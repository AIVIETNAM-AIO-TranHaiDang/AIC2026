"""Language-general spoken-entity NER for the Answer Ledger.

Fills the Chronicle's ``ChronicleRecord.entities`` (persons/orgs/places) from
each shot's speech + caption text, as a resumable stage keyed by ``shot_key``
(``entities.jsonl``) that mirrors the ASR/OCR stages: single-process for
resume, and a work-stealing :class:`EntityProcessor` for the data-parallel
multi-GPU path (see :func:`aic.parallel.run_local_stage`).

Two config-selectable backends (note 17 §4b — the corpus is bilingual, so no
Vietnamese-only tool):

- ``gliner`` — the local GLiNER-multi model: zero-shot, multilingual, runs on
  CPU, weights cached under ``paths.models_dir`` via ``HF_HOME``. A local torch
  stage, so it fans across GPUs.
- ``llm`` — an OpenAI-compatible endpoint asked for a ``{persons, orgs,
  places}`` JSON object; an endpoint stage, so it scales across the pool.

``entity_backend: auto`` picks ``llm`` when the endpoint is enabled, else
``gliner``. Any per-shot failure degrades to empty entities, never raises.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import httpx

from aic.chronicle.jobs import CHRONICLE_MANIFEST, JobStats
from aic.chronicle.schema import ChronicleRecord, Entities
from aic.config import Config, LedgerConfig
from aic.manifest import (
    ManifestWriter,
    read_manifest,
    shard_manifest_path,
)
from aic.models_cache import apply_model_cache_env
from aic.oaicompat import (
    ChatCompletionsClient,
    EndpointError,
    build_endpoint_pool,
    parse_json_reply,
    sampling_payload,
)

logger = logging.getLogger(__name__)

ENTITIES_MANIFEST = "entities.jsonl"

# Default NER prompt for the llm backend; overridable via ledger.llm
# system_prompt (an override must still demand this exact JSON object).
_LLM_SYSTEM_PROMPT = (
    "You extract named entities from a short passage that may be in any "
    "language. Reply with a single JSON object with exactly these fields, "
    "each a list of distinct verbatim strings from the text (empty list if "
    "none):\n"
    '"persons": people\'s names;\n'
    '"orgs": organisations, teams, companies, institutions;\n'
    '"places": locations, cities, countries, venues.\n'
    "Do not translate, invent, or explain. Output only the JSON object."
)

_LLM_REPAIR_PROMPT = (
    "Your previous reply was not the required JSON object. Reply again with "
    "ONLY the JSON object {persons, orgs, places}, no prose, no code fences."
)


def shot_ner_text(record: ChronicleRecord) -> str:
    """The text an NER backend reads for one shot: speech + captions.

    Both Vietnamese (``caption_vi``, ``asr_text``) and English (``caption_en``)
    sources are joined, so a multilingual backend sees names in whatever
    language they were spoken or written. Empty when the shot has no text
    (a field-footage shot), which yields no entities.
    """
    parts = [record.asr_text, record.caption_vi, record.caption_en]
    return "\n".join(part for part in parts if part)


def _entities_from_buckets(buckets: dict[str, list[str]]) -> Entities:
    """Build an :class:`Entities` from a bucket dict, deduping order-stably."""

    def dedup(values: object) -> list[str]:
        seen: dict[str, None] = {}
        for value in values if isinstance(values, list) else []:
            text = str(value).strip()
            if text:
                seen.setdefault(text, None)
        return list(seen)

    return Entities(
        persons=dedup(buckets.get("persons", [])),
        orgs=dedup(buckets.get("orgs", [])),
        places=dedup(buckets.get("places", [])),
    )


class EntityBackend(Protocol):
    def extract(self, text: str) -> Entities:
        """Extract named entities from one passage of text."""
        ...


class GlinerEntityBackend:
    """Local GLiNER-multi NER; weights cached under ``paths.models_dir``.

    The model is imported and loaded lazily on first use (``gliner`` is an
    optional dependency, absent on a box that only runs the llm backend). The
    ``label_map`` from config assigns each GLiNER label to a Chronicle bucket,
    so the label set is reusable without touching code.
    """

    def __init__(self, cfg: LedgerConfig, models_dir: Path) -> None:
        self._cfg = cfg.gliner
        # Ensure HF_HOME points inside the project tree before the first
        # download, covering gliner (which resolves weights through
        # huggingface_hub's env-honouring cache). Idempotent (setdefault).
        apply_model_cache_env(models_dir)
        # label -> bucket, from the config's bucket -> labels map.
        self._label_to_bucket: dict[str, str] = {}
        for bucket, labels in self._cfg.label_map.items():
            for label in labels:
                self._label_to_bucket[label] = bucket
        self._labels = list(self._label_to_bucket)
        self._model = None

    def _resolve_device(self) -> str:
        if self._cfg.device != "auto":
            return self._cfg.device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def _ensure_model(self):
        if self._model is None:
            from gliner import GLiNER

            self._model = GLiNER.from_pretrained(
                self._cfg.model_id, map_location=self._resolve_device()
            )
            logger.info("loaded GLiNER %s", self._cfg.model_id)
        return self._model

    def warmup(self) -> None:
        """Force the weights to download/load once (call in the parent before
        spawning data-parallel workers, so they don't race on the cache)."""
        self._ensure_model()

    def extract(self, text: str) -> Entities:
        if not text.strip():
            return Entities()
        model = self._ensure_model()
        buckets: dict[str, list[str]] = {"persons": [], "orgs": [], "places": []}
        for span in model.predict_entities(
            text, self._labels, threshold=self._cfg.threshold
        ):
            bucket = self._label_to_bucket.get(span.get("label", ""))
            if bucket is not None:
                buckets[bucket].append(span.get("text", ""))
        return _entities_from_buckets(buckets)


class LlmEntityError(EndpointError):
    """Raised when an llm NER request fails; the caller degrades to empty."""


class LlmEntityBackend:
    """NER through one or more OpenAI-compatible chat-completions endpoints."""

    def __init__(
        self,
        cfg: LedgerConfig,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._cfg = cfg.llm
        self._prompt = cfg.llm.system_prompt or _LLM_SYSTEM_PROMPT
        self._pool = build_endpoint_pool(
            provider=cfg.llm.provider,
            base_url=cfg.llm.base_url,
            model=cfg.llm.model,
            api_key_env=cfg.llm.api_key_env,
            timeout_s=cfg.llm.timeout_s,
            host_max_retries=cfg.llm.host_max_retries,
            rate_limit_max_retries=cfg.llm.rate_limit_max_retries,
            rate_limit_backoff_s=cfg.llm.rate_limit_backoff_s,
            extra_fields_for=lambda _model: sampling_payload(cfg.llm),
            transport=transport,
            error_cls=LlmEntityError,
        )

    def _messages(self, text: str) -> list[dict]:
        return [
            {"role": "system", "content": self._prompt},
            {"role": "user", "content": text},
        ]

    def _parse(self, reply: str) -> Entities:
        payload = parse_json_reply(reply)
        if not isinstance(payload, dict):
            raise LlmEntityError("entity reply was not a JSON object")
        return _entities_from_buckets(payload)

    def _extract_one(self, client: ChatCompletionsClient, text: str) -> Entities:
        messages = self._messages(text)
        reply = client.complete(
            messages, max_tokens=self._cfg.max_output_tokens, json_response=True
        )
        try:
            return self._parse(reply)
        except (LlmEntityError, json.JSONDecodeError):
            # One repair retry (X3), then a typed-empty fallback.
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": _LLM_REPAIR_PROMPT})
            repaired = client.complete(
                messages, max_tokens=self._cfg.max_output_tokens, json_response=True
            )
            return self._parse(repaired)

    def extract(self, text: str) -> Entities:
        if not text.strip():
            return Entities()
        [result] = self._pool.run_batch([text], self._extract_one, 1)
        if isinstance(result, Exception):
            raise result
        return result

    def extract_many(self, texts: list[str]) -> list[Entities | Exception]:
        """Extract for many shots concurrently across the endpoint pool."""
        return self._pool.run_batch(texts, self._extract_one, self._cfg.num_parallel)

    def close(self) -> None:
        self._pool.close()


def build_entity_backend(
    cfg: LedgerConfig,
    models_dir: Path,
    transport: httpx.BaseTransport | None = None,
) -> EntityBackend:
    """Build the entity backend the config resolves to (gliner or llm)."""
    backend = cfg.resolved_entity_backend()
    if backend == "gliner":
        return GlinerEntityBackend(cfg, models_dir)
    return LlmEntityBackend(cfg, transport=transport)


def _pending_records(cfg: Config, done: set[str]) -> list[ChronicleRecord]:
    """Assembled shots not yet in the entity manifest, as validated records."""
    records = []
    for raw in read_manifest(cfg.paths.manifests_dir / CHRONICLE_MANIFEST):
        if raw.get("shot_key") in done:
            continue
        raw = dict(raw)
        raw.pop("shot_key", None)
        records.append(ChronicleRecord.model_validate(raw))
    return records


def _extract_records(
    backend: EntityBackend,
    records: list[ChronicleRecord],
    manifest: ManifestWriter,
    on_item: Callable[[int], None] | None,
) -> tuple[int, int]:
    """Extract entities for a list of shots, writing one manifest row each.

    An endpoint backend (``extract_many``) fans the shots across its pool; a
    local backend runs per shot. A failed shot is recorded ``status=failed``
    (retried on the next resume), never aborting the run.
    """
    processed = failed = 0
    batch_extract = getattr(backend, "extract_many", None)
    if batch_extract is not None:
        results = batch_extract([shot_ner_text(r) for r in records])
        for record, result in zip(records, results, strict=True):
            if isinstance(result, Exception):
                logger.error("entity NER failed for %s: %s", record.shot_key, result)
                manifest.append({"shot_key": record.shot_key, "status": "failed"})
                failed += 1
            else:
                manifest.append(
                    {"shot_key": record.shot_key, "status": "ok", **result.model_dump()}
                )
                processed += 1
            if on_item is not None:
                on_item(1)
        return processed, failed
    for record in records:
        try:
            entities = backend.extract(shot_ner_text(record))
        except Exception as exc:  # noqa: BLE001 - survive one bad shot
            logger.error("entity NER failed for %s: %s", record.shot_key, exc)
            manifest.append({"shot_key": record.shot_key, "status": "failed"})
            failed += 1
        else:
            manifest.append(
                {"shot_key": record.shot_key, "status": "ok", **entities.model_dump()}
            )
            processed += 1
        if on_item is not None:
            on_item(1)
    return processed, failed


def run_entity_job(
    cfg: Config,
    backend: EntityBackend,
    *,
    video_filter: set[str] | None = None,
    shard_tag: str = "",
    on_item: Callable[[int], None] | None = None,
) -> JobStats:
    """Extract entities for every assembled shot missing from the manifest.

    ``video_filter``/``shard_tag``/``on_item`` support data-parallel execution
    (see :func:`aic.parallel.run_local_stage`); the defaults keep the original
    single-process behaviour. Only ``status=ok`` rows count as done, so a
    transient failure is retried on the next resume (X4).
    """
    canonical = cfg.paths.manifests_dir / ENTITIES_MANIFEST
    manifest_path = shard_manifest_path(canonical, shard_tag)
    done = {
        row["shot_key"]
        for row in read_manifest(canonical)
        if row.get("status") == "ok"
    }
    if shard_tag:
        done |= {
            row["shot_key"]
            for row in read_manifest(manifest_path)
            if row.get("status") == "ok"
        }
    records = _pending_records(cfg, done)
    if video_filter is not None:
        records = [r for r in records if r.video_id in video_filter]
    with ManifestWriter(manifest_path, "shot_key") as manifest:
        processed, failed = _extract_records(backend, records, manifest, on_item)
    return JobStats(processed=processed, skipped=len(done), failed=failed)


class EntityProcessor:
    """Per-video entity extraction for the work-stealing multi-GPU path.

    Groups each video's pending shots once and holds one open rank-manifest
    writer so a worker can process videos pulled from the shared queue in any
    order; the per-shot work matches :func:`run_entity_job`.
    """

    def __init__(
        self,
        cfg: Config,
        backend: EntityBackend,
        shard_tag: str,
        on_item: Callable[[int], None] | None = None,
    ) -> None:
        self._backend = backend
        self._on_item = on_item
        canonical = cfg.paths.manifests_dir / ENTITIES_MANIFEST
        manifest_path = shard_manifest_path(canonical, shard_tag)
        done = {
            row["shot_key"]
            for row in read_manifest(canonical)
            if row.get("status") == "ok"
        }
        if shard_tag:
            done |= {
                row["shot_key"]
                for row in read_manifest(manifest_path)
                if row.get("status") == "ok"
            }
        self._pending: dict[str, list[ChronicleRecord]] = {}
        for record in _pending_records(cfg, done):
            self._pending.setdefault(record.video_id, []).append(record)
        # Already-done shots counted PER VIDEO, so a multi-GPU run's summed
        # skipped total is the corpus total, not len(done) once per worker.
        self._skipped_by_video: dict[str, int] = {}
        for shot_key in done:
            video_id = shot_key.split(":", 1)[0]
            self._skipped_by_video[video_id] = (
                self._skipped_by_video.get(video_id, 0) + 1
            )
        self._writer = ManifestWriter(manifest_path, "shot_key")
        self._processed = self._skipped = self._failed = 0

    def process(self, video_id: str) -> None:
        self._skipped += self._skipped_by_video.get(video_id, 0)
        records = self._pending.get(video_id, [])
        if not records:
            return
        processed, failed = _extract_records(
            self._backend, records, self._writer, self._on_item
        )
        self._processed += processed
        self._failed += failed

    def close(self) -> dict[str, int]:
        self._writer.close()
        if hasattr(self._backend, "close"):
            self._backend.close()
        return {
            "processed": self._processed,
            "skipped": self._skipped,
            "failed": self._failed,
        }


def load_entities(cfg: Config) -> dict[str, Entities]:
    """Read entities.jsonl into a shot_key -> Entities map (ok rows only)."""
    result: dict[str, Entities] = {}
    for row in read_manifest(cfg.paths.manifests_dir / ENTITIES_MANIFEST):
        if row.get("status") != "ok":
            continue
        result[row["shot_key"]] = Entities(
            persons=row.get("persons", []),
            orgs=row.get("orgs", []),
            places=row.get("places", []),
        )
    return result


def completed_entity_keys(cfg: Config) -> set[str]:
    """Shot keys with a successful entity row (for planning)."""
    return {
        row["shot_key"]
        for row in read_manifest(cfg.paths.manifests_dir / ENTITIES_MANIFEST)
        if row.get("status") == "ok"
    }
