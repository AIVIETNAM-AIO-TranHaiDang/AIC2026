"""The offline detector-counter pass: per-shot counts for the Answer Ledger.

A resumable stage keyed by ``shot_key`` (``ledger_count.jsonl``) that mirrors the
entities stage: single-process for resume, and a work-stealing
:class:`CounterProcessor` for the data-parallel multi-GPU path
(:func:`aic.parallel.run_local_stage`). Sharding is by VIDEO — a worker owns
whole videos — so a tracking backend that consumes a whole video's frames still
parallelises (frames of one video are never split across workers).

Per shot: run the configured detector over that shot's keyframes for the concept
list, reconcile the per-frame counts to one per-moment count (median, risk 4.3),
and write ``{shot_key, status, people_count, salient_objects, counts}``. A failed
shot is recorded ``status=failed`` and retried on the next resume. The ledger
build joins these rows into ``people_count`` / ``salient_objects``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from aic.chronicle.jobs import JobStats
from aic.config import Config
from aic.ingest.pipeline import KEYFRAMES_MANIFEST
from aic.manifest import ManifestWriter, read_manifest, shard_manifest_path
from aic.vqa.counter import Counter, build_counter, reconcile_objects

logger = logging.getLogger(__name__)

COUNT_MANIFEST = "ledger_count.jsonl"


def _shot_frames(
    cfg: Config, video_filter: set[str] | None
) -> dict[str, tuple[str, list[str]]]:
    """Map each shot_key -> (video_id, keyframe image paths in frame order).

    Reads the keyframes manifest (produced by ingestion), so the count pass
    needs only ingest to have run, not the chronicle text stages.
    """
    grouped: dict[str, tuple[str, list[tuple[int, str]]]] = {}
    for row in read_manifest(cfg.paths.manifests_dir / KEYFRAMES_MANIFEST):
        video_id = row["video_id"]
        if video_filter is not None and video_id not in video_filter:
            continue
        shot_key = f"{video_id}:{row['shot_id']}"
        entry = grouped.setdefault(shot_key, (video_id, []))
        entry[1].append((row["frame_idx"], row["image_path"]))
    return {
        shot_key: (video_id, [path for _idx, path in sorted(frames)])
        for shot_key, (video_id, frames) in grouped.items()
    }


def count_shot(counter: Counter, cfg: Config, image_paths: list[str]) -> dict:
    """Detector counts for one shot's keyframes, reconciled to per-moment.

    Returns the manifest payload (without shot_key/status): the reconciled
    per-concept counts, the person count, and the present concepts.
    """
    counter_cfg = cfg.chronicle.ledger.counter
    per_frame = counter.detect(image_paths, counter_cfg.concepts)
    objects = reconcile_objects(per_frame, min_count=1)
    return {
        "people_count": objects.get(counter_cfg.person_concept, 0),
        "salient_objects": sorted(objects),
        "counts": objects,
    }


def _done_keys(canonical: Path, manifest_path: Path, shard_tag: str) -> set[str]:
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
    return done


def _count_shots(
    counter: Counter,
    cfg: Config,
    shots: list[tuple[str, list[str]]],
    manifest: ManifestWriter,
    on_item: Callable[[int], None] | None,
) -> tuple[int, int]:
    """Count each (shot_key, frames) pair, one manifest row each."""
    processed = failed = 0
    for shot_key, image_paths in shots:
        try:
            payload = count_shot(counter, cfg, image_paths)
        except Exception as exc:  # noqa: BLE001 - survive one bad shot
            logger.error("counter failed for %s: %s", shot_key, exc)
            manifest.append({"shot_key": shot_key, "status": "failed"})
            failed += 1
        else:
            manifest.append({"shot_key": shot_key, "status": "ok", **payload})
            processed += 1
        if on_item is not None:
            on_item(1)
    return processed, failed


def run_counter_job(
    cfg: Config,
    counter: Counter,
    *,
    video_filter: set[str] | None = None,
    shard_tag: str = "",
    on_item: Callable[[int], None] | None = None,
) -> JobStats:
    """Count every shot missing from the manifest. Resumable (status=ok done)."""
    canonical = cfg.paths.manifests_dir / COUNT_MANIFEST
    manifest_path = shard_manifest_path(canonical, shard_tag)
    done = _done_keys(canonical, manifest_path, shard_tag)
    shots = [
        (shot_key, image_paths)
        for shot_key, (_video_id, image_paths) in _shot_frames(
            cfg, video_filter
        ).items()
        if shot_key not in done
    ]
    with ManifestWriter(manifest_path, "shot_key") as manifest:
        processed, failed = _count_shots(counter, cfg, shots, manifest, on_item)
    return JobStats(processed=processed, skipped=len(done), failed=failed)


class CounterProcessor:
    """Per-video counting for the work-stealing multi-GPU path.

    Holds one resident detector and an open rank-manifest writer; ``process``
    handles whole videos pulled from the shared queue in any order.
    """

    def __init__(
        self,
        cfg: Config,
        counter: Counter,
        shard_tag: str,
        on_item: Callable[[int], None] | None = None,
    ) -> None:
        self._cfg = cfg
        self._counter = counter
        self._on_item = on_item
        canonical = cfg.paths.manifests_dir / COUNT_MANIFEST
        manifest_path = shard_manifest_path(canonical, shard_tag)
        done = _done_keys(canonical, manifest_path, shard_tag)
        self._pending: dict[str, list[tuple[str, list[str]]]] = {}
        for shot_key, (video_id, image_paths) in _shot_frames(cfg, None).items():
            if shot_key in done:
                continue
            self._pending.setdefault(video_id, []).append((shot_key, image_paths))
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
        shots = self._pending.get(video_id, [])
        if not shots:
            return
        processed, failed = _count_shots(
            self._counter, self._cfg, shots, self._writer, self._on_item
        )
        self._processed += processed
        self._failed += failed

    def close(self) -> dict[str, int]:
        self._writer.close()
        return {
            "processed": self._processed,
            "skipped": self._skipped,
            "failed": self._failed,
        }


def _build_stage_counter(cfg: Config) -> Counter:
    """The configured counter with the resolved HF token (gated weights)."""
    counter = build_counter(
        cfg.chronicle.ledger.counter,
        cfg.paths.models_dir,
        cfg.paths.resolve_hf_token(),
    )
    warmup = getattr(counter, "warmup", None)
    if warmup is not None:
        warmup()
    return counter


def make_counter_worker(cfg, tag, on_item) -> CounterProcessor:
    return CounterProcessor(cfg, _build_stage_counter(cfg), tag, on_item)


def load_counts(cfg: Config) -> dict[str, dict]:
    """Read ledger_count.jsonl into a shot_key -> {people_count, salient_objects}
    map (ok rows only), for joining onto the ledger build."""
    result: dict[str, dict] = {}
    for row in read_manifest(cfg.paths.manifests_dir / COUNT_MANIFEST):
        if row.get("status") != "ok":
            continue
        result[row["shot_key"]] = {
            "people_count": row.get("people_count"),
            "salient_objects": row.get("salient_objects", []),
        }
    return result


def shot_video_map(cfg: Config) -> dict[str, str]:
    """Light shot_key -> video_id map for stage planning (no image paths)."""
    return {
        f"{row['video_id']}:{row['shot_id']}": row["video_id"]
        for row in read_manifest(cfg.paths.manifests_dir / KEYFRAMES_MANIFEST)
    }


def completed_count_keys(cfg: Config) -> set[str]:
    """Shot keys with a successful count row (for stage planning)."""
    return {
        row["shot_key"]
        for row in read_manifest(cfg.paths.manifests_dir / COUNT_MANIFEST)
        if row.get("status") == "ok"
    }
