"""Resumable Chronicle extraction jobs and the per-shot assembler.

Each extractor writes its own manifest (asr.jsonl per video, ocr.jsonl per
keyframe, captions.jsonl per shot) so the expensive stages can run and resume
independently — on different machines if need be. The assembler then joins
everything into chronicle.jsonl, one validated record per shot; shots whose
caption is missing are assembled anyway (degraded rows are legal).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from aic.chronicle.asr import (
    AsrBackend,
    AsrSegment,
    align_segments_to_shots,
    passes_speech_rate,
)
from aic.chronicle.caption import OpenAICompatCaptioner
from aic.chronicle.ocr import OcrBackend, merge_near_identical
from aic.chronicle.schema import ChronicleRecord, OcrLine, normalize_vietnamese
from aic.chronicle.translate import Translator
from aic.config import Config
from aic.ingest.pipeline import (
    KEYFRAMES_MANIFEST,
    SHOTS_MANIFEST,
    VIDEOS_MANIFEST,
)
from aic.manifest import (
    ManifestWriter,
    completed_keys,
    read_manifest,
    rewrite_manifest,
    shard_manifest_path,
)

logger = logging.getLogger(__name__)

ASR_MANIFEST = "asr.jsonl"
OCR_MANIFEST = "ocr.jsonl"
CAPTIONS_MANIFEST = "captions.jsonl"
CHRONICLE_MANIFEST = "chronicle.jsonl"


@dataclass(frozen=True)
class JobStats:
    processed: int
    skipped: int
    failed: int


def reset_asr_records(cfg: Config, video_ids: set[str] | None = None) -> int:
    """Drop ASR manifest rows so the next resumable run redoes them.

    ``video_ids`` None drops every row (a settings change, e.g. enabling the
    VAD filter, invalidates all transcripts); a set drops only those videos.
    Returns the number of rows removed. The subsequent ``run_asr_job`` then
    treats the affected videos as pending again — the one sanctioned way to
    re-run a stage whose artifacts are valid-but-wrong rather than corrupt
    (corruption is the verify/repair pass's job).
    """
    path = cfg.paths.manifests_dir / ASR_MANIFEST
    if video_ids is None:
        return rewrite_manifest(path, lambda row: False)
    return rewrite_manifest(path, lambda row: row.get("video_id") not in video_ids)


def _done_videos(cfg: Config) -> list[dict]:
    return [
        record
        for record in read_manifest(cfg.paths.manifests_dir / VIDEOS_MANIFEST)
        if record.get("status") == "done"
    ]


def run_asr_job(
    cfg: Config,
    backend: AsrBackend,
    *,
    video_filter: set[str] | None = None,
    shard_tag: str = "",
    on_item: Callable[[int], None] | None = None,
) -> JobStats:
    """Transcribe every ingested video that has an audio track.

    ``video_filter``/``shard_tag``/``on_item`` support data-parallel execution
    (see :func:`aic.parallel.run_local_stage`); the defaults keep the original
    single-process behaviour.
    """
    canonical = cfg.paths.manifests_dir / ASR_MANIFEST
    manifest_path = shard_manifest_path(canonical, shard_tag)
    done = completed_keys(canonical, "video_id")
    if shard_tag:
        done |= completed_keys(manifest_path, "video_id")
    processed = skipped = failed = 0
    with ManifestWriter(manifest_path, "video_id") as manifest:
        for video in _done_videos(cfg):
            video_id = video["video_id"]
            if video_filter is not None and video_id not in video_filter:
                continue
            if video_id in done:
                skipped += 1
                continue
            if not video.get("has_audio", False):
                manifest.append(
                    {"video_id": video_id, "segments": [], "reason": "no_audio"}
                )
                processed += 1
                if on_item is not None:
                    on_item(1)
                continue
            try:
                segments = backend.transcribe(Path(video["path"]))
            except Exception as exc:  # noqa: BLE001 - job must survive one bad file
                logger.error("ASR failed for %s: %s", video_id, exc)
                failed += 1
                if on_item is not None:
                    on_item(1)
                continue
            manifest.append(
                {
                    "video_id": video_id,
                    "segments": [
                        {
                            "start_ms": s.start_ms,
                            "end_ms": s.end_ms,
                            "text": s.text,
                            "confidence": s.confidence,
                        }
                        for s in segments
                    ],
                }
            )
            processed += 1
            if on_item is not None:
                on_item(1)
    return JobStats(processed, skipped, failed)


def _transcribe_one(backend: AsrBackend, video: dict, manifest: ManifestWriter) -> str:
    """Transcribe one video and append its ASR record.

    Returns ``"processed"`` or ``"failed"``. Shared per-video logic behind the
    single-process :func:`run_asr_job` and the multi-GPU :class:`AsrProcessor`.
    """
    video_id = video["video_id"]
    if not video.get("has_audio", False):
        manifest.append({"video_id": video_id, "segments": [], "reason": "no_audio"})
        return "processed"
    try:
        segments = backend.transcribe(Path(video["path"]))
    except Exception as exc:  # noqa: BLE001 - job must survive one bad file
        logger.error("ASR failed for %s: %s", video_id, exc)
        return "failed"
    manifest.append(
        {
            "video_id": video_id,
            "segments": [
                {
                    "start_ms": s.start_ms,
                    "end_ms": s.end_ms,
                    "text": s.text,
                    "confidence": s.confidence,
                }
                for s in segments
            ],
        }
    )
    return "processed"


class AsrProcessor:
    """Per-video transcription for the work-stealing multi-GPU path.

    Indexes the ingested videos once and holds one open rank-manifest writer so
    a worker can transcribe videos pulled from the shared queue in any order;
    :func:`_transcribe_one` does the identical per-video work as the
    single-process :func:`run_asr_job`.
    """

    def __init__(
        self,
        cfg: Config,
        backend: AsrBackend,
        shard_tag: str,
        on_item: Callable[[int], None] | None = None,
    ) -> None:
        self._backend = backend
        self._on_item = on_item
        self._videos_by_id = {v["video_id"]: v for v in _done_videos(cfg)}
        canonical = cfg.paths.manifests_dir / ASR_MANIFEST
        manifest_path = shard_manifest_path(canonical, shard_tag)
        self._done = completed_keys(canonical, "video_id")
        if shard_tag:
            self._done |= completed_keys(manifest_path, "video_id")
        self._writer = ManifestWriter(manifest_path, "video_id")
        self._processed = self._skipped = self._failed = 0

    def process(self, video_id: str) -> None:
        if video_id in self._done:
            self._skipped += 1
            return
        video = self._videos_by_id.get(video_id)
        if video is None:
            return
        if _transcribe_one(self._backend, video, self._writer) == "processed":
            self._processed += 1
        else:
            self._failed += 1
        if self._on_item is not None:
            self._on_item(1)

    def close(self) -> dict[str, int]:
        self._writer.close()
        return {
            "processed": self._processed,
            "skipped": self._skipped,
            "failed": self._failed,
        }


def _write_ocr_record(manifest: ManifestWriter, record: dict, lines) -> None:
    manifest.append(
        {
            "keyframe_id": record["keyframe_id"],
            "video_id": record["video_id"],
            "shot_id": record["shot_id"],
            "lines": [
                {"text": line.text, "confidence": line.confidence, "bbox": line.bbox}
                for line in lines
            ],
        }
    )


def _run_ocr_batched(
    cfg: Config,
    backend: OcrBackend,
    pending: list[dict],
    manifest: ManifestWriter,
    on_item: Callable[[int], None] | None,
) -> tuple[int, int]:
    """Dispatch OCR for an endpoint backend in batches across its pool."""
    processed = failed = 0
    batch_size = cfg.chronicle.ocr.batch_size
    for start in range(0, len(pending), batch_size):
        chunk = pending[start : start + batch_size]
        images: list[np.ndarray] = []
        decoded: list[dict] = []  # records that decoded, aligned to images
        for record in chunk:
            try:
                with Image.open(record["image_path"]) as img:
                    images.append(np.array(img.convert("RGB")))
                decoded.append(record)
            except Exception as exc:  # noqa: BLE001 - survive one bad frame
                logger.error("OCR failed for %s: %s", record["keyframe_id"], exc)
                failed += 1
                if on_item is not None:
                    on_item(1)
        results = backend.read_many(images)
        for record, result in zip(decoded, results, strict=True):
            if isinstance(result, Exception):
                logger.error("OCR failed for %s: %s", record["keyframe_id"], result)
                failed += 1
            else:
                _write_ocr_record(manifest, record, result)
                processed += 1
            if on_item is not None:
                on_item(1)
    return processed, failed


def run_ocr_job(
    cfg: Config,
    backend: OcrBackend,
    *,
    video_filter: set[str] | None = None,
    shard_tag: str = "",
    on_item: Callable[[int], None] | None = None,
) -> JobStats:
    """Read on-screen text from every keyframe image.

    ``video_filter``/``shard_tag``/``on_item`` support data-parallel execution
    (see :func:`aic.parallel.run_local_stage`); the defaults keep the original
    single-process behaviour. An endpoint backend that exposes ``read_many``
    (the VLM backend) fans its frames out across its endpoint pool in batches;
    a local backend (EasyOCR) is read serially per frame.
    """
    canonical = cfg.paths.manifests_dir / OCR_MANIFEST
    manifest_path = shard_manifest_path(canonical, shard_tag)
    done = completed_keys(canonical, "keyframe_id")
    if shard_tag:
        done |= completed_keys(manifest_path, "keyframe_id")
    processed = skipped = failed = 0
    batch_reader = getattr(backend, "read_many", None)
    with ManifestWriter(manifest_path, "keyframe_id") as manifest:
        pending: list[dict] = []
        for record in read_manifest(cfg.paths.manifests_dir / KEYFRAMES_MANIFEST):
            keyframe_id = record["keyframe_id"]
            if video_filter is not None and record["video_id"] not in video_filter:
                continue
            if keyframe_id in done:
                skipped += 1
                continue
            if batch_reader is not None:
                pending.append(record)
                continue
            try:
                with Image.open(record["image_path"]) as img:
                    lines = backend.read(np.array(img.convert("RGB")))
            except Exception as exc:  # noqa: BLE001 - job must survive one bad frame
                logger.error("OCR failed for %s: %s", keyframe_id, exc)
                failed += 1
                if on_item is not None:
                    on_item(1)
                continue
            _write_ocr_record(manifest, record, lines)
            processed += 1
            if on_item is not None:
                on_item(1)
        if batch_reader is not None:
            processed_b, failed_b = _run_ocr_batched(
                cfg, backend, pending, manifest, on_item
            )
            processed += processed_b
            failed += failed_b
    return JobStats(processed, skipped, failed)


class OcrProcessor:
    """Per-video OCR for the work-stealing multi-GPU path.

    Groups each video's pending keyframes once and holds one open rank-manifest
    writer so a worker can OCR videos pulled from the shared queue in any order.
    A local EasyOCR backend is read per frame; a backend exposing ``read_many``
    (the VLM) fans that video's frames across its endpoint pool via the shared
    :func:`_run_ocr_batched`, matching :func:`run_ocr_job`.
    """

    def __init__(
        self,
        cfg: Config,
        backend: OcrBackend,
        shard_tag: str,
        on_item: Callable[[int], None] | None = None,
    ) -> None:
        self._cfg = cfg
        self._backend = backend
        self._on_item = on_item
        self._batch_reader = getattr(backend, "read_many", None)
        canonical = cfg.paths.manifests_dir / OCR_MANIFEST
        manifest_path = shard_manifest_path(canonical, shard_tag)
        done = completed_keys(canonical, "keyframe_id")
        if shard_tag:
            done |= completed_keys(manifest_path, "keyframe_id")
        self._pending: dict[str, list[dict]] = {}
        self._skipped_by_video: dict[str, int] = {}
        for record in read_manifest(cfg.paths.manifests_dir / KEYFRAMES_MANIFEST):
            video_id = record["video_id"]
            if record["keyframe_id"] in done:
                self._skipped_by_video[video_id] = (
                    self._skipped_by_video.get(video_id, 0) + 1
                )
            else:
                self._pending.setdefault(video_id, []).append(record)
        self._writer = ManifestWriter(manifest_path, "keyframe_id")
        self._processed = self._skipped = self._failed = 0

    def process(self, video_id: str) -> None:
        self._skipped += self._skipped_by_video.get(video_id, 0)
        records = self._pending.get(video_id, [])
        if not records:
            return
        if self._batch_reader is not None:
            processed, failed = _run_ocr_batched(
                self._cfg, self._backend, records, self._writer, self._on_item
            )
            self._processed += processed
            self._failed += failed
            return
        for record in records:
            try:
                with Image.open(record["image_path"]) as img:
                    lines = self._backend.read(np.array(img.convert("RGB")))
            except Exception as exc:  # noqa: BLE001 - job must survive one bad frame
                logger.error("OCR failed for %s: %s", record["keyframe_id"], exc)
                self._failed += 1
                if self._on_item is not None:
                    self._on_item(1)
                continue
            _write_ocr_record(self._writer, record, lines)
            self._processed += 1
            if self._on_item is not None:
                self._on_item(1)

    def close(self) -> dict[str, int]:
        self._writer.close()
        return {
            "processed": self._processed,
            "skipped": self._skipped,
            "failed": self._failed,
        }


def _load_shot_frames(image_paths: list[str], frames_per_shot: int) -> list[np.ndarray]:
    """Decode up to ``frames_per_shot`` keyframes of a shot into RGB arrays."""
    frames = []
    for path in image_paths[:frames_per_shot]:
        with Image.open(path) as img:
            frames.append(np.array(img.convert("RGB")))
    return frames


def run_caption_job(cfg: Config, captioner: OpenAICompatCaptioner) -> JobStats:
    """Caption every shot from its keyframes via the configured endpoint(s).

    Shots are dispatched to the captioner's endpoint pool in batches of
    ``caption.batch_size``; each batch's results are written before the next
    starts, so the manifest advances for resume and a mixed/failing endpoint
    pool degrades per shot rather than aborting the run.
    """
    caption_cfg = cfg.chronicle.caption
    manifest_path = cfg.paths.manifests_dir / CAPTIONS_MANIFEST
    # Only successful (or legitimately empty) shots count as done: a
    # status=failed row records the error but must not block the retry on
    # the next resume — endpoint failures are usually transient. The
    # assembler keys captions by shot_key and a later ok row wins.
    done = {
        record["shot_key"]
        for record in read_manifest(manifest_path)
        if record.get("status") in ("ok", "no_keyframes")
    }

    frames_by_shot: dict[str, list[str]] = {}
    for record in read_manifest(cfg.paths.manifests_dir / KEYFRAMES_MANIFEST):
        key = f"{record['video_id']}:{record['shot_id']}"
        frames_by_shot.setdefault(key, []).append(record["image_path"])

    processed = skipped = failed = 0
    with ManifestWriter(manifest_path, "shot_key") as manifest:
        # First pass: split shots into skip / no-keyframe (written now) / pending.
        pending: list[tuple[str, list[str]]] = []
        for shot in read_manifest(cfg.paths.manifests_dir / SHOTS_MANIFEST):
            shot_key = f"{shot['video_id']}:{shot['shot_id']}"
            if shot_key in done:
                skipped += 1
                continue
            image_paths = frames_by_shot.get(shot_key, [])
            if not image_paths:
                # Every keyframe of this shot was dropped by dedup/blank
                # filters; there is nothing to caption.
                manifest.append({"shot_key": shot_key, "status": "no_keyframes"})
                processed += 1
                continue
            pending.append((shot_key, image_paths))

        # Second pass: dispatch pending shots across the endpoint pool in
        # batches, writing each batch before moving on. One aggregate
        # progress bar with ok/failed counts replaces per-request logging
        # (httpx request lines are capped at WARNING by setup_logging).
        from tqdm.auto import tqdm
        from tqdm.contrib.logging import logging_redirect_tqdm

        with (
            logging_redirect_tqdm(),
            tqdm(total=len(pending), desc="captions", unit="shot") as bar,
        ):
            for start in range(0, len(pending), caption_cfg.batch_size):
                batch = pending[start : start + caption_cfg.batch_size]
                frame_batches = [
                    _load_shot_frames(image_paths, caption_cfg.frames_per_shot)
                    for _shot_key, image_paths in batch
                ]
                results = captioner.caption_shots(frame_batches)
                for (shot_key, _paths), result in zip(batch, results, strict=True):
                    if isinstance(result, Exception):
                        logger.error("caption failed for %s: %s", shot_key, result)
                        manifest.append(
                            {
                                "shot_key": shot_key,
                                "status": "failed",
                                "reason": str(result),
                            }
                        )
                        failed += 1
                    else:
                        manifest.append(
                            {
                                "shot_key": shot_key,
                                "status": "ok",
                                "caption_en": result.caption_en,
                                "caption_vi": result.caption_vi,
                                "actions": result.actions,
                                "scene": result.scene,
                                "confidence": result.confidence,
                            }
                        )
                        processed += 1
                    bar.update(1)
                bar.set_postfix(ok=processed, failed=failed)
    return JobStats(processed, skipped, failed)


def _split_overlay_lines(
    shot_lines: dict[str, list[OcrLine]],
    shots_of_video: list[str],
    min_recurrence: float,
) -> tuple[dict[str, list[OcrLine]], list[OcrLine]]:
    """Split one video's OCR into per-shot scene text and video overlay.

    A normalized line text present in at least ``min_recurrence`` of the
    video's shots is a persistent overlay (watermark, news ticker, program
    bug): it identifies the video, not the moment, so it moves out of every
    shot's evidence into one per-video list (best-confidence copy kept).
    """
    total_shots = len(shots_of_video)
    shots_containing: dict[str, set[str]] = {}
    for shot_key, lines in shot_lines.items():
        for line in lines:
            shots_containing.setdefault(
                normalize_vietnamese(line.text), set()
            ).add(shot_key)
    overlay_texts = {
        text
        for text, shots in shots_containing.items()
        if total_shots > 0 and len(shots) / total_shots >= min_recurrence
    }
    scene_by_shot: dict[str, list[OcrLine]] = {}
    best_overlay: dict[str, OcrLine] = {}
    for shot_key, lines in shot_lines.items():
        kept = []
        for line in lines:
            text = normalize_vietnamese(line.text)
            if text in overlay_texts:
                current = best_overlay.get(text)
                if current is None or line.confidence > current.confidence:
                    best_overlay[text] = line
            else:
                kept.append(line)
        scene_by_shot[shot_key] = kept
    overlay = sorted(best_overlay.values(), key=lambda line: -line.confidence)
    return scene_by_shot, overlay


def assemble_chronicle(cfg: Config, translator: Translator | None = None) -> JobStats:
    """Join shots + ASR + OCR + captions into one record per shot.

    Assembly is deterministic and cheap, so it rebuilds from scratch each run
    (no resume): re-running after adding captions upgrades degraded rows.
    The ASR rate gate and the overlay split both apply here too, so changing
    those settings heals an existing corpus without re-running extraction.
    """
    manifests = cfg.paths.manifests_dir
    min_rate = cfg.chronicle.asr.min_chars_per_second

    asr_by_video: dict[str, list[AsrSegment]] = {}
    for record in read_manifest(manifests / ASR_MANIFEST):
        asr_by_video[record["video_id"]] = [
            AsrSegment(
                start_ms=s["start_ms"],
                end_ms=s["end_ms"],
                text=s["text"],
                confidence=s["confidence"],
            )
            for s in record.get("segments", [])
            if passes_speech_rate(s["text"], s["start_ms"], s["end_ms"], min_rate)
        ]

    ocr_by_shot: dict[str, list[OcrLine]] = {}
    for record in read_manifest(manifests / OCR_MANIFEST):
        key = f"{record['video_id']}:{record['shot_id']}"
        ocr_by_shot.setdefault(key, []).extend(
            OcrLine(
                text=line["text"],
                confidence=line["confidence"],
                bbox=line.get("bbox"),
            )
            for line in record.get("lines", [])
        )

    captions_by_shot: dict[str, dict] = {
        record["shot_key"]: record
        for record in read_manifest(manifests / CAPTIONS_MANIFEST)
        if record.get("status") == "ok"
    }

    shots_by_video: dict[str, list[dict]] = {}
    for shot in read_manifest(manifests / SHOTS_MANIFEST):
        shots_by_video.setdefault(shot["video_id"], []).append(shot)

    records: list[ChronicleRecord] = []
    for video_id, shots in shots_by_video.items():
        shot_windows = [
            (shot["shot_id"], shot["start_ms"], shot["end_ms"]) for shot in shots
        ]
        asr_per_shot = align_segments_to_shots(
            asr_by_video.get(video_id, []), shot_windows
        )
        shot_keys = [f"{video_id}:{shot['shot_id']}" for shot in shots]
        video_ocr = {
            key: merge_near_identical(ocr_by_shot.get(key, []))
            for key in shot_keys
        }
        overlay: list[OcrLine] = []
        if cfg.chronicle.overlay_min_recurrence is not None:
            video_ocr, overlay = _split_overlay_lines(
                video_ocr, shot_keys, cfg.chronicle.overlay_min_recurrence
            )
        for shot, shot_key in zip(shots, shot_keys, strict=True):
            caption = captions_by_shot.get(shot_key, {})
            records.append(
                ChronicleRecord(
                    video_id=video_id,
                    shot_id=shot["shot_id"],
                    t_start_ms=shot["start_ms"],
                    t_end_ms=shot["end_ms"],
                    caption_en=caption.get("caption_en"),
                    caption_vi=caption.get("caption_vi"),
                    actions=caption.get("actions", []),
                    scene=caption.get("scene"),
                    caption_confidence=caption.get("confidence"),
                    ocr=video_ocr.get(shot_key, []),
                    # The overlay is per video; every shot row carries it so
                    # the textstack can build the video document from any
                    # subset of rows without a separate manifest.
                    overlay_ocr=overlay,
                    asr_text=asr_per_shot.get(shot["shot_id"]) or None,
                )
            )

    if translator is not None:
        # Native caption_vi (request_caption_vi) wins; translation fills
        # the rest, so the two sources compose per shot.
        to_translate = [r for r in records if r.caption_en and not r.caption_vi]
        translations = translator.translate([r.caption_en for r in to_translate])
        for record, caption_vi in zip(to_translate, translations, strict=True):
            record.caption_vi = caption_vi

    if cfg.chronicle.ledger.enabled:
        # Join the resumable NER stage's output into the (deferred) entities
        # field. Function-level import breaks the jobs<->entities cycle. When
        # the ledger is disabled this is skipped and KIS output is unchanged.
        from aic.chronicle.entities import load_entities

        entities_by_shot = load_entities(cfg)
        for record in records:
            entities = entities_by_shot.get(record.shot_key)
            if entities is not None:
                record.entities = entities

    chronicle_path = manifests / CHRONICLE_MANIFEST
    chronicle_path.unlink(missing_ok=True)
    degraded = 0
    with ManifestWriter(chronicle_path, "shot_key") as manifest:
        for record in records:
            degraded += int(record.degraded)
            manifest.append({"shot_key": record.shot_key, **record.model_dump()})
    logger.info(
        "chronicle assembled: %d shots (%d without caption)",
        len(records),
        degraded,
    )
    return JobStats(processed=len(records), skipped=0, failed=0)


def load_chronicle(cfg: Config) -> list[ChronicleRecord]:
    """Read chronicle.jsonl back into validated records."""
    records = []
    for raw in read_manifest(cfg.paths.manifests_dir / CHRONICLE_MANIFEST):
        raw = dict(raw)
        raw.pop("shot_key", None)
        records.append(ChronicleRecord.model_validate(raw))
    return records
