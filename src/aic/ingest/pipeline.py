"""Per-video ingestion orchestration with resumable manifests.

For each video: probe -> decode low-res -> detect shots -> plan keyframes ->
extract and filter keyframe images -> append records to the three manifests
(videos, shots, keyframes). A failed video is recorded with its reason and the
run continues; a re-run with the same manifests skips completed videos.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from aic.config import Config
from aic.ids import make_keyframe_id
from aic.ingest.dedup import PHashDeduper, is_low_information
from aic.ingest.keyframes import plan_keyframes
from aic.ingest.shots import ShotDetector
from aic.ingest.video import (
    VideoError,
    extract_frames_at,
    iter_frames,
    probe_video,
)
from aic.manifest import ManifestWriter, completed_keys, shard_manifest_path

logger = logging.getLogger(__name__)

VIDEOS_MANIFEST = "videos.jsonl"
SHOTS_MANIFEST = "shots.jsonl"
KEYFRAMES_MANIFEST = "keyframes.jsonl"


@dataclass(frozen=True)
class IngestStats:
    processed: int
    skipped: int
    failed: int
    keyframes_written: int


def discover_videos(videos_dir: Path, extensions: list[str]) -> list[Path]:
    """List ingestable video files, sorted for deterministic ordering."""
    if not videos_dir.is_dir():
        raise FileNotFoundError(
            f"videos directory not found: {videos_dir} "
            "(drop input videos there, or point paths.data_root elsewhere)"
        )
    normalised = {ext.lower().lstrip(".") for ext in extensions}
    return sorted(
        path
        for path in videos_dir.iterdir()
        if path.is_file() and path.suffix.lower().lstrip(".") in normalised
    )


def run_ingest(
    cfg: Config,
    detector: ShotDetector,
    limit: int | None = None,
    *,
    video_filter: set[str] | None = None,
    shard_tag: str = "",
    on_item: Callable[[int], None] | None = None,
) -> IngestStats:
    """Ingest every not-yet-processed video in the configured corpus.

    ``video_filter`` restricts the run to those video ids (a data-parallel
    worker's shard); ``shard_tag`` routes the three manifests to per-shard
    files so concurrent workers never interleave writes; ``on_item`` is called
    once per handled video for a progress bar. The defaults reproduce the
    original single-process behaviour exactly.
    """
    paths = cfg.paths
    manifests_dir = paths.manifests_dir
    videos = discover_videos(paths.videos_dir, cfg.ingest.video_extensions)
    if video_filter is not None:
        videos = [path for path in videos if path.stem in video_filter]
    if limit is not None:
        videos = videos[:limit]
    canonical = manifests_dir / VIDEOS_MANIFEST
    done = completed_keys(canonical, "video_id")
    if shard_tag:
        done |= completed_keys(shard_manifest_path(canonical, shard_tag), "video_id")

    deduper = PHashDeduper(cfg.ingest.dedup.phash_max_distance)
    processed = skipped = failed = keyframes_written = 0
    with (
        ManifestWriter(
            shard_manifest_path(canonical, shard_tag), "video_id"
        ) as videos_out,
        ManifestWriter(
            shard_manifest_path(manifests_dir / SHOTS_MANIFEST, shard_tag), "video_id"
        ) as shots_out,
        ManifestWriter(
            shard_manifest_path(manifests_dir / KEYFRAMES_MANIFEST, shard_tag),
            "keyframe_id",
        ) as keyframes_out,
    ):
        for path in videos:
            video_id = path.stem
            if video_id in done:
                skipped += 1
                continue
            try:
                written = _ingest_one(
                    path, cfg, detector, deduper, videos_out, shots_out, keyframes_out
                )
                keyframes_written += written
                processed += 1
            except VideoError as exc:
                logger.error("ingest failed for %s: %s", path.name, exc)
                videos_out.append(
                    {"video_id": video_id, "status": "failed", "reason": str(exc)}
                )
                failed += 1
            if on_item is not None:
                on_item(1)
    return IngestStats(processed, skipped, failed, keyframes_written)


class IngestProcessor:
    """Per-video ingestion for the work-stealing multi-GPU path.

    Holds the shot detector, one per-worker pHash deduper, and the three open
    rank-manifest writers, so a worker can ingest videos pulled from the shared
    queue in any order. The deduper is scoped to the worker (not the whole
    corpus), matching how the previous static-shard path already deduped within
    a shard; :func:`_ingest_one` does the identical per-video work as the
    single-process :func:`run_ingest`.
    """

    def __init__(
        self,
        cfg: Config,
        detector: ShotDetector,
        shard_tag: str,
        on_item: Callable[[int], None] | None = None,
    ) -> None:
        self._cfg = cfg
        self._detector = detector
        self._on_item = on_item
        self._deduper = PHashDeduper(cfg.ingest.dedup.phash_max_distance)
        self._paths_by_id = {
            path.stem: path
            for path in discover_videos(
                cfg.paths.videos_dir, cfg.ingest.video_extensions
            )
        }
        manifests_dir = cfg.paths.manifests_dir
        canonical = manifests_dir / VIDEOS_MANIFEST
        self._done = completed_keys(canonical, "video_id")
        if shard_tag:
            self._done |= completed_keys(
                shard_manifest_path(canonical, shard_tag), "video_id"
            )
        self._videos_out = ManifestWriter(
            shard_manifest_path(canonical, shard_tag), "video_id"
        )
        self._shots_out = ManifestWriter(
            shard_manifest_path(manifests_dir / SHOTS_MANIFEST, shard_tag), "video_id"
        )
        self._keyframes_out = ManifestWriter(
            shard_manifest_path(manifests_dir / KEYFRAMES_MANIFEST, shard_tag),
            "keyframe_id",
        )
        self._processed = self._skipped = self._failed = 0
        self._keyframes_written = 0

    def process(self, video_id: str) -> None:
        if video_id in self._done:
            self._skipped += 1
            return
        path = self._paths_by_id[video_id]
        try:
            self._keyframes_written += _ingest_one(
                path,
                self._cfg,
                self._detector,
                self._deduper,
                self._videos_out,
                self._shots_out,
                self._keyframes_out,
            )
            self._processed += 1
        except VideoError as exc:
            logger.error("ingest failed for %s: %s", path.name, exc)
            self._videos_out.append(
                {"video_id": video_id, "status": "failed", "reason": str(exc)}
            )
            self._failed += 1
        if self._on_item is not None:
            self._on_item(1)

    def close(self) -> dict[str, int]:
        self._videos_out.close()
        self._shots_out.close()
        self._keyframes_out.close()
        return {
            "processed": self._processed,
            "skipped": self._skipped,
            "failed": self._failed,
            "keyframes": self._keyframes_written,
        }


def _ingest_one(
    path: Path,
    cfg: Config,
    detector: ShotDetector,
    deduper: PHashDeduper,
    videos_out: ManifestWriter,
    shots_out: ManifestWriter,
    keyframes_out: ManifestWriter,
) -> int:
    info = probe_video(path)
    logger.debug(
        "ingesting %s (%.1fs, vfr=%s, audio=%s)",
        info.video_id,
        info.duration_ms / 1000,
        info.is_vfr,
        info.has_audio,
    )

    detect_width, detect_height = detector.frame_size
    pts_ms: list[int] = []
    lowres: list[np.ndarray] = []
    for ts, frame in iter_frames(path, width=detect_width, height=detect_height):
        pts_ms.append(ts)
        lowres.append(frame)
    if not lowres:
        raise VideoError(f"{path}: no decodable frames")
    lowres_array = np.stack(lowres)

    shots = detector.detect(lowres_array, pts_ms)

    targets: dict[int, tuple[int, int]] = {}
    for shot in shots:
        for frame_idx in plan_keyframes(
            shot, pts_ms, lowres_array, cfg.ingest.keyframes
        ):
            targets[pts_ms[frame_idx]] = (shot.shot_id, frame_idx)

    images = extract_frames_at(
        path, sorted(targets), max_side=cfg.ingest.keyframes.max_side
    )

    keyframes_dir = cfg.paths.keyframes_dir / info.video_id
    keyframes_dir.mkdir(parents=True, exist_ok=True)

    def write_keyframe(ts: int, shot_id: int, frame_idx: int, image) -> None:
        keyframe_id = make_keyframe_id(info.video_id, shot_id, frame_idx)
        image_path = keyframes_dir / f"{keyframe_id}.jpg"
        Image.fromarray(image).save(
            image_path, quality=cfg.ingest.keyframes.jpeg_quality
        )
        keyframes_out.append(
            {
                "keyframe_id": keyframe_id,
                "video_id": info.video_id,
                "shot_id": shot_id,
                "frame_idx": frame_idx,
                "timestamp_ms": ts,
                "image_path": str(image_path),
            }
        )

    written = 0
    dropped_blank = dropped_dup = 0
    written_by_shot: dict[int, int] = {}
    # First duplicate-dropped frame per shot, kept as a salvage candidate so
    # a shot never ends up with zero keyframes (an unretrievable shot).
    salvage: dict[int, tuple[int, int, np.ndarray]] = {}
    for ts in sorted(targets):
        shot_id, frame_idx = targets[ts]
        image = images.get(ts)
        if image is None:
            logger.warning("%s: no frame extracted near %d ms", info.video_id, ts)
            continue
        if is_low_information(image, cfg.ingest.dedup.min_entropy_bits):
            dropped_blank += 1
            continue
        if deduper.is_duplicate(info.video_id, image):
            if shot_id not in salvage:
                salvage[shot_id] = (ts, frame_idx, image)
            dropped_dup += 1
            continue
        write_keyframe(ts, shot_id, frame_idx, image)
        written_by_shot[shot_id] = written_by_shot.get(shot_id, 0) + 1
        written += 1

    if cfg.ingest.dedup.keep_one_per_shot:
        # De-duplication may have emptied a shot entirely (its visuals repeat
        # elsewhere); keep its first non-blank frame so the shot stays
        # reachable by the visual and caption channels. Blank-only shots
        # (entropy gate) still get nothing.
        for shot_id, (ts, frame_idx, image) in sorted(salvage.items()):
            if written_by_shot.get(shot_id, 0) > 0:
                continue
            write_keyframe(ts, shot_id, frame_idx, image)
            written += 1
            dropped_dup -= 1

    for shot in shots:
        shots_out.append(
            {
                "video_id": info.video_id,
                "shot_id": shot.shot_id,
                "start_ms": shot.start_ms,
                "end_ms": shot.end_ms,
                "boundary_confidence": shot.boundary_confidence,
            }
        )
    videos_out.append(
        {
            "video_id": info.video_id,
            "status": "done",
            "path": info.path,
            "duration_ms": info.duration_ms,
            "fps_average": info.fps_average,
            "has_audio": info.has_audio,
            "is_vfr": info.is_vfr,
            "n_shots": len(shots),
            "n_keyframes": written,
            "n_dropped_blank": dropped_blank,
            "n_dropped_duplicate": dropped_dup,
        }
    )
    logger.info(
        "%s: %d shots, %d keyframes (%d blank, %d duplicates dropped)",
        info.video_id,
        len(shots),
        written,
        dropped_blank,
        dropped_dup,
    )
    return written
