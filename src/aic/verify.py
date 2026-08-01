"""Artifact integrity check and repair before a resumable offline run.

An interrupted run (a killed Colab session, a Ctrl-C, a power loss) can leave
the corpus in a state the manifests alone do not describe: a truncated keyframe
image or embedding shard that a manifest still lists as done, duplicate rows
from a stage that was re-run after a partial pass, or a downstream record that
points at an upstream artifact that has since gone. The resume machinery trusts
the manifest, so it would skip these broken items forever.

``verify_and_repair`` walks the persisted artifacts in dependency order
(ingest -> embed -> OCR/caption/ASR), opens and decodes every file a manifest
references, and drops any manifest row whose artifact is missing or corrupt
(deleting the bad file). Because each stage also drops rows whose *input* no
longer exists, a repair upstream cascades downstream automatically: removing a
broken video's keyframes makes the embed/OCR/caption rows that referenced them
orphans, and they are dropped too. After the pass, the normal resume redoes
exactly the removed work plus whatever was still pending; the Chronicle and the
vector index are rebuilt from scratch on their next run, so they need no special
handling here.

Scope: this operates on the **canonical (merged) manifests**. If a multi-GPU
run was interrupted before its merge step (leftover ``name.rankK.jsonl`` files),
re-run that stage first — the workers resume from their shard files and merge —
then verify. Only the currently configured embedding model's directory is
checked (``embed.model_id``).

It is CPU-only and imports no model libraries. Gated by ``verify.on_resume``
because the decode check reads every artifact once (a full-corpus pass); enable
it after a known-bad interruption, not on every resume.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from aic.chronicle.jobs import ASR_MANIFEST, CAPTIONS_MANIFEST, OCR_MANIFEST
from aic.config import Config
from aic.embed.job import EMBEDDINGS_MANIFEST, embeddings_dir_for
from aic.embed.shard_io import iter_shard_files, load_shard
from aic.ingest.pipeline import (
    KEYFRAMES_MANIFEST,
    SHOTS_MANIFEST,
    VIDEOS_MANIFEST,
)
from aic.manifest import read_manifest, rewrite_manifest

logger = logging.getLogger(__name__)


@dataclass
class RepairReport:
    """What the pass removed, so the caller can log it and know work was requeued."""

    broken_videos: list[str] = field(default_factory=list)
    corrupt_shards: list[str] = field(default_factory=list)
    dropped_keyframe_rows: int = 0
    dropped_embed_rows: int = 0
    dropped_ocr_rows: int = 0
    dropped_caption_rows: int = 0
    dropped_asr_rows: int = 0

    @property
    def total_repairs(self) -> int:
        return (
            len(self.broken_videos)
            + len(self.corrupt_shards)
            + self.dropped_embed_rows
            + self.dropped_ocr_rows
            + self.dropped_caption_rows
            + self.dropped_asr_rows
        )


def _image_ok(path: Path) -> bool:
    """True if ``path`` exists and decodes fully (catches truncated writes)."""
    if not path.is_file():
        return False
    try:
        with Image.open(path) as img:
            img.load()  # force full pixel decode; a torn JPEG raises here
    except (OSError, ValueError, UnidentifiedImageError):
        return False
    return True


def _load_shard_safe(path: Path) -> np.ndarray | None:
    """Return the shard array at ``path``, or None if missing/unreadable.

    Format-agnostic (safetensors or .npy); a truncated or corrupt file raises,
    which is caught and reported as missing so the repair drops its rows.
    """
    if not path.is_file():
        return None
    try:
        return load_shard(path)
    except (OSError, ValueError, EOFError):
        return None


def _broken_ingest_videos(cfg: Config) -> tuple[set[str], list[dict]]:
    """Video ids whose ingest artifacts are corrupt, duplicated, or partial.

    Returns ``(broken_videos, keyframe_rows)``. A video is broken if it is
    marked done but any of its keyframe images is missing or won't decode, or
    its keyframe manifest has duplicate ids (a re-run after a partial pass), or
    it has keyframe/shot rows without a done marker (an interrupted video), or
    its videos-manifest row records a failed ingest (dropping the row requeues
    the video: the failure may have been transient, e.g. a network-drive
    hiccup, and the ordinary resume skips every recorded video id).
    """
    manifests = cfg.paths.manifests_dir
    done_videos: set[str] = set()
    failed: set[str] = set()
    for row in read_manifest(manifests / VIDEOS_MANIFEST):
        if row.get("status") == "done":
            done_videos.add(row["video_id"])
        else:
            failed.add(row["video_id"])
    keyframe_rows = list(read_manifest(manifests / KEYFRAMES_MANIFEST))
    shot_video_ids = {
        row["video_id"] for row in read_manifest(manifests / SHOTS_MANIFEST)
    }

    id_counts = Counter(row["keyframe_id"] for row in keyframe_rows)
    duplicated = {
        row["video_id"] for row in keyframe_rows if id_counts[row["keyframe_id"]] > 1
    }
    referenced_video_ids = {row["video_id"] for row in keyframe_rows} | shot_video_ids
    partial = (referenced_video_ids - done_videos) | failed

    corrupt: set[str] = set()
    already_flagged = duplicated | partial
    for row in keyframe_rows:
        video_id = row["video_id"]
        if video_id in already_flagged or video_id in corrupt:
            continue
        if video_id in done_videos and not _image_ok(Path(row["image_path"])):
            logger.warning(
                "verify: keyframe image %s of video %s is missing/corrupt",
                row["image_path"],
                video_id,
            )
            corrupt.add(video_id)

    return duplicated | partial | corrupt, keyframe_rows


def _repair_ingest(
    cfg: Config, report: RepairReport
) -> tuple[set[str], set[str], set[str]]:
    """Drop broken videos' rows and keyframe images; return the surviving sets.

    Returns ``(surviving_keyframe_ids, surviving_shot_keys, surviving_done_videos)``
    for the downstream stages to cascade against.
    """
    manifests = cfg.paths.manifests_dir
    broken, keyframe_rows = _broken_ingest_videos(cfg)

    if broken:
        report.broken_videos = sorted(broken)
        for video_id in broken:
            keyframe_dir = cfg.paths.keyframes_dir / video_id
            for image in keyframe_dir.glob("*.jpg"):
                image.unlink()
            if keyframe_dir.is_dir():
                keyframe_dir.rmdir()
        keep = lambda row: row["video_id"] not in broken  # noqa: E731
        rewrite_manifest(manifests / VIDEOS_MANIFEST, keep)
        rewrite_manifest(manifests / SHOTS_MANIFEST, keep)
        report.dropped_keyframe_rows = rewrite_manifest(
            manifests / KEYFRAMES_MANIFEST, keep
        )

    surviving_keyframes = {
        row["keyframe_id"] for row in keyframe_rows if row["video_id"] not in broken
    }
    surviving_shot_keys = {
        f"{row['video_id']}:{row['shot_id']}"
        for row in read_manifest(manifests / SHOTS_MANIFEST)
    }
    surviving_done_videos = {
        row["video_id"]
        for row in read_manifest(manifests / VIDEOS_MANIFEST)
        if row.get("status") == "done"
    }
    return surviving_keyframes, surviving_shot_keys, surviving_done_videos


def _repair_embed(
    cfg: Config, surviving_keyframes: set[str], report: RepairReport
) -> None:
    """Drop embed rows that are orphaned or backed by a corrupt/missing shard."""
    out_dir = embeddings_dir_for(cfg, cfg.embed.model_id)
    manifest_path = out_dir / EMBEDDINGS_MANIFEST
    if not manifest_path.is_file():
        return
    rows = list(read_manifest(manifest_path))

    # First cut: rows whose keyframe is gone (cascade from an ingest repair).
    live = [row for row in rows if row["keyframe_id"] in surviving_keyframes]

    # Second cut: rows whose shard will not load or is too short (truncated).
    by_shard: dict[str, list[dict]] = {}
    for row in live:
        by_shard.setdefault(row["shard"], []).append(row)
    corrupt_shards: set[str] = set()
    for shard, shard_rows in by_shard.items():
        array = _load_shard_safe(out_dir / shard)
        max_row = max(row["row"] for row in shard_rows)
        if array is None or max_row >= len(array):
            logger.warning("verify: embedding shard %s is missing/corrupt", shard)
            corrupt_shards.add(shard)
    final = [row for row in live if row["shard"] not in corrupt_shards]

    report.corrupt_shards = sorted(corrupt_shards)
    if len(final) != len(rows):
        # Set-based membership: a row survives iff its (keyframe_id, shard,
        # row) triple survived both cuts. List membership here was quadratic
        # and stalled for hours on a full-corpus embed manifest.
        final_keys = {(row["keyframe_id"], row["shard"], row["row"]) for row in final}
        report.dropped_embed_rows = rewrite_manifest(
            manifest_path,
            lambda row: (row["keyframe_id"], row["shard"], row["row"]) in final_keys,
        )

    # Delete every shard no longer referenced by a surviving row: the corrupt
    # ones, plus any orphaned wholly by the row drops (e.g. a truncated shard an
    # interrupted embed left behind, never entered in the manifest is untouched;
    # this removes ones that were referenced but are now dead).
    referenced = {row["shard"] for row in final}
    for shard_file in iter_shard_files(out_dir):
        if shard_file.name not in referenced:
            shard_file.unlink()


def _drop_orphans(path: Path, key: str, alive: Iterable[str]) -> int:
    """Drop manifest rows whose ``key`` value is not in ``alive``."""
    alive_set = set(alive)
    return rewrite_manifest(path, lambda row: row.get(key) in alive_set)


def verify_and_repair(cfg: Config) -> RepairReport:
    """Check every persisted artifact and repair the corpus for a clean resume.

    See the module docstring for the model. Safe to call when there is nothing
    to repair (it then only reads); returns a :class:`RepairReport` describing
    what, if anything, was removed and thereby requeued.
    """
    report = RepairReport()
    manifests = cfg.paths.manifests_dir

    surviving_keyframes, surviving_shot_keys, surviving_videos = _repair_ingest(
        cfg, report
    )
    _repair_embed(cfg, surviving_keyframes, report)
    report.dropped_ocr_rows = _drop_orphans(
        manifests / OCR_MANIFEST, "keyframe_id", surviving_keyframes
    )
    report.dropped_caption_rows = _drop_orphans(
        manifests / CAPTIONS_MANIFEST, "shot_key", surviving_shot_keys
    )
    report.dropped_asr_rows = _drop_orphans(
        manifests / ASR_MANIFEST, "video_id", surviving_videos
    )

    if report.total_repairs:
        logger.info(
            "verify: repaired corpus — %d video(s) requeued, %d corrupt shard(s), "
            "dropped rows: embed=%d ocr=%d caption=%d asr=%d",
            len(report.broken_videos),
            len(report.corrupt_shards),
            report.dropped_embed_rows,
            report.dropped_ocr_rows,
            report.dropped_caption_rows,
            report.dropped_asr_rows,
        )
    else:
        logger.info("verify: corpus clean, nothing to repair")
    return report


def maybe_repair(cfg: Config) -> RepairReport | None:
    """Run :func:`verify_and_repair` only when ``verify.on_resume`` is enabled.

    The entry-point scripts call this before their stage so an opted-in run
    heals a corpus left inconsistent by a previous interruption; with the flag
    off (the default) it is a no-op and the resume proceeds unchanged.
    """
    if not cfg.verify.on_resume:
        return None
    logger.info("verify.on_resume is set: checking artifact integrity before resume")
    return verify_and_repair(cfg)
