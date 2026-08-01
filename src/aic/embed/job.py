"""Resumable keyframe-embedding job.

Reads the keyframes manifest from Phase 2, encodes images in batches, and
writes embedding shards (safetensors or .npy, per ``embed.shard_format``) plus
an embeddings manifest recording, for every keyframe, its shard, row, and the
encoder ``model_id`` — the provenance the index build and the version guard
rely on.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from aic.config import Config
from aic.embed.encoders import ImageTextEncoder
from aic.embed.shard_io import load_shard, next_shard_index, save_shard
from aic.ingest.pipeline import KEYFRAMES_MANIFEST
from aic.manifest import (
    ManifestWriter,
    completed_keys,
    read_manifest,
    shard_manifest_path,
)

logger = logging.getLogger(__name__)

EMBEDDINGS_MANIFEST = "manifest.jsonl"


def model_slug(model_id: str) -> str:
    """Filesystem-safe directory name for a model id."""
    return model_id.replace("/", "--")


def embeddings_dir_for(cfg: Config, model_id: str) -> Path:
    return cfg.paths.embeddings_dir / "keyframes" / model_slug(model_id)


@dataclass(frozen=True)
class EmbedStats:
    embedded: int
    skipped: int
    failed: int


def run_embed_keyframes(
    cfg: Config,
    encoder: ImageTextEncoder,
    limit: int | None = None,
    *,
    video_filter: set[str] | None = None,
    shard_tag: str = "",
    on_item: Callable[[int], None] | None = None,
) -> EmbedStats:
    """Embed every not-yet-embedded keyframe image.

    ``video_filter`` restricts the run to keyframes of those video ids (a
    data-parallel worker's shard); ``shard_tag`` namespaces both the manifest
    and the ``.npy`` shard files so concurrent workers never collide;
    ``on_item`` is called with the batch size once per batch for a progress
    bar. The defaults reproduce the original single-process behaviour.
    """
    out_dir = embeddings_dir_for(cfg, encoder.model_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    canonical = out_dir / EMBEDDINGS_MANIFEST
    manifest_path = shard_manifest_path(canonical, shard_tag)

    records = list(read_manifest(cfg.paths.manifests_dir / KEYFRAMES_MANIFEST))
    in_scope = [
        r
        for r in records
        if video_filter is None or r["video_id"] in video_filter
    ]
    done = completed_keys(canonical, "keyframe_id")
    if shard_tag:
        done |= completed_keys(manifest_path, "keyframe_id")
    pending = [r for r in in_scope if r["keyframe_id"] not in done]
    if limit is not None:
        pending = pending[:limit]
    if not pending:
        return EmbedStats(embedded=0, skipped=len(in_scope), failed=0)

    # A worker counts only its own shards so its indices stay contiguous; the
    # tag keeps names unique across workers writing into the same directory.
    shard_prefix = f"shard-{shard_tag}-" if shard_tag else "shard-"
    next_shard = next_shard_index(out_dir, shard_prefix)

    with ManifestWriter(manifest_path, "keyframe_id") as manifest:
        embedded, failed, _ = _embed_records(
            encoder, out_dir, shard_prefix, next_shard, manifest, pending,
            cfg.embed.batch_size, cfg.embed.shard_format, cfg.embed.store_dtype,
            on_item,
        )
    return EmbedStats(
        embedded=embedded, skipped=len(in_scope) - len(pending), failed=failed
    )


def _embed_records(
    encoder: ImageTextEncoder,
    out_dir: Path,
    shard_prefix: str,
    next_shard: int,
    manifest: ManifestWriter,
    records: list[dict],
    batch_size: int,
    shard_format: str,
    store_dtype: str | None,
    on_item: Callable[[int], None] | None,
) -> tuple[int, int, int]:
    """Encode ``records`` in batches, saving one shard per batch.

    Returns ``(embedded, failed, next_shard)``. Shared by the single-process
    :func:`run_embed_keyframes` (which passes the whole pending list, so
    batches pack across videos) and the multi-GPU :class:`EmbedProcessor`
    (which passes one video's frames at a time); both write identical rows and
    rank-namespaced shards, so a sharded-then-merged run matches a solo run.
    The shard filename recorded in the manifest carries its extension, so the
    format is per-shard and the loader round-trips it.
    """
    embedded = failed = 0
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        images, kept = [], []
        for record in batch:
            try:
                with Image.open(record["image_path"]) as img:
                    images.append(np.array(img.convert("RGB")))
                kept.append(record)
            except OSError as exc:
                logger.error(
                    "cannot read keyframe image %s: %s", record["image_path"], exc
                )
                failed += 1
        if not kept:
            if on_item is not None:
                on_item(len(batch))
            continue
        vectors = encoder.encode_images(images)
        shard_name = save_shard(
            out_dir, f"{shard_prefix}{next_shard:05d}", vectors,
            shard_format, store_dtype,
        )
        next_shard += 1
        for row, record in enumerate(kept):
            manifest.append(
                {
                    "keyframe_id": record["keyframe_id"],
                    "video_id": record["video_id"],
                    "shot_id": record["shot_id"],
                    "timestamp_ms": record["timestamp_ms"],
                    "shard": shard_name,
                    "row": row,
                    "model_id": encoder.model_id,
                    "dim": int(vectors.shape[1]),
                }
            )
        embedded += len(kept)
        if on_item is not None:
            on_item(len(batch))
    return embedded, failed, next_shard


class EmbedProcessor:
    """Per-video keyframe embedding for the work-stealing multi-GPU path.

    Reads the keyframes manifest once, groups each video's still-pending frames
    in memory, and holds one open rank-manifest writer so a worker can embed
    videos pulled from the shared queue in any order without re-scanning the
    manifest. Batching is per video; with the small embed batch sizes used in
    practice a video's frames still fill batches, and the rows/shards written
    match :func:`run_embed_keyframes`, so a merged run reproduces a solo run.
    """

    def __init__(
        self,
        cfg: Config,
        encoder: ImageTextEncoder,
        shard_tag: str,
        on_item: Callable[[int], None] | None = None,
    ) -> None:
        self._encoder = encoder
        self._on_item = on_item
        self._batch_size = cfg.embed.batch_size
        self._shard_format = cfg.embed.shard_format
        self._store_dtype = cfg.embed.store_dtype
        self._out_dir = embeddings_dir_for(cfg, encoder.model_id)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        canonical = self._out_dir / EMBEDDINGS_MANIFEST
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
        self._shard_prefix = f"shard-{shard_tag}-" if shard_tag else "shard-"
        self._next_shard = next_shard_index(self._out_dir, self._shard_prefix)
        self._writer = ManifestWriter(manifest_path, "keyframe_id")
        self._embedded = self._skipped = self._failed = 0

    def process(self, video_id: str) -> None:
        self._skipped += self._skipped_by_video.get(video_id, 0)
        records = self._pending.get(video_id, [])
        embedded, failed, self._next_shard = _embed_records(
            self._encoder, self._out_dir, self._shard_prefix, self._next_shard,
            self._writer, records, self._batch_size, self._shard_format,
            self._store_dtype, self._on_item,
        )
        self._embedded += embedded
        self._failed += failed

    def close(self) -> dict[str, int]:
        self._writer.close()
        return {
            "embedded": self._embedded,
            "skipped": self._skipped,
            "failed": self._failed,
        }


def load_embeddings(cfg: Config, model_id: str) -> tuple[np.ndarray, list[dict]]:
    """Load all shards back into ``(vectors, records)`` aligned row-for-row.

    Streams one shard at a time into a preallocated float32 matrix, so peak
    memory is the output plus a single shard — holding every shard in a dict
    and then stacking doubled the footprint at full-corpus scale.
    """
    out_dir = embeddings_dir_for(cfg, model_id)
    records = list(read_manifest(out_dir / EMBEDDINGS_MANIFEST))
    if not records:
        raise FileNotFoundError(
            f"no embeddings manifest under {out_dir}; run the embed job first"
        )
    dim = int(records[0]["dim"])
    rows_by_shard: dict[str, list[tuple[int, int]]] = {}
    for out_row, record in enumerate(records):
        if record["model_id"] != model_id:
            raise ValueError(
                f"embeddings manifest row for {record['keyframe_id']} was "
                f"produced by {record['model_id']!r}, expected {model_id!r}"
            )
        if int(record["dim"]) != dim:
            raise ValueError(
                f"embeddings manifest row for {record['keyframe_id']} has "
                f"dim {record['dim']}, expected {dim}"
            )
        rows_by_shard.setdefault(record["shard"], []).append(
            (out_row, record["row"])
        )
    # float32 regardless of the stored (possibly float16) shard dtype.
    vectors = np.empty((len(records), dim), dtype=np.float32)
    for shard, pairs in rows_by_shard.items():
        array = load_shard(out_dir / shard)
        for out_row, shard_row in pairs:
            vectors[out_row] = array[shard_row]
    return vectors, records
