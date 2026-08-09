"""Lightweight keyframe metadata loading for retrieval-only services.

The normal embedding loader reconstructs the full vector matrix from every
shard because index-building jobs need those vectors.  A serving process that
already has a persisted vector index only needs the manifest fields used to
map search hits back to videos, shots, and timestamps.
"""

from __future__ import annotations

from aic.config import Config
from aic.embed.job import EMBEDDINGS_MANIFEST, embeddings_dir_for
from aic.manifest import read_manifest


def load_keyframe_meta_manifest(
    cfg: Config, model_id: str
) -> dict[str, tuple[str, int, int]]:
    """Read keyframe metadata without opening any embedding shard.

    The persisted vector index remains the source of searchable vectors.  The
    manifest is only used for ``keyframe_id -> (video, shot, timestamp)``.
    Model provenance and duplicate IDs are checked so a stale or incorrectly
    merged manifest still fails loudly at service startup.
    """

    manifest_path = embeddings_dir_for(cfg, model_id) / EMBEDDINGS_MANIFEST
    metadata: dict[str, tuple[str, int, int]] = {}
    for record in read_manifest(manifest_path):
        keyframe_id = str(record["keyframe_id"])
        record_model = str(record["model_id"])
        if record_model != model_id:
            raise ValueError(
                f"embeddings manifest row for {keyframe_id} was produced by "
                f"{record_model!r}, expected {model_id!r}"
            )
        if keyframe_id in metadata:
            raise ValueError(
                f"duplicate keyframe_id in embeddings manifest: {keyframe_id}"
            )
        metadata[keyframe_id] = (
            str(record["video_id"]),
            int(record["shot_id"]),
            int(record["timestamp_ms"]),
        )
    if not metadata:
        raise FileNotFoundError(
            f"no embeddings manifest under {manifest_path.parent}; "
            "run the embed job first"
        )
    return metadata
