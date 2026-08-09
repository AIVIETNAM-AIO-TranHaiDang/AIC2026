from __future__ import annotations

import json
from pathlib import Path

import pytest

from aic.embed.job import EMBEDDINGS_MANIFEST, embeddings_dir_for
from aic.retrieval.fast_meta import load_keyframe_meta_manifest


class _Paths:
    def __init__(self, embeddings_dir: Path) -> None:
        self.embeddings_dir = embeddings_dir


class _Config:
    def __init__(self, embeddings_dir: Path) -> None:
        self.paths = _Paths(embeddings_dir)


def _write_manifest(root: Path, model_id: str, records: list[dict]) -> _Config:
    cfg = _Config(root / "embeddings")
    directory = embeddings_dir_for(cfg, model_id)  # type: ignore[arg-type]
    directory.mkdir(parents=True)
    (directory / EMBEDDINGS_MANIFEST).write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return cfg


def test_load_keyframe_meta_manifest_does_not_require_shards(tmp_path: Path) -> None:
    model_id = "example/encoder"
    cfg = _write_manifest(
        tmp_path,
        model_id,
        [
            {
                "keyframe_id": "L21_V001_s2_f30",
                "video_id": "L21_V001",
                "shot_id": 2,
                "timestamp_ms": 1200,
                "shard": "missing-on-purpose.safetensors",
                "row": 0,
                "model_id": model_id,
                "dim": 1152,
            }
        ],
    )

    assert load_keyframe_meta_manifest(cfg, model_id) == {  # type: ignore[arg-type]
        "L21_V001_s2_f30": ("L21_V001", 2, 1200)
    }


def test_load_keyframe_meta_manifest_rejects_model_mismatch(tmp_path: Path) -> None:
    cfg = _write_manifest(
        tmp_path,
        "example/encoder",
        [
            {
                "keyframe_id": "frame-1",
                "video_id": "video-1",
                "shot_id": 0,
                "timestamp_ms": 0,
                "model_id": "different/encoder",
            }
        ],
    )

    with pytest.raises(ValueError, match="different/encoder"):
        load_keyframe_meta_manifest(cfg, "example/encoder")  # type: ignore[arg-type]

