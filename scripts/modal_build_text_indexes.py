"""Build Chronicle semantic/literal indexes on Modal with BGE-M3.

The source Chronicle and published indexes live in the ``aic-embeddings``
Volume.  Indexes are first built and validated on container-local storage;
only complete directories are then published.  An existing published index
is renamed to a timestamped backup instead of being deleted.

Run on the workspace that owns ``aic-embeddings``::

    modal run scripts/modal_build_text_indexes.py
"""

from __future__ import annotations

import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import modal

LOCAL_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/opt/aic")
BUILD_ROOT = Path("/tmp/aic-text-index-build")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(str(LOCAL_ROOT / "requirements.txt"))
    .add_local_dir(LOCAL_ROOT / "src", remote_path=str(REMOTE_ROOT / "src"), copy=True)
    .add_local_dir(
        LOCAL_ROOT / "configs", remote_path=str(REMOTE_ROOT / "configs"), copy=True
    )
)

app = modal.App("aic-build-text-indexes")
results = modal.Volume.from_name("aic-embeddings", create_if_missing=False)
model_cache = modal.Volume.from_name("aic-model-cache", create_if_missing=True)


def _publish_directory(source: Path, destination: Path) -> str | None:
    """Publish one complete index and preserve any previous version."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    incoming = destination.parent / f".{destination.name}.next-{uuid4().hex}"
    shutil.copytree(source, incoming)

    backup: Path | None = None
    if destination.exists():
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        backup = destination.parent / f"{destination.name}.backup-{stamp}"
        destination.rename(backup)

    try:
        incoming.rename(destination)
    except Exception:
        if backup is not None and not destination.exists():
            backup.rename(destination)
        raise
    return backup.name if backup is not None else None


@app.function(
    image=image,
    gpu="L4",
    cpu=8.0,
    memory=32768,
    timeout=6 * 60 * 60,
    max_containers=1,
    volumes={"/mnt/out": results, "/mnt/models": model_cache},
)
def build() -> dict[str, object]:
    """Build, validate, and publish all configured Chronicle text indexes."""

    os.chdir(REMOTE_ROOT)
    sys.path.insert(0, str(REMOTE_ROOT / "src"))

    if BUILD_ROOT.exists():
        shutil.rmtree(BUILD_ROOT)
    BUILD_ROOT.mkdir(parents=True)
    (BUILD_ROOT / "manifests").symlink_to(
        "/mnt/out/manifests", target_is_directory=True
    )

    chronicle_path = BUILD_ROOT / "manifests" / "chronicle.jsonl"
    if not chronicle_path.is_file():
        raise FileNotFoundError(
            "missing /mnt/out/manifests/chronicle.jsonl; assemble OCR/ASR first"
        )

    from aic.chronicle.jobs import load_chronicle
    from aic.config import load_config
    from aic.index.sparse import SparseIndex
    from aic.index.vector import VectorIndex
    from aic.models_cache import apply_model_cache_env
    from aic.textstack.embedder import build_textstack_embedders
    from aic.textstack.job import (
        build_text_indexes,
        factoid_index_dir,
        literal_index_dir,
        overlay_index_dir,
        semantic_index_dir,
    )

    cfg = load_config(REMOTE_ROOT / "configs" / "t0.yaml")
    cfg.paths.data_root = BUILD_ROOT
    cfg.paths.models_dir = Path("/mnt/models")
    cfg.index.device = "cpu"
    apply_model_cache_env(cfg.paths.models_dir)

    records = load_chronicle(cfg)
    shot_keys = [record.shot_key for record in records]
    if len(shot_keys) != len(set(shot_keys)):
        raise RuntimeError("chronicle contains duplicate (video_id, shot_id) keys")

    shots_with_ocr = sum(bool(record.ocr) for record in records)
    shots_with_asr = sum(bool(record.asr_text) for record in records)
    if shots_with_ocr == 0 or shots_with_asr == 0:
        raise RuntimeError(
            "chronicle must contain both OCR and ASR before text indexing: "
            f"ocr={shots_with_ocr}, asr={shots_with_asr}"
        )

    print(
        f"Chronicle validated: shots={len(records)}, "
        f"with_ocr={shots_with_ocr}, with_asr={shots_with_asr}",
        flush=True,
    )
    dense_embedder, sparse_embedder = build_textstack_embedders(
        cfg.textstack, cfg.paths.models_dir
    )
    stats = build_text_indexes(cfg, dense_embedder, sparse_embedder)

    semantic = VectorIndex.load(semantic_index_dir(cfg), device="cpu")
    literal = SparseIndex.load(literal_index_dir(cfg))
    if len(semantic) != stats.semantic_docs:
        raise RuntimeError(
            f"semantic index count mismatch: {len(semantic)} != {stats.semantic_docs}"
        )
    if len(literal) != stats.literal_docs:
        raise RuntimeError(
            f"literal index count mismatch: {len(literal)} != {stats.literal_docs}"
        )

    candidate_dirs = (
        semantic_index_dir(cfg),
        literal_index_dir(cfg),
        overlay_index_dir(cfg),
        factoid_index_dir(cfg),
    )
    published: dict[str, dict[str, object]] = {}
    for source in candidate_dirs:
        if not source.is_dir():
            continue
        destination = Path("/mnt/out/indexes") / source.name
        backup = _publish_directory(source, destination)
        size_bytes = sum(
            path.stat().st_size for path in source.rglob("*") if path.is_file()
        )
        published[source.name] = {"bytes": size_bytes, "backup": backup}

    results.commit()
    model_cache.commit()
    print(f"Published indexes: {sorted(published)}", flush=True)
    return {
        "shots": len(records),
        "shots_with_ocr": shots_with_ocr,
        "shots_with_asr": shots_with_asr,
        "semantic_docs": stats.semantic_docs,
        "literal_docs": stats.literal_docs,
        "overlay_docs": stats.overlay_docs,
        "factoid_docs": stats.factoid_docs,
        "model_id": dense_embedder.model_id,
        "semantic_dim": semantic.dim,
        "published": published,
    }


@app.local_entrypoint()
def main() -> None:
    print(build.remote())
