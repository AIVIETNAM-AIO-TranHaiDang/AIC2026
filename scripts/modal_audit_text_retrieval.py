"""Audit Chronicle text retrieval by channel on Modal.

This diagnostic does not modify any corpus artifact.  It loads the persisted
BGE-M3 semantic/literal indexes, ranks a query in each channel independently,
and compares the ranking with exact OCR/ASR occurrences in Chronicle.

Example::

    modal run scripts/modal_audit_text_retrieval.py --query "NGUYÊN LIỆU" \
        --expected-shots "L26_V078:18,L26_V078:94,L26_V078:95"
"""

from __future__ import annotations

import os
import sys
import tempfile
import unicodedata
from pathlib import Path

import modal

LOCAL_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/opt/aic")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(str(LOCAL_ROOT / "requirements.txt"))
    .add_local_dir(LOCAL_ROOT / "src", remote_path=str(REMOTE_ROOT / "src"), copy=True)
    .add_local_dir(
        LOCAL_ROOT / "configs", remote_path=str(REMOTE_ROOT / "configs"), copy=True
    )
)

app = modal.App("aic-audit-text-retrieval")
results = modal.Volume.from_name("aic-embeddings", create_if_missing=False)
model_cache = modal.Volume.from_name("aic-model-cache", create_if_missing=False)


def _normalise(text: str) -> str:
    return unicodedata.normalize("NFC", text).casefold().strip()


def _rank_map(hits: list[tuple[str, float]]) -> dict[str, int]:
    return {shot_key: rank for rank, (shot_key, _) in enumerate(hits, start=1)}


def _channel_report(
    hits: list[tuple[str, float]],
    exact_ocr: set[str],
    exact_asr: set[str],
    expected_shots: list[str],
    records_by_key: dict[str, object],
) -> dict[str, object]:
    ranks = _rank_map(hits)
    top = []
    for rank, (shot_key, score) in enumerate(hits[:20], start=1):
        record = records_by_key[shot_key]
        top.append(
            {
                "rank": rank,
                "shot_key": shot_key,
                "score": round(score, 6),
                "exact_ocr": shot_key in exact_ocr,
                "exact_asr": shot_key in exact_asr,
                "ocr": [line.text for line in record.ocr],
                "asr": record.asr_text,
            }
        )

    def first_rank(keys: set[str]) -> int | None:
        return min((ranks[key] for key in keys if key in ranks), default=None)

    return {
        "first_exact_ocr_rank": first_rank(exact_ocr),
        "first_exact_asr_rank": first_rank(exact_asr),
        "exact_ocr_in_top": {
            str(k): sum(shot_key in exact_ocr for shot_key, _ in hits[:k])
            for k in (1, 3, 5, 10, 100)
        },
        "exact_asr_in_top": {
            str(k): sum(shot_key in exact_asr for shot_key, _ in hits[:k])
            for k in (1, 3, 5, 10, 100)
        },
        "expected_ranks": {key: ranks.get(key) for key in expected_shots},
        "top20": top,
    }


@app.function(
    image=image,
    gpu="L4",
    cpu=4.0,
    memory=16384,
    timeout=30 * 60,
    volumes={"/mnt/out": results, "/mnt/models": model_cache},
)
def audit(query: str, expected_shots: list[str]) -> dict[str, object]:
    os.chdir(REMOTE_ROOT)
    sys.path.insert(0, str(REMOTE_ROOT / "src"))

    audit_root = Path(tempfile.mkdtemp(prefix="aic-text-audit-"))
    (audit_root / "manifests").symlink_to(
        "/mnt/out/manifests", target_is_directory=True
    )
    (audit_root / "indexes").symlink_to(
        "/mnt/out/indexes", target_is_directory=True
    )

    from aic.chronicle.jobs import load_chronicle
    from aic.config import load_config
    from aic.index.sparse import SparseIndex
    from aic.index.vector import VectorIndex
    from aic.models_cache import apply_model_cache_env
    from aic.textstack.embedder import build_textstack_embedders
    from aic.textstack.job import literal_index_dir, semantic_index_dir

    cfg = load_config(REMOTE_ROOT / "configs" / "t0.yaml")
    cfg.paths.data_root = audit_root
    cfg.paths.models_dir = Path("/mnt/models")
    cfg.index.device = "cpu"
    apply_model_cache_env(cfg.paths.models_dir)

    records = load_chronicle(cfg)
    records_by_key = {record.shot_key: record for record in records}
    unknown = [key for key in expected_shots if key not in records_by_key]
    if unknown:
        raise ValueError(f"expected shots absent from Chronicle: {unknown}")

    needle = _normalise(query)
    exact_ocr = {
        record.shot_key
        for record in records
        if any(needle in _normalise(line.text) for line in record.ocr)
    }
    exact_asr = {
        record.shot_key
        for record in records
        if record.asr_text and needle in _normalise(record.asr_text)
    }

    dense_embedder, sparse_embedder = build_textstack_embedders(
        cfg.textstack, cfg.paths.models_dir
    )
    # The default BGE-M3 config returns the same object for both channels;
    # encoding once supplies both the dense vector and learned-sparse weights.
    if dense_embedder is sparse_embedder:
        encoded = dense_embedder.encode_queries([query])
        dense_encoded = sparse_encoded = encoded
    else:
        dense_encoded = dense_embedder.encode_queries([query])
        sparse_encoded = sparse_embedder.encode_queries([query])

    semantic = VectorIndex.load(semantic_index_dir(cfg), device="cpu")
    literal = SparseIndex.load(literal_index_dir(cfg))
    semantic_hits = semantic.search(
        dense_encoded.dense,
        top_k=len(semantic),
        query_model_id=dense_embedder.model_id,
    )[0]
    literal_hits = literal.search(sparse_encoded.sparse[0], top_k=len(literal))

    return {
        "query": query,
        "model_id": dense_embedder.model_id,
        "chronicle_shots": len(records),
        "exact_ocr_shots": len(exact_ocr),
        "exact_asr_shots": len(exact_asr),
        "semantic": _channel_report(
            semantic_hits,
            exact_ocr,
            exact_asr,
            expected_shots,
            records_by_key,
        ),
        "literal": _channel_report(
            literal_hits,
            exact_ocr,
            exact_asr,
            expected_shots,
            records_by_key,
        ),
    }


@app.local_entrypoint()
def main(query: str, expected_shots: str = "") -> None:
    expected = [value.strip() for value in expected_shots.split(",") if value.strip()]
    print(audit.remote(query, expected))
