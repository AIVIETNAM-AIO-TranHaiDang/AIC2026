"""Build the Chronicle text indexes: dense (semantic) and sparse (literal).

Two document views per shot (see :class:`aic.chronicle.schema.ChronicleRecord`):
``semantic_text`` (captions + speech, dense channel) and ``literal_text``
(on-screen text + speech, sparse channel). Shots whose view is empty are
simply absent from that index — a missing channel entry is a legal degraded
state, never an error.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from aic.chronicle.jobs import load_chronicle
from aic.config import Config
from aic.index.sparse import SparseIndex
from aic.index.vector import VectorIndex
from aic.textstack.embedder import TextEmbedder

logger = logging.getLogger(__name__)

SEMANTIC_INDEX_DIR = "chronicle-semantic"
LITERAL_INDEX_DIR = "chronicle-literal"
OVERLAY_INDEX_DIR = "chronicle-overlay"
FACTOID_INDEX_DIR = "chronicle-factoid"


@dataclass(frozen=True)
class TextIndexStats:
    semantic_docs: int
    literal_docs: int
    total_shots: int
    overlay_docs: int = 0
    factoid_docs: int = 0


def semantic_index_dir(cfg: Config) -> Path:
    return cfg.paths.indexes_dir / SEMANTIC_INDEX_DIR


def literal_index_dir(cfg: Config) -> Path:
    return cfg.paths.indexes_dir / LITERAL_INDEX_DIR


def overlay_index_dir(cfg: Config) -> Path:
    return cfg.paths.indexes_dir / OVERLAY_INDEX_DIR


def factoid_index_dir(cfg: Config) -> Path:
    return cfg.paths.indexes_dir / FACTOID_INDEX_DIR


def build_text_indexes(
    cfg: Config,
    dense_embedder: TextEmbedder,
    sparse_embedder: TextEmbedder | None = None,
) -> TextIndexStats:
    """Embed chronicle documents and persist both indexes.

    ``dense_embedder`` builds the semantic vector index; ``sparse_embedder``
    (defaulting to the dense one, the single-BGE-M3 case) must produce
    learned-sparse weights for the literal index.
    """
    if sparse_embedder is None:
        sparse_embedder = dense_embedder
    records = load_chronicle(cfg)
    if not records:
        raise FileNotFoundError(
            "chronicle is empty; run the chronicle jobs and assembler first"
        )

    semantic_ids, semantic_docs = [], []
    literal_ids, literal_docs = [], []
    overlay_by_video: dict[str, str] = {}
    for record in records:
        semantic = record.semantic_text()
        if semantic:
            semantic_ids.append(record.shot_key)
            semantic_docs.append(semantic)
        literal = record.literal_text()
        if literal:
            literal_ids.append(record.shot_key)
            literal_docs.append(literal)
        # The overlay list is identical on every row of a video (the
        # assembler stamps the per-video split onto each shot), so the
        # first non-empty row provides the video's document.
        if record.video_id not in overlay_by_video:
            overlay = record.overlay_text()
            if overlay:
                overlay_by_video[record.video_id] = overlay

    if semantic_docs:
        embeddings = dense_embedder.encode(semantic_docs)
        VectorIndex(
            vectors=embeddings.dense,
            ids=semantic_ids,
            model_id=dense_embedder.model_id,
            backend=cfg.index.backend,
        ).save(semantic_index_dir(cfg))
    if literal_docs:
        embeddings = sparse_embedder.encode(literal_docs)
        if len(embeddings.sparse) != len(literal_docs):
            raise ValueError(
                f"sparse embedder {sparse_embedder.model_id!r} is dense-only; "
                "the literal index needs a learned-sparse embedder (bge_m3)"
            )
        SparseIndex(weights=embeddings.sparse, ids=literal_ids).save(
            literal_index_dir(cfg)
        )
    if overlay_by_video:
        overlay_ids = sorted(overlay_by_video)
        overlay_docs = [overlay_by_video[video_id] for video_id in overlay_ids]
        embeddings = sparse_embedder.encode(overlay_docs)
        if len(embeddings.sparse) == len(overlay_docs):
            SparseIndex(weights=embeddings.sparse, ids=overlay_ids).save(
                overlay_index_dir(cfg)
            )
        else:
            logger.warning(
                "sparse embedder %s is dense-only; overlay video index skipped",
                sparse_embedder.model_id,
            )

    # Answer Ledger factoid index (VQA Track A): one flat document per shot
    # with text, indexed sparsely like the literal channel. Off unless the
    # ledger is enabled; empty rows (no screen text, no entities) are skipped.
    factoid_docs_n = 0
    if cfg.chronicle.ledger.enabled:
        from aic.chronicle.ledger import load_ledger

        factoid_ids, factoid_docs = [], []
        for row in load_ledger(cfg):
            doc = row.factoid_text()
            if doc:
                factoid_ids.append(row.shot_key)
                factoid_docs.append(doc)
        if factoid_docs:
            embeddings = sparse_embedder.encode(factoid_docs)
            if len(embeddings.sparse) == len(factoid_docs):
                SparseIndex(weights=embeddings.sparse, ids=factoid_ids).save(
                    factoid_index_dir(cfg)
                )
                factoid_docs_n = len(factoid_docs)
            else:
                logger.warning(
                    "sparse embedder %s is dense-only; factoid index skipped",
                    sparse_embedder.model_id,
                )

    logger.info(
        "text indexes built: %d semantic docs, %d literal docs, "
        "%d overlay video docs, %d factoid docs over %d shots",
        len(semantic_docs),
        len(literal_docs),
        len(overlay_by_video),
        factoid_docs_n,
        len(records),
    )
    return TextIndexStats(
        semantic_docs=len(semantic_docs),
        literal_docs=len(literal_docs),
        total_shots=len(records),
        overlay_docs=len(overlay_by_video),
        factoid_docs=factoid_docs_n,
    )
