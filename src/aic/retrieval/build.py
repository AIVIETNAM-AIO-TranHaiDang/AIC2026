"""Assemble the FusionRetriever from persisted artifacts.

The one place that knows where Phase 2-5 outputs live on disk; tests build
retrievers from in-memory fakes instead, and scripts call this.
"""

from __future__ import annotations

import logging
from pathlib import Path

from aic.config import Config
from aic.embed.encoders import ImageTextEncoder
from aic.embed.job import embeddings_dir_for, load_embeddings, model_slug
from aic.index.sparse import SparseIndex
from aic.index.vector import VectorIndex
from aic.ingest.pipeline import SHOTS_MANIFEST
from aic.manifest import read_manifest
from aic.retrieval.dense import DenseVisualChannel
from aic.retrieval.fusion import (
    FACTOID_SPARSE,
    Channel,
    FusionRetriever,
    LiteralSparseChannel,
    SemanticDenseChannel,
    ShotMeta,
)
from aic.retrieval.temporal import OverlayVideoPrior
from aic.textstack.embedder import TextEmbedder
from aic.textstack.job import (
    factoid_index_dir,
    literal_index_dir,
    overlay_index_dir,
    semantic_index_dir,
)

logger = logging.getLogger(__name__)


def load_overlay_prior(
    cfg: Config, sparse_embedder: TextEmbedder
) -> OverlayVideoPrior | None:
    """The overlay video prior, when configured and built.

    Missing artifacts degrade to None (no prior) with a warning only when
    the config actually asks for the prior — the usual
    degraded-but-working rule.
    """
    if cfg.retrieval.temporal.video_prior_weight <= 0:
        return None
    if not overlay_index_dir(cfg).is_dir():
        logger.warning(
            "video_prior_weight is set but no overlay index exists; run "
            "build_text_indexes with chronicle.overlay_min_recurrence set"
        )
        return None
    return OverlayVideoPrior(
        embedder=sparse_embedder,
        index=SparseIndex.load(overlay_index_dir(cfg)),
    )


def keyframe_index_dir(cfg: Config, model_id: str) -> Path:
    return cfg.paths.indexes_dir / f"keyframes-{model_slug(model_id)}"


def build_keyframe_index(cfg: Config, model_id: str) -> VectorIndex:
    """Build (and persist) the keyframe vector index from embedding shards."""
    vectors, records = load_embeddings(cfg, model_id)
    index = VectorIndex(
        vectors=vectors,
        ids=[record["keyframe_id"] for record in records],
        model_id=model_id,
        backend=cfg.index.backend,
        device=cfg.index.device,
        gpu_float16=cfg.index.gpu_float16,
    )
    index.save(keyframe_index_dir(cfg, model_id))
    return index


def load_keyframe_meta(cfg: Config, model_id: str) -> dict[str, tuple[str, int, int]]:
    """keyframe_id -> (video_id, shot_id, timestamp_ms) from the embed manifest."""
    _, records = load_embeddings(cfg, model_id)
    return {
        record["keyframe_id"]: (
            record["video_id"],
            record["shot_id"],
            record["timestamp_ms"],
        )
        for record in records
    }


def load_shot_meta(cfg: Config) -> dict[str, ShotMeta]:
    """Shot metadata from the Phase 2 shots manifest.

    Read from ingest output (not the Chronicle) so the retriever also works
    on a corpus that has embeddings but no Chronicle yet.
    """
    meta = {}
    for record in read_manifest(cfg.paths.manifests_dir / SHOTS_MANIFEST):
        key = f"{record['video_id']}:{record['shot_id']}"
        meta[key] = ShotMeta(
            video_id=record["video_id"],
            shot_id=record["shot_id"],
            t_start_ms=record["start_ms"],
            t_end_ms=record["end_ms"],
        )
    if not meta:
        raise FileNotFoundError(
            "shots manifest is empty; run ingestion (Phase 2) first"
        )
    return meta


def build_channels(
    cfg: Config,
    encoder: ImageTextEncoder,
    text_embedder: TextEmbedder,
    fixture_dir: Path | None = None,
    sparse_text_embedder: TextEmbedder | None = None,
) -> list[Channel]:
    """Instantiate every channel whose artifacts exist on disk.

    ``text_embedder`` serves the semantic dense channel;
    ``sparse_text_embedder`` (defaulting to the same object) serves the
    literal sparse channel — pass both when textstack.dense is configured.
    Channels whose artifacts are missing are skipped with a warning —
    the degraded-but-working principle. Shared by the Phase 5 fusion
    retriever and the Phase 6 spec dispatcher.
    """
    if sparse_text_embedder is None:
        sparse_text_embedder = text_embedder
    channels: list[Channel] = []

    embed_dir = embeddings_dir_for(cfg, encoder.model_id)
    if embed_dir.is_dir():
        index_dir = keyframe_index_dir(cfg, encoder.model_id)
        if index_dir.is_dir():
            index = VectorIndex.load(
                index_dir,
                device=cfg.index.device,
                gpu_float16=cfg.index.gpu_float16,
            )
        else:
            index = build_keyframe_index(cfg, encoder.model_id)
        channels.append(
            DenseVisualChannel(
                encoder=encoder,
                index=index,
                keyframe_meta=load_keyframe_meta(cfg, encoder.model_id),
                fixture_dir=fixture_dir,
                query_clip=cfg.retrieval.query_clip,
                keyframe_oversample=cfg.retrieval.keyframe_oversample,
            )
        )
    else:
        logger.warning("no keyframe embeddings found; dense visual channel off")

    if semantic_index_dir(cfg).is_dir():
        channels.append(
            SemanticDenseChannel(
                embedder=text_embedder,
                index=VectorIndex.load(
                    semantic_index_dir(cfg),
                    device=cfg.index.device,
                    gpu_float16=cfg.index.gpu_float16,
                ),
            )
        )
    else:
        logger.warning("no semantic index found; semantic dense channel off")

    if literal_index_dir(cfg).is_dir():
        channels.append(
            LiteralSparseChannel(
                embedder=sparse_text_embedder,
                index=SparseIndex.load(literal_index_dir(cfg)),
            )
        )
    else:
        logger.warning("no literal index found; literal sparse channel off")

    # Answer Ledger factoid channel (VQA Track A). Added only when its index
    # exists; inert for KIS because the FusionRetriever skips any channel that
    # is absent from the fusion weights, and KIS profiles do not weight
    # factoid_sparse — so this cannot change KIS output.
    if factoid_index_dir(cfg).is_dir():
        channels.append(
            LiteralSparseChannel(
                embedder=sparse_text_embedder,
                index=SparseIndex.load(factoid_index_dir(cfg)),
                name=FACTOID_SPARSE,
            )
        )

    if not channels:
        raise FileNotFoundError(
            "no retrieval channel has artifacts; run the embed/chronicle/text "
            "index jobs first"
        )
    return channels


def build_fusion_retriever(
    cfg: Config,
    encoder: ImageTextEncoder,
    text_embedder: TextEmbedder,
    fixture_dir: Path | None = None,
    sparse_text_embedder: TextEmbedder | None = None,
) -> FusionRetriever:
    """Wire persisted indexes into the Phase 5 online engine."""
    channels = build_channels(
        cfg, encoder, text_embedder, fixture_dir, sparse_text_embedder
    )
    weights = cfg.retrieval.fusion.channel_weights
    active = {c.name for c in channels}
    return FusionRetriever(
        channels=channels,
        weights={k: v for k, v in weights.items() if k in active},
        rrf_k=cfg.retrieval.fusion.rrf_k,
        top_k_per_channel=cfg.retrieval.fusion.top_k_per_channel,
        shot_meta=load_shot_meta(cfg),
        temporal_cfg=cfg.retrieval.temporal,
        video_prior=load_overlay_prior(
            cfg, sparse_text_embedder or text_embedder
        ),
        chunking_cfg=cfg.retrieval.visual_chunking,
    )
