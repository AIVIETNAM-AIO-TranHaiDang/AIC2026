"""Production assembly of the service state from persisted artifacts.

This is the service-side sibling of :func:`aic.retrieval.build
.build_fusion_retriever`: it loads the same channels and metadata, adds the
Cortex compiler, the dense feedback engine, evidence bundles, the session
store, and the Phase 7 escalation toolbox + QPP advisor. Tests never call
this — they inject fakes into :class:`aic.service.app.ServiceState`
directly.
"""

from __future__ import annotations

import logging

from aic.config import Config
from aic.cortex.compiler import CortexCompiler
from aic.cortex.dispatch import SpecDispatcher
from aic.cortex.feedback import DenseFeedbackEngine
from aic.embed.encoders import ImageTextEncoder, build_image_text_encoder
from aic.escalate.base import Escalation
from aic.escalate.cache import ModelCache
from aic.escalate.imagine import ImagineMatch
from aic.escalate.qpp import QppAdvisor, QppError
from aic.escalate.rerank import LateInteractionReranker
from aic.escalate.verify import VlmVerify
from aic.index.vector import VectorIndex
from aic.retrieval.build import (
    build_channels,
    build_keyframe_index,
    keyframe_index_dir,
    load_keyframe_meta,
    load_overlay_prior,
    load_shot_meta,
)
from aic.service.app import ServiceState
from aic.service.bundles import EvidenceBundle, load_bundles
from aic.service.engine import OnlineEngine
from aic.service.sessions import SessionStore
from aic.textstack.embedder import TextEmbedder, build_textstack_embedders

logger = logging.getLogger(__name__)


def _keyframe_search_assets(
    cfg: Config, encoder: ImageTextEncoder
) -> tuple[VectorIndex, dict[str, tuple[str, int, int]]] | None:
    """The keyframe index + metadata, shared by feedback and Imagine->Match."""
    try:
        index_dir = keyframe_index_dir(cfg, encoder.model_id)
        if index_dir.is_dir():
            index = VectorIndex.load(
                index_dir,
                device=cfg.index.device,
                gpu_float16=cfg.index.gpu_float16,
            )
        else:
            index = build_keyframe_index(cfg, encoder.model_id)
        return index, load_keyframe_meta(cfg, encoder.model_id)
    except FileNotFoundError:
        return None


def _build_escalations(
    cfg: Config,
    encoder: ImageTextEncoder,
    sparse_embedder: TextEmbedder,
    bundles: dict[str, EvidenceBundle],
    keyframe_assets: tuple[VectorIndex, dict[str, tuple[str, int, int]]] | None,
) -> tuple[dict[str, Escalation], dict[str, float]]:
    """Every enabled escalation whose dependencies exist on disk."""
    escalations: dict[str, Escalation] = {}
    timeouts: dict[str, float] = {}
    cache = ModelCache(cfg.escalations.max_resident_models)

    rerank_cfg = cfg.escalations.rerank
    if rerank_cfg.enabled:
        if hasattr(sparse_embedder, "encode_colbert"):
            tool = LateInteractionReranker(
                embedder=sparse_embedder,
                evidence_text={k: b.evidence_text for k, b in bundles.items()},
                cfg=rerank_cfg,
            )
            escalations[tool.name] = tool
            timeouts[tool.name] = rerank_cfg.timeout_s
        else:
            logger.warning(
                "rerank escalation needs a ColBERT-capable embedder "
                "(bge_m3); disabled"
            )

    imagine_cfg = cfg.escalations.imagine
    if imagine_cfg.enabled:
        if keyframe_assets is not None:
            index, keyframe_meta = keyframe_assets
            tool = ImagineMatch(
                encoder=encoder,
                index=index,
                keyframe_meta=keyframe_meta,
                cfg=imagine_cfg,
                models_dir=cfg.paths.models_dir,
                cache=cache,
                rrf_k=cfg.retrieval.fusion.rrf_k,
            )
            escalations[tool.name] = tool
            timeouts[tool.name] = imagine_cfg.timeout_s
        else:
            logger.warning(
                "imagine escalation needs keyframe embeddings; disabled"
            )

    verify_cfg = cfg.escalations.vlm_verify
    if verify_cfg.enabled:
        tool = VlmVerify(
            bundles=bundles,
            frames_dir=cfg.paths.keyframes_dir,
            cfg=verify_cfg,
        )
        escalations[tool.name] = tool
        timeouts[tool.name] = verify_cfg.timeout_s

    return escalations, timeouts


def _load_qpp(cfg: Config) -> QppAdvisor | None:
    if cfg.qpp.trained_artifact is None:
        return None
    try:
        return QppAdvisor.load(cfg.qpp.trained_artifact, cfg.qpp)
    except QppError as exc:
        logger.warning("QPP advisor disabled: %s", exc)
        return None


def build_service_state(cfg: Config) -> ServiceState:
    """Load every artifact the online service needs."""
    encoder = build_image_text_encoder(cfg.embed, cfg.paths.models_dir)
    dense_embedder, sparse_embedder = build_textstack_embedders(
        cfg.textstack, cfg.paths.models_dir
    )
    channels = build_channels(
        cfg, encoder, dense_embedder, sparse_text_embedder=sparse_embedder
    )
    shot_meta = load_shot_meta(cfg)
    dispatcher = SpecDispatcher(
        channels=channels,
        channel_weights=cfg.retrieval.fusion.channel_weights,
        dispatch_cfg=cfg.cortex.dispatch,
        rrf_k=cfg.retrieval.fusion.rrf_k,
        top_k_per_channel=cfg.retrieval.fusion.top_k_per_channel,
        shot_meta=shot_meta,
        temporal_cfg=cfg.retrieval.temporal,
        video_prior=load_overlay_prior(cfg, sparse_embedder),
        chunking_cfg=cfg.retrieval.visual_chunking,
    )
    # A missing chronicle yields empty bundles (read_manifest tolerates
    # absent files): the service degrades to bare shot metadata.
    bundles = load_bundles(cfg)
    if not bundles:
        logger.warning("no chronicle found; serving without evidence bundles")
    # VQA Track A: the answer-ledger lookup, additive and off unless both
    # vqa.enabled and the ledger was built. Missing ledger.jsonl -> empty map.
    ledger: dict[str, object] = {}
    if cfg.vqa.enabled and cfg.chronicle.ledger.enabled:
        from aic.chronicle.ledger import load_ledger_map

        ledger = dict(load_ledger_map(cfg))
        if not ledger:
            logger.warning(
                "vqa.enabled but no ledger.jsonl; QA cards will lack provenance"
            )
    # VQA Track B: the per-candidate grounded reader, off unless its endpoint
    # is enabled. Lazy pool build, so no endpoint is contacted here.
    qa_reader = None
    if cfg.vqa.enabled and cfg.vqa.reader.enabled:
        from aic.vqa.reader import QaReader

        qa_reader = QaReader(
            cfg.vqa.reader,
            frames_dir=cfg.paths.keyframes_dir,
            labels=cfg.vqa.evidence_labels,
        )
    # VQA phase 4: the online count program for the 'count' archetype, built
    # only when a detector-counter backend is configured. build_counter is lazy
    # (no weights load here); the model loads on the first count question.
    count_program = None
    if cfg.vqa.enabled and cfg.chronicle.ledger.counter.enabled:
        from aic.vqa.count import CountProgram
        from aic.vqa.counter import build_counter

        counter = build_counter(
            cfg.chronicle.ledger.counter,
            cfg.paths.models_dir,
            cfg.paths.resolve_hf_token(),
        )
        count_reader = qa_reader.read_count if qa_reader is not None else None
        count_program = CountProgram(
            counter,
            count_reader,
            cfg.vqa,
            cfg.chronicle.ledger.counter,
            cfg.paths.keyframes_dir,
        )
    # Per-video shot index for the VQA evidence-pack window scans (built once;
    # scanning the whole corpus per candidate costs seconds per question on a
    # large corpus). None when VQA is off — no QA route reads it then.
    video_shot_index = None
    if cfg.vqa.enabled:
        from aic.vqa.evidence import build_video_shot_index

        video_shot_index = build_video_shot_index(shot_meta)
    keyframe_assets = _keyframe_search_assets(cfg, encoder)
    if keyframe_assets is None:
        logger.warning("no keyframe embeddings; feedback/similar disabled")
        feedback_engine = None
    else:
        index, keyframe_meta = keyframe_assets
        feedback_engine = DenseFeedbackEngine(
            encoder=encoder,
            index=index,
            keyframe_meta=keyframe_meta,
            cfg=cfg.service.feedback,
        )
    escalations, timeouts = _build_escalations(
        cfg, encoder, sparse_embedder, bundles, keyframe_assets
    )
    # The automatic rerank IS the vlm_verify escalation instance (the
    # config validator guarantees vlm_verify.enabled when auto_rerank is
    # on); reusing it keeps one endpoint pool and one prompt.
    auto_rerank = None
    if cfg.retrieval.auto_rerank.enabled:
        auto_rerank = escalations.get("vlm_verify")
        if auto_rerank is None:
            logger.warning(
                "auto_rerank enabled but the vlm_verify escalation did not "
                "build; searches run without the automatic rerank"
            )
    return ServiceState(
        compiler=CortexCompiler(cfg.cortex),
        engine=OnlineEngine(
            dispatcher=dispatcher,
            feedback_engine=feedback_engine,
        ),
        bundles=bundles,
        sessions=SessionStore(
            max_sessions=cfg.service.max_sessions,
            max_constraints=cfg.service.max_constraints,
        ),
        shot_meta=shot_meta,
        frames_dir=cfg.paths.keyframes_dir,
        negation_filter=cfg.cortex.dispatch.negation_filter,
        escalations=escalations,
        escalation_timeouts=timeouts,
        qpp=_load_qpp(cfg),
        auto_rerank=auto_rerank,
        auto_rerank_timeout_s=cfg.retrieval.auto_rerank.timeout_s,
        vqa_enabled=cfg.vqa.enabled,
        ledger=ledger,
        vqa_archetypes=cfg.vqa.archetypes,
        qa_reader=qa_reader,
        vqa_cfg=cfg.vqa,
        count_program=count_program,
        video_shot_index=video_shot_index,
    )
