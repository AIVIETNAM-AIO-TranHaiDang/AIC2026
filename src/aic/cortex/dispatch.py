"""Spec-to-channel dispatch: per-field sub-queries fused with weighted RRF.

Each spec field targets the channels that index the matching evidence
(visual phrases hit the embedding channels, OCR literals hit the sparse
channel verbatim, ...), and every sub-query joins one RRF race with weight
``channel_weight * field_weight``. Negations never become sub-queries; they
are applied as evidence-text filters after fusion (note 09 Phase 6:
embedding negation is unreliable).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field

from aic.config import DispatchConfig
from aic.cortex.compiler import CortexCompiler
from aic.cortex.spec import QuerySpec
from aic.eval.fixture import QueryCase
from aic.retrieval.base import SearchResult
from aic.retrieval.fusion import (
    DENSE_VISUAL,
    FACTOID_SPARSE,
    LITERAL_SPARSE,
    SEMANTIC_DENSE,
    Channel,
    ShotCandidate,
    ShotMeta,
    ground_candidates,
    query_text,
    rrf_fuse,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AdhocQuery:
    """Duck-types the fields channels read from an eval QueryCase.

    Online queries have no ground truth, which QueryCase requires, so the
    service dispatches these instead; channels only touch query_id, kind,
    text, clip_path, and reveals.
    """

    query_id: str
    text: str
    kind: str = "kis_t"
    clip_path: str | None = None
    reveals: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SubQuery:
    """One planned channel probe: ``key`` is unique within the RRF race."""

    key: str
    spec_field: str
    text: str
    channel: str
    weight: float


@dataclass(frozen=True)
class DispatchDetail:
    """A dispatch run with its per-sub-query rankings kept.

    The fused list is what retrieval uses; the rankings feed the QPP
    advisor's channel-agreement feature (Phase 7), which needs to see how
    much the channels corroborate each other before fusion flattened them.
    """

    candidates: list[ShotCandidate]
    rankings: dict[str, list[ShotCandidate]]
    weights: dict[str, float]


# Which channels index the evidence each spec field describes. The factoid
# channel (the Answer Ledger index) is listed for the text fields but only
# activates when a weight for it is supplied — KIS profiles never weight it,
# and the QA path passes it as an extra weight (vqa.factoid_weight) — so KIS
# dispatch plans are byte-identical with or without a built factoid index.
_FIELD_CHANNELS = {
    "visual_phrases": (DENSE_VISUAL, SEMANTIC_DENSE),
    "paraphrases": (DENSE_VISUAL, SEMANTIC_DENSE),
    "entity_terms": (SEMANTIC_DENSE, LITERAL_SPARSE, FACTOID_SPARSE),
    "ocr_literals": (LITERAL_SPARSE, FACTOID_SPARSE),
    "asr_phrases": (LITERAL_SPARSE, SEMANTIC_DENSE, FACTOID_SPARSE),
}


def build_plan(
    spec: QuerySpec,
    dispatch_cfg: DispatchConfig,
    channel_weights: Mapping[str, float],
) -> list[SubQuery]:
    """The full dispatch plan for a spec, deterministic order."""
    field_weights = {
        "visual_phrases": dispatch_cfg.visual_weight,
        "paraphrases": dispatch_cfg.visual_weight * dispatch_cfg.paraphrase_weight,
        "entity_terms": dispatch_cfg.entity_weight,
        "ocr_literals": dispatch_cfg.ocr_literal_weight,
        "asr_phrases": dispatch_cfg.asr_weight,
    }
    plan = []
    for spec_field, channels in _FIELD_CHANNELS.items():
        for i, text in enumerate(getattr(spec, spec_field)):
            if not text.strip():
                continue
            for channel in channels:
                channel_weight = channel_weights.get(channel)
                if channel_weight is None:
                    continue  # channel disabled by config: the ablation switch
                plan.append(
                    SubQuery(
                        key=f"{spec_field}[{i}].{channel}",
                        spec_field=spec_field,
                        text=text.strip(),
                        channel=channel,
                        weight=channel_weight * field_weights[spec_field],
                    )
                )
    return plan


# Channels the raw (untranslated) query is dispatched to when
# dispatch.raw_query_weight > 0: the multilingual text channels only. The
# dense visual tower is English-trained, so the raw query never goes there.
_RAW_QUERY_CHANNELS = (SEMANTIC_DENSE, LITERAL_SPARSE)


def apply_visual_chunking(
    plan: list[SubQuery], chunking_cfg
) -> tuple[list[SubQuery], dict[str, int] | None]:
    """Expand multi-sentence dense-visual sub-queries into sentence chunks.

    Returns the (possibly rewritten) plan and, when exactly one sub-query
    was chunked, the label->sentence-position map for the temporal order
    bonus. When several sub-queries chunk (two long paraphrases at once),
    their sentence orders are unrelated, so no order map is produced —
    the chunks still dispatch, only the bonus stays silent. Only the dense
    visual channel is touched: SigLIP is the encoder that cannot read past
    the first sentence (note 15); the text channels read the full query.
    """
    if chunking_cfg is None or not chunking_cfg.enabled:
        return plan, None
    from aic.retrieval.chunking import split_query_sentences

    rewritten: list[SubQuery] = []
    order_maps: list[dict[str, int]] = []
    for sub in plan:
        if sub.channel != DENSE_VISUAL:
            rewritten.append(sub)
            continue
        chunks = split_query_sentences(sub.text, chunking_cfg.max_chunks)
        if len(chunks) <= 1:
            rewritten.append(sub)
            continue
        prefix = sub.key.rsplit(".", 1)[0]
        order: dict[str, int] = {}
        for j, chunk in enumerate(chunks):
            key = f"{prefix}.chunk{j}.{DENSE_VISUAL}"
            order[key] = j
            rewritten.append(
                SubQuery(
                    key=key,
                    spec_field=sub.spec_field,
                    text=chunk,
                    channel=DENSE_VISUAL,
                    weight=sub.weight * chunking_cfg.chunk_weight,
                )
            )
        order_maps.append(order)
    return rewritten, order_maps[0] if len(order_maps) == 1 else None


class SpecDispatcher:
    """Executes a dispatch plan against the Phase 5 channels.

    ``shot_meta`` + ``temporal_cfg`` switch the final fusion to temporal
    window scoring (note 15); ``video_prior`` adds the overlay video prior.
    All three default to None, which keeps the plain per-shot RRF.
    """

    def __init__(
        self,
        channels: list[Channel],
        channel_weights: Mapping[str, float],
        dispatch_cfg: DispatchConfig,
        rrf_k: int,
        top_k_per_channel: int,
        shot_meta: Mapping[str, ShotMeta] | None = None,
        temporal_cfg=None,
        video_prior=None,
        chunking_cfg=None,
    ) -> None:
        self._channels = {channel.name: channel for channel in channels}
        self._channel_weights = dict(channel_weights)
        self._dispatch_cfg = dispatch_cfg
        self._rrf_k = rrf_k
        self._top_k_per_channel = top_k_per_channel
        self._shot_meta = shot_meta
        self._temporal_cfg = temporal_cfg
        self._video_prior = video_prior
        self._chunking_cfg = chunking_cfg

    def rank_spec(
        self,
        spec: QuerySpec,
        raw_query: str = "",
        extra_channel_weights: Mapping[str, float] | None = None,
    ) -> list[ShotCandidate]:
        return self.rank_spec_detailed(
            spec, raw_query, extra_channel_weights=extra_channel_weights
        ).candidates

    def _raw_query_subs(self, raw_query: str) -> list[SubQuery]:
        weight = self._dispatch_cfg.raw_query_weight
        if weight <= 0 or not raw_query.strip():
            return []
        return [
            SubQuery(
                key=f"raw_query.{channel}",
                spec_field="raw_query",
                text=raw_query.strip(),
                channel=channel,
                weight=self._channel_weights[channel] * weight,
            )
            for channel in _RAW_QUERY_CHANNELS
            if channel in self._channel_weights
        ]

    def _fuse(
        self,
        rankings: dict[str, list[ShotCandidate]],
        weights: dict[str, float],
        query: str,
        label_order: dict[str, int] | None = None,
    ) -> list[ShotCandidate]:
        if (
            self._temporal_cfg is None
            or self._temporal_cfg.window_ms == 0
            or self._shot_meta is None
        ):
            return rrf_fuse(rankings, weights, self._rrf_k)
        from aic.retrieval.temporal import fuse_windows

        prior = None
        if (
            self._video_prior is not None
            and self._temporal_cfg.video_prior_weight > 0
        ):
            prior = self._video_prior.contributions(
                query,
                self._temporal_cfg.video_prior_weight,
                self._rrf_k,
                self._top_k_per_channel,
            )
        return fuse_windows(
            rankings,
            weights,
            self._rrf_k,
            dict(self._shot_meta),
            self._temporal_cfg,
            video_prior=prior,
            label_order=label_order,
        )

    def rank_spec_detailed(
        self,
        spec: QuerySpec,
        raw_query: str = "",
        extra_channel_weights: Mapping[str, float] | None = None,
    ) -> DispatchDetail:
        # Group the plan's sub-queries per channel so each channel embeds
        # all its texts in ONE encoder call (rank_texts) — a spec fans out
        # to a dozen sub-queries, and per-sub-query encoding paid the model
        # dispatch overhead every time. Channels without rank_texts (test
        # fakes) keep the per-sub-query path.
        #
        # extra_channel_weights (the QA path's factoid probe) extend the
        # profile weights for THIS call only; sub-queries whose channel is
        # not actually loaded are dropped below, so passing a weight for a
        # never-built index is harmless.
        weights_in_effect = dict(self._channel_weights)
        if extra_channel_weights:
            weights_in_effect.update(extra_channel_weights)
        plan = build_plan(spec, self._dispatch_cfg, weights_in_effect)
        plan.extend(self._raw_query_subs(raw_query))
        plan, label_order = apply_visual_chunking(plan, self._chunking_cfg)
        by_channel: dict[str, list[SubQuery]] = {}
        for sub in plan:
            if sub.channel in self._channels:
                by_channel.setdefault(sub.channel, []).append(sub)

        rankings: dict[str, list[ShotCandidate]] = {}
        weights: dict[str, float] = {}
        for channel_name, subs in by_channel.items():
            channel = self._channels[channel_name]
            rank_texts = getattr(channel, "rank_texts", None)
            if rank_texts is not None:
                results = rank_texts(
                    [sub.text for sub in subs], self._top_k_per_channel
                )
            else:
                results = []
                for sub in subs:
                    case = AdhocQuery(query_id=sub.key, text=sub.text)
                    if not channel.supports(case):
                        results.append([])
                        continue
                    results.append(channel.rank(case, self._top_k_per_channel))
            for sub, candidates in zip(subs, results, strict=True):
                if candidates:
                    rankings[sub.key] = candidates
                    weights[sub.key] = sub.weight
        if not rankings:
            return DispatchDetail(candidates=[], rankings={}, weights={})
        query_for_prior = raw_query.strip() or "\n".join(
            spec.ocr_literals + spec.entity_terms + spec.visual_phrases
        )
        return DispatchDetail(
            candidates=self._fuse(rankings, weights, query_for_prior, label_order),
            rankings=rankings,
            weights=weights,
        )


def filter_negations(
    candidates: list[ShotCandidate],
    negations: list[str],
    evidence_text: Mapping[str, str],
) -> list[ShotCandidate]:
    """Drop candidates whose evidence contains a negated phrase.

    Case-folded substring match over the shot's combined caption/OCR/ASR
    text. Shots with no evidence text are kept: absence of evidence is not
    evidence of the negated thing.
    """
    if not negations:
        return candidates
    needles = [n.casefold() for n in negations if n.strip()]
    if not needles:
        return candidates
    kept = []
    for candidate in candidates:
        haystack = evidence_text.get(candidate.shot_key, "").casefold()
        if haystack and any(needle in haystack for needle in needles):
            continue
        kept.append(candidate)
    return kept


class CortexRetriever:
    """Compile -> dispatch -> filter -> ground. The Phase 6 fast path.

    Implements the harness Retriever protocol so ablation #2 (spec dispatch
    vs. the single-string Phase 5 baseline) runs on the same fixture
    metrics. KIS-V cases carry a clip instead of text and bypass the Cortex
    into the plain fusion engine.
    """

    def __init__(
        self,
        compiler: CortexCompiler,
        dispatcher: SpecDispatcher,
        shot_meta: dict[str, ShotMeta],
        clip_retriever=None,
        evidence_text: Mapping[str, str] | None = None,
        negation_filter: bool = True,
        auto_rerank=None,
        auto_rerank_timeout_s: float = 20.0,
    ) -> None:
        """``clip_retriever`` is any Retriever handling kis_v cases
        (normally the Phase 5 FusionRetriever); ``evidence_text`` maps
        shot_key to combined caption/OCR/ASR text for negation filtering.
        ``auto_rerank`` is an Escalation (normally VlmVerify) run
        automatically on the fused candidates under
        ``auto_rerank_timeout_s``; any failure or timeout keeps the fused
        order (degrade-to-unchanged, note 15)."""
        self._compiler = compiler
        self._dispatcher = dispatcher
        self._shot_meta = shot_meta
        self._clip_retriever = clip_retriever
        self._evidence_text = evidence_text or {}
        self._negation_filter = negation_filter
        self._auto_rerank = auto_rerank
        self._auto_rerank_timeout_s = auto_rerank_timeout_s

    def search(self, case: QueryCase, top_k: int) -> list[SearchResult]:
        if case.kind == "kis_v":
            if self._clip_retriever is None:
                logger.warning("no clip retriever configured; kis_v unsupported")
                return []
            return self._clip_retriever.search(case, top_k)
        raw_query = query_text(case)
        spec = self._compiler.compile(raw_query)
        return self.search_spec(spec, top_k, raw_query=raw_query)

    def _apply_auto_rerank(
        self, spec: QuerySpec, raw_query: str, candidates: list[ShotCandidate]
    ) -> list[ShotCandidate]:
        if self._auto_rerank is None or not candidates:
            return candidates
        from aic.escalate.base import EscalationRequest, run_with_timeout

        request = EscalationRequest(
            query=raw_query or "\n".join(spec.visual_phrases),
            spec=spec,
            candidates=candidates,
        )
        try:
            outcome = run_with_timeout(
                self._auto_rerank, request, self._auto_rerank_timeout_s
            )
        except Exception as exc:  # noqa: BLE001 - the automatic stage must
            # degrade to the fused order on ANY failure, not only the
            # declared escalation errors: a search must never be broken by
            # its own optional improver.
            logger.warning("auto rerank skipped: %s", exc)
            return candidates
        if outcome.note:
            logger.info("auto rerank note: %s", outcome.note)
        return outcome.candidates

    def search_spec(
        self, spec: QuerySpec, top_k: int, raw_query: str = ""
    ) -> list[SearchResult]:
        candidates = self._dispatcher.rank_spec(spec, raw_query)
        if self._negation_filter:
            candidates = filter_negations(
                candidates, spec.negations, self._evidence_text
            )
        candidates = self._apply_auto_rerank(spec, raw_query, candidates)
        return ground_candidates(candidates, self._shot_meta, top_k)
