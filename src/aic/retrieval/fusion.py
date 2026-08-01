"""Channel race and weighted reciprocal-rank fusion (the online fast path).

Channels rank shots independently (dense visual over keyframes, semantic
dense over captions+speech, literal sparse over OCR+speech); fusion combines
their rankings with weighted RRF. RRF is rank-based on purpose: raw scores
from different embedding spaces are not comparable, ranks always are
(note 08, step 4; note 09 Phase 5 edge cases).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from aic.eval.fixture import QueryCase
from aic.index.sparse import SparseIndex
from aic.index.vector import VectorIndex
from aic.retrieval.base import SearchResult
from aic.textstack.embedder import TextEmbedder

logger = logging.getLogger(__name__)

DENSE_VISUAL = "dense_visual"
SEMANTIC_DENSE = "semantic_dense"
LITERAL_SPARSE = "literal_sparse"
FACTOID_SPARSE = "factoid_sparse"


@dataclass(frozen=True)
class ShotCandidate:
    shot_key: str
    score: float
    timestamp_ms: int | None = None
    """Best evidence timestamp, when the channel knows one (dense visual
    knows its best keyframe; text channels do not)."""


class Channel(Protocol):
    @property
    def name(self) -> str: ...

    def supports(self, case: QueryCase) -> bool: ...

    def rank(self, case: QueryCase, top_k: int) -> list[ShotCandidate]: ...


def query_text(case: QueryCase) -> str:
    """The text a channel should embed for this case.

    KIS-C accumulates every revealed detail into the query; proper session
    narrowing arrives with the Phase 6 service, this keeps evaluation honest
    until then.
    """
    parts = [case.text or ""]
    parts.extend(case.reveals)
    return "\n".join(p for p in parts if p).strip()


class SemanticDenseChannel:
    """Query text vs. chronicle semantic documents (dense text embedder)."""

    def __init__(self, embedder: TextEmbedder, index: VectorIndex) -> None:
        self._embedder = embedder
        self._index = index

    @property
    def name(self) -> str:
        return SEMANTIC_DENSE

    def supports(self, case: QueryCase) -> bool:
        return bool(query_text(case))

    def rank(self, case: QueryCase, top_k: int) -> list[ShotCandidate]:
        return self.rank_texts([query_text(case)], top_k)[0]

    def rank_texts(self, texts: list[str], top_k: int) -> list[list[ShotCandidate]]:
        """Rank several text queries with one embedder call (batch dispatch)."""
        embeddings = self._embedder.encode_queries(texts)
        hits_per_query = self._index.search(
            embeddings.dense, top_k, query_model_id=self._embedder.model_id
        )
        return [
            [ShotCandidate(shot_key=key, score=score) for key, score in hits]
            for hits in hits_per_query
        ]


class LiteralSparseChannel:
    """Query text vs. a sparse chronicle index (BGE-M3 lexical weights).

    Serves both the literal (OCR+speech) index and, under a different
    ``name`` (``factoid_sparse``), the Answer Ledger factoid index — the
    retrieval mechanism is identical, only the index and channel weight
    differ.
    """

    def __init__(
        self, embedder: TextEmbedder, index: SparseIndex, name: str = LITERAL_SPARSE
    ) -> None:
        self._embedder = embedder
        self._index = index
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def supports(self, case: QueryCase) -> bool:
        return bool(query_text(case))

    def rank(self, case: QueryCase, top_k: int) -> list[ShotCandidate]:
        return self.rank_texts([query_text(case)], top_k)[0]

    def rank_texts(self, texts: list[str], top_k: int) -> list[list[ShotCandidate]]:
        """Rank several text queries with one embedder call (batch dispatch)."""
        embeddings = self._embedder.encode_queries(texts)
        return [
            [
                ShotCandidate(shot_key=key, score=score)
                for key, score in self._index.search(sparse, top_k)
            ]
            for sparse in embeddings.sparse
        ]


def rrf_fuse(
    rankings: dict[str, list[ShotCandidate]],
    weights: dict[str, float],
    rrf_k: int,
) -> list[ShotCandidate]:
    """Weighted reciprocal-rank fusion across channel rankings.

    Deterministic: ties break on shot_key. A shot's timestamp comes from the
    highest-weighted channel that provided one.
    """
    fused: dict[str, float] = {}
    timestamps: dict[str, tuple[float, int]] = {}
    for channel_name, candidates in rankings.items():
        weight = weights.get(channel_name)
        if weight is None:
            continue
        for rank, candidate in enumerate(candidates, start=1):
            fused[candidate.shot_key] = fused.get(candidate.shot_key, 0.0) + (
                weight / (rrf_k + rank)
            )
            if candidate.timestamp_ms is not None:
                current = timestamps.get(candidate.shot_key)
                if current is None or weight > current[0]:
                    timestamps[candidate.shot_key] = (
                        weight,
                        candidate.timestamp_ms,
                    )
    ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
    return [
        ShotCandidate(
            shot_key=key,
            score=score,
            timestamp_ms=timestamps.get(key, (0.0, None))[1],
        )
        for key, score in ordered
    ]


@dataclass(frozen=True)
class ShotMeta:
    video_id: str
    shot_id: int
    t_start_ms: int
    t_end_ms: int

    @property
    def midpoint_ms(self) -> int:
        return (self.t_start_ms + self.t_end_ms) // 2


def ground_candidates(
    candidates: list[ShotCandidate],
    shot_meta: dict[str, ShotMeta],
    top_k: int,
) -> list[SearchResult]:
    """Fused shot candidates -> submittable ``(video_id, timestamp)`` results.

    The timestamp is the candidate's best evidence timestamp when a channel
    provided one (dense visual knows its best keyframe), else the shot
    midpoint. Candidates without metadata are dropped with a warning.
    """
    results = []
    for candidate in candidates[:top_k]:
        meta = shot_meta.get(candidate.shot_key)
        if meta is None:
            logger.warning(
                "fused candidate %s has no shot metadata; dropped",
                candidate.shot_key,
            )
            continue
        timestamp = (
            candidate.timestamp_ms
            if candidate.timestamp_ms is not None
            else meta.midpoint_ms
        )
        results.append(
            SearchResult(
                video_id=meta.video_id,
                timestamp_ms=timestamp,
                score=candidate.score,
                keyframe_id=None,
            )
        )
    return results


class FusionRetriever:
    """The Phase 5 engine: channel race -> RRF -> temporal grounding.

    Implements the harness Retriever protocol, so ablations (toggling
    channels via config weights) are measured with the same fixture metrics
    as everything else. Grounding is score-profile: the winning shot's
    timestamp is its best dense-visual keyframe when known, else the shot
    midpoint. ``temporal_cfg`` (window fusion, note 15) and ``video_prior``
    (the overlay index) are optional; both default to the plain per-shot
    behaviour.
    """

    def __init__(
        self,
        channels: list[Channel],
        weights: dict[str, float],
        rrf_k: int,
        top_k_per_channel: int,
        shot_meta: dict[str, ShotMeta],
        temporal_cfg=None,
        video_prior=None,
        chunking_cfg=None,
    ) -> None:
        known = {channel.name for channel in channels}
        unknown = set(weights) - known
        if unknown:
            raise ValueError(
                f"channel weights reference unknown channels: {sorted(unknown)}"
            )
        self._channels = channels
        self._weights = weights
        self._rrf_k = rrf_k
        self._top_k_per_channel = top_k_per_channel
        self._shot_meta = shot_meta
        self._temporal_cfg = temporal_cfg
        self._video_prior = video_prior
        self._chunking_cfg = chunking_cfg

    def _fuse(
        self,
        rankings: dict[str, list[ShotCandidate]],
        weights: dict[str, float],
        query: str,
        label_order: dict[str, int] | None = None,
    ) -> list[ShotCandidate]:
        if self._temporal_cfg is None or self._temporal_cfg.window_ms == 0:
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
            self._shot_meta,
            self._temporal_cfg,
            video_prior=prior,
            label_order=label_order,
        )

    def _dense_chunks(self, channel: Channel, case: QueryCase) -> list[str] | None:
        """The query's sentence chunks, when this channel should chunk.

        Only the dense visual channel on a text query, only when
        retrieval.visual_chunking is enabled, only when the query actually
        has several sentences, and only when the channel can batch-rank
        texts (KIS-V clips have no text to split).
        """
        cfg = self._chunking_cfg
        if cfg is None or not cfg.enabled:
            return None
        if channel.name != DENSE_VISUAL or case.kind == "kis_v":
            return None
        if getattr(channel, "rank_texts", None) is None:
            return None
        from aic.retrieval.chunking import split_query_sentences

        chunks = split_query_sentences(query_text(case), cfg.max_chunks)
        return chunks if len(chunks) > 1 else None

    def search(self, case: QueryCase, top_k: int) -> list[SearchResult]:
        rankings: dict[str, list[ShotCandidate]] = {}
        weights = dict(self._weights)
        label_order: dict[str, int] | None = None
        for channel in self._channels:
            if channel.name not in self._weights:
                continue
            if not channel.supports(case):
                continue
            chunks = self._dense_chunks(channel, case)
            if chunks is not None:
                results = channel.rank_texts(chunks, self._top_k_per_channel)
                chunk_weight = (
                    self._weights[channel.name] * self._chunking_cfg.chunk_weight
                )
                order: dict[str, int] = {}
                for j, candidates in enumerate(results):
                    if not candidates:
                        continue
                    label = f"query_chunk[{j}].{channel.name}"
                    rankings[label] = candidates
                    weights[label] = chunk_weight
                    order[label] = j
                if len(order) > 1:
                    label_order = order
                continue
            candidates = channel.rank(case, self._top_k_per_channel)
            if candidates:
                rankings[channel.name] = candidates
        if not rankings:
            logger.warning("no channel produced candidates for %s", case.query_id)
            return []
        fused = self._fuse(rankings, weights, query_text(case), label_order)
        return ground_candidates(fused, self._shot_meta, top_k)
