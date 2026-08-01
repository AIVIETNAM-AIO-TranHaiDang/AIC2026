"""The engine surface the service talks to, and its production assembly.

The FastAPI app depends on this small protocol so tests drive the endpoints
with in-memory fakes while production wires the real Phase 5/6 machinery
(spec dispatcher over the fusion channels, Rocchio feedback over the
keyframe index).
"""

from __future__ import annotations

import logging
from typing import Protocol

from aic.cortex.spec import QuerySpec
from aic.retrieval.fusion import ShotCandidate

logger = logging.getLogger(__name__)


class SearchEngine(Protocol):
    def rank_spec(
        self, spec: QuerySpec, raw_query: str = ""
    ) -> list[ShotCandidate]: ...

    def rank_feedback(
        self,
        query_text: str,
        positive_shots: list[str],
        negative_shots: list[str],
        top_k: int,
    ) -> list[ShotCandidate]: ...

    def rank_similar(self, shot_key: str, top_k: int) -> list[ShotCandidate]: ...


class OnlineEngine:
    """Production engine: dispatcher for search, dense engine for feedback.

    The feedback engine is optional — without keyframe embeddings the
    service still serves text search (degraded but working).
    ``raw_query`` carries the operator's untranslated query alongside the
    compiled spec (dispatch.raw_query_weight and the overlay video prior
    read it; empty string keeps the spec-only behaviour)."""

    def __init__(self, dispatcher, feedback_engine=None) -> None:
        self._dispatcher = dispatcher
        self._feedback = feedback_engine

    def rank_spec(
        self, spec: QuerySpec, raw_query: str = ""
    ) -> list[ShotCandidate]:
        return self._dispatcher.rank_spec(spec, raw_query)

    def rank_spec_detailed(self, spec: QuerySpec, raw_query: str = ""):
        """Fused candidates plus per-sub-query rankings (QPP features).

        Optional on the protocol: the app falls back to :meth:`rank_spec`
        for engines (test fakes) that do not expose it.
        """
        return self._dispatcher.rank_spec_detailed(spec, raw_query)

    def rank_qa_spec(
        self, spec: QuerySpec, raw_query: str = "", factoid_weight: float = 0.0
    ) -> list[ShotCandidate]:
        """QA-path ranking: the KIS dispatch plus the factoid channel.

        With ``factoid_weight`` > 0 the spec's text clauses also probe the
        Answer Ledger factoid index (note 18 §3.2's lookup); 0 (or a
        never-built index) degrades to exactly :meth:`rank_spec`. Optional on
        the protocol — the app falls back to :meth:`rank_spec` for engines
        (test fakes) that do not expose it.
        """
        if factoid_weight <= 0:
            return self.rank_spec(spec, raw_query)
        from aic.retrieval.fusion import FACTOID_SPARSE

        return self._dispatcher.rank_spec(
            spec,
            raw_query,
            extra_channel_weights={FACTOID_SPARSE: factoid_weight},
        )

    def rank_feedback(
        self,
        query_text: str,
        positive_shots: list[str],
        negative_shots: list[str],
        top_k: int,
    ) -> list[ShotCandidate]:
        if self._feedback is None:
            logger.warning("no dense feedback engine; returning empty feedback")
            return []
        return self._feedback.rank_feedback(
            query_text, positive_shots, negative_shots, top_k
        )

    def rank_similar(self, shot_key: str, top_k: int) -> list[ShotCandidate]:
        if self._feedback is None:
            logger.warning("no dense feedback engine; similar unsupported")
            return []
        return self._feedback.rank_similar(shot_key, top_k)
