"""Late-interaction re-rank: BGE-M3 ColBERT MaxSim over evidence text.

The T0 realisation of note 08's step 6b: the query's per-token vectors are
matched against each candidate's Chronicle evidence text (caption + OCR +
ASR), scored with MaxSim, and the pool is re-ordered. Multi-vectors are
computed on demand for the pool only — no precomputed patch store — which
keeps the tool free of extra artifacts and models (BGE-M3 is already
resident for the sparse channel).

The MaxSim formula mirrors FlagEmbedding's own ``colbert_score`` (verified
against 1.4.0): mean over query tokens of the max inner product over
document tokens.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np

from aic.config import RerankEscalationConfig
from aic.escalate.base import EscalationOutcome, EscalationRequest

logger = logging.getLogger(__name__)


def maxsim(query_vecs: np.ndarray, doc_vecs: np.ndarray) -> float:
    """ColBERT MaxSim between one query and one document."""
    if not len(query_vecs) or not len(doc_vecs):
        return 0.0
    return float((query_vecs @ doc_vecs.T).max(axis=1).mean())


class LateInteractionReranker:
    """Re-orders the top pool by MaxSim against evidence text."""

    def __init__(
        self,
        embedder,
        evidence_text: Mapping[str, str],
        cfg: RerankEscalationConfig,
    ) -> None:
        """``embedder`` must expose ``encode_colbert`` (BGE-M3); the
        builder checks this before wiring the tool."""
        self._embedder = embedder
        self._evidence_text = evidence_text
        self._cfg = cfg

    @property
    def name(self) -> str:
        return "rerank"

    def run(self, request: EscalationRequest) -> EscalationOutcome:
        pool = request.candidates[: self._cfg.pool_size]
        tail = request.candidates[self._cfg.pool_size :]
        if not pool:
            return EscalationOutcome(request.candidates, note="nothing to re-rank")
        scored = [
            (candidate, self._evidence_text.get(candidate.shot_key, "").strip())
            for candidate in pool
        ]
        with_text = [(c, text) for c, text in scored if text]
        without_text = [c for c, text in scored if not text]
        if not with_text:
            return EscalationOutcome(
                request.candidates,
                note="no candidate has evidence text to re-rank against",
            )
        query_vecs = self._embedder.encode_colbert([request.query])[0]
        doc_vecs = self._embedder.encode_colbert([text for _c, text in with_text])
        ranked = sorted(
            zip(with_text, doc_vecs, strict=True),
            key=lambda item: (-maxsim(query_vecs, item[1]), item[0][0].shot_key),
        )
        # Candidates without evidence cannot be scored; they keep their
        # relative order below the scored ones (absence of evidence is not
        # evidence of a mismatch).
        reordered = [c for (c, _text), _vecs in ranked] + without_text + tail
        return EscalationOutcome(reordered)
