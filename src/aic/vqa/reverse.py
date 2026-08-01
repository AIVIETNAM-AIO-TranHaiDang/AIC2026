"""Answer-first reverse retrieval + the EV submit hint (note 18 phase 5).

Reverse retrieval inverts the direction for a weak-locator question: instead of
finding the moment then reading it, it hypothesises plausible answers and
searches their surface forms in the literal/factoid indexes. A hypothesis wins
only through a DOUBLE gate — it must retrieve a concentrated moment AND that
moment must independently match the locator clause — which is what stops the
self-fulfilling loop where "searching X finds text containing X and confirms it"
(risk 5.2). The EV hint is a transparent rule table advising submit / wait /
escalate from the current answer state (advisory only, the QPP contract).

Everything here is pure or composes injected callables (an LLM hypothesiser and
an index searcher), so it is fully CPU-testable with fakes.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from statistics import median

from aic.config import EvHintConfig, VqaReverseConfig

logger = logging.getLogger(__name__)

# A search over the literal/factoid indexes: a surface form -> (shot_key, score)
# hits, best first. Production wraps the spec dispatcher; tests inject a fake.
Searcher = Callable[[str], list[tuple[str, float]]]
# The LLM (or ledger-seed) hypothesiser: question -> candidate answer strings.
Hypothesiser = Callable[[str], list[str]]
# How well a moment matches the locator clause (the second gate).
LocatorScorer = Callable[[str], float]


@dataclass(frozen=True)
class ReverseCard:
    """A reverse-retrieval answer: found by searching the answer, not the moment."""

    answer: str
    shot_key: str
    score: float

    def provenance(self) -> dict:
        return {"kind": "reverse", "note": "found by searching the answer"}


def _peakiness(scores: list[float]) -> float:
    """max / median of a score list; 1.0 for a flat/degenerate distribution."""
    if not scores:
        return 1.0
    med = median(scores)
    if med <= 0:
        return 1.0
    return max(scores) / med


def is_weak_locator(
    top_scores: list[float], has_track_a_answer: bool, cfg: VqaReverseConfig
) -> bool:
    """Whether reverse retrieval should fire for this question (note 18 §3.3).

    Weak when Track A produced no answer-bearing card, or the top-M fused scores
    are flat (peakiness below ``trigger_ratio`` — no clear winning moment).
    """
    if not has_track_a_answer:
        return True
    if not top_scores:
        return True
    return _peakiness(top_scores) < cfg.trigger_ratio


def passes_double_gate(
    hits: list[tuple[str, float]], locator_score: float, cfg: VqaReverseConfig
) -> bool:
    """Both gates for a hypothesis (risk 5.2): concentrated AND locator-consistent.

    Gate A: the hypothesis's search returns a concentrated moment — its top
    hit's peakiness over the TOP ``gate_top_m`` hits is at least
    ``trigger_ratio``, or it is the only hit. The slice matters: over a full
    RRF tail the max/median ratio is always high (hundreds of 1/(k+rank)
    scores shrink the median), which would wave every hypothesis through.
    Gate B: that moment also scores at least ``min_locator_score`` against
    the locator clause — retrieval success on the answer string is NOT on its
    own evidence the answer fits the QUESTION.
    """
    if not hits:
        return False
    scores = [s for _k, s in hits[: cfg.gate_top_m]]
    concentrated = len(hits) == 1 or _peakiness(scores) >= cfg.trigger_ratio
    return concentrated and locator_score >= cfg.min_locator_score


class ReverseProgram:
    """Hypothesise answers, search them, keep only double-gated winners."""

    def __init__(
        self,
        hypothesiser: Hypothesiser,
        searcher: Searcher,
        locator_scorer: LocatorScorer,
        cfg: VqaReverseConfig,
    ) -> None:
        self._hypothesiser = hypothesiser
        self._searcher = searcher
        self._locator_scorer = locator_scorer
        self._cfg = cfg

    def run(
        self, question: str, seed_answers: list[str] | None = None
    ) -> list[ReverseCard]:
        """Reverse cards for a question, best first (empty when none survive).

        ``seed_answers`` (e.g. ledger entities of locator-adjacent shots) are
        used instead of the LLM when provided — cheaper and grounded (risk 5.1).
        Never raises: a hypothesiser or searcher failure yields no cards.
        """
        try:
            hypotheses = seed_answers or self._hypothesiser(question)
        except Exception as exc:  # noqa: BLE001 - reverse is optional
            logger.warning("reverse hypothesiser failed: %s", exc)
            return []
        cards: list[ReverseCard] = []
        for hypothesis in list(hypotheses)[: self._cfg.max_hypotheses]:
            if not hypothesis.strip():
                continue
            try:
                hits = self._searcher(hypothesis)
            except Exception as exc:  # noqa: BLE001 - skip a bad hypothesis
                logger.warning("reverse search failed for %r: %s", hypothesis, exc)
                continue
            if not hits:
                continue
            top_key, top_score = hits[0]
            locator_score = self._locator_scorer(top_key)
            if passes_double_gate(hits, locator_score, self._cfg):
                cards.append(
                    ReverseCard(
                        answer=hypothesis.strip(),
                        shot_key=top_key,
                        score=top_score * self._cfg.reverse_weight,
                    )
                )
        cards.sort(key=lambda c: -c.score)
        return cards


# EV submit hint (note 18 §3.6): submit_now | wait_for_reads | escalate_zoom.
EV_HINTS = ("submit_now", "wait_for_reads", "escalate_zoom")


def ev_hint(features: dict, cfg: EvHintConfig) -> str:
    """Advise submit / wait / escalate from the current answer state.

    Transparent rule table (advisory only, never fires anything):
    - a confident winner (cross-track agreement, or enough agreeing moments)
      that was not disconfirmed -> submit_now (until the per-task budget is spent);
    - otherwise, reads still pending -> wait_for_reads;
    - otherwise -> escalate_zoom.
    ``features``: agreeing_count, cross_track_agree, disconfirmed,
    track_b_pending, submits_used.
    """
    if features.get("submits_used", 0) >= cfg.max_early_submits:
        return "wait_for_reads"
    disconfirmed = bool(features.get("disconfirmed", False))
    cross = bool(features.get("cross_track_agree", False))
    agree = int(features.get("agreeing_count", 0))
    if not disconfirmed and (cross or agree >= cfg.min_agreeing_for_submit):
        return "submit_now"
    if int(features.get("track_b_pending", 0)) > 0:
        return "wait_for_reads"
    return "escalate_zoom"
