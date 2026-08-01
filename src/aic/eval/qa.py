"""The QA branch of the harness: answer + moment scoring for `kind: qa` cases.

A KIS case is scored by rank alone; a QA case needs two judgments — did the
system read the right answer, and did it point at the right moment — plus their
conjunction, the DRES-judged `qa_correct`. This module defines the answerer
contract the harness routes QA cases through, the normalised answer match, and
the per-case metric builder. It is pure and fake-testable; the production
answerer that composes the Cortex + ledger + reader lives in
`scripts/evaluate_fixture.py`, mirroring how that script re-wires the online
KIS path for evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from aic.config import VqaNormalizeConfig
from aic.eval.fixture import QueryCase
from aic.eval.metrics import QaCaseMetric, is_hit
from aic.retrieval.base import SearchResult
from aic.vqa.normalize import normalize_answer


@dataclass(frozen=True)
class QaOutcome:
    """What a QA answerer returns for one case.

    ``results`` are the located moments (best first) scored for ``moment_ok``
    exactly like a KIS result list; ``answer`` is the system's single best
    answer (None when it declined); ``archetype``/``answer_type`` come from the
    compiled plan and drive normalisation and the per-archetype breakdown.
    """

    results: list[SearchResult] = field(default_factory=list)
    answer: str | None = None
    archetype: str = "other"
    answer_type: str = "text"


class QaAnswerer(Protocol):
    def answer(self, case: QueryCase, top_k: int) -> QaOutcome:
        """Answer one QA case and locate its moment, best result first."""
        ...


def answer_matches(
    answer: str | None,
    accepted_answers: list[str],
    answer_type: str,
    normalize_cfg: VqaNormalizeConfig,
) -> bool:
    """True when ``answer`` normalises equal to any accepted answer.

    Both sides go through the same configured normalisation (number words ↔
    digits, colour synonyms, unit stripping — all language-general via
    VqaNormalizeConfig), so "ba" matches "3" and "xanh dương" matches "xanh
    lam" under the default Vietnamese lexicon. An empty answer never matches.
    """
    if not answer:
        return False
    got = normalize_answer(answer, answer_type, normalize_cfg)
    if not got:
        return False
    return any(
        got == normalize_answer(accepted, answer_type, normalize_cfg)
        for accepted in accepted_answers
    )


def qa_case_metric(
    case: QueryCase,
    outcome: QaOutcome,
    tolerance_ms: int,
    normalize_cfg: VqaNormalizeConfig,
    elapsed_s: float = 0.0,
) -> QaCaseMetric:
    """Score one QA case into a :class:`QaCaseMetric`.

    ``moment_ok`` applies the KIS hit tolerance to the SUBMITTED moment —
    ``results[0]``, which the answerer orders as the moment backing its answer
    (the top Track A card, or the winning Track B group's member). The
    archetype is the labelled one when present (so a mislabelled compiler
    archetype does not move the case between buckets), else the compiler's.
    """
    moment_ok = bool(outcome.results) and is_hit(
        outcome.results[0], case.truths, tolerance_ms
    )
    answer_ok = answer_matches(
        outcome.answer, case.accepted_answers, outcome.answer_type, normalize_cfg
    )
    archetype = case.archetype or outcome.archetype
    return QaCaseMetric(
        archetype=archetype,
        answer_ok=answer_ok,
        moment_ok=moment_ok,
        elapsed_s=elapsed_s,
    )
