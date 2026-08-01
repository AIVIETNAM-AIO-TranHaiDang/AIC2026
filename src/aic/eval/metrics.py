"""Retrieval metrics for the KIS evaluation harness.

The primary quantities, per docs/note/09-kis-implementation-plan.md Phase 1:

- Recall@k over the fixture cases,
- median and mean rank of the first hit (misses take a configured penalty rank),
- a time-to-target proxy that converts rank into the seconds an operator would
  spend scanning results, since the competition scores time, not rank.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from aic.eval.fixture import GroundTruth
from aic.retrieval.base import SearchResult


def is_hit(result: SearchResult, truths: list[GroundTruth], tolerance_ms: int) -> bool:
    """A result hits if it lands inside any labelled range, widened by the tolerance."""
    for truth in truths:
        if result.video_id != truth.video_id:
            continue
        if (
            truth.t_start_ms - tolerance_ms
            <= result.timestamp_ms
            <= (truth.t_end_ms + tolerance_ms)
        ):
            return True
    return False


def rank_of_first_hit(
    results: list[SearchResult],
    truths: list[GroundTruth],
    tolerance_ms: int,
) -> int | None:
    """1-based rank of the first hit, or None if no result hits."""
    for rank, result in enumerate(results, start=1):
        if is_hit(result, truths, tolerance_ms):
            return rank
    return None


def recall_at_k(ranks: list[int | None], k: int) -> float:
    """Fraction of cases whose first hit is at rank <= k."""
    if not ranks:
        raise ValueError("ranks must not be empty")
    hits = sum(1 for rank in ranks if rank is not None and rank <= k)
    return hits / len(ranks)


def time_to_target_proxy_s(
    rank: int | None,
    per_rank_cost_s: float,
    miss_penalty_rank: int,
) -> float:
    """Seconds an operator would spend reaching the target at this rank.

    A miss is charged as if the target sat at the penalty rank, which keeps the
    proxy finite and comparable across systems. Escalation latency is added by
    the caller once escalations exist (Phase 7).
    """
    effective_rank = miss_penalty_rank if rank is None else min(rank, miss_penalty_rank)
    return effective_rank * per_rank_cost_s


@dataclass(frozen=True)
class MetricsReport:
    n_cases: int
    n_misses: int
    recall_at: dict[int, float]
    median_rank: float
    mean_rank: float
    mean_time_to_target_s: float
    per_case_ranks: dict[str, int | None]


@dataclass(frozen=True)
class QaCaseMetric:
    """One QA case's outcome, the three DRES-relevant booleans + wall time."""

    archetype: str
    answer_ok: bool
    moment_ok: bool
    elapsed_s: float = 0.0

    @property
    def qa_correct(self) -> bool:
        """The DRES-judged quantity: right answer AND right moment."""
        return self.answer_ok and self.moment_ok


@dataclass(frozen=True)
class ArchetypeRates:
    n: int
    answer_accuracy: float
    moment_hit_rate: float
    qa_correct_rate: float


@dataclass(frozen=True)
class QaMetricsReport:
    n_cases: int
    answer_accuracy: float
    moment_hit_rate: float
    qa_correct_rate: float
    mean_time_per_case_s: float
    per_archetype: dict[str, ArchetypeRates]
    per_case: dict[str, dict]


def _rates(metrics: list[QaCaseMetric]) -> tuple[float, float, float]:
    n = len(metrics)
    return (
        sum(m.answer_ok for m in metrics) / n,
        sum(m.moment_ok for m in metrics) / n,
        sum(m.qa_correct for m in metrics) / n,
    )


def summarize_qa(per_case: dict[str, QaCaseMetric]) -> QaMetricsReport:
    """Aggregate QA case outcomes into overall + per-archetype rates.

    Kept entirely separate from :func:`summarize` (risk 3.4): a KIS-only
    fixture never calls this, so its :class:`MetricsReport` is byte-identical.
    """
    if not per_case:
        raise ValueError("per_case must not be empty")
    metrics = list(per_case.values())
    answer_accuracy, moment_hit_rate, qa_correct_rate = _rates(metrics)
    by_archetype: dict[str, list[QaCaseMetric]] = {}
    for metric in metrics:
        by_archetype.setdefault(metric.archetype, []).append(metric)
    per_archetype = {}
    for archetype, group in by_archetype.items():
        answer, moment, correct = _rates(group)
        per_archetype[archetype] = ArchetypeRates(
            n=len(group),
            answer_accuracy=answer,
            moment_hit_rate=moment,
            qa_correct_rate=correct,
        )
    return QaMetricsReport(
        n_cases=len(metrics),
        answer_accuracy=answer_accuracy,
        moment_hit_rate=moment_hit_rate,
        qa_correct_rate=qa_correct_rate,
        mean_time_per_case_s=float(statistics.fmean(m.elapsed_s for m in metrics)),
        per_archetype=per_archetype,
        per_case={
            query_id: {
                "archetype": m.archetype,
                "answer_ok": m.answer_ok,
                "moment_ok": m.moment_ok,
                "qa_correct": m.qa_correct,
                "elapsed_s": round(m.elapsed_s, 3),
            }
            for query_id, m in per_case.items()
        },
    )


def summarize(
    per_case_ranks: dict[str, int | None],
    k_values: list[int],
    miss_penalty_rank: int,
    per_rank_cost_s: float,
) -> MetricsReport:
    """Aggregate per-case first-hit ranks into a report."""
    if not per_case_ranks:
        raise ValueError("per_case_ranks must not be empty")
    ranks = list(per_case_ranks.values())
    effective = [miss_penalty_rank if rank is None else rank for rank in ranks]
    proxies = [
        time_to_target_proxy_s(rank, per_rank_cost_s, miss_penalty_rank)
        for rank in ranks
    ]
    return MetricsReport(
        n_cases=len(ranks),
        n_misses=sum(1 for rank in ranks if rank is None),
        recall_at={k: recall_at_k(ranks, k) for k in k_values},
        median_rank=float(statistics.median(effective)),
        mean_rank=float(statistics.fmean(effective)),
        mean_time_to_target_s=float(statistics.fmean(proxies)),
        per_case_ranks=dict(per_case_ranks),
    )
