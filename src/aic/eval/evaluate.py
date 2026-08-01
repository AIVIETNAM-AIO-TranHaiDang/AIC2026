"""The harness runner: evaluate any Retriever against a labelled fixture.

Merging a retrieval component without a number from this runner violates the
Phase 1 exit rule in docs/note/09-kis-implementation-plan.md.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict
from pathlib import Path

from aic.config import EvalConfig, VqaNormalizeConfig
from aic.eval.fixture import EvalFixture
from aic.eval.metrics import (
    MetricsReport,
    QaMetricsReport,
    rank_of_first_hit,
    summarize,
    summarize_qa,
)
from aic.eval.qa import QaAnswerer, QaOutcome, qa_case_metric
from aic.manifest import write_json_atomic
from aic.retrieval.base import Retriever, SearchResult

logger = logging.getLogger(__name__)


def evaluate_with_results(
    retriever: Retriever,
    fixture: EvalFixture,
    eval_cfg: EvalConfig,
) -> tuple[MetricsReport, dict[str, list[SearchResult]]]:
    """Aggregate metrics *and* return the per-case result lists.

    Queries the retriever exactly once per case and keeps its results, so a
    caller that wants to inspect what was returned (a debug dump) does not
    have to search again — important when a single search is an LLM call.

    ``top_k`` is the largest configured k: results beyond it cannot change any
    reported metric, so asking for more would only cost latency.
    """
    top_k = max(eval_cfg.k_values)
    per_case_ranks: dict[str, int | None] = {}
    per_case_results: dict[str, list[SearchResult]] = {}
    # QA cases are scored on a different axis (answer + moment) and summarised
    # separately, so this KIS loop skips them; a KIS-only fixture is therefore
    # scored exactly as before (report byte-identical, risk 3.4).
    kis_cases = [case for case in fixture.cases if case.kind != "qa"]
    # One progress bar over the cases with per-case wall time: on the cortex
    # path a single case can spend minutes in LLM calls (compile + automatic
    # rerank), and a silent loop is indistinguishable from a hang.
    from tqdm.auto import tqdm
    from tqdm.contrib.logging import logging_redirect_tqdm

    with (
        logging_redirect_tqdm(),
        tqdm(total=len(kis_cases), desc="eval", unit="case") as bar,
    ):
        for case in kis_cases:
            started = time.perf_counter()
            results = retriever.search(case, top_k=top_k)
            elapsed_s = time.perf_counter() - started
            if len(results) > top_k:
                raise ValueError(
                    f"retriever returned {len(results)} results for case "
                    f"{case.query_id}, more than the requested {top_k}"
                )
            per_case_results[case.query_id] = results
            rank = rank_of_first_hit(results, case.truths, eval_cfg.hit_tolerance_ms)
            per_case_ranks[case.query_id] = rank
            logger.debug("case %s: first hit at rank %s", case.query_id, rank)
            bar.set_postfix_str(f"last={case.query_id} rank={rank} {elapsed_s:.1f}s")
            bar.update(1)
    report = summarize(
        per_case_ranks,
        k_values=eval_cfg.k_values,
        miss_penalty_rank=eval_cfg.miss_penalty_rank,
        per_rank_cost_s=eval_cfg.per_rank_cost_s,
    )
    return report, per_case_results


def evaluate(
    retriever: Retriever,
    fixture: EvalFixture,
    eval_cfg: EvalConfig,
) -> MetricsReport:
    """Run every KIS fixture case through the retriever and aggregate metrics."""
    report, _ = evaluate_with_results(retriever, fixture, eval_cfg)
    return report


def evaluate_qa_with_results(
    answerer: QaAnswerer,
    fixture: EvalFixture,
    eval_cfg: EvalConfig,
    normalize_cfg: VqaNormalizeConfig,
) -> tuple[QaMetricsReport, dict[str, QaOutcome]]:
    """Score the fixture's ``kind: qa`` cases through a QA answerer.

    Each case is answered and located once; ``moment_ok`` reuses the KIS hit
    tolerance, ``answer_ok`` the configured normalised match, and their
    conjunction is ``qa_correct``. The per-case tqdm postfix shows ``A=``
    (answer) / ``M=`` (moment) so a stalled reader is visible (X10). Raises if
    the fixture has no QA cases (the caller checks first).
    """
    qa_cases = [case for case in fixture.cases if case.kind == "qa"]
    if not qa_cases:
        raise ValueError("fixture has no qa cases")
    top_k = max(eval_cfg.k_values)
    per_case_metric = {}
    per_case_outcome: dict[str, QaOutcome] = {}
    from tqdm.auto import tqdm
    from tqdm.contrib.logging import logging_redirect_tqdm

    with (
        logging_redirect_tqdm(),
        tqdm(total=len(qa_cases), desc="qa-eval", unit="case") as bar,
    ):
        for case in qa_cases:
            started = time.perf_counter()
            outcome = answerer.answer(case, top_k=top_k)
            elapsed_s = time.perf_counter() - started
            if len(outcome.results) > top_k:
                raise ValueError(
                    f"qa answerer returned {len(outcome.results)} results for "
                    f"case {case.query_id}, more than the requested {top_k}"
                )
            metric = qa_case_metric(
                case, outcome, eval_cfg.hit_tolerance_ms, normalize_cfg, elapsed_s
            )
            per_case_metric[case.query_id] = metric
            per_case_outcome[case.query_id] = outcome
            bar.set_postfix_str(
                f"last={case.query_id} A={int(metric.answer_ok)} "
                f"M={int(metric.moment_ok)} {elapsed_s:.1f}s"
            )
            bar.update(1)
    return summarize_qa(per_case_metric), per_case_outcome


def report_to_markdown(report: MetricsReport) -> str:
    """Render a report as a compact Markdown table for implementation notes."""
    lines = [
        "| metric | value |",
        "| --- | --- |",
        f"| cases | {report.n_cases} |",
        f"| misses | {report.n_misses} |",
    ]
    for k in sorted(report.recall_at):
        lines.append(f"| recall@{k} | {report.recall_at[k]:.3f} |")
    lines.append(f"| median rank | {report.median_rank:.1f} |")
    lines.append(f"| mean rank | {report.mean_rank:.1f} |")
    lines.append(f"| mean time-to-target (s) | {report.mean_time_to_target_s:.1f} |")
    return "\n".join(lines)


def qa_report_to_markdown(report: QaMetricsReport) -> str:
    """Render a QA report as a compact Markdown table (overall + archetype)."""
    lines = [
        "| qa metric | value |",
        "| --- | --- |",
        f"| cases | {report.n_cases} |",
        f"| answer accuracy | {report.answer_accuracy:.3f} |",
        f"| moment hit rate | {report.moment_hit_rate:.3f} |",
        f"| qa correct (both) | {report.qa_correct_rate:.3f} |",
        f"| mean time/case (s) | {report.mean_time_per_case_s:.1f} |",
    ]
    if report.per_archetype:
        lines.append("")
        lines.append("| archetype | n | answer | moment | correct |")
        lines.append("| --- | --- | --- | --- | --- |")
        for archetype in sorted(report.per_archetype):
            rate = report.per_archetype[archetype]
            lines.append(
                f"| {archetype} | {rate.n} | {rate.answer_accuracy:.3f} | "
                f"{rate.moment_hit_rate:.3f} | {rate.qa_correct_rate:.3f} |"
            )
    return "\n".join(lines)


def save_report(report: MetricsReport, path: Path | str) -> None:
    """Persist the full report (including per-case ranks) as JSON."""
    write_json_atomic(path, asdict(report))


def save_qa_report(report: QaMetricsReport, path: Path | str) -> None:
    """Persist the full QA report (overall + per-archetype + per-case) as JSON."""
    write_json_atomic(path, asdict(report))
