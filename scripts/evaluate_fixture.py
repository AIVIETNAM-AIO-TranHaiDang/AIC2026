"""Evaluate the fusion retriever against the labelled fixture.

Prints the metrics table and saves the full JSON report. Channel ablations
are run by editing retrieval.fusion.channel_weights in the config (removing
a channel disables it) and re-running this script.

Usage:
    python scripts/evaluate_fixture.py --config configs/t0.yaml \
        --fixture data/fixture.yaml [--report data/reports/baseline.json]

Pass ``--debug-top N`` to also dump, per case, the top-N returned
``(video_id, timestamp_ms)`` with a HIT marker against the labelled ranges.
This separates a genuine retrieval weakness (top results are the *wrong*
video) from a labelling artifact (top results are the *right* video but fall
outside a truth window that was drawn too narrowly).

Pass ``--use-cortex`` to evaluate the Phase 6 online path (Cortex compile ->
spec dispatch) instead of the raw single-string fusion, so the measured
numbers match what production serves (including VI->EN translation of the
visual query). ``--only-channels`` composes with both paths.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aic.config import load_config  # noqa: E402
from aic.log import setup_logging  # noqa: E402
from aic.models_cache import apply_model_cache_env  # noqa: E402


def _build_dispatcher(cfg, encoder, dense_embedder, sparse_embedder, fixture_dir):
    """The shared spec dispatcher + shot metadata + evidence bundles.

    The compile->dispatch->fuse core both the KIS retriever and the QA answerer
    build on; kept in one place so the two harness paths use the same fusion.
    """
    from aic.cortex.dispatch import SpecDispatcher
    from aic.retrieval.build import (
        build_channels,
        load_overlay_prior,
        load_shot_meta,
    )
    from aic.service.bundles import load_bundles

    channels = build_channels(
        cfg,
        encoder,
        dense_embedder,
        fixture_dir=fixture_dir,
        sparse_text_embedder=sparse_embedder,
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
    return dispatcher, shot_meta, load_bundles(cfg)


def _build_cortex_retriever(cfg, encoder, dense_embedder, sparse_embedder, fixture_dir):
    """Wire the Phase 6 online retriever (compile -> dispatch -> filter -> ground).

    Mirrors the construction in scripts/train_qpp.py and the service so the
    harness measures the same path production serves — including temporal
    window fusion, the overlay video prior, and the automatic VLM rerank
    when the profile enables them.
    """
    from aic.cortex.compiler import CortexCompiler
    from aic.cortex.dispatch import CortexRetriever

    dispatcher, shot_meta, bundles = _build_dispatcher(
        cfg, encoder, dense_embedder, sparse_embedder, fixture_dir
    )
    evidence = {key: bundle.evidence_text for key, bundle in bundles.items()}
    auto_rerank = None
    if cfg.retrieval.auto_rerank.enabled:
        from aic.escalate.verify import VlmVerify

        auto_rerank = VlmVerify(
            bundles=bundles,
            frames_dir=cfg.paths.keyframes_dir,
            cfg=cfg.escalations.vlm_verify,
        )
        print(
            "auto rerank: vlm_verify on every case "
            f"(top_n={cfg.escalations.vlm_verify.top_n}, "
            f"timeout={cfg.retrieval.auto_rerank.timeout_s:.0f}s)"
        )
    return CortexRetriever(
        compiler=CortexCompiler(cfg.cortex),
        dispatcher=dispatcher,
        shot_meta=shot_meta,
        clip_retriever=None,
        evidence_text=evidence,
        negation_filter=cfg.cortex.dispatch.negation_filter,
        auto_rerank=auto_rerank,
        auto_rerank_timeout_s=cfg.retrieval.auto_rerank.timeout_s,
    )


class CortexQaAnswerer:
    """The QA branch of the harness: compile -> locate -> Track A/B answer.

    Composes the same pieces the `/api/qa` route does (compile_qa, the spec
    dispatcher, the Answer Ledger, and — when a reader endpoint is configured —
    the phase-2 reader + vote), through the shared :mod:`aic.vqa.answer`
    derivation, so the measured answer is the one the operator would submit.
    """

    def __init__(
        self, cfg, compiler, dispatcher, shot_meta, bundles, ledger, qa_reader
    ):
        self._cfg = cfg
        self._compiler = compiler
        self._dispatcher = dispatcher
        self._shot_meta = shot_meta
        self._bundles = bundles
        self._ledger = ledger
        self._qa_reader = qa_reader
        self._evidence = {k: b.evidence_text for k, b in bundles.items()}

    def _video_of(self, shot_key: str) -> str:
        bundle = self._bundles.get(shot_key)
        if bundle is not None:
            return bundle.video_id
        meta = self._shot_meta.get(shot_key)
        return meta.video_id if meta is not None else ""

    def answer(self, case, top_k):
        from aic.eval.qa import QaOutcome
        from aic.retrieval.base import SearchResult
        from aic.retrieval.fusion import FACTOID_SPARSE, ground_candidates
        from aic.vqa.answer import ledger_answer, track_b_groups

        plan = self._compiler.compile_qa(case.question, self._cfg.vqa.archetypes)
        # Match the /api/qa route: the locator also probes the Answer Ledger
        # factoid channel at vqa.factoid_weight (inert when 0 or not built).
        extra = None
        if self._cfg.vqa.factoid_weight > 0:
            extra = {FACTOID_SPARSE: self._cfg.vqa.factoid_weight}
        candidates = self._dispatcher.rank_spec(
            plan.locator, case.question, extra_channel_weights=extra
        )
        # Match the /api/qa route: filter negations before Track A/B when the
        # profile enables it, so the harness measures the served path.
        if self._cfg.cortex.dispatch.negation_filter:
            from aic.cortex.dispatch import filter_negations

            candidates = filter_negations(
                candidates, plan.locator.negations, self._evidence
            )
        results = ground_candidates(candidates, self._shot_meta, top_k)

        # Track A: the first candidate whose ledger row answers this archetype.
        answer = None
        for candidate in candidates[:top_k]:
            guess, _ = ledger_answer(
                self._ledger.get(candidate.shot_key), plan.archetype
            )
            if guess:
                answer = guess
                break

        # Track B: read + vote when a reader endpoint is configured. The winning
        # group's member is the submitted moment, so it leads the result list.
        if self._qa_reader is not None:
            groups = track_b_groups(
                self._qa_reader,
                self._cfg.vqa,
                plan,
                case.question,
                candidates,
                self._bundles,
                self._shot_meta,
                self._video_of,
                [answer] if answer else [],
            )
            if groups:
                answer = groups[0].answer
                winner = groups[0].members[0]
                # Prepend the winning moment, then re-truncate: the grounded
                # list is already top_k long, and the harness rejects an
                # over-length result list (this crashed real-corpus QA runs
                # whenever Track B produced a winner).
                results = [
                    SearchResult(
                        video_id=winner.video_id,
                        timestamp_ms=winner.timestamp_ms,
                        score=groups[0].score,
                    ),
                    *results,
                ][:top_k]

        return QaOutcome(
            results=results,
            answer=answer,
            archetype=plan.archetype,
            answer_type=plan.expected_answer_type,
        )


def _build_qa_answerer(cfg, encoder, dense_embedder, sparse_embedder, fixture_dir):
    """Build the production QA answerer for the harness (mirrors state.py)."""
    from aic.cortex.compiler import CortexCompiler

    dispatcher, shot_meta, bundles = _build_dispatcher(
        cfg, encoder, dense_embedder, sparse_embedder, fixture_dir
    )
    ledger: dict = {}
    if cfg.vqa.enabled and cfg.chronicle.ledger.enabled:
        from aic.chronicle.ledger import load_ledger_map

        ledger = dict(load_ledger_map(cfg))
    qa_reader = None
    if cfg.vqa.enabled and cfg.vqa.reader.enabled:
        from aic.vqa.reader import QaReader

        qa_reader = QaReader(
            cfg.vqa.reader,
            frames_dir=cfg.paths.keyframes_dir,
            labels=cfg.vqa.evidence_labels,
        )
        print("qa answerer: Track B reader on")
    return CortexQaAnswerer(
        cfg,
        CortexCompiler(cfg.cortex),
        dispatcher,
        shot_meta,
        bundles,
        ledger,
        qa_reader,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Profile config YAML.")
    parser.add_argument("--fixture", required=True, help="Fixture YAML path.")
    parser.add_argument("--report", default=None, help="Where to save the JSON report.")
    parser.add_argument(
        "--only-channels",
        default=None,
        help=(
            "Comma-separated fusion channels to keep active (ablation), e.g. "
            "'dense_visual' or 'semantic_dense,literal_sparse'. Others are "
            "disabled for this run without editing the config."
        ),
    )
    parser.add_argument(
        "--use-cortex",
        action="store_true",
        help=(
            "Evaluate the Phase 6 online path (Cortex compile -> spec dispatch) "
            "instead of the raw single-string fusion. This is what production "
            "runs: the Cortex folds VI->EN translation into compilation, so the "
            "visual channel sees English. Needs the cortex LLM key set (else the "
            "Cortex falls back to the raw query, matching the default path)."
        ),
    )
    parser.add_argument(
        "--compare",
        default=None,
        help=(
            "Second profile config evaluated in the SAME process right after "
            "the first, reusing the already-loaded encoder and text "
            "embedders (an A/B over retrieval levers, e.g. visual chunking, "
            "without paying a second multi-minute model load - repeated cold "
            "loads in one GPU session have frozen mid-materialisation under "
            "memory pressure). The embed and textstack sections must equal "
            "the primary config's, since the models are shared. Prints both "
            "reports and a side-by-side summary."
        ),
    )
    parser.add_argument(
        "--compare-report",
        default=None,
        help="Where to save the --compare run's JSON report.",
    )
    parser.add_argument(
        "--debug-top",
        type=int,
        default=0,
        help="If > 0, dump the top-N returned results per case with HIT markers.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    cfg = load_config(args.config)
    apply_model_cache_env(cfg.paths.models_dir)

    if args.only_channels:
        if args.compare:
            raise SystemExit(
                "--only-channels cannot be combined with --compare; run the "
                "channel ablation per config instead"
            )
        keep = {name.strip() for name in args.only_channels.split(",") if name.strip()}
        weights = cfg.retrieval.fusion.channel_weights
        unknown = keep - set(weights)
        if unknown:
            raise SystemExit(
                f"--only-channels not in config channel_weights: {sorted(unknown)}"
            )
        for name in list(weights):
            if name not in keep:
                del weights[name]
        print(f"ablation: active channels = {sorted(weights)}")

    runs = [(args.config, cfg, args.report)]
    if args.compare:
        compare_cfg = load_config(args.compare)
        if compare_cfg.embed != cfg.embed or compare_cfg.textstack != cfg.textstack:
            raise SystemExit(
                "--compare requires identical embed and textstack sections: "
                "the loaded models are shared between the two runs, so a "
                "config that changes them would be evaluated with the wrong "
                "models"
            )
        runs.append((args.compare, compare_cfg, args.compare_report))

    from aic.embed.encoders import build_image_text_encoder
    from aic.eval.evaluate import (
        evaluate_qa_with_results,
        evaluate_with_results,
        qa_report_to_markdown,
        report_to_markdown,
        save_qa_report,
        save_report,
    )
    from aic.eval.fixture import load_fixture
    from aic.eval.metrics import is_hit
    from aic.retrieval.build import build_fusion_retriever
    from aic.textstack.embedder import build_textstack_embedders

    fixture_path = Path(args.fixture)
    fixture = load_fixture(fixture_path)
    has_kis = any(case.kind != "qa" for case in fixture.cases)
    has_qa = any(case.kind == "qa" for case in fixture.cases)
    encoder = build_image_text_encoder(cfg.embed, cfg.paths.models_dir)
    dense_embedder, sparse_embedder = build_textstack_embedders(
        cfg.textstack, cfg.paths.models_dir
    )

    summaries: list[tuple[str, object]] = []
    for config_path, run_cfg, report_path in runs:
        if len(runs) > 1:
            print(f"\n=== config: {config_path} ===")

        # QA cases (kind: qa) are scored on the answer+moment axis by a QA
        # answerer; they are summarised separately so a KIS-only fixture is
        # byte-identical. A mixed fixture runs both.
        if has_qa:
            answerer = _build_qa_answerer(
                run_cfg, encoder, dense_embedder, sparse_embedder, fixture_path.parent
            )
            qa_report, _ = evaluate_qa_with_results(
                answerer, fixture, run_cfg.eval, run_cfg.vqa.normalize
            )
            print(qa_report_to_markdown(qa_report))
            if report_path:
                qa_report_path = str(Path(report_path).with_suffix(".qa.json"))
                save_qa_report(qa_report, qa_report_path)
                print(f"qa report saved to {qa_report_path}")

        if not has_kis:
            summaries.append((config_path, None))
            continue

        if args.use_cortex:
            retriever = _build_cortex_retriever(
                run_cfg, encoder, dense_embedder, sparse_embedder, fixture_path.parent
            )
            print(
                "retriever: cortex online path "
                f"(cortex.enabled={run_cfg.cortex.enabled})"
            )
        else:
            retriever = build_fusion_retriever(
                run_cfg,
                encoder=encoder,
                text_embedder=dense_embedder,
                fixture_dir=fixture_path.parent,
                sparse_text_embedder=sparse_embedder,
            )

        # One search pass per case, reused for both the metrics and the debug
        # dump, so --debug-top does not re-query the retriever: on the cortex
        # path every search is an LLM call, and re-querying doubled them into
        # rate limits.
        report, per_case_results = evaluate_with_results(
            retriever, fixture, run_cfg.eval
        )
        print(report_to_markdown(report))
        if report_path:
            save_report(report, report_path)
            print(f"report saved to {report_path}")
        summaries.append((config_path, report))

        if args.debug_top > 0:
            tol = run_cfg.eval.hit_tolerance_ms
            print(
                f"\n--- top-{args.debug_top} results per case "
                f"(tolerance {tol} ms) ---"
            )
            for case in fixture.cases:
                if case.query_id not in report.per_case_ranks:
                    continue  # a qa case: scored on the answer+moment axis
                truths = ", ".join(
                    f"{t.video_id}[{t.t_start_ms}-{t.t_end_ms}]" for t in case.truths
                )
                rank = report.per_case_ranks[case.query_id]
                print(f"\n[{case.query_id}] {case.kind}  rank={rank}  truth: {truths}")
                for i, result in enumerate(
                    per_case_results[case.query_id][: args.debug_top], start=1
                ):
                    mark = "HIT " if is_hit(result, case.truths, tol) else "    "
                    kf = result.keyframe_id or ""
                    print(
                        f"  {i:>3} {mark}{result.video_id} @ "
                        f"{result.timestamp_ms:>6} ms"
                        f"  score={result.score:.4f}  {kf}"
                    )

    if len([s for s in summaries if s[1] is not None]) > 1:
        print("\n--- side by side ---")
        for config_path, report in summaries:
            if report is None:
                continue  # qa-only run: no KIS side-by-side
            recalls = "  ".join(
                f"R@{k}={v:.3f}" for k, v in sorted(report.recall_at.items())
            )
            print(
                f"{config_path}: {recalls}  median={report.median_rank:.1f}"
                f"  mean={report.mean_rank:.1f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
