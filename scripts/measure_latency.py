"""Measure /api/search latency: p50/p95 wall time plus per-stage breakdown.

The Phase 6 exit criterion (warm p50 < 500 ms) and the Phase 10 go/no-go
evidence (is vector search the bottleneck at full scale?) both come from
this script. Queries default to the fixture's text cases so the numbers
reflect real workload, and every response's ``timings_ms`` (compile / rank
/ filter / advise / respond) is aggregated so the bottleneck is named, not
guessed.

By default the app runs in-process (no network noise); pass --url to
measure a separately running service instead.

Usage:
    python scripts/measure_latency.py --config configs/t0.yaml \
        --fixture data/fixture.yaml [--repeats 5] [--url http://host:8000] \
        [--report data/reports/latency.json]
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aic.config import load_config  # noqa: E402
from aic.log import setup_logging  # noqa: E402
from aic.models_cache import apply_model_cache_env  # noqa: E402


def _percentiles(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "p50": statistics.median(ordered),
        "p95": ordered[min(int(round(0.95 * (len(ordered) - 1))), len(ordered) - 1)],
        "max": ordered[-1],
        "n": len(ordered),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Profile config YAML.")
    parser.add_argument("--fixture", required=True, help="Fixture YAML path.")
    parser.add_argument(
        "--repeats", type=int, default=5, help="Timed passes over the query set."
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Untimed passes first (the target is WARM latency).",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Measure a running service instead of an in-process app.",
    )
    parser.add_argument("--report", default=None, help="Save the JSON report here.")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()

    setup_logging(args.log_level)
    cfg = load_config(args.config)
    apply_model_cache_env(cfg.paths.models_dir)

    import time

    from aic.eval.fixture import load_fixture
    from aic.manifest import write_json_atomic
    from aic.retrieval.fusion import query_text

    queries = [
        query_text(case)
        for case in load_fixture(args.fixture).cases
        if case.kind != "kis_v"
    ]
    if not queries:
        print("fixture has no text cases; nothing to measure")
        return 1

    if args.url:
        import httpx

        client = httpx.Client(base_url=args.url, timeout=120.0)
        post = client.post
    else:
        from fastapi.testclient import TestClient

        from aic.service.app import create_app
        from aic.service.state import build_service_state

        client = TestClient(create_app(build_service_state(cfg), cfg.service))
        post = client.post

    def run_pass(record: bool, wall: list[float], stages: dict[str, list[float]]):
        for query in queries:
            start = time.perf_counter()
            response = post("/api/search", json={"query": query})
            elapsed_ms = (time.perf_counter() - start) * 1000
            response.raise_for_status()
            if record:
                wall.append(elapsed_ms)
                for stage, ms in (response.json().get("timings_ms") or {}).items():
                    stages.setdefault(stage, []).append(ms)

    wall: list[float] = []
    stages: dict[str, list[float]] = {}
    for _ in range(args.warmup):
        run_pass(False, wall, stages)
    for _ in range(args.repeats):
        run_pass(True, wall, stages)

    report = {
        "queries": len(queries),
        "repeats": args.repeats,
        "wall_ms": _percentiles(wall),
        "stages_ms": {stage: _percentiles(ms) for stage, ms in stages.items()},
    }
    print(f"\n{len(queries)} queries x {args.repeats} repeats (warm)")
    print(
        f"wall: p50 {report['wall_ms']['p50']:.0f} ms, "
        f"p95 {report['wall_ms']['p95']:.0f} ms "
        "(target: p50 < 500 ms, note 09 Phase 6)"
    )
    for stage, values in sorted(
        report["stages_ms"].items(), key=lambda item: -item[1]["p50"]
    ):
        print(f"  {stage}: p50 {values['p50']:.0f} ms, p95 {values['p95']:.0f} ms")
    if args.report:
        write_json_atomic(args.report, report)
        print(f"report saved to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
