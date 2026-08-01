"""Cold-start drill: load everything the service needs and time it.

Phase 9's one-command warm-up — run it before a round (or after a crash)
and read the total against the < 5 minute target from note 09. Loads every
index and model the fast path touches and pushes one throwaway query
through the engine so lazy weights are resident before the first real
query.

Usage:
    python scripts/warmup.py --config configs/t0.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aic.config import load_config  # noqa: E402
from aic.log import setup_logging  # noqa: E402
from aic.models_cache import apply_model_cache_env  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Profile config YAML.")
    parser.add_argument(
        "--query",
        default="a person speaking in a news studio",
        help="Throwaway query used to warm the lazy models.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    cfg = load_config(args.config)
    apply_model_cache_env(cfg.paths.models_dir)

    from aic.cortex.spec import fallback_spec
    from aic.service.state import build_service_state

    timings: dict[str, float] = {}
    total_start = time.perf_counter()

    start = time.perf_counter()
    state = build_service_state(cfg)
    timings["load artifacts (indexes, bundles, sessions)"] = (
        time.perf_counter() - start
    )

    # The engine loads encoders lazily; one spec through the fast path
    # forces every fast-path model into memory. The fallback spec is used
    # on purpose: warm-up must not depend on the LLM endpoint.
    start = time.perf_counter()
    # The raw query is passed too so the raw-query dispatch and the overlay
    # video prior warm alongside the channels.
    candidates = state.engine.rank_spec(fallback_spec(args.query), args.query)
    timings["warm fast-path models (first query)"] = time.perf_counter() - start

    start = time.perf_counter()
    state.engine.rank_spec(fallback_spec(args.query), args.query)
    timings["warm repeat query"] = time.perf_counter() - start

    total = time.perf_counter() - total_start
    print()
    for stage, seconds in timings.items():
        print(f"{stage}: {seconds:.1f}s")
    print(f"total cold start: {total:.1f}s (target < 300s, note 09 Phase 9)")
    print(
        f"fast path returned {len(candidates)} candidates; "
        f"escalations wired: {sorted(state.escalations) or 'none'}; "
        f"QPP advisor: {'on' if state.qpp is not None else 'off'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
