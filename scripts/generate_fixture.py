"""Generate a synthetic evaluation fixture from the ingested corpus.

A VLM writes one AIC-style Vietnamese KIS query per randomly sampled truth
window (labels exact by construction; see note 15 and the fixture_gen
config section). Which endpoint writes the queries follows the profile:
the API profiles point fixture_gen at Gemini, the local profile at the
self-hosted model. Keep the hand-labelled fixture as a holdout — never
merge the two files.

Usage:
    python scripts/generate_fixture.py --config configs/t0.yaml \
        [--out data/fixture.synthetic.yaml] [--count 20]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aic.config import load_config  # noqa: E402
from aic.log import setup_logging  # noqa: E402
from aic.models_cache import apply_model_cache_env  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Profile config YAML.")
    parser.add_argument(
        "--out",
        default="data/fixture.synthetic.yaml",
        help="Where to write the generated fixture YAML.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Override fixture_gen.count (e.g. a small smoke run).",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    cfg = load_config(args.config)
    apply_model_cache_env(cfg.paths.models_dir)

    if cfg.fixture_gen is None:
        raise SystemExit(
            "this profile has no fixture_gen section; add one (see "
            "configs/t0.yaml) to choose the generation endpoint"
        )
    gen_cfg = cfg.fixture_gen
    if args.count is not None:
        gen_cfg = gen_cfg.model_copy(update={"count": args.count})

    from aic.eval.generate import generate_fixture

    stats = generate_fixture(cfg, gen_cfg, Path(args.out))
    print(
        f"fixture written to {args.out}: {stats.generated} case(s), "
        f"{stats.failed} failed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
