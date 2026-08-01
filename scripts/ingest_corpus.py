"""Ingest the video corpus: shots, keyframes, and manifests.

Runs on CPU (dev machine, slow) and GPU (Colab / server, set
ingest.shots.device or leave 'auto'). The run is resumable: videos already
present in the videos manifest are skipped, so a killed Colab session simply
re-runs the same command.

Usage:
    python scripts/ingest_corpus.py --config configs/t0.yaml [--limit N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aic.config import load_config  # noqa: E402
from aic.log import setup_logging  # noqa: E402
from aic.models_cache import apply_model_cache_env  # noqa: E402
from aic.parallel import run_local_stage  # noqa: E402
from aic.verify import maybe_repair  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Profile config YAML.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N videos (pilot runs; single-GPU only).",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Data-parallel workers, one per GPU. Default: every visible GPU "
        "(1 / CPU runs in-process).",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    cfg = load_config(args.config)
    apply_model_cache_env(cfg.paths.models_dir)
    maybe_repair(cfg)
    stats = run_local_stage(
        "ingest", args.config, cfg, args.num_gpus, desc="shots", limit=args.limit
    )
    print(
        f"processed={stats.get('processed', 0)} skipped={stats.get('skipped', 0)} "
        f"failed={stats.get('failed', 0)} keyframes={stats.get('keyframes', 0)}"
    )
    return 1 if stats.get("failed") and not stats.get("processed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
