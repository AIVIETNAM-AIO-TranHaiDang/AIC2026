"""Build the Chronicle text indexes (semantic dense + literal sparse).

Runs after scripts/build_chronicle.py.

Usage:
    python scripts/build_text_indexes.py --config configs/t0.yaml
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
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    cfg = load_config(args.config)
    apply_model_cache_env(cfg.paths.models_dir)

    from aic.textstack.embedder import build_textstack_embedders
    from aic.textstack.job import build_text_indexes

    dense_embedder, sparse_embedder = build_textstack_embedders(
        cfg.textstack, cfg.paths.models_dir
    )
    stats = build_text_indexes(cfg, dense_embedder, sparse_embedder)
    print(
        f"semantic_docs={stats.semantic_docs} literal_docs={stats.literal_docs} "
        f"shots={stats.total_shots}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
