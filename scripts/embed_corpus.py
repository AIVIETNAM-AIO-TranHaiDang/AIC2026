"""Embed ingested keyframes and build the keyframe vector index.

Runs after scripts/ingest_corpus.py. Resumable at batch granularity; the
index is rebuilt from shards at the end of every run. On the GPU server the
same command simply picks up device 'auto' as CUDA.

Usage:
    python scripts/embed_corpus.py --config configs/t0.yaml \
        [--limit N] [--num-gpus K]
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
        help="Embed at most N keyframes (pilot runs; single-GPU only).",
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
        "embed", args.config, cfg, args.num_gpus, desc="embed", limit=args.limit
    )
    print(
        f"embedded={stats.get('embedded', 0)} skipped={stats.get('skipped', 0)} "
        f"failed={stats.get('failed', 0)}"
    )
    # The index is rebuilt from the merged shards; model_id is fixed by config,
    # so no encoder needs to stay resident in the parent after embedding.
    if stats.get("embedded") or stats.get("skipped"):
        from aic.retrieval.build import build_keyframe_index

        index = build_keyframe_index(cfg, cfg.embed.model_id)
        print(f"index built: {len(index)} vectors, dim={index.dim}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
