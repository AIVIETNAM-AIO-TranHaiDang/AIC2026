"""Assert GPU and CPU FAISS search agree before trusting index.device: cuda.

Phase 10's exit criterion is "no change at all to retrieval results": this
script loads the same persisted index twice (CPU and GPU), runs identical
queries through both, and fails loudly on any top-k id mismatch. With
--float16 it additionally reports the fp16 score drift so the precision
trade-off is measured, not assumed.

Run on the GPU server (needs the faiss-gpu-cu12 build from
requirements-gpu.txt); on a faiss-cpu machine it exits with a clear message
instead of pretending to pass.

Usage:
    python scripts/faiss_gpu_parity.py --config configs/t0.yaml \
        [--queries 256] [--top-k 100] [--float16] [--seed 0]
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
        "--queries", type=int, default=256, help="Random unit query vectors."
    )
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument(
        "--float16",
        action="store_true",
        help="Also check the fp16-resident GPU index and report score drift.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    cfg = load_config(args.config)
    apply_model_cache_env(cfg.paths.models_dir)

    import faiss
    import numpy as np

    from aic.embed.encoders import build_image_text_encoder
    from aic.index.vector import VectorIndex
    from aic.retrieval.build import keyframe_index_dir

    if not hasattr(faiss, "StandardGpuResources") or faiss.get_num_gpus() < 1:
        print(
            "this faiss build has no usable GPU (faiss-cpu installed, or no "
            "CUDA device); install faiss-gpu-cu12 per requirements-gpu.txt "
            "and run on the GPU server"
        )
        return 1

    encoder = build_image_text_encoder(cfg.embed, cfg.paths.models_dir)
    index_dir = keyframe_index_dir(cfg, encoder.model_id)
    if not index_dir.is_dir():
        print(f"no keyframe index at {index_dir}; run the embed stage first")
        return 1

    cpu_index = VectorIndex.load(index_dir, device="cpu")
    rng = np.random.default_rng(args.seed)
    queries = rng.standard_normal((args.queries, cpu_index.dim)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    def top_ids(index: VectorIndex) -> list[list[str]]:
        hits = index.search(queries, args.top_k, query_model_id=index.model_id)
        return [[shot_id for shot_id, _score in row] for row in hits]

    def top_scores(index: VectorIndex) -> list[list[float]]:
        hits = index.search(queries, args.top_k, query_model_id=index.model_id)
        return [[score for _shot_id, score in row] for row in hits]

    cpu_ids = top_ids(cpu_index)
    variants = [("fp32", False)] + ([("fp16", True)] if args.float16 else [])
    failed = False
    for label, use_fp16 in variants:
        gpu_index = VectorIndex.load(index_dir, device="cuda", gpu_float16=use_fp16)
        gpu_ids = top_ids(gpu_index)
        mismatches = sum(
            1 for cpu_row, gpu_row in zip(cpu_ids, gpu_ids, strict=True)
            if cpu_row != gpu_row
        )
        cpu_scores = np.asarray(top_scores(cpu_index))
        gpu_scores = np.asarray(top_scores(gpu_index))
        drift = float(np.max(np.abs(cpu_scores - gpu_scores)))
        print(
            f"GPU {label}: {mismatches}/{args.queries} queries with a "
            f"different top-{args.top_k}; max |score drift| = {drift:.2e}"
        )
        if label == "fp32" and mismatches:
            failed = True
    if failed:
        print("PARITY FAILED: fp32 GPU results differ from CPU — do not enable")
        return 1
    print("parity OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
