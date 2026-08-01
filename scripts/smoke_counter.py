"""GPU smoke test for the VQA detector-counter (phase 4).

Runs the configured (or ``--backend`` overridden) detector-counter over one
video's keyframes, prints per-concept counts and the minutes-per-video figure
the note-18 honesty table needs before enabling ``ledger.counter`` corpus-wide.
Also asserts the gated-token path: with no HF token resolved for a gated
backend it exits with a clear "request access + set $VAR" message rather than a
raw 401 traceback.

GPU + gated weights only (SAM 3 via transformers v5, requirements-gpu). Local
tests are mock-based; run this on the server:

    python scripts/smoke_counter.py --config configs/t4-colab-vqa.yaml
    python scripts/smoke_counter.py --config configs/t0.yaml --backend sam3 \
        --video L01_V001
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aic.config import COUNTER_BACKENDS, load_config  # noqa: E402
from aic.log import setup_logging  # noqa: E402
from aic.models_cache import apply_model_cache_env  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Profile config YAML.")
    parser.add_argument(
        "--backend",
        default=None,
        help=f"Override counter.backend for this run ({list(COUNTER_BACKENDS)}).",
    )
    parser.add_argument(
        "--video", default=None, help="video_id to count (default: the first)."
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    cfg = load_config(args.config)
    apply_model_cache_env(cfg.paths.models_dir)

    counter_cfg = cfg.chronicle.ledger.counter
    if args.backend:
        counter_cfg = counter_cfg.model_copy(update={"backend": args.backend})

    token = cfg.paths.resolve_hf_token()
    if token is None:
        # The SAM family is HF-gated: fail with an actionable message, not a 401.
        raise SystemExit(
            f"counter.backend {counter_cfg.backend!r} weights "
            f"({counter_cfg.model_id}) are HF-gated. Request access on the model "
            f"page, then set the token via paths.hf_token_env "
            f"(${cfg.paths.hf_token_env}) or paths.hf_token."
        )

    from aic.vqa.counter import build_counter, reconcile_objects
    from aic.vqa.counter.stage import _shot_frames

    shots = _shot_frames(cfg, {args.video} if args.video else None)
    if not shots:
        raise SystemExit("no keyframes found; run ingestion first")
    video_id = args.video or next(iter(shots.values()))[0]
    shots = {k: v for k, v in shots.items() if v[0] == video_id}
    print(f"counting {len(shots)} shots of {video_id} with {counter_cfg.backend}")
    print(f"concepts: {counter_cfg.concepts}")

    counter = build_counter(counter_cfg, cfg.paths.models_dir, token)
    started = time.perf_counter()
    total: dict[str, int] = {}
    for shot_key, (_vid, image_paths) in sorted(shots.items()):
        per_frame = counter.detect(image_paths, counter_cfg.concepts)
        objects = reconcile_objects(per_frame, min_count=1)
        for concept, count in objects.items():
            total[concept] = total.get(concept, 0) + count
        print(f"  {shot_key}: {objects or '{}'}")
    elapsed_min = (time.perf_counter() - started) / 60.0
    print(f"\nper-concept totals over {video_id}: {total}")
    print(f"minutes/video: {elapsed_min:.2f} (record in note 18 honesty table)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
