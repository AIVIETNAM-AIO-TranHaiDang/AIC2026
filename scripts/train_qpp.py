"""Train the QPP advisor on the labelled fixture.

Runs every text fixture case through the Cortex + dispatcher exactly like
the online search path, extracts the QPP features, labels each case by
whether its target landed in the operator's top-k, fits the logistic
model, reports leave-one-out calibration, and writes the JSON artifact the
service loads via ``qpp.trained_artifact``.

Re-run whenever the fixture grows — 30 cases is the first useful size
(note 11); KIS-V cases are skipped (they bypass the Cortex, so the advisor
never sees them online either).

Usage:
    python scripts/train_qpp.py --config configs/t0.yaml \
        --fixture data/fixture.yaml --out data/qpp.json [--isotonic]
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
    parser.add_argument("--fixture", required=True, help="Fixture YAML path.")
    parser.add_argument("--out", required=True, help="Artifact JSON output path.")
    parser.add_argument(
        "--isotonic",
        action="store_true",
        help="Add isotonic calibration on the leave-one-out probabilities "
        "(only worthwhile once the fixture has dozens of cases).",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    cfg = load_config(args.config)
    apply_model_cache_env(cfg.paths.models_dir)

    import numpy as np

    from aic.cortex.compiler import CortexCompiler
    from aic.cortex.dispatch import SpecDispatcher, filter_negations
    from aic.embed.encoders import build_image_text_encoder
    from aic.escalate.qpp import (
        QppError,
        extract_features,
        save_artifact,
        train_artifact,
    )
    from aic.eval.fixture import load_fixture
    from aic.eval.metrics import rank_of_first_hit
    from aic.retrieval.build import (
        build_channels,
        load_overlay_prior,
        load_shot_meta,
    )
    from aic.retrieval.fusion import ground_candidates, query_text
    from aic.service.bundles import load_bundles
    from aic.textstack.embedder import build_textstack_embedders

    fixture = load_fixture(args.fixture)
    encoder = build_image_text_encoder(cfg.embed, cfg.paths.models_dir)
    dense_embedder, sparse_embedder = build_textstack_embedders(
        cfg.textstack, cfg.paths.models_dir
    )
    channels = build_channels(
        cfg, encoder, dense_embedder, sparse_text_embedder=sparse_embedder
    )
    shot_meta = load_shot_meta(cfg)
    # Same dispatcher construction as the service (state.py): QPP features
    # must be extracted from the ranking production actually serves.
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
    compiler = CortexCompiler(cfg.cortex)
    video_by_shot = {key: meta.video_id for key, meta in shot_meta.items()}
    bundles = load_bundles(cfg)
    evidence = {key: b.evidence_text for key, b in bundles.items()}

    # The advisor promises "hit in the top-k the operator sees".
    target_k = cfg.service.default_top_k

    features, labels, case_ids = [], [], []
    for case in fixture.cases:
        if case.kind == "kis_v":
            print(f"skipping {case.query_id}: kis_v bypasses the Cortex")
            continue
        raw_query = query_text(case)
        spec = compiler.compile(raw_query)
        detail = dispatcher.rank_spec_detailed(spec, raw_query)
        candidates = detail.candidates
        if cfg.cortex.dispatch.negation_filter:
            candidates = filter_negations(candidates, spec.negations, evidence)
        features.append(
            extract_features(
                spec, candidates, detail.rankings, video_by_shot, cfg.qpp, target_k
            )
        )
        results = ground_candidates(candidates, shot_meta, target_k)
        rank = rank_of_first_hit(results, case.truths, cfg.eval.hit_tolerance_ms)
        labels.append(1 if rank is not None else 0)
        case_ids.append(case.query_id)
        print(f"{case.query_id}: rank={rank} label={labels[-1]}")

    if not features:
        print("no usable cases in the fixture; nothing to train")
        return 1
    x = np.vstack(features)
    y = np.asarray(labels)

    # Leave-one-out probabilities: the honest calibration estimate at this
    # fixture size (note 11 recommends logistic + LOO, not a deep model).
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut

    loo_probs = np.zeros(len(y), dtype=np.float64)
    for train_rows, test_rows in LeaveOneOut().split(x):
        if len(np.unique(y[train_rows])) < 2:
            loo_probs[test_rows] = float(y[train_rows].mean())
            continue
        fold = LogisticRegression(max_iter=1000).fit(x[train_rows], y[train_rows])
        loo_probs[test_rows] = fold.predict_proba(x[test_rows])[:, 1]
    brier = float(np.mean((loo_probs - y) ** 2))

    calibration = None
    if args.isotonic:
        from sklearn.isotonic import IsotonicRegression

        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(loo_probs, y)
        calibration = {
            "x": iso.X_thresholds_.tolist(),
            "y": iso.y_thresholds_.tolist(),
        }

    try:
        artifact = train_artifact(x, y, target_k, calibration)
    except QppError as exc:
        print(f"\ncannot train yet: {exc}")
        print(
            "A one-class fixture (every case hit, or every case missed) "
            "gives the logistic model nothing to separate. Grow the fixture "
            "with harder cases — scripts/generate_fixture.py produces ~100 "
            "synthetic ones — and rerun. The service runs fine without the "
            "advisor (qpp.trained_artifact: null)."
        )
        return 1
    save_artifact(artifact, Path(args.out))

    print(f"\ncases: {len(y)} (hits {int(y.sum())}, misses {int(len(y) - y.sum())})")
    print(f"leave-one-out Brier score: {brier:.3f} (0 is perfect, 0.25 is coin)")
    for case_id, prob, label in zip(case_ids, loo_probs, y, strict=True):
        print(f"  {case_id}: P(hit)={prob:.2f} actual={label}")
    print(f"artifact saved to {args.out}")
    print(f"enable it with:\n  qpp:\n    trained_artifact: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
