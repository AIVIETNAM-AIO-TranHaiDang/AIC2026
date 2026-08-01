"""Report per-video modality coverage over the ingested + chronicled corpus.

A CPU-only readiness check that reads the JSONL manifests (no models loaded)
and prints, per video, the fraction of shots carrying ASR / OCR / captions and
the derived semantic / literal documents that feed the two text channels, plus
a concentration summary per text channel (coverage, single-video hub share,
spread of per-video rates).

Run this after ingest + embed + chronicle, *before* labelling a fixture, to
catch the uneven-coverage skew that makes equal-weight Reciprocal Rank Fusion
fragile (docs/note/11 section 5, docs/note/13). By default it only describes;
pass a threshold to also emit warnings and a non-zero exit code so it can gate
a pre-labelling check.

Usage:
    python scripts/check_corpus.py --config configs/t0.yaml
    python scripts/check_corpus.py --config configs/t0.yaml \
        --min-shot-coverage 0.2 --max-hub-share 0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aic.config import load_config  # noqa: E402
from aic.corpus_report import (  # noqa: E402
    build_corpus_report,
    coverage_warnings,
    render_report,
)
from aic.log import setup_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Profile config YAML.")
    parser.add_argument(
        "--min-shot-coverage",
        type=float,
        default=None,
        help=(
            "Advisory floor (0-1): warn when a text channel's corpus-wide "
            "coverage (docs/shots) is below this. Off by default. A channel far "
            "below this contributes mostly noise; ~0.15-0.25 is a reasonable "
            "floor on a mixed news corpus, but the right value is corpus-"
            "specific — this is a heuristic, not a contract."
        ),
    )
    parser.add_argument(
        "--max-hub-share",
        type=float,
        default=None,
        help=(
            "Advisory ceiling (0-1): warn when a single video holds more than "
            "this share of a text channel's documents (a fusion hub risk). Off "
            "by default. On a balanced corpus of N videos, an even share is "
            "~1/N; values well above that flag concentration."
        ),
    )
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()

    setup_logging(args.log_level)
    cfg = load_config(args.config)

    report = build_corpus_report(cfg)
    print(render_report(report))

    warnings = coverage_warnings(
        report,
        min_shot_coverage=args.min_shot_coverage,
        max_hub_share=args.max_hub_share,
    )
    if warnings:
        print("\nwarnings:")
        for message in warnings:
            print(f"  ! {message}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
