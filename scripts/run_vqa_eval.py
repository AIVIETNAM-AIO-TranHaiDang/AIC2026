"""One-command VQA evaluation driver for the GPU server (note 18/19).

Builds the Answer Ledger + factoid index on a VQA-enabled config, then scores a
`kind: qa` fixture on the answer + moment axis (per-archetype breakdown), so a
single command turns the KIS-processed corpus in data/ into a VQA effectiveness
report. It orchestrates the existing, tested entry points rather than
re-implementing them:

    build_chronicle.py   -> resumable ASR/OCR/caption (skipped when done for KIS)
                            + Answer Ledger NER + ledger.jsonl (+ counter if on)
    build_text_indexes.py -> semantic + literal + overlay + FACTOID indexes
    evaluate_fixture.py  -> the phase-3 QA harness (answer_ok / moment_ok /
                            qa_correct + per archetype)

Track A works with no endpoint. To also measure Track B, enable vqa.reader in
the config with a VLM endpoint (see configs/t0-vqa.yaml). Usage:

    python scripts/run_vqa_eval.py                      # defaults below
    python scripts/run_vqa_eval.py --num-gpus 2
    python scripts/run_vqa_eval.py --skip-build         # ledger+index already built
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from aic.config import load_config  # noqa: E402


def _run(step: str, cmd: list[str]) -> None:
    """Run one pipeline step, failing loudly (never swallow a build error)."""
    print(f"\n=== {step} ===\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/t0-vqa.yaml",
        help="VQA-enabled profile (ledger.enabled + vqa.enabled).",
    )
    parser.add_argument(
        "--fixture",
        default="data/fixture.qa.yaml",
        help="QA fixture (kind: qa cases) to score.",
    )
    parser.add_argument(
        "--report",
        default="data/reports/vqa.json",
        help="Where to save the report; the QA metrics also save to <report>.qa.json.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="GPUs for the resumable offline stages (entities/counter fan out).",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip the ledger + index build (reuse existing ledger.jsonl and "
        "the factoid index); go straight to the evaluation.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    # Fail early with a clear message if the config is not actually VQA-enabled.
    cfg = load_config(args.config)
    if not (cfg.vqa.enabled and cfg.chronicle.ledger.enabled):
        raise SystemExit(
            f"{args.config}: run_vqa_eval needs vqa.enabled AND "
            "chronicle.ledger.enabled (use configs/t0-vqa.yaml or a copy)."
        )
    if not Path(args.fixture).is_file():
        raise SystemExit(f"fixture not found: {args.fixture}")
    py = sys.executable

    if not args.skip_build:
        _run(
            "Answer Ledger (NER + ledger.jsonl, resumable)",
            [py, "scripts/build_chronicle.py", "--config", args.config,
             "--num-gpus", str(args.num_gpus)],
        )
        _run(
            "Text indexes (incl. the Track A factoid index)",
            [py, "scripts/build_text_indexes.py", "--config", args.config],
        )
    else:
        print("--skip-build: reusing the existing ledger.jsonl + factoid index")

    _run(
        "VQA evaluation (answer + moment + qa_correct, per archetype)",
        [py, "scripts/evaluate_fixture.py", "--use-cortex",
         "--config", args.config, "--fixture", args.fixture,
         "--report", args.report, "--log-level", args.log_level],
    )
    print(
        f"\nDone. QA metrics printed above; JSON at "
        f"{Path(args.report).with_suffix('.qa.json')}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
