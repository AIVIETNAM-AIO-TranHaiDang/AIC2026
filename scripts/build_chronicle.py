"""Run the Chronicle extraction jobs and assemble chronicle.jsonl.

Stages run in order (each independently resumable): ASR -> OCR -> captions
(only when chronicle.caption.enabled and the API key env is set) ->
assembly (+ translation when enabled) -> Answer Ledger NER (only when
chronicle.ledger.enabled, then a re-assembly to merge entities). Re-running
after captions arrive upgrades previously degraded rows.

Usage:
    python scripts/build_chronicle.py --config configs/t0.yaml
    python scripts/build_chronicle.py --config configs/t0.yaml --skip-asr
    python scripts/build_chronicle.py --config configs/t0.yaml --redo-asr
    python scripts/build_chronicle.py --config configs/t0.yaml \
        --redo-asr-videos K01_V009,L27_V013
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aic.config import load_config  # noqa: E402
from aic.log import setup_logging  # noqa: E402
from aic.models_cache import apply_model_cache_env  # noqa: E402
from aic.parallel import resolve_stage_gpus, run_local_stage  # noqa: E402
from aic.verify import maybe_repair  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Profile config YAML.")
    parser.add_argument("--skip-asr", action="store_true")
    parser.add_argument("--skip-ocr", action="store_true")
    parser.add_argument(
        "--skip-entities",
        action="store_true",
        help="Skip the Answer Ledger NER stage even when "
        "chronicle.ledger.enabled (it runs after the first assembly and the "
        "chronicle is re-assembled to merge entities).",
    )
    parser.add_argument(
        "--redo-asr",
        action="store_true",
        help="Drop every ASR manifest row before running, so all videos are "
        "re-transcribed with the current settings (use after changing ASR "
        "config, e.g. enabling vad_filter; the stage itself stays resumable).",
    )
    parser.add_argument(
        "--redo-asr-videos",
        default=None,
        help="Comma-separated video ids whose ASR rows are dropped before "
        "running, re-transcribing only them.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Data-parallel workers for the local torch stages (ASR, easyocr "
        "OCR), one per GPU. Default: every visible GPU. The vlm OCR backend "
        "is an external endpoint and always runs single-process here.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    cfg = load_config(args.config)
    apply_model_cache_env(cfg.paths.models_dir)

    if not args.skip_asr and cfg.chronicle.asr.backend == "faster_whisper":
        from aic.cuda_runtime import (
            CudaRuntimeError,
            ensure_faster_whisper_cuda12_runtime,
        )

        try:
            ensure_faster_whisper_cuda12_runtime(cfg.chronicle.asr.device)
        except CudaRuntimeError as exc:
            print(f"ASR runtime error: {exc}", file=sys.stderr)
            return 2

    maybe_repair(cfg)

    from aic.chronicle.caption import OpenAICompatCaptioner
    from aic.chronicle.jobs import (
        assemble_chronicle,
        reset_asr_records,
        run_caption_job,
    )
    from aic.chronicle.translate import build_translator

    models_dir = cfg.paths.models_dir
    stage_failures = 0
    if args.redo_asr or args.redo_asr_videos:
        video_ids = None
        if args.redo_asr_videos:
            video_ids = {
                v.strip() for v in args.redo_asr_videos.split(",") if v.strip()
            }
        dropped = reset_asr_records(cfg, video_ids)
        print(f"asr: dropped {dropped} manifest row(s) for re-transcription")
    if not args.skip_asr:
        stats = run_local_stage("asr", args.config, cfg, args.num_gpus, desc="asr")
        print(
            f"asr: processed={stats.get('processed', 0)} "
            f"skipped={stats.get('skipped', 0)} failed={stats.get('failed', 0)}"
        )
        stage_failures += stats.get("failed", 0)
    if not args.skip_ocr:
        # Only the torch backend (easyocr) is GPU-parallel; the vlm backend is
        # an OpenAI-compatible endpoint and stays single-process here. The
        # resolver turns the CLI default (None = auto-detect) into a concrete
        # worker count before the comparison below.
        ocr_gpus = resolve_stage_gpus(
            args.num_gpus, cfg.chronicle.ocr.backend == "easyocr"
        )
        if ocr_gpus > 1:
            # Fill the model cache from the parent before spawning workers;
            # concurrent first-run downloads race on the same temp.zip.
            from aic.chronicle.ocr import ensure_easyocr_models

            ensure_easyocr_models(cfg.chronicle.ocr, cfg.paths.models_dir)
        stats = run_local_stage("ocr", args.config, cfg, ocr_gpus, desc="ocr")
        print(
            f"ocr: processed={stats.get('processed', 0)} "
            f"skipped={stats.get('skipped', 0)} failed={stats.get('failed', 0)}"
        )
        stage_failures += stats.get("failed", 0)
    if cfg.chronicle.caption.enabled:
        captioner = OpenAICompatCaptioner(cfg.chronicle.caption)
        try:
            stats = run_caption_job(cfg, captioner)
        finally:
            captioner.close()
        print(
            f"captions: processed={stats.processed} skipped={stats.skipped} "
            f"failed={stats.failed}"
        )
        stage_failures += stats.failed

    translator = None
    if cfg.chronicle.translate.enabled:
        translator = build_translator(cfg.chronicle.translate, models_dir)
    stats = assemble_chronicle(cfg, translator)
    print(f"chronicle: {stats.processed} shots assembled")

    # Answer Ledger NER: fills ChronicleRecord.entities from a resumable
    # per-shot stage, then re-assembles to merge it in. Reads the assembled
    # chronicle, so it must run AFTER the first assembly. gliner is a local
    # torch stage (data-parallel across GPUs); llm is an endpoint stage
    # (single-process, scales across its pool). Skipped unless the ledger is
    # enabled AND spoken_entities is a requested field.
    ledger = cfg.chronicle.ledger
    if (
        not args.skip_entities
        and ledger.enabled
        and "spoken_entities" in ledger.fields
    ):
        backend_name = ledger.resolved_entity_backend()
        entity_gpus = resolve_stage_gpus(args.num_gpus, backend_name == "gliner")
        if entity_gpus > 1 and backend_name == "gliner":
            # Pre-download GLiNER weights in the parent so workers don't race.
            from aic.chronicle.entities import GlinerEntityBackend

            GlinerEntityBackend(ledger, models_dir).warmup()
        stats = run_local_stage(
            "entities", args.config, cfg, entity_gpus, desc="entities"
        )
        print(
            f"entities: processed={stats.get('processed', 0)} "
            f"skipped={stats.get('skipped', 0)} failed={stats.get('failed', 0)}"
        )
        stage_failures += stats.get("failed", 0)
        # Re-assemble so ChronicleRecord.entities is persisted in chronicle.jsonl.
        stats = assemble_chronicle(cfg, translator)
        print(f"chronicle: re-assembled {stats.processed} shots with entities")

    if ledger.enabled and ledger.counter.enabled:
        # Detector-counter pass (phase 4): fills people_count / salient_objects
        # per shot from the keyframes, data-parallel across GPUs. Local torch
        # stage, so it fans across the full device count. The gated weights
        # need the resolved HF token; the warmup pre-downloads once in the
        # parent so workers don't race.
        counter_gpus = resolve_stage_gpus(args.num_gpus, parallelizable=True)
        if counter_gpus > 1:
            from aic.vqa.counter.stage import _build_stage_counter

            _build_stage_counter(cfg)
        stats = run_local_stage(
            "ledger-count", args.config, cfg, counter_gpus, desc="ledger-count"
        )
        print(
            f"ledger-count: processed={stats.get('processed', 0)} "
            f"skipped={stats.get('skipped', 0)} failed={stats.get('failed', 0)}"
        )
        stage_failures += stats.get("failed", 0)

    if ledger.enabled:
        # Derive the per-shot factoid ledger from the assembled chronicle
        # (pure, whole-file rebuild). scripts/build_text_indexes.py then
        # embeds it into the Track A factoid index.
        from aic.chronicle.ledger import build_ledger

        rows = build_ledger(cfg)
        print(f"ledger: {rows} rows written")
    return 1 if stage_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
