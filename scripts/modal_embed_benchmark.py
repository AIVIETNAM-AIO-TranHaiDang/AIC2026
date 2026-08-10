"""Benchmark the source SigLIP2 encoder on several Modal GPUs.

Small L4 correctness smoke test:
    modal run scripts/modal_embed_benchmark.py::smoke

Sequential cost-bounded comparison (L4, A10, then L40S):
    modal run scripts/modal_embed_benchmark.py::run --sample-size 2048

The runner reads verified keyframe TARs from ``aic-keyframe-results`` and
writes JSON reports to ``aic-embedding-benchmarks``. It never edits the input
Volume or the source embedding/index format.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import modal

LOCAL_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/opt/aic")
INPUT_ROOT = Path("/keyframe-results")
MODEL_ROOT = Path("/models")
REPORT_ROOT = Path("/benchmark-results")

MODEL_ID = "google/siglip2-so400m-patch16-384"
FULL_CORPUS_SIZE = 171_059
CPU_CORES = 4.0
MEMORY_MIB = 16_384
MEMORY_GIB = MEMORY_MIB / 1024
TIMEOUT_SECONDS = 25 * 60

# https://modal.com/pricing, checked 2026-08-08. Reports retain the date and
# rates so future readers never mistake an old cost estimate for live billing.
PRICING_CHECKED_DATE = "2026-08-08"
GPU_USD_PER_SECOND = {
    "L4": 0.000222,
    "A10": 0.000306,
    "L40S": 0.000542,
    "A100": 0.000694,
}
BATCH_SIZES = {
    "L4": (8, 16, 32),
    "A10": (8, 16, 32),
    "L40S": (16, 32, 64),
    "A100": (32, 64, 128),
}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy==2.5.1",
        "Pillow==12.3.0",
        "pydantic==2.13.4",
        "PyYAML==6.0.3",
        "safetensors==0.8.0",
        "torch==2.13.0",
        "torchvision==0.28.0",
        "transformers==4.57.6",
    )
    .add_local_dir(LOCAL_ROOT / "src", remote_path=str(REMOTE_ROOT / "src"), copy=True)
)

app = modal.App("aic-siglip2-benchmark")
keyframe_results = modal.Volume.from_name(
    "aic-keyframe-results", create_if_missing=False
).with_mount_options(read_only=True)
model_cache = modal.Volume.from_name("aic-model-cache", create_if_missing=True)
benchmark_results = modal.Volume.from_name(
    "aic-embedding-benchmarks", create_if_missing=True
)

FUNCTION_OPTIONS = {
    "image": image,
    "cpu": CPU_CORES,
    "memory": MEMORY_MIB,
    "timeout": TIMEOUT_SECONDS,
    "max_containers": 1,
    "volumes": {
        str(INPUT_ROOT): keyframe_results,
        str(MODEL_ROOT): model_cache,
        str(REPORT_ROOT): benchmark_results,
    },
}


def _imports() -> tuple[object, object, object, object, object]:
    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    from aic.embed.benchmark import (  # noqa: PLC0415
        estimated_compute_cost,
        extract_keyframe_sample,
        vector_health,
    )
    from aic.embed.encoders import Siglip2Encoder  # noqa: PLC0415

    return np, torch, Image, Siglip2Encoder, (
        estimated_compute_cost,
        extract_keyframe_sample,
        vector_health,
    )


def _encode_sample(encoder: object, images: list[object], batch_size: int) -> object:
    import numpy as np

    chunks = [
        encoder.encode_images(images[start : start + batch_size])
        for start in range(0, len(images), batch_size)
    ]
    return np.concatenate(chunks, axis=0)


def _run_end_to_end(
    encoder: object,
    sample: list[object],
    batch_size: int,
    *,
    np: object,
    torch: object,
    Image: object,
    vector_health: object,
) -> dict[str, object]:
    from aic.embed.shard_io import save_shard
    from aic.manifest import ManifestWriter

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    embedded = 0
    with tempfile.TemporaryDirectory(prefix="aic-e2e-") as temporary:
        output = Path(temporary)
        with ManifestWriter(output / "manifest.jsonl", "keyframe_id") as manifest:
            for shard_index, start in enumerate(range(0, len(sample), batch_size)):
                records = sample[start : start + batch_size]
                images = []
                for record in records:
                    with Image.open(record.local_path) as image_file:
                        images.append(np.array(image_file.convert("RGB")))
                vectors = encoder.encode_images(images)
                health = vector_health(vectors, len(records))
                shard = save_shard(
                    output,
                    f"shard-{shard_index:05d}",
                    vectors,
                    "safetensors",
                    None,
                )
                for row, record in enumerate(records):
                    manifest.append(
                        {
                            "keyframe_id": record.keyframe_id,
                            "shard": shard,
                            "row": row,
                            "model_id": MODEL_ID,
                            "dim": health["dim"],
                        }
                    )
                embedded += len(records)
        output_bytes = sum(path.stat().st_size for path in output.iterdir())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "status": "ok",
        "batch_size": batch_size,
        "images": embedded,
        "elapsed_seconds": round(elapsed, 6),
        "images_per_second": embedded / elapsed,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "output_bytes": output_bytes,
        "includes": ["jpeg_decode", "siglip2_encode", "safetensors", "manifest"],
    }


def _benchmark(gpu_label: str, sample_size: int, smoke: bool) -> dict[str, object]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    np, torch, Image, Siglip2Encoder, helpers = _imports()
    estimated_compute_cost, extract_keyframe_sample, vector_health = helpers
    batches = (8,) if smoke else BATCH_SIZES[gpu_label]
    total_started = time.perf_counter()
    benchmark_results.reload()

    with tempfile.TemporaryDirectory(prefix="aic-siglip2-sample-") as temporary:
        sample_started = time.perf_counter()
        sample = extract_keyframe_sample(
            INPUT_ROOT, Path(temporary), sample_size
        )
        sample_seconds = time.perf_counter() - sample_started
        packages = sorted({record.source_package for record in sample})

        decode_started = time.perf_counter()
        decoded = []
        for record in sample:
            with Image.open(record.local_path) as image_file:
                decoded.append(np.array(image_file.convert("RGB")))
        decode_seconds = time.perf_counter() - decode_started

        encoder = Siglip2Encoder(
            model_id=MODEL_ID,
            device="cuda",
            batch_size=max(batches),
            models_dir=MODEL_ROOT,
        )
        model_started = time.perf_counter()
        warmup_count = min(8, len(decoded))
        warmup = encoder.encode_images(decoded[:warmup_count])
        torch.cuda.synchronize()
        model_load_seconds = time.perf_counter() - model_started
        model_cache.commit()
        warmup_health = vector_health(warmup, warmup_count)

        trials: list[dict[str, object]] = []
        reference = None
        for batch_size in batches:
            try:
                encoder.encode_images(decoded[: min(batch_size, len(decoded))])
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                vectors = _encode_sample(encoder, decoded, batch_size)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                health = vector_health(vectors, sample_size)
                trial: dict[str, object] = {
                    "status": "ok",
                    "batch_size": batch_size,
                    "images": sample_size,
                    "elapsed_seconds": round(elapsed, 6),
                    "images_per_second": sample_size / elapsed,
                    "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3,
                    **health,
                }
                if reference is None:
                    reference = vectors
                    trial["min_cosine_vs_first_batch"] = 1.0
                else:
                    cosines = np.sum(reference * vectors, axis=1)
                    trial["min_cosine_vs_first_batch"] = float(np.min(cosines))
                trials.append(trial)
                del vectors
            except torch.cuda.OutOfMemoryError as exc:
                trials.append(
                    {
                        "status": "oom",
                        "batch_size": batch_size,
                        "error": str(exc),
                    }
                )
                torch.cuda.empty_cache()

        successful = [trial for trial in trials if trial["status"] == "ok"]
        if not successful:
            raise RuntimeError(f"every batch size failed on {gpu_label}")
        best = max(successful, key=lambda trial: trial["images_per_second"])
        end_to_end = _run_end_to_end(
            encoder,
            sample,
            int(best["batch_size"]),
            np=np,
            torch=torch,
            Image=Image,
            vector_health=vector_health,
        )

    total_elapsed = time.perf_counter() - total_started
    gpu_rate = GPU_USD_PER_SECOND[gpu_label]
    total_cost = estimated_compute_cost(
        total_elapsed,
        gpu_usd_per_second=gpu_rate,
        cpu_cores=CPU_CORES,
        memory_gib=MEMORY_GIB,
    )
    projected_seconds = FULL_CORPUS_SIZE / end_to_end["images_per_second"]
    projected_cost = estimated_compute_cost(
        projected_seconds,
        gpu_usd_per_second=gpu_rate,
        cpu_cores=CPU_CORES,
        memory_gib=MEMORY_GIB,
    )
    report: dict[str, object] = {
        "status": "ok",
        "benchmark": "siglip2-keyframe-embedding",
        "smoke": smoke,
        "created_at": datetime.now(UTC).isoformat(),
        "gpu_requested": gpu_label,
        "gpu_actual": torch.cuda.get_device_name(0),
        "model_id": MODEL_ID,
        "sample_images": sample_size,
        "sample_packages": packages,
        "sample_extract_seconds": sample_seconds,
        "sample_decode_seconds": decode_seconds,
        "model_load_and_warmup_seconds": model_load_seconds,
        "vector_dim": warmup_health["dim"],
        "trials": trials,
        "best_batch_size": best["batch_size"],
        "end_to_end": end_to_end,
        "total_function_seconds": total_elapsed,
        "pricing": {
            "checked_date": PRICING_CHECKED_DATE,
            "gpu_usd_per_second": gpu_rate,
            "cpu_cores": CPU_CORES,
            "memory_gib": MEMORY_GIB,
        },
        "estimated_this_run_usd": total_cost,
        "projected_full_corpus": {
            "images": FULL_CORPUS_SIZE,
            "seconds_excluding_cold_start": projected_seconds,
            "estimated_compute_usd_excluding_cold_start": projected_cost,
        },
    }
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    history_path = REPORT_ROOT / f"siglip2-{gpu_label.lower()}-{timestamp}.json"
    latest_path = REPORT_ROOT / f"siglip2-{gpu_label.lower()}-latest.json"
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
        temporary_report = Path(handle.name)
    try:
        temporary_report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        shutil.copyfile(temporary_report, history_path)
        shutil.copyfile(temporary_report, latest_path)
    finally:
        temporary_report.unlink(missing_ok=True)
    benchmark_results.commit()
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


@app.function(gpu="L4", **FUNCTION_OPTIONS)
def benchmark_l4(sample_size: int = 2048, smoke: bool = False) -> dict[str, object]:
    return _benchmark("L4", sample_size, smoke)


@app.function(gpu="A10", **FUNCTION_OPTIONS)
def benchmark_a10(sample_size: int = 2048) -> dict[str, object]:
    return _benchmark("A10", sample_size, False)


@app.function(gpu="L40S", **FUNCTION_OPTIONS)
def benchmark_l40s(sample_size: int = 2048) -> dict[str, object]:
    return _benchmark("L40S", sample_size, False)


@app.function(gpu="A100-80GB", **FUNCTION_OPTIONS)
def benchmark_a100(sample_size: int = 2048) -> dict[str, object]:
    return _benchmark("A100", sample_size, False)


@app.local_entrypoint(name="smoke")
def smoke(sample_size: int = 64) -> None:
    """Run the cheapest correctness check before the multi-GPU benchmark."""
    report = benchmark_l4.remote(sample_size, True)
    print(json.dumps(report, ensure_ascii=False, indent=2))


@app.local_entrypoint(name="run")
def run(sample_size: int = 2048) -> None:
    """Run each candidate sequentially so credit use stays bounded."""
    reports = []
    for function in (benchmark_l4, benchmark_a10, benchmark_l40s, benchmark_a100):
        reports.append(function.remote(sample_size))
    summary = {
        report["gpu_requested"]: {
            "best_batch_size": report["best_batch_size"],
            "images_per_second": report["end_to_end"]["images_per_second"],
            "estimated_this_run_usd": report["estimated_this_run_usd"],
            "projected_full_corpus": report["projected_full_corpus"],
        }
        for report in reports
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
