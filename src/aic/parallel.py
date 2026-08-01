"""Data-parallel execution of the resumable per-video torch stages.

Four offline stages are local, GPU-bound, and embarrassingly parallel over
videos: shot detection (ingest), keyframe embedding, ASR, and easyocr OCR.
Each already lists its work, skips what a manifest says is done, and appends
one record per artifact. This module fans that work across N GPUs:

- one worker process per GPU, pinned with ``CUDA_VISIBLE_DEVICES`` so each sees
  a single card as ``cuda:0`` (no per-model device plumbing needed);
- a shared work queue of video ids that all workers pull from: each worker
  builds its model once, then takes the next id whenever it finishes one, so a
  fast GPU keeps grabbing work instead of idling while a slow GPU grinds
  through a static shard (the videos are unequal in length, so static
  round-robin left cards idle);
- per-shard manifests (``name.rankK.jsonl``) that the parent concatenates into
  the canonical manifest once every worker is done, so concurrent appends never
  interleave into one file;
- a single aggregate tqdm progress bar over the whole corpus, driven by ticks
  the workers push through a queue.

Because the queue is shared, a crashed worker's *queued* ids are drained by the
survivors; only the one id it was mid-flight on waits for the next resumable
re-run (the manifest records what finished). If the *parent* dies (Ctrl-C, a
session limit) the shard manifests it never merged are folded into the
canonical manifest at the start of the next run, before planning, so finished
work is never re-done.

Endpoint stages (caption, VLM OCR) are intentionally excluded: they scale by
fanning requests across OpenAI-compatible endpoints, not by owning a GPU.

The single-process path (``num_gpus`` resolving to 1, e.g. the CPU dev box) is
still driven through here so it gets the same progress bar without spawning any
subprocess.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue as queue_mod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from aic.config import Config
from aic.manifest import completed_keys, read_manifest

logger = logging.getLogger(__name__)

# Poll interval for the parent's progress drain: short enough to keep the bar
# responsive, long enough not to busy-spin.
_DRAIN_POLL_S = 0.5


def resolve_num_gpus(requested: int | None) -> int:
    """Number of worker processes to run.

    ``requested`` None -> every visible CUDA device (``torch.cuda`` query); an
    explicit value is honoured but capped at the visible device count when that
    count is positive (asking for more cards than exist only slows things). The
    floor is 1 so a CPU box or a no-GPU request still runs single-process. On a
    machine with no CUDA an explicit value is honoured as given, which lets the
    parallel path be exercised on a CPU-only dev box.
    """
    try:
        import torch

        available = torch.cuda.device_count()
    except Exception as exc:  # torch missing or CUDA query failed
        logger.warning("could not query CUDA devices (%s); assuming 0", exc)
        available = 0

    if requested is None:
        return available if available > 0 else 1
    if requested < 1:
        raise ValueError(f"num_gpus must be >= 1, got {requested}")
    if available > 0 and requested > available:
        logger.warning(
            "requested %d GPUs but only %d visible; using %d",
            requested,
            available,
            available,
        )
        return available
    return requested


def resolve_stage_gpus(requested: int | None, parallelizable: bool) -> int:
    """Worker count for a stage that may be pinned single-process.

    ``parallelizable`` False (an endpoint-backed stage, e.g. the vlm OCR
    backend, which scales across its endpoint pool instead of owning GPUs)
    always yields 1. Otherwise ``requested`` resolves exactly like
    :func:`resolve_num_gpus` — in particular the CLI default ``None`` becomes
    a concrete count *here*, so callers can compare or branch on the result
    without ever seeing ``None`` (regression: ``build_chronicle`` compared the
    raw CLI value with ``> 1`` and crashed on the auto-detect default).
    """
    if not parallelizable:
        return 1
    return resolve_num_gpus(requested)


def shard_tag(rank: int) -> str:
    return f"rank{rank}"


def gpu_labels(num_workers: int) -> list[str]:
    """The ``CUDA_VISIBLE_DEVICES`` value each worker rank should pin.

    When the parent process already runs under a mask (an operator gave this
    job GPUs ``1,3`` on a shared server), rank k must map to the k-th entry
    of that mask — writing the plain rank would grab globally-indexed cards
    the operator excluded. Without a parent mask the rank is the device
    index. Entries pass through verbatim, so UUID/MIG identifiers work too.
    """
    parent = os.environ.get("CUDA_VISIBLE_DEVICES")
    if parent is None:
        return [str(rank) for rank in range(num_workers)]
    visible = [part.strip() for part in parent.split(",") if part.strip()]
    if num_workers > len(visible):
        raise ValueError(
            f"{num_workers} workers requested but CUDA_VISIBLE_DEVICES "
            f"exposes only {len(visible)} device(s): {parent!r}"
        )
    return visible[:num_workers]


def merge_shard_manifests(canonical_paths: Sequence[Path]) -> int:
    """Fold every ``name.rank*.jsonl`` sibling into ``name``, then delete it.

    Returns the number of shard files folded. Discovery is by glob rather than
    by the current worker count: an interrupted run leaves its shards behind —
    possibly from a larger ``--num-gpus`` than today's — and every one of them
    must fold back or the next plan re-does work those shards already recorded.
    Idempotent across re-runs: a worker only writes rows for keys not already
    in the canonical manifest, so concatenating adds no duplicates. A worker
    that had nothing to do wrote a shard with no rows, which folds harmlessly.
    """
    folded = 0
    for canonical in canonical_paths:
        pattern = f"{canonical.stem}.rank*{canonical.suffix}"
        for shard_path in sorted(canonical.parent.glob(pattern)):
            if shard_path == canonical:
                continue
            with (
                canonical.open("a", encoding="utf-8") as out,
                shard_path.open("r", encoding="utf-8") as src,
            ):
                for line in src:
                    out.write(line)
            shard_path.unlink()
            folded += 1
    return folded


@dataclass(frozen=True)
class StagePlan:
    """What the parent needs to shard and to size the progress bar.

    ``shard_ids`` are the video ids that still have pending work (the unit of
    sharding). ``total`` and ``unit`` size the bar (videos for ingest/asr,
    keyframes for embed/ocr). ``merge_paths`` are the canonical manifests the
    per-shard files fold back into.
    """

    shard_ids: list[str]
    total: int
    unit: str
    merge_paths: list[Path]


def _plan_ingest(cfg: Config) -> StagePlan:
    from aic.ingest.pipeline import (
        KEYFRAMES_MANIFEST,
        SHOTS_MANIFEST,
        VIDEOS_MANIFEST,
        discover_videos,
    )

    md = cfg.paths.manifests_dir
    done = completed_keys(md / VIDEOS_MANIFEST, "video_id")
    pending = [
        path.stem
        for path in discover_videos(cfg.paths.videos_dir, cfg.ingest.video_extensions)
        if path.stem not in done
    ]
    return StagePlan(
        pending,
        len(pending),
        "video",
        [md / VIDEOS_MANIFEST, md / SHOTS_MANIFEST, md / KEYFRAMES_MANIFEST],
    )


def _plan_embed(cfg: Config) -> StagePlan:
    from aic.embed.job import EMBEDDINGS_MANIFEST, embeddings_dir_for
    from aic.ingest.pipeline import KEYFRAMES_MANIFEST

    out_dir = embeddings_dir_for(cfg, cfg.embed.model_id)
    done = completed_keys(out_dir / EMBEDDINGS_MANIFEST, "keyframe_id")
    pending = [
        r
        for r in read_manifest(cfg.paths.manifests_dir / KEYFRAMES_MANIFEST)
        if r["keyframe_id"] not in done
    ]
    video_ids = list(dict.fromkeys(r["video_id"] for r in pending))
    return StagePlan(video_ids, len(pending), "img", [out_dir / EMBEDDINGS_MANIFEST])


def _plan_asr(cfg: Config) -> StagePlan:
    from aic.chronicle.jobs import ASR_MANIFEST, _done_videos

    md = cfg.paths.manifests_dir
    done = completed_keys(md / ASR_MANIFEST, "video_id")
    pending = [v["video_id"] for v in _done_videos(cfg) if v["video_id"] not in done]
    return StagePlan(pending, len(pending), "video", [md / ASR_MANIFEST])


def _plan_ocr(cfg: Config) -> StagePlan:
    from aic.chronicle.jobs import OCR_MANIFEST
    from aic.ingest.pipeline import KEYFRAMES_MANIFEST

    md = cfg.paths.manifests_dir
    done = completed_keys(md / OCR_MANIFEST, "keyframe_id")
    pending = [
        r
        for r in read_manifest(md / KEYFRAMES_MANIFEST)
        if r["keyframe_id"] not in done
    ]
    video_ids = list(dict.fromkeys(r["video_id"] for r in pending))
    return StagePlan(video_ids, len(pending), "img", [md / OCR_MANIFEST])


def _plan_entities(cfg: Config) -> StagePlan:
    from aic.chronicle.entities import ENTITIES_MANIFEST, completed_entity_keys
    from aic.chronicle.jobs import CHRONICLE_MANIFEST

    md = cfg.paths.manifests_dir
    done = completed_entity_keys(cfg)
    pending = [
        r
        for r in read_manifest(md / CHRONICLE_MANIFEST)
        if r.get("shot_key") not in done
    ]
    video_ids = list(dict.fromkeys(r["video_id"] for r in pending))
    return StagePlan(video_ids, len(pending), "shot", [md / ENTITIES_MANIFEST])


def _plan_ledger_count(cfg: Config) -> StagePlan:
    from aic.vqa.counter.stage import (
        COUNT_MANIFEST,
        completed_count_keys,
        shot_video_map,
    )

    md = cfg.paths.manifests_dir
    done = completed_count_keys(cfg)
    pending = {sk: vid for sk, vid in shot_video_map(cfg).items() if sk not in done}
    video_ids = list(dict.fromkeys(pending.values()))
    return StagePlan(video_ids, len(pending), "shot", [md / COUNT_MANIFEST])


def _run_ledger_count(cfg, video_filter, tag, on_item, limit):
    from aic.vqa.counter.stage import _build_stage_counter, run_counter_job

    counter = _build_stage_counter(cfg)
    s = run_counter_job(
        cfg, counter, video_filter=video_filter, shard_tag=tag, on_item=on_item
    )
    return {"processed": s.processed, "skipped": s.skipped, "failed": s.failed}


def _run_ingest(cfg, video_filter, tag, on_item, limit):
    from aic.ingest.pipeline import run_ingest
    from aic.ingest.shots import build_shot_detector

    detector = build_shot_detector(cfg.ingest.shots)
    s = run_ingest(
        cfg,
        detector,
        limit=limit,
        video_filter=video_filter,
        shard_tag=tag,
        on_item=on_item,
    )
    return {
        "processed": s.processed,
        "skipped": s.skipped,
        "failed": s.failed,
        "keyframes": s.keyframes_written,
    }


def _run_embed(cfg, video_filter, tag, on_item, limit):
    from aic.embed.encoders import build_image_text_encoder
    from aic.embed.job import run_embed_keyframes

    encoder = build_image_text_encoder(cfg.embed, cfg.paths.models_dir)
    s = run_embed_keyframes(
        cfg,
        encoder,
        limit=limit,
        video_filter=video_filter,
        shard_tag=tag,
        on_item=on_item,
    )
    return {"embedded": s.embedded, "skipped": s.skipped, "failed": s.failed}


def _run_asr(cfg, video_filter, tag, on_item, limit):
    from aic.chronicle.asr import build_asr_backend
    from aic.chronicle.jobs import run_asr_job

    backend = build_asr_backend(cfg.chronicle.asr, cfg.paths.models_dir)
    s = run_asr_job(
        cfg, backend, video_filter=video_filter, shard_tag=tag, on_item=on_item
    )
    return {"processed": s.processed, "skipped": s.skipped, "failed": s.failed}


def _run_ocr(cfg, video_filter, tag, on_item, limit):
    from aic.chronicle.jobs import run_ocr_job
    from aic.chronicle.ocr import build_ocr_backend

    backend = build_ocr_backend(cfg.chronicle.ocr, cfg.paths.models_dir)
    s = run_ocr_job(
        cfg, backend, video_filter=video_filter, shard_tag=tag, on_item=on_item
    )
    return {"processed": s.processed, "skipped": s.skipped, "failed": s.failed}


def _run_entities(cfg, video_filter, tag, on_item, limit):
    from aic.chronicle.entities import build_entity_backend, run_entity_job

    backend = build_entity_backend(cfg.chronicle.ledger, cfg.paths.models_dir)
    s = run_entity_job(
        cfg, backend, video_filter=video_filter, shard_tag=tag, on_item=on_item
    )
    return {"processed": s.processed, "skipped": s.skipped, "failed": s.failed}


# Stage registry: name -> (planner, runner). Kept module-level so a spawned
# worker can import and dispatch by name (only picklable strings cross the
# process boundary, never a closure).
_PLANNERS: dict[str, Callable[[Config], StagePlan]] = {
    "ingest": _plan_ingest,
    "embed": _plan_embed,
    "asr": _plan_asr,
    "ocr": _plan_ocr,
    "entities": _plan_entities,
    "ledger-count": _plan_ledger_count,
}
_RUNNERS: dict[str, Callable[..., dict[str, int]]] = {
    "ingest": _run_ingest,
    "embed": _run_embed,
    "asr": _run_asr,
    "ocr": _run_ocr,
    "entities": _run_entities,
    "ledger-count": _run_ledger_count,
}


class StageWorker(Protocol):
    """A resident model plus its open shard writer, driven one video at a time.

    A worker builds its model once, then ``process(video_id)`` is called for
    each id pulled from the shared queue; ``close()`` flushes the shard manifest
    and returns the stage's aggregate stats. Each stage's ``*Processor`` class
    implements this; the factories below build one (model included) per worker.
    """

    def process(self, video_id: str) -> None: ...

    def close(self) -> dict[str, int]: ...


def _make_ingest_worker(cfg, tag, on_item) -> StageWorker:
    from aic.ingest.pipeline import IngestProcessor
    from aic.ingest.shots import build_shot_detector

    return IngestProcessor(cfg, build_shot_detector(cfg.ingest.shots), tag, on_item)


def _make_embed_worker(cfg, tag, on_item) -> StageWorker:
    from aic.embed.encoders import build_image_text_encoder
    from aic.embed.job import EmbedProcessor

    encoder = build_image_text_encoder(cfg.embed, cfg.paths.models_dir)
    return EmbedProcessor(cfg, encoder, tag, on_item)


def _make_asr_worker(cfg, tag, on_item) -> StageWorker:
    from aic.chronicle.asr import build_asr_backend
    from aic.chronicle.jobs import AsrProcessor

    backend = build_asr_backend(cfg.chronicle.asr, cfg.paths.models_dir)
    return AsrProcessor(cfg, backend, tag, on_item)


def _make_ocr_worker(cfg, tag, on_item) -> StageWorker:
    from aic.chronicle.jobs import OcrProcessor
    from aic.chronicle.ocr import build_ocr_backend

    backend = build_ocr_backend(cfg.chronicle.ocr, cfg.paths.models_dir)
    return OcrProcessor(cfg, backend, tag, on_item)


def _make_entities_worker(cfg, tag, on_item) -> StageWorker:
    from aic.chronicle.entities import EntityProcessor, build_entity_backend

    backend = build_entity_backend(cfg.chronicle.ledger, cfg.paths.models_dir)
    return EntityProcessor(cfg, backend, tag, on_item)


def _make_ledger_count_worker(cfg, tag, on_item) -> StageWorker:
    from aic.vqa.counter.stage import make_counter_worker

    return make_counter_worker(cfg, tag, on_item)


# Multi-GPU work-stealing: name -> factory that builds a resident-model worker.
# Kept module-level so a spawned worker can import and dispatch by name (only
# picklable strings cross the process boundary, never a closure).
_WORKER_FACTORIES: dict[str, Callable[..., StageWorker]] = {
    "ingest": _make_ingest_worker,
    "embed": _make_embed_worker,
    "asr": _make_asr_worker,
    "ocr": _make_ocr_worker,
    "entities": _make_entities_worker,
    "ledger-count": _make_ledger_count_worker,
}


def _sum_stats(stats_list: Sequence[dict[str, int]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for stats in stats_list:
        for key, value in stats.items():
            totals[key] = totals.get(key, 0) + value
    return totals


def _worker_entry(stage, config_path, rank, gpu_label, work_queue, tag, out_queue):
    """Child-process entry point: pin one GPU, then steal videos from the queue.

    ``CUDA_VISIBLE_DEVICES`` is set to ``gpu_label`` (rank remapped through
    the parent's own mask by :func:`gpu_labels`) before any torch import so
    the model lands on the assigned card. The worker builds its model once,
    then pulls video ids from the shared ``work_queue`` until it draws the
    ``None`` sentinel, processing each. Progress ticks and the final stats
    (or an error) go back to the parent through ``out_queue``.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_label
    worker: StageWorker | None = None
    try:
        from aic.config import load_config
        from aic.models_cache import apply_model_cache_env

        cfg = load_config(config_path)
        apply_model_cache_env(cfg.paths.models_dir)

        def tick(n: int = 1) -> None:
            out_queue.put(int(n))

        worker = _WORKER_FACTORIES[stage](cfg, tag, tick)
        while True:
            video_id = work_queue.get()
            if video_id is None:
                break
            worker.process(video_id)
        stats = worker.close()
        worker = None
        out_queue.put(("done", rank, stats))
    except Exception as exc:  # noqa: BLE001 - report, never hang the parent
        import traceback

        if worker is not None:
            # Flush whatever finished before the crash so a re-run resumes past it.
            try:
                worker.close()
            except Exception:  # noqa: BLE001 - best effort, already failing
                pass
        out_queue.put(("err", rank, f"{exc}\n{traceback.format_exc()}"))


def _run_parallel(
    stage: str, config_path: str, plan: StagePlan, num_gpus: int, desc: str
) -> dict[str, int]:
    from tqdm.auto import tqdm
    from tqdm.contrib.logging import logging_redirect_tqdm

    ctx = mp.get_context("spawn")
    out_queue: Any = ctx.Queue()
    work_queue: Any = ctx.Queue()
    for video_id in plan.shard_ids:
        work_queue.put(video_id)
    for _ in range(num_gpus):
        work_queue.put(None)  # one sentinel per worker so each stops exactly once

    procs = []
    labels = gpu_labels(num_gpus)
    for rank in range(num_gpus):
        proc = ctx.Process(
            target=_worker_entry,
            args=(
                stage,
                config_path,
                rank,
                labels[rank],
                work_queue,
                shard_tag(rank),
                out_queue,
            ),
        )
        proc.start()
        procs.append(proc)

    stats_list: list[dict[str, int]] = []
    errors: list[str] = []
    terminated = 0
    with (
        logging_redirect_tqdm(),
        tqdm(total=plan.total, desc=desc, unit=plan.unit) as bar,
    ):
        while terminated < num_gpus:
            try:
                msg = out_queue.get(timeout=_DRAIN_POLL_S)
            except queue_mod.Empty:
                # Guard against a hard-killed worker that never sent a sentinel.
                if all(not p.is_alive() for p in procs) and out_queue.empty():
                    break
                continue
            if isinstance(msg, tuple) and msg and msg[0] == "done":
                stats_list.append(msg[2])
                terminated += 1
            elif isinstance(msg, tuple) and msg and msg[0] == "err":
                errors.append(f"gpu {msg[1]}: {msg[2]}")
                terminated += 1
            else:
                bar.update(int(msg))

    for proc in procs:
        proc.join()
    # A crashed worker can leave its unclaimed ids (and sentinel) in the queue;
    # drop them without blocking the parent's exit on the feeder thread.
    work_queue.cancel_join_thread()

    merge_shard_manifests(plan.merge_paths)

    dead = [p.exitcode for p in procs if p.exitcode not in (0, None)]
    if errors or dead:
        detail = "; ".join(errors) or f"worker exit codes {dead}"
        count = len(errors) or len(dead)
        raise RuntimeError(f"{stage}: {count} worker(s) failed: {detail}")
    return _sum_stats(stats_list)


def run_local_stage(
    stage: str,
    config_path: str,
    cfg: Config,
    num_gpus: int | None,
    desc: str,
    limit: int | None = None,
) -> dict[str, int]:
    """Run a local torch stage, single- or multi-process, with one progress bar.

    ``num_gpus`` None auto-detects; the run stays single-process (no subprocess
    overhead) whenever it resolves to 1 or there is at most one shard of work.
    ``limit`` is a single-process-only pilot cap; combined with multiple GPUs
    it is ignored with a warning, since sharding a capped run is ambiguous.
    Returns the aggregate stats dict for the caller to print.
    """
    if stage not in _PLANNERS:
        known = sorted(_PLANNERS)
        raise ValueError(f"unknown stage {stage!r}; expected one of {known}")

    plan = _PLANNERS[stage](cfg)
    # An interrupted earlier run (parent killed before its merge) leaves
    # ``name.rankK.jsonl`` shards the planners cannot see — fold them into the
    # canonical manifests first and re-plan, or their finished work is re-done.
    folded = merge_shard_manifests(plan.merge_paths)
    if folded:
        logger.info(
            "%s: folded %d leftover shard manifest(s) from an interrupted run",
            desc,
            folded,
        )
        plan = _PLANNERS[stage](cfg)
    resolved = resolve_num_gpus(num_gpus)
    workers = min(resolved, len(plan.shard_ids)) if plan.shard_ids else 1

    if workers > 1 and limit is not None:
        logger.warning("--limit is ignored when running on multiple GPUs")
        limit = None

    if workers <= 1:
        from tqdm.auto import tqdm
        from tqdm.contrib.logging import logging_redirect_tqdm

        total = plan.total if limit is None else min(plan.total, limit)
        with (
            logging_redirect_tqdm(),
            tqdm(total=total, desc=desc, unit=plan.unit) as bar,
        ):
            return _RUNNERS[stage](cfg, None, "", lambda n=1: bar.update(int(n)), limit)

    logger.info("%s: data-parallel across %d GPUs", desc, workers)
    return _run_parallel(stage, config_path, plan, workers, desc)
