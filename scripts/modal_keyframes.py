"""Run the existing OmniShotCut/keyframe ingest safely on Modal.

Stage one official ZIP into persistent storage (CPU only):
    modal run --detach scripts/modal_keyframes.py::stage \
      --archive Videos_L21_a

Submit one or more per-video GPU jobs after staging:
    modal run --detach scripts/modal_keyframes.py::run \
      --archive Videos_L21_a --video-ids L21_V001,L21_V002

Each video commits its own TAR and JSON report to ``aic-keyframe-results``.
Completed videos are skipped on later submissions, while an interrupted ZIP
download resumes from the bytes already stored in ``aic-source-zips``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import modal

LOCAL_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/opt/aic")
OMNISHOTCUT_COMMIT = "3331cd3163f7f17cd6d7c8fc12ffde22894ace01"
DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024
DOWNLOAD_LOG_BYTES = 256 * 1024 * 1024

stage_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("PyYAML==6.0.3")
    .add_local_dir(
        LOCAL_ROOT / "src", remote_path=str(REMOTE_ROOT / "src"), copy=True
    )
)

gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git", "libglib2.0-0", "libgl1")
    .pip_install_from_requirements(str(LOCAL_ROOT / "requirements-modal-ingest.txt"))
    .run_commands(
        "python -m pip install --no-deps "
        "git+https://github.com/UVA-Computer-Vision-Lab/OmniShotCut.git@"
        f"{OMNISHOTCUT_COMMIT}"
    )
    .add_local_dir(LOCAL_ROOT / "src", remote_path=str(REMOTE_ROOT / "src"), copy=True)
    .add_local_dir(
        LOCAL_ROOT / "scripts",
        remote_path=str(REMOTE_ROOT / "scripts"),
        copy=True,
    )
    .add_local_dir(
        LOCAL_ROOT / "configs",
        remote_path=str(REMOTE_ROOT / "configs"),
        copy=True,
    )
)

app = modal.App("aic-keyframes")
source_zips = modal.Volume.from_name("aic-source-zips", create_if_missing=False)
model_cache = modal.Volume.from_name("aic-model-cache", create_if_missing=False)
results = modal.Volume.from_name("aic-keyframe-results", create_if_missing=False)


def _download_resumable(url: str, destination: Path) -> int:
    """Download with HTTP Range, preserving bytes from interrupted attempts."""
    offset = destination.stat().st_size if destination.is_file() else 0
    headers = {"User-Agent": "AIC2026-keyframes"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    started = time.monotonic()

    try:
        response = urllib.request.urlopen(request, timeout=120)
    except urllib.error.HTTPError as exc:
        # A complete file whose ready marker was not written can legitimately
        # receive 416.  Its ZIP structure is validated by the caller.
        if exc.code == 416 and offset:
            print("server reports no bytes remain; validating existing ZIP", flush=True)
            return offset
        raise

    with response:
        status = getattr(response, "status", response.getcode())
        append = bool(offset and status == 206)
        if append:
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {offset}-"):
                raise RuntimeError(
                    f"server resumed at an unexpected offset: {content_range!r}"
                )
        elif offset:
            # Servers may ignore Range and answer 200; restart instead of
            # appending a second full ZIP to the partial file.
            print("server ignored Range; restarting ZIP download", flush=True)
            offset = 0

        mode = "ab" if append else "wb"
        downloaded = offset
        next_log = ((downloaded // DOWNLOAD_LOG_BYTES) + 1) * DOWNLOAD_LOG_BYTES
        with destination.open(mode) as handle:
            while chunk := response.read(DOWNLOAD_CHUNK_BYTES):
                handle.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_log:
                    print(f"downloaded {downloaded / 1024**3:.2f} GiB", flush=True)
                    next_log += DOWNLOAD_LOG_BYTES
            handle.flush()
            os.fsync(handle.fileno())

    elapsed = time.monotonic() - started
    print(
        f"download attempt complete: {downloaded / 1024**3:.2f} GiB "
        f"stored in {elapsed:.1f}s",
        flush=True,
    )
    return downloaded


def _validate_zip(zip_path: Path, spec: object) -> int:
    """Check the central directory and exact official video selection."""
    from aic.modal_keyframes import select_zip_members  # noqa: PLC0415

    with zipfile.ZipFile(zip_path) as archive_file:
        selected = select_zip_members(archive_file.namelist(), spec, None)
    return len(selected)


@app.function(
    image=stage_image,
    cpu=1.0,
    memory=2_048,
    timeout=6 * 60 * 60,
    max_containers=1,
    volumes={"/inputs": source_zips},
)
def stage_archive(archive: str) -> dict[str, object]:
    """Persist one official ZIP; partial downloads remain resumable."""
    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from aic.modal_keyframes import (  # noqa: PLC0415
        get_archive_spec,
        sha256_file,
        validate_ready_archive,
    )

    spec = get_archive_spec(archive)
    source_zips.reload()
    try:
        marker = validate_ready_archive(Path("/inputs"), archive)
    except ValueError as exc:
        print(f"stage check: {exc}", flush=True)
    else:
        return {**marker, "status": "already_ready"}

    zip_path = Path("/inputs") / f"{archive}.zip"
    marker_path = Path("/inputs") / f"{archive}.ready.json"
    initial_bytes = zip_path.stat().st_size if zip_path.is_file() else 0

    try:
        try:
            video_count = _validate_zip(zip_path, spec)
            print("existing ZIP is complete; creating its ready marker", flush=True)
        except (OSError, ValueError, zipfile.BadZipFile):
            if initial_bytes:
                print(
                    f"resuming {archive} from {initial_bytes / 1024**3:.2f} GiB",
                    flush=True,
                )
            else:
                print(f"downloading {spec.url}", flush=True)
            _download_resumable(spec.url, zip_path)
            try:
                video_count = _validate_zip(zip_path, spec)
            except (OSError, ValueError, zipfile.BadZipFile):
                if not initial_bytes:
                    raise
                # The remote object changed or supplied an unusable range.
                # Reset only this explicitly selected archive and retry once.
                print("resumed ZIP is invalid; restarting it once", flush=True)
                zip_path.unlink(missing_ok=True)
                source_zips.commit()
                _download_resumable(spec.url, zip_path)
                video_count = _validate_zip(zip_path, spec)

        marker: dict[str, object] = {
            "status": "ready",
            "archive": archive,
            "url": spec.url,
            "videos": video_count,
            "artifact": zip_path.name,
            "artifact_bytes": zip_path.stat().st_size,
            "artifact_sha256": sha256_file(zip_path),
        }
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".json", delete=False
        ) as handle:
            json.dump(marker, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            temporary_marker = Path(handle.name)
        try:
            shutil.copyfile(temporary_marker, marker_path)
        finally:
            temporary_marker.unlink(missing_ok=True)
        source_zips.commit()
        print(json.dumps(marker, ensure_ascii=False, indent=2), flush=True)
        return marker
    finally:
        # Commit useful partial bytes when a network interruption raises.
        source_zips.commit()


@app.function(
    image=gpu_image,
    gpu="L4",
    cpu=4.0,
    memory=24_576,
    timeout=3 * 60 * 60,
    max_containers=2,
    volumes={"/inputs": source_zips, "/models": model_cache, "/results": results},
)
def process_video(archive: str, video_id: str) -> dict[str, object]:
    """Run unchanged source for one video and checkpoint that video only."""
    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from aic.modal_keyframes import (  # noqa: PLC0415
        completed_video_result,
        get_archive_spec,
        make_runtime_config,
        output_stem,
        select_zip_members,
        sha256_file,
        validate_ingest_output,
        validate_ready_archive,
    )

    spec = get_archive_spec(archive)
    artifact_stem = output_stem(archive, video_id)
    source_zips.reload()
    results.reload()

    completed = completed_video_result(Path("/results"), archive, video_id)
    if completed is not None:
        skipped = {**completed, "status": "skipped", "reason": "already_complete"}
        print(json.dumps(skipped, ensure_ascii=False, indent=2), flush=True)
        return skipped

    validate_ready_archive(Path("/inputs"), archive)
    zip_path = Path("/inputs") / f"{archive}.zip"
    work_root = Path(tempfile.mkdtemp(prefix=f"{artifact_stem}-", dir="/tmp"))
    data_root = work_root / "data"
    videos_dir = data_root / "videos"
    videos_dir.mkdir(parents=True)

    with zipfile.ZipFile(zip_path) as archive_file:
        selected = select_zip_members(archive_file.namelist(), spec, video_id)
        member_name, selected_id = selected[0]
        destination = videos_dir / f"{selected_id}.mp4"
        print(f"extracting {selected_id} from staged {archive}.zip", flush=True)
        with archive_file.open(member_name) as source, destination.open("wb") as out:
            shutil.copyfileobj(source, out, length=DOWNLOAD_CHUNK_BYTES)

    runtime_config = work_root / "modal-keyframes.yaml"
    make_runtime_config(
        REMOTE_ROOT / "configs" / "modal-keyframes.yaml", runtime_config
    )
    command = [
        sys.executable,
        str(REMOTE_ROOT / "scripts" / "ingest_corpus.py"),
        "--config",
        str(runtime_config),
        "--num-gpus",
        "1",
    ]
    print("running existing source: " + " ".join(command), flush=True)
    started = time.monotonic()
    try:
        subprocess.run(command, cwd=work_root, check=True)
    finally:
        model_cache.commit()
    elapsed = time.monotonic() - started

    counts = validate_ingest_output(data_root, {video_id})
    result_path = Path("/results") / f"{artifact_stem}-keyframes.tar"
    local_artifact = work_root / f"{artifact_stem}-keyframes.tar.part"
    with tarfile.open(local_artifact, mode="w") as bundle:
        bundle.add(data_root / "keyframes", arcname="data/keyframes")
        bundle.add(data_root / "manifests", arcname="data/manifests")
    shutil.copyfile(local_artifact, result_path)

    report: dict[str, object] = {
        "status": "done",
        "archive": archive,
        "video_id": video_id,
        "gpu": "L4",
        "config_profile": "configs/modal-keyframes.yaml",
        "base_profile": "configs/t0.yaml",
        "omnishotcut_commit": OMNISHOTCUT_COMMIT,
        "elapsed_seconds": round(elapsed, 3),
        "artifact": result_path.name,
        "artifact_bytes": result_path.stat().st_size,
        "artifact_sha256": sha256_file(result_path),
        **counts,
    }
    report_path = Path("/results") / f"{artifact_stem}-report.json"
    local_report = work_root / f"{artifact_stem}-report.json.part"
    local_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copyfile(local_report, report_path)
    results.commit()
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def _local_helpers() -> tuple[object, object]:
    sys.path.insert(0, str(LOCAL_ROOT / "src"))
    from aic.modal_keyframes import get_archive_spec, parse_video_ids  # noqa: PLC0415

    return get_archive_spec, parse_video_ids


@app.local_entrypoint(name="stage")
def stage(archive: str = "Videos_L21_a") -> None:
    """Stage a resumable source ZIP using CPU resources only."""
    get_archive_spec, _ = _local_helpers()
    get_archive_spec(archive)
    report = stage_archive.remote(archive)
    print(json.dumps(report, ensure_ascii=False, indent=2))


@app.local_entrypoint(name="run")
def run(archive: str = "Videos_L21_a", video_ids: str = "") -> None:
    """Submit all archive videos, or a comma-separated subset, then return."""
    _, parse_video_ids = _local_helpers()
    selected = parse_video_ids(archive, video_ids)
    process_video.spawn_map([archive] * len(selected), selected)
    print(
        f"submitted {len(selected)} video job(s) for {archive}; "
        "completed outputs will be skipped"
    )
