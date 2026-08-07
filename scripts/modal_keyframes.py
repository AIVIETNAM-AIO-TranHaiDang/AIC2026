"""Run the existing OmniShotCut/keyframe ingest on Modal.

Pilot (one official video):
    modal run --detach scripts/modal_keyframes.py \
      --archive Videos_L21_a --video-id L21_V001

Whole archive (use only after the pilot output is checked):
    modal run --detach scripts/modal_keyframes.py --archive Videos_L21_a

The GPU container downloads the selected official ZIP, calls the unchanged
``scripts/ingest_corpus.py``, validates its outputs, and stores a plain TAR plus
a JSON report in the persistent ``aic-keyframe-results`` Modal Volume.
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
import urllib.request
import zipfile
from pathlib import Path

import modal

LOCAL_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/opt/aic")
OMNISHOTCUT_COMMIT = "3331cd3163f7f17cd6d7c8fc12ffde22894ace01"

image = (
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
model_cache = modal.Volume.from_name("aic-model-cache", create_if_missing=False)
results = modal.Volume.from_name("aic-keyframe-results", create_if_missing=False)


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "AIC2026-keyframes"})
    started = time.monotonic()
    downloaded = 0
    next_log = 256 * 1024 * 1024
    with urllib.request.urlopen(request, timeout=120) as response:
        with destination.open("wb") as handle:
            while chunk := response.read(8 * 1024 * 1024):
                handle.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_log:
                    print(f"downloaded {downloaded / 1024**3:.2f} GiB", flush=True)
                    next_log += 256 * 1024 * 1024
    elapsed = time.monotonic() - started
    print(f"download complete: {downloaded / 1024**3:.2f} GiB in {elapsed:.1f}s")


@app.function(
    image=image,
    gpu="L4",
    cpu=4.0,
    memory=24_576,
    timeout=20 * 60 * 60,
    volumes={"/models": model_cache, "/results": results},
)
def process_archive(archive: str, video_id: str | None = None) -> dict[str, object]:
    # Imports from the copied project source happen inside the cloud container.
    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from aic.modal_keyframes import (  # noqa: PLC0415
        get_archive_spec,
        make_runtime_config,
        output_stem,
        select_zip_members,
        sha256_file,
        validate_ingest_output,
    )

    spec = get_archive_spec(archive)
    artifact_stem = output_stem(archive, video_id)
    work_root = Path(tempfile.mkdtemp(prefix=f"{artifact_stem}-", dir="/tmp"))
    data_root = work_root / "data"
    videos_dir = data_root / "videos"
    videos_dir.mkdir(parents=True)

    zip_path = work_root / f"{archive}.zip"
    print(f"downloading {spec.url}", flush=True)
    _download(spec.url, zip_path)

    with zipfile.ZipFile(zip_path) as archive_file:
        selected = select_zip_members(archive_file.namelist(), spec, video_id)
        for member_name, selected_id in selected:
            destination = videos_dir / f"{selected_id}.mp4"
            print(f"extracting {selected_id}", flush=True)
            with (
                archive_file.open(member_name) as source,
                destination.open("wb") as out,
            ):
                shutil.copyfileobj(source, out, length=8 * 1024 * 1024)
    zip_path.unlink()

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
        # Preserve a checkpoint that finished downloading even if a later video fails.
        model_cache.commit()
    elapsed = time.monotonic() - started

    expected_ids = {selected_id for _, selected_id in selected}
    counts = validate_ingest_output(data_root, expected_ids)

    result_path = Path("/results") / f"{artifact_stem}-keyframes.tar"
    partial_path = result_path.with_suffix(result_path.suffix + ".part")
    with tarfile.open(partial_path, mode="w") as bundle:
        bundle.add(data_root / "keyframes", arcname="data/keyframes")
        bundle.add(data_root / "manifests", arcname="data/manifests")
    os.replace(partial_path, result_path)

    report: dict[str, object] = {
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
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    results.commit()
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


@app.local_entrypoint()
def main(archive: str = "Videos_L21_a", video_id: str = "L21_V001") -> None:
    """Submit one pilot by default; pass an empty video-id for a whole ZIP."""
    selected_video = video_id.strip() or None
    report = process_archive.remote(archive, selected_video)
    print(json.dumps(report, ensure_ascii=False, indent=2))
