"""Transfer aic-keyframe-results volume from old workspace to new workspace.

Run on the OLD workspace profile:
    MODAL_PROFILE=tranhaidang2005ct modal run scripts/transfer_volume.py

This function mounts the old volume (read-only), then uses the Modal CLI
to upload each file to the new workspace's volume.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import modal

app = modal.App("aic-volume-transfer")

old_volume = modal.Volume.from_name(
    "aic-keyframe-results", create_if_missing=False
).with_mount_options(read_only=True)

VOLUME_MOUNT = Path("/old-keyframes")


@app.function(
    image=modal.Image.debian_slim(python_version="3.12"),
    volumes={str(VOLUME_MOUNT): old_volume},
    cpu=4.0,
    memory=8192,
    timeout=60 * 60,
    max_containers=1,
)
def transfer() -> dict[str, object]:
    """Read all files from old volume and return manifest."""
    files = sorted(VOLUME_MOUNT.iterdir())
    total = len(files)
    total_bytes = sum(f.stat().st_size for f in files)

    print(f"Found {total} files, total {total_bytes / 1024**2:.1f} MiB")

    manifest = []
    for f in files:
        manifest.append({
            "name": f.name,
            "size_bytes": f.stat().st_size,
        })

    return {
        "total_files": total,
        "total_bytes": total_bytes,
        "total_mib": round(total_bytes / 1024**2, 1),
        "manifest": manifest,
    }


@app.local_entrypoint()
def main() -> None:
    """Download from old workspace, upload to new workspace."""
    result = transfer.remote()
    total = result["total_files"]
    print(f"\n=== {total} files, {result['total_mib']} MiB total ===\n")

    tmp_dir = Path(tempfile.mkdtemp(prefix="aic-transfer-"))
    old_profile = os.environ.get("MODAL_PROFILE", "tranhaidang2005ct")
    new_profile = "new-workspace"
    volume_name = "aic-keyframe-results"

    try:
        transferred = 0
        errors = []
        started = time.time()

        for item in result["manifest"]:
            fname = item["name"]
            transferred += 1
            local_path = tmp_dir / fname
            size_mib = item["size_bytes"] / 1024**2

            print(f"[{transferred}/{total}] {fname} ({size_mib:.1f} MiB)")

            try:
                # Download from old workspace
                subprocess.run(
                    ["modal", "volume", "get", volume_name, fname, str(local_path)],
                    env={**os.environ, "MODAL_PROFILE": old_profile},
                    check=True,
                    capture_output=True,
                )

                # Upload to new workspace
                subprocess.run(
                    ["modal", "volume", "put", volume_name, str(local_path), fname],
                    env={**os.environ, "MODAL_PROFILE": new_profile},
                    check=True,
                    capture_output=True,
                )

                # Clean up
                local_path.unlink(missing_ok=True)

            except subprocess.CalledProcessError as e:
                errors.append(f"{fname}: {e.stderr.decode()[:200]}")
                print(f"  ERROR: {errors[-1]}")

        elapsed = time.time() - started
        print(f"\n=== Done in {elapsed/60:.1f} min ===")
        print(f"Transferred: {transferred - len(errors)}/{total}")
        if errors:
            print(f"Errors: {len(errors)}")
            for e in errors:
                print(f"  - {e}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
