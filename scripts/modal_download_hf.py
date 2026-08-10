"""Download keyframe packages from Hugging Face to a Modal Volume.

Run on the NEW workspace profile:
    MODAL_PROFILE=new-workspace modal run scripts/modal_download_hf.py
"""

from __future__ import annotations
import os
import modal

HF_HUB_VERSION = "1.27.0"
image = modal.Image.debian_slim(python_version="3.12").pip_install(f"huggingface_hub=={HF_HUB_VERSION}", "hf_transfer==0.1.6")
app = modal.App("aic-keyframes-download")

results = modal.Volume.from_name("aic-keyframe-results", create_if_missing=True)

@app.function(
    image=image,
    cpu=4.0,
    memory=8192,
    timeout=24 * 60 * 60,
    volumes={"/results": results},
    env={"HF_HUB_ENABLE_HF_TRANSFER": "1"},
    secrets=[modal.Secret.from_name("huggingface-aic")]
)
def download_all(repo_id: str) -> None:
    from huggingface_hub import snapshot_download
    import shutil
    from pathlib import Path
    
    print(f"Downloading from {repo_id} to /tmp/hf-cache")
    
    local_dir = Path("/tmp/hf-cache")
    local_dir.mkdir(parents=True, exist_ok=True)
    
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        max_workers=16,
    )
    
    print("Flattening files and moving to /results volume...")
    results_dir = Path("/results")
    
    count = 0
    for root, _, files in os.walk(local_dir):
        for file in files:
            if file.endswith(".tar") or file.endswith(".json") or file.endswith(".jsonl"):
                src = Path(root) / file
                dst = results_dir / file
                shutil.move(str(src), str(dst))
                count += 1
                    
    print(f"Moved {count} files to volume.")
    
    # Commit the volume state
    results.commit()
    
    results_dir_count = len(list(results_dir.iterdir()))
    print(f"Total files now in /results: {results_dir_count}")

@app.local_entrypoint()
def main(repo_id: str = "Neezidow/AIC2026-keyframes") -> None:
    print(f"Starting download from {repo_id} to Modal volume...")
    download_all.remote(repo_id)
    print("Download complete.")
