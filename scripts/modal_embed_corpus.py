"""Run full corpus embedding on Modal L40S.

Run on the NEW workspace profile:
    MODAL_PROFILE=new-workspace modal run scripts/modal_embed_corpus.py
"""

from __future__ import annotations
import os
import subprocess
from pathlib import Path
import modal

LOCAL_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/opt/aic")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(str(LOCAL_ROOT / "requirements.txt"))
    .add_local_dir(LOCAL_ROOT / "src", remote_path=str(REMOTE_ROOT / "src"), copy=True)
    .add_local_dir(LOCAL_ROOT / "configs", remote_path=str(REMOTE_ROOT / "configs"), copy=True)
    .add_local_dir(LOCAL_ROOT / "scripts", remote_path=str(REMOTE_ROOT / "scripts"), copy=True)
)

app = modal.App("aic-embed-corpus")

# Mounts
keyframe_results = modal.Volume.from_name("aic-keyframe-results", create_if_missing=False).with_mount_options(read_only=True)
model_cache = modal.Volume.from_name("aic-model-cache", create_if_missing=True)
embed_results = modal.Volume.from_name("aic-embeddings", create_if_missing=True)

@app.function(
    image=image,
    gpu="A100-40GB",
    cpu=4.0,
    memory=16384,
    timeout=4 * 60 * 60,
    max_containers=1,
    volumes={
        "/opt/aic/raw-keyframes": keyframe_results,
        "/opt/aic/data/models": model_cache,
        "/mnt/out": embed_results,
    }
)
def run_embed() -> None:
    import tarfile
    import shutil
    import sys
    import runpy
    os.chdir("/opt/aic")
    
    Path("/mnt/out/embeddings").mkdir(parents=True, exist_ok=True)
    Path("/mnt/out/indexes").mkdir(parents=True, exist_ok=True)
    
    Path("data/embeddings").symlink_to("/mnt/out/embeddings")
    Path("data/indexes").symlink_to("/mnt/out/indexes")
    
    print("Preparing data from TARs...", flush=True)
    raw_tars = list(Path("/opt/aic/raw-keyframes").glob("*.tar"))
    if not raw_tars:
        raise RuntimeError("No TAR files found in /opt/aic/raw-keyframes")
    
    # Extract to /opt/aic/data (JPEGs go into /opt/aic/data/keyframes/..., manifest into /opt/aic/data/manifests/)
    # But since /opt/aic/data/keyframes is NOT a volume anymore, we can just extract to disk.
    Path("data/manifests").mkdir(parents=True, exist_ok=True)
    
    # We aggregate the manifests
    with open("data/manifests/keyframes.jsonl", "w") as out_kf:
        for i, tar_path in enumerate(raw_tars):
            if i % 50 == 0:
                print(f"Extracting {i}/{len(raw_tars)}...", flush=True)
            with tarfile.open(tar_path, "r") as tar:
                # Extract JPEGs to data/
                for member in tar.getmembers():
                    if member.name.startswith("data/keyframes/") and member.name.endswith(".jpg"):
                        tar.extract(member, path=".")
                    elif member.name == "data/manifests/keyframes.jsonl":
                        f = tar.extractfile(member)
                        if f:
                            out_kf.write(f.read().decode("utf-8"))
                            
    print("Data preparation complete. Running embed_corpus.py...", flush=True)
    sys.argv = ["scripts/embed_corpus.py", "--config", "configs/t0.yaml"]
    
    try:
        runpy.run_path("scripts/embed_corpus.py", run_name="__main__")
    except SystemExit as e:
        if e.code != 0:
            raise RuntimeError(f"Embedding failed with code {e.code}")
    
    embed_results.commit()
    print("Embedding completed and volume committed.", flush=True)

@app.local_entrypoint()
def main() -> None:
    print("Submitting full embedding job to Modal L40S...")
    run_embed.remote()
    print("Done!")
