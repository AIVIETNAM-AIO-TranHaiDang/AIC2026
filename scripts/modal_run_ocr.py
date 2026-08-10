import modal
import os
import sys
import runpy
from pathlib import Path

LOCAL_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/opt/aic")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(str(LOCAL_ROOT / "requirements.txt"))
    .add_local_dir(LOCAL_ROOT / "src", remote_path=str(REMOTE_ROOT / "src"), copy=True)
    .add_local_dir(LOCAL_ROOT / "configs", remote_path=str(REMOTE_ROOT / "configs"), copy=True)
    .add_local_dir(LOCAL_ROOT / "scripts", remote_path=str(REMOTE_ROOT / "scripts"), copy=True)
)

app = modal.App("aic-ocr-pipeline")
model_cache = modal.Volume.from_name("aic-model-cache", create_if_missing=True)
embed_results = modal.Volume.from_name("aic-embeddings", create_if_missing=False)
frames_volume = modal.Volume.from_name("aic-frames", create_if_missing=False)

@app.function(
    image=image,
    gpu="L4:4", # 4 L4 GPUs for faster OCR processing
    cpu=16.0,
    memory=32768,
    timeout=8 * 60 * 60, # 8 hours
    max_containers=1,
    volumes={
        "/opt/aic/data/models": model_cache,
        "/mnt/out": embed_results,
        "/mnt/frames": frames_volume,
    }
)
def run_ocr() -> None:
    os.chdir("/opt/aic")
    
    # We output to /mnt/out/manifests
    Path("/mnt/out/manifests").mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(parents=True, exist_ok=True)
    
    # Symlink manifests from volume
    try:
        os.symlink("/mnt/out/manifests", "data/manifests")
    except FileExistsError:
        pass
    
    # Symlink keyframes from volume
    # aic-frames has the keyframe folders at its root (e.g., L21_V001/...)
    try:
        os.symlink("/mnt/frames/keyframes", "data/keyframes")
    except FileExistsError:
        pass
        
    print("Running build_chronicle.py for OCR...", flush=True)
    # run with --skip-asr and --skip-entities
    sys.argv = ["/opt/aic/scripts/build_chronicle.py", "--config", "/opt/aic/configs/t0.yaml", "--skip-asr", "--skip-entities", "--num-gpus", "4"]
    
    try:
        runpy.run_path("/opt/aic/scripts/build_chronicle.py", run_name="__main__")
    except SystemExit as e:
        if e.code != 0:
            raise RuntimeError(f"OCR failed with code {e.code}")
    
    embed_results.commit()
    print("OCR completed and volume committed.", flush=True)

@app.local_entrypoint()
def main():
    print("Starting OCR pipeline...")
    run_ocr.remote()
    print("Done OCR!")
