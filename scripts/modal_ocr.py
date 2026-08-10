import modal
import os
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

app = modal.App("aic-ocr")
keyframe_results = modal.Volume.from_name("aic-keyframe-results", create_if_missing=False).with_mount_options(read_only=True)
model_cache = modal.Volume.from_name("aic-model-cache", create_if_missing=True)
embed_results = modal.Volume.from_name("aic-embeddings", create_if_missing=True)
frames_volume = modal.Volume.from_name("aic-frames", create_if_missing=True)

@app.function(
    image=image,
    gpu="L4:4",
    cpu=16.0,
    memory=32768,
    timeout=4 * 60 * 60,
    max_containers=1,
    volumes={
        "/opt/aic/raw-keyframes": keyframe_results,
        "/opt/aic/data/models": model_cache,
        "/mnt/out": embed_results,
        "/mnt/frames": frames_volume,
    }
)
def run_ocr() -> None:
    import tarfile
    import sys
    import runpy
    import shutil
    os.chdir("/opt/aic")
    
    # We will output manifests to the persistent volume
    Path("/mnt/out/manifests").mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(parents=True, exist_ok=True)
    Path("data/manifests").symlink_to("/mnt/out/manifests")
    
    # Symlink keyframes to the frames volume to save them permanently
    Path("/mnt/frames/keyframes").mkdir(parents=True, exist_ok=True)
    Path("data/keyframes").symlink_to("/mnt/frames/keyframes")
    
    print("Preparing data from TARs...", flush=True)
    raw_tars = list(Path("/opt/aic/raw-keyframes").glob("*.tar"))
    if not raw_tars:
        raise RuntimeError("No TAR files found in /opt/aic/raw-keyframes")
    
    # We aggregate the manifests so that build_chronicle and serve can read them
    # We also extract the JPEGs to data/keyframes
    with open("data/manifests/keyframes.jsonl", "w") as out_kf, \
         open("data/manifests/shots.jsonl", "w") as out_shots, \
         open("data/manifests/videos.jsonl", "w") as out_vids:
        for i, tar_path in enumerate(raw_tars):
            if i % 50 == 0:
                print(f"Extracting {i}/{len(raw_tars)}...", flush=True)
            with tarfile.open(tar_path, "r") as tar:
                for member in tar.getmembers():
                    if member.name.startswith("data/keyframes/") and member.name.endswith(".jpg"):
                        tar.extract(member, path=".")
                    elif member.name == "data/manifests/keyframes.jsonl":
                        f = tar.extractfile(member)
                        if f: out_kf.write(f.read().decode("utf-8"))
                    elif member.name == "data/manifests/shots.jsonl":
                        f = tar.extractfile(member)
                        if f: out_shots.write(f.read().decode("utf-8"))
                    elif member.name == "data/manifests/videos.jsonl":
                        f = tar.extractfile(member)
                        if f: out_vids.write(f.read().decode("utf-8"))
                            
    print("Data preparation complete. Running build_chronicle.py...", flush=True)
    sys.argv = ["scripts/build_chronicle.py", "--config", "configs/t0.yaml", "--skip-asr", "--skip-entities", "--num-gpus", "4"]
    
    try:
        runpy.run_path("scripts/build_chronicle.py", run_name="__main__")
    except SystemExit as e:
        if e.code != 0:
            raise RuntimeError(f"OCR failed with code {e.code}")
    
    embed_results.commit()
    frames_volume.commit()
    print("OCR completed and volumes committed.", flush=True)
