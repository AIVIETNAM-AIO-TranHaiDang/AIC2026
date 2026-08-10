import modal
import os
from pathlib import Path

LOCAL_ROOT = Path(__file__).resolve().parents[1]

app = modal.App("aic-extract")
keyframe_results = modal.Volume.from_name("aic-keyframe-results", create_if_missing=False).with_mount_options(read_only=True)
embed_results = modal.Volume.from_name("aic-embeddings", create_if_missing=False)
frames_volume = modal.Volume.from_name("aic-frames", create_if_missing=True)

@app.function(
    image=modal.Image.debian_slim(python_version="3.12"),
    cpu=4.0,
    memory=8192,
    timeout=2 * 60 * 60,
    volumes={
        "/opt/aic/raw-keyframes": keyframe_results,
        "/mnt/out": embed_results,
        "/mnt/frames": frames_volume,
    }
)
def extract_data() -> None:
    import tarfile
    
    Path("/mnt/out/manifests").mkdir(parents=True, exist_ok=True)
    Path("/mnt/frames/keyframes").mkdir(parents=True, exist_ok=True)
    
    raw_tars = list(Path("/opt/aic/raw-keyframes").glob("*.tar"))
    if not raw_tars:
        raise RuntimeError("No TAR files found in /opt/aic/raw-keyframes")
    
    print("Extracting images and manifests to persistent volumes...", flush=True)
    with open("/mnt/out/manifests/keyframes.jsonl", "w") as out_kf, \
         open("/mnt/out/manifests/shots.jsonl", "w") as out_shots, \
         open("/mnt/out/manifests/videos.jsonl", "w") as out_vids:
         
        for i, tar_path in enumerate(raw_tars):
            if i % 50 == 0:
                print(f"Extracting {i}/{len(raw_tars)}...", flush=True)
            with tarfile.open(tar_path, "r") as tar:
                for member in tar.getmembers():
                    if member.name.startswith("data/keyframes/") and member.name.endswith(".jpg"):
                        # Extract directly into frames_volume
                        member.name = member.name.replace("data/keyframes/", "")
                        tar.extract(member, path="/mnt/frames/keyframes")
                    elif member.name == "data/manifests/keyframes.jsonl":
                        f = tar.extractfile(member)
                        if f: out_kf.write(f.read().decode("utf-8"))
                    elif member.name == "data/manifests/shots.jsonl":
                        f = tar.extractfile(member)
                        if f: out_shots.write(f.read().decode("utf-8"))
                    elif member.name == "data/manifests/videos.jsonl":
                        f = tar.extractfile(member)
                        if f: out_vids.write(f.read().decode("utf-8"))
                            
    embed_results.commit()
    frames_volume.commit()
    print("Extraction complete!", flush=True)
