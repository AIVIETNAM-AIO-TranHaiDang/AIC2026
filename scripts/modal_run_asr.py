import modal
import os
import sys
import runpy
from pathlib import Path
import urllib.request
import zipfile

LOCAL_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/opt/aic")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(str(LOCAL_ROOT / "requirements.txt"))
    .pip_install("nvidia-cublas-cu12", "nvidia-cudnn-cu12==9.*")
    # we need ffmpeg for video decoding
    .apt_install("ffmpeg", "libsm6", "libxext6")
    .add_local_dir(LOCAL_ROOT / "src", remote_path=str(REMOTE_ROOT / "src"), copy=True)
    .add_local_dir(LOCAL_ROOT / "configs", remote_path=str(REMOTE_ROOT / "configs"), copy=True)
    .add_local_dir(LOCAL_ROOT / "scripts", remote_path=str(REMOTE_ROOT / "scripts"), copy=True)
    # Upload the downloaded videos.jsonl so the job knows what to transcribe
    .add_local_file(LOCAL_ROOT / "local_temp_manifests" / "videos.jsonl", remote_path=str(REMOTE_ROOT / "data" / "manifests" / "videos.jsonl"))
)

app = modal.App("aic-asr-pipeline")
source_zips = modal.Volume.from_name("aic-source-zips", create_if_missing=True)
model_cache = modal.Volume.from_name("aic-model-cache", create_if_missing=True)
asr_results = modal.Volume.from_name("aic-asr-results", create_if_missing=True)

ZIPS = [
    "Videos_L21_a.zip", "Videos_L22_a.zip", "Videos_L23_a.zip",
    "Videos_L24_a.zip", "Videos_L25_a.zip", "Videos_L26_a.zip",
    "Videos_L26_b.zip"
]

@app.function(
    image=image,
    volumes={"/mnt/zips": source_zips},
    timeout=4 * 60 * 60, # 4 hours
)
def stage_zips():
    """Download ZIPs to the volume."""
    print("Staging ZIPs...")
    base_url = "https://aic-data.ledo.io.vn/"
    for zip_name in ZIPS:
        dest_path = Path(f"/mnt/zips/{zip_name}")
        if dest_path.exists():
            print(f"{zip_name} already exists. Skipping.")
            continue
        print(f"Downloading {zip_name}...")
        url = base_url + zip_name
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(str(dest_path), 'wb') as out_file:
            import shutil
            shutil.copyfileobj(response, out_file)
        print(f"Downloaded {zip_name}")
    source_zips.commit()
    print("All ZIPs staged!")


@app.function(
    image=image,
    gpu="L4:2", # 2 GPUs for faster-whisper parallelization
    cpu=16.0,
    memory=32768,
    timeout=12 * 60 * 60, # 12 hours timeout to be safe
    volumes={
        "/mnt/zips": source_zips,
        "/opt/aic/data/models": model_cache,
        "/mnt/out": asr_results,
    }
)
def run_asr():
    os.chdir("/opt/aic")
    Path("data/videos").mkdir(parents=True, exist_ok=True)
    Path("data/manifests").mkdir(parents=True, exist_ok=True)
    Path("/mnt/out/manifests").mkdir(parents=True, exist_ok=True)
    
    # Symlink to save the final output
    try:
        os.symlink("/mnt/out/manifests/asr.jsonl", "data/manifests/asr.jsonl")
    except FileExistsError:
        pass

    # Extract all ZIPs to data/videos
    print("Extracting videos from ZIPs...")
    for zip_name in ZIPS:
        zip_path = Path(f"/mnt/zips/{zip_name}")
        if not zip_path.exists():
            raise RuntimeError(f"Missing ZIP: {zip_path}. Did you run stage_zips?")
        print(f"Extracting {zip_name}...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for member in zf.infolist():
                if member.filename.endswith(".mp4"):
                    # Extract directly to data/videos ignoring original folder structure if any
                    source = zf.open(member)
                    target_path = Path("data/videos") / Path(member.filename).name
                    if target_path.exists(): continue
                    with open(target_path, "wb") as target:
                        import shutil
                        shutil.copyfileobj(source, target)
    
    print("Extraction complete. Running build_chronicle.py for ASR...", flush=True)
    # run with --skip-ocr and --skip-entities
    sys.argv = ["/opt/aic/scripts/build_chronicle.py", "--config", "/opt/aic/configs/t0.yaml", "--skip-ocr", "--skip-entities", "--num-gpus", "2"]
    try:
        runpy.run_path("/opt/aic/scripts/build_chronicle.py", run_name="__main__")
    except SystemExit as e:
        if e.code != 0:
            raise RuntimeError(f"ASR failed with code {e.code}")
    
    asr_results.commit()
    print("ASR completed and committed.", flush=True)

@app.function(timeout=16*60*60)
def master():
    print("Starting ASR pipeline on Modal...")
    stage_zips.remote()
    run_asr.remote()
    print("Done! You can download results using: modal volume get aic-asr-results manifests/asr.jsonl .")
