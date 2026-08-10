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
)

app = modal.App("aic-server")

embed_results = modal.Volume.from_name("aic-embeddings", create_if_missing=False)
model_cache = modal.Volume.from_name("aic-model-cache", create_if_missing=False)
frames_volume = modal.Volume.from_name("aic-frames", create_if_missing=True)

@app.function(
    image=image,
    gpu="T4",
    cpu=4.0,
    memory=16384,
    min_containers=1,
    max_containers=1,
    timeout=600,
    volumes={
        "/mnt/out": embed_results,
        "/mnt/models": model_cache,
        "/mnt/frames": frames_volume,
    }
)
@modal.asgi_app()
def fastapi_app():
    import sys
    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, HTMLResponse
    import threading
    import subprocess
    
    real_app = None
    is_ready = False
    loading_error = None
    
    def load_heavy_stuff():
        nonlocal real_app, is_ready, loading_error
        try:
            os.chdir("/opt/aic")
            print("Copying 13GB of artifacts to local NVMe (/tmp) to avoid Modal heartbeat timeout...", flush=True)
            Path("/tmp/aic_data").mkdir(parents=True, exist_ok=True)
            
            subprocess.run(["cp", "-a", "/mnt/out/embeddings", "/tmp/aic_data/embeddings"], check=False)
            subprocess.run(["cp", "-a", "/mnt/out/indexes", "/tmp/aic_data/indexes"], check=False)
            
            Path("data").mkdir(parents=True, exist_ok=True)
            Path("data/embeddings").symlink_to("/tmp/aic_data/embeddings")
            Path("data/indexes").symlink_to("/tmp/aic_data/indexes")
            Path("data/manifests").symlink_to("/mnt/out/manifests")
            Path("data/keyframes").symlink_to("/mnt/frames/keyframes")
            Path("data/models").symlink_to("/mnt/models")
            
            from aic.config import load_config
            from aic.models_cache import apply_model_cache_env
            from aic.service.app import create_app
            from aic.service.state import build_service_state
            
            cfg = load_config("configs/t0.yaml")
            apply_model_cache_env(cfg.paths.models_dir)
            cfg.paths.data_root = Path("/opt/aic/data")
            
            print("Building state...", flush=True)
            state = build_service_state(cfg)
            real_app = create_app(state, cfg.service)
            is_ready = True
            print("App is ready!", flush=True)
        except Exception as e:
            import traceback
            loading_error = traceback.format_exc()
            print(f"Error loading: {e}", flush=True)
            
    threading.Thread(target=load_heavy_stuff, daemon=True).start()

    async def asgi_proxy(scope, receive, send):
        if scope["type"] != "http":
            if is_ready and real_app:
                return await real_app(scope, receive, send)
            return

        if loading_error:
            response = HTMLResponse(f"<html><body><h2>Error Loading:</h2><pre>{loading_error}</pre></body></html>")
            return await response(scope, receive, send)
        if not is_ready:
            if scope.get("path", "").startswith("/api/"):
                response = JSONResponse({"error": "Loading data..."}, status_code=503)
            else:
                response = HTMLResponse("<html><head><meta http-equiv='refresh' content='3'></head><body style='background:#111;color:#fff;display:flex;justify-content:center;align-items:center;height:100vh;font-family:sans-serif;'><h2>Đang nạp 13GB Dữ liệu vào RAM (Khoảng 1-2 phút).<br><br>Vui lòng chờ, trang sẽ tự làm mới...</h2></body></html>")
            return await response(scope, receive, send)
            
        return await real_app(scope, receive, send)
        
    return asgi_proxy
