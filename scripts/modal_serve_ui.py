"""Deploy the AIC retrieval UI on Modal.

This entry point reuses the already-built SigLIP2 embeddings/FAISS index and
the keyframe volume.  It deliberately keeps FAISS on CPU for exact parity;
the L4 is used for query-side model inference when the installed Torch build
has CUDA support.

Development (laptop must stay on)::

    MODAL_PROFILE=new-workspace modal serve scripts/modal_serve_ui.py

Persistent deployment (laptop may be turned off)::

    MODAL_PROFILE=new-workspace modal deploy scripts/modal_serve_ui.py
"""

from __future__ import annotations

import os
import shutil
import threading
from pathlib import Path

import modal

LOCAL_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/opt/aic")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(str(LOCAL_ROOT / "requirements.txt"))
    .add_local_dir(LOCAL_ROOT / "src", remote_path=str(REMOTE_ROOT / "src"), copy=True)
    .add_local_dir(
        LOCAL_ROOT / "configs", remote_path=str(REMOTE_ROOT / "configs"), copy=True
    )
)

app = modal.App("aic-ui-siglip2")

embed_results = modal.Volume.from_name("aic-embeddings", create_if_missing=False)
model_cache = modal.Volume.from_name("aic-model-cache", create_if_missing=False)
frames_volume = modal.Volume.from_name("aic-frames", create_if_missing=False)


def _replace_link(path: Path, target: str) -> None:
    """Create a symlink, tolerating a warm container from a prior request."""

    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    path.symlink_to(target)


@app.function(
    image=image,
    gpu="L4",
    cpu=4.0,
    memory=16384,
    # Keep the service scaled to zero when idle; the first request incurs a
    # cold start while the index and model are loaded.
    min_containers=0,
    max_containers=1,
    timeout=1800,
    secrets=[modal.Secret.from_name("gemini-aic")],
    volumes={
        "/mnt/out": embed_results,
        "/mnt/models": model_cache,
        "/mnt/frames": frames_volume,
    },
)
@modal.asgi_app()
def fastapi_app():
    import sys

    sys.path.insert(0, str(REMOTE_ROOT / "src"))

    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse

    real_app = None
    is_ready = False
    loading_error = None

    def load_heavy_stuff() -> None:
        nonlocal real_app, is_ready, loading_error
        try:
            os.chdir(REMOTE_ROOT)
            # Copy the index/embedding files to the container's local NVMe.
            # The source Volumes remain the durable copy; /tmp is disposable.
            print("Copying embedding artifacts to local container storage...", flush=True)
            data_root = Path("/tmp/aic_data")
            data_root.mkdir(parents=True, exist_ok=True)
            shutil.copytree(
                "/mnt/out/embeddings", data_root / "embeddings", dirs_exist_ok=True
            )
            shutil.copytree(
                "/mnt/out/indexes", data_root / "indexes", dirs_exist_ok=True
            )

            Path("data").mkdir(parents=True, exist_ok=True)
            _replace_link(Path("data/embeddings"), str(data_root / "embeddings"))
            _replace_link(Path("data/indexes"), str(data_root / "indexes"))
            _replace_link(Path("data/manifests"), "/mnt/out/manifests")
            _replace_link(Path("data/keyframes"), "/mnt/frames/keyframes")
            _replace_link(Path("data/models"), "/mnt/models")

            from aic.config import load_config
            from aic.models_cache import apply_model_cache_env
            from aic.service.app import create_app
            from aic.service.state import build_service_state

            cfg = load_config("configs/t0.yaml")
            # The stored FAISS index is queried on CPU for exact, reproducible
            # results.  SigLIP2 query encoding remains auto (GPU when available).
            cfg.index.device = "cpu"
            cfg.index.gpu_float16 = False
            apply_model_cache_env(cfg.paths.models_dir)
            cfg.paths.data_root = Path("/opt/aic/data")

            print("Building retrieval service state (CPU FAISS)...", flush=True)
            state = build_service_state(cfg)
            real_app = create_app(state, cfg.service)
            is_ready = True
            print("AIC UI is ready.", flush=True)
        except Exception:
            import traceback

            loading_error = traceback.format_exc()
            print(f"Error loading AIC UI:\n{loading_error}", flush=True)

    threading.Thread(target=load_heavy_stuff, daemon=True).start()

    async def asgi_proxy(scope, receive, send):
        if scope["type"] != "http":
            if is_ready and real_app:
                return await real_app(scope, receive, send)
            return

        if loading_error:
            response = HTMLResponse(
                "<html><body><h2>Error loading AIC UI</h2>"
                f"<pre>{loading_error}</pre></body></html>",
                status_code=500,
            )
            return await response(scope, receive, send)
        if not is_ready:
            if scope.get("path", "").startswith("/api/"):
                response = JSONResponse(
                    {"error": "AIC data is still loading; retry shortly."},
                    status_code=503,
                )
            else:
                response = HTMLResponse(
                    "<html><head><meta http-equiv='refresh' content='3'></head>"
                    "<body style='background:#111;color:#fff;display:flex;"
                    "justify-content:center;align-items:center;height:100vh;"
                    "font-family:sans-serif'><h2>Đang nạp dữ liệu và model..."
                    "<br><br>Trang sẽ tự làm mới.</h2></body></html>"
                )
            return await response(scope, receive, send)

        return await real_app(scope, receive, send)

    # FastAPI is only used as a tiny ASGI-compatible wrapper while the real
    # pipeline app is built in the background.
    return asgi_proxy

