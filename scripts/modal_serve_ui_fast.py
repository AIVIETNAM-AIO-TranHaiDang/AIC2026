"""Optimized Modal deployment for the full SigLIP2 retrieval UI.

This is additive: ``modal_serve_ui.py`` remains available as the conservative
deployment.  The optimized path avoids copying or loading the raw embedding
shards at startup.  It copies only the files consumed by ``VectorIndex.load``
and reads keyframe metadata directly from the embedding manifest.

Deploy into the existing app name/URL::

    MODAL_PROFILE=new-workspace modal deploy scripts/modal_serve_ui_fast.py
"""

from __future__ import annotations

import os
import shutil
import threading
import time
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

# Reuse the existing app name so deploying this script replaces the slow
# revision and preserves the public URL.
app = modal.App("aic-ui-siglip2")

embed_results = modal.Volume.from_name("aic-embeddings", create_if_missing=False)
model_cache = modal.Volume.from_name("aic-model-cache", create_if_missing=False)
frames_volume = modal.Volume.from_name("aic-frames", create_if_missing=False)


def _replace_link(path: Path, target: str) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    path.symlink_to(target)


def _copy_required_index(source: Path, destination: Path) -> None:
    """Copy only artifacts read by the current ``VectorIndex.load`` path."""

    destination.mkdir(parents=True, exist_ok=True)
    for filename in ("meta.json", "ids.json", "vectors.npy"):
        source_path = source / filename
        if not source_path.is_file():
            raise FileNotFoundError(f"required index artifact missing: {source_path}")
        size_mib = source_path.stat().st_size / (1024 * 1024)
        started = time.perf_counter()
        print(f"Copying {filename} ({size_mib:.1f} MiB)...", flush=True)
        shutil.copyfile(source_path, destination / filename)
        elapsed = time.perf_counter() - started
        print(f"Copied {filename} in {elapsed:.1f}s", flush=True)


@app.function(
    image=image,
    gpu="L4",
    cpu=4.0,
    memory=16384,
    min_containers=1,
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

    from fastapi.responses import HTMLResponse, JSONResponse

    real_app = None
    is_ready = False
    loading_error = None

    def load_heavy_stuff() -> None:
        nonlocal real_app, is_ready, loading_error
        try:
            os.chdir(REMOTE_ROOT)

            from aic.config import load_config
            from aic.models_cache import apply_model_cache_env

            cfg = load_config("configs/t0.yaml")
            cfg.index.device = "cpu"
            cfg.index.gpu_float16 = False

            Path("data").mkdir(parents=True, exist_ok=True)
            _replace_link(Path("data/embeddings"), "/mnt/out/embeddings")
            _replace_link(Path("data/manifests"), "/mnt/out/manifests")
            _replace_link(Path("data/keyframes"), "/mnt/frames/keyframes")
            _replace_link(Path("data/models"), "/mnt/models")

            # Derive the index directory from the configured model rather than
            # hard-coding the SigLIP2 slug into the deployment script.
            from aic.embed.job import model_slug

            index_name = f"keyframes-{model_slug(cfg.embed.model_id)}"
            local_indexes = Path("/tmp/aic_data/indexes")
            local_index = local_indexes / index_name
            _copy_required_index(Path("/mnt/out/indexes") / index_name, local_index)
            _replace_link(Path("data/indexes"), str(local_indexes))

            cfg.paths.data_root = Path("/opt/aic/data")
            apply_model_cache_env(cfg.paths.models_dir)

            # Install the retrieval-only metadata loader before service.state
            # imports its function references. Both assignments are explicit
            # so the behavior remains correct if that module was imported by a
            # future dependency earlier in startup.
            from aic.retrieval import build as retrieval_build
            from aic.retrieval.fast_meta import load_keyframe_meta_manifest

            retrieval_build.load_keyframe_meta = load_keyframe_meta_manifest

            from aic.service import state as service_state
            from aic.service.app import create_app

            service_state.load_keyframe_meta = load_keyframe_meta_manifest

            print(
                "Building retrieval state without loading embedding shards...",
                flush=True,
            )
            state = service_state.build_service_state(cfg)
            real_app = create_app(state, cfg.service)
            is_ready = True
            print("AIC UI is ready (optimized startup).", flush=True)
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
                    "font-family:sans-serif'><h2>Đang nạp index và model..."
                    "<br><br>Trang sẽ tự làm mới.</h2></body></html>"
                )
            return await response(scope, receive, send)

        return await real_app(scope, receive, send)

    return asgi_proxy

