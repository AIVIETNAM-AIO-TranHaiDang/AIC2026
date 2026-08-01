"""Deploy one llama-server per GPU and print the joined endpoint list.

Auto-detects the visible GPU count and launches that many OpenAI-compatible
llama-server instances, each pinned to one GPU (``CUDA_VISIBLE_DEVICES``) on its
own port, then prints the comma-separated base-URL list to set as
``LOCAL_LLM_BASE_URL`` — which the config's ``resolve_endpoints`` splits so every
LLM stage fans its requests across all of them. Works anywhere N GPUs are
visible (Kaggle 2x T4, a multi-GPU server), not just Colab.

Background by default (the servers persist after this exits, for notebooks);
pass --foreground to stream their logs and keep them tied to this process.

Usage:
    python scripts/serve_local_llms.py \
        --server-bin /content/llama.cpp/build/bin/llama-server \
        --model data/models/qwen35-2b/Qwen3.5-2B-Q4_K_M.gguf \
        --mmproj data/models/qwen35-2b/mmproj-F16.gguf
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aic.localllm import ServerSpec, joined_base_url, plan_servers  # noqa: E402
from aic.log import setup_logging  # noqa: E402
from aic.parallel import resolve_num_gpus  # noqa: E402

# llama-server flag defaults, matching the T4 notebook (-ngl offloads all
# layers to the GPU, -c caps the KV-cache context). All overridable below.
# The context must hold the largest request any stage sends: vlm_verify
# ships top_n candidates with one keyframe each, and a VL projector spends
# several hundred to ~1.2k tokens per image — 4096 fits captioning (2
# frames/shot) but 400s the verify ("exceeds the available context size").
# 16384 covers verify at top_n 8 with margin; the KV cache of a 2B-class
# GGUF at 16k is small next to a T4's 16 GB.
_DEFAULT_NGL = 99
_DEFAULT_CTX = 16384
_DEFAULT_BASE_PORT = 8080
_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_HEALTH_TIMEOUT_S = 240.0
_HEALTH_POLL_S = 2.0


def _is_up(health_url: str) -> bool:
    try:
        with urllib.request.urlopen(health_url, timeout=2):
            return True
    except Exception:  # noqa: BLE001 - any failure means "not up yet"
        return False


def _wait_healthy(specs: list[ServerSpec], timeout_s: float) -> list[ServerSpec]:
    """Poll every server's /health until up or the deadline; return the dead."""
    deadline = time.monotonic() + timeout_s
    pending = list(specs)
    while pending and time.monotonic() < deadline:
        pending = [spec for spec in pending if not _is_up(spec.health_url)]
        if pending:
            time.sleep(_HEALTH_POLL_S)
    return pending


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to the GGUF model.")
    parser.add_argument(
        "--mmproj",
        default=None,
        help="Vision projector GGUF (required for caption / VLM-verify).",
    )
    parser.add_argument(
        "--server-bin",
        default=None,
        help="llama-server binary. Default: the one on PATH.",
    )
    parser.add_argument("--host", default=_DEFAULT_HOST)
    parser.add_argument("--base-port", type=int, default=_DEFAULT_BASE_PORT)
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Servers to launch. Default: every visible GPU (min 1).",
    )
    parser.add_argument("--ngl", type=int, default=_DEFAULT_NGL)
    parser.add_argument("--ctx", type=int, default=_DEFAULT_CTX)
    parser.add_argument(
        "--health-timeout", type=float, default=_DEFAULT_HEALTH_TIMEOUT_S
    )
    parser.add_argument(
        "--log-dir",
        default=".",
        help="Where per-GPU llama-server logs go in background mode.",
    )
    parser.add_argument(
        "--foreground",
        action="store_true",
        help="Stream server logs and stay attached (Ctrl-C stops them). "
        "Default: background (servers persist after this exits).",
    )
    parser.add_argument("--log-level", default="INFO")
    args, extra = parser.parse_known_args()
    # Anything after a literal "--" is passed through to llama-server verbatim.
    if extra and extra[0] == "--":
        extra = extra[1:]

    setup_logging(args.log_level)

    server_bin = args.server_bin or shutil.which("llama-server")
    if not server_bin:
        parser.error("llama-server not found on PATH; pass --server-bin")

    num_gpus = resolve_num_gpus(args.num_gpus)
    specs = plan_servers(
        server_bin=server_bin,
        model=args.model,
        num_gpus=num_gpus,
        host=args.host,
        base_port=args.base_port,
        ngl=args.ngl,
        ctx=args.ctx,
        mmproj=args.mmproj,
        extra_args=extra,
    )

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    procs: list[subprocess.Popen] = []
    for spec in specs:
        env = {**_os_environ(), **spec.env}
        if args.foreground:
            proc = subprocess.Popen(spec.argv, env=env)
        else:
            log_path = log_dir / f"llama-server.gpu{spec.gpu_index}.log"
            proc = subprocess.Popen(
                spec.argv,
                env=env,
                stdout=open(log_path, "a"),
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        procs.append(proc)
        print(f"launched gpu {spec.gpu_index} -> {spec.base_url} (pid {proc.pid})")

    dead = _wait_healthy(specs, args.health_timeout)
    if dead:
        urls = ", ".join(spec.base_url for spec in dead)
        print(f"ERROR: {len(dead)} server(s) did not become healthy: {urls}")
        if not args.foreground:
            print(f"see {log_dir}/llama-server.gpu*.log")
        for proc in procs:
            proc.terminate()
        return 1

    url_list = joined_base_url(specs)
    print(f"\nall {len(specs)} server(s) healthy")
    print(f"LOCAL_LLM_BASE_URL={url_list}")
    print(f'export LOCAL_LLM_BASE_URL="{url_list}"')

    if args.foreground:
        print("\nforeground mode: Ctrl-C to stop all servers")
        try:
            for proc in procs:
                proc.wait()
        except KeyboardInterrupt:
            for proc in procs:
                proc.terminate()
    return 0


def _os_environ() -> dict[str, str]:
    import os

    return dict(os.environ)


if __name__ == "__main__":
    raise SystemExit(main())
