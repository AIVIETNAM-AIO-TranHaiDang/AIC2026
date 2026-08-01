"""Plan a one-server-per-GPU local LLM deployment.

The offline LLM stages (captioning, Query Cortex, VLM-verify) scale by pointing
at several OpenAI-compatible endpoints (the ``EndpointPool`` fan-out). On a box
with N GPUs — Kaggle's 2x T4, or a multi-GPU server — the cheapest way to get N
endpoints is to run N copies of ``llama-server``, each pinned to one GPU via
``CUDA_VISIBLE_DEVICES`` on its own port. This module builds that launch plan;
``scripts/serve_local_llms.py`` executes it (launch, health-check, print the
joined base-URL list) so nothing GPU- or process-specific is hardcoded.

The planning is pure and side-effect-free so it can be unit-tested without a
GPU or the ``llama-server`` binary.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from aic.parallel import gpu_labels

# llama.cpp health endpoint served next to the OpenAI-compatible /v1 routes.
HEALTH_PATH = "/health"


@dataclass(frozen=True)
class ServerSpec:
    """One llama-server instance: which GPU, which port, and how to launch it."""

    gpu_index: int
    port: int
    base_url: str
    health_url: str
    argv: list[str]
    env: dict[str, str] = field(default_factory=dict)


def plan_servers(
    *,
    server_bin: str,
    model: str,
    num_gpus: int,
    host: str,
    base_port: int,
    ngl: int,
    ctx: int,
    mmproj: str | None = None,
    extra_args: Sequence[str] = (),
) -> list[ServerSpec]:
    """Build one :class:`ServerSpec` per GPU.

    Server ``i`` is pinned to one card via ``CUDA_VISIBLE_DEVICES`` and gets
    port ``base_port + i``. The card label comes from
    :func:`aic.parallel.gpu_labels`, which remaps rank ``i`` through the
    parent's own ``CUDA_VISIBLE_DEVICES`` mask — under an operator restriction
    like ``1,3`` the plain rank would grab globally-indexed cards the operator
    excluded. ``mmproj`` (the vision projector) is added when given — required
    for the multimodal caption / VLM-verify stages. Extra llama-server flags
    are appended verbatim.
    """
    if num_gpus < 1:
        raise ValueError(f"num_gpus must be >= 1, got {num_gpus}")
    labels = gpu_labels(num_gpus)
    specs: list[ServerSpec] = []
    for i in range(num_gpus):
        port = base_port + i
        argv = [
            server_bin,
            "-m", model,
            "--host", host,
            "--port", str(port),
            "-ngl", str(ngl),
            "-c", str(ctx),
            "--jinja",
        ]
        if mmproj:
            argv += ["--mmproj", mmproj]
        argv += list(extra_args)
        specs.append(
            ServerSpec(
                gpu_index=i,
                port=port,
                base_url=f"http://{host}:{port}/v1",
                health_url=f"http://{host}:{port}{HEALTH_PATH}",
                argv=argv,
                env={"CUDA_VISIBLE_DEVICES": labels[i]},
            )
        )
    return specs


def joined_base_url(specs: Sequence[ServerSpec]) -> str:
    """Comma-joined base URLs, the value for ``LOCAL_LLM_BASE_URL``.

    The config's ``resolve_endpoints`` splits this back into a per-endpoint
    list, so one env var carries the whole multi-GPU deployment.
    """
    return ",".join(spec.base_url for spec in specs)
