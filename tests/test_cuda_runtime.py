from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from aic import cuda_runtime


def test_cpu_runtime_does_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cuda_runtime.os,
        "execvpe",
        lambda *_args, **_kwargs: pytest.fail("CPU setup must not re-exec"),
    )

    cuda_runtime.ensure_faster_whisper_cuda12_runtime("cpu")


def test_visible_runtime_does_not_reexec(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cuda_runtime, "_cuda_requested", lambda _device: True)
    monkeypatch.setattr(cuda_runtime, "_library_loadable", lambda _name: True)
    monkeypatch.setattr(
        cuda_runtime.os,
        "execvpe",
        lambda *_args, **_kwargs: pytest.fail("visible runtime must not re-exec"),
    )

    cuda_runtime.ensure_faster_whisper_cuda12_runtime("cuda")


def test_missing_runtime_has_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cuda_runtime, "_cuda_requested", lambda _device: True)
    monkeypatch.setattr(cuda_runtime, "_library_loadable", lambda _name: False)
    monkeypatch.setattr(cuda_runtime, "_pip_library_dirs", list)

    with pytest.raises(cuda_runtime.CudaRuntimeError, match="nvidia-cublas-cu12"):
        cuda_runtime.ensure_faster_whisper_cuda12_runtime("cuda")


def test_pip_library_dirs_handles_missing_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_spec(_module_name: str):
        raise ModuleNotFoundError

    monkeypatch.setattr(cuda_runtime.importlib.util, "find_spec", missing_spec)

    assert cuda_runtime._pip_library_dirs() == []


def test_pip_runtime_reexecs_with_library_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cublas = tmp_path / "cublas"
    cudnn = tmp_path / "cudnn"
    cublas.mkdir()
    cudnn.mkdir()
    monkeypatch.setattr(cuda_runtime, "_cuda_requested", lambda _device: True)
    monkeypatch.setattr(cuda_runtime, "_library_loadable", lambda _name: False)
    monkeypatch.setattr(cuda_runtime, "_pip_library_dirs", lambda: [cublas, cudnn])
    monkeypatch.delenv(cuda_runtime._REEXEC_MARKER, raising=False)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing")
    captured: dict[str, object] = {}

    def fake_execvpe(executable: str, argv: list[str], environment: dict[str, str]):
        captured.update(executable=executable, argv=argv, environment=environment)
        raise RuntimeError("reexec intercepted")

    monkeypatch.setattr(cuda_runtime.os, "execvpe", fake_execvpe)

    with pytest.raises(RuntimeError, match="reexec intercepted"):
        cuda_runtime.ensure_faster_whisper_cuda12_runtime("cuda")

    environment = captured["environment"]
    assert isinstance(environment, dict)
    assert environment[cuda_runtime._REEXEC_MARKER] == "1"
    assert environment["LD_LIBRARY_PATH"].split(os.pathsep) == [
        str(cublas),
        str(cudnn),
        "/existing",
    ]
    assert captured["executable"] == sys.executable
    assert captured["argv"] == [sys.executable, *sys.argv]
