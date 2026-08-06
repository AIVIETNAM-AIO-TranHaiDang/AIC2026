"""Native CUDA runtime setup for non-PyTorch inference engines.

PyTorch wheels carry their own versioned CUDA runtime, but CTranslate2's Linux
wheel loads the CUDA 12 cuBLAS/cuDNN sonames dynamically.  A process using a
CUDA 13 PyTorch environment can therefore see the GPU while still failing at
the first Faster-Whisper kernel with ``libcublas.so.12`` missing.

The official Faster-Whisper installation supports installing the CUDA 12
libraries as Python packages.  Their directories must be present in
``LD_LIBRARY_PATH`` *before Python starts*, so this module re-executes the
current entry point once with those directories prepended.  CPU runs and
processes whose system loader already finds the libraries are unchanged.
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
import sys
from pathlib import Path

_CUDA12_LIBRARIES = {
    "nvidia.cublas.lib": "libcublas.so.12",
    "nvidia.cudnn.lib": "libcudnn.so.9",
}
_REEXEC_MARKER = "AIC_FASTER_WHISPER_CUDA12_REEXEC"


class CudaRuntimeError(RuntimeError):
    """Raised when GPU Faster-Whisper lacks its CUDA 12 runtime."""


def _library_loadable(soname: str) -> bool:
    try:
        ctypes.CDLL(soname)
    except OSError:
        return False
    return True


def _pip_library_dirs() -> list[Path]:
    """Return pip-installed CUDA 12 library directories when complete."""
    directories: list[Path] = []
    for module_name, soname in _CUDA12_LIBRARIES.items():
        try:
            spec = importlib.util.find_spec(module_name)
        except ModuleNotFoundError:
            return []
        locations = () if spec is None else spec.submodule_search_locations or ()
        directory = next(
            (
                Path(location)
                for location in locations
                if (Path(location) / soname).is_file()
            ),
            None,
        )
        if directory is None:
            return []
        directories.append(directory)
    return directories


def _cuda_requested(device: str) -> bool:
    if device == "cpu":
        return False
    if device == "auto":
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    return device.startswith("cuda")


def ensure_faster_whisper_cuda12_runtime(device: str) -> None:
    """Make the CUDA 12 runtime visible before GPU CTranslate2 inference.

    When pip-provided libraries are found but not yet loadable, this function
    replaces the current process with an identical command whose
    ``LD_LIBRARY_PATH`` includes them.  It returns normally for CPU execution
    or when the runtime is already visible.
    """
    if sys.platform != "linux" or not _cuda_requested(device):
        return
    if all(_library_loadable(soname) for soname in _CUDA12_LIBRARIES.values()):
        return

    directories = _pip_library_dirs()
    install_hint = (
        "install the Faster-Whisper CUDA 12 runtime with: "
        "python -m pip install nvidia-cublas-cu12 'nvidia-cudnn-cu12==9.*'"
    )
    if not directories:
        raise CudaRuntimeError(
            "GPU Faster-Whisper requires libcublas.so.12 and libcudnn.so.9; "
            + install_hint
        )
    if os.environ.get(_REEXEC_MARKER) == "1":
        raise CudaRuntimeError(
            "CUDA 12 runtime packages are installed but their libraries are still "
            "not loadable after configuring LD_LIBRARY_PATH"
        )

    environment = os.environ.copy()
    existing = [
        path
        for path in environment.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        if path
    ]
    prefixes = [str(path) for path in directories]
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(
        prefixes + [path for path in existing if path not in prefixes]
    )
    environment[_REEXEC_MARKER] = "1"
    os.execvpe(
        sys.executable,
        [sys.executable, *sys.argv],
        environment,
    )
