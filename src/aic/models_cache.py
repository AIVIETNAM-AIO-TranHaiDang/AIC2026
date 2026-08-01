"""Keep every downloaded model weight inside the project tree.

Project rule (CLAUDE.md): weights live under ``paths.models_dir``
(``data/models`` by default, gitignored), never in user-level caches such as
``~/.cache/huggingface``. Two mechanisms enforce it:

- every backend passes an explicit cache/download argument
  (``cache_dir``, ``download_root``, ``model_storage_directory``), and
- entry-point scripts call :func:`apply_model_cache_env` before importing any
  model library, which covers libraries that only honour environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path


def hf_cache_dir(models_dir: Path) -> Path:
    return models_dir / "hf"


def torch_cache_dir(models_dir: Path) -> Path:
    return models_dir / "torch"


def whisper_cache_dir(models_dir: Path) -> Path:
    return models_dir / "whisper"


def ocr_cache_dir(models_dir: Path) -> Path:
    return models_dir / "ocr"


def apply_model_cache_env(models_dir: Path) -> None:
    """Point cache environment variables into the project models directory.

    Uses ``setdefault`` so an explicitly configured environment still wins.
    Must run before the first import of transformers/torch-hub consumers.
    """
    targets = {
        "HF_HOME": hf_cache_dir(models_dir),
        "TORCH_HOME": torch_cache_dir(models_dir),
    }
    for name, path in targets.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(name, str(path))
