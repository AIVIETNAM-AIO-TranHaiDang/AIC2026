"""On-disk format for keyframe embedding shards.

Each shard is one 2-D array of row vectors. Two formats are supported and may
coexist in the same directory, because every manifest row records the shard's
filename *with its extension*, so the loader picks the format per file:

- ``safetensors`` (default): records the dtype explicitly and loads without
  executing any code — the safe, dtype-flexible option.
- ``npy``: the legacy NumPy format, kept for shards already written that way
  and for environments that prefer it.

``store_dtype`` (from ``embed.store_dtype``) casts vectors before writing —
e.g. ``float16`` halves shard size — while :func:`load_shard` returns the stored
dtype and the embed job upcasts to float32 for the index. A dtype NumPy cannot
store natively (bfloat16) is rejected for the ``npy`` format at config time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

from aic.config import resolve_store_dtype

SHARD_PREFIX = "shard-"
_TENSOR_KEY = "embeddings"
_EXTENSIONS = {"safetensors": ".safetensors", "npy": ".npy"}


def next_shard_index(out_dir: Path, shard_prefix: str) -> int:
    """One past the highest existing ``<prefix>NNNNN.<ext>`` index, any format.

    ``max + 1`` rather than a count so deleting a shard from the middle of the
    sequence (the repair pass) never makes a later write collide with a
    survivor. Counts both shard formats so a mixed directory stays consistent.
    """
    indices: list[int] = []
    for path in out_dir.glob(f"{shard_prefix}*"):
        if path.suffix not in _EXTENSIONS.values():
            continue
        stem_num = path.stem[len(shard_prefix) :]
        if stem_num.isdigit():
            indices.append(int(stem_num))
    return max(indices) + 1 if indices else 0


def iter_shard_files(out_dir: Path):
    """Yield every shard file in ``out_dir`` (either format), sorted."""
    for path in sorted(out_dir.glob(f"{SHARD_PREFIX}*")):
        if path.suffix in _EXTENSIONS.values():
            yield path


def save_shard(
    out_dir: Path,
    stem: str,
    array: np.ndarray,
    shard_format: str,
    store_dtype: str | None,
) -> str:
    """Write one shard and return its filename (with extension).

    ``stem`` is the extension-less name (e.g. ``shard-00000``); the extension
    is chosen from ``shard_format`` and returned as part of the filename so the
    caller records it in the manifest and the loader can round-trip it.
    """
    if store_dtype is not None:
        dtype = resolve_store_dtype(store_dtype)
        if array.dtype != dtype:
            array = array.astype(dtype)
    array = np.ascontiguousarray(array)
    filename = stem + _EXTENSIONS[shard_format]
    path = out_dir / filename
    if shard_format == "safetensors":
        save_file({_TENSOR_KEY: array}, path)
    else:
        np.save(path, array)
    return filename


def load_shard(path: Path | str) -> np.ndarray:
    """Load a shard, choosing the format from the file suffix.

    Returns the stored dtype (the embed job upcasts to float32 for the index).
    The safetensors array is materialised with a copy so the memory-mapped file
    handle is released immediately — otherwise a later delete (the repair pass)
    could fail on Windows.
    """
    path = Path(path)
    if path.suffix == ".safetensors":
        return np.array(load_file(path)[_TENSOR_KEY])
    return np.load(path)
