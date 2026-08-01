"""Exact inner-product vector index with an encoder-version guard.

Two interchangeable backends behind one class: FAISS (IndexFlatIP) and a plain
numpy matmul. Both are exact — at competition scale (about 3 GB of fp16
vectors, note 06/08) exact search is fast enough, and approximate indexes only
enter at tier T2. The index directory records the producing ``model_id``;
searching with vectors from any other encoder raises, because a silent
encoder mismatch returns plausible-looking garbage.

GPU residency (Phase 10) is a runtime property, never an artifact property:
the CPU index stays the source of truth (FAISS serializes CPU indexes), and
``device='auto'|'cuda'`` clones it onto the GPU for search only. GPU support
is detected from the installed faiss build (``hasattr(faiss,
"StandardGpuResources")``), not from CUDA alone — faiss-cpu and faiss-gpu
are conflicting distributions of the same module, so a CUDA machine with
faiss-cpu must quietly stay on the CPU under 'auto'. The GPU clone API was
verified against the faiss sources: ``index_cpu_to_gpu(resources, device,
index, options)`` with ``GpuClonerOptions.useFloat16`` mapping to
``GpuIndexFlatConfig.useFloat16`` (fp16 storage) for flat indexes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from aic.manifest import write_json_atomic

logger = logging.getLogger(__name__)

_META_FILE = "meta.json"
_IDS_FILE = "ids.json"
_VECTORS_FILE = "vectors.npy"
_FAISS_FILE = "index.faiss"

_BACKENDS = ("numpy", "faiss")


class IndexError_(RuntimeError):
    """Raised on index build/load/search contract violations."""


class VectorIndex:
    """Exact top-k inner-product search over L2-normalised vectors."""

    def __init__(
        self,
        vectors: np.ndarray,
        ids: list[str],
        model_id: str,
        backend: str,
        device: str = "cpu",
        gpu_float16: bool = False,
    ) -> None:
        if backend not in _BACKENDS:
            raise IndexError_(
                f"unknown index backend {backend!r}; known: {', '.join(_BACKENDS)}"
            )
        if vectors.ndim != 2 or len(vectors) != len(ids):
            raise IndexError_(
                f"vectors {vectors.shape} do not align with {len(ids)} ids"
            )
        if not np.isfinite(vectors).all():
            raise IndexError_("index vectors contain NaN/Inf")
        self._vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self._ids = list(ids)
        self._model_id = model_id
        self._backend = backend
        self._row_by_id: dict[str, int] | None = None
        self._faiss_index = None
        self._gpu_index = None
        self._gpu_resources = None
        if backend == "numpy" and device == "cuda":
            raise IndexError_(
                "the numpy index backend has no GPU path; set index.backend "
                "to faiss or index.device to auto/cpu"
            )
        if backend == "faiss":
            import faiss

            self._faiss_index = faiss.IndexFlatIP(self._vectors.shape[1])
            self._faiss_index.add(self._vectors)
            if self._resolve_device(faiss, device) == "cuda":
                self._to_gpu(faiss, gpu_float16)

    @staticmethod
    def _resolve_device(faiss_module, device: str) -> str:
        """Where search should run, honouring the faiss build's abilities."""
        if device == "cpu":
            return "cpu"
        has_gpu = (
            hasattr(faiss_module, "StandardGpuResources")
            and hasattr(faiss_module, "get_num_gpus")
            and faiss_module.get_num_gpus() > 0
        )
        if device == "cuda" and not has_gpu:
            raise IndexError_(
                "index.device is 'cuda' but this faiss build has no usable "
                "GPU (faiss-cpu installed, or no CUDA device visible); "
                "install faiss-gpu-cu12 per requirements-gpu.txt or use "
                "device auto/cpu"
            )
        return "cuda" if has_gpu else "cpu"

    def _to_gpu(self, faiss_module, gpu_float16: bool) -> None:
        """Clone the CPU index onto GPU 0; fall back to CPU on failure.

        The resources object must outlive the GPU index, so it is stored on
        the instance. A GPU OOM at load time degrades to CPU search with a
        warning (note 09 Phase 10 edge case) instead of taking the service
        down.
        """
        try:
            resources = faiss_module.StandardGpuResources()
            options = faiss_module.GpuClonerOptions()
            options.useFloat16 = gpu_float16
            self._gpu_index = faiss_module.index_cpu_to_gpu(
                resources, 0, self._faiss_index, options
            )
            self._gpu_resources = resources
            logger.info(
                "vector index on GPU (%d vectors, float16=%s)",
                len(self._ids),
                gpu_float16,
            )
        except (RuntimeError, MemoryError) as exc:
            self._gpu_index = None
            self._gpu_resources = None
            logger.warning(
                "GPU index load failed (%s); serving from the CPU index", exc
            )

    @property
    def model_id(self) -> str:
        return self._model_id

    def vectors_for(self, ids: list[str]) -> np.ndarray:
        """The stored vectors for ``ids`` (row-aligned), for relevance
        feedback. Raises on unknown ids so a stale id never yields a silent
        zero vector."""
        if self._row_by_id is None:
            self._row_by_id = {id_: row for row, id_ in enumerate(self._ids)}
        try:
            rows = [self._row_by_id[id_] for id_ in ids]
        except KeyError as exc:
            raise IndexError_(f"unknown vector id: {exc.args[0]!r}") from exc
        return self._vectors[rows]

    @property
    def dim(self) -> int:
        return int(self._vectors.shape[1])

    def __len__(self) -> int:
        return len(self._ids)

    def search(
        self, queries: np.ndarray, top_k: int, query_model_id: str
    ) -> list[list[tuple[str, float]]]:
        """Top-k ``(id, score)`` per query row, best first.

        ``query_model_id`` is the encoder that produced ``queries``; it must
        match the encoder that built this index.
        """
        if query_model_id != self._model_id:
            raise IndexError_(
                f"query vectors come from {query_model_id!r} but this index was "
                f"built with {self._model_id!r}; rebuild or re-encode"
            )
        if queries.ndim != 2 or queries.shape[1] != self.dim:
            raise IndexError_(
                f"query shape {queries.shape} does not match index dim {self.dim}"
            )
        k = min(top_k, len(self._ids))
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        if self._gpu_index is not None:
            scores, indices = self._gpu_index.search(queries, k)
        elif self._faiss_index is not None:
            scores, indices = self._faiss_index.search(queries, k)
        else:
            similarity = queries @ self._vectors.T
            indices = np.argsort(-similarity, axis=1)[:, :k]
            scores = np.take_along_axis(similarity, indices, axis=1)
        return [
            [
                (self._ids[int(idx)], float(score))
                for idx, score in zip(row_idx, row_score, strict=True)
            ]
            for row_idx, row_score in zip(indices, scores, strict=True)
        ]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            directory / _META_FILE,
            {
                "model_id": self._model_id,
                "backend": self._backend,
                "dim": self.dim,
                "count": len(self._ids),
            },
        )
        write_json_atomic(directory / _IDS_FILE, self._ids)
        np.save(directory / _VECTORS_FILE, self._vectors)
        if self._faiss_index is not None:
            import faiss

            faiss.write_index(self._faiss_index, str(directory / _FAISS_FILE))

    @classmethod
    def load(
        cls,
        directory: Path,
        device: str = "cpu",
        gpu_float16: bool = False,
    ) -> VectorIndex:
        """Load an index; ``device``/``gpu_float16`` are runtime choices.

        The artifact on disk is identical regardless of device — GPU
        residency never changes what is stored, only where search runs.
        """
        meta_path = directory / _META_FILE
        if not meta_path.is_file():
            raise IndexError_(f"no index metadata at {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        ids = json.loads((directory / _IDS_FILE).read_text(encoding="utf-8"))
        # Memory-map: the constructor's float32/contiguity pass is a no-copy
        # view over the map, faiss copies rows into its own storage during
        # add, and afterwards the OS can evict the mapped pages — halving the
        # resident footprint versus loading a second in-RAM matrix. The
        # stored matrix stays the source for vectors_for (feedback).
        vectors = np.load(directory / _VECTORS_FILE, mmap_mode="r")
        index = cls(
            vectors=vectors,
            ids=ids,
            model_id=meta["model_id"],
            backend=meta["backend"],
            device=device,
            gpu_float16=gpu_float16,
        )
        if len(index) != meta["count"] or index.dim != meta["dim"]:
            raise IndexError_(f"index at {directory} is inconsistent with its metadata")
        return index
