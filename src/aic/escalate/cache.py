"""LRU cache for heavy escalation models — the single-GPU economy mode.

On a multi-GPU server every escalation model is pinned to its own card via
its ``device`` config field and this cache never needs to evict. On a single
rented GPU (e.g. a T4), ``escalations.max_resident_models`` bounds how many
heavy models stay loaded; the least recently used is dropped and CUDA's
cached blocks are released so the next load fits.
"""

from __future__ import annotations

import gc
import logging
import threading
from collections import OrderedDict
from collections.abc import Callable

logger = logging.getLogger(__name__)


class ModelCache:
    """Thread-safe LRU of name -> loaded model."""

    def __init__(self, max_resident: int) -> None:
        if max_resident <= 0:
            raise ValueError("max_resident must be positive")
        self._max_resident = max_resident
        self._models: OrderedDict[str, object] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, name: str, loader: Callable[[], object]) -> object:
        """The cached model, loading (and possibly evicting) as needed."""
        with self._lock:
            if name in self._models:
                self._models.move_to_end(name)
                return self._models[name]
            while len(self._models) >= self._max_resident:
                evicted_name, evicted = self._models.popitem(last=False)
                logger.info("evicting model %r from the escalation cache", evicted_name)
                del evicted
                self._release_memory()
            model = loader()
            self._models[name] = model
            return model

    def __len__(self) -> int:
        with self._lock:
            return len(self._models)

    @staticmethod
    def _release_memory() -> None:
        gc.collect()
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
