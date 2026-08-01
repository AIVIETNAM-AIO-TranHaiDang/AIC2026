"""Text embedders producing dense and learned-sparse representations.

Two embedders behind one registry:

- ``bge_m3`` — verified against FlagEmbedding 1.4.0:
  ``BGEM3FlagModel(model_name_or_path, normalize_embeddings=True,
  use_fp16=..., devices=..., cache_dir=..., batch_size=...)`` and
  ``encode(sentences, return_dense=True, return_sparse=True)`` returning
  ``{"dense_vecs": np.ndarray, "lexical_weights": list[dict[str, float]]}``.
  One model serves both the semantic (dense) and literal (sparse) channels,
  Vietnamese included. Queries and documents share one encoding.
- ``qwen3_embedding`` — verified against the Qwen3-Embedding model card
  (transformers >= 4.51; this project pins 4.57.6): ``AutoTokenizer`` with
  ``padding_side='left'``, ``AutoModel``, last-token pooling over
  ``last_hidden_state``, and an instruction prefix on queries only
  (``Instruct: {task}\\nQuery:{query}``). Dense-only: it may only be
  configured as ``textstack.dense`` (the semantic channel); the sparse
  channel keeps BGE-M3.

The dense/sparse split is wired by :func:`build_textstack_embedders`, which
returns the same object twice when no dense override is configured.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from aic.config import DenseTextConfig, TextStackConfig
from aic.embed.encoders import l2_normalize
from aic.models_cache import hf_cache_dir

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TextEmbeddings:
    dense: np.ndarray
    """L2-normalised float32 vectors [n, d]."""
    sparse: list[dict[str, float]]
    """Per-text learned term weights (token id -> weight). Empty list for
    dense-only embedders."""


class TextEmbedder(Protocol):
    @property
    def model_id(self) -> str: ...

    def encode(self, texts: list[str]) -> TextEmbeddings:
        """Encode documents."""
        ...

    def encode_queries(self, texts: list[str]) -> TextEmbeddings:
        """Encode queries; instruction-tuned embedders add their prefix here."""
        ...


class BgeM3Embedder:
    """BGE-M3 via FlagEmbedding, lazy-loaded, weights cached in-project."""

    def __init__(
        self, cfg: TextStackConfig | DenseTextConfig, models_dir: Path
    ) -> None:
        self._cfg = cfg
        self._cache_dir = hf_cache_dir(models_dir)
        self._model = None

    @property
    def model_id(self) -> str:
        return self._cfg.model_id

    def _ensure_loaded(self):
        if self._model is None:
            import torch
            from FlagEmbedding import BGEM3FlagModel

            device = self._cfg.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = BGEM3FlagModel(
                self._cfg.model_id,
                normalize_embeddings=True,
                use_fp16=device.startswith("cuda"),
                devices=device,
                cache_dir=str(self._cache_dir),
                batch_size=self._cfg.batch_size,
            )
            logger.info("loaded %s on %s", self._cfg.model_id, device)
        return self._model

    def encode(self, texts: list[str]) -> TextEmbeddings:
        if not texts:
            raise ValueError("encode needs at least one text")
        model = self._ensure_loaded()
        output = model.encode(texts, return_dense=True, return_sparse=True)
        dense = l2_normalize(np.asarray(output["dense_vecs"], dtype=np.float32))
        sparse = [
            {term: float(weight) for term, weight in doc.items()}
            for doc in output["lexical_weights"]
        ]
        return TextEmbeddings(dense=dense, sparse=sparse)

    def encode_queries(self, texts: list[str]) -> TextEmbeddings:
        # BGE-M3 is symmetric: queries and documents share one encoding.
        return self.encode(texts)

    def encode_colbert(self, texts: list[str]) -> list[np.ndarray]:
        """Per-token multi-vectors for late-interaction MaxSim (Phase 7).

        Verified against FlagEmbedding 1.4.0: ``encode(...,
        return_colbert_vecs=True)`` returns ``colbert_vecs`` as one
        ``[n_tokens, dim]`` array per text (the CLS token is excluded by
        the library). Only BGE-M3 offers this; the re-rank escalation
        checks for the method's presence.
        """
        if not texts:
            raise ValueError("encode_colbert needs at least one text")
        model = self._ensure_loaded()
        output = model.encode(
            texts,
            return_dense=False,
            return_sparse=False,
            return_colbert_vecs=True,
        )
        return [
            np.asarray(vecs, dtype=np.float32) for vecs in output["colbert_vecs"]
        ]


class Qwen3Embedder:
    """Qwen3-Embedding via transformers; dense-only, instruction-aware."""

    def __init__(self, cfg: DenseTextConfig, models_dir: Path) -> None:
        self._cfg = cfg
        self._cache_dir = hf_cache_dir(models_dir)
        self._model = None

    @property
    def model_id(self) -> str:
        return self._cfg.model_id

    def _ensure_loaded(self):
        if self._model is None:
            import torch
            from transformers import AutoModel, AutoTokenizer

            device = self._cfg.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device.startswith("cuda") else torch.float32
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._cfg.model_id,
                cache_dir=str(self._cache_dir),
                padding_side="left",
            )
            self._model = AutoModel.from_pretrained(
                self._cfg.model_id, cache_dir=str(self._cache_dir), dtype=dtype
            ).to(device)
            self._model.eval()
            self._device = device
            self._torch = torch
            logger.info("loaded %s on %s", self._cfg.model_id, device)
        return self._model

    def _encode_raw(self, texts: list[str]) -> np.ndarray:
        model = self._ensure_loaded()
        chunks = []
        for start in range(0, len(texts), self._cfg.batch_size):
            batch = texts[start : start + self._cfg.batch_size]
            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self._cfg.max_length,
                return_tensors="pt",
            ).to(self._device)
            with self._torch.no_grad():
                outputs = model(**inputs)
            # Left padding puts every sequence's last real token at -1
            # (the pooling recipe from the Qwen3-Embedding model card).
            pooled = outputs.last_hidden_state[:, -1]
            chunks.append(pooled.float().cpu().numpy())
        return l2_normalize(np.concatenate(chunks, axis=0))

    def encode(self, texts: list[str]) -> TextEmbeddings:
        if not texts:
            raise ValueError("encode needs at least one text")
        return TextEmbeddings(dense=self._encode_raw(texts), sparse=[])

    def encode_queries(self, texts: list[str]) -> TextEmbeddings:
        if not texts:
            raise ValueError("encode_queries needs at least one text")
        prefixed = [
            f"Instruct: {self._cfg.query_instruction}\nQuery:{text}"
            for text in texts
        ]
        return TextEmbeddings(dense=self._encode_raw(prefixed), sparse=[])


_TEXT_EMBEDDERS = {
    "bge_m3": BgeM3Embedder,
    "qwen3_embedding": Qwen3Embedder,
}


def build_text_embedder(
    cfg: TextStackConfig | DenseTextConfig, models_dir: Path
) -> TextEmbedder:
    try:
        embedder_cls = _TEXT_EMBEDDERS[cfg.model]
    except KeyError:
        known = ", ".join(sorted(_TEXT_EMBEDDERS))
        raise ValueError(
            f"unknown text embedder {cfg.model!r}; known embedders: {known}"
        ) from None
    return embedder_cls(cfg, models_dir)


def build_textstack_embedders(
    cfg: TextStackConfig, models_dir: Path
) -> tuple[TextEmbedder, TextEmbedder]:
    """Return ``(dense_embedder, sparse_embedder)`` per the config.

    Without a ``dense`` override both roles are the same object, so the
    model loads once and the semantic/literal indexes stay consistent.
    """
    sparse = build_text_embedder(cfg, models_dir)
    if cfg.dense is None:
        return sparse, sparse
    return build_text_embedder(cfg.dense, models_dir), sparse
