"""Image-text encoders behind a registry (the retrieval backbone).

Every encoder produces L2-normalised float32 vectors so cosine similarity is a
plain inner product, and exposes its ``model_id`` so indexes can refuse queries
encoded with a different model (the encoder-version guard from note 09).

The concrete API used by :class:`Siglip2Encoder` was verified against two
transformers generations: 4.57.6 (the project pin) returns the pooled
feature tensor from ``get_image_features``/``get_text_features``, while
newer releases return a ``BaseModelOutputWithPooling`` whose
``pooler_output`` is that same tensor (verified from both sources).
:func:`_pooled_features` bridges the two; any other return shape fails
loudly rather than silently embedding the wrong thing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import numpy as np

from aic.config import EmbedConfig
from aic.models_cache import hf_cache_dir

logger = logging.getLogger(__name__)


class EncoderError(RuntimeError):
    """Raised when encoding fails or produces invalid vectors."""


class ImageTextEncoder(Protocol):
    @property
    def model_id(self) -> str: ...

    def encode_images(self, images: list[np.ndarray]) -> np.ndarray:
        """Encode RGB uint8 images to L2-normalised float32 vectors [n, d]."""
        ...

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode texts to L2-normalised float32 vectors [n, d]."""
        ...


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalise rows; reject NaN/zero rows loudly.

    One NaN row silently poisons an inner-product index (note 09, Phase 3 edge
    cases), so this is the single choke point every encoder output passes
    through.
    """
    if vectors.ndim != 2:
        raise EncoderError(f"expected [n, d] vectors, got shape {vectors.shape}")
    if not np.isfinite(vectors).all():
        raise EncoderError("encoder produced non-finite values (NaN/Inf)")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise EncoderError("encoder produced a zero-norm vector")
    return (vectors / norms).astype(np.float32)


def _text_max_length(model_config) -> int | None:
    """The text tower's position-embedding capacity, if the config exposes it.

    SigLIP/SigLIP2 text towers carry a fixed-size position table (64 for the
    released checkpoints) and were trained on inputs padded to exactly that
    length (model card); a longer tokenization raises inside the text
    embeddings. Read from the checkpoint config, never assumed.
    """
    text_config = getattr(model_config, "text_config", None)
    length = getattr(text_config, "max_position_embeddings", None)
    return int(length) if length else None


def _pooled_features(outputs, torch_module):
    """The pooled feature tensor across transformers generations.

    transformers <= 4.57 returns the tensor directly; newer releases return
    a model output whose ``pooler_output`` holds the identical tensor. A
    mean over hidden states is deliberately NOT a fallback: it lives in a
    different embedding space and would poison the index silently.
    """
    if torch_module.is_tensor(outputs):
        return outputs
    pooled = getattr(outputs, "pooler_output", None)
    if pooled is None or not torch_module.is_tensor(pooled):
        raise EncoderError(
            "unexpected feature output type "
            f"{type(outputs).__name__}; expected a tensor or a model output "
            "with pooler_output (transformers API change?)"
        )
    return pooled


class Siglip2Encoder:
    """SigLIP 2 via transformers, loaded lazily, weights cached in-project."""

    def __init__(
        self, model_id: str, device: str, batch_size: int, models_dir: Path
    ) -> None:
        self._model_id = model_id
        self._device_setting = device
        self._batch_size = batch_size
        self._cache_dir = hf_cache_dir(models_dir)
        self._model = None
        self._processor = None

    @property
    def model_id(self) -> str:
        return self._model_id

    def _ensure_loaded(self):
        if self._model is None:
            import torch
            from transformers import AutoModel, AutoProcessor

            device = self._device_setting
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device.startswith("cuda") else torch.float32
            self._model = AutoModel.from_pretrained(
                self._model_id, cache_dir=str(self._cache_dir), dtype=dtype
            ).to(device)
            self._model.eval()
            self._processor = AutoProcessor.from_pretrained(
                self._model_id, cache_dir=str(self._cache_dir)
            )
            self._text_max_length = _text_max_length(self._model.config)
            self._device = device
            self._torch = torch
            logger.info("loaded %s on %s", self._model_id, device)
        return self._model

    def encode_images(self, images: list[np.ndarray]) -> np.ndarray:
        model = self._ensure_loaded()
        chunks = []
        for start in range(0, len(images), self._batch_size):
            batch = images[start : start + self._batch_size]
            inputs = self._processor(images=batch, return_tensors="pt").to(self._device)
            with self._torch.no_grad():
                features = _pooled_features(
                    model.get_image_features(**inputs), self._torch
                )
            chunks.append(features.float().cpu().numpy())
        return l2_normalize(np.concatenate(chunks, axis=0))

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        model = self._ensure_loaded()
        chunks = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            # padding="max_length" is how SigLIP was trained; truncation is
            # mandatory with it, because padding alone never shortens an
            # over-length query (e.g. a long fixture sentence) and the text
            # tower hard-fails past its position table. max_length=None
            # falls back to the tokenizer's own model_max_length.
            inputs = self._processor(
                text=batch,
                padding="max_length",
                truncation=True,
                max_length=self._text_max_length,
                return_tensors="pt",
            ).to(self._device)
            with self._torch.no_grad():
                features = _pooled_features(
                    model.get_text_features(**inputs), self._torch
                )
            chunks.append(features.float().cpu().numpy())
        return l2_normalize(np.concatenate(chunks, axis=0))


class PECoreEncoder:
    """Meta Perception Encoder (PE-Core) via the perception_models package.

    Verified against facebookresearch/perception_models (commit pinned in
    requirements-gpu.txt): ``pe.CLIP.from_config(name, pretrained=True)``
    downloads ``facebook/{name}/{name}.pt`` through ``hf_hub_download`` (so
    the project cache env applies), ``transforms.get_image_transform(
    model.image_size)`` returns a PIL->tensor callable, ``transforms.
    get_text_tokenizer(model.context_length)`` returns a tokenizer whose
    ``__call__`` yields a ``[n, context_length]`` LongTensor, and
    ``encode_image``/``encode_text`` accept ``normalize=True``.

    ``model_id`` here is the PE config name (e.g. PE-Core-L14-336). The text
    tower is English-first: queries must be translated before encoding,
    which the query path already does. Licence note: the released weights
    are Apache-2.0, the reference code is FAIR Noncommercial Research —
    fine for this research competition, flagged in note 10.
    """

    def __init__(
        self, model_id: str, device: str, batch_size: int, models_dir: Path
    ) -> None:
        self._model_id = model_id
        self._device_setting = device
        self._batch_size = batch_size
        self._models_dir = models_dir
        self._model = None

    @property
    def model_id(self) -> str:
        return self._model_id

    def _ensure_loaded(self):
        if self._model is None:
            from aic.models_cache import apply_model_cache_env

            # fetch_pe_checkpoint has no cache_dir argument; it honours the
            # HF cache env, which this pins into the project tree.
            apply_model_cache_env(self._models_dir)
            import torch

            try:
                import core.vision_encoder.pe as pe
                import core.vision_encoder.transforms as pe_transforms
            except ImportError as exc:
                raise EncoderError(
                    "perception_models is not installed in this environment; "
                    "see requirements-gpu.txt for the pinned install"
                ) from exc

            device = self._device_setting
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            model = pe.CLIP.from_config(self._model_id, pretrained=True)
            self._model = model.to(device)
            self._model.eval()
            self._preprocess = pe_transforms.get_image_transform(model.image_size)
            self._tokenizer = pe_transforms.get_text_tokenizer(model.context_length)
            self._device = device
            self._torch = torch
            logger.info("loaded PE-Core %s on %s", self._model_id, device)
        return self._model

    def encode_images(self, images: list[np.ndarray]) -> np.ndarray:
        from PIL import Image

        model = self._ensure_loaded()
        chunks = []
        for start in range(0, len(images), self._batch_size):
            batch = images[start : start + self._batch_size]
            pixel_values = self._torch.stack(
                [self._preprocess(Image.fromarray(image)) for image in batch]
            ).to(self._device)
            with self._torch.no_grad():
                features = model.encode_image(pixel_values, normalize=True)
            chunks.append(features.float().cpu().numpy())
        return l2_normalize(np.concatenate(chunks, axis=0))

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        model = self._ensure_loaded()
        chunks = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            tokens = self._tokenizer(batch).to(self._device)
            with self._torch.no_grad():
                features = model.encode_text(tokens, normalize=True)
            chunks.append(features.float().cpu().numpy())
        return l2_normalize(np.concatenate(chunks, axis=0))


_ENCODERS = {
    "siglip2": Siglip2Encoder,
    "pe_core": PECoreEncoder,
}


def build_image_text_encoder(cfg: EmbedConfig, models_dir: Path) -> ImageTextEncoder:
    """Instantiate the configured encoder by registry name."""
    try:
        encoder_cls = _ENCODERS[cfg.model]
    except KeyError:
        known = ", ".join(sorted(_ENCODERS))
        raise ValueError(
            f"unknown image-text encoder {cfg.model!r}; known encoders: {known}"
        ) from None
    return encoder_cls(
        model_id=cfg.model_id,
        device=cfg.device,
        batch_size=cfg.batch_size,
        models_dir=models_dir,
    )
