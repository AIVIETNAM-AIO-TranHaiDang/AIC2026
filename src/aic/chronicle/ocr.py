"""On-screen text extraction behind a registry.

Two backends:

- ``easyocr`` — the Tier-0 CPU-friendly default (verified against 1.7.2:
  ``Reader(lang_list, gpu=..., model_storage_directory=...)`` and
  ``readtext(image, detail=1)`` returning ``(box, text, confidence)``
  tuples; Vietnamese is in its supported language list).
- ``vlm`` — any OCR-capable vision-language model behind an
  OpenAI-compatible endpoint. The primary target is PaddleOCR-VL-1.6
  (SOTA document/text parsing as of mid-2026, 109 languages including
  Vietnamese, 0.9B parameters) served by vLLM, which avoids importing the
  PaddlePaddle dependency tree into this environment entirely; general
  VLMs such as Qwen3-VL work with an overridden prompt. See
  docs/note/10-model-sota-audit-2026-07.md for the selection evidence.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import httpx
import numpy as np

from aic.chronicle.caption import image_to_data_url, sampling_payload
from aic.chronicle.schema import OcrLine, normalize_vietnamese
from aic.config import OcrConfig
from aic.models_cache import ocr_cache_dir
from aic.oaicompat import ChatCompletionsClient, EndpointError, build_endpoint_pool

logger = logging.getLogger(__name__)

# VLM endpoints return plain text without per-line confidence; lines are
# recorded at full confidence and min_confidence does not apply.
VLM_LINE_CONFIDENCE = 1.0


def _normalize_box(
    box: list, width: int, height: int
) -> list[float] | None:
    """Axis-aligned normalised [x0, y0, x1, y1] from an EasyOCR polygon.

    EasyOCR ``detail=1`` returns a 4-point polygon in pixel coordinates; the
    ledger only needs the enclosing box in 0-1 space for reading-order sorting.
    Returns None if the geometry is unusable (zero-sized frame), so the row
    degrades to manifest order rather than carrying a bad box.
    """
    if width <= 0 or height <= 0 or not box:
        return None
    xs = [float(point[0]) for point in box]
    ys = [float(point[1]) for point in box]
    return [
        min(xs) / width,
        min(ys) / height,
        max(xs) / width,
        max(ys) / height,
    ]


class OcrBackend(Protocol):
    def read(self, image: np.ndarray) -> list[OcrLine]:
        """Extract text lines from an RGB uint8 image."""
        ...


def ensure_easyocr_models(cfg: OcrConfig, models_dir: Path) -> None:
    """Download the EasyOCR models once, before data-parallel workers race.

    Two workers started against an empty cache both download to the same
    ``<storage>/temp.zip`` and delete it out from under each other (seen on
    Kaggle 2x T4: the first frames of each worker fail with ENOENT). A
    throwaway CPU reader in the parent fills the cache first, so every
    worker finds the models present and never downloads. No-op when the
    models already exist — EasyOCR checks before downloading.
    """
    import easyocr

    storage = ocr_cache_dir(models_dir)
    storage.mkdir(parents=True, exist_ok=True)
    easyocr.Reader(
        cfg.languages,
        gpu=False,
        model_storage_directory=str(storage),
        verbose=False,
    )


class EasyOcrBackend:
    """EasyOCR adapter; models download into the project models directory."""

    def __init__(self, cfg: OcrConfig, models_dir: Path) -> None:
        self._cfg = cfg
        self._storage = ocr_cache_dir(models_dir)
        self._reader = None

    def _ensure_reader(self):
        if self._reader is None:
            import easyocr
            import torch

            self._storage.mkdir(parents=True, exist_ok=True)
            self._reader = easyocr.Reader(
                self._cfg.languages,
                gpu=torch.cuda.is_available(),
                model_storage_directory=str(self._storage),
                verbose=False,
            )
            logger.info("loaded EasyOCR for languages %s", self._cfg.languages)
        return self._reader

    def read(self, image: np.ndarray) -> list[OcrLine]:
        reader = self._ensure_reader()
        height, width = image.shape[0], image.shape[1]
        lines = []
        for box, text, confidence in reader.readtext(image, detail=1):
            if confidence < self._cfg.min_confidence:
                continue
            normalized = normalize_vietnamese(text)
            if normalized:
                lines.append(
                    OcrLine(
                        text=normalized,
                        confidence=float(confidence),
                        bbox=_normalize_box(box, width, height),
                    )
                )
        return lines


class VlmOcrError(EndpointError):
    """Raised when a VLM OCR request fails; the job logs and moves on."""


def _ocr_lines_from_reply(text: str) -> list[OcrLine]:
    """Split a VLM reply into normalized, non-empty OCR lines."""
    lines = []
    for raw_line in (text or "").splitlines():
        normalized = normalize_vietnamese(raw_line)
        if normalized:
            lines.append(OcrLine(text=normalized, confidence=VLM_LINE_CONFIDENCE))
    return lines


class VlmOcrBackend:
    """OCR through one or more OpenAI-compatible chat-completions endpoints.

    Sends one frame per request with the configured prompt (PaddleOCR-VL's
    ``"OCR:"`` by default) and splits the reply into lines. Endpoints are
    self-deployed, so the API key is optional, mirroring the captioner; a list
    of ``base_url`` fans OCR out and fails over across the pool (see
    :func:`aic.config.resolve_endpoints`). A blank frame yields an empty reply
    and therefore no lines, which is not an error.
    """

    def __init__(
        self,
        cfg: OcrConfig,
        models_dir: Path,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        # models_dir is part of the registry constructor contract; a remote
        # endpoint stores its own weights, so it is unused here.
        del models_dir
        assert cfg.base_url and cfg.model  # enforced by OcrConfig validation
        self._cfg = cfg
        self._pool = build_endpoint_pool(
            provider="local",
            base_url=cfg.base_url,
            model=cfg.model,
            api_key_env=cfg.api_key_env,
            timeout_s=cfg.timeout_s,
            host_max_retries=cfg.host_max_retries,
            rate_limit_max_retries=cfg.rate_limit_max_retries,
            rate_limit_backoff_s=cfg.rate_limit_backoff_s,
            extra_fields_for=lambda _model: sampling_payload(cfg),
            transport=transport,
            error_cls=VlmOcrError,
        )

    def _read_one(
        self, client: ChatCompletionsClient, image: np.ndarray
    ) -> list[OcrLine]:
        data_url = image_to_data_url(image, self._cfg.image_jpeg_quality)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": self._cfg.prompt},
                ],
            }
        ]
        text = client.complete(
            messages,
            max_tokens=self._cfg.max_output_tokens,
            json_response=False,
            allow_empty=True,
        )
        return _ocr_lines_from_reply(text)

    def read(self, image: np.ndarray) -> list[OcrLine]:
        """Read one frame with pool-level failover across endpoints."""
        [result] = self._pool.run_batch([image], self._read_one, 1)
        if isinstance(result, Exception):
            raise result
        return result

    def read_many(
        self, images: list[np.ndarray]
    ) -> list[list[OcrLine] | Exception]:
        """Read many frames concurrently across the endpoint pool.

        Returns one entry per input frame in order: its OCR lines on success,
        or the exception that ended its attempts (the caller records that
        keyframe as failed).
        """
        return self._pool.run_batch(images, self._read_one, self._cfg.num_parallel)

    def close(self) -> None:
        self._pool.close()


def merge_near_identical(lines: list[OcrLine]) -> list[OcrLine]:
    """Collapse repeats of the same on-screen text within a shot.

    Lower-thirds and tickers persist across many keyframes, so the same string
    arrives once per sampled frame. Case-folded NFC equality keeps the highest
    confidence occurrence, preserving first-seen order.
    """
    best: dict[str, OcrLine] = {}
    order: list[str] = []
    for line in lines:
        key = normalize_vietnamese(line.text).casefold()
        if key not in best:
            best[key] = line
            order.append(key)
        elif line.confidence > best[key].confidence:
            best[key] = line
    return [best[key] for key in order]


_OCR_BACKENDS = {
    "easyocr": EasyOcrBackend,
    "vlm": VlmOcrBackend,
}


def build_ocr_backend(cfg: OcrConfig, models_dir: Path) -> OcrBackend:
    try:
        backend_cls = _OCR_BACKENDS[cfg.backend]
    except KeyError:
        known = ", ".join(sorted(_OCR_BACKENDS))
        raise ValueError(
            f"unknown OCR backend {cfg.backend!r}; known backends: {known}"
        ) from None
    return backend_cls(cfg, models_dir)
