"""Imagine->Match: diffusion renders of the spec, image->image search.

Note 08 step 6a: when a KIS-T description matches nothing textually, a
few-step diffusion model renders what the spec *looks like*; the renders are
junk-gated by their image-text cosine against the prompt (the CLIP-score
gate note 09 mandates) and the survivors' centroid probes the keyframe
index. The render ranking is RRF-fused with the current pool so strong
existing evidence is never thrown away.

Diffusers API verified against 0.39.0: ``AutoPipelineForText2Image
.from_pretrained(model_id, cache_dir=..., torch_dtype=...)``, pipeline call
fields ``prompt / num_inference_steps / guidance_scale /
num_images_per_prompt / width / height / generator / output_type``, and
``output_type="np"`` returning float images in [0, 1].
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from aic.config import ImagineEscalationConfig
from aic.cortex.feedback import rank_vector_over_keyframes
from aic.embed.encoders import ImageTextEncoder, l2_normalize
from aic.escalate.base import EscalationOutcome, EscalationRequest
from aic.escalate.cache import ModelCache
from aic.index.vector import VectorIndex
from aic.models_cache import hf_cache_dir
from aic.retrieval.fusion import rrf_fuse

logger = logging.getLogger(__name__)

_POOL_RANKING = "pool"
_IMAGINE_RANKING = "imagine"


class ImagineMatch:
    """Render the spec, gate the junk, search by image, fuse with the pool."""

    def __init__(
        self,
        encoder: ImageTextEncoder,
        index: VectorIndex,
        keyframe_meta: dict[str, tuple[str, int, int]],
        cfg: ImagineEscalationConfig,
        models_dir: Path,
        cache: ModelCache,
        rrf_k: int,
        pipeline_loader=None,
    ) -> None:
        """``pipeline_loader`` overrides the diffusers load (tests inject a
        fake pipeline; production leaves it None)."""
        self._encoder = encoder
        self._index = index
        self._keyframe_meta = keyframe_meta
        self._cfg = cfg
        self._models_dir = models_dir
        self._cache = cache
        self._rrf_k = rrf_k
        self._pipeline_loader = pipeline_loader or self._load_pipeline

    @property
    def name(self) -> str:
        return "imagine"

    def _load_pipeline(self):
        try:
            import torch
            from diffusers import AutoPipelineForText2Image
        except ImportError as exc:
            raise RuntimeError(
                "diffusers is not installed in this environment; the "
                "imagine escalation needs it (see requirements.txt)"
            ) from exc
        device = self._cfg.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        pipeline = AutoPipelineForText2Image.from_pretrained(
            self._cfg.model_id,
            cache_dir=str(hf_cache_dir(self._models_dir)),
            torch_dtype=dtype,
        ).to(device)
        pipeline.set_progress_bar_config(disable=True)
        logger.info("loaded %s on %s", self._cfg.model_id, device)
        return pipeline

    def _prompt(self, request: EscalationRequest) -> str:
        phrases = [p for p in request.spec.visual_phrases if p.strip()]
        return "; ".join(phrases) if phrases else request.query

    def _render(self, prompt: str) -> list[np.ndarray]:
        pipeline = self._cache.get("imagine.diffusion", self._pipeline_loader)
        generator = None
        if self._cfg.seed is not None:
            import torch

            generator = torch.Generator().manual_seed(self._cfg.seed)
        output = pipeline(
            prompt=prompt,
            num_inference_steps=self._cfg.num_inference_steps,
            guidance_scale=self._cfg.guidance_scale,
            num_images_per_prompt=self._cfg.num_images,
            width=self._cfg.width,
            height=self._cfg.height,
            generator=generator,
            output_type="np",
        )
        return [
            (np.clip(image, 0.0, 1.0) * 255).round().astype(np.uint8)
            for image in output.images
        ]

    def run(self, request: EscalationRequest) -> EscalationOutcome:
        prompt = self._prompt(request)
        if not prompt.strip():
            return EscalationOutcome(
                request.candidates, note="empty prompt; nothing to render"
            )
        try:
            renders = self._render(prompt)
        except Exception as exc:  # degrade, never break the console
            logger.warning("imagine render failed: %s", exc)
            return EscalationOutcome(
                request.candidates, note=f"render failed: {exc}"
            )
        image_vectors = self._encoder.encode_images(renders)
        text_vector = self._encoder.encode_texts([prompt])[0]
        scores = image_vectors @ text_vector
        kept = image_vectors[scores >= self._cfg.min_clip_score]
        if not len(kept):
            # The junk gate rejected everything: report it to the operator
            # instead of silently returning the unescalated list (note 09).
            return EscalationOutcome(
                request.candidates,
                note="no usable renders (all failed the CLIP-score gate)",
            )
        centroid = l2_normalize(kept.mean(axis=0, keepdims=True).astype(np.float32))
        imagined = rank_vector_over_keyframes(
            self._index,
            self._keyframe_meta,
            self._encoder.model_id,
            centroid,
            max(len(request.candidates), 1),
        )
        if not request.candidates:
            return EscalationOutcome(imagined)
        fused = rrf_fuse(
            {_POOL_RANKING: request.candidates, _IMAGINE_RANKING: imagined},
            {_POOL_RANKING: 1.0, _IMAGINE_RANKING: self._cfg.fusion_weight},
            self._rrf_k,
        )
        return EscalationOutcome(fused)
