"""SAM 3 / SAM 3.1 detector-counter adapter (Promptable Concept Segmentation).

Verified against the HF ``transformers`` SAM3 integration (``Sam3Model`` /
``Sam3Processor``, contributed by yonigozlan/ronghanghu): a text concept prompt
returns instance masks + boxes + scores, so counting a concept is
``len(post_process_instance_segmentation(...)['scores'])`` above a threshold.
``facebook/sam3.1`` and ``facebook/sam3`` are HF-**gated** — the download 401s
without an access-granted token, resolved from ``paths`` (``hf_token`` /
``hf_token_env``). transformers for SAM 3 (v5.x) conflicts with the dev pin, so
this backend's dependency lives in ``requirements-gpu.txt`` and the import is
guarded + lazy; it runs only on the GPU server (mock-based tests locally,
``scripts/smoke_counter.py`` on the box).
"""

from __future__ import annotations

import logging
from pathlib import Path

from aic.config import CounterConfig
from aic.models_cache import apply_model_cache_env
from aic.vqa.counter import ConceptDetections, Detection, FrameDetections

logger = logging.getLogger(__name__)


def _chunks(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


class Sam3Counter:
    """Per-frame concept counter over the transformers SAM 3 family.

    Not a video tracker (``is_native = False``): it counts instances in each
    frame independently, so the caller reconciles a moment's frames by median
    (aic.vqa.counter.reconcile_count).
    """

    is_native = False

    def __init__(
        self, cfg: CounterConfig, models_dir: Path, hf_token: str | None
    ) -> None:
        self._cfg = cfg
        self._hf_token = hf_token
        # HF weights land under paths.models_dir (never a user-level cache).
        apply_model_cache_env(models_dir)
        self._model = None
        self._processor = None

    def _resolve_device(self) -> str:
        if self._cfg.device != "auto":
            return self._cfg.device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def _ensure_model(self):
        if self._model is not None:
            return
        import torch
        from transformers import Sam3Model, Sam3Processor

        device = self._resolve_device()
        load_kwargs: dict = {}
        if self._hf_token:
            load_kwargs["token"] = self._hf_token
        if self._cfg.image_size is not None:
            from transformers import Sam3Config

            config = Sam3Config.from_pretrained(self._cfg.model_id, **load_kwargs)
            config.image_size = self._cfg.image_size
            load_kwargs["config"] = config
            proc_size = {"height": self._cfg.image_size, "width": self._cfg.image_size}
        else:
            proc_size = None
        model = Sam3Model.from_pretrained(self._cfg.model_id, **load_kwargs)
        model = model.to(device)
        if device.startswith("cuda"):
            model = model.half()  # fp16 (the plan's VRAM figures assume it)
        model.eval()
        proc_kwargs = {k: v for k, v in load_kwargs.items() if k != "config"}
        if proc_size is not None:
            proc_kwargs["size"] = proc_size
        self._processor = Sam3Processor.from_pretrained(
            self._cfg.model_id, **proc_kwargs
        )
        self._model = model
        self._torch = torch
        self._device = device
        logger.info("loaded SAM counter %s on %s", self._cfg.model_id, device)

    def warmup(self) -> None:
        """Load the weights once (call before spawning data-parallel workers)."""
        self._ensure_model()

    @staticmethod
    def _load_frame(frame):
        from PIL import Image

        if isinstance(frame, (str, Path)):
            with Image.open(frame) as img:
                return img.convert("RGB")
        return frame.convert("RGB") if hasattr(frame, "convert") else frame

    def _detect_concept(self, images: list, concept: str) -> list[ConceptDetections]:
        """Detect one concept across a list of frames (chunked by max_batch)."""
        out: list[ConceptDetections] = []
        for chunk in _chunks(images, self._cfg.max_batch):
            inputs = self._processor(
                images=chunk, text=[concept] * len(chunk), return_tensors="pt"
            ).to(self._device)
            with self._torch.no_grad():
                outputs = self._model(**inputs)
            results = self._processor.post_process_instance_segmentation(
                outputs,
                threshold=self._cfg.confidence,
                mask_threshold=self._cfg.mask_threshold,
                target_sizes=inputs.get("original_sizes").tolist(),
            )
            for res in results:
                boxes = res["boxes"].tolist()
                scores = res["scores"].tolist()
                out.append(
                    ConceptDetections(
                        concept=concept,
                        detections=[
                            Detection(box=tuple(box), score=float(score))
                            for box, score in zip(boxes, scores, strict=True)
                        ],
                    )
                )
        return out

    def detect(self, frames: list, concepts: list[str]) -> list[FrameDetections]:
        self._ensure_model()
        images = [self._load_frame(f) for f in frames]
        # Per concept across all frames, then transpose into per-frame dicts.
        by_concept = {c: self._detect_concept(images, c) for c in concepts}
        return [
            {concept: by_concept[concept][i] for concept in concepts}
            for i in range(len(images))
        ]
