"""Shot boundary detection behind a registry interface.

The detector consumes frames that the caller already decoded (one decode pass
is shared between detection and motion analysis) and returns PTS-anchored
shots. Each detector declares the frame geometry it needs via ``frame_size``
so the pipeline decodes once at the right resolution. Implementations are
selected by config name via :func:`build_shot_detector`, which is what makes
tier upgrades config-level.

Backends:

- ``transnetv2`` — the T0 default (transnetv2-pytorch 1.0.5, bundled
  weights, runs on CPU).
- ``omnishotcut`` — the 2026 SBD SOTA (shot-query transformer). Verified
  against the upstream source (UVA-Computer-Vision-Lab/OmniShotCut):
  ``omnishotcut.load(checkpoint_path, filename=...)`` downloads from the HF
  hub when given a repo id, and ``model.inference(video, mode=...,
  overlap=...)`` accepts a ``(T, H, W, 3)`` uint8 array already sized to the
  checkpoint's ``process_height``/``process_width`` (the "val" transform
  normalises but does not resize). The two documented modes differ in
  return shape: ``"clean_shot"`` returns ``[start, end]`` frame ranges for
  non-transition shots only, while ``"default"`` returns a
  ``(ranges, intra_labels, inter_labels)`` tuple covering every detected
  shot including gradual transitions. Upstream hard-codes CUDA, so this
  backend is GPU-only; install per requirements-gpu.txt.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from aic.config import ShotDetectionConfig

logger = logging.getLogger(__name__)

# Input geometry required by TransNetV2 (height, width, channels).
TRANSNET_FRAME_HEIGHT = 27
TRANSNET_FRAME_WIDTH = 48


@dataclass(frozen=True)
class Shot:
    shot_id: int
    start_frame: int
    end_frame: int
    start_ms: int
    end_ms: int
    boundary_confidence: float
    """Transition probability at the cut that opened this shot.

    1.0 for the first shot of a video. Low values flag gradual transitions
    worth human inspection (Phase 2 edge case in note 09).
    """


class ShotDetector(Protocol):
    @property
    def frame_size(self) -> tuple[int, int]:
        """``(width, height)`` the detector needs its input frames in."""
        ...

    def detect(self, frames: np.ndarray, pts_ms: list[int]) -> list[Shot]:
        """Split ``frames`` (uint8, [N, H, W, 3] at ``frame_size``) into shots.

        ``pts_ms[i]`` is the timestamp of ``frames[i]``; shots carry both frame
        indices and timestamps so downstream stages never re-derive time.
        """
        ...


class TransNetV2Detector:
    """TransNetV2 (transnetv2-pytorch) adapter.

    The model is loaded lazily on first use so that constructing the detector
    (e.g. in tests that never call it) costs nothing. Verified against
    transnetv2-pytorch 1.0.5 source: ``TransNetV2(device=...)`` resolves
    'auto' itself and moves the weights, but ``predict_frames`` — unlike its
    ``predict_video`` — does NOT move the input tensor, so this adapter
    moves frames to ``model.device`` (mixed devices otherwise fail inside
    conv3d on GPU machines).
    """

    def __init__(
        self, threshold: float, min_shot_duration_ms: int, device: str
    ) -> None:
        self._threshold = threshold
        self._min_shot_duration_ms = min_shot_duration_ms
        self._device = device
        self._model = None

    @property
    def frame_size(self) -> tuple[int, int]:
        return (TRANSNET_FRAME_WIDTH, TRANSNET_FRAME_HEIGHT)

    def _ensure_model(self):
        if self._model is None:
            import torch  # deferred: torch import is slow and GPU-touching
            from transnetv2_pytorch import TransNetV2

            self._model = TransNetV2(device=self._device)
            self._model.eval()
            self._torch = torch
        return self._model

    def detect(self, frames: np.ndarray, pts_ms: list[int]) -> list[Shot]:
        expected = (TRANSNET_FRAME_HEIGHT, TRANSNET_FRAME_WIDTH, 3)
        if frames.ndim != 4 or frames.shape[1:] != expected:
            raise ValueError(
                f"TransNetV2 expects frames [N, {expected[0]}, {expected[1]}, 3], "
                f"got {frames.shape}"
            )
        if frames.dtype != np.uint8:
            raise ValueError(f"frames must be uint8, got {frames.dtype}")
        if len(pts_ms) != len(frames):
            raise ValueError(
                f"pts_ms has {len(pts_ms)} entries for {len(frames)} frames"
            )
        if len(frames) == 0:
            return []

        model = self._ensure_model()
        # model.device is the torch.device the package resolved (it accepts
        # 'auto'); predict_frames does not move inputs, so we must.
        tensor = self._torch.from_numpy(np.ascontiguousarray(frames)).to(
            model.device
        )
        with self._torch.no_grad():
            single_frame_pred, _ = model.predict_frames(tensor, quiet=True)
        probabilities = single_frame_pred.cpu().numpy()
        scene_ranges = model.predictions_to_scenes(
            probabilities, threshold=self._threshold
        )
        shots = self._ranges_to_shots(scene_ranges, probabilities, pts_ms)
        return _merge_short_shots(shots, self._min_shot_duration_ms)

    @staticmethod
    def _ranges_to_shots(
        scene_ranges: np.ndarray,
        probabilities: np.ndarray,
        pts_ms: list[int],
    ) -> list[Shot]:
        shots = []
        for shot_id, (start_frame, end_frame) in enumerate(scene_ranges.tolist()):
            confidence = (
                1.0 if start_frame == 0 else float(probabilities[start_frame - 1])
            )
            shots.append(
                Shot(
                    shot_id=shot_id,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    start_ms=pts_ms[start_frame],
                    end_ms=pts_ms[end_frame],
                    boundary_confidence=confidence,
                )
            )
        return shots


def _merge_short_shots(shots: list[Shot], min_duration_ms: int) -> list[Shot]:
    """Merge shots shorter than the minimum into their predecessor.

    Rapid montage cuts otherwise explode the keyframe count (Phase 2 edge
    case). The first shot is never dropped: if it is short it absorbs its
    successor instead.
    """
    if min_duration_ms <= 0 or len(shots) <= 1:
        return shots
    merged: list[Shot] = []
    for shot in shots:
        duration = shot.end_ms - shot.start_ms
        if merged and duration < min_duration_ms:
            previous = merged[-1]
            merged[-1] = Shot(
                shot_id=previous.shot_id,
                start_frame=previous.start_frame,
                end_frame=shot.end_frame,
                start_ms=previous.start_ms,
                end_ms=shot.end_ms,
                boundary_confidence=previous.boundary_confidence,
            )
        else:
            merged.append(shot)
    # A short first shot absorbs forward; re-check once the list is built.
    if len(merged) > 1 and merged[0].end_ms - merged[0].start_ms < min_duration_ms:
        first, second = merged[0], merged[1]
        merged[:2] = [
            Shot(
                shot_id=0,
                start_frame=first.start_frame,
                end_frame=second.end_frame,
                start_ms=first.start_ms,
                end_ms=second.end_ms,
                boundary_confidence=first.boundary_confidence,
            )
        ]
    return [
        Shot(
            shot_id=index,
            start_frame=shot.start_frame,
            end_frame=shot.end_frame,
            start_ms=shot.start_ms,
            end_ms=shot.end_ms,
            boundary_confidence=shot.boundary_confidence,
        )
        for index, shot in enumerate(merged)
    ]


class OmniShotCutDetector:
    """OmniShotCut adapter (GPU-only, see the module docstring).

    The inference mode comes from config: ``clean_shot`` keeps
    non-transition shots only, so dissolve/wipe frames fall between shots
    and never become keyframes; ``default`` keeps every detected shot,
    transitions included, so no frame span is unreachable. The model
    exposes no per-cut probability, so ``boundary_confidence`` is always
    1.0 for this backend.
    """

    def __init__(self, cfg: ShotDetectionConfig) -> None:
        self._cfg = cfg
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "OmniShotCut requires CUDA (upstream loads the model on "
                    "'cuda' unconditionally); use shots.model: transnetv2 on "
                    "CPU machines"
                )
            import omnishotcut

            if self._cfg.checkpoint_filename:
                self._model = omnishotcut.load(
                    self._cfg.checkpoint, filename=self._cfg.checkpoint_filename
                )
            else:
                self._model = omnishotcut.load(self._cfg.checkpoint)
            logger.info("loaded OmniShotCut from %s", self._cfg.checkpoint)
        return self._model

    @property
    def frame_size(self) -> tuple[int, int]:
        # The checkpoint's model_args carry the trained input geometry; the
        # inference transform normalises but does not resize, so frames must
        # be decoded at exactly this size (verified against upstream source,
        # pinned commit in requirements-gpu.txt).
        args = self._ensure_model()._model_args
        return (args.process_width, args.process_height)

    def detect(self, frames: np.ndarray, pts_ms: list[int]) -> list[Shot]:
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(
                f"OmniShotCut expects frames [N, H, W, 3], got {frames.shape}"
            )
        if frames.dtype != np.uint8:
            raise ValueError(f"frames must be uint8, got {frames.dtype}")
        if len(pts_ms) != len(frames):
            raise ValueError(
                f"pts_ms has {len(pts_ms)} entries for {len(frames)} frames"
            )
        if len(frames) == 0:
            return []
        model = self._ensure_model()
        result = model.inference(
            frames, mode=self._cfg.mode, overlap=self._cfg.overlap_frames
        )
        if self._cfg.mode == "clean_shot":
            ranges = result
        else:
            # "default" mode also returns per-shot transition labels, which
            # the Shot schema does not carry; every range becomes a shot.
            ranges, _intra_labels, _inter_labels = result
        last = len(pts_ms) - 1
        shots = []
        for shot_id, (start_frame, end_frame) in enumerate(ranges):
            start_frame = max(0, min(int(start_frame), last))
            end_frame = max(start_frame, min(int(end_frame), last))
            shots.append(
                Shot(
                    shot_id=shot_id,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    start_ms=pts_ms[start_frame],
                    end_ms=pts_ms[end_frame],
                    boundary_confidence=1.0,
                )
            )
        if not shots:
            # clean_shot can drop everything on transition-only footage; a
            # single whole-video shot keeps the video retrievable.
            logger.warning("OmniShotCut returned no shots; using one full shot")
            shots = [
                Shot(
                    shot_id=0,
                    start_frame=0,
                    end_frame=last,
                    start_ms=pts_ms[0],
                    end_ms=pts_ms[last],
                    boundary_confidence=0.0,
                )
            ]
        return _merge_short_shots(shots, self._cfg.min_shot_duration_ms)


_DETECTORS = {
    "transnetv2": lambda cfg: TransNetV2Detector(
        threshold=cfg.threshold,
        min_shot_duration_ms=cfg.min_shot_duration_ms,
        device=cfg.device,
    ),
    "omnishotcut": OmniShotCutDetector,
}


def build_shot_detector(cfg: ShotDetectionConfig) -> ShotDetector:
    """Instantiate the configured shot detector by registry name."""
    try:
        factory = _DETECTORS[cfg.model]
    except KeyError:
        known = ", ".join(sorted(_DETECTORS))
        raise ValueError(
            f"unknown shot detector {cfg.model!r}; known detectors: {known}"
        ) from None
    return factory(cfg)
