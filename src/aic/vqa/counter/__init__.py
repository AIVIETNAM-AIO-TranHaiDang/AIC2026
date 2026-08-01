"""Detector-counter backends for the ledger count pass + online count program.

A backend implements one interface — :meth:`Counter.detect(frames, concepts)`,
returning per-frame instance detections for each concept — so every backend is
drop-in (the note-10 model-registry discipline; no stage hardcodes one model).
The shipped, verified set is the SAM 3 family (``aic.config.COUNTER_BACKENDS``);
adding another detector/counter is a one-file drop-in registered in
``_COUNTER_FACTORIES`` after its own verify-before-use pass on GPU.

The count-reconciliation helpers here are pure (CPU-testable with fake
detections); only the backend adapters touch heavy models, import-guarded and
lazy so a box without the dependency still imports this package.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Protocol

from aic.config import CounterConfig

Box = tuple[float, float, float, float]  # xyxy, absolute pixels


@dataclass(frozen=True)
class Detection:
    """One detected instance of a concept in one frame."""

    box: Box
    score: float


@dataclass(frozen=True)
class ConceptDetections:
    """Every instance of one concept found in one frame."""

    concept: str
    detections: list[Detection] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.detections)

    @property
    def mean_confidence(self) -> float:
        if not self.detections:
            return 0.0
        return sum(d.score for d in self.detections) / len(self.detections)

    def union_box(self) -> Box | None:
        """The bounding box enclosing every instance (the zoom crop), or None."""
        if not self.detections:
            return None
        xs0 = [d.box[0] for d in self.detections]
        ys0 = [d.box[1] for d in self.detections]
        xs1 = [d.box[2] for d in self.detections]
        ys1 = [d.box[3] for d in self.detections]
        return (min(xs0), min(ys0), max(xs1), max(ys1))


# One frame's detections for several concepts: concept -> its instances.
FrameDetections = dict[str, ConceptDetections]


class Counter(Protocol):
    def detect(
        self, frames: list, concepts: list[str]
    ) -> list[FrameDetections]:
        """Detect every instance of each concept in each frame.

        ``frames`` are PIL images or image paths; returns one
        :data:`FrameDetections` per input frame, in order. A per-frame backend
        returns independent per-frame counts (reconciled by the caller); a
        tracking backend (``is_native``) already returns unique-across-frames
        counts.

        Contract: every returned :data:`FrameDetections` MUST carry an entry
        for EVERY requested concept, with an empty detection list when the
        concept is absent in that frame. Reconciliation medians over the
        entries present, so an omitted zero-count frame would silently bias
        the count upward.
        """
        ...

    @property
    def is_native(self) -> bool:
        """True when the backend counts unique instances across frames itself
        (a video tracker), so the caller must NOT median across frames."""
        ...


def reconcile_count(per_frame_counts: list[int], is_native: bool) -> int:
    """One count for a moment from its per-frame counts (note 18 §3, risk 4.3).

    Per-frame backends (e.g. SAM 3) see the SAME instances in every keyframe of
    a shot, so summing double-counts; the median across frames is the robust
    per-moment count. A native tracking backend already returns unique instances
    per frame, so the shot's count is the peak (max) it reports, not a median.
    Empty input is 0.
    """
    if not per_frame_counts:
        return 0
    if is_native:
        return max(per_frame_counts)
    return int(round(median(per_frame_counts)))


def reconcile_objects(
    per_frame: list[FrameDetections], min_count: int = 1
) -> dict[str, int]:
    """Reconciled per-concept count across a moment's frames.

    Concepts whose reconciled count is below ``min_count`` are dropped, so
    ``salient_objects`` lists only concepts actually present in the moment.
    """
    concepts: dict[str, list[int]] = {}
    for frame in per_frame:
        for concept, dets in frame.items():
            concepts.setdefault(concept, []).append(dets.count)
    counts = {
        concept: reconcile_count(per_frame_counts, is_native=False)
        for concept, per_frame_counts in concepts.items()
    }
    return {c: n for c, n in counts.items() if n >= min_count}


# Backend registry: name -> factory(cfg, models_dir, hf_token) -> Counter.
# Only verified adapters appear here; CounterConfig.backend is validated against
# aic.config.COUNTER_BACKENDS, so an unregistered name never reaches this map.
def _build_sam(cfg: CounterConfig, models_dir: Path, hf_token: str | None) -> Counter:
    from aic.vqa.counter.sam import Sam3Counter

    return Sam3Counter(cfg, models_dir, hf_token)


_COUNTER_FACTORIES: dict[str, Callable[..., Counter]] = {
    "sam3.1": _build_sam,
    "sam3": _build_sam,
}


def build_counter(
    cfg: CounterConfig, models_dir: Path, hf_token: str | None
) -> Counter:
    """Build the configured detector-counter backend (weights load lazily)."""
    factory = _COUNTER_FACTORIES.get(cfg.backend)
    if factory is None:  # defensive: config validation should prevent this
        raise ValueError(
            f"no counter adapter for backend {cfg.backend!r}; "
            f"registered: {sorted(_COUNTER_FACTORIES)}"
        )
    return factory(cfg, models_dir, hf_token)
