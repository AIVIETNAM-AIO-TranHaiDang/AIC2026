"""The online count program (note 18 phase 4, §3): detector-first counting.

A ``count`` question routes here instead of trusting a VLM read. The located
moment's keyframes go to the detector-counter for the question's concept; the
per-frame instance counts are reconciled to one per-moment number (median, so
the same person in four keyframes is one person, not four — risk 4.3). A VLM
read gives an independent estimate: when it lands within ``adjudicate_margin``
of the detector it re-reads the crop (the union of the concept's boxes) as the
adjudicator; beyond the margin the detector wins (note 17 §4b). With no counter
configured the program returns None and the caller degrades to the plain read.

The decision, reconciliation, and crop maths are pure (CPU-testable with a fake
counter + fake reader); only the detector and the reader touch models.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from aic.config import CounterConfig, VqaConfig
from aic.vqa.counter import (
    Box,
    ConceptDetections,
    Counter,
    FrameDetections,
    reconcile_count,
)

logger = logging.getLogger(__name__)

# A reader that answers "how many <concept>?" over some frames (optionally
# cropped to a box in absolute pixels), returning an integer or None on
# failure/abstention. The production impl is aic.vqa.reader.QaReader.read_count;
# tests inject a fake.
CountReader = Callable[[str, str, list[str], "Box | None"], "int | None"]


@dataclass(frozen=True)
class CountResult:
    concept: str
    count: int
    detector_count: int
    reader_count: int | None
    source: str  # "detector" or "adjudicated"
    video_id: str
    timestamp_ms: int

    def provenance(self) -> dict:
        note = f"detector: {self.detector_count}"
        if self.reader_count is not None:
            note += f", reader: {self.reader_count}"
        note += f" — {self.source}: {self.count}"
        return {"kind": "count", "concept": self.concept, "note": note}


def resolve_concept(plan, counter_cfg: CounterConfig) -> str:
    """The concept to count: the compiler's, else the person concept fallback."""
    return plan.concept or counter_cfg.person_concept


def union_box(per_frame: list[FrameDetections], concept: str) -> Box | None:
    """The box enclosing every detected instance of the concept across frames."""
    boxes = [
        frame[concept].union_box()
        for frame in per_frame
        if concept in frame and frame[concept].union_box() is not None
    ]
    if not boxes:
        return None
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def pad_box(box: Box, pad_frac: float) -> Box:
    """Expand a box by a fraction of its size on every side (for a zoom crop).

    Coordinates may go slightly out of the image; the reader clamps to each
    frame's real size when it crops, so this stays image-size-agnostic.
    """
    x0, y0, x1, y1 = box
    dx = (x1 - x0) * pad_frac
    dy = (y1 - y0) * pad_frac
    return (x0 - dx, y0 - dy, x1 + dx, y1 + dy)


def adjudicate_within_margin(
    detector_count: int, reader_count: int | None, margin: int
) -> bool:
    """Whether the VLM should adjudicate: it has a count within the margin."""
    return reader_count is not None and abs(detector_count - reader_count) <= margin


def apply_score_floor(
    per_frame: list[FrameDetections], floor: float
) -> list[FrameDetections]:
    """Drop detections scoring below the query-time floor (vqa.count).

    Applied before reconciliation AND before the crop box is drawn, so a
    tightened floor changes both the count and the zoom. A floor of 0 returns
    the input unchanged (every detection already passed the backend's own
    confidence threshold).
    """
    if floor <= 0.0:
        return per_frame
    return [
        {
            concept: ConceptDetections(
                concept=concept,
                detections=[d for d in dets.detections if d.score >= floor],
            )
            for concept, dets in frame.items()
        }
        for frame in per_frame
    ]


class CountProgram:
    """Detector-first counting for one located moment (degrades to None)."""

    def __init__(
        self,
        counter: Counter | None,
        count_reader: CountReader | None,
        vqa_cfg: VqaConfig,
        counter_cfg: CounterConfig,
        frames_dir: Path | None,
    ) -> None:
        self._counter = counter
        self._count_reader = count_reader
        self._vqa_cfg = vqa_cfg
        self._counter_cfg = counter_cfg
        self._frames_dir = frames_dir

    @property
    def available(self) -> bool:
        return self._counter is not None

    def _frame_paths(self, frame_names: list[str]) -> list[str]:
        if self._frames_dir is None:
            return list(frame_names)
        return [str(self._frames_dir / name) for name in frame_names]

    def count(self, plan, pack, video_id: str) -> CountResult | None:
        """Count the concept in one moment; None when no counter or no frames.

        Never raises: a detector or reader failure is logged and the moment is
        reported with whatever count survived (degrade-to-partial).
        """
        if self._counter is None:
            return None
        concept = resolve_concept(plan, self._counter_cfg)
        frame_paths = self._frame_paths(pack.frames)
        if not frame_paths:
            return None
        try:
            per_frame = self._counter.detect(frame_paths, [concept])
        except Exception as exc:  # noqa: BLE001 - degrade, never break the route
            logger.warning("counter failed for %s: %s", pack.shot_key, exc)
            return None
        per_frame = apply_score_floor(
            per_frame, self._vqa_cfg.count.min_mask_confidence
        )
        counts = [
            frame[concept].count for frame in per_frame if concept in frame
        ]
        detector_count = reconcile_count(counts, self._counter.is_native)

        reader_count = None
        source = "detector"
        final = detector_count
        if self._count_reader is not None:
            reader_count = self._safe_read(concept, plan, frame_paths, None)
            margin = self._vqa_cfg.count.adjudicate_margin
            if adjudicate_within_margin(detector_count, reader_count, margin):
                crop = union_box(per_frame, concept)
                pad = self._vqa_cfg.count.crop_pad_frac
                crop = pad_box(crop, pad) if crop else None
                adjudicated = self._safe_read(concept, plan, frame_paths, crop)
                if adjudicated is not None:
                    final, source = adjudicated, "adjudicated"

        return CountResult(
            concept=concept,
            count=final,
            detector_count=detector_count,
            reader_count=reader_count,
            source=source,
            video_id=video_id,
            timestamp_ms=pack.grounded_timestamp_ms,
        )

    def _safe_read(self, concept, plan, frame_paths, crop) -> int | None:
        try:
            return self._count_reader(concept, plan.question_core, frame_paths, crop)
        except Exception as exc:  # noqa: BLE001 - the reader is only corroboration
            logger.warning("count reader failed: %s", exc)
            return None
