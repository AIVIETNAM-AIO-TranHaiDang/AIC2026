"""Adaptive keyframe sampling within detected shots.

Three mechanisms combine (all tunable via config, see note 09 Phase 2):

- base fractional positions inside every shot (start / middles / end),
- gap filling so no two sampled frames are further apart than ``max_gap_ms``
  (a two-minute talking-head shot must not hide a mid-shot event),
- one extra frame at the strongest motion peak of long, high-motion shots,
  computed from the low-res frames already decoded for shot detection.
"""

from __future__ import annotations

import numpy as np

from aic.config import KeyframeConfig
from aic.ingest.shots import Shot


def plan_keyframes(
    shot: Shot,
    pts_ms: list[int],
    lowres_frames: np.ndarray,
    cfg: KeyframeConfig,
) -> list[int]:
    """Choose frame indices to keep for one shot, sorted and de-duplicated."""
    if shot.end_frame >= len(pts_ms):
        raise ValueError(
            f"shot {shot.shot_id} ends at frame {shot.end_frame} but only "
            f"{len(pts_ms)} timestamps are known"
        )
    span = shot.end_frame - shot.start_frame
    chosen = {shot.start_frame + round(position * span) for position in cfg.positions}
    chosen.update(_fill_gaps(sorted(chosen), pts_ms, cfg.max_gap_ms))

    if cfg.motion_min_diff > 0:
        duration = shot.end_ms - shot.start_ms
        if duration >= cfg.motion_min_shot_ms:
            peak = _motion_peak(
                lowres_frames, shot.start_frame, shot.end_frame, cfg.motion_min_diff
            )
            if peak is not None:
                chosen.add(peak)
    return sorted(chosen)


def _fill_gaps(
    sorted_frames: list[int], pts_ms: list[int], max_gap_ms: int
) -> set[int]:
    """Insert evenly spaced frames wherever consecutive picks are too far apart."""
    extras: set[int] = set()
    for left, right in zip(sorted_frames, sorted_frames[1:], strict=False):
        gap_ms = pts_ms[right] - pts_ms[left]
        if gap_ms <= max_gap_ms:
            continue
        n_extra = gap_ms // max_gap_ms
        for i in range(1, int(n_extra) + 1):
            frame = left + round(i * (right - left) / (n_extra + 1))
            if left < frame < right:
                extras.add(frame)
    return extras


def _motion_peak(
    lowres_frames: np.ndarray,
    start_frame: int,
    end_frame: int,
    min_diff: float,
) -> int | None:
    """Frame index of the strongest inter-frame change inside the shot.

    Uses mean absolute difference of consecutive low-res frames (0-255 scale).
    Returns None when the peak is below ``min_diff`` — a static shot earns no
    extra keyframe.
    """
    if end_frame - start_frame < 2:
        return None
    window = lowres_frames[start_frame : end_frame + 1].astype(np.int16)
    diffs = np.abs(np.diff(window, axis=0)).mean(axis=(1, 2, 3))
    peak_offset = int(np.argmax(diffs))
    if float(diffs[peak_offset]) < min_diff:
        return None
    return start_frame + peak_offset + 1
