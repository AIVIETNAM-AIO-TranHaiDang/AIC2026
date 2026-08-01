"""The ID contract shared by every pipeline stage.

Everything in the system is keyed by ``(video_id, shot_id, keyframe_id,
timestamp_ms)``:

- ``video_id`` is the video file stem (no extension, no directories),
- ``shot_id`` is a zero-based shot index within the video,
- ``keyframe_id`` is globally unique: ``{video_id}_s{shot_id}_f{frame_idx}``,
- timestamps are integer milliseconds derived from stream PTS, never from
  ``frame_idx / fps`` arithmetic (which drifts on variable-frame-rate video).

See docs/note/09-kis-implementation-plan.md, "Cross-cutting engineering
contracts".
"""

from __future__ import annotations

import math
import re
from fractions import Fraction
from pathlib import Path

_VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_KEYFRAME_ID_PATTERN = re.compile(
    r"^(?P<video_id>[A-Za-z0-9][A-Za-z0-9._-]*)_s(?P<shot_id>\d+)_f(?P<frame_idx>\d+)$"
)


class IdError(ValueError):
    """Raised when an identifier violates the ID contract."""


def video_id_from_path(path: Path | str) -> str:
    """Derive and validate the video id from a file path."""
    stem = Path(path).stem
    if not _VIDEO_ID_PATTERN.match(stem):
        raise IdError(
            f"video file stem {stem!r} contains characters outside the ID contract "
            "(allowed: letters, digits, '.', '_', '-'; must not start with a symbol)"
        )
    return stem


def make_keyframe_id(video_id: str, shot_id: int, frame_idx: int) -> str:
    """Build the globally unique keyframe id."""
    if not _VIDEO_ID_PATTERN.match(video_id):
        raise IdError(f"invalid video_id {video_id!r}")
    if shot_id < 0 or frame_idx < 0:
        raise IdError("shot_id and frame_idx must be non-negative")
    return f"{video_id}_s{shot_id}_f{frame_idx}"


def parse_keyframe_id(keyframe_id: str) -> tuple[str, int, int]:
    """Split a keyframe id back into ``(video_id, shot_id, frame_idx)``."""
    match = _KEYFRAME_ID_PATTERN.match(keyframe_id)
    if match is None:
        raise IdError(f"invalid keyframe_id {keyframe_id!r}")
    return (
        match.group("video_id"),
        int(match.group("shot_id")),
        int(match.group("frame_idx")),
    )


def pts_to_ms(pts: int, time_base: Fraction) -> int:
    """Convert a stream PTS value to integer milliseconds.

    Half-up rounding (not truncation, not banker's rounding) keeps the maximum
    error at half a millisecond and is deterministic on exact .5 values, which
    exact Fraction arithmetic does produce for common time bases.
    """
    return math.floor(pts * time_base * 1000 + Fraction(1, 2))
