"""Video probing and decoding on top of PyAV.

Timestamps always come from stream PTS (converted to integer milliseconds by
:func:`aic.ids.pts_to_ms`), never from ``frame_index / fps`` arithmetic, which
drifts on the variable-frame-rate footage that broadcast recordings often are.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np

from aic.ids import pts_to_ms, video_id_from_path

logger = logging.getLogger(__name__)

# Number of leading inter-frame gaps inspected to flag variable frame rate;
# spread of gaps beyond this fraction of their median marks the video VFR.
_VFR_PROBE_FRAMES = 120
_VFR_SPREAD_FRACTION = 0.1


class VideoError(RuntimeError):
    """Raised when a video cannot be opened or decoded."""


@dataclass(frozen=True)
class VideoInfo:
    video_id: str
    path: str
    duration_ms: int
    width: int
    height: int
    fps_average: float
    has_audio: bool
    is_vfr: bool


def probe_video(path: Path | str) -> VideoInfo:
    """Open a video, read its stream properties, and sniff for VFR timing."""
    path = Path(path)
    video_id = video_id_from_path(path)
    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise VideoError(f"{path}: no video stream")
            stream = container.streams.video[0]
            time_base = stream.time_base
            if time_base is None:
                raise VideoError(f"{path}: video stream has no time base")

            gaps: list[int] = []
            last_pts: int | None = None
            first_pts: int | None = None
            end_ms = 0
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                if first_pts is None:
                    first_pts = frame.pts
                if last_pts is not None:
                    gaps.append(frame.pts - last_pts)
                last_pts = frame.pts
                end_ms = pts_to_ms(frame.pts, time_base)
                if len(gaps) >= _VFR_PROBE_FRAMES:
                    break
            if first_pts is None:
                raise VideoError(f"{path}: could not decode any frame")

            is_vfr = False
            if gaps:
                median_gap = sorted(gaps)[len(gaps) // 2]
                spread = max(gaps) - min(gaps)
                is_vfr = median_gap > 0 and spread > median_gap * _VFR_SPREAD_FRACTION

            if stream.duration is not None:
                duration_ms = pts_to_ms(stream.duration, time_base)
            elif container.duration is not None:
                duration_ms = round(container.duration / av.time_base * 1000)
            else:
                duration_ms = end_ms
            return VideoInfo(
                video_id=video_id,
                path=str(path),
                duration_ms=duration_ms,
                width=stream.codec_context.width,
                height=stream.codec_context.height,
                fps_average=float(stream.average_rate or 0.0),
                has_audio=bool(container.streams.audio),
                is_vfr=is_vfr,
            )
    except av.error.FFmpegError as exc:
        raise VideoError(f"{path}: {exc}") from exc


def iter_frames(
    path: Path | str,
    width: int,
    height: int,
) -> Iterator[tuple[int, np.ndarray]]:
    """Decode every frame resized to ``width x height`` RGB.

    Yields ``(timestamp_ms, frame)`` with frames as uint8 arrays of shape
    ``(height, width, 3)``. Frames without a PTS are skipped (they cannot be
    addressed by the ID contract).
    """
    path = Path(path)
    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise VideoError(f"{path}: no video stream")
            stream = container.streams.video[0]
            time_base = stream.time_base
            if time_base is None:
                raise VideoError(f"{path}: video stream has no time base")
            for frame in container.decode(stream):
                if frame.pts is None:
                    logger.warning("%s: skipping frame without PTS", path)
                    continue
                rgb = frame.reformat(width=width, height=height, format="rgb24")
                yield pts_to_ms(frame.pts, time_base), rgb.to_ndarray()
    except av.error.FFmpegError as exc:
        raise VideoError(f"{path}: {exc}") from exc


def _frame_to_array(frame: av.VideoFrame, out_w: int, out_h: int) -> np.ndarray:
    """Resize one decoded frame to ``out_w x out_h`` RGB and return the array.

    Isolated so the extraction loop resizes each chosen frame exactly once
    (and so a test can count how often that happens).
    """
    return frame.reformat(width=out_w, height=out_h, format="rgb24").to_ndarray()


def extract_frames_at(
    path: Path | str,
    targets_ms: list[int],
    max_side: int,
) -> dict[int, np.ndarray]:
    """Extract the frame nearest to each target timestamp, at display size.

    Single sequential decode with a two-pointer sweep: frames arrive in
    presentation (PTS-ascending) order and the targets are sorted, so the frame
    nearest a target is one of the two that straddle it. We buffer just the
    previous frame and, the moment decoding passes a target, pick the closer of
    the two and resize it once. Cost is O(frames) to decode plus O(targets)
    resizes — never the O(frames x targets) of resizing on every improvement,
    which made long videos quadratic in length.
    """
    path = Path(path)
    if not targets_ms:
        return {}
    targets = sorted(set(targets_ms))
    result: dict[int, np.ndarray] = {}
    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise VideoError(f"{path}: no video stream")
            stream = container.streams.video[0]
            time_base = stream.time_base
            if time_base is None:
                raise VideoError(f"{path}: video stream has no time base")
            src_w = stream.codec_context.width
            src_h = stream.codec_context.height
            scale = min(max_side / max(src_w, src_h), 1.0)
            # Even dimensions keep every downstream codec and library happy.
            out_w = max(2, int(src_w * scale) // 2 * 2)
            out_h = max(2, int(src_h * scale) // 2 * 2)

            ti = 0
            prev_frame = None
            prev_ts = 0
            prev_arr: np.ndarray | None = None  # memoized resize of prev_frame
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                ts = pts_to_ms(frame.pts, time_base)
                cur_arr: np.ndarray | None = None  # resized once, if this frame wins
                # Resolve every target this frame has now reached or passed.
                while ti < len(targets) and ts >= targets[ti]:
                    target = targets[ti]
                    if prev_frame is not None and abs(prev_ts - target) <= abs(
                        ts - target
                    ):
                        if prev_arr is None:
                            prev_arr = _frame_to_array(prev_frame, out_w, out_h)
                        result[target] = prev_arr
                    else:
                        if cur_arr is None:
                            cur_arr = _frame_to_array(frame, out_w, out_h)
                        result[target] = cur_arr
                    ti += 1
                if ti >= len(targets):
                    break
                prev_frame, prev_ts, prev_arr = frame, ts, cur_arr
            # Targets beyond the last decoded frame take that last frame.
            if ti < len(targets) and prev_frame is not None:
                if prev_arr is None:
                    prev_arr = _frame_to_array(prev_frame, out_w, out_h)
                while ti < len(targets):
                    result[targets[ti]] = prev_arr
                    ti += 1
    except av.error.FFmpegError as exc:
        raise VideoError(f"{path}: {exc}") from exc
    return result


def decode_audio_mono(path: Path | str, rate: int) -> np.ndarray:
    """Decode the first audio track as mono float32 at ``rate`` Hz.

    Returns an empty array when the file has no audio stream (a legal corpus
    state). The resampler flush call (``resample(None)``) drains buffered
    samples; the whole path is exercised by a live round-trip test.
    """
    path = Path(path)
    try:
        with av.open(str(path)) as container:
            if not container.streams.audio:
                return np.zeros(0, dtype=np.float32)
            resampler = av.AudioResampler(format="flt", layout="mono", rate=rate)
            chunks: list[np.ndarray] = []
            for frame in container.decode(audio=0):
                for resampled in resampler.resample(frame):
                    chunks.append(resampled.to_ndarray().reshape(-1))
            for resampled in resampler.resample(None):
                chunks.append(resampled.to_ndarray().reshape(-1))
    except av.error.FFmpegError as exc:
        raise VideoError(f"{path}: {exc}") from exc
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32)
