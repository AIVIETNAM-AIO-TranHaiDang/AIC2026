"""Submission serialization — the single module that knows the format.

The official 2026 submission format is not yet published; when it lands,
this module is the only place that changes (note 09 Phase 8). Until then
two provisional formats cover the likely shapes: a JSON object and a CSV
line, both carrying exactly the ``(video_id, timestamp)`` contract the whole
pipeline is keyed on.
"""

from __future__ import annotations

import json


def serialize_submission(video_id: str, timestamp_ms: int, fmt: str) -> str:
    """One submission in the configured format."""
    if timestamp_ms < 0:
        raise ValueError(f"timestamp_ms must be non-negative, got {timestamp_ms}")
    if fmt == "json":
        return json.dumps(
            {"video_id": video_id, "timestamp_ms": timestamp_ms},
            ensure_ascii=False,
        )
    if fmt == "csv":
        return f"{video_id},{timestamp_ms}"
    raise ValueError(f"unknown submission format {fmt!r}")
