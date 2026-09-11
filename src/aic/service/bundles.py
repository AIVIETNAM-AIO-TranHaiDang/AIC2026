"""Evidence bundles: everything the operator sees about one shot.

Built by joining the Chronicle with the keyframes manifest. Bundles power
the UI result cards, the negation filter (via ``evidence_text``), and the
question planner (via the scene/text/speech attributes).
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import Field

from aic.chronicle.jobs import load_chronicle
from aic.config import Config, StrictModel
from aic.ingest.pipeline import KEYFRAMES_MANIFEST
from aic.manifest import read_manifest

logger = logging.getLogger(__name__)


class KeyframeRef(StrictModel):
    """One keyframe exposed to the KIS operator and submission queue."""

    keyframe_id: str
    frame_id: int
    timestamp_ms: int
    name: str


class EvidenceBundle(StrictModel):
    shot_key: str
    video_id: str
    shot_id: int
    t_start_ms: int
    t_end_ms: int
    caption: str | None = None
    scene: str | None = None
    actions: list[str] = Field(default_factory=list)
    ocr_lines: list[str] = Field(default_factory=list)
    asr_text: str | None = None
    frames: list[str] = Field(
        default_factory=list,
        description="Keyframe image file names (relative to the keyframes "
        "directory), timeline order.",
    )
    frame_refs: list[KeyframeRef] = Field(
        default_factory=list,
        description="Keyframe submission metadata in the same order as frames.",
    )

    @property
    def midpoint_ms(self) -> int:
        return (self.t_start_ms + self.t_end_ms) // 2

    @property
    def evidence_text(self) -> str:
        """Combined searchable text, for the negation filter."""
        parts = [self.caption or "", self.asr_text or "", *self.ocr_lines]
        parts.extend(self.actions)
        return "\n".join(p for p in parts if p)

    @property
    def has_text(self) -> bool:
        return bool(self.ocr_lines)

    @property
    def has_speech(self) -> bool:
        return bool(self.asr_text)


def load_bundles(cfg: Config) -> dict[str, EvidenceBundle]:
    """shot_key -> bundle, from chronicle.jsonl plus the keyframes manifest."""
    frames_by_shot: dict[str, list[KeyframeRef]] = {}
    for record in read_manifest(cfg.paths.manifests_dir / KEYFRAMES_MANIFEST):
        key = f"{record['video_id']}:{record['shot_id']}"
        # Keyframe images live under a per-video subdirectory of the
        # keyframes dir; the bundle stores that keyframes-dir-relative path
        # (forward slashes, so it doubles as the /frames URL path). A bare
        # file name here 404'd every console thumbnail and starved
        # VLM-verify of its frames.
        name = f"{record['video_id']}/{Path(record['image_path']).name}"
        frames_by_shot.setdefault(key, []).append(
            KeyframeRef(
                keyframe_id=str(record["keyframe_id"]),
                frame_id=int(record["frame_idx"]),
                timestamp_ms=int(record["timestamp_ms"]),
                name=name,
            )
        )

    bundles = {}
    for chron in load_chronicle(cfg):
        frame_refs = sorted(
            frames_by_shot.get(chron.shot_key, []), key=lambda ref: ref.timestamp_ms
        )
        bundles[chron.shot_key] = EvidenceBundle(
            shot_key=chron.shot_key,
            video_id=chron.video_id,
            shot_id=chron.shot_id,
            t_start_ms=chron.t_start_ms,
            t_end_ms=chron.t_end_ms,
            caption=chron.caption_en,
            scene=chron.scene,
            actions=chron.actions,
            ocr_lines=[line.text for line in chron.ocr],
            asr_text=chron.asr_text,
            frames=[ref.name for ref in frame_refs],
            frame_refs=frame_refs,
        )
    logger.info("loaded %d evidence bundles", len(bundles))
    return bundles
