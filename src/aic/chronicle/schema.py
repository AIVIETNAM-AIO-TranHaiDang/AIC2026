"""The Chronicle record: one structured, validated row per shot.

A row with missing caption fields is legal ("degraded"): the system then falls
back to OCR/ASR channels for that shot and is never worse than the note-06
baseline. Entity fields exist in the schema but are populated by a later
iteration (the entity/face axis is deferred; see implementation notes).
"""

from __future__ import annotations

import unicodedata

from pydantic import Field

from aic.config import StrictModel


class OcrLine(StrictModel):
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: list[float] | None = Field(
        default=None,
        min_length=4,
        max_length=4,
        description="Normalised [x0, y0, x1, y1] box of the line in the "
        "frame (0-1). None on legacy rows and backends that do not return "
        "geometry; the ledger's reading-order sort falls back to manifest "
        "order when it is absent (never wrong, only un-sorted). Exactly four "
        "values — the reading-order sort unpacks it.",
    )


class Entities(StrictModel):
    persons: list[str] = Field(default_factory=list)
    orgs: list[str] = Field(default_factory=list)
    places: list[str] = Field(default_factory=list)


class ChronicleRecord(StrictModel):
    video_id: str
    shot_id: int = Field(ge=0)
    t_start_ms: int = Field(ge=0)
    t_end_ms: int = Field(ge=0)
    caption_en: str | None = None
    caption_vi: str | None = None
    actions: list[str] = Field(default_factory=list)
    scene: str | None = None
    caption_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    entities: Entities = Field(default_factory=Entities)
    ocr: list[OcrLine] = Field(default_factory=list)
    overlay_ocr: list[OcrLine] = Field(
        default_factory=list,
        description="Persistent-overlay lines (watermarks, tickers, program "
        "bugs) split out of the per-shot OCR by recurrence "
        "(chronicle.overlay_min_recurrence). Identical across a video's "
        "rows; indexed per VIDEO as the retrieval video prior, never as "
        "shot evidence.",
    )
    asr_text: str | None = None

    @property
    def shot_key(self) -> str:
        return f"{self.video_id}:{self.shot_id}"

    @property
    def degraded(self) -> bool:
        """True when the shot has no caption (OCR/ASR-only retrieval)."""
        return self.caption_en is None

    def semantic_text(self) -> str:
        """Text for the dense (meaning) channel: captions plus speech."""
        parts = [self.caption_vi, self.caption_en, self.asr_text]
        parts.extend(self.actions)
        if self.scene:
            parts.append(self.scene)
        return "\n".join(p for p in parts if p)

    def literal_text(self) -> str:
        """Text for the sparse (exact-match) channel: on-screen text + speech.

        Overlay lines are deliberately absent: a watermark matching every
        shot of its video is noise at shot granularity (it enters retrieval
        through the per-video overlay index instead).
        """
        parts = [line.text for line in self.ocr]
        if self.asr_text:
            parts.append(self.asr_text)
        return "\n".join(p for p in parts if p)

    def overlay_text(self) -> str:
        """Text for the per-video overlay document (the video prior)."""
        return "\n".join(line.text for line in self.overlay_ocr if line.text)


def normalize_vietnamese(text: str) -> str:
    """NFC-normalise so Vietnamese diacritics compare consistently.

    OCR backends emit a mix of composed and decomposed Unicode; without one
    canonical form, identical on-screen strings fail to merge or match.
    """
    return unicodedata.normalize("NFC", text).strip()
