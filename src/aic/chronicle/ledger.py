"""The Answer Ledger: one typed factoid row per shot (note 18 §2).

A pure derivation over the assembled Chronicle — the expensive NER lives in the
separate resumable ``entities`` stage, so the ledger uses the assembler pattern
(rebuild ``ledger.jsonl`` whole each run). Phase 1 populates the two free-tier
text fields, ``screen_text`` (scene OCR sorted into human reading order, the
ViTextVQA lesson) and ``spoken_entities`` (flattened from
``ChronicleRecord.entities``). The rows also render a flat ``factoid_text()``
document that the text stack indexes for the VQA Track A lookup.
"""

from __future__ import annotations

import logging
from statistics import median

from aic.chronicle.jobs import load_chronicle
from aic.chronicle.schema import ChronicleRecord, OcrLine
from aic.config import Config, LedgerConfig, StrictModel
from aic.manifest import ManifestWriter, read_manifest

logger = logging.getLogger(__name__)

LEDGER_MANIFEST = "ledger.jsonl"


class LedgerRow(StrictModel):
    """The pre-computed answer space for one shot; empty fields are legal."""

    shot_key: str
    screen_text: list[str] = []
    spoken_entities: list[str] = []
    people_count: int | None = None
    salient_objects: list[str] = []

    @property
    def is_empty(self) -> bool:
        """True when no factoid field carries searchable text.

        ``people_count`` is a number, not a search term, so it does not by
        itself make a shot indexable; ``salient_objects`` (concept names) does.
        """
        return not (self.screen_text or self.spoken_entities or self.salient_objects)

    def factoid_text(self) -> str:
        """One flat searchable document, no JSON (the Track A index input)."""
        parts = (
            list(self.screen_text)
            + list(self.spoken_entities)
            + list(self.salient_objects)
        )
        return ". ".join(part for part in parts if part)


def reading_order(lines: list[OcrLine], row_band_factor: float) -> list[OcrLine]:
    """Sort OCR lines into human reading order (top-to-bottom, left-to-right).

    Uses each line's normalised bbox: lines are grouped into rows (bands) by
    vertical position, then ordered by band and then by left edge. A new band
    starts when a line's y-centre exceeds the current band's by
    ``row_band_factor`` of the median line height. If any line lacks a bbox
    (legacy corpora, VLM OCR), the manifest order is kept unchanged — degraded,
    never wrong (the ViTextVQA token-order lesson only applies when geometry
    is available).
    """
    if not lines or any(line.bbox is None for line in lines):
        return list(lines)
    metrics = []
    for line in lines:
        x0, y0, _x1, y1 = line.bbox  # type: ignore[misc]  # bbox is not None here
        metrics.append((line, x0, (y0 + y1) / 2.0, y1 - y0))
    med_height = median(height for _line, _x0, _yc, height in metrics) or 0.0
    threshold = row_band_factor * med_height
    by_y = sorted(metrics, key=lambda m: m[2])
    band = 0
    band_yc = by_y[0][2]
    banded = []
    for line, x0, yc, _height in by_y:
        if yc - band_yc > threshold:
            band += 1
            band_yc = yc
        banded.append((band, x0, line))
    banded.sort(key=lambda item: (item[0], item[1]))
    return [line for _band, _x0, line in banded]


def _screen_text(record: ChronicleRecord, cfg: LedgerConfig) -> list[str]:
    """Reading-order scene-OCR text above the confidence floor.

    ``record.ocr`` already excludes persistent overlays (the assembler's
    overlay split), so this is per-shot scene text, not tickers/watermarks.
    """
    kept = [line for line in record.ocr if line.confidence >= cfg.min_ocr_confidence]
    return [line.text for line in reading_order(kept, cfg.row_band_factor)]


def _spoken_entities(record: ChronicleRecord) -> list[str]:
    """Persons + orgs + places from the Chronicle entities, order-deduped."""
    entities = record.entities
    return list(
        dict.fromkeys(entities.persons + entities.orgs + entities.places)
    )


def build_ledger_rows(
    records: list[ChronicleRecord],
    cfg: LedgerConfig,
    counts: dict[str, dict] | None = None,
) -> list[LedgerRow]:
    """Pure function: assembled shots -> typed ledger rows.

    A field is populated only when it is in ``cfg.fields``; an absent field
    stays empty, so a profile can enable a subset without producing partial
    junk. ``counts`` is the shot_key -> {people_count, salient_objects} map from
    the offline counter stage (empty/None when the counter is off). Deterministic
    given the chronicle and counts.
    """
    want_screen = "screen_text" in cfg.fields
    want_entities = "spoken_entities" in cfg.fields
    want_people = "people_count" in cfg.fields
    want_objects = "salient_objects" in cfg.fields
    counts = counts or {}
    rows = []
    for record in records:
        count_row = counts.get(record.shot_key, {})
        rows.append(
            LedgerRow(
                shot_key=record.shot_key,
                screen_text=_screen_text(record, cfg) if want_screen else [],
                spoken_entities=_spoken_entities(record) if want_entities else [],
                people_count=count_row.get("people_count") if want_people else None,
                salient_objects=(
                    count_row.get("salient_objects", []) if want_objects else []
                ),
            )
        )
    return rows


def build_ledger(cfg: Config) -> int:
    """Rebuild ledger.jsonl from the assembled chronicle. Returns row count.

    Assembler pattern (cheap, deterministic): the whole file is rewritten each
    run. Every shot gets a row (empty fields are legal); the factoid index
    skips the empty ones.
    """
    records = load_chronicle(cfg)
    counts = None
    if cfg.chronicle.ledger.counter.enabled:
        from aic.vqa.counter.stage import load_counts

        counts = load_counts(cfg)
    rows = build_ledger_rows(records, cfg.chronicle.ledger, counts)
    path = cfg.paths.manifests_dir / LEDGER_MANIFEST
    path.unlink(missing_ok=True)
    with ManifestWriter(path, "shot_key") as manifest:
        for row in rows:
            manifest.append(row.model_dump())
    logger.info("ledger built: %d rows", len(rows))
    return len(rows)


def load_ledger(cfg: Config) -> list[LedgerRow]:
    """Read ledger.jsonl back into validated rows (empty when absent)."""
    return [
        LedgerRow.model_validate(row)
        for row in read_manifest(cfg.paths.manifests_dir / LEDGER_MANIFEST)
    ]


def load_ledger_map(cfg: Config) -> dict[str, LedgerRow]:
    """Ledger rows keyed by shot_key, for joining onto retrieved moments."""
    return {row.shot_key: row for row in load_ledger(cfg)}
