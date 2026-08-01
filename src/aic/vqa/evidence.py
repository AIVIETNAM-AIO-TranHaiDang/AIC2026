"""Evidence packs for Track B: what one candidate moment's reader sees.

A candidate is a moment (a shot with a grounded timestamp); the reader needs
the frames AROUND that moment (across neighbouring shots inside the zoom
window, nearest first) plus the moment's own text (OCR in reading order,
speech, caption). Pure and deterministic, so it is unit-tested without any
model.
"""

from __future__ import annotations

from aic.config import StrictModel, VqaConfig
from aic.retrieval.fusion import ShotCandidate, ShotMeta
from aic.service.bundles import EvidenceBundle


class EvidencePack(StrictModel):
    """One candidate's evidence for a grounded read."""

    shot_key: str
    grounded_timestamp_ms: int
    frames: list[str]  # keyframes-dir-relative names, nearest-shot first
    text_block: str


def _anchor_ms(candidate: ShotCandidate, bundle: EvidenceBundle | None) -> int:
    if candidate.timestamp_ms is not None:
        return candidate.timestamp_ms
    return bundle.midpoint_ms if bundle is not None else 0


def _text_block(bundle: EvidenceBundle | None, cfg: VqaConfig) -> str:
    """Source-labelled text for the moment (OCR reading order, speech, caption).

    Labels come from ``cfg.evidence_labels`` (English by default, override to
    localise); the values stay in the footage's original language.
    """
    if bundle is None:
        return ""
    labels = cfg.evidence_labels
    parts = []
    if bundle.ocr_lines:
        parts.append(f"{labels['ocr']}: " + " | ".join(bundle.ocr_lines))
    if bundle.asr_text:
        parts.append(f"{labels['asr']}: " + bundle.asr_text)
    if bundle.caption:
        parts.append(f"{labels['caption']}: " + bundle.caption)
    return "\n".join(parts)[: cfg.reader.max_evidence_chars]


def build_video_shot_index(
    shot_meta: dict[str, ShotMeta],
) -> dict[str, list[tuple[int, str]]]:
    """``video_id -> [(midpoint_ms, shot_key), ...]`` for window gathering.

    Build once and pass to :func:`build_evidence_pack`: the window scan is
    then linear in ONE video's shots instead of the whole corpus per
    candidate (top_m candidates x a large corpus adds whole seconds of pure
    Python per question otherwise).
    """
    index: dict[str, list[tuple[int, str]]] = {}
    for key, meta in shot_meta.items():
        index.setdefault(meta.video_id, []).append((meta.midpoint_ms, key))
    return index


def build_evidence_pack(
    candidate: ShotCandidate,
    bundles: dict[str, EvidenceBundle],
    shot_meta: dict[str, ShotMeta],
    cfg: VqaConfig,
    video_index: dict[str, list[tuple[int, str]]] | None = None,
) -> EvidencePack:
    """Frames within ± zoom_window_ms across shots (nearest first) + text.

    Frames are gathered ACROSS the shots whose midpoint falls inside the window
    of the same video, not just the anchor shot — the temporal fusion rewards
    multi-shot windows, so the evidence must match what scored. Capped at
    ``reader.frames_per_candidate``. ``video_index`` (from
    :func:`build_video_shot_index`) narrows the window scan to the candidate's
    video; None scans ``shot_meta`` whole (same result, corpus-linear).
    """
    bundle = bundles.get(candidate.shot_key)
    meta = shot_meta.get(candidate.shot_key)
    video_id = meta.video_id if meta is not None else (
        bundle.video_id if bundle is not None else ""
    )
    anchor = _anchor_ms(candidate, bundle)

    # Shots of this video within the window, nearest midpoint to the anchor
    # first; the anchor shot itself sorts first (distance 0).
    if video_index is not None:
        candidates_in_video = video_index.get(video_id, [])
        in_window = [
            (abs(mid - anchor), key)
            for mid, key in candidates_in_video
            if abs(mid - anchor) <= cfg.zoom_window_ms
        ]
    else:
        in_window = [
            (abs(m.midpoint_ms - anchor), key)
            for key, m in shot_meta.items()
            if m.video_id == video_id
            and abs(m.midpoint_ms - anchor) <= cfg.zoom_window_ms
        ]
    in_window.sort()
    if not in_window and bundle is not None:
        in_window = [(0, candidate.shot_key)]

    frames: list[str] = []
    limit = cfg.reader.frames_per_candidate
    for _dist, key in in_window:
        shot_bundle = bundles.get(key)
        if shot_bundle is None:
            continue
        for name in shot_bundle.frames:
            if name not in frames:
                frames.append(name)
                if len(frames) >= limit:
                    break
        if len(frames) >= limit:
            break

    return EvidencePack(
        shot_key=candidate.shot_key,
        grounded_timestamp_ms=anchor,
        frames=frames,
        text_block=_text_block(bundle, cfg),
    )
