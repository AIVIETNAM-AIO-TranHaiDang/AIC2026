"""Shared VQA answer derivation for the service route and the eval harness.

Both `/api/qa` (aic.service.app) and the QA branch of the harness
(scripts/evaluate_fixture.py) compile a question, locate candidate moments, and
turn them into an answer — Track A reads a first guess off the Answer Ledger,
Track B reads and votes across moments. Keeping that derivation here means the
number the harness measures is produced by the same code the operator sees.
"""

from __future__ import annotations

from collections.abc import Callable

from aic.config import VqaConfig
from aic.retrieval.fusion import ShotCandidate, ShotMeta
from aic.service.bundles import EvidenceBundle
from aic.vqa.evidence import build_evidence_pack, build_video_shot_index
from aic.vqa.normalize import normalize_answer
from aic.vqa.vote import AnswerGroup, ReadResult, apply_disconfirmation, vote_answers

# archetype -> the ledger field that directly answers it (a non-null Track A
# answer). Other archetypes get a located moment with visible evidence only.
ARCHETYPE_ANSWER_FIELD = {
    "read_text": "screen_text",
    "entity": "spoken_entities",
}

# Ledger fields shown as provenance when the archetype has no direct field.
_PROVENANCE_FALLBACK_FIELDS = ("screen_text", "spoken_entities")


def ledger_answer(row: object, archetype: str) -> tuple[str | None, dict | None]:
    """Track A first-guess answer + provenance from one ledger row.

    Returns ``(answer_guess, provenance)``. ``answer_guess`` is non-null only
    when the archetype maps directly to a ledger field with a value; otherwise
    the row still yields visible provenance (the first non-empty fallback
    field) so the operator sees the evidence behind a located moment.
    """
    if row is None:
        return None, None
    field = ARCHETYPE_ANSWER_FIELD.get(archetype)
    values = getattr(row, field, None) if field else None
    if values:
        return values[0], {"kind": "ledger", "field": field, "value": values[0]}
    for fallback in _PROVENANCE_FALLBACK_FIELDS:
        vals = getattr(row, fallback, None)
        if vals:
            return None, {"kind": "ledger", "field": fallback, "value": vals[0]}
    return None, None


def track_b_groups(
    reader,
    vqa_cfg: VqaConfig,
    plan,
    question: str,
    candidates: list[ShotCandidate],
    bundles: dict[str, EvidenceBundle],
    shot_meta: dict[str, ShotMeta],
    video_of: Callable[[str], str],
    track_a_answers: list[str],
    video_index: dict[str, list[tuple[int, str]]] | None = None,
) -> list[AnswerGroup]:
    """Read the top-M candidates, vote across moments, guard with disconfirm.

    Pure orchestration over the phase-2 pieces (evidence packs, the reader,
    voting, disconfirmation); shared by the service route and the harness so
    both derive answer groups identically. ``video_of`` maps a shot_key to its
    video_id (bundle- or metadata-backed); ``track_a_answers`` are the Track A
    guesses whose normalised form earns the cross-track vote bonus.
    ``video_index`` (aic.vqa.evidence.build_video_shot_index) is reused when
    the caller holds one; otherwise it is built once here so the top-M window
    scans stay per-video, never top_m x corpus.
    """
    pool = candidates[: vqa_cfg.top_m]
    if not pool:
        return []
    if video_index is None:
        video_index = build_video_shot_index(shot_meta)
    packs = [
        build_evidence_pack(candidate, bundles, shot_meta, vqa_cfg, video_index)
        for candidate in pool
    ]
    question_text = plan.question_core or question
    reads = reader.read_many(packs, question_text)

    packs_by_shot = {pack.shot_key: pack for pack in packs}
    results = [
        ReadResult(
            shot_key=candidate.shot_key,
            video_id=video_of(candidate.shot_key),
            candidate_score=candidate.score,
            grounded_timestamp_ms=pack.grounded_timestamp_ms,
            read=read,
        )
        for candidate, pack, read in zip(pool, packs, reads, strict=True)
    ]
    track_a_normalized = {
        normalize_answer(answer, plan.expected_answer_type, vqa_cfg.normalize)
        for answer in track_a_answers
    }
    track_a_normalized.discard("")

    groups = vote_answers(
        results, plan.expected_answer_type, vqa_cfg, track_a_normalized
    )
    return apply_disconfirmation(
        reader, groups, packs_by_shot, question_text, vqa_cfg
    )
