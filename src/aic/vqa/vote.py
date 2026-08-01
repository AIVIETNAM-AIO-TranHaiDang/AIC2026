"""Cross-candidate voting and disconfirmation for Track B (note 18 §3.4).

Independently-read candidate moments are grouped by their normalised answer
(the language-general normaliser); a group's score is the sum of its members'
retrieval scores plus a
one-off bonus when the same normalised answer also appears on a Track A card
(cross-track agreement — a genuinely independent signal). A malformed answer
for the question's type is demoted below well-formed ones, never dropped. One
optional disconfirmation call guards the winner. All pure except
:func:`apply_disconfirmation`, which takes a duck-typed reader so tests inject
a fake.
"""

from __future__ import annotations

from pydantic import Field

from aic.config import StrictModel, VqaConfig
from aic.vqa.evidence import EvidencePack
from aic.vqa.normalize import fold_diacritics, is_valid_for_type, normalize_answer
from aic.vqa.reader import QaRead


class ReadResult(StrictModel):
    """One candidate moment paired with its grounded read (a vote input)."""

    shot_key: str
    video_id: str
    candidate_score: float = Field(
        description="The moment's retrieval (fusion) score; the vote weight."
    )
    grounded_timestamp_ms: int = Field(
        description="The candidate's grounded keyframe timestamp — the frame "
        "submitted when this member wins (the KIS grounding rule)."
    )
    read: QaRead


class GroupMember(StrictModel):
    """One moment inside an answer group, best-scoring first."""

    shot_key: str
    video_id: str
    score: float
    timestamp_ms: int
    confidence: float


class AnswerGroup(StrictModel):
    """One candidate answer with every moment that read it."""

    answer: str = Field(
        description="Display answer: the canonical form for number/color, the "
        "verbatim read (diacritics kept) otherwise."
    )
    normalized: str = Field(description="Comparison key the members share.")
    score: float = Field(description="Sum of member scores + cross-track bonus.")
    valid: bool = Field(
        description="Whether the answer is well-formed for the question type; "
        "invalid groups rank below valid ones."
    )
    confident: bool = Field(
        default=False,
        description="At least vqa.vote_min_agree moments agree on this "
        "answer (the LongVidSearch cross-candidate signal); the console "
        "marks confident groups for early submission.",
    )
    members: list[GroupMember]


def _display(normalized: str, best_raw: str, answer_type: str) -> str:
    """Canonical form for number/color; the verbatim read otherwise.

    A number vote is clearer as "3" than as whichever surface form ("ba")
    happened to win, and a colour as its canonical name; a name or free-text
    answer must keep its original case and diacritics.
    """
    if answer_type in ("number", "color"):
        return normalized
    return best_raw


def vote_answers(
    results: list[ReadResult],
    answer_type: str,
    cfg: VqaConfig,
    track_a_normalized: set[str],
) -> list[AnswerGroup]:
    """Group answerable reads by normalised answer, ranked best group first.

    Abstentions (``answerable=False``) and answers that normalise to empty are
    ignored. Groups are ranked ``(valid, score)`` with a deterministic
    ``normalized`` tie-break, so a well-formed answer never loses to a
    malformed one at equal score and the order is reproducible.
    """
    grouped: dict[str, list[ReadResult]] = {}
    for result in results:
        if not result.read.answerable:
            continue
        normalized = normalize_answer(result.read.answer, answer_type, cfg.normalize)
        if not normalized:
            continue
        grouped.setdefault(normalized, []).append(result)

    # Looser diacritic tier for the CROSS-TRACK bonus only (trap 2.5: the fold
    # never merges groups — "màu" and "mau" stay distinct answers — but a
    # diacritics-less VLM read of a diacritics-bearing Track A card still earns
    # the agreement bonus).
    track_a_folded = {fold_diacritics(a) for a in track_a_normalized}

    groups: list[AnswerGroup] = []
    for normalized, members in grouped.items():
        score = sum(member.candidate_score for member in members)
        if cfg.track_a_bonus > 0 and (
            normalized in track_a_normalized
            or fold_diacritics(normalized) in track_a_folded
        ):
            score += cfg.track_a_bonus
        ordered = sorted(members, key=lambda m: (-m.candidate_score, m.shot_key))
        best_raw = ordered[0].read.answer
        groups.append(
            AnswerGroup(
                answer=_display(normalized, best_raw, answer_type),
                normalized=normalized,
                score=score,
                valid=is_valid_for_type(normalized, answer_type, cfg.normalize),
                confident=len(members) >= cfg.vote_min_agree,
                members=[
                    GroupMember(
                        shot_key=m.shot_key,
                        video_id=m.video_id,
                        score=m.candidate_score,
                        timestamp_ms=m.grounded_timestamp_ms,
                        confidence=m.read.confidence,
                    )
                    for m in ordered
                ],
            )
        )
    groups.sort(key=lambda g: (not g.valid, -g.score, g.normalized))
    return groups


def apply_disconfirmation(
    reader,
    groups: list[AnswerGroup],
    packs_by_shot: dict[str, EvidencePack],
    question: str,
    cfg: VqaConfig,
) -> list[AnswerGroup]:
    """Run one disconfirmation call against the runner-up; demote on a hit.

    Only fires with ``vqa.disconfirm`` on, a reader present, and at least two
    groups. The runner-up moment's evidence is asked whether it CONTRADICTS the
    winner's answer; a contradiction moves the winner below the runner-up. Any
    reader failure returns False (the vote stands — degrade to unchanged).
    ``reader`` is anything exposing ``disconfirm(pack, question, answer) ->
    bool`` (:class:`aic.vqa.reader.QaReader` in production, a fake in tests).
    """
    if not cfg.disconfirm or reader is None or len(groups) < 2:
        return groups
    winner, runner_up = groups[0], groups[1]
    pack = packs_by_shot.get(runner_up.members[0].shot_key)
    if pack is None:
        return groups
    if reader.disconfirm(pack, question, winner.answer):
        return [runner_up, winner, *groups[2:]]
    return groups
