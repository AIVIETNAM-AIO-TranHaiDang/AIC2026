"""The labelled evaluation fixture: queries with ground-truth segments.

The fixture is a YAML file listing the dev corpus and the labelled query
cases. A case may carry several acceptable ground-truth ranges because news
footage is frequently re-broadcast; labelling only one copy would punish a
correct answer (see the Phase 1 edge cases in note 09).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from aic.config import ConfigError, StrictModel

QueryKind = Literal["kis_t", "kis_v", "kis_c", "qa"]


class GroundTruth(StrictModel):
    video_id: str
    t_start_ms: int = Field(ge=0)
    t_end_ms: int = Field(ge=0)

    @model_validator(mode="after")
    def _range_ordered(self) -> GroundTruth:
        if self.t_end_ms < self.t_start_ms:
            raise ValueError(f"ground truth for {self.video_id}: t_end_ms < t_start_ms")
        return self


class QueryCase(StrictModel):
    query_id: str
    kind: QueryKind
    text: str | None = Field(
        default=None, description="Vietnamese query text (kis_t, kis_c stage 0)."
    )
    clip_path: str | None = Field(
        default=None,
        description="Path of the example clip relative to the fixture file (kis_v).",
    )
    reveals: list[str] = Field(
        default_factory=list,
        description="Progressively revealed detail texts (kis_c only).",
    )
    question: str | None = Field(
        default=None,
        description="The question asked about the moment (kind 'qa' only).",
    )
    accepted_answers: list[str] = Field(
        default_factory=list,
        description="Answers a DRES assessor would accept, paraphrases "
        "included (kind 'qa'); matched after Vietnamese normalisation.",
    )
    archetype: str | None = Field(
        default=None,
        description="Optional labelled question archetype (count/read_text/"
        "...). None lets the compiler decide; a label also tests the compiler "
        "and drives the per-archetype metric breakdown.",
    )
    truths: list[GroundTruth] = Field(min_length=1)

    @model_validator(mode="after")
    def _kind_requirements(self) -> QueryCase:
        if self.kind == "kis_t" and not self.text:
            raise ValueError(f"case {self.query_id}: kis_t requires 'text'")
        if self.kind == "kis_v" and not self.clip_path:
            raise ValueError(f"case {self.query_id}: kis_v requires 'clip_path'")
        if self.kind == "kis_c":
            if not self.text:
                raise ValueError(f"case {self.query_id}: kis_c requires 'text'")
            if not self.reveals:
                raise ValueError(
                    f"case {self.query_id}: kis_c requires at least one reveal"
                )
        if self.kind != "kis_c" and self.reveals:
            raise ValueError(f"case {self.query_id}: 'reveals' is only valid for kis_c")
        if self.kind == "qa":
            if not self.question:
                raise ValueError(f"case {self.query_id}: qa requires 'question'")
            if not self.accepted_answers:
                raise ValueError(
                    f"case {self.query_id}: qa requires at least one accepted_answer"
                )
        elif self.question or self.accepted_answers or self.archetype is not None:
            raise ValueError(
                f"case {self.query_id}: 'question'/'accepted_answers'/'archetype' "
                "are only valid for kind 'qa'"
            )
        return self


class EvalFixture(StrictModel):
    corpus_videos: list[str] = Field(
        min_length=1, description="video_ids the fixture queries run against."
    )
    cases: list[QueryCase] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> EvalFixture:
        known = set(self.corpus_videos)
        seen_ids: set[str] = set()
        for case in self.cases:
            if case.query_id in seen_ids:
                raise ValueError(f"duplicate query_id {case.query_id!r}")
            seen_ids.add(case.query_id)
            for truth in case.truths:
                if truth.video_id not in known:
                    raise ValueError(
                        f"case {case.query_id}: ground-truth video "
                        f"{truth.video_id!r} is not in corpus_videos"
                    )
        return self


def load_fixture(path: Path | str) -> EvalFixture:
    """Load and validate an evaluation fixture from YAML."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"fixture file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"fixture file {path} is not valid YAML: {exc}") from exc
    try:
        return EvalFixture.model_validate(raw)
    except ValueError as exc:
        raise ConfigError(f"fixture file {path} failed validation: {exc}") from exc
