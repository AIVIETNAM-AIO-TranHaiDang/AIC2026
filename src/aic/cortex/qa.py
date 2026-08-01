"""The typed QA plan the Cortex compiles a question into (note 18 §3.1).

A question becomes an *answer program*: an archetype (which track/tool should
handle it), a ``locator`` (a reused :class:`QuerySpec` describing the moment to
find), the question core, and the expected answer type. Phase 1 uses only the
locator (Track A lookup); later phases route on the archetype.
"""

from __future__ import annotations

from pydantic import Field

from aic.config import StrictModel
from aic.cortex.spec import QuerySpec, fallback_spec

# Expected answer shape; guides normalisation/validation in later phases. Free
# text is the safe default. A format contract, not a per-corpus tunable.
ANSWER_TYPES = ("number", "text", "entity", "color", "yes_no", "duration")

# The archetype used whenever compilation fails or returns an unknown value —
# the never-fail path (mirrors spec.fallback_spec for KIS).
FALLBACK_ARCHETYPE = "other"


class QaPlan(StrictModel):
    """One compiled question. ``locator`` is never all-empty (fallback fills it)."""

    archetype: str = Field(
        description="Which program handles the question (count/read_text/"
        "entity/attribute/cumulative/other); validated against vqa.archetypes."
    )
    locator: QuerySpec = Field(
        description="The moment to find, as a reused KIS query spec."
    )
    question_core: str = Field(
        description="The question stripped to what is actually asked, verbatim."
    )
    expected_answer_type: str = Field(
        default="text",
        description=f"Expected answer shape, one of {list(ANSWER_TYPES)}.",
    )
    concept: str | None = Field(
        default=None,
        description="For a count question, the countable object noun phrase the "
        "detector-counter should segment (e.g. 'person in a red shirt'); None "
        "otherwise. The count program falls back to counter.person_concept.",
    )


def fallback_qa_plan(question: str) -> QaPlan:
    """The never-fail plan: archetype 'other', the whole question as locator.

    Whatever fails in QA compilation, this dispatches exactly like a plain KIS
    query for the moment, so Track A still returns located evidence.
    """
    return QaPlan(
        archetype=FALLBACK_ARCHETYPE,
        locator=fallback_spec(question),
        question_core=question.strip(),
        expected_answer_type="text",
    )
