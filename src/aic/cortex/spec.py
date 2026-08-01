"""The typed query spec the Cortex compiles raw queries into.

The field set follows note 09 Phase 6: visual phrases feed the embedding
channels, OCR literals feed the sparse channel verbatim, ASR phrases target
speech, temporal hints and negations are carried as metadata/filters (never
embedded), and paraphrases join the channel race at reduced weight.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from aic.config import StrictModel

SpecSource = Literal["cortex", "fallback"]


class QuerySpec(StrictModel):
    """One compiled query. Lists may be empty, but never all of them."""

    visual_phrases: list[str] = Field(
        default_factory=list,
        description="What the target moment looks like, in English (the "
        "Cortex folds translation into compilation).",
    )
    entity_terms: list[str] = Field(
        default_factory=list,
        description="Named entities / proper nouns, verbatim.",
    )
    ocr_literals: list[str] = Field(
        default_factory=list,
        description="Text expected on screen, verbatim in original script.",
    )
    asr_phrases: list[str] = Field(
        default_factory=list,
        description="Phrases expected in speech, in the spoken language.",
    )
    temporal_hints: list[str] = Field(
        default_factory=list,
        description="Ordering/timing hints ('after the anchor', 'at the "
        "end'); consumed by TRAKE/temporal logic, not embedded.",
    )
    negations: list[str] = Field(
        default_factory=list,
        description="Things the target is NOT; applied as filters.",
    )
    paraphrases: list[str] = Field(
        default_factory=list,
        description="Alternative phrasings of the visual description.",
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="The Cortex's certainty that it parsed the query "
        "correctly; a QPP feature in Phase 7.",
    )
    source: SpecSource = Field(
        default="cortex",
        description="'fallback' marks specs built without the LLM (disabled "
        "cortex, LLM failure, or empty reply) - the raw query as one visual "
        "phrase.",
    )

    def truncated(self, max_items: int) -> QuerySpec:
        """A copy with every list capped at ``max_items`` entries."""
        return QuerySpec(
            visual_phrases=self.visual_phrases[:max_items],
            entity_terms=self.entity_terms[:max_items],
            ocr_literals=self.ocr_literals[:max_items],
            asr_phrases=self.asr_phrases[:max_items],
            temporal_hints=self.temporal_hints[:max_items],
            negations=self.negations[:max_items],
            paraphrases=self.paraphrases[:max_items],
            confidence=self.confidence,
            source=self.source,
        )

    @property
    def is_empty(self) -> bool:
        """True when nothing dispatchable survived compilation."""
        return not (
            self.visual_phrases
            or self.entity_terms
            or self.ocr_literals
            or self.asr_phrases
            or self.paraphrases
        )


def fallback_spec(raw_query: str) -> QuerySpec:
    """The raw query as a single visual phrase.

    This is the "never dumber than baseline" guarantee (note 09 Phase 6):
    whatever fails in the smart layer, dispatching this spec equals the
    Phase 5 single-string behaviour. An empty query yields an empty spec,
    which dispatch answers with no results rather than an error.
    """
    phrase = raw_query.strip()
    return QuerySpec(visual_phrases=[phrase] if phrase else [], source="fallback")
