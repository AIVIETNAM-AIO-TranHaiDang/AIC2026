"""Question planner: which one question best splits the candidates?

Given the current result set, suggest the attribute whose value distribution
has maximum entropy — the question whose answer eliminates the most
candidates in expectation (note 09 Phase 8). Attributes come from the
evidence schema: the caption's scene class, on-screen-text presence, and
speech presence. The planner returns structure (attribute + value counts);
the UI renders the human question, so no prose lives here.
"""

from __future__ import annotations

import math
from collections import Counter

from aic.service.bundles import EvidenceBundle

# Attribute name -> extractor over a bundle. Booleans stringify so every
# attribute is a categorical distribution.
_ATTRIBUTES = {
    "scene": lambda b: b.scene or "unknown",
    "has_text": lambda b: str(b.has_text).lower(),
    "has_speech": lambda b: str(b.has_speech).lower(),
}


def _entropy(counts: Counter) -> float:
    total = sum(counts.values())
    return -sum(
        (n / total) * math.log2(n / total) for n in counts.values() if n
    )


def suggest_question(bundles: list[EvidenceBundle]) -> dict | None:
    """The most informative attribute split, or None when nothing splits.

    Returns ``{"attribute": name, "counts": {value: n, ...}}``. An attribute
    with a single observed value carries no information and is skipped; with
    fewer than two candidates there is nothing to split.
    """
    if len(bundles) < 2:
        return None
    best: tuple[float, str, Counter] | None = None
    for name, extract in _ATTRIBUTES.items():
        counts = Counter(extract(bundle) for bundle in bundles)
        if len(counts) < 2:
            continue
        entropy = _entropy(counts)
        if best is None or entropy > best[0]:
            best = (entropy, name, counts)
    if best is None:
        return None
    _entropy_value, attribute, counts = best
    return {"attribute": attribute, "counts": dict(counts)}
