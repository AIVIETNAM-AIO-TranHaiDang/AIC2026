"""Sparse (lexical) index over learned term weights.

BGE-M3 emits per-text ``{token_id: weight}`` maps; relevance is the dot
product of query and document weights. An inverted index makes that a walk
over the query's terms only — exact, dependency-free, and fast at
chronicle scale (one document per shot).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from aic.manifest import write_json_atomic

_DOCS_FILE = "docs.json"


class SparseIndex:
    """Inverted index over sparse term-weight documents."""

    def __init__(self, weights: list[dict[str, float]], ids: list[str]) -> None:
        if len(weights) != len(ids):
            raise ValueError(
                f"{len(weights)} weight maps do not align with {len(ids)} ids"
            )
        self._ids = list(ids)
        self._postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for row, doc in enumerate(weights):
            for term, weight in doc.items():
                if weight > 0:
                    self._postings[term].append((row, float(weight)))
        self._weights = [dict(doc) for doc in weights]

    def __len__(self) -> int:
        return len(self._ids)

    def search(
        self, query_weights: dict[str, float], top_k: int
    ) -> list[tuple[str, float]]:
        """Top-k ``(id, score)`` by sparse dot product, best first."""
        scores: dict[int, float] = defaultdict(float)
        for term, query_weight in query_weights.items():
            if query_weight <= 0:
                continue
            for row, doc_weight in self._postings.get(term, ()):
                scores[row] += query_weight * doc_weight
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [(self._ids[row], score) for row, score in ranked[:top_k]]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            directory / _DOCS_FILE,
            {"ids": self._ids, "weights": self._weights},
        )

    @classmethod
    def load(cls, directory: Path) -> SparseIndex:
        payload = json.loads((directory / _DOCS_FILE).read_text(encoding="utf-8"))
        return cls(weights=payload["weights"], ids=payload["ids"])
