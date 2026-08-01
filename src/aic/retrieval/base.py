"""The retriever contract every search backend implements.

The evaluation harness (Phase 1) and the online service (Phase 6) both talk to
this interface, so any component that can rank ``(video_id, timestamp_ms)``
candidates for a query case is measurable and swappable.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from aic.eval.fixture import QueryCase


@dataclass(frozen=True)
class SearchResult:
    video_id: str
    timestamp_ms: int
    score: float
    keyframe_id: str | None = None


class Retriever(Protocol):
    def search(self, case: QueryCase, top_k: int) -> list[SearchResult]:
        """Return up to ``top_k`` results, best first."""
        ...


class RandomBaselineRetriever:
    """Uniform random guesses over the corpus.

    Exists to prove the harness works and to anchor the metric floor: every
    real retriever must beat this by a wide margin, and the fixture is broken
    if this baseline ever scores well.
    """

    def __init__(self, video_durations_ms: Mapping[str, int], seed: int) -> None:
        if not video_durations_ms:
            raise ValueError("video_durations_ms must not be empty")
        for video_id, duration in video_durations_ms.items():
            if duration <= 0:
                raise ValueError(
                    f"video {video_id!r} has non-positive duration {duration}"
                )
        self._durations = dict(video_durations_ms)
        self._rng = random.Random(seed)

    def search(self, case: QueryCase, top_k: int) -> list[SearchResult]:
        video_ids = list(self._durations)
        results = []
        for rank in range(top_k):
            video_id = self._rng.choice(video_ids)
            timestamp_ms = self._rng.randrange(self._durations[video_id])
            results.append(
                SearchResult(
                    video_id=video_id,
                    timestamp_ms=timestamp_ms,
                    score=float(top_k - rank),
                )
            )
        return results
