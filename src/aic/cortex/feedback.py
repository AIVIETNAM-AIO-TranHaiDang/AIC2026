"""Rocchio relevance feedback over the dense visual channel (VPRF).

The operator marks shots as "more like this" / "not this"; the query vector
moves toward the positive keyframes and away from the negative ones:

    q' = alpha * q + beta * mean(positives) - gamma * mean(negatives)

Feedback runs on the dense visual channel only — it is the channel with a
geometric query — and the re-ranked list replaces the fused one for that
step. Positive/negative shots map to their keyframe vectors through the
keyframe metadata already loaded for retrieval.
"""

from __future__ import annotations

import logging

import numpy as np

from aic.config import FeedbackConfig
from aic.embed.encoders import ImageTextEncoder, l2_normalize
from aic.index.vector import VectorIndex
from aic.retrieval.fusion import ShotCandidate

logger = logging.getLogger(__name__)


def rank_vector_over_keyframes(
    index: VectorIndex,
    keyframe_meta: dict[str, tuple[str, int, int]],
    query_model_id: str,
    query: np.ndarray,
    top_k: int,
) -> list[ShotCandidate]:
    """Rank shots by their best keyframe for a ``[1, dim]`` query vector.

    Keyframes are over-fetched (several may belong to one shot) and pooled
    to the best-scoring keyframe per shot — the same pooling the dense
    visual channel uses. Shared by relevance feedback, similar-shot search,
    and the Imagine->Match escalation.
    """
    hits = index.search(
        query,
        min(top_k * 4, len(index)),
        query_model_id=query_model_id,
    )[0]
    best_per_shot: dict[str, ShotCandidate] = {}
    for keyframe_id, score in hits:
        meta = keyframe_meta.get(keyframe_id)
        if meta is None:
            continue
        video_id, shot_id, timestamp_ms = meta
        shot_key = f"{video_id}:{shot_id}"
        if shot_key not in best_per_shot:
            best_per_shot[shot_key] = ShotCandidate(
                shot_key=shot_key, score=score, timestamp_ms=timestamp_ms
            )
        if len(best_per_shot) >= top_k:
            break
    return sorted(best_per_shot.values(), key=lambda c: -c.score)


def rocchio_update(
    query: np.ndarray,
    positives: np.ndarray,
    negatives: np.ndarray,
    cfg: FeedbackConfig,
) -> np.ndarray:
    """The updated, re-normalised query vector. Inputs are ``[n, dim]``."""
    updated = cfg.alpha * query
    if len(positives):
        updated = updated + cfg.beta * positives.mean(axis=0, keepdims=True)
    if len(negatives):
        updated = updated - cfg.gamma * negatives.mean(axis=0, keepdims=True)
    return l2_normalize(updated.astype(np.float32))


class DenseFeedbackEngine:
    """Feedback + similar-shot search over the keyframe index."""

    def __init__(
        self,
        encoder: ImageTextEncoder,
        index: VectorIndex,
        keyframe_meta: dict[str, tuple[str, int, int]],
        cfg: FeedbackConfig,
    ) -> None:
        """``keyframe_meta`` maps keyframe_id -> (video_id, shot_id, ts_ms),
        the same mapping the dense visual channel uses."""
        self._encoder = encoder
        self._index = index
        self._keyframe_meta = keyframe_meta
        self._cfg = cfg
        self._keyframes_by_shot: dict[str, list[str]] = {}
        for keyframe_id, (video_id, shot_id, _ts) in keyframe_meta.items():
            key = f"{video_id}:{shot_id}"
            self._keyframes_by_shot.setdefault(key, []).append(keyframe_id)

    def _shot_vectors(self, shot_keys: list[str]) -> np.ndarray:
        keyframe_ids = []
        for shot_key in shot_keys:
            ids = self._keyframes_by_shot.get(shot_key)
            if not ids:
                logger.warning("feedback shot %s has no keyframes; ignored", shot_key)
                continue
            keyframe_ids.extend(ids)
        if not keyframe_ids:
            return np.zeros((0, self._index.dim), dtype=np.float32)
        return self._index.vectors_for(keyframe_ids)

    def rank_feedback(
        self,
        query_text: str,
        positive_shots: list[str],
        negative_shots: list[str],
        top_k: int,
    ) -> list[ShotCandidate]:
        """Re-rank with the Rocchio-updated query vector."""
        query = self._encoder.encode_texts([query_text])
        updated = rocchio_update(
            query,
            self._shot_vectors(positive_shots),
            self._shot_vectors(negative_shots),
            self._cfg,
        )
        return self._rank_vector(updated, top_k)

    def rank_similar(self, shot_key: str, top_k: int) -> list[ShotCandidate]:
        """Shots most similar to ``shot_key`` (its keyframe centroid)."""
        vectors = self._shot_vectors([shot_key])
        if not len(vectors):
            return []
        centroid = l2_normalize(
            vectors.mean(axis=0, keepdims=True).astype(np.float32)
        )
        # The probed shot itself always tops its own similarity list.
        return [
            candidate
            for candidate in self._rank_vector(centroid, top_k + 1)
            if candidate.shot_key != shot_key
        ][:top_k]

    def _rank_vector(self, query: np.ndarray, top_k: int) -> list[ShotCandidate]:
        return rank_vector_over_keyframes(
            self._index,
            self._keyframe_meta,
            self._encoder.model_id,
            query,
            top_k,
        )
