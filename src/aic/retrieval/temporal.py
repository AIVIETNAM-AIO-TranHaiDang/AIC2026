"""Temporal window fusion: score windows of adjacent shots, not lone shots.

KIS queries describe a scene that unfolds over several shots; per-shot RRF
lets the fragments compete against each other instead of accumulate
(note 15). Here the per-(sub-query) rankings are re-aggregated per video:
candidates whose shot midpoints lie within ``window_ms`` of each other form
a window, the window collects each sub-query's best contribution inside it
(soft coverage), cross-channel agreement multiplies the score, and an
optional per-video prior (overlay text) is added. The window's best shot is
what enters the final ranking, so the submission contract — one
``(video_id, timestamp)`` per result — is unchanged.

With ``window_ms == 0`` the function returns plain weighted RRF, bit for
bit, which is both the ablation switch and the compatibility guarantee.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping

from aic.config import TemporalFusionConfig
from aic.index.sparse import SparseIndex
from aic.retrieval.fusion import ShotCandidate, ShotMeta, rrf_fuse
from aic.textstack.embedder import TextEmbedder

logger = logging.getLogger(__name__)


def channel_of_label(label: str) -> str:
    """Channel name from a dispatch sub-query key (``field[i].channel``).

    Plain channel names (the FusionRetriever case) pass through unchanged.
    """
    return label.rsplit(".", 1)[-1]


class OverlayVideoPrior:
    """Ranks videos by their persistent-overlay text (the video prior).

    Overlay lines (watermarks, tickers, program names) are noise at shot
    granularity but identify the right VIDEO; this searches the per-video
    overlay index built by the textstack job.
    """

    def __init__(self, embedder: TextEmbedder, index: SparseIndex) -> None:
        self._embedder = embedder
        self._index = index

    def contributions(
        self, text: str, weight: float, rrf_k: int, top_k: int
    ) -> dict[str, float]:
        """video_id -> RRF-shaped contribution ``weight / (rrf_k + rank)``."""
        if not text.strip() or weight <= 0:
            return {}
        embeddings = self._embedder.encode_queries([text])
        if not embeddings.sparse:
            return {}
        hits = self._index.search(embeddings.sparse[0], top_k)
        return {
            video_id: weight / (rrf_k + rank)
            for rank, (video_id, _score) in enumerate(hits, start=1)
        }


def fuse_windows(
    rankings: dict[str, list[ShotCandidate]],
    weights: dict[str, float],
    rrf_k: int,
    shot_meta: dict[str, ShotMeta],
    cfg: TemporalFusionConfig,
    channel_of: Callable[[str], str] = channel_of_label,
    video_prior: dict[str, float] | None = None,
    label_order: Mapping[str, int] | None = None,
) -> list[ShotCandidate]:
    """Window-level fusion over per-label rankings (see module docstring).

    ``rankings``/``weights``/``rrf_k`` are exactly what :func:`rrf_fuse`
    takes; ``channel_of`` maps a ranking label to its channel for the
    agreement bonus. Every window is the maximal candidate run ending at
    each shot (two-pointer over midpoints), so window count is linear in
    candidates and member scans are bounded by how many shots physically
    fit in ``window_ms``. Shots that head no window keep their per-shot RRF
    order below the window results, so result depth never shrinks.

    ``label_order`` maps ordered sub-query labels (the sentence chunks of
    retrieval.visual_chunking) to their sentence position. When set and
    ``cfg.order_bonus`` > 0, a window whose ordered labels hit shots in
    non-decreasing time order — earlier sentences matching earlier shots —
    scores * (1 + order_bonus): sentence order in a KIS query usually
    mirrors event order (vitrivr's temporal scoring, VBS).
    """
    base = rrf_fuse(rankings, weights, rrf_k)
    if cfg.window_ms == 0:
        return base

    # Per-shot, per-label contribution (the addends rrf_fuse summed).
    contrib: dict[str, dict[str, float]] = {}
    for label, candidates in rankings.items():
        weight = weights.get(label)
        if weight is None:
            continue
        for rank, candidate in enumerate(candidates, start=1):
            per_shot = contrib.setdefault(candidate.shot_key, {})
            per_shot[label] = per_shot.get(label, 0.0) + weight / (rrf_k + rank)

    base_by_key = {candidate.shot_key: candidate for candidate in base}
    prior = video_prior or {}

    # Candidates grouped per video, in midpoint order.
    by_video: dict[str, list[tuple[int, str]]] = {}
    dropped = 0
    for shot_key in contrib:
        meta = shot_meta.get(shot_key)
        if meta is None:
            dropped += 1
            continue
        by_video.setdefault(meta.video_id, []).append(
            (meta.midpoint_ms, shot_key)
        )
    if dropped:
        logger.warning(
            "%d fused candidate(s) lack shot metadata; excluded from windows",
            dropped,
        )

    # Best window score per representative shot (dedup across the
    # overlapping maximal windows that share a representative).
    best_for_rep: dict[str, float] = {}
    for video_id, members in by_video.items():
        members.sort()
        left = 0
        for right in range(len(members)):
            while members[right][0] - members[left][0] > cfg.window_ms:
                left += 1
            best_per_label: dict[str, float] = {}
            best_mid: dict[str, int] = {}
            for mid, shot_key in members[left : right + 1]:
                for label, value in contrib[shot_key].items():
                    if value > best_per_label.get(label, 0.0):
                        best_per_label[label] = value
                        best_mid[label] = mid
            channels = {channel_of(label) for label in best_per_label}
            score = sum(best_per_label.values())
            if cfg.agreement_bonus > 0 and len(channels) > 1:
                score *= 1.0 + cfg.agreement_bonus * (len(channels) - 1)
            if cfg.order_bonus > 0 and label_order:
                ordered = sorted(
                    (label_order[label], best_mid[label])
                    for label in best_per_label
                    if label in label_order
                )
                if len(ordered) > 1 and all(
                    earlier[1] <= later[1]
                    # Adjacent pairs: lengths differ by one, so strict=False.
                    for earlier, later in zip(ordered, ordered[1:], strict=False)
                ):
                    score *= 1.0 + cfg.order_bonus
            score += prior.get(video_id, 0.0)
            # Deterministic: best per-shot RRF score, ties on shot_key
            # ascending (matching rrf_fuse's own tie rule).
            representative = min(
                (
                    (-base_by_key[shot_key].score, shot_key)
                    for _mid, shot_key in members[left : right + 1]
                )
            )[1]
            current = best_for_rep.get(representative)
            if current is None or score > current:
                best_for_rep[representative] = score

    windows = sorted(
        best_for_rep.items(), key=lambda item: (-item[1], item[0])
    )
    results = [
        ShotCandidate(
            shot_key=shot_key,
            score=score,
            timestamp_ms=base_by_key[shot_key].timestamp_ms,
        )
        for shot_key, score in windows
    ]
    emitted = set(best_for_rep)
    results.extend(
        candidate for candidate in base if candidate.shot_key not in emitted
    )
    return results
