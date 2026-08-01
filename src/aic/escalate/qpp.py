"""The QPP advisor: calibrated P(hit in top-k) from post-retrieval signals.

Features come straight from what Phase 5/6 already compute (note 11): fused
score margins, cross-channel rank agreement, temporal compactness of the top
candidates, and the Cortex's own spec confidence. A logistic model is
trained on the labelled fixture by scripts/train_qpp.py and stored as a
transparent JSON artifact (no pickles), so the service can score a query
without scikit-learn.

Advisory only: the service renders the probability and a "consider
escalating" hint; the human decides (never auto-submit, note 09 Phase 7).
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np

from aic.config import QppConfig
from aic.cortex.spec import QuerySpec
from aic.manifest import write_json_atomic
from aic.retrieval.fusion import ShotCandidate

logger = logging.getLogger(__name__)

# The feature contract. Order matters: artifacts store the names they were
# trained with and the advisor refuses a mismatch (the same guard idea as
# the encoder-version check on indexes).
FEATURE_NAMES = (
    "top1_margin",
    "deep_margin",
    "channel_agreement",
    "temporal_compactness",
    "spec_confidence",
    "spec_is_fallback",
    "pool_fill",
)

ARTIFACT_VERSION = 1


class QppError(RuntimeError):
    """Raised on artifact contract violations."""


def _margin(scores: list[float], depth: int) -> float:
    """Relative fused-score drop from rank 1 to ``depth`` (0 when flat)."""
    if len(scores) < 2 or scores[0] <= 0:
        return 0.0
    rank = min(depth, len(scores)) - 1
    return max((scores[0] - scores[rank]) / scores[0], 0.0)


def _per_channel_ranks(
    rankings: dict[str, list[ShotCandidate]], top_n: int
) -> dict[str, dict[str, int]]:
    """Best rank per shot per channel, sub-queries of one channel merged.

    Dispatch keys look like ``visual_phrases[0].dense_visual``; everything
    after the last dot is the channel name (Phase 5 rankings use the bare
    channel name already).
    """
    merged: dict[str, dict[str, int]] = {}
    for key, candidates in rankings.items():
        channel = key.rsplit(".", 1)[-1]
        ranks = merged.setdefault(channel, {})
        for rank, candidate in enumerate(candidates[:top_n], start=1):
            current = ranks.get(candidate.shot_key)
            if current is None or rank < current:
                ranks[candidate.shot_key] = rank
    return merged


def _spearman(ranks_a: dict[str, int], ranks_b: dict[str, int], top_n: int) -> float:
    """Spearman correlation over the union of two channels' top shots.

    A shot one channel missed takes rank ``top_n + 1``: full disagreement
    about it is penalised but bounded.
    """
    union = sorted(set(ranks_a) | set(ranks_b))
    if len(union) < 2:
        return 0.0
    missing = top_n + 1
    a = np.array([ranks_a.get(key, missing) for key in union], dtype=np.float64)
    b = np.array([ranks_b.get(key, missing) for key in union], dtype=np.float64)
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    denom = np.linalg.norm(a_centered) * np.linalg.norm(b_centered)
    if denom == 0:
        return 0.0
    return float(a_centered @ b_centered / denom)


def channel_agreement(
    rankings: dict[str, list[ShotCandidate]], top_n: int
) -> float:
    """Mean pairwise rank correlation across channels, in [-1, 1].

    One (or zero) active channels cannot disagree with anything: that lack
    of corroboration is scored 0, not 1.
    """
    per_channel = _per_channel_ranks(rankings, top_n)
    channels = sorted(per_channel)
    if len(channels) < 2:
        return 0.0
    correlations = []
    for i, first in enumerate(channels):
        for second in channels[i + 1 :]:
            correlations.append(
                _spearman(per_channel[first], per_channel[second], top_n)
            )
    return float(np.mean(correlations))


def temporal_compactness(
    candidates: list[ShotCandidate],
    video_by_shot: dict[str, str],
    top_n: int,
) -> float:
    """Share of the top-N candidates that sit in the modal video."""
    top = candidates[:top_n]
    if not top:
        return 0.0
    counts: dict[str, int] = {}
    for candidate in top:
        video = video_by_shot.get(candidate.shot_key)
        if video is None:
            continue
        counts[video] = counts.get(video, 0) + 1
    if not counts:
        return 0.0
    return max(counts.values()) / len(top)


def extract_features(
    spec: QuerySpec,
    candidates: list[ShotCandidate],
    rankings: dict[str, list[ShotCandidate]],
    video_by_shot: dict[str, str],
    cfg: QppConfig,
    top_k: int,
) -> np.ndarray:
    """The feature vector, in FEATURE_NAMES order."""
    scores = [candidate.score for candidate in candidates]
    features = np.array(
        [
            _margin(scores, 2),
            _margin(scores, cfg.margin_top_n),
            channel_agreement(rankings, cfg.agreement_top_n),
            temporal_compactness(candidates, video_by_shot, cfg.compactness_top_n),
            spec.confidence if spec.confidence is not None else 0.0,
            1.0 if spec.source == "fallback" else 0.0,
            min(len(candidates) / top_k, 1.0) if top_k > 0 else 0.0,
        ],
        dtype=np.float64,
    )
    return features


def train_artifact(
    features: np.ndarray,
    labels: np.ndarray,
    target_k: int,
    calibration: dict[str, list[float]] | None = None,
) -> dict:
    """Fit the logistic model and package the JSON artifact.

    scikit-learn is imported here (not at module top) so the service never
    needs it; only the training script does.
    """
    from sklearn.linear_model import LogisticRegression

    classes = np.unique(labels)
    if len(classes) < 2:
        raise QppError(
            "QPP training needs both hits and misses in the fixture "
            f"outcomes; got only label {classes.tolist()}"
        )
    model = LogisticRegression(max_iter=1000)
    model.fit(features, labels)
    return {
        "artifact_version": ARTIFACT_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "coef": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "target_k": target_k,
        "n_cases": int(len(labels)),
        "calibration": calibration,
    }


class QppAdvisor:
    """Scores a query's ranking against the trained artifact."""

    def __init__(self, artifact: dict, cfg: QppConfig) -> None:
        if artifact.get("artifact_version") != ARTIFACT_VERSION:
            raise QppError(
                f"QPP artifact version {artifact.get('artifact_version')!r} "
                f"does not match the supported {ARTIFACT_VERSION}"
            )
        if tuple(artifact.get("feature_names", ())) != FEATURE_NAMES:
            raise QppError(
                "QPP artifact was trained on different features "
                f"({artifact.get('feature_names')}); retrain with "
                "scripts/train_qpp.py"
            )
        self._coef = np.asarray(artifact["coef"], dtype=np.float64)
        self._intercept = float(artifact["intercept"])
        self._target_k = int(artifact["target_k"])
        self._calibration = artifact.get("calibration")
        self._cfg = cfg

    @classmethod
    def load(cls, path: Path, cfg: QppConfig) -> QppAdvisor:
        try:
            artifact = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise QppError(f"cannot read QPP artifact at {path}: {exc}") from exc
        return cls(artifact, cfg)

    @property
    def target_k(self) -> int:
        return self._target_k

    def probability(self, features: np.ndarray) -> float:
        raw = 1.0 / (1.0 + math.exp(-(float(self._coef @ features) + self._intercept)))
        if self._calibration:
            raw = float(
                np.interp(raw, self._calibration["x"], self._calibration["y"])
            )
        return min(max(raw, 0.0), 1.0)

    def advise(
        self,
        spec: QuerySpec,
        candidates: list[ShotCandidate],
        rankings: dict[str, list[ShotCandidate]],
        video_by_shot: dict[str, str],
    ) -> dict:
        """The advisor block for the search response."""
        features = extract_features(
            spec, candidates, rankings, video_by_shot, self._cfg, self._target_k
        )
        p_hit = self.probability(features)
        hint = None
        if p_hit < self._cfg.hint_threshold:
            hint = (
                f"target unlikely in top {self._target_k} "
                f"(P={p_hit:.0%}); consider escalating"
            )
        return {"p_hit": p_hit, "target_k": self._target_k, "hint": hint}


def save_artifact(artifact: dict, path: Path) -> None:
    write_json_atomic(path, artifact)
