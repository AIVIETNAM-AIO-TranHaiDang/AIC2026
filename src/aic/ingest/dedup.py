"""Keyframe filtering: blank-frame rejection and near-duplicate removal.

News broadcasts repeat studio shots, station idents, and transition graphics.
Two cheap filters run at ingest time (note 09, Phase 2):

- an entropy gate drops black/blank/graphic-card frames,
- perceptual-hash de-duplication drops near-identical keyframes within a video
  and exact repeats across videos.

Embedding-cosine de-duplication (the second, semantic pass) is a pure function
here; the ingest pipeline applies it in Phase 3 once embeddings exist.
"""

from __future__ import annotations

import imagehash
import numpy as np
from PIL import Image


def entropy_bits(image: np.ndarray) -> float:
    """Shannon entropy (bits) of the grayscale intensity histogram."""
    if image.ndim == 3:
        gray = image.mean(axis=2)
    else:
        gray = image
    histogram, _ = np.histogram(gray, bins=256, range=(0, 255))
    total = histogram.sum()
    if total == 0:
        return 0.0
    probabilities = histogram[histogram > 0] / total
    return float(-(probabilities * np.log2(probabilities)).sum())


def is_low_information(image: np.ndarray, min_entropy_bits: float) -> bool:
    return entropy_bits(image) < min_entropy_bits


class PHashDeduper:
    """Perceptual-hash duplicate detector.

    Within one video, a new keyframe is a duplicate if its hash is within
    ``max_distance`` (Hamming) of any accepted keyframe of that video — a
    linear scan, acceptable at per-video keyframe counts. Across videos only
    exact hash repeats are dropped (station idents, shared intros); anything
    looser would need an ANN structure and risks dropping legitimate
    near-identical news scenes from different broadcasts.
    """

    def __init__(self, max_distance: int) -> None:
        if max_distance < 0:
            raise ValueError("max_distance must be non-negative")
        self._max_distance = max_distance
        self._per_video: dict[str, list[imagehash.ImageHash]] = {}
        self._global_exact: set[str] = set()

    def is_duplicate(self, video_id: str, image: np.ndarray) -> bool:
        """Check an image and, when new, remember it."""
        candidate = imagehash.phash(Image.fromarray(image))
        exact_key = str(candidate)
        if exact_key in self._global_exact:
            return True
        for accepted in self._per_video.get(video_id, []):
            if candidate - accepted <= self._max_distance:
                return True
        self._per_video.setdefault(video_id, []).append(candidate)
        self._global_exact.add(exact_key)
        return False


def cosine_duplicate_indices(vectors: np.ndarray, threshold: float) -> set[int]:
    """Indices of rows that duplicate an earlier row at cosine >= threshold.

    Pure math over L2-normalisable vectors; the Phase 3 pipeline feeds it
    keyframe embeddings for the semantic de-duplication pass. Quadratic in the
    number of rows — callers batch per video, not per corpus.
    """
    if vectors.ndim != 2:
        raise ValueError(f"expected [n, d] vectors, got shape {vectors.shape}")
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("zero-norm vector cannot be compared by cosine")
    normalised = vectors / norms
    similarity = normalised @ normalised.T
    duplicates: set[int] = set()
    n = len(vectors)
    for j in range(1, n):
        if j in duplicates:
            continue
        for i in range(j):
            if i in duplicates:
                continue
            if similarity[i, j] >= threshold:
                duplicates.add(j)
                break
    return duplicates
