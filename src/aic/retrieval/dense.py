"""The dense visual channel: queries against the keyframe vector index.

Serves two query forms with one index: text-to-image (KIS-T/C through the
encoder's text tower) and image-to-image (KIS-V example clips through the
image tower, several sampled frames averaged into one query vector).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from aic.config import QueryClipConfig
from aic.embed.encoders import ImageTextEncoder
from aic.eval.fixture import QueryCase
from aic.index.vector import VectorIndex
from aic.ingest.video import iter_frames, probe_video
from aic.retrieval.fusion import DENSE_VISUAL, ShotCandidate, query_text

logger = logging.getLogger(__name__)


class DenseVisualChannel:
    """Ranks shots by their best-matching keyframe."""

    def __init__(
        self,
        encoder: ImageTextEncoder,
        index: VectorIndex,
        keyframe_meta: dict[str, tuple[str, int, int]],
        fixture_dir: Path | None = None,
        query_clip: QueryClipConfig | None = None,
        keyframe_oversample: int = 1,
    ) -> None:
        """``keyframe_meta`` maps keyframe_id -> (video_id, shot_id, ts_ms).

        ``fixture_dir`` anchors relative KIS-V clip paths; without it only
        text queries are supported. ``query_clip`` controls KIS-V clip
        sampling; None uses the config model's defaults.
        ``keyframe_oversample`` searches that many times ``top_k`` keyframes
        before the per-shot collapse, so the channel still returns ``top_k``
        shots (several keyframes of one shot otherwise crowd out distinct
        shots; see retrieval.keyframe_oversample in the config).
        """
        self._encoder = encoder
        self._index = index
        self._keyframe_meta = keyframe_meta
        self._fixture_dir = fixture_dir
        self._query_clip = query_clip or QueryClipConfig()
        self._oversample = max(1, keyframe_oversample)

    @property
    def name(self) -> str:
        return DENSE_VISUAL

    def supports(self, case: QueryCase) -> bool:
        if case.kind == "kis_v":
            return self._fixture_dir is not None
        return bool(query_text(case))

    def rank(self, case: QueryCase, top_k: int) -> list[ShotCandidate]:
        query = self._encode_query(case)
        return self._rank_vectors(query, top_k)[0]

    def rank_texts(self, texts: list[str], top_k: int) -> list[list[ShotCandidate]]:
        """Rank several text queries with one encoder call (batch dispatch)."""
        queries = self._encoder.encode_texts(texts)
        return self._rank_vectors(queries, top_k)

    def _rank_vectors(
        self, queries: np.ndarray, top_k: int
    ) -> list[list[ShotCandidate]]:
        hits_per_query = self._index.search(
            queries, top_k * self._oversample, query_model_id=self._encoder.model_id
        )
        rankings = []
        for hits in hits_per_query:
            best_per_shot: dict[str, ShotCandidate] = {}
            for keyframe_id, score in hits:
                meta = self._keyframe_meta.get(keyframe_id)
                if meta is None:
                    logger.warning("keyframe %s has no metadata; dropped", keyframe_id)
                    continue
                video_id, shot_id, timestamp_ms = meta
                shot_key = f"{video_id}:{shot_id}"
                if shot_key not in best_per_shot:
                    best_per_shot[shot_key] = ShotCandidate(
                        shot_key=shot_key, score=score, timestamp_ms=timestamp_ms
                    )
                if len(best_per_shot) >= top_k:
                    break
            rankings.append(
                sorted(best_per_shot.values(), key=lambda c: -c.score)
            )
        return rankings

    def _encode_query(self, case: QueryCase) -> np.ndarray:
        if case.kind == "kis_v":
            return self._encode_clip(case)
        return self._encoder.encode_texts([query_text(case)])

    def _encode_clip(self, case: QueryCase) -> np.ndarray:
        if self._fixture_dir is None:
            raise ValueError("KIS-V requires a fixture_dir for clip paths")
        clip_path = self._fixture_dir / case.clip_path
        clip_cfg = self._query_clip
        info = probe_video(clip_path)
        step = max(info.duration_ms // (clip_cfg.frames + 1), 1)
        targets = sorted({step * (i + 1) for i in range(clip_cfg.frames)})
        # Nearest frame per target (two-pointer over the PTS-ascending
        # decode). A window test here once collected the first N frames of
        # the clip instead — the target windows overlap into one contiguous
        # span — collapsing the query to near-duplicate consecutive frames.
        frames = []
        ti = 0
        prev: tuple[int, np.ndarray] | None = None
        for ts, frame in iter_frames(
            clip_path, width=clip_cfg.decode_width, height=clip_cfg.decode_height
        ):
            while ti < len(targets) and ts >= targets[ti]:
                target = targets[ti]
                if prev is not None and abs(prev[0] - target) <= abs(ts - target):
                    frames.append(prev[1])
                else:
                    frames.append(frame)
                ti += 1
            if ti >= len(targets):
                break
            prev = (ts, frame)
        # Targets beyond the last decoded frame take that last frame.
        while ti < len(targets) and prev is not None:
            frames.append(prev[1])
            ti += 1
        if not frames:
            raise ValueError(f"no frames decodable from query clip {clip_path}")
        vectors = self._encoder.encode_images(frames)
        mean = vectors.mean(axis=0, keepdims=True)
        norm = np.linalg.norm(mean)
        if norm == 0:
            raise ValueError("clip query vectors averaged to zero")
        return (mean / norm).astype(np.float32)
