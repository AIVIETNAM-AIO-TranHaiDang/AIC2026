"""Speech transcription behind a registry, with hallucination gating.

Two backends:

- ``faster_whisper`` — verified against version 1.2.1:
  ``WhisperModel(model_size_or_path, device=..., compute_type=...,
  download_root=...)`` and ``transcribe(path, language=...)`` returning
  ``(segments, info)`` where each segment carries ``start``, ``end``,
  ``text``, ``avg_logprob``, and ``no_speech_prob``. PhoWhisper
  (Vietnamese-tuned Whisper) is served by this same backend after a
  CTranslate2 conversion — point ``model_size`` at the converted directory.
- ``qwen3_asr`` — the official ``qwen_asr`` package (0.0.6), verified
  against its source: ``Qwen3ASRModel.from_pretrained(repo_id,
  forced_aligner=..., forced_aligner_kwargs=..., **kwargs->AutoModel)`` and
  ``transcribe(audio, language=..., return_time_stamps=True)`` accepting a
  ``(np.ndarray, sample_rate)`` tuple and returning ``ASRTranscription``
  objects whose ``time_stamps.items`` are word spans with ``text`` /
  ``start_time`` / ``end_time`` in seconds. Long audio is chunked
  internally at low-energy boundaries. The package pins
  ``transformers==4.57.6`` (matching this project) but its librosa/numba
  tree downgrades numpy, so it installs on the GPU server per
  requirements-gpu.txt, never in this venv.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from aic.chronicle.schema import normalize_vietnamese
from aic.config import AsrConfig
from aic.models_cache import hf_cache_dir, whisper_cache_dir

logger = logging.getLogger(__name__)

# Audio sample rate Qwen3-ASR operates at (SAMPLE_RATE in qwen_asr.inference.utils).
QWEN3_ASR_SAMPLE_RATE = 16000

# The qwen_asr package validates against English language names and does no
# code mapping of its own (verified from its utils: normalize_language_name
# only fixes casing). Map the ISO codes faster-whisper accepts, covering the
# full SUPPORTED_LANGUAGES list of qwen_asr 0.0.6, so one config spelling
# switches backends. Unlisted values pass through unchanged. A None language
# means auto-detect for both backends (verified: faster-whisper 1.2.1
# defaults transcribe(language=None) to detection, and qwen_asr 0.0.6
# declares transcribe(language: Optional[...] = None) with model-side
# language identification).
_ISO_TO_QWEN_LANGUAGE = {
    "ar": "Arabic",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fa": "Persian",
    "fi": "Finnish",
    "fil": "Filipino",
    "fr": "French",
    "hi": "Hindi",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "mk": "Macedonian",
    "ms": "Malay",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sv": "Swedish",
    "th": "Thai",
    "tl": "Filipino",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "yue": "Cantonese",
    "zh": "Chinese",
}


@dataclass(frozen=True)
class AsrSegment:
    start_ms: int
    end_ms: int
    text: str
    confidence: float
    """Exponentiated average log-probability, in (0, 1]."""


def passes_speech_rate(
    text: str, start_ms: int, end_ms: int, min_chars_per_second: float | None
) -> bool:
    """False when a segment says too little for its span (rate gate).

    Music-induced hallucinations decode with HIGH confidence, so the
    logprob/no-speech gates miss them; their giveaway is a long audio span
    carrying a short phrase (a measured case: 70 characters over 45 s). A
    non-positive span keeps the segment — the rate is undefined, and
    dropping on missing timestamps would silently discard good speech.
    """
    if min_chars_per_second is None:
        return True
    span_s = (end_ms - start_ms) / 1000.0
    if span_s <= 0:
        return True
    return len(text.strip()) / span_s >= min_chars_per_second


class AsrBackend(Protocol):
    def transcribe(self, video_path: Path) -> list[AsrSegment]:
        """Transcribe the audio track; empty list when there is none."""
        ...


class FasterWhisperAsr:
    """faster-whisper adapter with music/jingle hallucination gating."""

    def __init__(self, cfg: AsrConfig, models_dir: Path) -> None:
        self._cfg = cfg
        self._download_root = whisper_cache_dir(models_dir)
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            self._download_root.mkdir(parents=True, exist_ok=True)
            self._model = WhisperModel(
                self._cfg.model_size,
                device=self._cfg.device,
                compute_type=self._cfg.compute_type,
                download_root=str(self._download_root),
            )
            logger.info(
                "loaded faster-whisper %s (%s)",
                self._cfg.model_size,
                self._cfg.device,
            )
        return self._model

    def transcribe(self, video_path: Path) -> list[AsrSegment]:
        import math

        model = self._ensure_model()
        try:
            # Anti-hallucination knobs (all config-driven, defaults = the
            # library defaults verified against faster-whisper 1.2.1): VAD
            # keeps music/silence away from the decoder, and disabling
            # previous-text conditioning stops a hallucinated phrase from
            # propagating window to window (the repeated-line failure mode
            # observed on broadcast audio).
            segments, _info = model.transcribe(
                str(video_path),
                language=self._cfg.language,
                vad_filter=self._cfg.vad_filter,
                condition_on_previous_text=self._cfg.condition_on_previous_text,
                hallucination_silence_threshold=(
                    self._cfg.hallucination_silence_threshold
                ),
                no_repeat_ngram_size=self._cfg.no_repeat_ngram_size,
                repetition_penalty=self._cfg.repetition_penalty,
            )
        except (RuntimeError, ValueError) as exc:
            # Videos without an audio stream fail at decode time; that is a
            # legal corpus state recorded by ingest, not a pipeline error.
            logger.warning("ASR skipped for %s: %s", video_path.name, exc)
            return []
        results = []
        gated = 0
        for segment in segments:
            if segment.no_speech_prob > self._cfg.max_no_speech_prob:
                continue
            if segment.avg_logprob < self._cfg.min_avg_logprob:
                continue
            text = normalize_vietnamese(segment.text)
            if not text:
                continue
            start_ms = round(segment.start * 1000)
            end_ms = round(segment.end * 1000)
            if not passes_speech_rate(
                text, start_ms, end_ms, self._cfg.min_chars_per_second
            ):
                gated += 1
                continue
            results.append(
                AsrSegment(
                    start_ms=start_ms,
                    end_ms=end_ms,
                    text=text,
                    confidence=float(math.exp(segment.avg_logprob)),
                )
            )
        if gated:
            logger.info(
                "ASR rate gate dropped %d segment(s) for %s",
                gated,
                video_path.name,
            )
        return results


class Qwen3Asr:
    """Qwen3-ASR adapter producing gap-split segments from word timestamps.

    The model reports no per-segment confidence or no-speech probability, so
    the whisper gating fields do not apply; its robustness on speech-over-
    music is a model property rather than a post-filter. Segments carry
    confidence 1.0.
    """

    def __init__(self, cfg: AsrConfig, models_dir: Path) -> None:
        self._cfg = cfg
        self._cache_dir = hf_cache_dir(models_dir)
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            import torch

            try:
                from qwen_asr import Qwen3ASRModel
            except ImportError as exc:
                raise RuntimeError(
                    "qwen-asr is not installed in this environment; see "
                    "requirements-gpu.txt (its dependency tree downgrades "
                    "numpy, so it stays out of the dev venv)"
                ) from exc

            device = self._cfg.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
            self._model = Qwen3ASRModel.from_pretrained(
                self._cfg.model_id,
                forced_aligner=self._cfg.aligner_id,
                forced_aligner_kwargs={
                    "dtype": dtype,
                    "device_map": device,
                    "cache_dir": str(self._cache_dir),
                },
                dtype=dtype,
                device_map=device,
                cache_dir=str(self._cache_dir),
            )
            logger.info(
                "loaded Qwen3-ASR %s + aligner %s on %s",
                self._cfg.model_id,
                self._cfg.aligner_id,
                device,
            )
        return self._model

    def transcribe(self, video_path: Path) -> list[AsrSegment]:
        from aic.ingest.video import VideoError, decode_audio_mono

        try:
            audio = decode_audio_mono(video_path, QWEN3_ASR_SAMPLE_RATE)
        except VideoError as exc:
            logger.warning("ASR skipped for %s: %s", video_path.name, exc)
            return []
        if audio.size == 0:
            return []
        model = self._ensure_model()
        language = self._cfg.language
        if language is not None:
            language = _ISO_TO_QWEN_LANGUAGE.get(language.lower(), language)
        results = model.transcribe(
            audio=(audio, QWEN3_ASR_SAMPLE_RATE),
            language=language,
            return_time_stamps=True,
        )
        result = results[0]
        if not result.text.strip() or result.time_stamps is None:
            return []
        words = [
            (normalize_vietnamese(item.text), item.start_time, item.end_time)
            for item in result.time_stamps.items
        ]
        return _words_to_segments(
            [w for w in words if w[0]], self._cfg.segment_gap_ms
        )


def _words_to_segments(
    words: list[tuple[str, float, float]],
    gap_ms: int,
) -> list[AsrSegment]:
    """Group word spans (text, start_s, end_s) into segments split on silence.

    Word-level timestamps are too fine for the manifests; joining runs of
    words separated by less than ``gap_ms`` yields utterance-like segments
    that align to shots the same way whisper segments do.
    """
    segments: list[AsrSegment] = []
    current: list[str] = []
    seg_start_ms = seg_end_ms = 0
    for text, start_s, end_s in words:
        start_ms, end_ms = round(start_s * 1000), round(end_s * 1000)
        if current and start_ms - seg_end_ms > gap_ms:
            segments.append(
                AsrSegment(
                    start_ms=seg_start_ms,
                    end_ms=seg_end_ms,
                    text=" ".join(current),
                    confidence=1.0,
                )
            )
            current = []
        if not current:
            seg_start_ms = start_ms
            seg_end_ms = end_ms
        current.append(text)
        seg_end_ms = max(seg_end_ms, end_ms)
    if current:
        segments.append(
            AsrSegment(
                start_ms=seg_start_ms,
                end_ms=seg_end_ms,
                text=" ".join(current),
                confidence=1.0,
            )
        )
    return segments


def align_segments_to_shots(
    segments: list[AsrSegment],
    shots: list[tuple[int, int, int]],
) -> dict[int, str]:
    """Join segment texts per overlapping shot.

    ``shots`` rows are ``(shot_id, start_ms, end_ms)``. A segment spanning a
    boundary contributes to every shot it overlaps (Phase 4 alignment rule):
    speech about a scene should be findable from any shot it covers.
    """
    per_shot: dict[int, list[str]] = {}
    for shot_id, shot_start, shot_end in shots:
        for segment in segments:
            if segment.end_ms > shot_start and segment.start_ms < shot_end:
                per_shot.setdefault(shot_id, []).append(segment.text)
    return {shot_id: " ".join(texts) for shot_id, texts in per_shot.items()}


_ASR_BACKENDS = {
    "faster_whisper": FasterWhisperAsr,
    "qwen3_asr": Qwen3Asr,
}


def build_asr_backend(cfg: AsrConfig, models_dir: Path) -> AsrBackend:
    try:
        backend_cls = _ASR_BACKENDS[cfg.backend]
    except KeyError:
        known = ", ".join(sorted(_ASR_BACKENDS))
        raise ValueError(
            f"unknown ASR backend {cfg.backend!r}; known backends: {known}"
        ) from None
    return backend_cls(cfg, models_dir)
