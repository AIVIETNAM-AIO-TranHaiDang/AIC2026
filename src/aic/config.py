"""Typed configuration for the AIC pipeline.

One YAML file per hardware/deployment profile (configs/t0.yaml, t1.yaml, ...).
Every tunable threshold, model name, path, and batch size lives here; code must
not contain magic values. Unknown keys are rejected so typos fail loudly.

Secrets are never stored in config files. String values may reference
environment variables with ``${VAR_NAME}`` and are resolved at load time.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_ENV_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")

# dtypes NumPy stores natively, so the legacy .npy shard format can hold them.
# Anything else (e.g. bfloat16 via the optional ml_dtypes package) needs
# safetensors, which records the dtype explicitly.
_NPY_SAFE_DTYPES = frozenset(
    {np.dtype("float16"), np.dtype("float32"), np.dtype("float64")}
)


class ConfigError(ValueError):
    """Raised when a config file is missing, malformed, or fails validation."""


def resolve_store_dtype(name: str) -> np.dtype:
    """Parse an embedding store-dtype name to a NumPy dtype.

    Imports ``ml_dtypes`` first if it is installed, so names it registers with
    NumPy (``bfloat16``) resolve; raises :class:`ConfigError` on an unknown name.
    """
    try:
        import ml_dtypes  # noqa: F401  (registers extra dtypes with NumPy)
    except ImportError:
        pass
    try:
        return np.dtype(name)
    except TypeError as exc:
        raise ConfigError(
            f"embed.store_dtype {name!r} is not a known dtype; use e.g. "
            "'float16' / 'float32', or install ml_dtypes for 'bfloat16'"
        ) from exc


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def resolve_endpoints(
    base_url: str | list[str] | None,
    model: str | list[str],
) -> list[tuple[str | None, str]]:
    """Pair endpoint base URLs with the model each one serves.

    A stage may target several OpenAI-compatible servers at once (e.g. one
    box running Qwen, another running Gemma). The pairing rule is:

    - one model, any number of endpoints -> that model is applied to every
      endpoint (broadcast);
    - several models -> the count must equal the number of endpoints, paired
      by position (endpoint[i] serves model[i]).

    A ``None`` base URL means "let the provider preset resolve it" and is a
    single endpoint. Returns ``(base_url, model)`` pairs; the caller builds one
    client per pair.
    """
    # A single comma-separated string expands to a list, so one env var can
    # carry every endpoint of a multi-GPU deploy (LOCAL_LLM_BASE_URL=
    # "http://127.0.0.1:8080/v1,http://127.0.0.1:8081/v1") without editing the
    # YAML — the portable knob scripts/serve_local_llms.py exports.
    if isinstance(base_url, str) and "," in base_url:
        base_url = [part.strip() for part in base_url.split(",") if part.strip()]

    base_urls: list[str | None]
    if base_url is None:
        base_urls = [None]
    elif isinstance(base_url, str):
        base_urls = [base_url]
    else:
        base_urls = list(base_url)
        if not base_urls:
            raise ConfigError("base_url list must not be empty")

    models = [model] if isinstance(model, str) else list(model)
    if not models:
        raise ConfigError("model must not be empty")

    if len(models) == 1:
        models = models * len(base_urls)
    elif len(models) != len(base_urls):
        raise ConfigError(
            f"model has {len(models)} entries but base_url has "
            f"{len(base_urls)} endpoint(s); pass a single model (applied to "
            "every endpoint) or exactly one model per endpoint"
        )
    return list(zip(base_urls, models, strict=True))


class ProjectConfig(StrictModel):
    profile: Literal["t0", "t1", "t2"]
    mode: Literal["offline", "api", "hybrid"]


class PathsConfig(StrictModel):
    data_root: Path
    models_dir: Path | None = Field(
        default=None,
        description="Where downloaded model weights are cached. Unset (the "
        "default) resolves to <data_root>/models, keeping weights inside the "
        "project tree (CLAUDE.md's default); set it to point the cache at a "
        "shared or persistent location reused across runs and services (a "
        "mounted drive, a Kaggle dataset). It is passed to each model library's "
        "explicit cache argument, never a user-level cache like ~/.cache.",
    )
    hf_token_env: str | None = Field(
        default="HF_TOKEN",
        description="NAME of the environment variable holding a Hugging Face "
        "access token, used to download gated weights (e.g. SAM 3.1). The "
        "value is read from the environment, never stored here — the strict, "
        "secret-safe path (mirrors api_key_env). None disables it.",
    )
    hf_token: str | None = Field(
        default=None,
        description="The HF access token VALUE, inline, for convenience. If "
        "you set this you MUST gitignore this config file and never push it "
        "to a remote; committed profiles keep it null. When set it wins over "
        "hf_token_env; otherwise the token is read from that env var.",
    )

    def resolve_hf_token(self) -> str | None:
        """The effective HF token: inline value, else the named env var."""
        if self.hf_token:
            return self.hf_token
        if self.hf_token_env:
            return os.environ.get(self.hf_token_env)
        return None

    @model_validator(mode="after")
    def _default_models_dir(self) -> PathsConfig:
        # Resolve the optional override to the in-project default so every
        # caller can read a concrete path from cfg.paths.models_dir.
        if self.models_dir is None:
            self.models_dir = self.data_root / "models"
        return self

    @property
    def videos_dir(self) -> Path:
        return self.data_root / "videos"

    @property
    def keyframes_dir(self) -> Path:
        return self.data_root / "keyframes"

    @property
    def manifests_dir(self) -> Path:
        return self.data_root / "manifests"

    @property
    def embeddings_dir(self) -> Path:
        return self.data_root / "embeddings"

    @property
    def indexes_dir(self) -> Path:
        return self.data_root / "indexes"


class ShotDetectionConfig(StrictModel):
    model: str = Field(description="Registry name of the shot detector.")
    device: str = Field(description="Torch device string, or 'auto'.")
    threshold: float = Field(
        gt=0.0,
        lt=1.0,
        description="Cut probability threshold. Used by transnetv2; "
        "omnishotcut predicts ranges directly and ignores it.",
    )
    min_shot_duration_ms: int = Field(ge=0)
    checkpoint: str | None = Field(
        default=None,
        description="For model 'omnishotcut': local checkpoint path or a "
        "Hugging Face repo id (the official one is uva-cv-lab/OmniShotCut).",
    )
    checkpoint_filename: str | None = Field(
        default=None,
        description="For model 'omnishotcut': checkpoint filename inside the "
        "HF repo. None keeps the upstream default (OmniShotCut_ckpt.pth).",
    )
    overlap_frames: int = Field(
        default=20,
        ge=0,
        description="For model 'omnishotcut': overlap between adjacent "
        "inference windows (upstream default 20).",
    )
    mode: Literal["clean_shot", "default"] = Field(
        default="clean_shot",
        description="For model 'omnishotcut': 'clean_shot' keeps only "
        "non-transition shots (dissolve/wipe frames fall between shots and "
        "never become keyframes); 'default' keeps every detected shot "
        "including gradual transitions.",
    )

    @model_validator(mode="after")
    def _omnishotcut_needs_checkpoint(self) -> ShotDetectionConfig:
        if self.model == "omnishotcut" and not self.checkpoint:
            raise ValueError("shot detector 'omnishotcut' requires checkpoint")
        return self


class KeyframeConfig(StrictModel):
    positions: list[float] = Field(
        min_length=1,
        description="Fractional positions inside a shot to sample, in [0, 1].",
    )
    max_gap_ms: int = Field(
        gt=0,
        description="Insert extra keyframes so no gap inside a shot exceeds this.",
    )
    motion_min_diff: float = Field(
        ge=0.0,
        description=(
            "Mean absolute low-res frame difference (0-255 scale) above which a "
            "motion peak earns an extra keyframe. 0 disables motion extras."
        ),
    )
    motion_min_shot_ms: int = Field(
        ge=0,
        description="Only shots at least this long are scanned for motion peaks.",
    )
    max_side: int = Field(gt=0, description="Longest side of saved keyframe images.")
    jpeg_quality: int = Field(ge=1, le=100)

    @model_validator(mode="after")
    def _positions_in_range(self) -> KeyframeConfig:
        if any(p < 0.0 or p > 1.0 for p in self.positions):
            raise ValueError("keyframe positions must be within [0, 1]")
        return self


class DedupConfig(StrictModel):
    phash_max_distance: int = Field(
        ge=0,
        description=(
            "Hamming distance at or below which two keyframes in the same video "
            "count as duplicates. Exact matches are also deduplicated across videos."
        ),
    )
    min_entropy_bits: float = Field(
        ge=0.0,
        description="Grayscale entropy below which a frame is dropped as blank.",
    )
    keep_one_per_shot: bool = Field(
        default=True,
        description="When de-duplication would leave a shot with zero "
        "keyframes, keep its first non-blank frame anyway, so every "
        "non-blank shot stays reachable by the visual and caption channels "
        "(a shot without keyframes is invisible to both). Blank shots "
        "(all frames under min_entropy_bits) still get nothing. False "
        "restores the pure-filter behaviour.",
    )


class IngestConfig(StrictModel):
    video_extensions: list[str] = Field(min_length=1)
    shots: ShotDetectionConfig
    keyframes: KeyframeConfig
    dedup: DedupConfig


class EvalConfig(StrictModel):
    k_values: list[int] = Field(min_length=1)
    hit_tolerance_ms: int = Field(
        ge=0,
        description=(
            "A result within this margin outside the labelled ground-truth range "
            "still counts as a hit."
        ),
    )
    miss_penalty_rank: int = Field(
        gt=0,
        description="Rank assigned to queries whose target was never returned.",
    )
    per_rank_cost_s: float = Field(
        ge=0.0,
        description=(
            "Seconds the operator is assumed to spend inspecting one result, used "
            "by the time-to-target proxy."
        ),
    )

    @model_validator(mode="after")
    def _k_positive(self) -> EvalConfig:
        if any(k <= 0 for k in self.k_values):
            raise ValueError("k_values must be positive")
        return self


class EmbedConfig(StrictModel):
    model: str = Field(description="Registry name of the image-text encoder.")
    model_id: str = Field(description="Hugging Face model id.")
    device: str = Field(description="Torch device string, or 'auto'.")
    batch_size: int = Field(gt=0)
    shard_format: Literal["safetensors", "npy"] = Field(
        default="safetensors",
        description="On-disk format for keyframe embedding shards. "
        "'safetensors' (default) records the dtype explicitly and loads "
        "without executing code; 'npy' is the legacy NumPy format. Shards "
        "already written in the other format are not migrated automatically.",
    )
    store_dtype: str | None = Field(
        default=None,
        description="dtype to store embedding shards as. None (default) keeps "
        "the encoder's output dtype (float32). Set e.g. 'float16' to halve "
        "shard size; vectors are upcast to float32 on load for the index. A "
        "dtype NumPy cannot store natively (e.g. 'bfloat16', needs the optional "
        "ml_dtypes package) requires shard_format: safetensors.",
    )

    @model_validator(mode="after")
    def _check_store_dtype(self) -> EmbedConfig:
        if self.store_dtype is not None:
            dtype = resolve_store_dtype(self.store_dtype)
            if self.shard_format == "npy" and dtype not in _NPY_SAFE_DTYPES:
                raise ConfigError(
                    f"embed.store_dtype {self.store_dtype!r} is not a native "
                    "NumPy dtype; set shard_format: safetensors to store it"
                )
        return self


class IndexConfig(StrictModel):
    backend: Literal["faiss", "numpy"]
    device: Literal["auto", "cpu", "cuda"] = Field(
        default="auto",
        description="Where FAISS search runs (Phase 10). 'auto' uses the GPU "
        "only when CUDA is visible AND the installed faiss build has GPU "
        "support (faiss-cpu and faiss-gpu are conflicting distributions of "
        "the same module, so CUDA alone proves nothing); otherwise CPU. "
        "'cuda' fails loudly when GPU support is missing. The numpy backend "
        "ignores this field. Serialization always happens from the CPU copy.",
    )
    gpu_float16: bool = Field(
        default=False,
        description="Store the GPU-resident index in float16, halving VRAM "
        "at a small score-precision cost (flat-index storage option, "
        "GpuIndexFlatConfig.useFloat16). Measure the drift with "
        "scripts/faiss_gpu_parity.py before enabling in competition.",
    )


class AsrConfig(StrictModel):
    backend: str = Field(description="Registry name of the ASR backend.")
    model_size: str | None = Field(
        default=None,
        description="For backend 'faster_whisper': model size or a converted "
        "CTranslate2 model path (e.g. tiny, large-v3, or a PhoWhisper "
        "conversion directory).",
    )
    device: str
    compute_type: str = Field(
        default="auto",
        description="For backend 'faster_whisper': ctranslate2 compute type "
        "(auto, int8, float16, ...).",
    )
    language: str | None = Field(
        default=None,
        description="Language hint as an ISO code (vi). Works unchanged "
        "for both backends: qwen3_asr wants English names (Vietnamese), "
        "and every ISO code for its 30 supported languages is mapped "
        "automatically. None or 'auto' lets either backend detect the "
        "language per video (useful for mixed corpora; a fixed hint is "
        "faster and more stable on a known-language corpus).",
    )

    @field_validator("language")
    @classmethod
    def _auto_means_detect(cls, value: str | None) -> str | None:
        # Both backends spell auto-detection as language=None; accept the
        # natural config spelling too.
        if value is not None and value.lower() == "auto":
            return None
        return value
    max_no_speech_prob: float = Field(
        ge=0.0,
        le=1.0,
        description="Segments above this no-speech probability are dropped "
        "(music/jingle hallucination gate). faster_whisper only; qwen3_asr "
        "reports no such probability.",
    )
    min_avg_logprob: float = Field(
        description="Segments below this average log-probability are dropped. "
        "faster_whisper only."
    )
    vad_filter: bool = Field(
        default=False,
        description="For backend 'faster_whisper': run Silero VAD before "
        "decoding so music/silence stretches are never transcribed — the "
        "primary defence against looped hallucinations on broadcast audio. "
        "Default False is the library default (verified against "
        "faster-whisper 1.2.1); profiles for news corpora should enable it.",
    )
    condition_on_previous_text: bool = Field(
        default=True,
        description="For backend 'faster_whisper': feed the previous window's "
        "text as decoder context. The library default (True, verified against "
        "1.2.1) helps continuity but propagates a hallucinated phrase across "
        "the whole file; set False on music-heavy corpora to stop loops.",
    )
    hallucination_silence_threshold: float | None = Field(
        default=None,
        description="For backend 'faster_whisper': when word timestamps "
        "detect a silence gap longer than this many seconds inside a "
        "segment, the segment is treated as a probable hallucination and "
        "re-decoded past the gap. None (library default) disables the check.",
    )
    no_repeat_ngram_size: int = Field(
        default=0,
        ge=0,
        description="For backend 'faster_whisper': block repeating n-grams "
        "of this size during decoding (ctranslate2 option). 0 (library "
        "default) disables the constraint.",
    )
    repetition_penalty: float = Field(
        default=1.0,
        gt=0.0,
        description="For backend 'faster_whisper': multiplicative penalty on "
        "already-generated tokens. 1.0 (library default) is no penalty; "
        "values slightly above 1.0 discourage the repeated-line failure mode.",
    )
    min_chars_per_second: float | None = Field(
        default=None,
        gt=0.0,
        description="Segments whose text length (characters) divided by "
        "their audio span (seconds) falls below this rate are dropped as "
        "probable hallucinations. Music-induced Whisper hallucinations "
        "carry HIGH decoder confidence (a measured case: one 45 s span at "
        "confidence 0.9 holding a 70-character phrase, ~1.6 chars/s), so "
        "the logprob/no-speech gates cannot catch them; speaking rate can — "
        "real Vietnamese speech runs an order of magnitude faster. None "
        "(default) disables the gate. Applied at transcription AND when the "
        "assembler joins segments to shots, so an already-persisted ASR "
        "manifest heals on the next assemble without re-transcribing.",
    )
    model_id: str | None = Field(
        default=None,
        description="For backend 'qwen3_asr': HF repo id of the ASR model "
        "(e.g. Qwen/Qwen3-ASR-1.7B).",
    )
    aligner_id: str | None = Field(
        default=None,
        description="For backend 'qwen3_asr': HF repo id of the forced "
        "aligner that provides timestamps "
        "(e.g. Qwen/Qwen3-ForcedAligner-0.6B).",
    )
    segment_gap_ms: int = Field(
        default=800,
        gt=0,
        description="For backend 'qwen3_asr': aligned words separated by a "
        "silence longer than this are split into separate segments.",
    )

    @model_validator(mode="after")
    def _backend_fields(self) -> AsrConfig:
        if self.backend == "faster_whisper" and not self.model_size:
            raise ValueError("ASR backend 'faster_whisper' requires model_size")
        if self.backend == "qwen3_asr" and not (self.model_id and self.aligner_id):
            raise ValueError(
                "ASR backend 'qwen3_asr' requires model_id and aligner_id "
                "(timestamps need the forced aligner)"
            )
        return self


class SamplingFields(StrictModel):
    """Sampling knobs shared by every module that talks to an
    OpenAI-compatible chat-completions endpoint (captioning, VLM OCR, the
    Query Cortex).

    The fields are optional; None omits the parameter so the server applies
    its own defaults. ``top_k`` is not part of the OpenAI API
    (api.openai.com rejects unknown arguments) but is accepted by
    self-deployed servers such as vLLM, which takes it as a top-level
    request field (verified against the vLLM chat protocol source).
    """

    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. None omits the parameter.",
    )
    top_p: float | None = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description="Nucleus sampling probability mass. None omits it.",
    )
    top_k: int | None = Field(
        default=None,
        gt=0,
        description="Top-k sampling cutoff. Self-deployed servers only "
        "(vLLM and friends); api.openai.com rejects it.",
    )


class MultiEndpointFields(StrictModel):
    """Fan-out knobs for a stage that may target several OpenAI-compatible
    endpoints at once (see :func:`resolve_endpoints`).

    Defaults reproduce the original single-endpoint, serial behaviour exactly:
    one endpoint with ``num_parallel=1`` runs one request at a time.
    """

    num_parallel: int = Field(
        default=1,
        ge=1,
        description="Concurrent requests sent to EACH endpoint. Only the "
        "batch stages (chronicle caption, VLM OCR) use it; the online "
        "single-call stages (Query Cortex, VLM verify) issue one request per "
        "action and ignore it. Keep at 1 for llama.cpp vision (its clip "
        "attaches to slot 0 only); raise for vLLM, which batches efficiently.",
    )
    host_max_retries: int = Field(
        default=3,
        ge=1,
        description="Consecutive request failures on ONE endpoint before it "
        "is evicted from the pool for the rest of the run; a success resets "
        "the counter. Only connection/transport errors count, never a valid "
        "reply with unusable content.",
    )
    rate_limit_max_retries: int = Field(
        default=3,
        ge=0,
        description="Same-endpoint retries for HTTP 429 (rate limited) and "
        "503 (overloaded) — the alive-but-unwilling statuses — before the "
        "request counts as a transport failure. Each wait honours the "
        "Retry-After header when present, else exponential backoff from "
        "rate_limit_backoff_s. Other errors still fail over immediately. 0 "
        "disables the retry — without it a burst of 429s from a rate-limited "
        "API evicts a perfectly healthy endpoint.",
    )
    rate_limit_backoff_s: float = Field(
        default=2.0,
        gt=0.0,
        description="Base wait before the first rate-limit retry; doubles "
        "each further retry (2s, 4s, 8s, ...). Ignored when the server "
        "sends Retry-After.",
    )


class EndpointRequestConfig(SamplingFields, MultiEndpointFields):
    """Sampling, fan-out, plus frame-encoding knobs for modules that send
    images to an OpenAI-compatible endpoint."""

    image_jpeg_quality: int = Field(
        default=85,
        ge=1,
        le=100,
        description="JPEG quality for frames encoded into the request. "
        "Higher is sharper but larger (more tokens/latency on VLM side).",
    )


class OcrConfig(EndpointRequestConfig):
    backend: str = Field(description="Registry name of the OCR backend.")
    languages: list[str] = Field(
        min_length=1,
        description="ISO codes for backend 'easyocr', which has no "
        "language detection: Reader(lang_list) is a mandatory argument "
        "(verified against 1.7.2), so an explicit list is a library "
        "constraint, not a preference. The 'vlm' backend is prompt-driven "
        "and ignores this field.",
    )
    min_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Lines below this confidence are dropped. Applies to "
        "backends that report per-line confidence (easyocr); VLM backends "
        "do not, so they ignore it.",
    )
    base_url: str | list[str] | None = Field(
        default=None,
        description="For backend 'vlm': base URL of an OpenAI-compatible "
        "server hosting an OCR-capable VLM (e.g. PaddleOCR-VL served by "
        "vLLM, or a general VLM such as Qwen3-VL). A list fans OCR out over "
        "several endpoints (see chronicle.caption for the pairing rule).",
    )
    model: str | list[str] | None = Field(
        default=None,
        description="For backend 'vlm': model name as served by the "
        "endpoint. A single model is applied to every endpoint; a list must "
        "have one entry per endpoint.",
    )
    api_key_env: str | None = Field(
        default=None,
        description="For backend 'vlm': environment variable holding the "
        "API key. Optional; if unset or empty the request is sent without "
        "authentication (typical for self-deployed servers).",
    )
    prompt: str = Field(
        default="OCR:",
        description="For backend 'vlm': the instruction sent with each "
        "frame. The default is PaddleOCR-VL's documented element-level "
        "recognition prompt; override for general-purpose VLMs.",
    )
    timeout_s: float = Field(default=60.0, gt=0.0)
    max_output_tokens: int = Field(default=1024, gt=0)
    batch_size: int = Field(
        default=128,
        gt=0,
        description="For backend 'vlm': keyframes dispatched to the endpoint "
        "pool per round before results are written and the manifest advances. "
        "Bounds decoded frames in memory and the resume granularity; it does "
        "not cap concurrency (num_parallel does). Ignored by 'easyocr'.",
    )

    @model_validator(mode="after")
    def _vlm_needs_endpoint(self) -> OcrConfig:
        if self.backend == "vlm":
            if not self.base_url or not self.model:
                raise ValueError("OCR backend 'vlm' requires base_url and model")
            resolve_endpoints(self.base_url, self.model)
        return self


class CaptionConfig(EndpointRequestConfig):
    enabled: bool = Field(
        description="Captions are optional: a Chronicle without them is legal "
        "and the system degrades to OCR/ASR channels."
    )
    provider: Literal["openai", "gemini", "local"]
    base_url: str | list[str] | None = Field(
        default=None,
        description="Endpoint base URL. Required for provider 'local' (a "
        "self-deployed OpenAI-compatible server such as vLLM/Ollama/TGI); "
        "preset for openai/gemini but overridable (e.g. a proxy). A list "
        "fans captioning out across several endpoints kept busy in parallel; "
        "pass one model for all of them, or one model per endpoint.",
    )
    model: str | list[str]
    api_key_env: str = Field(
        description="Name of the environment variable holding the API key. "
        "For provider 'local' the key is optional: if the variable is unset "
        "the request is sent without authentication."
    )
    thinking: Literal["default", "off", "minimal", "low", "medium", "high"] = Field(
        default="default",
        description="Reasoning effort for the captioning call. 'default' "
        "omits the parameter. Gemini <3 maps to thinking_budget (off=0); "
        "Gemini >=3 always thinks and maps to thinking_level, where "
        "off/minimal clamp to the lowest level. OpenAI maps to "
        "reasoning_effort. Provider 'local' maps off/minimal to "
        "chat_template_kwargs {enable_thinking: false} and low/medium/high "
        "to {enable_thinking: true} (Qwen-style templates on vLLM; servers "
        "that do not know the kwarg should stay on 'default'). Reasoning "
        "text a local model still emits (<think> blocks or a separate "
        "reasoning field) is stripped from the reply before parsing.",
    )
    system_prompt: str | None = Field(
        default=None,
        description="Overrides the built-in captioning system prompt "
        "(which targets news broadcasts and English captions). Any "
        "override must still demand a JSON object with the fields "
        "caption_en, actions, scene, and confidence - the parser "
        "validates that schema and rejects replies without it (plus "
        "caption_vi when request_caption_vi is on).",
    )
    request_caption_vi: bool = Field(
        default=False,
        description="Ask the captioner for a native Vietnamese caption_vi "
        "field alongside caption_en (the built-in prompt gains the field; "
        "a system_prompt override must demand it too). The assembler "
        "prefers this native caption_vi and falls back to offline "
        "translation of caption_en for shots without one when "
        "chronicle.translate.enabled is set — the two sources compose. "
        "Native Vietnamese reads better but costs output tokens per shot; "
        "translation is free of API cost but goes through the MT model.",
    )
    frames_per_shot: int = Field(gt=0)
    batch_size: int = Field(
        default=64,
        gt=0,
        description="Shots dispatched to the endpoint pool per round before "
        "their results are written and the manifest advances. Bounds decoded "
        "frames held in memory and the resume granularity; it does not cap "
        "concurrency (that is num_parallel per endpoint). Lower it on a "
        "memory-constrained box.",
    )
    timeout_s: float = Field(gt=0.0)
    max_output_tokens: int = Field(gt=0)

    @model_validator(mode="after")
    def _local_needs_base_url(self) -> CaptionConfig:
        if self.provider == "local" and not self.base_url:
            raise ValueError("caption provider 'local' requires base_url")
        resolve_endpoints(self.base_url, self.model)
        if self.top_k is not None and self.provider != "local":
            raise ValueError(
                "top_k is only supported by self-deployed servers (provider "
                "'local'); api.openai.com rejects unknown arguments and the "
                "Gemini compatibility layer does not document it"
            )
        return self


class TranslateConfig(StrictModel):
    enabled: bool
    model_id: str
    device: str
    src_lang: str = Field(
        description="NLLB language code, e.g. eng_Latn. NLLB has no "
        "language detection: the tokenizer needs the source token and "
        "generation is forced to the target token, so both codes are "
        "architecture requirements, not preferences."
    )
    tgt_lang: str = Field(description="NLLB language code, e.g. vie_Latn.")
    batch_size: int = Field(gt=0)
    max_new_tokens: int = Field(
        default=256,
        gt=0,
        description="Generation cap per translated text. Without an "
        "explicit cap the model's own default silently truncates long "
        "inputs (NLLB defaults to 200 total tokens).",
    )


# The Chronicle entity buckets, mirroring aic.chronicle.schema.Entities
# (config cannot import schema — schema imports StrictModel from here). A
# format contract, not a tunable: the NER label_map keys must be a subset.
_ENTITY_BUCKETS = ("persons", "orgs", "places")

# Ledger fields whose extractor is implemented in the current phase; the
# LedgerConfig.fields list is validated against this so a not-yet-built field
# (e.g. people_count before phase 4) fails loudly instead of silently missing.
# Phase 1 ships the two free-tier text fields; later phases extend this tuple
# as their extractors land (scene_time, people_count, salient_objects, ...).
LEDGER_FIELDS = (
    "screen_text",
    "spoken_entities",
    "people_count",
    "salient_objects",
)

# Detector-counter backends with a verified adapter shipped in
# aic.vqa.counter. The SAM 3 family (sam3.1 default, sam3 switch) shares the
# transformers Sam3Model API. Other detector/counters from the plan (countvid,
# countgd++, rt-counter) are registry extension points: each needs its own
# verify-before-use pass on GPU before an adapter is added here — they are NOT
# listed until implemented, so selecting one that has no adapter fails loudly.
COUNTER_BACKENDS = ("sam3.1", "sam3")


class EntityGlinerConfig(StrictModel):
    """Local GLiNER backend for spoken-entity NER (language-general).

    GLiNER is a generalist zero-shot NER model: it extracts any entity type
    named at inference time, works across languages (the multilingual
    checkpoint), and runs on CPU. Chosen over a Vietnamese-only tool because
    the corpus is bilingual (note 17 §4b). Weights land in paths.models_dir.
    """

    model_id: str = Field(
        default="urchade/gliner_multi-v2.1",
        description="HuggingFace GLiNER checkpoint. The default is the "
        "multilingual, Apache-2.0 model; any GLiNER-family id trades size for "
        "accuracy. Avoid urchade/gliner_multi (cc-by-nc, non-commercial).",
    )
    device: str = Field(
        default="auto",
        description="Torch device: 'auto' (cuda if available else cpu), "
        "'cpu', or an explicit 'cuda:N'. Passed to GLiNER as map_location.",
    )
    label_map: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "persons": ["person"],
            "orgs": ["organization"],
            "places": ["location"],
        },
        description="Maps each Chronicle entity bucket (persons/orgs/places) "
        "to the GLiNER zero-shot labels that fill it. Reusable without "
        "retraining: add labels or buckets here, never in code.",
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum span confidence to keep an entity; the GLiNER "
        "model-card default is 0.5.",
    )

    @model_validator(mode="after")
    def _buckets_known(self) -> EntityGlinerConfig:
        unknown = set(self.label_map) - set(_ENTITY_BUCKETS)
        if unknown:
            raise ValueError(
                f"entity label_map keys {sorted(unknown)} are not Chronicle "
                f"entity buckets {list(_ENTITY_BUCKETS)}"
            )
        return self


class EntityLlmConfig(SamplingFields, MultiEndpointFields):
    """LLM backend for spoken-entity NER over an OpenAI-compatible endpoint.

    Higher accuracy than the local GLiNER model when a capable text LLM is
    deployed, at the cost of needing that endpoint. Reuses the shared
    EndpointPool (failover, eviction, rate-limit backoff). Disabled by
    default; the ledger falls back to GLiNER when this is off.
    """

    enabled: bool = Field(
        default=False,
        description="True routes NER through this endpoint (or lets "
        "entity_backend 'auto' prefer it). False keeps the local GLiNER path.",
    )
    provider: Literal["openai", "gemini", "local"] = "local"
    base_url: str | list[str] | None = Field(
        default=None,
        description="Endpoint base URL. Required for provider 'local'; a list "
        "gives several endpoints for load-balancing and failover.",
    )
    model: str | list[str] = Field(
        default="",
        description="Model name(s) on the endpoint. Required when enabled.",
    )
    api_key_env: str | None = Field(
        default=None,
        description="Environment variable holding the API key; optional for "
        "provider 'local'.",
    )
    timeout_s: float = Field(default=30.0, gt=0.0)
    max_output_tokens: int = Field(default=400, gt=0)
    system_prompt: str | None = Field(
        default=None,
        description="Overrides the built-in NER prompt. Any override must "
        "still demand the {persons, orgs, places} JSON object the parser "
        "validates.",
    )

    @model_validator(mode="after")
    def _endpoint_constraints(self) -> EntityLlmConfig:
        if not self.enabled:
            return self
        if self.provider == "local" and not self.base_url:
            raise ValueError("ledger entity llm provider 'local' requires base_url")
        if not self.model:
            raise ValueError("ledger entity llm requires model when enabled")
        resolve_endpoints(self.base_url, self.model)
        return self


class CounterConfig(StrictModel):
    """Detector-counter backend (note 18 phase 4), off by default.

    Fills the ledger's ``people_count`` / ``salient_objects`` offline and powers
    the online count program; its masks/boxes define the zoom crop. The backend
    is registry-selectable (no stage hardcodes one model); the shipped, verified
    set is the SAM 3 family (``COUNTER_BACKENDS``). Off leaves the ledger's free
    text tier unchanged.
    """

    enabled: bool = Field(
        default=False,
        description="Master switch for the offline count pass. False leaves "
        "the ledger's text-only fields (screen_text, spoken_entities) "
        "unchanged.",
    )
    backend: str = Field(
        default="sam3.1",
        description="Detector-counter backend, from the shipped registry "
        f"{list(COUNTER_BACKENDS)}: 'sam3.1' (default; concept segmentation + "
        "tracking, ~halved VRAM vs SAM 3, T4-viable) or 'sam3' (the 11/2025 "
        "release). Both are HF-gated (request access, supply paths.hf_token). "
        "Other detector/counters are extension points added after their own "
        "verify pass.",
    )
    model_id: str = Field(
        default="facebook/sam3.1",
        description="HF repo id for the backend checkpoint (gated for the SAM "
        "family). Cached under paths.models_dir; downloaded with the resolved "
        "HF token.",
    )
    device: str = Field(
        default="auto",
        description="'auto' (cuda if available else cpu), 'cuda', 'cuda:N', or "
        "'cpu'. The offline stage fans one worker per GPU (data-parallel).",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Instance score threshold: a detected instance counts only "
        "when its confidence is at least this. Passed to the backend's "
        "post-processing (SAM 3: post_process_instance_segmentation threshold).",
    )
    mask_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Mask binarisation threshold (a mask pixel is foreground "
        "when its probability is at least this). A DIFFERENT knob from "
        "'confidence': raising confidence filters whole instances, raising "
        "this erodes every mask (and the boxes derived from it). 0.5 is the "
        "transformers post-processing default; leave it unless masks bleed.",
    )
    concepts: list[str] = Field(
        default_factory=lambda: ["person"],
        min_length=1,
        description="Concept noun phrases the detector counts per shot, filling "
        "salient_objects (and people_count from the person concept). "
        "Corpus-tunable (a news corpus: person, car, motorbike, flag, boat, "
        "animal, ...); the question's concept is resolved at query time.",
    )
    person_concept: str = Field(
        default="person",
        description="Which entry of 'concepts' is the person count written to "
        "people_count. Must be one of concepts; language-neutral default.",
    )
    max_batch: int = Field(
        default=4,
        gt=0,
        description="Keyframe images sent to the detector per forward pass. "
        "Small default sized for a 16 GB T4: SAM 3.1 fp16 is ~10-12 GB for a "
        "single image but a 16-image batch can reach 18-22 GB and OOM; raise "
        "on a larger GPU. Config the batch, never assume the box.",
    )
    image_size: int | None = Field(
        default=None,
        description="Optional square inference resolution override. None uses "
        "the backend default (SAM 3 is trained at 1008 px; lowering trades "
        "accuracy for speed/VRAM per the model card).",
    )

    @field_validator("backend")
    @classmethod
    def _backend_shipped(cls, value: str) -> str:
        if value not in COUNTER_BACKENDS:
            raise ValueError(
                f"counter.backend {value!r} has no shipped adapter; "
                f"implemented: {list(COUNTER_BACKENDS)}"
            )
        return value

    @model_validator(mode="after")
    def _person_in_concepts(self) -> CounterConfig:
        if self.person_concept not in self.concepts:
            raise ValueError(
                f"counter.person_concept {self.person_concept!r} must be one "
                f"of counter.concepts {self.concepts}"
            )
        return self


class LedgerConfig(StrictModel):
    """The offline Answer Ledger pass (note 18 §2), off by default.

    When enabled, the Chronicle gains a language-general NER stage that fills
    ChronicleRecord.entities, and a per-shot ledger.jsonl of typed factoids is
    built for the VQA Track A lookup. Every field is optional; a profile can
    enable only the free-tier text fields and skip heavy backends.
    """

    enabled: bool = Field(
        default=False,
        description="Master switch. False leaves the KIS pipeline "
        "bit-identical (no NER stage, no ledger).",
    )
    fields: list[str] = Field(
        default_factory=lambda: ["screen_text", "spoken_entities"],
        min_length=1,
        description="Which ledger factoid fields to populate. Validated "
        f"against the implemented set {list(LEDGER_FIELDS)}.",
    )
    entity_backend: Literal["auto", "llm", "gliner"] = Field(
        default="auto",
        description="NER backend for spoken_entities: 'gliner' (local, "
        "multilingual, zero-shot, CPU), 'llm' (an endpoint, highest "
        "accuracy), or 'auto' (llm when ledger.llm.enabled, else gliner).",
    )
    gliner: EntityGlinerConfig = EntityGlinerConfig()
    llm: EntityLlmConfig = EntityLlmConfig()
    counter: CounterConfig = CounterConfig()
    min_ocr_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Drop OCR lines below this confidence from screen_text. "
        "0.0 (default) keeps every line; raise to suppress low-confidence "
        "reads from the factoid text.",
    )
    row_band_factor: float = Field(
        default=0.6,
        gt=0.0,
        description="Reading-order sort: start a new text row (band) when a "
        "line's y-centre exceeds the current band's by this fraction of the "
        "median line height. Lower splits rows more eagerly.",
    )

    @field_validator("fields")
    @classmethod
    def _fields_known(cls, value: list[str]) -> list[str]:
        unknown = [f for f in value if f not in LEDGER_FIELDS]
        if unknown:
            raise ValueError(
                f"ledger fields {unknown} are not implemented; known: "
                f"{list(LEDGER_FIELDS)}"
            )
        return value

    @model_validator(mode="after")
    def _backend_available(self) -> LedgerConfig:
        if self.entity_backend == "llm" and not self.llm.enabled:
            raise ValueError(
                "ledger.entity_backend 'llm' requires ledger.llm.enabled; use "
                "'gliner' or 'auto' otherwise"
            )
        count_fields = {"people_count", "salient_objects"}
        if count_fields & set(self.fields) and not self.counter.enabled:
            raise ValueError(
                "ledger.fields include a count field "
                f"({sorted(count_fields & set(self.fields))}) but "
                "ledger.counter.enabled is false; enable the counter or drop "
                "the field"
            )
        return self

    def resolved_entity_backend(self) -> Literal["llm", "gliner"]:
        """Concrete backend after resolving 'auto'."""
        if self.entity_backend == "auto":
            return "llm" if self.llm.enabled else "gliner"
        return self.entity_backend


class ChronicleConfig(StrictModel):
    asr: AsrConfig
    ocr: OcrConfig
    caption: CaptionConfig
    translate: TranslateConfig
    ledger: LedgerConfig = LedgerConfig()
    overlay_min_recurrence: float | None = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description="Persistent-overlay OCR split (note 15): a normalized "
        "OCR line appearing in at least this fraction of a video's shots is "
        "moved from the shot's scene-text evidence into overlay_ocr "
        "(watermarks, news tickers, program bugs). Overlay text stops "
        "competing as per-shot evidence — where it drowned real scene text — "
        "but is still indexed per VIDEO for the retrieval video prior "
        "(retrieval.temporal.video_prior_weight). None (default) keeps every "
        "line as shot evidence, the pre-split behaviour.",
    )


class DenseTextConfig(StrictModel):
    """Optional override for the semantic (dense) channel embedder.

    The literal (sparse) channel always uses the top-level textstack
    embedder, which must produce learned-sparse weights (BGE-M3).
    """

    model: str = Field(description="Registry name of the dense text embedder.")
    model_id: str
    device: str
    batch_size: int = Field(gt=0)
    query_instruction: str = Field(
        default="Given a Vietnamese news video search query, retrieve the "
        "shot descriptions that match the queried moment",
        description="Instruction prepended to queries (not documents) by "
        "instruction-tuned embedders such as Qwen3-Embedding.",
    )
    max_length: int = Field(
        default=8192,
        gt=0,
        description="Tokenizer truncation length for documents and queries.",
    )


class TextStackConfig(StrictModel):
    model: str = Field(description="Registry name of the text embedder.")
    model_id: str
    device: str
    batch_size: int = Field(gt=0)
    dense: DenseTextConfig | None = Field(
        default=None,
        description="When set, the semantic dense index/channel uses this "
        "embedder instead of the top-level one; the sparse channel keeps "
        "the top-level embedder. This is the dense-embedder ablation switch.",
    )

    @model_validator(mode="after")
    def _top_level_must_produce_sparse(self) -> TextStackConfig:
        if self.model == "qwen3_embedding":
            raise ValueError(
                "qwen3_embedding is dense-only; put it under textstack.dense "
                "and keep a sparse-capable embedder (bge_m3) at the top level"
            )
        return self


class FusionConfig(StrictModel):
    rrf_k: int = Field(
        gt=0, description="Reciprocal-rank-fusion constant (rank smoothing)."
    )
    top_k_per_channel: int = Field(gt=0)
    channel_weights: dict[str, float] = Field(
        min_length=1,
        description="Per-channel RRF weight; a channel absent here is disabled, "
        "which is how ablations toggle channels.",
    )

    @model_validator(mode="after")
    def _weights_positive(self) -> FusionConfig:
        for name, weight in self.channel_weights.items():
            if weight <= 0:
                raise ValueError(f"channel weight for {name!r} must be positive")
        return self


class DispatchConfig(StrictModel):
    """Per-field weight multipliers for spec-to-channel dispatch.

    A sub-query's RRF weight is ``channel_weight * field_weight`` (times the
    paraphrase multiplier for paraphrases), so ablations can tune fields and
    channels independently.
    """

    visual_weight: float = Field(
        default=1.0,
        gt=0.0,
        description="visual_phrases -> dense visual + semantic dense.",
    )
    entity_weight: float = Field(
        default=0.8,
        gt=0.0,
        description="entity_terms -> semantic dense + literal sparse.",
    )
    ocr_literal_weight: float = Field(
        default=1.5,
        gt=0.0,
        description="ocr_literals -> literal sparse. Boosted above the other "
        "fields so a pure on-screen-text query can win fusion outright "
        "(note 09, Phase 5/6 edge case).",
    )
    asr_weight: float = Field(
        default=1.0,
        gt=0.0,
        description="asr_phrases -> literal sparse + semantic dense.",
    )
    paraphrase_weight: float = Field(
        default=0.5,
        gt=0.0,
        description="Multiplier applied to paraphrases of visual_phrases.",
    )
    negation_filter: bool = Field(
        default=True,
        description="Drop candidates whose evidence text contains a negated "
        "phrase. Negations are carried as filters, never embedded "
        "(embedding negation is unreliable; note 09 Phase 6).",
    )
    raw_query_weight: float = Field(
        default=0.0,
        ge=0.0,
        description="When > 0, the RAW (untranslated) query text joins the "
        "RRF race as one extra sub-query against the semantic dense and "
        "literal sparse channels at channel_weight * this. Both channels "
        "embed with the multilingual BGE-M3, so a Vietnamese query matches "
        "Vietnamese evidence (caption_vi, ASR transcripts, on-screen text) "
        "directly — removing the VI->EN translation single point of failure "
        "from the text channels (note 15). 0 (default) keeps the "
        "translation-only behaviour. The dense visual channel never gets "
        "the raw query: its text tower is English-trained.",
    )


class CortexConfig(SamplingFields, MultiEndpointFields):
    """The Query Cortex: LLM compilation of a raw query into a typed spec.

    Provider-agnostic like the captioner (OpenAI-compatible endpoint); the
    default is the Gemini compatibility endpoint. When disabled (or on any
    LLM failure) the raw query becomes a single visual phrase, so the system
    is never dumber than the Phase 5 baseline.
    """

    enabled: bool = Field(
        default=True,
        description="False skips the LLM entirely and always uses the "
        "fallback spec (offline mode).",
    )
    provider: Literal["openai", "gemini", "local"] = "gemini"
    base_url: str | list[str] | None = Field(
        default=None,
        description="Endpoint base URL. Required for provider 'local'; "
        "preset for openai/gemini but overridable. A list gives the Cortex "
        "several endpoints for load-balancing and failover (one query is one "
        "request, so num_parallel does not apply here).",
    )
    model: str | list[str] = "gemini-3.1-flash-lite"
    api_key_env: str = Field(
        default="GEMINI_API_KEY",
        description="Environment variable holding the API key. Optional "
        "for provider 'local'.",
    )
    thinking: Literal["default", "off", "minimal", "low", "medium", "high"] = Field(
        default="default",
        description="Reasoning effort, mapped per provider exactly like "
        "chronicle.caption.thinking.",
    )
    system_prompt: str | None = Field(
        default=None,
        description="Overrides the built-in compilation prompt. Any "
        "override must still demand the QuerySpec JSON schema the parser "
        "validates (visual_phrases, entity_terms, ocr_literals, "
        "asr_phrases, temporal_hints, negations, paraphrases, confidence).",
    )
    qa_system_prompt: str | None = Field(
        default=None,
        description="Overrides the built-in VQA question-compilation prompt "
        "(compile_qa). Any override must still demand the QaPlan JSON schema "
        "the parser validates (archetype, locator{...}, question_core, "
        "expected_answer_type). Only read when vqa.enabled.",
    )
    paraphrases: int = Field(
        default=2,
        ge=0,
        description="Paraphrases of the query requested from the LLM; they "
        "join the channel race at dispatch.paraphrase_weight.",
    )
    max_list_items: int = Field(
        default=8,
        gt=0,
        description="Per-field cap on spec list lengths; extremely long "
        "KIS-C accumulations are truncated rather than dispatched whole.",
    )
    timeout_s: float = Field(default=30.0, gt=0.0)
    max_output_tokens: int = Field(default=800, gt=0)
    dispatch: DispatchConfig = DispatchConfig()

    @model_validator(mode="after")
    def _endpoint_constraints(self) -> CortexConfig:
        if self.provider == "local" and not self.base_url:
            raise ValueError("cortex provider 'local' requires base_url")
        resolve_endpoints(self.base_url, self.model)
        if self.top_k is not None and self.provider != "local":
            raise ValueError(
                "top_k is only supported by self-deployed servers "
                "(provider 'local')"
            )
        return self


class QppConfig(StrictModel):
    """The QPP advisor (Phase 7): a calibrated P(hit in top-k) hint.

    Advisory only — it renders a "likely to miss, consider escalating" hint
    and never fires an escalation itself (the 2026-07 operator-trigger
    revision in note 09). The model artifact is trained on the labelled
    fixture by scripts/train_qpp.py; without one the hint is simply absent.
    """

    trained_artifact: Path | None = Field(
        default=None,
        description="Path to the trained advisor artifact — the small JSON "
        "file scripts/train_qpp.py produces from labelled fixture outcomes "
        "(NOT a neural model checkpoint). None (default) disables the "
        "advisor entirely: no hint is rendered.",
    )
    hint_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Predicted P(target in the operator's top-k) below "
        "which the console shows the 'consider escalating' hint.",
    )
    margin_top_n: int = Field(
        default=5,
        gt=1,
        description="The deep margin feature compares the top-1 fused score "
        "against the score at this rank.",
    )
    agreement_top_n: int = Field(
        default=50,
        gt=0,
        description="Channel-agreement rank correlation is computed over "
        "each channel's top-N shots.",
    )
    compactness_top_n: int = Field(
        default=10,
        gt=0,
        description="Temporal compactness looks at how many of the top-N "
        "fused candidates share one video.",
    )


class RerankEscalationConfig(StrictModel):
    """Late-interaction re-rank (BGE-M3 ColBERT MaxSim over evidence text)."""

    enabled: bool = Field(
        default=True,
        description="The re-rank tool costs no extra model (BGE-M3 is "
        "already resident for the sparse channel), so it defaults on.",
    )
    pool_size: int = Field(
        default=200,
        gt=0,
        description="Only the top-N fused candidates are re-scored; the "
        "rest keep their order below them.",
    )
    timeout_s: float = Field(default=20.0, gt=0.0)


class ImagineEscalationConfig(StrictModel):
    """Imagine->Match: few-step diffusion renders of the spec, then
    image->image search against the keyframe index (note 08, step 6a).

    Defaults follow the SDXL-Turbo model card: a 1-step distilled model
    trained without classifier-free guidance (guidance_scale must stay 0.0)
    at a native 512x512.
    """

    enabled: bool = Field(
        default=False,
        description="Off by default: the diffusion model is a multi-GB "
        "download and wants a GPU. Enable per profile on the server.",
    )
    model_id: str = Field(
        default="stabilityai/sdxl-turbo",
        description="Diffusers text-to-image model id.",
    )
    device: str = Field(
        default="auto",
        description="Torch device for the diffusion pipeline; pin a card "
        "(cuda:1) on multi-GPU servers so the tool stays warm.",
    )
    num_images: int = Field(
        default=3, gt=0, description="Diverse renders per escalation."
    )
    num_inference_steps: int = Field(
        default=1,
        gt=0,
        description="SDXL-Turbo is distilled for single-step generation "
        "(model card); more steps only pay off on non-turbo models.",
    )
    guidance_scale: float = Field(
        default=0.0,
        ge=0.0,
        description="SDXL-Turbo was trained without classifier-free "
        "guidance and the model card requires 0.0; raise only for models "
        "that document CFG support.",
    )
    width: int = Field(default=512, gt=0)
    height: int = Field(default=512, gt=0)
    min_clip_score: float = Field(
        default=0.2,
        description="Renders whose image-text cosine against the prompt "
        "falls below this are junk-gated out (note 09 Phase 7 edge case). "
        "If every render fails the gate the tool reports it instead of "
        "silently returning the unescalated list.",
    )
    fusion_weight: float = Field(
        default=1.0,
        gt=0.0,
        description="RRF weight of the render-based ranking when fused "
        "with the current candidate pool (pool weight is 1.0).",
    )
    seed: int | None = Field(
        default=None,
        description="Fixed diffusion seed for reproducible renders; None "
        "draws a fresh seed per escalation (more diverse retries).",
    )
    timeout_s: float = Field(default=60.0, gt=0.0)


class VlmVerifyConfig(EndpointRequestConfig):
    """VLM verify: listwise re-rank of the top bundles (note 08, step 6c).

    Provider-agnostic like the captioner and the Cortex; the default is the
    Gemini compatibility endpoint.
    """

    enabled: bool = Field(
        default=False,
        description="Off by default: needs an API key or a local VLM "
        "endpoint. Enable per profile.",
    )
    provider: Literal["openai", "gemini", "local"] = "gemini"
    base_url: str | list[str] | None = Field(
        default=None,
        description="Endpoint base URL. Required for provider 'local'; "
        "preset for openai/gemini but overridable. A list gives verify "
        "several endpoints for load-balancing and failover (one re-rank is "
        "one request, so num_parallel does not apply here).",
    )
    model: str | list[str] = "gemini-3.1-flash-lite"
    api_key_env: str = Field(
        default="GEMINI_API_KEY",
        description="Environment variable holding the API key. Optional "
        "for provider 'local'.",
    )
    thinking: Literal["default", "off", "minimal", "low", "medium", "high"] = Field(
        default="default",
        description="Reasoning effort, mapped per provider exactly like "
        "chronicle.caption.thinking.",
    )
    system_prompt: str | None = Field(
        default=None,
        description="Overrides the built-in verification prompt. Any "
        "override must still demand the {\"order\": [...]} JSON the parser "
        "validates.",
    )
    top_n: int = Field(
        default=20,
        gt=0,
        description="How many top candidates the VLM re-orders.",
    )
    frames_per_candidate: int = Field(
        default=1,
        ge=0,
        description="Keyframes attached per candidate; 0 sends evidence "
        "text only (cheaper, weaker).",
    )
    max_evidence_chars: int = Field(
        default=500,
        gt=0,
        description="Per-candidate cap on the evidence text sent to the "
        "VLM, keeping listwise requests bounded.",
    )
    timeout_s: float = Field(default=60.0, gt=0.0)
    max_output_tokens: int = Field(default=800, gt=0)

    @model_validator(mode="after")
    def _endpoint_constraints(self) -> VlmVerifyConfig:
        if self.enabled and self.provider == "local" and not self.base_url:
            raise ValueError("vlm_verify provider 'local' requires base_url")
        if self.top_k is not None and self.provider != "local":
            raise ValueError(
                "top_k is only supported by self-deployed servers "
                "(provider 'local')"
            )
        return self


class EscalationsConfig(StrictModel):
    """The Phase 7 escalation toolbox: operator-triggered tools.

    Each escalation is a console button + hotkey; a disabled escalation has
    no button and no endpoint. Model placement is per-tool ``device`` (pin
    each model to its own card on multi-GPU servers); the LRU cache below is
    the single-GPU economy mode.
    """

    max_resident_models: int = Field(
        default=1,
        gt=0,
        description="Heavy escalation models kept loaded at once; beyond "
        "it the least recently used is evicted (single-GPU economy mode; "
        "irrelevant when each model has its own GPU).",
    )
    rerank: RerankEscalationConfig = RerankEscalationConfig()
    imagine: ImagineEscalationConfig = ImagineEscalationConfig()
    vlm_verify: VlmVerifyConfig = VlmVerifyConfig()


class FeedbackConfig(StrictModel):
    """Rocchio relevance-feedback weights for the dense visual channel.

    Defaults are the classic Rocchio values (alpha=1.0, beta=0.75,
    gamma=0.25) from the SMART literature; the fixture ablation tunes them.
    """

    alpha: float = Field(default=1.0, ge=0.0, description="Original query.")
    beta: float = Field(default=0.75, ge=0.0, description="Positive centroid.")
    gamma: float = Field(default=0.25, ge=0.0, description="Negative centroid.")


class ServiceConfig(StrictModel):
    """The online FastAPI service (Phase 6) and operator UI (Phase 8)."""

    host: str = "127.0.0.1"
    port: int = Field(default=8000, gt=0, lt=65536)
    default_top_k: int = Field(
        default=100,
        gt=0,
        description="Results returned per search unless the request says "
        "otherwise.",
    )
    monotone_narrowing: bool = Field(
        default=True,
        description="KIS-C reveals restrict new results to shots already in "
        "the previous candidate set, so adding a constraint never grows the "
        "set; if the intersection would be empty the unfiltered results are "
        "kept (the target may have sat outside the earlier top-k).",
    )
    max_sessions: int = Field(
        default=64,
        gt=0,
        description="In-memory session cap; the oldest idle session is "
        "evicted beyond it.",
    )
    max_constraints: int = Field(
        default=12,
        gt=1,
        description="KIS-C constraint-stack cap; beyond it the two oldest "
        "constraints merge into one so no revealed detail is lost.",
    )
    submission_format: Literal["json", "csv"] = Field(
        default="json",
        description="Serialization of (video_id, timestamp) submissions. "
        "The official 2026 format lands in aic/service/submission.py only.",
    )
    feedback: FeedbackConfig = FeedbackConfig()
    streaming: bool = Field(
        default=False,
        description="Register the additive NDJSON streaming routes "
        "(/api/search/stream, /api/qa/stream) that emit the immediately-usable "
        "payload first (KIS fused ranking; VQA Track A cards) and then stream "
        "each slower stage as it finishes (KIS auto-rerank; VQA "
        "reverse/count/Track B). Off keeps only the one-shot JSON routes, whose "
        "behaviour is byte-identical either way. The operator console "
        "auto-detects the /stream routes and falls back to the one-shot route "
        "on 404, so enabling this is purely additive.",
    )
    log_file: Path | None = Field(
        default=None,
        description="Rotating service log file (Phase 9 hardening: disk "
        "never fills from logs). None logs to stderr only.",
    )
    log_max_bytes: int = Field(
        default=10_000_000,
        gt=0,
        description="Rotation size of the service log file.",
    )
    log_backups: int = Field(
        default=5,
        ge=0,
        description="Rotated log files kept before the oldest is deleted.",
    )


class QueryClipConfig(StrictModel):
    """How KIS-V example clips are sampled into a query vector."""

    frames: int = Field(
        default=4,
        gt=0,
        description="Frames sampled evenly from the query clip and averaged "
        "into one query vector.",
    )
    decode_width: int = Field(
        default=720,
        gt=0,
        description="Decode width for query clips; match the keyframe "
        "max_side so the image tower sees comparable inputs.",
    )
    decode_height: int = Field(default=405, gt=0)


class TemporalFusionConfig(StrictModel):
    """Window-level scoring over the per-shot RRF race (note 15).

    Queries describe a scene spanning several shots; per-shot fusion lets
    partial evidence in adjacent shots compete instead of accumulate. With a
    window, candidates of one video within ``window_ms`` of each other form
    a window whose score sums each sub-query's best contribution inside it,
    so fragments of one compound query reinforce the same region.
    """

    window_ms: int = Field(
        default=0,
        ge=0,
        description="Maximum span (midpoint to midpoint, ms) of a candidate "
        "window. 0 (default) disables window fusion entirely — ranking is "
        "the plain per-shot RRF, bit-identical to the pre-temporal engine.",
    )
    agreement_bonus: float = Field(
        default=0.0,
        ge=0.0,
        description="Multiplicative reward for cross-channel agreement: a "
        "window hit by N distinct channels scores * (1 + bonus * (N - 1)). "
        "Independent channels (visual, caption, OCR/speech) hitting the "
        "same few seconds is triangulated evidence, worth more than the "
        "same mass from one channel. Multiplicative, so it is scale-free "
        "against the RRF weights. 0 disables the reward.",
    )
    video_prior_weight: float = Field(
        default=0.0,
        ge=0.0,
        description="Weight of the overlay-text video prior: each window "
        "gains this * 1/(rrf_k + rank of its video in the overlay index "
        "ranking). Overlay lines (watermarks, tickers, program names — see "
        "chronicle.overlay_min_recurrence) identify the right VIDEO even "
        "though they are noise at shot granularity. 0 disables the prior; "
        "it also requires window fusion (window_ms > 0) and a built overlay "
        "index to have any effect.",
    )
    order_bonus: float = Field(
        default=0.0,
        ge=0.0,
        description="Multiplicative reward for a window whose ordered "
        "sub-queries (the sentence chunks of retrieval.visual_chunking) "
        "match in the query's sentence order: earlier sentences hitting "
        "earlier shots scores * (1 + bonus). Sentence order in a KIS query "
        "usually mirrors event order (vitrivr's temporal scoring, VBS). "
        "0 disables it; it only fires when window fusion is on AND a query "
        "was actually chunked, so it is inert without visual_chunking.",
    )


class AutoRerankConfig(StrictModel):
    """Automatic VLM listwise rerank of the fused top (note 15).

    Reuses the escalations.vlm_verify endpoint, prompt, and top_n settings —
    this knob only makes that same listwise rerank run automatically after
    fusion instead of waiting for the operator button. Any failure or
    timeout degrades to the unchanged ranking.
    """

    enabled: bool = Field(
        default=False,
        description="Run the vlm_verify listwise rerank on every search "
        "result before it is returned. Directly attacks the top-1 vs top-k "
        "gap, at the cost of one VLM call of added latency per search — "
        "budget it against the round timer before enabling in a live round.",
    )
    timeout_s: float = Field(
        default=20.0,
        gt=0.0,
        description="Hard timeout for the automatic rerank call; on expiry "
        "the unchanged fused ranking is returned (degrade-to-unchanged).",
    )


class VisualChunkingConfig(StrictModel):
    """Sentence chunking of long queries for the dense visual channel.

    SigLIP-family text towers were trained on short alt-text (64-position
    table) and attend mostly to the FIRST sentence of a multi-sentence
    input (arXiv 2602.22419), so a 3-sentence KIS query fed whole loses
    sentences 2-3. With chunking, each sentence becomes its own
    dense-visual sub-query and rejoins the existing RRF/temporal-window
    fusion — each sentence finds its own moment, and the window covering
    them all wins. Applies only to the dense visual channel; the
    multilingual text channels read the full query natively. Off is the
    previous behaviour, bit for bit.
    """

    enabled: bool = Field(
        default=False,
        description="Split multi-sentence dense-visual queries into one "
        "sub-query per sentence. Affects the Cortex fallback spec, long "
        "paraphrases, and the no-cortex baseline; single-sentence queries "
        "are unchanged.",
    )
    max_chunks: int = Field(
        default=6,
        ge=1,
        description="Upper bound on chunks per query; extra sentences merge "
        "into the last chunk. Bounds encoder and index cost per query.",
    )
    chunk_weight: float = Field(
        default=1.0,
        gt=0.0,
        description="RRF weight multiplier applied to each chunk sub-query "
        "(relative to the weight the unsplit query would have had). Under "
        "plain per-shot RRF, N chunks contribute up to N times the unsplit "
        "mass — lower this toward 1/N to keep the channel balance; under "
        "window fusion the per-label-best sum is the intended coverage "
        "semantics and 1.0 is the natural value.",
    )


class RetrievalConfig(StrictModel):
    fusion: FusionConfig
    query_clip: QueryClipConfig = QueryClipConfig()
    temporal: TemporalFusionConfig = TemporalFusionConfig()
    auto_rerank: AutoRerankConfig = AutoRerankConfig()
    visual_chunking: VisualChunkingConfig = VisualChunkingConfig()
    keyframe_oversample: int = Field(
        default=4,
        ge=1,
        description="The dense visual channel searches "
        "top_k_per_channel * this many keyframes before collapsing them to "
        "their best shot, then keeps top_k_per_channel shots. Without the "
        "oversample the channel returned top_k keyframes of ~3 per shot, so "
        "it entered rank fusion with a third as many candidates as the text "
        "channels. Raise it if keyframes-per-shot grows (denser sampling).",
    )


class FixtureGenConfig(EndpointRequestConfig):
    """Synthetic fixture generation: a VLM writes AIC-style Vietnamese KIS
    queries for randomly sampled truth windows of the ingested corpus
    (note 15). Labels are exact by construction, so the generated fixture
    scales ablation measurement and QPP training beyond the hand-labelled
    cases. The model sees ONLY the window's keyframes — never the corpus
    captions — so the queries do not leak the caption channel's vocabulary.
    """

    provider: Literal["openai", "gemini", "local"]
    base_url: str | list[str] | None = Field(
        default=None,
        description="Endpoint base URL. Required for provider 'local'; "
        "preset for openai/gemini but overridable. A list fans generation "
        "out across several endpoints.",
    )
    model: str | list[str]
    api_key_env: str = Field(
        description="Environment variable holding the API key. Optional "
        "for provider 'local'.",
    )
    thinking: Literal["default", "off", "minimal", "low", "medium", "high"] = Field(
        default="default",
        description="Reasoning effort for the generation call; same "
        "provider mapping as chronicle.caption.thinking.",
    )
    system_prompt: str | None = Field(
        default=None,
        description="Overrides the built-in generation system prompt. Any "
        'override must still demand a JSON object {"text": ...} — the '
        "parser validates that shape and rejects replies without it.",
    )
    count: int = Field(
        default=100,
        gt=0,
        description="Number of synthetic cases to generate.",
    )
    window_ms: int = Field(
        default=10000,
        gt=0,
        description="Span of each sampled truth window. Match the "
        "hand-labelled fixture's window length so the two sets measure the "
        "same task.",
    )
    frames_per_case: int = Field(
        default=4,
        gt=0,
        description="Keyframes of the window sent to the model per case.",
    )
    seed: int = Field(
        default=0,
        description="RNG seed for window sampling; a fixed seed regenerates "
        "the identical case set.",
    )
    timeout_s: float = Field(default=120.0, gt=0.0)
    max_output_tokens: int = Field(default=500, gt=0)
    qa_fraction: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of sampled windows generated as VQA cases "
        "(kind 'qa') instead of KIS queries; 0 keeps the KIS-only behaviour. "
        "The sampled-window flow is shared: a qa window asks the VLM for a "
        "question + accepted answers + archetype grounded ONLY in the frames "
        "it saw, and to self-assert the answer is visible from those frames "
        "or the case is discarded (counted as a failure).",
    )
    qa_system_prompt: str | None = Field(
        default=None,
        description="Overrides the built-in QA-generation prompt. Any "
        "override must still demand the JSON object {question_vi, "
        "accepted_answers, archetype, answer_visible} the parser validates.",
    )
    qa_archetypes: list[str] = Field(
        default_factory=lambda: list(_VQA_ARCHETYPES),
        description="Archetypes the generator stratifies qa windows over so "
        "the synthetic set is not all read_text (risk 3.5); each qa window is "
        "assigned one target archetype round-robin (the model may relabel).",
    )

    @model_validator(mode="after")
    def _local_needs_base_url(self) -> FixtureGenConfig:
        if self.provider == "local" and not self.base_url:
            raise ValueError("fixture_gen provider 'local' requires base_url")
        resolve_endpoints(self.base_url, self.model)
        if self.top_k is not None and self.provider != "local":
            raise ValueError(
                "top_k is only supported by self-deployed servers (provider "
                "'local'); api.openai.com rejects unknown arguments and the "
                "Gemini compatibility layer does not document it"
            )
        return self


class VerifyConfig(StrictModel):
    """Integrity check + repair of persisted artifacts before a resume.

    An interrupted offline run (a killed Colab session, a Ctrl-C) can leave a
    truncated keyframe image or embedding shard on disk that the manifest still
    lists as done, or duplicate/partial manifest rows. When ``on_resume`` is
    true, ``aic.verify.verify_and_repair`` runs before the stage: it opens and
    decodes every manifest-referenced artifact, deletes any that are missing or
    corrupt together with their manifest rows (cascading the drop to downstream
    stages whose input vanished), so the normal resume then redoes exactly the
    broken items plus whatever was still pending. It is off by default because
    the decode check reads every artifact once — a full-corpus pass — which is
    only worth paying after a known-bad interruption.
    """

    on_resume: bool = Field(
        default=False,
        description="Run artifact integrity check + repair before resuming an "
        "offline stage. Off by default; the check decodes every referenced "
        "file, so enable it after an interrupted run, not on every resume.",
    )


_VQA_ARCHETYPES = ("count", "read_text", "entity", "attribute", "cumulative", "other")

# The evidence-source labels the reader's prompt is built from. A format
# contract with aic.vqa.evidence/reader (they index these exact keys), so an
# override must supply every one of them — validated at load, not at query time.
_EVIDENCE_LABEL_KEYS = ("ocr", "asr", "caption", "question", "proposed_answer")


class QaReaderConfig(SamplingFields, MultiEndpointFields):
    """Track B grounded reader: a VLM reads an evidence pack per candidate.

    Same OpenAI-compatible endpoint shape as escalations.vlm_verify (mirror it
    by analogy). Disabled by default; with the reader off the QA path is Track
    A only (phase-1 behaviour), so enabling phase 2 is opt-in.
    """

    enabled: bool = Field(
        default=False,
        description="False keeps the QA path Track A only (no per-candidate "
        "VLM read); True runs Track B grounded reads.",
    )
    provider: Literal["openai", "gemini", "local"] = "local"
    base_url: str | list[str] | None = Field(
        default=None,
        description="Endpoint base URL. Required for provider 'local'; a list "
        "gives several endpoints for the per-candidate fan-out.",
    )
    model: str | list[str] = Field(
        default="",
        description="Reader model name(s); required when enabled.",
    )
    api_key_env: str | None = Field(
        default=None,
        description="Environment variable holding the API key; optional for "
        "provider 'local'.",
    )
    thinking: Literal["default", "off", "minimal", "low", "medium", "high"] = "default"
    system_prompt: str | None = Field(
        default=None,
        description="Overrides the built-in reader prompt. Any override must "
        "still demand the QaRead JSON object (answerable, answer, "
        "evidence_timestamp_ms, confidence).",
    )
    count_system_prompt: str | None = Field(
        default=None,
        description="Overrides the built-in system prompt of the count read "
        "(the VLM's independent estimate + crop adjudication). Any override "
        "must still demand a bare integer reply, which the parser extracts.",
    )
    count_user_prompt: str | None = Field(
        default=None,
        description="Overrides the built-in user prompt of the count read. "
        "'{question_label}' (the configured vqa.evidence_labels question "
        "label), '{question}', and '{concept}' are replaced verbatim "
        "(str.replace, so literal braces elsewhere are safe); an override "
        "should keep question and concept.",
    )
    disconfirm_system_prompt: str | None = Field(
        default=None,
        description="Overrides the built-in disconfirmation prompt (the one "
        "negatively-framed check against the runner-up moment). Any override "
        "must still demand the {contradicts, why} JSON object.",
    )
    frames_per_candidate: int = Field(
        default=4,
        gt=0,
        description="Keyframes packed into each candidate's read. Each image "
        "costs ~300-1.2k projector tokens, so a small number keeps the "
        "request under a local server's context (X1); never a listwise call.",
    )
    image_jpeg_quality: int = Field(
        default=85,
        ge=1,
        le=100,
        description="JPEG quality for frames encoded into the read request.",
    )
    max_evidence_chars: int = Field(
        default=2000,
        gt=0,
        description="Cap on the text block (OCR/ASR/caption) per candidate.",
    )
    timeout_s: float = Field(
        default=30.0,
        gt=0.0,
        description="Per-request httpx timeout. MUST be <= vqa.timeout_s so an "
        "abandoned slow read does not occupy the server past the wall (X2/X9).",
    )
    max_output_tokens: int = Field(default=256, gt=0)

    @model_validator(mode="after")
    def _endpoint_constraints(self) -> QaReaderConfig:
        if not self.enabled:
            return self
        if self.provider == "local" and not self.base_url:
            raise ValueError("qa reader provider 'local' requires base_url")
        if not self.model:
            raise ValueError("qa reader requires model when enabled")
        resolve_endpoints(self.base_url, self.model)
        return self


class VqaNormalizeConfig(StrictModel):
    """Language-general answer normalisation (note 18 §3.5).

    Every language- or corpus-specific table is config, so the same code
    serves any language by swapping these lexicons; the defaults are the
    Vietnamese set the AIC corpus needs. The compound-number grammar
    (``ten_word``/``tens_word``) is the one structural rule that is
    Vietnamese-shaped — leave both empty for a language without it and only
    the single-word lexicon (plus digits, always) applies.
    """

    number_words: dict[str, int] = Field(
        default_factory=lambda: {
            "không": 0,
            "một": 1,
            "mốt": 1,
            "hai": 2,
            "ba": 3,
            "bốn": 4,
            "tư": 4,
            "năm": 5,
            "lăm": 5,
            "sáu": 6,
            "bảy": 7,
            "bẩy": 7,
            "tám": 8,
            "chín": 9,
        },
        description="Maps a spoken cardinal word to its integer, so a "
        "number answer read as a word ('ba') folds to the same key as its "
        "digit ('3'). Language-specific, hence config; the defaults are the "
        "Vietnamese cardinals 0-9 (with the positional variants mốt/tư/lăm). "
        "Digits always parse regardless of this table.",
    )
    ten_word: str = Field(
        default="mười",
        description="The word for exactly ten (Vietnamese 'mười'), used by "
        "the compound rule 'ten_word <unit>' -> 10+unit. Empty disables it.",
    )
    tens_word: str = Field(
        default="mươi",
        description="The multiplier word in '<n> tens_word [unit]' "
        "(Vietnamese 'mươi': hai mươi ba -> 23). Empty disables the compound "
        "rule, leaving single-word + digit parsing only (for languages "
        "without this construction).",
    )
    color_lexicon: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "đỏ": ["đỏ", "red"],
            "cam": ["cam", "orange"],
            "vàng": ["vàng", "yellow"],
            "lục": ["lục", "xanh lá", "xanh lá cây", "green"],
            "lam": ["lam", "xanh dương", "xanh lam", "xanh nước biển", "blue"],
            "tím": ["tím", "purple", "violet"],
            "trắng": ["trắng", "white"],
            "đen": ["đen", "black"],
            "nâu": ["nâu", "brown"],
            "hồng": ["hồng", "pink"],
            "xám": ["xám", "grey", "gray"],
        },
        description="Maps each canonical colour to its Vietnamese/English "
        "surface variants, so 'xanh dương'/'xanh lam' fold to one answer. "
        "Corpus-tunable; the defaults are the common Vietnamese news palette.",
    )
    unit_strip: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "number": ["người", "cái", "chiếc", "con", "lần", "%", "phần trăm"],
        },
        description="Trailing units stripped from an answer BEFORE comparison, "
        "per answer_type. 'other' and 'duration' strip nothing (a duration IS "
        "its unit). Units are corpus-dependent, hence config not code.",
    )


class VqaCountConfig(StrictModel):
    """Online count program (note 18 phase 4): detector-first counting."""

    min_mask_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Query-time instance score floor: detections below this "
        "are dropped BEFORE the count is reconciled and before the crop box "
        "is drawn. Applied on top of ledger.counter.confidence (which the "
        "backend already enforced), so it can only tighten, never loosen.",
    )
    adjudicate_margin: int = Field(
        default=1,
        ge=0,
        description="When the detector count and the VLM read differ by AT "
        "MOST this many, the VLM adjudicates with the masked/boxed crops; "
        "beyond it the detector wins (note 17 §4b). 0 = always trust the "
        "detector.",
    )
    crop_pad_frac: float = Field(
        default=0.15,
        ge=0.0,
        description="Fraction of the union box's size added on every side "
        "before the adjudication crop, so instances at the box edge keep "
        "context. The reader clamps the padded box to each frame.",
    )


class VqaReverseConfig(StrictModel):
    """Answer-first reverse retrieval (note 18 §3.3, phase 5), off by default.

    For a weak-locator question, hypothesise plausible answers and search their
    surface forms in the literal/factoid indexes; a hypothesis wins only when it
    both retrieves a concentrated moment AND that moment matches the locator
    (the double gate that stops the self-fulfilling loop, risk 5.2).
    """

    enabled: bool = Field(
        default=False,
        description="Master switch (vqa.reverse_retrieval). Off keeps the "
        "phase-2 behaviour byte-identical.",
    )
    trigger_ratio: float = Field(
        default=1.3,
        ge=1.0,
        description="Peakiness = top fused score / median of the top-M. The "
        "locator is 'weak' (reverse fires) when peakiness is BELOW this (a flat "
        "distribution, no clear winner), or when Track A produced no "
        "answer-bearing card. Also the concentration gate for a hypothesis's "
        "own search. Calibrate on the phase-3 fixture distributions; 1.0 never "
        "fires on flatness alone.",
    )
    max_hypotheses: int = Field(
        default=5,
        gt=0,
        description="Cap on answer hypotheses per question (bounds cost; the "
        "whole pass is max_hypotheses sparse queries, no VLM reads).",
    )
    min_locator_score: float = Field(
        default=0.0,
        description="A hypothesis's retrieved moment must ALSO score at least "
        "this against the locator clause to win (the second gate). Corpus- and "
        "score-scale dependent; calibrate on the fixture.",
    )
    reverse_weight: float = Field(
        default=0.5,
        ge=0.0,
        description="Weight a reverse card enters the answer vote with, so a "
        "sharp sparse match does not dominate honest reads (risk 5.3). "
        "Provenance labels it 'found by searching the answer'.",
    )
    gate_top_m: int = Field(
        default=8,
        gt=0,
        description="How many of a hypothesis's top hits the concentration "
        "gate (gate A) computes peakiness over. Over a full RRF tail the "
        "max/median ratio is always well above any sane trigger_ratio "
        "(hundreds of 1/(k+rank) scores make the median tiny), so the gate "
        "would pass everything; a short slice makes it discriminate flat "
        "from peaked. Size it like vqa.top_m.",
    )


class EvHintConfig(StrictModel):
    """The EV submit hint (note 18 §3.6, phase 5): a transparent rule table.

    Advisory only (the QPP contract): it never fires anything, only suggests
    submit_now / wait_for_reads / escalate_zoom from the current answer state.
    A trained model may replace the rules once phase-3 timing data justifies it.
    """

    enabled: bool = Field(
        default=False,
        description="Master switch. Off omits the hint from /api/qa.",
    )
    min_agreeing_for_submit: int = Field(
        default=2,
        ge=1,
        description="Submit-now when the winning answer group has at least this "
        "many agreeing moments (cross-moment agreement).",
    )
    max_early_submits: int = Field(
        default=3,
        ge=0,
        description="Budget on submit_now hints per task (DRES penalties are "
        "real even if resubmission is allowed, risk 5.5); beyond it the hint "
        "advises waiting/escalating. The service scopes a 'task' as one "
        "question: re-asking the same question spends the budget, a new "
        "question resets it.",
    )


class VqaConfig(StrictModel):
    """The online VQA / Q&A path (note 18 §3), off by default.

    When disabled the service exposes only the KIS endpoints; enabling it adds
    the Track A answer-lookup route over the ledger factoid index. Track B
    (per-candidate reads + voting) is off until the reader endpoint is enabled.
    Reverse retrieval / EV hint (phase 5) are not stubbed here.
    """

    enabled: bool = Field(
        default=False,
        description="Master switch. False leaves the KIS service and its "
        "outputs bit-identical (no /api/qa route).",
    )
    archetypes: list[str] = Field(
        default_factory=lambda: list(_VQA_ARCHETYPES),
        min_length=1,
        description="Question archetypes the compiler may assign; must "
        f"include 'other' (the never-fail fallback). Known: {list(_VQA_ARCHETYPES)}.",
    )
    top_m: int = Field(
        default=8,
        gt=0,
        description="Candidate moments Track A joins with their ledger rows, "
        "and Track B reads per question.",
    )
    reader: QaReaderConfig = QaReaderConfig()
    zoom_window_ms: int = Field(
        default=10000,
        gt=0,
        description="Half-window (ms) around a candidate's timestamp from "
        "which Track B gathers keyframes across shots for the evidence pack.",
    )
    vote_min_agree: int = Field(
        default=2,
        gt=0,
        description="Minimum candidate reads agreeing on a normalised answer "
        "for the group to be flagged 'confident' (LongVidSearch "
        "cross-candidate vote); the console marks confident groups so the "
        "operator can submit early without waiting for the EV hint.",
    )
    disconfirm: bool = Field(
        default=True,
        description="Run one disconfirmation call against the runner-up "
        "moment before submitting the winner (CoVe/CASHEW-shaped). Off skips "
        "it; a failure always degrades to the un-checked vote.",
    )
    track_a_bonus: float = Field(
        default=0.5,
        ge=0.0,
        description="Score added to an answer group when the same normalised "
        "answer also appears on a Track A card (cross-track agreement — the "
        "strongest signal). 0 disables the bonus.",
    )
    timeout_s: float = Field(
        default=30.0,
        gt=0.0,
        description="Wall-clock budget for the Track B read stage. The reader "
        "endpoint's timeout_s must be <= this so abandoned reads never occupy "
        "the server past the wall (X2/X9).",
    )
    factoid_weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight the QA locator's text clauses (entity_terms, "
        "ocr_literals, asr_phrases) and reverse-retrieval hypotheses enter "
        "the Answer Ledger factoid channel with. Applies ONLY to the QA "
        "dispatch — KIS searches never probe the factoid index — and is "
        "inert when the ledger/factoid index was not built. 0 disables the "
        "factoid probe entirely.",
    )
    normalize: VqaNormalizeConfig = VqaNormalizeConfig()
    count: VqaCountConfig = VqaCountConfig()
    reverse: VqaReverseConfig = VqaReverseConfig()
    ev_hint: EvHintConfig = EvHintConfig()
    evidence_labels: dict[str, str] = Field(
        default_factory=lambda: {
            "ocr": "On-screen text",
            "asr": "Speech",
            "caption": "Caption",
            "question": "Question",
            "proposed_answer": "Proposed answer",
        },
        description="Labels prefixed to each evidence source in the reader's "
        "prompt (the values below the label stay in the footage's original "
        "language). English by default so the prompt is language-neutral; "
        "override to localise. Keys: ocr, asr, caption, question, "
        "proposed_answer.",
    )

    @field_validator("archetypes")
    @classmethod
    def _archetypes_valid(cls, value: list[str]) -> list[str]:
        unknown = [a for a in value if a not in _VQA_ARCHETYPES]
        if unknown:
            raise ValueError(
                f"unknown archetypes {unknown}; known: {list(_VQA_ARCHETYPES)}"
            )
        if "other" not in value:
            raise ValueError("archetypes must include 'other' (the fallback)")
        return value

    @field_validator("evidence_labels")
    @classmethod
    def _labels_complete(cls, value: dict[str, str]) -> dict[str, str]:
        # The evidence builder and reader index these keys directly; a partial
        # override must fail at config load, not as a KeyError mid-question.
        missing = [k for k in _EVIDENCE_LABEL_KEYS if k not in value]
        unknown = [k for k in value if k not in _EVIDENCE_LABEL_KEYS]
        if missing or unknown:
            raise ValueError(
                f"evidence_labels must define exactly {list(_EVIDENCE_LABEL_KEYS)}"
                + (f"; missing {missing}" if missing else "")
                + (f"; unknown {unknown}" if unknown else "")
            )
        return value

    @model_validator(mode="after")
    def _reader_timeout_within_wall(self) -> VqaConfig:
        if self.reader.enabled and self.reader.timeout_s > self.timeout_s:
            raise ValueError(
                "vqa.reader.timeout_s must be <= vqa.timeout_s so an abandoned "
                f"read does not outlive the wall ({self.reader.timeout_s} > "
                f"{self.timeout_s})"
            )
        return self


class Config(StrictModel):
    project: ProjectConfig
    paths: PathsConfig
    ingest: IngestConfig
    embed: EmbedConfig
    index: IndexConfig
    chronicle: ChronicleConfig
    textstack: TextStackConfig
    retrieval: RetrievalConfig
    eval: EvalConfig
    cortex: CortexConfig = CortexConfig()
    vqa: VqaConfig = VqaConfig()
    escalations: EscalationsConfig = EscalationsConfig()
    qpp: QppConfig = QppConfig()
    service: ServiceConfig = ServiceConfig()
    verify: VerifyConfig = VerifyConfig()
    fixture_gen: FixtureGenConfig | None = Field(
        default=None,
        description="Synthetic fixture generation settings; only "
        "scripts/generate_fixture.py reads this section. None means the "
        "profile does not support generation (the script fails with a "
        "pointer here).",
    )

    @model_validator(mode="after")
    def _auto_rerank_needs_vlm_verify(self) -> Config:
        needs_verify = self.retrieval.auto_rerank.enabled
        if needs_verify and not self.escalations.vlm_verify.enabled:
            raise ValueError(
                "retrieval.auto_rerank.enabled requires "
                "escalations.vlm_verify.enabled: the automatic rerank reuses "
                "that escalation's endpoint, prompt, and top_n settings"
            )
        return self


def _resolve_env(value: object) -> object:
    """Recursively substitute ``${VAR}`` references in string values."""
    if isinstance(value, str):

        def substitute(match: re.Match[str]) -> str:
            name = match.group(1)
            resolved = os.environ.get(name)
            if resolved is None:
                raise ConfigError(f"environment variable {name!r} is not set")
            return resolved

        return _ENV_PATTERN.sub(substitute, value)
    if isinstance(value, dict):
        return {k: _resolve_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env(v) for v in value]
    return value


def load_config(path: Path | str) -> Config:
    """Load and validate a profile config from a YAML file."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"config file {path} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"config file {path} must contain a mapping at top level")
    resolved = _resolve_env(raw)
    try:
        return Config.model_validate(resolved)
    except ValueError as exc:
        raise ConfigError(f"config file {path} failed validation: {exc}") from exc
