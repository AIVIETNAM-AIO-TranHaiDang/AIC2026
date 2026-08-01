"""Shot captioning through any OpenAI-compatible chat-completions endpoint.

One captioner covers all three deployment modes:

- ``openai`` — api.openai.com (key required),
- ``gemini`` — Google's OpenAI-compatibility endpoint (key required),
- ``local`` — a self-deployed OpenAI-compatible server (vLLM, Ollama, TGI,
  ...) at a user-supplied ``base_url``; the API key is optional.

The provider mapping, thinking control, sampling fields, and reasoning-model
reply handling live in :mod:`aic.oaicompat` (shared with VLM OCR and the
Query Cortex); the helpers are re-exported here for convenience.

The prompt constrains output to observable content and JSON; an invalid
response gets one repair retry, then the shot is recorded without a caption
(degraded Chronicle rows are legal by design).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable

import httpx
import numpy as np
from pydantic import ValidationError, field_validator

from aic.config import CaptionConfig, StrictModel
from aic.oaicompat import (
    ChatCompletionsClient,
    EndpointError,
    build_endpoint_pool,
    parse_json_reply,
)
from aic.oaicompat import (
    gemini_major_version as gemini_major_version,
)
from aic.oaicompat import (
    image_to_data_url as image_to_data_url,
)
from aic.oaicompat import (
    sampling_payload as sampling_payload,
)
from aic.oaicompat import (
    strip_reasoning as strip_reasoning,
)
from aic.oaicompat import (
    thinking_payload as thinking_payload,
)

logger = logging.getLogger(__name__)

# Default system prompt; overridable via caption.system_prompt in config
# (overrides must keep the JSON field contract the parser validates).
_SYSTEM_PROMPT = (
    "You describe keyframes sampled from one shot of a video. The video can "
    "be any kind of content, so make no assumption about its genre. "
    "Report only what is visibly observable. Do not guess identities, "
    "locations, or events that are not directly readable from the frames. "
    "Respond with a single JSON object with exactly these fields: "
    '"caption_en" (one or two sentences, English), '
    '"actions" (list of short verb phrases), '
    '"scene" (a short lowercase label for the setting, e.g. indoor, outdoor, '
    "studio, interview, crowd, nature, urban, sports, graphic, or other), "
    '"confidence" (0.0-1.0, your certainty in the description).'
)

_CAPTION_VI_PROMPT_FIELD = (
    ' "caption_vi" (the same description in natural Vietnamese, one or two '
    "sentences),"
)

_REPAIR_PROMPT = (
    "Your previous reply was not valid JSON with the required fields. "
    "Reply again with ONLY the JSON object, no prose, no code fences."
)


class CaptionError(EndpointError):
    """Raised when a caption cannot be obtained for a shot."""


class CaptionResult(StrictModel):
    caption_en: str
    caption_vi: str | None = None
    actions: list[str] = []
    scene: str | None = None
    confidence: float | None = None

    @field_validator("actions", mode="before")
    @classmethod
    def _actions_from_string(cls, value):
        # VLMs occasionally emit the list as one comma-joined string
        # ("drone,pan") or a bare word ("sitting"); both carry the intended
        # content, so coerce instead of failing the whole caption.
        if value is None:
            return []
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value


def _system_prompt(cfg: CaptionConfig) -> str:
    if cfg.system_prompt:
        return cfg.system_prompt
    if not cfg.request_caption_vi:
        return _SYSTEM_PROMPT
    # Splice the caption_vi demand into the field list right after
    # caption_en, keeping the rest of the contract identical.
    marker = '"caption_en" (one or two sentences, English),'
    return _SYSTEM_PROMPT.replace(marker, marker + _CAPTION_VI_PROMPT_FIELD)


def _parse_caption_json(text: str, require_caption_vi: bool) -> CaptionResult:
    try:
        payload = parse_json_reply(text)
    except json.JSONDecodeError as exc:
        raise CaptionError(f"captioner returned non-JSON output: {exc}") from exc
    if isinstance(payload, dict) and payload.get("confidence") is not None:
        payload["confidence"] = min(max(float(payload["confidence"]), 0.0), 1.0)
    try:
        result = CaptionResult.model_validate(payload)
    except (ValidationError, TypeError) as exc:
        raise CaptionError(f"caption JSON failed validation: {exc}") from exc
    if require_caption_vi and not (result.caption_vi or "").strip():
        raise CaptionError("caption JSON lacks the requested caption_vi field")
    return result


def _caption_once(
    complete: Callable[..., str],
    images: list[np.ndarray],
    cfg: CaptionConfig,
) -> CaptionResult:
    """Caption one shot through ``complete(messages, ...)``, with one repair
    retry for an invalid JSON reply.

    ``complete`` is either a single endpoint's ``ChatCompletionsClient.complete``
    (batch fan-out, so the repair stays on the same endpoint) or the pool's
    ``complete`` (single-shot, with pool-level failover). A transport failure
    propagates so the pool can fail over; only a content failure is repaired.
    """
    if not images:
        raise CaptionError("caption_shot needs at least one image")
    images = images[: cfg.frames_per_shot]
    content: list[dict] = [
        {"type": "text", "text": f"These {len(images)} frames are from one shot."}
    ]
    content.extend(
        {
            "type": "image_url",
            "image_url": {"url": image_to_data_url(img, cfg.image_jpeg_quality)},
        }
        for img in images
    )
    messages = [
        {"role": "system", "content": _system_prompt(cfg)},
        {"role": "user", "content": content},
    ]
    reply = complete(messages, max_tokens=cfg.max_output_tokens, json_response=True)
    try:
        return _parse_caption_json(reply, cfg.request_caption_vi)
    except CaptionError:
        logger.warning("caption JSON invalid, retrying once with repair prompt")
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": _REPAIR_PROMPT})
        return _parse_caption_json(
            complete(messages, max_tokens=cfg.max_output_tokens, json_response=True),
            cfg.request_caption_vi,
        )


class OpenAICompatCaptioner:
    """Chat-completions captioner with one repair retry.

    Targets one or several OpenAI-compatible endpoints (see
    :func:`aic.config.resolve_endpoints`); a shot is captioned on one endpoint
    at a time, and the pool spreads shots across endpoints and fails over when
    one dies.
    """

    def __init__(
        self, cfg: CaptionConfig, transport: httpx.BaseTransport | None = None
    ) -> None:
        self._cfg = cfg

        def extra_fields_for(model_name: str) -> dict:
            per_model = cfg.model_copy(update={"model": model_name})
            return {**thinking_payload(per_model), **sampling_payload(per_model)}

        self._pool = build_endpoint_pool(
            provider=cfg.provider,
            base_url=cfg.base_url,
            model=cfg.model,
            api_key_env=cfg.api_key_env,
            timeout_s=cfg.timeout_s,
            host_max_retries=cfg.host_max_retries,
            rate_limit_max_retries=cfg.rate_limit_max_retries,
            rate_limit_backoff_s=cfg.rate_limit_backoff_s,
            extra_fields_for=extra_fields_for,
            transport=transport,
            error_cls=CaptionError,
        )

    def caption_shot(self, images: list[np.ndarray]) -> CaptionResult:
        """Caption one shot from up to ``frames_per_shot`` keyframes."""
        return _caption_once(self._pool.complete, images, self._cfg)

    def caption_shots(
        self, shots: list[list[np.ndarray]]
    ) -> list[CaptionResult | Exception]:
        """Caption many shots concurrently across the endpoint pool.

        Returns one entry per input shot in order: a :class:`CaptionResult` on
        success, or the :class:`CaptionError` that ended its attempts (the
        caller records that shot as failed — a degraded Chronicle row is legal).
        """

        def caption(client: ChatCompletionsClient, frames: list[np.ndarray]):
            return _caption_once(client.complete, frames, self._cfg)

        results = self._pool.run_batch(shots, caption, self._cfg.num_parallel)
        return [
            result if isinstance(result, (CaptionResult, Exception))
            else CaptionError("captioner returned no result")
            for result in results
        ]

    def close(self) -> None:
        self._pool.close()
