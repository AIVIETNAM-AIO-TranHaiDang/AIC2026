"""Shared client for OpenAI-compatible chat-completions endpoints.

Used by shot captioning, VLM OCR, and the Query Cortex, so the provider
mapping (openai / gemini / local), thinking control, sampling fields, and
reasoning-model reply handling live in exactly one place.

Thinking/reasoning control differs by provider and, for Gemini, by model
generation (verified against ai.google.dev/gemini-api/docs/openai and the
Gemini 3 developer guide, July 2026):

- Gemini 2.5-era models take ``thinking_config.thinking_budget`` (a token
  count; 0 disables thinking on the Flash variants);
- Gemini 3+ models take ``thinking_config.thinking_level`` (low/medium/high),
  always think, and reject ``thinking_budget`` requests with HTTP 400 —
  the two flags must never be mixed in one request;
- OpenAI takes ``reasoning_effort``;
- local servers take ``chat_template_kwargs`` ``enable_thinking`` (a
  top-level request field in vLLM's chat protocol, verified from source;
  kwargs a chat template does not use are filtered out harmlessly, and
  'default' sends nothing for servers that reject unknown fields).

Local reasoning models may still emit their thinking: vLLM with a reasoning
parser returns it in a separate ``reasoning`` field (``reasoning_content``
before the rename) which is ignored here, and without a parser the
``<think>...</think>`` block lands inline in ``content`` and is stripped.

API keys come from an environment variable named in config; they never touch
config files or git.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import queue
import re
import threading
import time
from collections.abc import Callable
from itertools import count
from typing import TypeVar

import httpx
import numpy as np
from PIL import Image

from aic.config import resolve_endpoints

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

PROVIDER_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
}

# thinking -> thinking_budget tokens for Gemini < 3 (2.5 era). The values
# follow the documented reasoning_effort mapping; 0 disables thinking.
_GEMINI_LEGACY_BUDGETS = {
    "off": 0,
    "minimal": 0,
    "low": 1024,
    "medium": 8192,
    "high": 24576,
}

# thinking -> thinking_level for Gemini >= 3, which cannot disable thinking:
# off/minimal clamp to the lowest supported level.
_GEMINI_LEVELS = {
    "off": "low",
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
}

_GEMINI_VERSION_PATTERN = re.compile(r"gemini-(\d+(?:\.\d+)?)")

# A leading think block emitted inline when the serving stack has no
# reasoning parser configured (the reasoning-model convention).
_THINK_BLOCK_PATTERN = re.compile(r"\A\s*<think>.*?</think>\s*", re.DOTALL)

# HTTP statuses worth retrying on the same endpoint after a wait: the server
# is alive but momentarily unwilling — 429 (rate limited) and 503 (overloaded),
# the two statuses that conventionally carry Retry-After. Other 5xx mean the
# endpoint is unwell: those must fail fast so the pool can fail over or evict
# instead of sleeping on a dead host.
_RETRYABLE_STATUSES = frozenset({429, 503})

# How much of an error response body to surface in the raised message —
# enough for the server's reason line, without dumping a whole HTML page.
_ERROR_BODY_MAX_CHARS = 300


class EndpointError(RuntimeError):
    """Raised when a chat-completions request or reply is unusable."""


class EndpointRequestError(EndpointError):
    """A transport/connectivity failure talking to one endpoint.

    Distinct from a content problem (a valid HTTP reply whose body is empty or
    malformed): only this error counts against an endpoint's health and is
    worth retrying on a different endpoint. A pool catches it to fail over; a
    single-endpoint caller sees it as an ordinary :class:`EndpointError`.

    ``status_code`` carries the HTTP status when one was received. A
    deterministic 4xx (a *rejected* request — see :func:`rejected_request`)
    proves the host is alive and must not count toward eviction: three
    oversized requests would otherwise kill live endpoints for every later
    caller of the pool.
    """

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def rejected_request(exc: EndpointRequestError) -> bool:
    """True when the endpoint answered with a non-retryable 4xx.

    The request was parsed and refused (too big for the context, unsupported
    modality, malformed field) — the host is demonstrably alive, so the
    failure belongs to the request, not the endpoint's health. 429 stays a
    health matter: it is retried in ``_post_with_retry`` and, once retries
    are exhausted, a permanently throttled host should rotate out.
    """
    status = getattr(exc, "status_code", None)
    return (
        status is not None
        and 400 <= status < 500
        and status not in _RETRYABLE_STATUSES
    )


def gemini_major_version(model: str) -> float | None:
    """Extract the Gemini generation from a model name, None if not Gemini."""
    match = _GEMINI_VERSION_PATTERN.search(model.lower())
    return float(match.group(1)) if match else None


def thinking_payload(cfg) -> dict:
    """Extra request fields implementing the configured thinking setting.

    ``cfg`` needs ``provider``, ``model``, and ``thinking`` attributes
    (CaptionConfig and CortexConfig both qualify). Returns exactly one
    mechanism (or none), per the mixing constraint in the module docstring.
    """
    if cfg.thinking == "default":
        return {}
    if cfg.provider == "local":
        enable = cfg.thinking not in ("off", "minimal")
        return {"chat_template_kwargs": {"enable_thinking": enable}}
    if cfg.provider == "openai":
        effort = "minimal" if cfg.thinking == "off" else cfg.thinking
        return {"reasoning_effort": effort}
    version = gemini_major_version(cfg.model)
    if version is None:
        logger.warning(
            "model %r does not look like a Gemini model; "
            "sending no thinking configuration",
            cfg.model,
        )
        return {}
    if version >= 3:
        level = _GEMINI_LEVELS[cfg.thinking]
        if cfg.thinking in ("off", "minimal") and level != cfg.thinking:
            logger.info(
                "Gemini %s cannot disable thinking; clamping %r to level %r",
                version,
                cfg.thinking,
                level,
            )
        return {"google": {"thinking_config": {"thinking_level": level}}}
    return {
        "google": {
            "thinking_config": {"thinking_budget": _GEMINI_LEGACY_BUDGETS[cfg.thinking]}
        }
    }


def sampling_payload(cfg) -> dict:
    """Optional sampling fields from a :class:`~aic.config.SamplingFields`.

    Fields left at None are omitted so the server applies its own defaults.
    """
    payload = {}
    if cfg.temperature is not None:
        payload["temperature"] = cfg.temperature
    if cfg.top_p is not None:
        payload["top_p"] = cfg.top_p
    if cfg.top_k is not None:
        payload["top_k"] = cfg.top_k
    return payload


def strip_reasoning(text: str) -> str:
    """Remove a leading inline ``<think>...</think>`` block, if any."""
    return _THINK_BLOCK_PATTERN.sub("", text)


def parse_json_reply(text: str) -> object:
    """Parse a JSON reply, tolerating a Markdown code fence around it.

    Raises ``json.JSONDecodeError`` when the payload is not JSON; callers
    wrap that in their own error type.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[len("json") :]
    return json.loads(cleaned)


def image_to_data_url(image: np.ndarray, jpeg_quality: int) -> str:
    """Encode an RGB uint8 array as a base64 JPEG data URL."""
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=jpeg_quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


class ChatCompletionsClient:
    """One provider-resolved endpoint plus the hardened completion call."""

    def __init__(
        self,
        provider: str,
        model: str,
        api_key_env: str | None,
        base_url: str | None,
        timeout_s: float,
        extra_fields: dict | None = None,
        transport: httpx.BaseTransport | None = None,
        error_cls: type[EndpointError] = EndpointError,
        rate_limit_max_retries: int = 0,
        rate_limit_backoff_s: float = 2.0,
    ) -> None:
        resolved = base_url or PROVIDER_BASE_URLS.get(provider)
        if not resolved:
            raise error_cls(f"no base URL for provider {provider!r}")
        api_key = os.environ.get(api_key_env) if api_key_env else None
        if not api_key and provider != "local":
            raise error_cls(
                f"environment variable {api_key_env!r} is not set; "
                f"provider {provider!r} needs an API key"
            )
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._model = model
        self._extra_fields = extra_fields or {}
        self._error_cls = error_cls
        self._rate_limit_max_retries = max(0, int(rate_limit_max_retries))
        self._rate_limit_backoff_s = rate_limit_backoff_s
        self._client = httpx.Client(
            base_url=resolved.rstrip("/"),
            headers=headers,
            timeout=timeout_s,
            transport=transport,
        )

    @staticmethod
    def _body_snippet(response: httpx.Response) -> str:
        """The head of an error response body, for the raised message.

        4xx bodies carry the actual reason (llama-server, for one, says
        "exceeds the available context size" there) while the status line
        alone says nothing actionable.
        """
        try:
            text = " ".join(response.text.split())
        except Exception:  # noqa: BLE001 - a broken body must not mask the error
            return ""
        if not text:
            return ""
        return f" | body: {text[:_ERROR_BODY_MAX_CHARS]}"

    def _retry_wait_s(self, response: httpx.Response, attempt: int) -> float:
        """Wait before retry ``attempt``: Retry-After if sent, else backoff."""
        retry_after = response.headers.get("Retry-After")
        if retry_after is not None:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                pass  # HTTP-date form; fall through to the backoff
        return self._rate_limit_backoff_s * (2**attempt)

    def _post_with_retry(self, payload: dict) -> httpx.Response:
        """POST the completion, retrying retryable statuses with a wait.

        A 429/5xx is the server saying "not right now", not a dead endpoint;
        retrying here keeps a healthy-but-throttled endpoint from being
        evicted by the pool's transport-failure counter. Retries exhausted
        (or any other HTTP error) raise :class:`EndpointRequestError`.
        """
        for attempt in range(self._rate_limit_max_retries + 1):
            try:
                response = self._client.post("/chat/completions", json=payload)
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if (
                    status not in _RETRYABLE_STATUSES
                    or attempt >= self._rate_limit_max_retries
                ):
                    raise EndpointRequestError(
                        f"completion request failed: {exc}"
                        f"{self._body_snippet(exc.response)}",
                        status_code=status,
                    ) from exc
                wait_s = self._retry_wait_s(exc.response, attempt)
                logger.warning(
                    "HTTP %d from endpoint; retry %d/%d in %.1fs",
                    status,
                    attempt + 1,
                    self._rate_limit_max_retries,
                    wait_s,
                )
                time.sleep(wait_s)
            except httpx.HTTPError as exc:
                raise EndpointRequestError(
                    f"completion request failed: {exc}"
                ) from exc
        raise EndpointRequestError("completion request failed: retries exhausted")

    def complete(
        self,
        messages: list[dict],
        max_tokens: int,
        json_response: bool = False,
        allow_empty: bool = False,
    ) -> str:
        """Send one completion request and return the reply text.

        ``allow_empty`` returns ``""`` for a valid reply that carries no
        content (a blank frame in OCR, say) instead of raising. A transport
        failure always raises :class:`EndpointRequestError` so a pool can fail
        over; a malformed or empty-when-not-allowed reply raises the
        configured ``error_cls`` (a content fault, not a host fault).
        """
        payload = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
            **self._extra_fields,
        }
        if json_response:
            payload["response_format"] = {"type": "json_object"}
        response = self._post_with_retry(payload)
        body = response.json()
        try:
            message = body["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise self._error_cls(
                f"unexpected completion payload shape: {exc}"
            ) from exc
        if not isinstance(message, dict):
            raise self._error_cls("unexpected completion payload shape: message")
        content = message.get("content")
        if not content:
            if allow_empty and not (
                message.get("reasoning") or message.get("reasoning_content")
            ):
                return ""
            # Reasoning models can spend the whole token budget thinking,
            # leaving the reasoning field populated and content empty.
            if message.get("reasoning") or message.get("reasoning_content"):
                raise self._error_cls(
                    "model returned reasoning but no content; raise "
                    "max_output_tokens or set thinking: off"
                )
            raise self._error_cls("completion contained no content")
        return strip_reasoning(content)

    def close(self) -> None:
        self._client.close()


class _EndpointHealth:
    """Thread-safe per-endpoint failure tracker shared across a pool's calls.

    An endpoint that raises ``max_retries`` *consecutive* transport errors is
    evicted (removed from rotation) for the rest of the run; any success resets
    its counter.
    """

    def __init__(self, n: int, max_retries: int) -> None:
        self._lock = threading.Lock()
        self._live = set(range(n))
        self._fails = [0] * n
        self.max_retries = max(1, int(max_retries))

    def live(self) -> list[int]:
        with self._lock:
            return sorted(self._live)

    def record_success(self, idx: int) -> None:
        with self._lock:
            self._fails[idx] = 0

    def record_failure(self, idx: int) -> tuple[bool, int]:
        """Count one transport error; return ``(evicted_now, consecutive)``."""
        with self._lock:
            self._fails[idx] += 1
            fails = self._fails[idx]
            if idx in self._live and fails >= self.max_retries:
                self._live.discard(idx)
                return True, fails
            return False, fails


class EndpointPool:
    """A set of interchangeable OpenAI-compatible endpoints with failover.

    Two access patterns share one health tracker:

    - :meth:`complete` — one request for the online single-call stages (Query
      Cortex, VLM verify). It picks a live endpoint (round-robin for load
      spread); on a transport failure it fails over to the next live one and
      only raises once every live endpoint has failed. A content fault (a
      valid reply the caller cannot use) propagates immediately without a host
      penalty.
    - :meth:`run_batch` — many independent items for the offline batch stages
      (captioning, VLM OCR). ``num_parallel`` worker threads per live endpoint
      pull from ONE shared queue, so a fast endpoint does more work and a dead
      endpoint's queued items are served by the survivors. An item whose
      in-flight request hits a transport error is retried on the remaining
      endpoints across a bounded number of rounds (cross-endpoint, so a mixed
      Qwen/Gemma pool degrades gracefully); a content fault is returned as the
      item's result with no retry.
    """

    def __init__(
        self,
        clients: list[ChatCompletionsClient],
        *,
        host_max_retries: int,
        error_cls: type[EndpointError] = EndpointError,
    ) -> None:
        if not clients:
            raise error_cls("EndpointPool needs at least one endpoint")
        self._clients = clients
        self._health = _EndpointHealth(len(clients), host_max_retries)
        self._error_cls = error_cls
        self._rr = count()

    @property
    def size(self) -> int:
        return len(self._clients)

    def complete(
        self,
        messages: list[dict],
        max_tokens: int,
        json_response: bool = False,
        allow_empty: bool = False,
    ) -> str:
        live = self._health.live()
        if not live:
            raise self._error_cls("no live endpoints remain in the pool")
        start = next(self._rr)
        order = [live[(start + k) % len(live)] for k in range(len(live))]
        last: Exception | None = None
        for idx in order:
            try:
                text = self._clients[idx].complete(
                    messages, max_tokens, json_response, allow_empty
                )
            except EndpointRequestError as exc:
                last = exc
                if rejected_request(exc):
                    # The host is alive and refused THIS request; try the
                    # next endpoint (a mixed pool may accept it) but leave
                    # the health counter alone.
                    logger.warning(
                        "endpoint %d rejected the request "
                        "(not counted toward eviction): %s",
                        idx,
                        exc,
                    )
                    continue
                evicted, fails = self._health.record_failure(idx)
                logger.warning(
                    "endpoint %d request failed (%d/%d consecutive)%s: %s",
                    idx,
                    fails,
                    self._health.max_retries,
                    "; evicted" if evicted else "",
                    exc,
                )
                continue
            self._health.record_success(idx)
            return text
        raise self._error_cls(
            f"all {len(order)} live endpoint(s) failed; last error: {last}"
        )

    def run_batch(
        self,
        items: list[_T],
        fn: Callable[[ChatCompletionsClient, _T], object],
        num_parallel: int,
    ) -> list[object]:
        """Run ``fn(client, item)`` for every item across the live endpoints.

        Worker threads (``num_parallel`` per live endpoint) pull from one
        shared queue, so a fast endpoint does more work and a dead endpoint's
        *queued* items are served by the survivors automatically. ``fn`` must
        raise :class:`EndpointRequestError` for a transport failure and any
        other exception for a content failure (kept as the item's result, no
        host penalty). A transport-failed item is requeued for another live
        endpoint (cross-endpoint failover) whenever one exists; with a single
        endpoint it is attempted once and then recorded as failed. The retry
        is bounded by eviction, so the pass always terminates. A *rejected*
        request (non-retryable 4xx, see :func:`rejected_request`) is recorded
        immediately with no health penalty and no requeue.

        Returns one result per item in input order: ``fn``'s value on success
        or the exception that ended its attempt.
        """
        results: list[object] = [None] * len(items)
        if not items:
            return results
        live = self._health.live()
        if not live:
            raise self._error_cls("no live endpoints remain in the pool")

        work: queue.Queue = queue.Queue()
        for idx in range(len(items)):
            work.put(idx)

        def worker(endpoint: int) -> None:
            while endpoint in self._health.live():
                try:
                    item_idx = work.get_nowait()
                except queue.Empty:
                    return
                try:
                    results[item_idx] = fn(self._clients[endpoint], items[item_idx])
                except EndpointRequestError as exc:
                    results[item_idx] = exc  # provisional; a retry may overwrite
                    if rejected_request(exc):
                        # Deterministic 4xx: the host is alive, the request
                        # is the problem. No health penalty, and no requeue —
                        # without eviction bounding the retries, requeueing
                        # a request every endpoint rejects would ping-pong
                        # forever.
                        logger.warning(
                            "endpoint %d rejected item "
                            "(bad request; not requeued): %s",
                            endpoint,
                            exc,
                        )
                        continue
                    evicted, fails = self._health.record_failure(endpoint)
                    if set(self._health.live()) - {endpoint}:
                        work.put(item_idx)  # another live endpoint can retry it
                    logger.warning(
                        "endpoint %d request failed (%d/%d consecutive)%s: %s",
                        endpoint,
                        fails,
                        self._health.max_retries,
                        "; evicted" if evicted else "",
                        exc,
                    )
                    if evicted:
                        return  # stop pulling; survivors drain the queue
                except Exception as exc:  # noqa: BLE001 - content fault, keep it
                    self._health.record_success(endpoint)
                    results[item_idx] = exc
                else:
                    self._health.record_success(endpoint)

        threads: list[threading.Thread] = []
        for endpoint in live:
            for _ in range(max(1, num_parallel)):
                thread = threading.Thread(
                    target=worker, args=(endpoint,), daemon=True
                )
                thread.start()
                threads.append(thread)
        for thread in threads:
            thread.join()

        # Items left unserved because every endpoint was evicted mid-run.
        for idx, result in enumerate(results):
            if result is None:
                results[idx] = self._error_cls("no live endpoint served this item")
        return results

    def close(self) -> None:
        for client in self._clients:
            client.close()


def build_endpoint_pool(
    *,
    provider: str,
    base_url: str | list[str] | None,
    model: str | list[str],
    api_key_env: str | None,
    timeout_s: float,
    host_max_retries: int,
    rate_limit_max_retries: int = 0,
    rate_limit_backoff_s: float = 2.0,
    extra_fields_for: Callable[[str], dict] | None = None,
    transport: httpx.BaseTransport | None = None,
    error_cls: type[EndpointError] = EndpointError,
) -> EndpointPool:
    """Build an :class:`EndpointPool` with one client per resolved endpoint.

    ``extra_fields_for(model)`` supplies the per-endpoint extra request fields
    (thinking/sampling), so a stage mixing model families still sends each
    endpoint the fields appropriate to the model it serves.
    """
    endpoints = resolve_endpoints(base_url, model)
    clients = [
        ChatCompletionsClient(
            provider=provider,
            model=endpoint_model,
            api_key_env=api_key_env,
            base_url=endpoint_url,
            timeout_s=timeout_s,
            extra_fields=(
                extra_fields_for(endpoint_model) if extra_fields_for else None
            ),
            transport=transport,
            error_cls=error_cls,
            rate_limit_max_retries=rate_limit_max_retries,
            rate_limit_backoff_s=rate_limit_backoff_s,
        )
        for endpoint_url, endpoint_model in endpoints
    ]
    return EndpointPool(
        clients, host_max_retries=host_max_retries, error_cls=error_cls
    )
