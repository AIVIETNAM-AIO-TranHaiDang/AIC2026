"""VLM verify: listwise re-order of the top evidence bundles.

Note 08 step 6c, miniReranker-style: the VLM sees the query plus a numbered
list of candidates (evidence text and optionally keyframes) and replies with
the candidate numbers best-first. Candidates the reply omits keep their
original relative order below the listed ones, and any endpoint failure
degrades to the unchanged pool with a note — the button is never worse than
not pressing it.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from aic.config import VlmVerifyConfig
from aic.escalate.base import EscalationOutcome, EscalationRequest
from aic.oaicompat import (
    EndpointError,
    EndpointPool,
    build_endpoint_pool,
    image_to_data_url,
    parse_json_reply,
    sampling_payload,
    thinking_payload,
)
from aic.service.bundles import EvidenceBundle

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You verify video-retrieval results. The user gives a search query and "
    "a numbered list of candidate shots with their evidence (caption, "
    "on-screen text, speech, and possibly keyframes). Order the candidates "
    "from best to worst match for the query. Reply with a JSON object "
    '{"order": [/* candidate numbers, best first */]} and nothing else.'
)


class VlmVerify:
    """Listwise re-rank of the top candidates through a VLM endpoint."""

    def __init__(
        self,
        bundles: Mapping[str, EvidenceBundle],
        frames_dir: Path | None,
        cfg: VlmVerifyConfig,
        transport=None,
    ) -> None:
        self._bundles = bundles
        self._frames_dir = frames_dir
        self._cfg = cfg
        self._transport = transport
        self._client: EndpointPool | None = None

    @property
    def name(self) -> str:
        return "vlm_verify"

    def _ensure_client(self) -> EndpointPool:
        if self._client is None:
            cfg = self._cfg

            def extra_fields_for(model_name: str) -> dict:
                per_model = cfg.model_copy(update={"model": model_name})
                return {**thinking_payload(per_model), **sampling_payload(cfg)}

            self._client = build_endpoint_pool(
                provider=cfg.provider,
                base_url=cfg.base_url,
                model=cfg.model,
                api_key_env=cfg.api_key_env,
                timeout_s=cfg.timeout_s,
                host_max_retries=cfg.host_max_retries,
                rate_limit_max_retries=cfg.rate_limit_max_retries,
                rate_limit_backoff_s=cfg.rate_limit_backoff_s,
                extra_fields_for=extra_fields_for,
                transport=self._transport,
            )
        return self._client

    def _candidate_images(self, bundle: EvidenceBundle) -> list[str]:
        if self._cfg.frames_per_candidate <= 0 or self._frames_dir is None:
            return []
        urls = []
        for name in bundle.frames[: self._cfg.frames_per_candidate]:
            path = self._frames_dir / name
            try:
                from PIL import Image

                with Image.open(path) as image:
                    array = np.asarray(image.convert("RGB"))
            except OSError as exc:
                logger.warning("keyframe %s unreadable: %s", path, exc)
                continue
            urls.append(image_to_data_url(array, self._cfg.image_jpeg_quality))
        return urls

    def _messages(self, request: EscalationRequest, pool) -> list[dict]:
        content: list[dict] = [
            {"type": "text", "text": f"Query: {request.query}"}
        ]
        for number, candidate in enumerate(pool, start=1):
            bundle = self._bundles.get(candidate.shot_key)
            evidence = (
                bundle.evidence_text[: self._cfg.max_evidence_chars]
                if bundle is not None
                else ""
            )
            content.append(
                {
                    "type": "text",
                    "text": f"Candidate {number}:\n{evidence or '(no evidence text)'}",
                }
            )
            if bundle is not None:
                for url in self._candidate_images(bundle):
                    content.append(
                        {"type": "image_url", "image_url": {"url": url}}
                    )
        return [
            {
                "role": "system",
                "content": self._cfg.system_prompt or _SYSTEM_PROMPT,
            },
            {"role": "user", "content": content},
        ]

    def _parse_order(self, reply: str, pool_size: int) -> list[int]:
        parsed = parse_json_reply(reply)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("order"), list):
            raise ValueError("reply lacks the required {'order': [...]} shape")
        order = []
        for item in parsed["order"]:
            if isinstance(item, bool) or not isinstance(item, int):
                continue
            if 1 <= item <= pool_size and item not in order:
                order.append(item)
        if not order:
            raise ValueError("reply ordered no valid candidate numbers")
        return order

    def run(self, request: EscalationRequest) -> EscalationOutcome:
        pool = request.candidates[: self._cfg.top_n]
        tail = request.candidates[self._cfg.top_n :]
        if not pool:
            return EscalationOutcome(request.candidates, note="nothing to verify")
        try:
            client = self._ensure_client()
            reply = client.complete(
                self._messages(request, pool),
                max_tokens=self._cfg.max_output_tokens,
                json_response=True,
            )
            order = self._parse_order(reply, len(pool))
        except (EndpointError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("vlm verify failed: %s", exc)
            return EscalationOutcome(
                request.candidates, note=f"verify failed: {exc}"
            )
        listed = [pool[number - 1] for number in order]
        listed_keys = {candidate.shot_key for candidate in listed}
        unlisted = [c for c in pool if c.shot_key not in listed_keys]
        return EscalationOutcome(listed + unlisted + tail)
