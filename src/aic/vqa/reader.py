"""Track B grounded reader: a VLM reads one candidate moment's evidence.

Mirrors :class:`aic.escalate.verify.VlmVerify` (lazy endpoint pool, frame
packing via ``image_to_data_url``, JSON parse with a single repair retry), but
reads each candidate INDEPENDENTLY rather than listwise: one grounded answer
per moment, so answers can be voted across moments (note 18 §3.4). A failed or
malformed read degrades to ``answerable=False`` (an abstention in the vote),
never raises. The pool's httpx timeout is ``reader.timeout_s``, which the
config validator pins ``<= vqa.timeout_s`` so an abandoned read cannot occupy
the server past the wall (X2/X9).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from pydantic import Field

from aic.config import QaReaderConfig, StrictModel
from aic.oaicompat import (
    ChatCompletionsClient,
    EndpointError,
    EndpointPool,
    build_endpoint_pool,
    image_to_data_url,
    parse_json_reply,
    sampling_payload,
    thinking_payload,
)
from aic.vqa.evidence import EvidencePack

logger = logging.getLogger(__name__)

# Default reader prompt; overridable via vqa.reader.system_prompt. Any override
# MUST keep the QaRead JSON contract the parser validates. Language-neutral by
# design (the corpus can be any language); it answers in the question's own
# language. The reader is never shown Track A's guess (X2.3 yes-bias): it is
# asked for the answer, not to confirm a hypothesis. Guessing is framed as
# worse than declining so an unreadable moment abstains instead of voting
# noise. Structure follows the persona/task/format prompting pattern.
_SYSTEM_PROMPT = (
    "You answer a question about ONE moment in a video, using only the "
    "evidence given (frames, on-screen text, speech, and a caption). The "
    "question and the evidence may be in any language. Reply with ONLY this "
    "JSON object and nothing else:\n"
    '{"answerable": true or false, "answer": "the answer as short text", '
    '"evidence_timestamp_ms": the millisecond timestamp your answer is '
    'visible at or null, "confidence": 0.0 to 1.0}\n'
    "If the answer is not visible in this moment's evidence, set "
    '"answerable" to false and "answer" to "" - declining is better than '
    "guessing. Answer in the same language as the question."
)

# Neutral English fallback labels when a caller does not pass localised ones
# (aic.config.VqaConfig.evidence_labels). Structural framing only; the values
# stay in the footage's language.
_DEFAULT_LABELS = {
    "ocr": "On-screen text",
    "asr": "Speech",
    "caption": "Caption",
    "question": "Question",
    "proposed_answer": "Proposed answer",
}

_REPAIR_PROMPT = (
    "Your previous reply was not the required JSON object. Reply again with "
    "ONLY the JSON object, no prose, no code fences."
)

# Default count-read prompts; overridable via vqa.reader.count_system_prompt /
# count_user_prompt. {question}/{concept} in the user prompt are substituted
# with str.replace (never str.format — an override may contain literal braces).
_COUNT_SYSTEM_PROMPT = (
    "You count objects in video frames. Reply with ONLY a single "
    "non-negative integer, no words."
)
_COUNT_USER_PROMPT = (
    "{question_label}: {question}\n"
    "How many '{concept}' are visible? Reply with only a single integer."
)

# Default disconfirmation prompt; overridable via
# vqa.reader.disconfirm_system_prompt.
_DISCONFIRM_SYSTEM_PROMPT = (
    "You check a proposed answer against ONE moment's evidence. Reply "
    "with ONLY this JSON object: "
    '{"contradicts": true or false, "why": "short reason"}. Set '
    '"contradicts" to true ONLY if the evidence clearly shows the '
    "answer is wrong; uncertainty is not contradiction."
)


class QaRead(StrictModel):
    """One candidate moment's grounded answer."""

    answerable: bool = Field(
        default=False,
        description="False marks an abstention: the moment's evidence does "
        "not answer the question, so the vote ignores it (not a zero-vote).",
    )
    answer: str = Field(
        default="",
        description="Short answer text as read from THIS moment; empty when "
        "unanswerable.",
    )
    evidence_timestamp_ms: int | None = Field(
        default=None,
        description="Millisecond timestamp the answer is visible at, clamped "
        "to the candidate window; None falls back to the grounded keyframe.",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="The reader's certainty for this moment.",
    )


def _abstention() -> QaRead:
    return QaRead(
        answerable=False, answer="", evidence_timestamp_ms=None, confidence=0.0
    )


class QaReader:
    """Per-candidate grounded reads through a VLM endpoint pool.

    Construction is lazy-safe: the pool is built on first read so no endpoint
    is contacted at service assembly. ``frames_dir`` is the keyframes root the
    pack's frame names are relative to (``paths.keyframes_dir``); None or a
    zero ``frames_per_candidate`` sends a text-only read (still valid — U10).
    """

    def __init__(
        self,
        cfg: QaReaderConfig,
        frames_dir: Path | None,
        labels: dict[str, str] | None = None,
        transport=None,
    ) -> None:
        self._cfg = cfg
        self._frames_dir = frames_dir
        self._labels = labels or _DEFAULT_LABELS
        self._transport = transport
        self._client: EndpointPool | None = None

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

    def _images(self, pack: EvidencePack) -> list[str]:
        if self._cfg.frames_per_candidate <= 0 or self._frames_dir is None:
            return []
        urls = []
        for name in pack.frames[: self._cfg.frames_per_candidate]:
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

    def read_count(
        self,
        concept: str,
        question: str,
        frame_paths: list[str],
        crop: tuple[float, float, float, float] | None = None,
    ) -> int | None:
        """Ask the VLM to count a concept in some frames (optional crop box).

        The count program's independent VLM estimate + its crop adjudicator
        (aic.vqa.count). Returns an integer, or None on failure/no digit. The
        crop is in absolute pixels and clamped to each frame; a language-neutral
        prompt asks for a bare number.
        """
        import re

        from PIL import Image

        if self._cfg.frames_per_candidate <= 0 or not frame_paths:
            return None
        user_text = (
            (self._cfg.count_user_prompt or _COUNT_USER_PROMPT)
            .replace("{question_label}", self._labels["question"])
            .replace("{question}", question)
            .replace("{concept}", concept)
        )
        content: list[dict] = [{"type": "text", "text": user_text}]
        for path in frame_paths[: self._cfg.frames_per_candidate]:
            try:
                with Image.open(path) as image:
                    rgb = image.convert("RGB")
                    if crop is not None:
                        w, h = rgb.size
                        box = (
                            max(0, int(crop[0])),
                            max(0, int(crop[1])),
                            min(w, int(crop[2])),
                            min(h, int(crop[3])),
                        )
                        if box[2] > box[0] and box[3] > box[1]:
                            rgb = rgb.crop(box)
                    array = np.asarray(rgb)
            except OSError as exc:
                logger.warning("count keyframe %s unreadable: %s", path, exc)
                continue
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_to_data_url(array, self._cfg.image_jpeg_quality)
                    },
                }
            )
        messages = [
            {
                "role": "system",
                "content": self._cfg.count_system_prompt or _COUNT_SYSTEM_PROMPT,
            },
            {"role": "user", "content": content},
        ]
        try:
            reply = self._ensure_client().complete(
                messages, self._cfg.max_output_tokens, json_response=False
            )
        except EndpointError as exc:
            logger.warning("count read failed: %s", exc)
            return None
        match = re.search(r"\d+", reply)
        return int(match.group()) if match else None

    def _messages(self, pack: EvidencePack, question: str) -> list[dict]:
        label = self._labels["question"]
        content: list[dict] = [{"type": "text", "text": f"{label}: {question}"}]
        if pack.text_block:
            content.append({"type": "text", "text": pack.text_block})
        for url in self._images(pack):
            content.append({"type": "image_url", "image_url": {"url": url}})
        return [
            {"role": "system", "content": self._cfg.system_prompt or _SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

    def _parse(self, reply: str, pack: EvidencePack) -> QaRead:
        parsed = parse_json_reply(reply)
        if not isinstance(parsed, dict):
            raise ValueError("reader reply was not a JSON object")
        # Tolerate extra keys: a chatty model adding e.g. "reason" must not
        # cost the moment its vote (QaRead is a StrictModel, so validating
        # the raw dict would reject the whole read over a harmless field).
        known = {k: v for k, v in parsed.items() if k in QaRead.model_fields}
        read = QaRead.model_validate(known)
        return read.model_copy(
            update={
                "evidence_timestamp_ms": _clamp_timestamp(
                    read.evidence_timestamp_ms, pack
                )
            }
        )

    def _read_one(
        self, client: ChatCompletionsClient, pack: EvidencePack, question: str
    ) -> QaRead:
        """The per-candidate ``run_batch`` worker: one read + one repair retry.

        Transport failures (``EndpointRequestError``) propagate so ``run_batch``
        can fail over; a content fault (bad JSON both times, empty reply)
        returns an abstention so the vote simply ignores the moment.
        """
        messages = self._messages(pack, question)
        reply = client.complete(
            messages, self._cfg.max_output_tokens, json_response=True
        )
        try:
            return self._parse(reply, pack)
        except (ValueError, json.JSONDecodeError):
            logger.warning("reader JSON invalid for %s; repairing once", pack.shot_key)
        repair = [
            *messages,
            {"role": "assistant", "content": reply},
            {"role": "user", "content": _REPAIR_PROMPT},
        ]
        reply = client.complete(repair, self._cfg.max_output_tokens, json_response=True)
        try:
            return self._parse(reply, pack)
        except (ValueError, json.JSONDecodeError):
            logger.warning("reader repair failed for %s; abstaining", pack.shot_key)
            return _abstention()

    def read_many(self, packs: list[EvidencePack], question: str) -> list[QaRead]:
        """One :class:`QaRead` per pack, fanned across the pool; never raises.

        Results are in input order. A candidate whose read raised (transport
        exhaustion after failover, or a content-fault error) degrades to an
        abstention, so a partial endpoint outage narrows the vote rather than
        failing the whole question.
        """
        if not packs:
            return []
        client = self._ensure_client()
        raw = client.run_batch(
            packs,
            lambda c, pack: self._read_one(c, pack, question),
            self._cfg.num_parallel,
        )
        reads: list[QaRead] = []
        for pack, result in zip(packs, raw, strict=True):
            if isinstance(result, QaRead):
                reads.append(result)
            else:
                logger.warning("reader failed for %s: %s", pack.shot_key, result)
                reads.append(_abstention())
        return reads

    def disconfirm(self, pack: EvidencePack, question: str, answer: str) -> bool:
        """Ask whether ``pack``'s evidence CONTRADICTS ``answer``; never raises.

        One negatively-framed call (CoVe/CASHEW-shaped) against the runner-up
        moment. Any failure returns False (the vote stands — degrade to the
        un-checked winner). Only this call ever sees a proposed answer, so the
        grounded read stays free of the yes-bias (X2.3).
        """
        system = self._cfg.disconfirm_system_prompt or _DISCONFIRM_SYSTEM_PROMPT
        content: list[dict] = [
            {"type": "text", "text": f"{self._labels['question']}: {question}"},
            {
                "type": "text",
                "text": f"{self._labels['proposed_answer']}: {answer}",
            },
        ]
        if pack.text_block:
            content.append({"type": "text", "text": pack.text_block})
        for url in self._images(pack):
            content.append({"type": "image_url", "image_url": {"url": url}})
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]
        try:
            client = self._ensure_client()
            reply = client.complete(
                messages, self._cfg.max_output_tokens, json_response=True
            )
            parsed = parse_json_reply(reply)
        except (EndpointError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("disconfirmation failed (%s); vote stands", exc)
            return False
        return isinstance(parsed, dict) and parsed.get("contradicts") is True

    def close(self) -> None:
        if self._client is not None:
            self._client.close()


def _clamp_timestamp(value: int | None, pack: EvidencePack) -> int:
    """Sanitise a read's timestamp against the pack's grounded anchor (X2.8).

    A model may omit or invent a negative timestamp; either falls back to the
    pack's grounded keyframe timestamp (the KIS grounding rule). The vote uses
    the grounded anchor for the submitted frame regardless, so this value is
    advisory — it only ever needs to be a plausible, non-negative millisecond.
    """
    if value is None or value < 0:
        return pack.grounded_timestamp_ms
    return value
