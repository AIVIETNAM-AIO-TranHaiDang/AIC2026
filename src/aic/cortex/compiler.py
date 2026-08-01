"""LLM compilation of a raw query into a :class:`QuerySpec`.

Provider-agnostic through the shared chat-completions client (Gemini by
default, any OpenAI-compatible endpoint by config). Failure handling is the
Phase 6 hard requirement: invalid JSON gets one repair retry, and any
remaining failure — including a disabled cortex or a missing endpoint —
falls back to the raw query as a single visual phrase. ``compile`` never
raises.
"""

from __future__ import annotations

import json
import logging

import httpx
from pydantic import ValidationError

from aic.config import CortexConfig
from aic.cortex.qa import ANSWER_TYPES, FALLBACK_ARCHETYPE, QaPlan, fallback_qa_plan
from aic.cortex.spec import QuerySpec, fallback_spec
from aic.oaicompat import (
    EndpointError,
    EndpointPool,
    build_endpoint_pool,
    parse_json_reply,
    sampling_payload,
    thinking_payload,
)

logger = logging.getLogger(__name__)

# Default compilation prompt; overridable via cortex.system_prompt in config
# (overrides must keep the QuerySpec JSON contract the parser validates).
# The default is deliberately domain- and language-agnostic: the corpus can be
# any kind of video in any language, so the prompt must not assume a genre
# (e.g. broadcast news) that would bias the compiled spec. Field languages
# still follow the index contents: visual phrases are matched by
# English-leaning embedding towers (so they are translated to English), while
# OCR/ASR literals are matched verbatim against on-screen text and speech in
# their original language. A genre-specialised prompt (e.g. Vietnamese news
# for the competition) belongs in cortex.system_prompt, not here; see
# docs/note/14.
_SYSTEM_PROMPT_TEMPLATE = (
    "You compile a search query into fields for a video retrieval system. "
    "The corpus can be any kind of video (news, sports, vlogs, film, "
    "user-generated, surveillance, and more) in any language, so make no "
    "assumption about the genre. The query itself may be written in any "
    "language. Respond with a single JSON object with exactly these fields:\n"
    '"visual_phrases": list of short English phrases describing what the '
    "target moment looks like - concrete subjects, actions, objects, and "
    "setting; translate any non-English description into English (the visual "
    "index is matched in English);\n"
    '"entity_terms": list of named entities (people, places, organisations, '
    "brands, titles) exactly as written in the query;\n"
    '"ocr_literals": list of strings expected to appear as on-screen text, '
    "verbatim, in their original language and script, diacritics preserved "
    "(do not translate);\n"
    '"asr_phrases": list of phrases likely spoken in the moment, in the '
    "language they would be spoken;\n"
    '"temporal_hints": list of ordering or timing constraints, in English;\n'
    '"negations": list of things the query says the target is NOT;\n'
    '"paraphrases": exactly {n_paraphrases} alternative English phrasings '
    "of the main visual description (empty list if zero requested);\n"
    '"confidence": 0.0-1.0, your certainty that you parsed the query '
    "correctly.\n"
    "Put a phrase only in the fields it belongs to; leave fields you have "
    "no evidence for as empty lists. Describe only what the query states or "
    "clearly implies - do not invent details."
)

_REPAIR_PROMPT = (
    "Your previous reply was not the required JSON object. Reply again with "
    "ONLY the JSON object, no prose, no code fences."
)

# Default VQA question-compilation prompt; overridable via
# cortex.qa_system_prompt. {archetypes} is the allowed archetype list.
_QA_SYSTEM_PROMPT_TEMPLATE = (
    "You compile a question about a moment in a video into a plan for a "
    "retrieval system. The corpus can be any kind of video in any language. "
    "Respond with a single JSON object with exactly these fields:\n"
    '"archetype": one of {archetypes} - the kind of question '
    "(count = how many; read_text = what does on-screen text say; entity = "
    "who/which/where; attribute = what colour/kind; cumulative = how many "
    "times/nights over a span; other = anything else);\n"
    '"locator": a JSON object describing the moment to find, with the same '
    "fields as a search query: "
    '"visual_phrases" (short English phrases of what the moment looks like, '
    "translated to English), "
    '"entity_terms" (named entities verbatim), '
    '"ocr_literals" (text expected on screen, verbatim original script), '
    '"asr_phrases" (phrases likely spoken, original language), '
    '"negations" (things the moment is NOT);\n'
    '"question_core": the question stripped to what is actually asked, '
    "verbatim in its original language;\n"
    '"expected_answer_type": one of number, text, entity, color, yes_no, '
    "duration;\n"
    '"concept": for a count question, the countable object noun phrase to '
    'detect (e.g. "person in a red shirt", "boat"), else null.\n'
    "Leave locator fields you have no evidence for as empty lists. Describe "
    "only what the question states or clearly implies."
)


class QaPlanParseError(EndpointError):
    """Internal QA compile failure; callers get the fallback plan instead."""


class CortexError(EndpointError):
    """Internal compiler failure; callers never see it (fallback instead)."""


def _parse_spec_json(text: str) -> QuerySpec:
    try:
        payload = parse_json_reply(text)
    except json.JSONDecodeError as exc:
        raise CortexError(f"cortex returned non-JSON output: {exc}") from exc
    if isinstance(payload, dict):
        payload.pop("source", None)  # the model does not decide provenance
        if payload.get("confidence") is not None:
            payload["confidence"] = min(
                max(float(payload["confidence"]), 0.0), 1.0
            )
    try:
        return QuerySpec.model_validate(payload)
    except (ValidationError, TypeError) as exc:
        raise CortexError(f"spec JSON failed validation: {exc}") from exc


def _parse_qa_json(text: str, archetypes: list[str]) -> QaPlan:
    try:
        payload = parse_json_reply(text)
    except json.JSONDecodeError as exc:
        raise QaPlanParseError(f"qa returned non-JSON output: {exc}") from exc
    if not isinstance(payload, dict):
        raise QaPlanParseError("qa reply was not a JSON object")
    locator_raw = payload.get("locator") or {}
    if not isinstance(locator_raw, dict):
        locator_raw = {}
    locator_raw.pop("source", None)  # the model does not decide provenance
    locator_raw.pop("confidence", None)
    try:
        locator = QuerySpec.model_validate(locator_raw)
    except (ValidationError, TypeError):
        locator = QuerySpec()
    archetype = payload.get("archetype")
    if archetype not in archetypes:
        archetype = FALLBACK_ARCHETYPE
    answer_type = payload.get("expected_answer_type")
    if answer_type not in ANSWER_TYPES:
        answer_type = "text"
    concept = payload.get("concept")
    return QaPlan(
        archetype=archetype,
        locator=locator,
        question_core=str(payload.get("question_core") or "").strip(),
        expected_answer_type=answer_type,
        concept=str(concept).strip() if isinstance(concept, str) and concept else None,
    )


class CortexCompiler:
    """Compiles queries; construction is lazy-safe and failure is silent.

    When the endpoint cannot even be constructed (missing API key), the
    compiler degrades to fallback specs permanently and logs once — the
    operator tool must keep working offline.
    """

    def __init__(
        self, cfg: CortexConfig, transport: httpx.BaseTransport | None = None
    ) -> None:
        self._cfg = cfg
        self._client: EndpointPool | None = None

        def extra_fields_for(model_name: str) -> dict:
            per_model = cfg.model_copy(update={"model": model_name})
            return {**thinking_payload(per_model), **sampling_payload(cfg)}

        if cfg.enabled:
            try:
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
                    transport=transport,
                    error_cls=CortexError,
                )
            except CortexError as exc:
                logger.warning(
                    "cortex endpoint unavailable (%s); every query will use "
                    "the fallback spec",
                    exc,
                )

    def compile(self, query: str) -> QuerySpec:
        """Compile ``query`` into a spec; never raises."""
        if self._client is None or not query.strip():
            return fallback_spec(query)
        messages = [
            {
                "role": "system",
                "content": self._cfg.system_prompt
                or _SYSTEM_PROMPT_TEMPLATE.format(
                    n_paraphrases=self._cfg.paraphrases
                ),
            },
            {"role": "user", "content": query},
        ]
        try:
            reply = self._client.complete(
                messages,
                max_tokens=self._cfg.max_output_tokens,
                json_response=True,
            )
        except CortexError as exc:
            logger.warning("cortex request failed (%s); using fallback", exc)
            return fallback_spec(query)
        try:
            spec = _parse_spec_json(reply)
        except CortexError:
            logger.warning("cortex JSON invalid, retrying once with repair prompt")
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": _REPAIR_PROMPT})
            try:
                spec = _parse_spec_json(
                    self._client.complete(
                        messages,
                        max_tokens=self._cfg.max_output_tokens,
                        json_response=True,
                    )
                )
            except CortexError as exc:
                logger.warning("cortex repair failed (%s); using fallback", exc)
                return fallback_spec(query)
        spec = spec.truncated(self._cfg.max_list_items)
        if spec.is_empty:
            # A structurally valid but empty spec would search for nothing;
            # the raw query is strictly better.
            logger.warning("cortex produced an empty spec; using fallback")
            return fallback_spec(query)
        return spec

    def compile_qa(self, question: str, archetypes: list[str]) -> QaPlan:
        """Compile a VQA question into a :class:`QaPlan`; never raises.

        ``archetypes`` is the allowed set (``vqa.archetypes``); an unknown or
        missing archetype falls back to ``other``. On any endpoint or JSON
        failure the whole question becomes the locator (the KIS-equivalent
        behaviour), so Track A still finds evidence.
        """
        if self._client is None or not question.strip():
            return fallback_qa_plan(question)
        system = self._cfg.qa_system_prompt or _QA_SYSTEM_PROMPT_TEMPLATE.format(
            archetypes=list(archetypes)
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ]
        try:
            reply = self._client.complete(
                messages, max_tokens=self._cfg.max_output_tokens, json_response=True
            )
        except CortexError as exc:
            logger.warning("qa compile request failed (%s); using fallback", exc)
            return fallback_qa_plan(question)
        try:
            plan = _parse_qa_json(reply, archetypes)
        except QaPlanParseError:
            logger.warning("qa JSON invalid, retrying once with repair prompt")
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": _REPAIR_PROMPT})
            try:
                plan = _parse_qa_json(
                    self._client.complete(
                        messages,
                        max_tokens=self._cfg.max_output_tokens,
                        json_response=True,
                    ),
                    archetypes,
                )
            except (QaPlanParseError, CortexError) as exc:
                logger.warning("qa repair failed (%s); using fallback", exc)
                return fallback_qa_plan(question)
        locator = plan.locator.truncated(self._cfg.max_list_items)
        if locator.is_empty:
            # A located moment needs at least one dispatchable clause; the raw
            # question is strictly better than nothing.
            locator = fallback_spec(plan.question_core or question)
        return plan.model_copy(update={"locator": locator})

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
