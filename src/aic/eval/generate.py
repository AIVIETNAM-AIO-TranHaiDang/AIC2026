"""Synthetic fixture generation: a VLM writes KIS queries for sampled windows.

Note 15: the 10-case hand-labelled fixture is too small to measure a change
— one flipped case moves recall by 10 points. Here windows of the ingested
corpus are sampled with exact labels by construction and a VLM writes an
AIC-style Vietnamese query for each, scaling measurement (and QPP training)
by an order of magnitude overnight. The model sees ONLY the window's
keyframes — never the corpus captions/ASR — so the queries cannot leak the
caption channel's vocabulary and unfairly reward one retrieval channel.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image

from aic.config import Config, FixtureGenConfig
from aic.eval.fixture import EvalFixture
from aic.ingest.pipeline import KEYFRAMES_MANIFEST, VIDEOS_MANIFEST
from aic.manifest import read_manifest
from aic.oaicompat import (
    ChatCompletionsClient,
    EndpointError,
    EndpointRequestError,
    build_endpoint_pool,
    image_to_data_url,
    parse_json_reply,
    sampling_payload,
    thinking_payload,
)

logger = logging.getLogger(__name__)

# Language- and domain-neutral by default (the corpus can be any kind of video
# in any language); a project that wants a fixed target language or genre sets
# fixture_gen.system_prompt. The model writes in the footage's own language.
_SYSTEM_PROMPT = (
    "You write text search queries for a video known-item-search task. The "
    "user shows you keyframes from ONE specific moment of one video. Write "
    "exactly one query, in the primary language of the footage (its "
    "on-screen text and speech), describing that moment: two to four "
    "sentences, concrete observable details (people, clothing and colours, "
    "objects, visible text, setting, actions), specific enough that only "
    "this moment matches. Never mention keyframes, images, or that you are "
    'describing pictures. Reply with a JSON object {"text": "<the query>"} '
    "and nothing else."
)


_REPAIR_PROMPT = (
    "Your previous reply was not valid JSON with the required field. "
    'Reply again with ONLY the JSON object {"text": "<the query>"}, no '
    "prose, no code fences."
)

# QA-mode generation (phase 3): the model writes a question + accepted answers
# grounded ONLY in the frames it saw, and self-asserts the answer is visible
# from them (anti-leak + anti-ambiguity, risk 3.1). Language- and domain-neutral
# by default (write in the footage's own language); a project fixes a target
# language via fixture_gen.qa_system_prompt. {archetype} is the target archetype
# the generator is stratified over so the set is not all one type.
_QA_SYSTEM_PROMPT = (
    "You write one question-and-answer item about ONE specific moment of a "
    "video, for a video Q&A task. You are shown keyframes from that moment. "
    "Write the question in the primary language of the footage (its on-screen "
    "text and speech). Aim for a '{archetype}' question (count = how many; "
    "read_text = what on-screen text says; entity = who/which/where; "
    "attribute = what colour/kind; cumulative = how many times over a span; "
    "other = anything else), but only if the frames actually support it. "
    "The question and answers must be answerable from these frames ALONE — "
    "never invent context you cannot see. Reply with ONLY this JSON object:\n"
    '{"question": "<the question>", '
    '"accepted_answers": ["<answer>", "<a paraphrase or variant>"], '
    '"archetype": "<count|read_text|entity|attribute|cumulative|other>", '
    '"answer_visible": true or false}\n'
    'Set "answer_visible" to false if the answer cannot be read from the '
    "frames; such items are discarded, so do not guess to fill the field."
)

_QA_REPAIR_PROMPT = (
    "Your previous reply was not the required JSON object. Reply again with "
    "ONLY the JSON object {question, accepted_answers, archetype, "
    "answer_visible}, no prose, no code fences."
)


class FixtureGenError(EndpointError):
    """Raised when a query cannot be generated for a window."""


@dataclass(frozen=True)
class SampledWindow:
    video_id: str
    t_start_ms: int
    t_end_ms: int
    frame_paths: list[str]
    is_qa: bool = False
    target_archetype: str | None = None

    @property
    def query_id(self) -> str:
        prefix = "qa" if self.is_qa else "syn"
        return f"{prefix}_{self.video_id}_{self.t_start_ms}"


@dataclass(frozen=True)
class GenerationStats:
    generated: int
    failed: int


def sample_windows(cfg: Config, gen_cfg: FixtureGenConfig) -> list[SampledWindow]:
    """Random non-overlapping truth windows anchored on real keyframes.

    Anchoring on keyframes (rather than uniform timestamps) guarantees every
    window has visual evidence to describe and to retrieve.
    """
    frames_by_video: dict[str, list[tuple[int, str]]] = {}
    for record in read_manifest(cfg.paths.manifests_dir / KEYFRAMES_MANIFEST):
        frames_by_video.setdefault(record["video_id"], []).append(
            (record["timestamp_ms"], record["image_path"])
        )
    anchors = [
        (video_id, ts, path)
        for video_id, frames in frames_by_video.items()
        for ts, path in frames
    ]
    if not anchors:
        raise FileNotFoundError(
            "keyframes manifest is empty; run ingestion before generating"
        )
    rng = random.Random(gen_cfg.seed)
    rng.shuffle(anchors)

    windows: list[SampledWindow] = []
    taken: dict[str, list[tuple[int, int]]] = {}
    half = gen_cfg.window_ms // 2
    for video_id, ts, _path in anchors:
        if len(windows) >= gen_cfg.count:
            break
        start = max(0, ts - half)
        end = start + gen_cfg.window_ms
        if any(
            s < end and start < e for s, e in taken.get(video_id, [])
        ):
            continue
        in_window = sorted(
            (frame_ts, path)
            for frame_ts, path in frames_by_video[video_id]
            if start <= frame_ts <= end
        )
        if not in_window:
            continue
        # Spread the evidence over the window: evenly spaced picks.
        count = min(gen_cfg.frames_per_case, len(in_window))
        step = len(in_window) / count
        picks = [in_window[int(i * step)][1] for i in range(count)]
        taken.setdefault(video_id, []).append((start, end))
        windows.append(
            SampledWindow(
                video_id=video_id,
                t_start_ms=start,
                t_end_ms=end,
                frame_paths=picks,
            )
        )
    if len(windows) < gen_cfg.count:
        logger.warning(
            "sampled %d/%d windows (corpus too small for more "
            "non-overlapping windows)",
            len(windows),
            gen_cfg.count,
        )
    return windows


def _parse_query_json(reply: str) -> str:
    try:
        payload = parse_json_reply(reply)
    except json.JSONDecodeError as exc:
        raise FixtureGenError(f"generator returned non-JSON output: {exc}") from exc
    text = payload.get("text") if isinstance(payload, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise FixtureGenError('generator reply lacks a non-empty "text" field')
    return text.strip()


def _frame_content(window: SampledWindow, gen_cfg: FixtureGenConfig) -> list[dict]:
    content: list[dict] = [
        {
            "type": "text",
            "text": (
                f"These {len(window.frame_paths)} frames are from one "
                "moment of one video."
            ),
        }
    ]
    for path in window.frame_paths:
        with Image.open(path) as img:
            array = np.array(img.convert("RGB"))
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image_to_data_url(array, gen_cfg.image_jpeg_quality)
                },
            }
        )
    return content


def _parse_qa_json(reply: str) -> dict:
    """Validate a QA-generation reply; raise to discard an unusable case.

    A ``answer_visible: false`` self-assertion discards the case (risk 3.1) —
    it is raised as a content failure so the run's failure counter records it,
    exactly like a malformed reply.
    """
    try:
        payload = parse_json_reply(reply)
    except json.JSONDecodeError as exc:
        raise FixtureGenError(f"qa generator returned non-JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise FixtureGenError("qa generator reply was not a JSON object")
    if payload.get("answer_visible") is not True:
        raise FixtureGenError("qa case discarded: answer_visible not asserted")
    question = payload.get("question")
    if not isinstance(question, str) or not question.strip():
        raise FixtureGenError('qa reply lacks a non-empty "question"')
    accepted = payload.get("accepted_answers")
    if not isinstance(accepted, list):
        raise FixtureGenError('qa reply "accepted_answers" is not a list')
    answers = [a.strip() for a in accepted if isinstance(a, str) and a.strip()]
    if not answers:
        raise FixtureGenError('qa reply has no non-empty "accepted_answers"')
    archetype = payload.get("archetype")
    return {
        "question": question.strip(),
        "accepted_answers": answers,
        "archetype": archetype if isinstance(archetype, str) and archetype else None,
    }


def _complete_with_repair(
    client: ChatCompletionsClient,
    messages: list[dict],
    gen_cfg: FixtureGenConfig,
    parse,
    repair_prompt: str,
    query_id: str,
):
    """One completion + a single repair retry on a content failure (X3).

    A transport failure propagates so ``run_batch`` can requeue the window; a
    content failure (bad JSON, failed self-assertion) is retried once and then
    raised, ending this window as a recorded failure.
    """
    reply = client.complete(
        messages, max_tokens=gen_cfg.max_output_tokens, json_response=True
    )
    try:
        return parse(reply)
    except FixtureGenError:
        logger.warning(
            "generator reply invalid for %s, retrying once with repair prompt",
            query_id,
        )
        messages = [
            *messages,
            {"role": "assistant", "content": reply},
            {"role": "user", "content": repair_prompt},
        ]
        return parse(
            client.complete(
                messages, max_tokens=gen_cfg.max_output_tokens, json_response=True
            )
        )


def _generate_one(
    client: ChatCompletionsClient,
    window: SampledWindow,
    gen_cfg: FixtureGenConfig,
) -> str | dict:
    """Generate one case body: a KIS query string, or a QA dict for a qa window."""
    if window.is_qa:
        archetype = window.target_archetype or "other"
        # replace, not str.format: the prompt embeds literal JSON braces that
        # str.format would misread as fields.
        system = (gen_cfg.qa_system_prompt or _QA_SYSTEM_PROMPT).replace(
            "{archetype}", archetype
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": _frame_content(window, gen_cfg)},
        ]
        return _complete_with_repair(
            client, messages, gen_cfg, _parse_qa_json, _QA_REPAIR_PROMPT,
            window.query_id,
        )
    messages = [
        {"role": "system", "content": gen_cfg.system_prompt or _SYSTEM_PROMPT},
        {"role": "user", "content": _frame_content(window, gen_cfg)},
    ]
    return _complete_with_repair(
        client, messages, gen_cfg, _parse_query_json, _REPAIR_PROMPT, window.query_id
    )


def _assign_qa_modes(
    windows: list[SampledWindow], gen_cfg: FixtureGenConfig
) -> list[SampledWindow]:
    """Flag a ``qa_fraction`` share of windows as qa, round-robin archetypes.

    Deterministic in ``seed`` (a distinct stream from window sampling) so a
    regenerated set is identical. Stratifying the archetype keeps the synthetic
    set from drifting to all read_text (risk 3.5).
    """
    if gen_cfg.qa_fraction <= 0.0 or not gen_cfg.qa_archetypes:
        return windows
    rng = random.Random(gen_cfg.seed + 1)
    out: list[SampledWindow] = []
    qa_index = 0
    for window in windows:
        if rng.random() < gen_cfg.qa_fraction:
            archetype = gen_cfg.qa_archetypes[qa_index % len(gen_cfg.qa_archetypes)]
            qa_index += 1
            out.append(replace(window, is_qa=True, target_archetype=archetype))
        else:
            out.append(window)
    return out


def generate_fixture(
    cfg: Config, gen_cfg: FixtureGenConfig, out_path: Path, transport=None
) -> GenerationStats:
    """Sample windows, generate queries, write an eval-compatible fixture.

    The output is validated through :class:`EvalFixture` before writing, so
    a generated file can never be one the harness refuses to load. Failed
    windows are skipped with a count (a smaller fixture is still useful).
    """
    windows = _assign_qa_modes(sample_windows(cfg, gen_cfg), gen_cfg)

    def extra_fields_for(model_name: str) -> dict:
        per_model = gen_cfg.model_copy(update={"model": model_name})
        return {**thinking_payload(per_model), **sampling_payload(per_model)}

    pool = build_endpoint_pool(
        provider=gen_cfg.provider,
        base_url=gen_cfg.base_url,
        model=gen_cfg.model,
        api_key_env=gen_cfg.api_key_env,
        timeout_s=gen_cfg.timeout_s,
        host_max_retries=gen_cfg.host_max_retries,
        rate_limit_max_retries=gen_cfg.rate_limit_max_retries,
        rate_limit_backoff_s=gen_cfg.rate_limit_backoff_s,
        extra_fields_for=extra_fields_for,
        transport=transport,
        error_cls=FixtureGenError,
    )

    from tqdm.auto import tqdm
    from tqdm.contrib.logging import logging_redirect_tqdm

    cases: list[dict] = []
    failed = 0
    try:
        with (
            logging_redirect_tqdm(),
            tqdm(total=len(windows), desc="queries", unit="case") as bar,
        ):

            def task(client: ChatCompletionsClient, window: SampledWindow) -> str:
                # Progress ticks on settled outcomes only: a transport
                # failure is requeued to another endpoint by run_batch, so
                # ticking it would overshoot the bar's total. tqdm.update
                # is thread-safe (its default lock).
                try:
                    result = _generate_one(client, window, gen_cfg)
                except EndpointRequestError:
                    raise
                except Exception:
                    bar.update(1)
                    raise
                bar.update(1)
                return result

            results = pool.run_batch(windows, task, gen_cfg.num_parallel)
            for window, result in zip(windows, results, strict=True):
                if isinstance(result, Exception):
                    logger.error(
                        "generation failed for %s: %s", window.query_id, result
                    )
                    failed += 1
                    continue
                truths = [
                    {
                        "video_id": window.video_id,
                        "t_start_ms": window.t_start_ms,
                        "t_end_ms": window.t_end_ms,
                    }
                ]
                if isinstance(result, dict):
                    case = {
                        "query_id": window.query_id,
                        "kind": "qa",
                        "question": result["question"],
                        "accepted_answers": result["accepted_answers"],
                        "truths": truths,
                    }
                    if result["archetype"] is not None:
                        case["archetype"] = result["archetype"]
                    cases.append(case)
                else:
                    cases.append(
                        {
                            "query_id": window.query_id,
                            "kind": "kis_t",
                            "text": result,
                            "truths": truths,
                        }
                    )
            bar.set_postfix(ok=len(cases), failed=failed)
    finally:
        pool.close()

    if not cases:
        raise FixtureGenError("no case generated; nothing written")
    corpus_videos = sorted(
        record["video_id"]
        for record in read_manifest(cfg.paths.manifests_dir / VIDEOS_MANIFEST)
        if record.get("status") == "done"
    )
    payload = {"corpus_videos": corpus_videos, "cases": cases}
    # Validate through the harness schema BEFORE writing: a generated file
    # must never be one load_fixture refuses.
    EvalFixture.model_validate(payload)

    import yaml

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return GenerationStats(generated=len(cases), failed=failed)
