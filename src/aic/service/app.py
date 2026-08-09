"""The FastAPI application: search, feedback, sessions, submit, and the UI.

The app is a pure function of injected state (engine, compiler, bundles,
sessions), so tests exercise every endpoint with fakes and no model ever
loads at import time. Production state comes from
:func:`aic.service.state.build_service_state`.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from aic.config import ServiceConfig
from aic.cortex.dispatch import DispatchDetail, filter_negations
from aic.cortex.spec import QuerySpec
from aic.escalate.base import (
    Escalation,
    EscalationRequest,
    EscalationTimeout,
    run_with_timeout,
)
from aic.retrieval.fusion import ShotCandidate, ShotMeta
from aic.service.bundles import EvidenceBundle
from aic.service.engine import SearchEngine
from aic.service.planner import suggest_question
from aic.service.sessions import SessionError, SessionStore
from aic.service.submission import serialize_submission
from aic.vqa.answer import ledger_answer, track_b_groups
from aic.vqa.evidence import build_evidence_pack
from aic.vqa.reverse import ReverseProgram, ev_hint, is_weak_locator

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"


@contextmanager
def _timed(timings: dict[str, float], stage: str):
    """Record a stage's wall time into the per-request timings (Phase 9)."""
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[stage] = round((time.perf_counter() - start) * 1000, 1)


@dataclass
class ServiceState:
    """Everything the endpoints close over.

    ``compiler`` is anything with ``compile(str) -> QuerySpec`` (the
    CortexCompiler in production, a fake in tests)."""

    compiler: object
    engine: SearchEngine
    bundles: dict[str, EvidenceBundle]
    sessions: SessionStore
    shot_meta: dict[str, ShotMeta] = field(default_factory=dict)
    frames_dir: Path | None = None
    negation_filter: bool = True
    escalations: dict[str, Escalation] = field(default_factory=dict)
    escalation_timeouts: dict[str, float] = field(default_factory=dict)
    qpp: object | None = None
    """A QppAdvisor (anything with ``advise(spec, candidates, rankings,
    video_by_shot) -> dict``), or None when no artifact is configured."""
    auto_rerank: Escalation | None = None
    """Escalation run automatically on every /api/search result
    (retrieval.auto_rerank); failures degrade to the unchanged ranking."""
    auto_rerank_timeout_s: float = 20.0
    vqa_enabled: bool = False
    """When True the additive /api/qa Track A route is registered; when False
    the service is the KIS system, bit-identical."""
    ledger: dict[str, object] = field(default_factory=dict)
    """shot_key -> LedgerRow, joined onto QA answer cards (empty without a
    built Answer Ledger)."""
    vqa_archetypes: list[str] = field(default_factory=list)
    """The archetypes compile_qa may assign (vqa.archetypes)."""
    qa_reader: object | None = None
    """Track B grounded reader (anything with ``read_many(packs, question) ->
    list[QaRead]`` and ``disconfirm(pack, question, answer) -> bool``); None
    keeps /api/qa Track A only (phase-1 behaviour)."""
    vqa_cfg: object | None = None
    """The VqaConfig, needed by Track B for evidence-pack windows, voting, and
    answer normalisation. None with a reader present disables Track B."""
    count_program: object | None = None
    """The online count program (aic.vqa.count.CountProgram) for the 'count'
    archetype; None (or not .available) degrades a count question to the read."""
    video_shot_index: dict[str, list[tuple[int, str]]] | None = None
    """video_id -> [(midpoint_ms, shot_key), ...] (aic.vqa.evidence
    .build_video_shot_index), so evidence-pack window scans stay per-video.
    None (test fakes) lets the pack builder scan shot_meta whole."""
    qa_task_state: dict = field(default_factory=dict)
    """The EV hint's per-task submit budget (note 18 §3.6): the current
    question and how many submit_now hints it has received. A new question
    resets the budget (one task = one question)."""

    def evidence_text(self) -> dict[str, str]:
        return {key: b.evidence_text for key, b in self.bundles.items()}

    def video_by_shot(self) -> dict[str, str]:
        """shot_key -> video_id, for the QPP compactness feature."""
        mapping = {key: meta.video_id for key, meta in self.shot_meta.items()}
        for key, bundle in self.bundles.items():
            mapping.setdefault(key, bundle.video_id)
        return mapping

    def rank_detailed(self, spec: QuerySpec, raw_query: str = "") -> DispatchDetail:
        """Detailed ranking when the engine offers it, plain otherwise."""
        detailed = getattr(self.engine, "rank_spec_detailed", None)
        if detailed is not None:
            return detailed(spec, raw_query)
        candidates = self.engine.rank_spec(spec, raw_query)
        return DispatchDetail(candidates=candidates, rankings={}, weights={})


class SearchRequest(BaseModel):
    query: str
    session_id: str | None = None
    top_k: int | None = Field(default=None, gt=0)


class FeedbackRequest(BaseModel):
    query: str
    positives: list[str] = Field(default_factory=list)
    negatives: list[str] = Field(default_factory=list)
    session_id: str | None = None
    top_k: int | None = Field(default=None, gt=0)


class SimilarRequest(BaseModel):
    shot_key: str
    top_k: int | None = Field(default=None, gt=0)


class RevealRequest(BaseModel):
    text: str


class QaRequest(BaseModel):
    question: str
    top_k: int | None = Field(default=None, gt=0)


class SubmitRequest(BaseModel):
    video_id: str
    timestamp_ms: int = Field(ge=0)
    session_id: str | None = None
    result_version: int | None = None


class EscalateRequest(BaseModel):
    name: str
    query: str
    session_id: str | None = None
    top_k: int | None = Field(default=None, gt=0)
    result_version: int | None = None


def _result_view(state: ServiceState, candidate: ShotCandidate) -> dict | None:
    bundle = state.bundles.get(candidate.shot_key)
    if bundle is not None:
        timestamp = (
            candidate.timestamp_ms
            if candidate.timestamp_ms is not None
            else bundle.midpoint_ms
        )
        return {
            "shot_key": bundle.shot_key,
            "video_id": bundle.video_id,
            "shot_id": bundle.shot_id,
            "t_start_ms": bundle.t_start_ms,
            "t_end_ms": bundle.t_end_ms,
            "timestamp_ms": timestamp,
            "score": candidate.score,
            "caption": bundle.caption,
            "scene": bundle.scene,
            "actions": bundle.actions,
            "ocr_lines": bundle.ocr_lines,
            "asr_text": bundle.asr_text,
            "frames": [f"/frames/{name}" for name in bundle.frames],
        }
    meta = state.shot_meta.get(candidate.shot_key)
    if meta is None:
        logger.warning("candidate %s has no bundle or metadata", candidate.shot_key)
        return None
    timestamp = (
        candidate.timestamp_ms
        if candidate.timestamp_ms is not None
        else meta.midpoint_ms
    )
    return {
        "shot_key": candidate.shot_key,
        "video_id": meta.video_id,
        "shot_id": meta.shot_id,
        "t_start_ms": meta.t_start_ms,
        "t_end_ms": meta.t_end_ms,
        "timestamp_ms": timestamp,
        "score": candidate.score,
        "caption": None,
        "scene": None,
        "actions": [],
        "ocr_lines": [],
        "asr_text": None,
        "frames": [],
    }


def _video_id_of(state: ServiceState, shot_key: str) -> str:
    bundle = state.bundles.get(shot_key)
    if bundle is not None:
        return bundle.video_id
    meta = state.shot_meta.get(shot_key)
    return meta.video_id if meta is not None else ""


def _run_track_b(
    state: ServiceState,
    plan,
    question: str,
    candidates: list[ShotCandidate],
    cards: list[dict],
) -> list[dict]:
    """Read the top-M candidates, vote across moments, guard with disconfirm.

    Returns the ranked answer groups as plain dicts. Delegates to the shared
    :func:`aic.vqa.answer.track_b_groups` so the harness measures the same
    derivation; the reader degrades a failed candidate to an abstention, so
    this never raises for endpoint trouble — a partial outage narrows the vote.
    """
    track_a_answers = [
        card["answer_guess"] for card in cards if card.get("answer_guess")
    ]
    groups = track_b_groups(
        state.qa_reader,
        state.vqa_cfg,
        plan,
        question,
        candidates,
        state.bundles,
        state.shot_meta,
        lambda shot_key: _video_id_of(state, shot_key),
        track_a_answers,
        video_index=state.video_shot_index,
    )
    return [group.model_dump() for group in groups]


def _qa_rank(state: ServiceState, plan, question: str) -> list[ShotCandidate]:
    """Locate a question's moment: the KIS dispatch plus the factoid probe.

    Uses the engine's ``rank_qa_spec`` (which adds the Answer Ledger factoid
    channel at ``vqa.factoid_weight``) when the engine offers it; plain
    ``rank_spec`` otherwise (test fakes, factoid weight 0). KIS searches never
    come through here, so their ranking is untouched.
    """
    vqa = state.vqa_cfg
    rank_qa = getattr(state.engine, "rank_qa_spec", None)
    if rank_qa is not None and vqa is not None and vqa.factoid_weight > 0:
        return rank_qa(plan.locator, question, vqa.factoid_weight)
    return state.engine.rank_spec(plan.locator, question)


def _run_count(state: ServiceState, plan, candidate: ShotCandidate) -> dict | None:
    """Detector-first count on the top located moment; None to degrade."""
    pack = build_evidence_pack(
        candidate,
        state.bundles,
        state.shot_meta,
        state.vqa_cfg,
        state.video_shot_index,
    )
    result = state.count_program.count(
        plan, pack, _video_id_of(state, candidate.shot_key)
    )
    if result is None:
        return None
    return {
        "concept": result.concept,
        "count": result.count,
        "source": result.source,
        "video_id": result.video_id,
        "timestamp_ms": result.timestamp_ms,
        "provenance": result.provenance(),
    }


def _run_reverse(
    state: ServiceState, plan, candidates: list[ShotCandidate]
) -> list[dict]:
    """Answer-first reverse retrieval on a weak-locator question (phase 5).

    Hypotheses seed from the ledger entities of the top located shots (grounded,
    no LLM call — risk 5.1); each is searched in the literal/factoid channels via
    the same engine, and only double-gated winners (concentrated AND matching the
    locator ranking) become cards. Additive; empty when nothing survives.
    """
    vqa = state.vqa_cfg
    locator_scores = {c.shot_key: c.score for c in candidates}
    rank_qa = getattr(state.engine, "rank_qa_spec", None)

    def searcher(text: str) -> list[tuple[str, float]]:
        # The hypothesis's surface form probes the literal AND (when built +
        # weighted) the factoid index — the two indexes that hold verbatim
        # answer strings — through the same engine as every other rank.
        spec = QuerySpec(ocr_literals=[text], asr_phrases=[text])
        if rank_qa is not None and vqa.factoid_weight > 0:
            hits = rank_qa(spec, text, vqa.factoid_weight)
        else:
            hits = state.engine.rank_spec(spec, text)
        return [(h.shot_key, h.score) for h in hits]

    seeds: list[str] = []
    for candidate in candidates[: vqa.top_m]:
        row = state.ledger.get(candidate.shot_key)
        if row is not None:
            seeds.extend(getattr(row, "spoken_entities", []))
    seeds = list(dict.fromkeys(seeds))

    program = ReverseProgram(
        hypothesiser=lambda _q: [],
        searcher=searcher,
        locator_scorer=lambda shot_key: locator_scores.get(shot_key, 0.0),
        cfg=vqa.reverse,
    )
    cards = program.run(plan.question_core, seed_answers=seeds or None)
    return [
        {
            "answer": card.answer,
            "shot_key": card.shot_key,
            "video_id": _video_id_of(state, card.shot_key),
            "score": card.score,
            "provenance": card.provenance(),
        }
        for card in cards
    ]


def _ev_features(response: dict, pending: int = 0, submits_used: int = 0) -> dict:
    """Answer-state features for the EV hint (phase 5).

    ``pending`` is the number of Track B reads not yet voted: 0 in the one-shot
    route (reads are synchronous, so nothing is pending once the response is
    built) and >0 for the preliminary hint of the streaming route, where the
    Track A payload ships before the reads land. ``submits_used`` is the
    per-task submit_now budget already spent (see :func:`_task_ev_hint`)."""
    groups = response.get("answer_groups") or []
    winner = groups[0] if groups else {}
    return {
        "agreeing_count": len(winner.get("members", [])),
        "cross_track_agree": bool(winner.get("valid"))
        and any(c.get("answer_guess") for c in response.get("cards", [])),
        "track_b_pending": pending,
        "submits_used": submits_used,
    }


def _task_ev_hint(
    state: ServiceState, question: str, response: dict, pending: int = 0
) -> str:
    """The EV hint with its per-task submit budget applied (note 18 §3.6).

    A 'task' is one question: a new question resets the budget, re-asking the
    same one (retries, the streaming final hint after the preliminary) keeps
    spending it, so max_early_submits caps how often the hint says submit_now
    for a single answer hunt (risk 5.5). Only a FINAL hint (nothing pending)
    spends budget — the streaming preliminary is advisory.
    """
    task = state.qa_task_state
    key = question.strip().casefold()
    if task.get("question") != key:
        task["question"] = key
        task["submits_used"] = 0
    features = _ev_features(response, pending, task.get("submits_used", 0))
    hint = ev_hint(features, state.vqa_cfg.ev_hint)
    if hint == "submit_now" and pending == 0:
        task["submits_used"] = task.get("submits_used", 0) + 1
    return hint


def _count_should_fire(state: ServiceState, plan, candidates: list) -> bool:
    """Whether the detector-first count program applies to this question.

    Shared by the one-shot and streaming QA routes so both fire the count on
    exactly the same condition (a 'count' archetype with an available counter
    and at least one located moment)."""
    program = state.count_program
    return (
        plan.archetype == "count"
        and program is not None
        and getattr(program, "available", False)
        and bool(candidates)
    )


def _reverse_should_fire(
    state: ServiceState, cards: list[dict], candidates: list
) -> bool:
    """Whether answer-first reverse retrieval applies (a weak locator, phase 5).

    Shared by the one-shot and streaming QA routes."""
    vqa = state.vqa_cfg
    if vqa is None or not vqa.reverse.enabled:
        return False
    top_scores = [c.score for c in candidates[: vqa.top_m]]
    has_answer = any(c.get("answer_guess") for c in cards)
    return is_weak_locator(top_scores, has_answer, vqa.reverse)


def _ndjson(event: str, payload: dict) -> bytes:
    """One newline-delimited JSON event for a streaming response.

    The client dispatches on ``event`` and re-renders that slice, so events may
    arrive in any order and an absent stage simply never emits its event."""
    return (json.dumps({"event": event, **payload}, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def _qa_card(
    state: ServiceState, candidate: ShotCandidate, archetype: str
) -> dict | None:
    """One answer card: a located moment, its ledger provenance, and — when
    the archetype maps directly to a ledger field — a first-guess answer."""
    bundle = state.bundles.get(candidate.shot_key)
    meta = state.shot_meta.get(candidate.shot_key)
    if bundle is not None:
        video_id, mid = bundle.video_id, bundle.midpoint_ms
        frames = [f"/frames/{name}" for name in bundle.frames]
    elif meta is not None:
        video_id, mid = meta.video_id, meta.midpoint_ms
        frames = []
    else:
        logger.warning("qa candidate %s has no bundle or metadata", candidate.shot_key)
        return None
    timestamp = candidate.timestamp_ms if candidate.timestamp_ms is not None else mid

    answer_guess, provenance = ledger_answer(
        state.ledger.get(candidate.shot_key), archetype
    )
    return {
        "shot_key": candidate.shot_key,
        "video_id": video_id,
        "timestamp_ms": timestamp,
        "score": candidate.score,
        "answer_guess": answer_guess,
        "provenance": provenance,
        "frames": frames,
    }


async def _qa_event_stream(
    state: ServiceState, question: str, top_k: int
) -> AsyncIterator[bytes]:
    """NDJSON event stream behind /api/qa/stream (note 18 §3, streaming).

    The immediately-usable Track A cards ship first; then reverse retrieval,
    the count program, and Track B stream in as each finishes, followed by the
    settled EV hint. Every stage delegates to the same helper the one-shot
    /api/qa route uses (run off the event loop via ``run_in_threadpool``), so
    the streamed answer is byte-for-byte what /api/qa would return — only the
    delivery is incremental. Stages emit in a fixed order; a stage that does
    not fire simply never emits its event, so the client keys on ``event``.
    """
    timings: dict[str, float] = {}
    with _timed(timings, "compile"):
        plan = await run_in_threadpool(
            state.compiler.compile_qa, question, state.vqa_archetypes
        )
    with _timed(timings, "rank"):
        candidates = await run_in_threadpool(_qa_rank, state, plan, question)
    if state.negation_filter:
        candidates = filter_negations(
            candidates, plan.locator.negations, state.evidence_text()
        )
    cards: list[dict] = []
    for candidate in candidates[:top_k]:
        card = _qa_card(state, candidate, plan.archetype)
        if card is not None:
            cards.append(card)
    # Immediate payload: Track A cards — usable the moment they land.
    yield _ndjson(
        "track_a",
        {
            "plan": {
                "archetype": plan.archetype,
                "expected_answer_type": plan.expected_answer_type,
                "question_core": plan.question_core,
            },
            "cards": cards,
        },
    )
    vqa = state.vqa_cfg
    reader_on = state.qa_reader is not None and vqa is not None
    # Preliminary EV hint: the reads are still pending, so this reflects a
    # 'wait for reads' state until Track B lands (track_b_pending > 0).
    if vqa is not None and vqa.ev_hint.enabled:
        pending = min(top_k, len(candidates)) if reader_on else 0
        hint = _task_ev_hint(state, question, {"cards": cards}, pending)
        yield _ndjson("hint", {"phase": "preliminary", "ev_hint": hint})
    # Reverse retrieval (cheap, local) first when it fires (phase 5).
    if _reverse_should_fire(state, cards, candidates):
        with _timed(timings, "reverse"):
            reverse_cards = await run_in_threadpool(
                _run_reverse, state, plan, candidates
            )
        if reverse_cards:
            yield _ndjson("reverse", {"reverse_answers": reverse_cards})
    # Count program (detector-first, GPU) next when it fires (phase 4).
    if _count_should_fire(state, plan, candidates):
        with _timed(timings, "count"):
            answer = await run_in_threadpool(_run_count, state, plan, candidates[0])
        if answer is not None:
            yield _ndjson("count", {"count_answer": answer})
    # Track B (slowest — per-candidate VLM reads voted into groups).
    answer_groups: list[dict] = []
    if reader_on:
        with _timed(timings, "read"):
            answer_groups = await run_in_threadpool(
                _run_track_b, state, plan, question, candidates, cards
            )
        yield _ndjson("reads", {"answer_groups": answer_groups})
    # Settled EV hint: reads are in now (pending 0), so the signal is final.
    if vqa is not None and vqa.ev_hint.enabled:
        hint = _task_ev_hint(
            state, question, {"cards": cards, "answer_groups": answer_groups}, 0
        )
        yield _ndjson("hint", {"phase": "final", "ev_hint": hint})
    yield _ndjson("done", {"timings_ms": timings})


def _respond(
    state: ServiceState,
    cfg: ServiceConfig,
    candidates: list[ShotCandidate],
    spec: QuerySpec | None,
    session_id: str | None,
    top_k: int,
) -> dict:
    """Shared tail: narrowing + versioning + views + planner suggestion."""
    version = None
    session_view = None
    if session_id is not None:
        keys = [c.shot_key for c in candidates]
        kept_keys, version = state.sessions.narrow_and_record(
            session_id, keys, cfg.monotone_narrowing
        )
        kept = set(kept_keys)
        candidates = [c for c in candidates if c.shot_key in kept]
        session_view = state.sessions.view(session_id)
    results = []
    for candidate in candidates[:top_k]:
        view = _result_view(state, candidate)
        if view is not None:
            results.append(view)
    top_bundles = [
        state.bundles[r["shot_key"]]
        for r in results
        if r["shot_key"] in state.bundles
    ]
    return {
        "results": results,
        "spec": spec.model_dump() if spec is not None else None,
        "result_version": version,
        "planner": suggest_question(top_bundles),
        "session": session_view,
    }


def _auto_rerank_or_degrade(
    state: ServiceState, query: str, spec: QuerySpec, candidates: list[ShotCandidate]
) -> list[ShotCandidate]:
    """Run the automatic listwise re-rank, degrading to the fused order on any
    failure or timeout (note 15). Shared by /api/search and its stream so the
    re-ranked result is derived identically in both."""
    try:
        outcome = run_with_timeout(
            state.auto_rerank,
            EscalationRequest(query=query, spec=spec, candidates=candidates),
            state.auto_rerank_timeout_s,
        )
        if outcome.note:
            logger.info("auto rerank note: %s", outcome.note)
        return outcome.candidates
    except Exception as exc:  # noqa: BLE001 - degrade on ANY failure: search
        # must never break on its optional stage.
        logger.warning("auto rerank skipped: %s", exc)
        return candidates


async def _search_event_stream(
    state: ServiceState,
    cfg: ServiceConfig,
    query: str,
    session_id: str | None,
    top_k: int,
) -> AsyncIterator[bytes]:
    """NDJSON event stream behind /api/search/stream (note 15, streaming).

    For a stateless search the fused ranking — already guaranteed never worse
    than the reranked one — ships immediately as a read-only 'results_preview'
    (no session narrowing, no result_version), so the operator sees candidates
    at fusion latency instead of waiting out the auto-rerank wall. The
    authoritative 'results' event is then built by the same _respond tail as
    /api/search, so narrowing and versioning happen exactly once and submit
    semantics are unchanged. When a KIS-C session is active the preview is
    omitted: narrowing/versioning must stay authoritative and single, so a
    session search streams only the one authoritative result.
    """
    timings: dict[str, float] = {}
    with _timed(timings, "compile"):
        spec = await run_in_threadpool(state.compiler.compile, query)
    with _timed(timings, "rank"):
        detail = await run_in_threadpool(state.rank_detailed, spec, query)
    candidates = detail.candidates
    if state.negation_filter:
        with _timed(timings, "filter"):
            candidates = filter_negations(
                candidates, spec.negations, state.evidence_text()
            )
    # Immediate read-only preview of the fused ranking (stateless search only).
    if session_id is None:
        preview = []
        for candidate in candidates[:top_k]:
            view = _result_view(state, candidate)
            if view is not None:
                preview.append(view)
        yield _ndjson(
            "results_preview", {"results": preview, "spec": spec.model_dump()}
        )
    # Auto-rerank (the slow stage) then the QPP advisor, exactly as /api/search.
    if state.auto_rerank is not None and candidates:
        with _timed(timings, "auto_rerank"):
            candidates = await run_in_threadpool(
                _auto_rerank_or_degrade, state, query, spec, candidates
            )
    advisor = None
    if state.qpp is not None:
        with _timed(timings, "advise"):
            advisor = await run_in_threadpool(
                state.qpp.advise,
                spec,
                candidates,
                detail.rankings,
                state.video_by_shot(),
            )
    with _timed(timings, "respond"):
        response = await run_in_threadpool(
            _respond, state, cfg, candidates, spec, session_id, top_k
        )
    response["advisor"] = advisor
    response["timings_ms"] = timings
    yield _ndjson("results", response)
    yield _ndjson("done", {"timings_ms": timings})


def create_app(state: ServiceState, cfg: ServiceConfig) -> FastAPI:
    app = FastAPI(title="AIC 2026 retrieval service")
    evidence = state.evidence_text()
    video_by_shot = state.video_by_shot()

    @app.exception_handler(SessionError)
    async def _unknown_session(request, exc: SessionError):
        raise HTTPException(status_code=404, detail="unknown session") from exc

    @app.middleware("http")
    async def _time_requests(request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"
        if request.url.path.startswith("/api/"):
            logger.info(
                "%s %s -> %s in %.1f ms",
                request.method,
                request.url.path,
                response.status_code,
                elapsed_ms,
            )
        return response

    def _joined_query(query: str, session_id: str | None) -> str:
        """The raw query plus every revealed KIS-C constraint."""
        if session_id is None:
            return query
        constraints = state.sessions.view(session_id)["constraints"]
        return "\n".join([query, *constraints]).strip()

    @app.post("/api/session")
    def create_session() -> dict:
        return state.sessions.create()

    @app.get("/api/session/{session_id}")
    def get_session(session_id: str) -> dict:
        return state.sessions.view(session_id)

    @app.post("/api/session/{session_id}/reveal")
    def reveal(session_id: str, request: RevealRequest) -> dict:
        return state.sessions.reveal(session_id, request.text)

    @app.post("/api/search")
    def search(request: SearchRequest) -> dict:
        timings: dict[str, float] = {}
        query = _joined_query(request.query, request.session_id)
        with _timed(timings, "compile"):
            spec = state.compiler.compile(query)
        with _timed(timings, "rank"):
            detail = state.rank_detailed(spec, query)
        candidates = detail.candidates
        if state.negation_filter:
            with _timed(timings, "filter"):
                candidates = filter_negations(candidates, spec.negations, evidence)
        if state.auto_rerank is not None and candidates:
            # Automatic listwise rerank of the fused top (note 15); any
            # failure or timeout keeps the fused order — the search reply
            # must never be worse or later than the rerank-less one by more
            # than its hard timeout.
            with _timed(timings, "auto_rerank"):
                candidates = _auto_rerank_or_degrade(state, query, spec, candidates)
        advisor = None
        if state.qpp is not None:
            with _timed(timings, "advise"):
                advisor = state.qpp.advise(
                    spec, candidates, detail.rankings, video_by_shot
                )
        with _timed(timings, "respond"):
            response = _respond(
                state,
                cfg,
                candidates,
                spec,
                request.session_id,
                request.top_k or cfg.default_top_k,
            )
        response["advisor"] = advisor
        response["timings_ms"] = timings
        return response

    if cfg.streaming:

        @app.post("/api/search/stream")
        def search_stream(request: SearchRequest) -> StreamingResponse:
            """Streaming twin of /api/search (note 15): the fused ranking ships
            immediately, the auto-rerank streams in as the final result.
            Additive — registered only when service.streaming."""
            query = _joined_query(request.query, request.session_id)
            top_k = request.top_k or cfg.default_top_k
            return StreamingResponse(
                _search_event_stream(
                    state, cfg, query, request.session_id, top_k
                ),
                media_type="application/x-ndjson",
            )

    @app.get("/api/escalations")
    def list_escalations() -> dict:
        return {
            "escalations": [
                {
                    "name": name,
                    "timeout_s": state.escalation_timeouts.get(name),
                }
                for name in sorted(state.escalations)
            ]
        }

    @app.post("/api/escalate")
    def escalate(request: EscalateRequest) -> dict:
        escalation = state.escalations.get(request.name)
        if escalation is None:
            raise HTTPException(
                status_code=404,
                detail=f"unknown or disabled escalation {request.name!r}",
            )
        if request.session_id is not None:
            current = state.sessions.current_version(request.session_id)
            if request.result_version != current:
                # Same stale protection as submit: escalating a result set
                # the operator is no longer looking at wastes the tool.
                raise HTTPException(
                    status_code=409,
                    detail=f"stale result set: version {request.result_version} "
                    f"escalated but session is at {current}",
                )
        timings: dict[str, float] = {}
        query = _joined_query(request.query, request.session_id)
        with _timed(timings, "compile"):
            spec = state.compiler.compile(query)
        with _timed(timings, "rank"):
            candidates = state.engine.rank_spec(spec, query)
        if state.negation_filter:
            candidates = filter_negations(candidates, spec.negations, evidence)
        timeout_s = state.escalation_timeouts.get(request.name)
        if timeout_s is None:
            raise HTTPException(
                status_code=500,
                detail=f"escalation {request.name!r} has no configured timeout",
            )
        with _timed(timings, "escalate"):
            try:
                outcome = run_with_timeout(
                    escalation,
                    EscalationRequest(query=query, spec=spec, candidates=candidates),
                    timeout_s,
                )
            except EscalationTimeout as exc:
                raise HTTPException(status_code=504, detail=str(exc)) from exc
        response = _respond(
            state,
            cfg,
            outcome.candidates,
            spec,
            request.session_id,
            request.top_k or cfg.default_top_k,
        )
        response["escalation"] = {"name": request.name, "note": outcome.note}
        response["timings_ms"] = timings
        return response

    @app.post("/api/feedback")
    def feedback(request: FeedbackRequest) -> dict:
        top_k = request.top_k or cfg.default_top_k
        candidates = state.engine.rank_feedback(
            request.query, request.positives, request.negatives, top_k
        )
        return _respond(
            state, cfg, candidates, None, request.session_id, top_k
        )

    @app.post("/api/similar")
    def similar(request: SimilarRequest) -> dict:
        top_k = request.top_k or cfg.default_top_k
        candidates = state.engine.rank_similar(request.shot_key, top_k)
        return _respond(state, cfg, candidates, None, None, top_k)

    if state.vqa_enabled:

        @app.post("/api/qa")
        def qa(request: QaRequest) -> dict:
            """VQA Track A (note 18 §3.2): compile the question, locate the
            moment through the existing retriever, and join each top moment
            with its ledger row into an answer card. No VLM call (phase 1)."""
            timings: dict[str, float] = {}
            with _timed(timings, "compile"):
                plan = state.compiler.compile_qa(
                    request.question, state.vqa_archetypes
                )
            with _timed(timings, "rank"):
                candidates = _qa_rank(state, plan, request.question)
            if state.negation_filter:
                candidates = filter_negations(
                    candidates, plan.locator.negations, evidence
                )
            top_k = request.top_k or cfg.default_top_k
            cards = []
            for candidate in candidates[:top_k]:
                card = _qa_card(state, candidate, plan.archetype)
                if card is not None:
                    cards.append(card)
            response = {
                "plan": {
                    "archetype": plan.archetype,
                    "expected_answer_type": plan.expected_answer_type,
                    "question_core": plan.question_core,
                },
                "cards": cards,
                "timings_ms": timings,
            }
            # Track B (note 18 §3.4): per-candidate grounded reads voted into
            # ranked answer groups. Additive — absent when the reader is off,
            # so Track A output is unchanged.
            if state.qa_reader is not None and state.vqa_cfg is not None:
                with _timed(timings, "read"):
                    response["answer_groups"] = _run_track_b(
                        state, plan, request.question, candidates, cards
                    )
            # Count program (note 18 §3, phase 4): a detector-first count for a
            # 'count' question, on the top located moment. Additive and
            # degrade-safe — absent when no counter is configured/available.
            if _count_should_fire(state, plan, candidates):
                with _timed(timings, "count"):
                    answer = _run_count(state, plan, candidates[0])
                if answer is not None:
                    response["count_answer"] = answer
            # Reverse retrieval (phase 5): fire only on a weak locator, behind
            # its flag. Additive; the seed hypothesiser is grounded in ledger
            # entities so it costs no LLM call.
            if _reverse_should_fire(state, cards, candidates):
                with _timed(timings, "reverse"):
                    reverse_cards = _run_reverse(state, plan, candidates)
                if reverse_cards:
                    response["reverse_answers"] = reverse_cards
            # EV submit hint (phase 5): advisory only, from the answer state,
            # with the per-task submit budget applied.
            vqa = state.vqa_cfg
            if vqa is not None and vqa.ev_hint.enabled:
                response["ev_hint"] = _task_ev_hint(
                    state, request.question, response
                )
            return response

        if cfg.streaming:

            @app.post("/api/qa/stream")
            def qa_stream(request: QaRequest) -> StreamingResponse:
                """Streaming twin of /api/qa (note 18): the same answer, but
                Track A ships first and each slower stage streams in as it
                finishes. Additive — registered only when service.streaming."""
                top_k = request.top_k or cfg.default_top_k
                return StreamingResponse(
                    _qa_event_stream(state, request.question, top_k),
                    media_type="application/x-ndjson",
                )

    @app.post("/api/submit")
    def submit(request: SubmitRequest) -> dict:
        if request.session_id is not None:
            current = state.sessions.current_version(request.session_id)
            if request.result_version != current:
                # Stale-submit protection: the operator is looking at an
                # older result set than the session's latest (note 09).
                raise HTTPException(
                    status_code=409,
                    detail=f"stale result set: version {request.result_version} "
                    f"submitted but session is at {current}",
                )
        return {
            "payload": serialize_submission(
                request.video_id, request.timestamp_ms, cfg.submission_format
            ),
            "format": cfg.submission_format,
        }

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(
            _STATIC_DIR / "index.html",
            headers={"Cache-Control": "no-store"},
        )

    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")
    if state.frames_dir is not None and state.frames_dir.is_dir():
        app.mount(
            "/frames", StaticFiles(directory=state.frames_dir), name="frames"
        )
    return app
