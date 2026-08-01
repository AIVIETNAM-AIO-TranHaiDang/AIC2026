"""The escalation contract: pure functions over the candidate pool.

Escalations are operator-triggered tools (the 2026-07 revision in note 09):
the service exposes each one as an endpoint plus a console button/hotkey,
and nothing fires automatically. Every escalation takes the current
candidate pool and returns a (usually re-ordered, possibly extended) pool;
internal failures degrade to the unchanged pool with an operator-facing
note, so a broken tool is never worse than not pressing the button.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from typing import Protocol

from aic.cortex.spec import QuerySpec
from aic.retrieval.fusion import ShotCandidate


class EscalationError(RuntimeError):
    """Raised when an escalation cannot run at all (bad wiring, bad name)."""


class EscalationTimeout(EscalationError):
    """Raised when an escalation exceeds its configured hard timeout."""


@dataclass(frozen=True)
class EscalationRequest:
    """Everything an escalation may look at."""

    query: str
    spec: QuerySpec
    candidates: list[ShotCandidate] = field(default_factory=list)


@dataclass(frozen=True)
class EscalationOutcome:
    """The re-ranked pool plus an optional operator-facing note.

    ``note`` reports degradations ("no usable renders", "endpoint failed")
    so the operator knows the button did nothing rather than wondering why
    the order looks unchanged.
    """

    candidates: list[ShotCandidate]
    note: str | None = None


class Escalation(Protocol):
    @property
    def name(self) -> str: ...

    def run(self, request: EscalationRequest) -> EscalationOutcome: ...


def run_with_timeout(
    escalation: Escalation,
    request: EscalationRequest,
    timeout_s: float,
) -> EscalationOutcome:
    """Run an escalation under its hard timeout (note 09 Phase 7).

    Python cannot kill a running thread: on timeout the worker is left to
    finish in the background and its result is discarded. The executor is
    per-call and shut down without waiting, so timed-out work never queues
    behind fresh requests.
    """
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(escalation.run, request)
        try:
            return future.result(timeout=timeout_s)
        except FuturesTimeoutError:
            raise EscalationTimeout(
                f"escalation {escalation.name!r} exceeded {timeout_s:.0f}s"
            ) from None
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
