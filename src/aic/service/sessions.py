"""KIS-C session state: constraint stack, versioned result sets, narrowing.

Server-side and in-memory: a browser refresh resumes the session by id, and
every mutation happens under one lock so concurrent operators cannot corrupt
a stack. Result sets are versioned so a submit always references the result
set the operator actually saw (stale-submit protection, note 09 Phase 8).
"""

from __future__ import annotations

import itertools
import threading
import uuid
from dataclasses import dataclass, field


@dataclass
class Session:
    session_id: str
    constraints: list[str] = field(default_factory=list)
    version: int = 0
    """Version of the most recent recorded result set; 0 = none yet."""
    last_candidates: frozenset[str] | None = None
    """Shot keys of the previous full candidate set, for narrowing."""
    last_access: int = 0

    def view(self) -> dict:
        return {
            "session_id": self.session_id,
            "constraints": list(self.constraints),
            "version": self.version,
        }


class SessionError(KeyError):
    """Raised for unknown session ids."""


class SessionStore:
    """Thread-safe in-memory sessions with LRU eviction beyond the cap."""

    def __init__(self, max_sessions: int, max_constraints: int) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self._counter = itertools.count(1)
        self._max_sessions = max_sessions
        self._max_constraints = max_constraints

    def create(self) -> dict:
        with self._lock:
            session = Session(session_id=uuid.uuid4().hex)
            session.last_access = next(self._counter)
            self._sessions[session.session_id] = session
            if len(self._sessions) > self._max_sessions:
                oldest = min(
                    self._sessions.values(), key=lambda s: s.last_access
                )
                del self._sessions[oldest.session_id]
            return session.view()

    def _get(self, session_id: str) -> Session:
        session = self._sessions.get(session_id)
        if session is None:
            raise SessionError(session_id)
        session.last_access = next(self._counter)
        return session

    def view(self, session_id: str) -> dict:
        with self._lock:
            return self._get(session_id).view()

    def reveal(self, session_id: str, text: str) -> dict:
        """Push a revealed constraint; the stack only ever grows in meaning.

        Beyond the cap the two oldest constraints merge into one, so very
        long KIS-C rounds keep every detail without unbounded growth
        (note 09 Phase 6 edge case).
        """
        with self._lock:
            session = self._get(session_id)
            cleaned = text.strip()
            if cleaned:
                session.constraints.append(cleaned)
                while len(session.constraints) > self._max_constraints:
                    merged = "; ".join(session.constraints[:2])
                    session.constraints[:2] = [merged]
            return session.view()

    def narrow_and_record(
        self, session_id: str, candidate_keys: list[str], monotone: bool
    ) -> tuple[list[str], int]:
        """Apply monotone narrowing and record the new result-set version.

        Returns the (possibly filtered) candidate keys in their given order
        and the new version. When narrowing would empty the set, the
        unfiltered candidates are kept — the target may have sat outside the
        previous top-k.
        """
        with self._lock:
            session = self._get(session_id)
            kept = candidate_keys
            if monotone and session.last_candidates is not None:
                filtered = [
                    key for key in candidate_keys if key in session.last_candidates
                ]
                if filtered:
                    kept = filtered
            session.last_candidates = frozenset(kept)
            session.version += 1
            return kept, session.version

    def current_version(self, session_id: str) -> int:
        with self._lock:
            return self._get(session_id).version
