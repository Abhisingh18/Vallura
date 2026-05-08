"""
In-memory session memory store.

Design decision (documented in README):
  In-memory storage is the right choice for this prototype because:
  1. Zero infrastructure dependency — no DB provisioning needed
  2. Sub-microsecond read/write latency
  3. Sufficient for a single-process demo / assignment
  4. The interface is stable — swapping to Redis/Postgres later requires
     only a new implementation of the same add_message / get_history API.

Trade-off: data is lost on restart. Acceptable for a prototype.
"""
from __future__ import annotations

import threading
from collections import defaultdict


class SessionMemory:
    """
    Thread-safe in-memory conversation store.

    Key: session_id (str)
    Value: ordered list of message dicts [{"role": "user"|"assistant", "content": "..."}]
    """

    def __init__(self, max_turns_per_session: int = 50) -> None:
        self._store: dict[str, list[dict[str, str]]] = defaultdict(list)
        self._max_turns = max_turns_per_session
        self._lock = threading.Lock()

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Append a message to the session history."""
        with self._lock:
            history = self._store[session_id]
            history.append({"role": role, "content": content})
            # Trim to prevent unbounded growth
            if len(history) > self._max_turns:
                self._store[session_id] = history[-self._max_turns:]

    def get_history(self, session_id: str) -> list[dict[str, str]]:
        """Return a copy of the session history (safe for concurrent reads)."""
        with self._lock:
            return list(self._store.get(session_id, []))

    def clear_session(self, session_id: str) -> None:
        """Remove all messages for a session."""
        with self._lock:
            self._store.pop(session_id, None)

    def session_exists(self, session_id: str) -> bool:
        """Check if a session has any messages."""
        with self._lock:
            return session_id in self._store and len(self._store[session_id]) > 0
