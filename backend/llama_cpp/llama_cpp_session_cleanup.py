from __future__ import annotations

import atexit
import logging
from threading import RLock
from typing import Protocol


class _ClosableSession(Protocol):
    def close(self) -> None: ...


_logger = logging.getLogger(__name__)
_sessions: set[_ClosableSession] = set()
_sessions_lock = RLock()


def track_session(session: _ClosableSession) -> None:
    with _sessions_lock:
        _sessions.add(session)


def untrack_session(session: _ClosableSession) -> None:
    with _sessions_lock:
        _sessions.discard(session)


def close_tracked_sessions() -> None:
    with _sessions_lock:
        sessions = tuple(_sessions)
        _sessions.clear()
    for session in sessions:
        try:
            session.close()
        except Exception:
            _logger.exception("Could not close a Llama.cpp session")


atexit.register(close_tracked_sessions)


__all__ = [
    "close_tracked_sessions",
    "track_session",
    "untrack_session",
]
