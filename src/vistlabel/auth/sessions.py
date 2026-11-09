# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Server-side sessions and login rate limiting.

The browser only ever holds an opaque random token in an ``HttpOnly``
cookie; the server stores a SHA-256 of the token with the session record.
Sessions are persisted atomically to ``workspace/auth/sessions.json`` so a
server restart does not sign the user out.
"""

from __future__ import annotations

import contextlib
import hashlib
import secrets
import threading
import time
from collections import deque
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, Field

from vistlabel.persistence import paths
from vistlabel.persistence.atomic import load_model, save_model

SESSION_COOKIE_NAME = "vistlabel_session"

#: Persist a session's sliding expiry at most this often; every API call
#: refreshes it in memory, and the frame-by-frame UI makes many.
_TOUCH_PERSIST_INTERVAL_SECONDS = 60


def _utc_now() -> datetime:
    return datetime.now(UTC)


class Session(BaseModel):
    session_id: str
    token_hash: str
    username: str
    created_at: str
    last_seen_at: str
    expires_at: str


class SessionFile(BaseModel):
    schema_version: int = 1
    sessions: list[Session] = Field(default_factory=list)


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class SessionStore:
    def __init__(
        self, workspace_root: Path, *, timeout_minutes: int = 120, persist: bool = True
    ) -> None:
        self._path = paths.sessions_file(workspace_root)
        self._timeout = timedelta(minutes=timeout_minutes)
        self._persist = persist
        self._lock = threading.Lock()
        self._sessions: dict[str, Session] = {}
        self._last_persisted_touch: dict[str, float] = {}
        self._load()

    @property
    def timeout(self) -> timedelta:
        return self._timeout

    def _load(self) -> None:
        if not self._persist or not self._path.exists():
            return
        try:
            data = load_model(self._path, SessionFile)
        except (OSError, ValueError):
            return
        now = _utc_now()
        for session in data.sessions:
            if datetime.fromisoformat(session.expires_at) > now:
                self._sessions[session.token_hash] = session

    def _save(self) -> None:
        if not self._persist:
            return
        save_model(self._path, SessionFile(sessions=list(self._sessions.values())), fsync=False)
        with contextlib.suppress(OSError):  # Windows ignores POSIX modes
            self._path.chmod(0o600)

    def create(self, username: str) -> tuple[str, Session]:
        token = secrets.token_urlsafe(32)
        now = _utc_now()
        session = Session(
            session_id=secrets.token_hex(6),
            token_hash=_hash_token(token),
            username=username,
            created_at=now.isoformat(timespec="milliseconds"),
            last_seen_at=now.isoformat(timespec="milliseconds"),
            expires_at=(now + self._timeout).isoformat(timespec="milliseconds"),
        )
        with self._lock:
            self._sessions[session.token_hash] = session
            self._save()
        return token, session

    def resolve(self, token: str | None) -> Session | None:
        if not token:
            return None
        token_hash = _hash_token(token)
        with self._lock:
            session = self._sessions.get(token_hash)
            if session is None:
                return None
            now = _utc_now()
            if datetime.fromisoformat(session.expires_at) <= now:
                del self._sessions[token_hash]
                self._save()
                return None
            # Sliding inactivity timeout.
            session.last_seen_at = now.isoformat(timespec="milliseconds")
            session.expires_at = (now + self._timeout).isoformat(timespec="milliseconds")
            last = self._last_persisted_touch.get(token_hash, 0.0)
            if time.monotonic() - last > _TOUCH_PERSIST_INTERVAL_SECONDS:
                self._last_persisted_touch[token_hash] = time.monotonic()
                self._save()
            return session

    def revoke(self, token: str | None) -> bool:
        if not token:
            return False
        with self._lock:
            removed = self._sessions.pop(_hash_token(token), None) is not None
            if removed:
                self._save()
            return removed

    def revoke_all(self) -> int:
        with self._lock:
            count = len(self._sessions)
            self._sessions.clear()
            self._save()
            return count

    def active_count(self) -> int:
        with self._lock:
            return len(self._sessions)


class LoginRateLimiter:
    """Sliding-window failed-attempt counter keyed by client address."""

    def __init__(self, limit: int = 10, window_seconds: int = 300) -> None:
        self._limit = limit
        self._window = window_seconds
        self._lock = threading.Lock()
        self._failures: dict[str, deque[float]] = {}

    def _prune(self, key: str, now: float) -> deque[float]:
        attempts = self._failures.setdefault(key, deque())
        while attempts and now - attempts[0] > self._window:
            attempts.popleft()
        return attempts

    def is_blocked(self, key: str) -> bool:
        with self._lock:
            return len(self._prune(key, time.monotonic())) >= self._limit

    def retry_after_seconds(self, key: str) -> int:
        with self._lock:
            attempts = self._prune(key, time.monotonic())
            if len(attempts) < self._limit:
                return 0
            return max(1, int(self._window - (time.monotonic() - attempts[0])))

    def record_failure(self, key: str) -> None:
        with self._lock:
            self._prune(key, time.monotonic()).append(time.monotonic())

    def reset(self, key: str) -> None:
        with self._lock:
            self._failures.pop(key, None)
