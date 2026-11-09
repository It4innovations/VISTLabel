# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Project editing lock with heartbeat and stale-lock recovery.

The first VISTLabel release supports one editing session per project. This is
a logical, JSON-file-based advisory lock (not an OS-level `flock`): the
holder is expected to call :meth:`ProjectLock.heartbeat` periodically while
editing, and any acquire attempt from a different owner treats a lock whose
heartbeat is older than `stale_after` as abandoned and takes it over. This
is deliberately not a database -- it is a small file next to the project it
protects, reconstructable and inspectable after a crash.

This is not the only protection against concurrent writes: every mutation
also goes through the optimistic-revision check in `projects.store`, so
correctness never relies on the lock alone.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import BaseModel

from vistlabel.persistence.atomic import load_model, save_model

DEFAULT_STALE_AFTER = timedelta(minutes=2)


class LockInfo(BaseModel):
    owner_id: str
    acquired_at: datetime
    last_heartbeat: datetime


class LockConflictError(Exception):
    """Raised by :meth:`ProjectLock.acquire` when another, non-stale owner
    already holds the lock."""

    def __init__(self, lock_path: Path, holder: LockInfo) -> None:
        self.lock_path = lock_path
        self.holder = holder
        super().__init__(
            f"{lock_path}: held by {holder.owner_id!r} "
            f"(last heartbeat {holder.last_heartbeat.isoformat()})"
        )


class LockNotHeldError(Exception):
    """Raised by :meth:`ProjectLock.heartbeat`/:meth:`release` when the
    caller is not (or is no longer) the current lock holder."""

    def __init__(self, lock_path: Path, owner_id: str) -> None:
        self.lock_path = lock_path
        self.owner_id = owner_id
        super().__init__(f"{lock_path}: {owner_id!r} does not currently hold this lock")


class ProjectLock:
    def __init__(self, lock_path: Path, stale_after: timedelta = DEFAULT_STALE_AFTER) -> None:
        self._lock_path = lock_path
        self._stale_after = stale_after

    def read(self) -> LockInfo | None:
        if not self._lock_path.exists():
            return None
        return load_model(self._lock_path, LockInfo)

    def _is_stale(self, info: LockInfo) -> bool:
        # ">=", not ">": an age of exactly stale_after must already count as
        # stale, or stale_after=0 ("immediately stale") would never fire when
        # both timestamps land in the same clock tick -- Windows' default
        # datetime resolution is ~15 ms, which is easily wider than two
        # consecutive acquire() calls.
        now = datetime.now(UTC)
        return now - info.last_heartbeat >= self._stale_after

    def acquire(self, owner_id: str) -> LockInfo:
        existing = self.read()
        if existing is not None and existing.owner_id != owner_id and not self._is_stale(existing):
            raise LockConflictError(self._lock_path, existing)
        now = datetime.now(UTC)
        info = LockInfo(owner_id=owner_id, acquired_at=now, last_heartbeat=now)
        save_model(self._lock_path, info)
        return info

    def heartbeat(self, owner_id: str) -> LockInfo:
        existing = self.read()
        if existing is None or existing.owner_id != owner_id:
            raise LockNotHeldError(self._lock_path, owner_id)
        updated = existing.model_copy(update={"last_heartbeat": datetime.now(UTC)})
        save_model(self._lock_path, updated)
        return updated

    def release(self, owner_id: str) -> None:
        existing = self.read()
        if existing is None:
            return
        if existing.owner_id != owner_id:
            raise LockNotHeldError(self._lock_path, owner_id)
        self._lock_path.unlink(missing_ok=True)

    def is_held_by_other(self, owner_id: str) -> bool:
        existing = self.read()
        if existing is None:
            return False
        return existing.owner_id != owner_id and not self._is_stale(existing)
