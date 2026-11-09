# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from vistlabel.persistence.locks import LockConflictError, LockNotHeldError, ProjectLock


def test_acquire_by_a_second_owner_raises_conflict(tmp_path: Path) -> None:
    lock = ProjectLock(tmp_path / "editor.lock.json")
    lock.acquire("session-a")

    with pytest.raises(LockConflictError):
        lock.acquire("session-b")


def test_reacquiring_with_the_same_owner_is_idempotent(tmp_path: Path) -> None:
    lock = ProjectLock(tmp_path / "editor.lock.json")
    lock.acquire("session-a")

    lock.acquire("session-a")  # must not raise


def test_stale_lock_can_be_taken_over(tmp_path: Path) -> None:
    lock = ProjectLock(tmp_path / "editor.lock.json", stale_after=timedelta(seconds=0))
    lock.acquire("session-a")

    # stale_after=0 means the lock is immediately considered stale.
    lock.acquire("session-b")

    holder = lock.read()
    assert holder is not None
    assert holder.owner_id == "session-b"


def test_heartbeat_refreshes_and_prevents_takeover(tmp_path: Path) -> None:
    lock = ProjectLock(tmp_path / "editor.lock.json", stale_after=timedelta(minutes=5))
    lock.acquire("session-a")

    lock.heartbeat("session-a")

    with pytest.raises(LockConflictError):
        lock.acquire("session-b")


def test_heartbeat_by_non_holder_raises(tmp_path: Path) -> None:
    lock = ProjectLock(tmp_path / "editor.lock.json")

    with pytest.raises(LockNotHeldError):
        lock.heartbeat("nobody")


def test_release_by_non_holder_raises(tmp_path: Path) -> None:
    lock = ProjectLock(tmp_path / "editor.lock.json")
    lock.acquire("session-a")

    with pytest.raises(LockNotHeldError):
        lock.release("session-b")


def test_release_by_holder_allows_new_acquire(tmp_path: Path) -> None:
    lock = ProjectLock(tmp_path / "editor.lock.json")
    lock.acquire("session-a")
    lock.release("session-a")

    lock.acquire("session-b")  # must not raise

    assert lock.is_held_by_other("session-a") is True
    assert lock.is_held_by_other("session-b") is False


def test_zero_stale_after_is_stale_within_the_same_clock_tick(tmp_path: Path) -> None:
    """Regression: with a strict ">" comparison, two acquire() calls landing
    in the same clock tick left age == 0, which did not exceed stale_after=0,
    so the takeover raised LockConflictError instead. Repeated because the
    failure only shows when both timestamps happen to be identical."""
    for attempt in range(200):
        lock = ProjectLock(tmp_path / f"editor-{attempt}.lock.json", stale_after=timedelta(0))
        lock.acquire("session-a")

        lock.acquire("session-b")

        holder = lock.read()
        assert holder is not None
        assert holder.owner_id == "session-b"


def test_a_lock_younger_than_stale_after_is_still_held(tmp_path: Path) -> None:
    """The looser comparison must not make live locks stealable."""
    lock = ProjectLock(tmp_path / "editor.lock.json", stale_after=timedelta(minutes=5))
    lock.acquire("session-a")

    with pytest.raises(LockConflictError):
        lock.acquire("session-b")
