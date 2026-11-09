# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from vistlabel.auth.account import AccountStore
from vistlabel.auth.passwords import WeakPasswordError, hash_password, verify_password
from vistlabel.auth.sessions import LoginRateLimiter, SessionStore
from vistlabel.settings import load_settings


def test_password_hash_is_argon2id_and_verifies() -> None:
    digest = hash_password("correct-horse-battery")
    assert digest.startswith("$argon2id$")
    assert verify_password(digest, "correct-horse-battery")
    assert not verify_password(digest, "wrong")
    assert not verify_password("garbage", "x")


@pytest.mark.parametrize("password", ["short", " leading-space", "trailing-space "])
def test_weak_passwords_are_rejected(password: str) -> None:
    with pytest.raises(WeakPasswordError):
        hash_password(password)


def test_account_store_never_persists_plaintext(tmp_path: Path) -> None:
    store = AccountStore(tmp_path)
    assert not store.is_configured()
    store.set_password("alice", "correct-horse-battery")
    raw = store.path.read_text(encoding="utf-8")
    assert "correct-horse-battery" not in raw
    assert json.loads(raw)["algorithm"] == "argon2id"
    assert store.path == tmp_path / "auth" / "account.json"
    assert store.verify("alice", "correct-horse-battery")
    assert not store.verify("alice", "nope")
    assert not store.verify("bob", "correct-horse-battery")


def test_changing_the_password_keeps_the_creation_time(tmp_path: Path) -> None:
    store = AccountStore(tmp_path)
    first = store.set_password("alice", "correct-horse-battery")
    second = store.set_password("alice", "another-long-password")
    assert second.created_at == first.created_at
    assert store.verify("alice", "another-long-password")
    assert not store.verify("alice", "correct-horse-battery")


def test_account_store_rejects_bad_username(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        AccountStore(tmp_path).set_password("bad name!", "correct-horse-battery")


def test_sessions_persist_and_expire(tmp_path: Path) -> None:
    store = SessionStore(tmp_path, timeout_minutes=30)
    token, _session = store.create("alice")
    assert token not in (tmp_path / "auth" / "sessions.json").read_text(encoding="utf-8")
    assert store.resolve(token) is not None
    assert store.resolve("unknown") is None

    reloaded = SessionStore(tmp_path, timeout_minutes=30)
    assert reloaded.resolve(token) is not None

    expired = SessionStore(tmp_path, timeout_minutes=30)
    record = expired.resolve(token)
    assert record is not None
    record.expires_at = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    assert expired.resolve(token) is None
    assert expired.active_count() == 0


def test_session_revocation(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    token, _ = store.create("alice")
    assert store.revoke(token)
    assert store.resolve(token) is None
    store.create("alice")
    assert store.revoke_all() == 1


def test_rate_limiter_blocks_after_limit() -> None:
    limiter = LoginRateLimiter(limit=3, window_seconds=60)
    for _ in range(3):
        assert not limiter.is_blocked("1.2.3.4")
        limiter.record_failure("1.2.3.4")
    assert limiter.is_blocked("1.2.3.4")
    assert limiter.retry_after_seconds("1.2.3.4") > 0
    assert not limiter.is_blocked("5.6.7.8")
    limiter.reset("1.2.3.4")
    assert not limiter.is_blocked("1.2.3.4")


def test_old_shared_api_key_section_fails_loudly(tmp_path: Path) -> None:
    config = tmp_path / "app.yaml"
    config.write_text("auth:\n  enabled: true\n  api_keys: [s3cr3t]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="vistlabel auth set-password"):
        load_settings(config)
