# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Browser sign-in: session cookie, page redirects, CSRF and rate limiting."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.conftest import (
    CSRF_HEADERS,
    TEST_PASSWORD,
    TEST_USERNAME,
    login,
    write_test_account,
)
from vistlabel.auth.account import AccountStore
from vistlabel.auth.sessions import SESSION_COOKIE_NAME
from vistlabel.main import cli, create_app
from vistlabel.settings import AppSettings


def test_unauthenticated_api_is_rejected_and_pages_redirect(anonymous_client: TestClient) -> None:
    response = anonymous_client.get("/api/v1/projects")
    assert response.status_code == 401
    assert response.json()["code"] == "AUTH_REQUIRED"

    page = anonymous_client.get("/", follow_redirects=False)
    assert page.status_code == 303
    assert page.headers["location"] == "/login?next=%2F"

    workspace = anonymous_client.get("/projects/abc?frame=3", follow_redirects=False)
    assert workspace.status_code == 303
    assert workspace.headers["location"] == "/login?next=%2Fprojects%2Fabc%3Fframe%3D3"

    assert anonymous_client.get("/help", follow_redirects=False).status_code == 303

    # Public endpoints stay reachable.
    assert anonymous_client.get("/api/v1/health").status_code == 200
    assert anonymous_client.get("/api/v1/version").status_code == 200
    assert anonymous_client.get("/static/css/app.css").status_code == 200
    assert anonymous_client.get("/login").status_code == 200
    assert anonymous_client.get("/api/v1/auth/status").json()["authenticated"] is False


def test_login_sets_httponly_cookie_and_logout_clears_it(anonymous_client: TestClient) -> None:
    response = anonymous_client.post(
        "/api/v1/auth/login", json={"username": TEST_USERNAME, "password": TEST_PASSWORD}
    )
    assert response.status_code == 200
    cookie = response.headers["set-cookie"]
    assert SESSION_COOKIE_NAME in cookie
    assert "HttpOnly" in cookie and "SameSite=lax" in cookie
    assert "Secure" not in cookie
    assert anonymous_client.get("/api/v1/auth/status").json()["username"] == TEST_USERNAME
    assert anonymous_client.get("/api/v1/projects").status_code == 200

    index = anonymous_client.get("/")
    assert index.status_code == 200
    assert 'id="logout-button"' in index.text and TEST_USERNAME in index.text

    # Signed in, /login goes straight on to the requested page.
    again = anonymous_client.get("/login?next=/help", follow_redirects=False)
    assert again.status_code == 303 and again.headers["location"] == "/help"

    anonymous_client.post("/api/v1/auth/logout")
    assert anonymous_client.get("/api/v1/projects").status_code == 401


def test_login_page_never_redirects_off_site(anonymous_client: TestClient) -> None:
    page = anonymous_client.get("/login?next=//evil.example/x")
    assert page.status_code == 200
    assert 'data-next="/"' in page.text


def test_secure_cookie_behind_https(app_settings: AppSettings) -> None:
    settings = app_settings.model_copy(
        update={"security": app_settings.security.model_copy(update={"secure_cookies": True})}
    )
    write_test_account(settings.workspace.root)
    client = TestClient(create_app(settings))
    response = client.post(
        "/api/v1/auth/login",
        json={"username": TEST_USERNAME, "password": TEST_PASSWORD},
        headers=CSRF_HEADERS,
    )
    assert response.status_code == 200
    assert "Secure" in response.headers["set-cookie"]


def test_invalid_credentials_and_rate_limit(anonymous_client: TestClient) -> None:
    anonymous_client.app.state.login_limiter._limit = 3  # type: ignore[attr-defined]
    for _ in range(3):
        response = anonymous_client.post(
            "/api/v1/auth/login", json={"username": TEST_USERNAME, "password": "wrong"}
        )
        assert response.status_code == 401
        assert response.json()["detail"] == "invalid username or password"
    blocked = anonymous_client.post(
        "/api/v1/auth/login", json={"username": TEST_USERNAME, "password": TEST_PASSWORD}
    )
    assert blocked.status_code == 429
    assert int(blocked.headers["retry-after"]) > 0


def test_csrf_header_is_required_for_state_changes(anonymous_client: TestClient) -> None:
    response = anonymous_client.post(
        "/api/v1/auth/login",
        json={"username": TEST_USERNAME, "password": TEST_PASSWORD},
        headers={"X-Requested-With": ""},
    )
    assert response.status_code == 403
    assert response.json()["code"] == "CSRF_REJECTED"

    login(anonymous_client)
    cross_site = anonymous_client.patch(
        "/api/v1/projects/whatever",
        json={},
        headers={**CSRF_HEADERS, "Origin": "https://evil.example"},
    )
    assert cross_site.status_code == 403
    assert cross_site.json()["code"] == "CSRF_REJECTED"


def test_no_account_configured_redirects_to_setup(app_settings: AppSettings) -> None:
    client = TestClient(create_app(app_settings))
    page = client.get("/", follow_redirects=False)
    assert page.status_code == 303 and page.headers["location"] == "/setup"
    setup = client.get("/setup")
    assert setup.status_code == 200
    assert "vistlabel auth set-password" in setup.text
    assert client.get("/login", follow_redirects=False).headers["location"] == "/setup"
    response = client.post(
        "/api/v1/auth/login", json={"username": "x", "password": "y"}, headers=CSRF_HEADERS
    )
    assert response.status_code == 503
    assert client.get("/api/v1/auth/status").json()["account_configured"] is False


def test_session_survives_process_restart(
    app_settings: AppSettings, configured_account: AccountStore
) -> None:
    first = TestClient(create_app(app_settings))
    login(first)
    cookie = first.cookies.get(SESSION_COOKIE_NAME)
    assert cookie
    second = TestClient(create_app(app_settings))
    second.cookies.set(SESSION_COOKIE_NAME, cookie)
    assert second.get("/api/v1/auth/status").json()["authenticated"] is True


def test_password_never_appears_in_logs(
    anonymous_client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    anonymous_client.post(
        "/api/v1/auth/login", json={"username": TEST_USERNAME, "password": "wrong-password-1"}
    )
    login(anonymous_client)
    assert "wrong-password-1" not in caplog.text
    assert TEST_PASSWORD not in caplog.text


def test_cli_set_password_status_and_revoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "app.yaml"
    workspace = tmp_path / "workspace"
    config.write_text(f"workspace:\n  root: {workspace.as_posix()}\n", encoding="utf-8")

    assert cli(["auth", "status", "--config", str(config)]) == 1

    monkeypatch.setenv("VISTLABEL_PASSWORD", "short")
    assert cli(["auth", "set-password", "--username", "lab", "--config", str(config)]) == 2

    monkeypatch.setenv("VISTLABEL_PASSWORD", "a-long-enough-password")
    assert cli(["auth", "set-password", "--username", "lab", "--config", str(config)]) == 0
    assert AccountStore(workspace).verify("lab", "a-long-enough-password")

    assert cli(["auth", "status", "--config", str(config)]) == 0
    assert "username: lab" in capsys.readouterr().out
    assert cli(["auth", "revoke-sessions", "--config", str(config)]) == 0

    monkeypatch.delenv("VISTLABEL_PASSWORD")
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert cli(["auth", "set-password", "--config", str(config)]) == 2


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    from vistlabel import __version__

    assert cli(["version"]) == 0
    assert capsys.readouterr().out.strip() == __version__
