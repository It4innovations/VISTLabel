# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
import uvicorn
from playwright.sync_api import Page

from tests.conftest import (
    CSRF_HEADERS,
    TEST_PASSWORD,
    TEST_USERNAME,
    login,
    write_test_account,
)
from vistlabel.main import create_app
from vistlabel.settings import AppSettings


@pytest.fixture
def live_server_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    settings = AppSettings(workspace={"root": tmp_path / "workspace"})
    write_test_account(settings.workspace.root)
    app = create_app(settings)
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    else:
        raise RuntimeError("uvicorn server did not start in time")

    port = server.servers[0].sockets[0].getsockname()[1]
    base_url = f"http://127.0.0.1:{port}"
    # The tests read and seed state with module-level httpx calls next to
    # the browser; route them through one signed-in client, so they don't
    # each need to carry the session cookie and the CSRF header.
    api = httpx.Client(base_url=base_url, headers=CSRF_HEADERS, timeout=30)
    login(api)
    for method in ("get", "post", "put", "patch", "delete"):
        monkeypatch.setattr(httpx, method, getattr(api, method))
    try:
        yield base_url
    finally:
        api.close()
        server.should_exit = True
        thread.join(timeout=5)


def sign_in(page: Page, base_url: str) -> None:
    """Sign in through the real login form, as an operator would."""
    page.goto(f"{base_url}/login")
    page.fill("#username", TEST_USERNAME)
    page.fill("#password", TEST_PASSWORD)
    page.click("#login-button")
    page.wait_for_url(f"{base_url}/")


@pytest.fixture(autouse=True)
def _signed_in(request: pytest.FixtureRequest) -> None:
    """Every browser test starts signed in, unless it is marked
    ``signed_out`` (the tests of the sign-in flow itself)."""
    if "live_server_url" not in request.fixturenames or "page" not in request.fixturenames:
        return
    if request.node.get_closest_marker("signed_out"):
        return
    sign_in(request.getfixturevalue("page"), request.getfixturevalue("live_server_url"))
