# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Playwright scaffolding smoke test.

This exercises the Playwright wiring itself -- a real browser context
issuing an HTTP request against a live VISTLabel server -- rather than a page
interaction. Marked ``browser`` so it can be isolated behind
``playwright install``.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

from tests.conftest import TEST_PASSWORD, TEST_USERNAME

pytestmark = pytest.mark.browser


def test_health_endpoint_is_reachable_through_a_real_browser_context(
    page: Page, live_server_url: str
) -> None:
    response = page.request.get(f"{live_server_url}/api/v1/health")

    assert response.ok
    assert response.json()["status"] in ("ok", "degraded", "error")


@pytest.mark.signed_out
def test_sign_in_and_sign_out_in_the_browser(page: Page, live_server_url: str) -> None:
    page.goto(f"{live_server_url}/help")
    page.wait_for_url(f"{live_server_url}/login?next=%2Fhelp")

    page.fill("#username", TEST_USERNAME)
    page.fill("#password", "not-the-password")
    page.click("#login-button")
    expect(page.locator("#login-error")).to_have_text("invalid username or password")

    page.fill("#password", TEST_PASSWORD)
    page.click("#login-button")
    page.wait_for_url(f"{live_server_url}/help")

    page.goto(f"{live_server_url}/")
    expect(page.locator("#current-user")).to_have_text(TEST_USERNAME)
    page.click("#logout-button")
    page.wait_for_url(f"{live_server_url}/login")
    assert page.request.get(f"{live_server_url}/api/v1/projects").status == 401
