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
from playwright.sync_api import Page

pytestmark = pytest.mark.browser


def test_health_endpoint_is_reachable_through_a_real_browser_context(
    page: Page, live_server_url: str
) -> None:
    response = page.request.get(f"{live_server_url}/api/v1/health")

    assert response.ok
    assert response.json()["status"] in ("ok", "degraded", "error")
