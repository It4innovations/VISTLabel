# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""The Help section in a real browser: reached from the Projects header,
navigated through its sidebar, and themed like the rest of the app."""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.browser


def test_help_is_reachable_from_projects_and_its_outline_jumps(
    page: Page, live_server_url: str
) -> None:
    errors: list[str] = []
    page.on(
        "console",
        lambda message: errors.append(message.text) if message.type == "error" else None,
    )
    page.goto(f"{live_server_url}/")

    page.click("#open-help-link")
    expect(page).to_have_url(f"{live_server_url}/help")
    expect(page.locator("#shortcut-table")).to_contain_text("Mark the current frame reviewed")

    page.click("#help-index >> text=User guide")
    expect(page).to_have_url(f"{live_server_url}/help/user-guide")
    expect(page.locator("#help-index li.active")).to_contain_text("User guide")

    page.click("#help-toc >> text=Export")
    expect(page).to_have_url(f"{live_server_url}/help/user-guide#export")
    expect(page.locator("h2#export")).to_be_in_viewport()

    # The theme toggle is wired up on the Help page too.
    page.click("#theme-toggle")
    expect(page.locator("html")).to_have_attribute("data-theme", "light")
    assert errors == []
