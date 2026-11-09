# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Browser-driven frame playback: Play/Pause button and Space, stopping at
the end, and handing over to the user on manual navigation or editing."""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

from tests.conftest import create_project_via_requests
from vistlabel.video.ffmpeg import FfmpegBinaries

pytestmark = pytest.mark.browser


def _open(page: Page, live_server_url: str, name: str) -> None:
    project_id = create_project_via_requests(live_server_url, name)
    page.goto(f"{live_server_url}/projects/{project_id}")
    expect(page.locator("#status-line")).to_contain_text("Ready", timeout=10_000)


def _frame(page: Page) -> int:
    return int(page.eval_on_selector("#frame-slider", "e => e.value"))


def test_frame_controls_are_not_covered_by_the_header_at_1280px(
    page: Page, live_server_url: str, ffmpeg_binaries: FfmpegBinaries
) -> None:
    page.set_viewport_size({"width": 1280, "height": 800})
    _open(page, live_server_url, "playback layout test")
    overlap = page.evaluate(
        """() => {
          const nav = document.getElementById('frame-nav').getBoundingClientRect();
          const last = document.getElementById('playback-speed').getBoundingClientRect();
          const next = document.querySelector('.lv-backend-picker').getBoundingClientRect();
          return { navOverflow: last.right - nav.right, intoBackend: last.right - next.left };
        }"""
    )
    # Half a pixel of slack: the speed select ends flush with the nav, and
    # Firefox's 1/60 px layout units leave float noise (~1e-5) on that edge.
    assert overlap["navOverflow"] < 0.5, overlap
    assert overlap["intoBackend"] < 0.5, overlap


def test_play_runs_to_the_last_frame_and_stops(
    page: Page, live_server_url: str, ffmpeg_binaries: FfmpegBinaries
) -> None:
    _open(page, live_server_url, "playback end test")  # 10 frames at 10 FPS
    page.select_option("#playback-speed", "2")
    page.click("#play-button")
    expect(page.locator("#play-button")).to_have_attribute("aria-pressed", "true")

    expect(page.locator("#status-line")).to_contain_text("End of video", timeout=10_000)
    expect(page.locator("#frame-slider")).to_have_value("9")
    expect(page.locator("#play-button")).to_have_attribute("aria-pressed", "false")

    # Play from the last frame starts over from frame 0.
    page.keyboard.press("Space")
    expect(page.locator("#status-line")).to_contain_text("End of video", timeout=10_000)


def test_space_toggles_playback_even_after_typing_a_frame_number(
    page: Page, live_server_url: str, ffmpeg_binaries: FfmpegBinaries
) -> None:
    _open(page, live_server_url, "playback space test")
    page.fill("#frame-number-input", "2")
    page.locator("#frame-number-input").press("Enter")
    expect(page.locator("#frame-slider")).to_have_value("2", timeout=10_000)
    page.select_option("#playback-speed", "0.25")  # 2.5 FPS: easy to stop midway

    # Focus is still in the number box -- Space must play, not be swallowed.
    page.keyboard.press("Space")
    expect(page.locator("#play-button")).to_have_attribute("aria-pressed", "true")
    page.wait_for_function("Number(document.getElementById('frame-slider').value) >= 3")
    page.keyboard.press("Space")
    expect(page.locator("#play-button")).to_have_attribute("aria-pressed", "false")
    expect(page.locator("#status-line")).to_contain_text("Paused at frame")

    stopped_at = _frame(page)
    page.wait_for_timeout(1_000)
    assert _frame(page) == stopped_at
    assert stopped_at < 9


def test_manual_navigation_and_editing_stop_playback(
    page: Page, live_server_url: str, ffmpeg_binaries: FfmpegBinaries
) -> None:
    _open(page, live_server_url, "playback handover test")
    page.select_option("#playback-speed", "0.25")

    page.click("#play-button")
    expect(page.locator("#play-button")).to_have_attribute("aria-pressed", "true")
    page.keyboard.press("ArrowLeft")
    expect(page.locator("#play-button")).to_have_attribute("aria-pressed", "false")

    page.click("#play-button")
    expect(page.locator("#play-button")).to_have_attribute("aria-pressed", "true")
    page.locator("#frame-canvas").click(position={"x": 20, "y": 20})
    expect(page.locator("#play-button")).to_have_attribute("aria-pressed", "false")
    stopped_at = _frame(page)
    page.wait_for_timeout(1_000)
    assert _frame(page) == stopped_at
