# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Tracking controls, exercised through the real annotation page's
"Propagate" control."""

from __future__ import annotations

import httpx
import pytest
from playwright.sync_api import Page, expect

from tests.conftest import create_project_via_requests
from vistlabel.video.ffmpeg import FfmpegBinaries

pytestmark = pytest.mark.browser


def test_propagate_button_tracks_the_seeded_object_forward(
    page: Page, live_server_url: str, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_via_requests(live_server_url, "propagate ui test")

    page.goto(f"{live_server_url}/projects/{project_id}")
    expect(page.locator("#status-line")).to_contain_text("Ready", timeout=10_000)

    page.fill("#new-object-label", "pipette")
    page.click("#add-object-button")
    expect(page.locator("#object-list li")).to_have_count(1)

    page.locator("#frame-canvas").click(position={"x": 20, "y": 20})
    expect(page.locator("#accept-candidate-button")).to_be_enabled(timeout=10_000)
    page.click("#accept-candidate-button")
    expect(page.locator("#status-line")).to_contain_text("Saved", timeout=10_000)

    # tiny_test_video.mp4 has 10 frames; the end-frame field defaults to
    # the last one.
    expect(page.locator("#propagation-end-frame")).to_have_value("9")
    page.click("#propagate-button")

    expect(page.locator("#status-line")).to_contain_text(
        "Propagation complete", timeout=20_000
    )
    expect(page.locator("#propagate-button")).to_be_enabled()
    expect(page.locator("#cancel-propagation-button")).to_be_disabled()

    # Verify server-side, not just the UI's own optimistic status text: the
    # last frame really was tracked, not silently skipped.
    annotation = httpx.get(
        f"{live_server_url}/api/v1/projects/{project_id}/frames/9/annotations"
    ).json()
    assert len(annotation["objects"]) == 1


def test_propagate_covers_active_objects_and_skips_deactivated_ones(
    page: Page, live_server_url: str, ffmpeg_binaries: FfmpegBinaries
) -> None:
    """Run submits every object with a saved mask on the current frame in
    one job; deactivating an object on that frame takes its mask out of
    the frame, so the run doesn't propagate it."""
    project_id = create_project_via_requests(live_server_url, "propagate deactivated test")

    page.goto(f"{live_server_url}/projects/{project_id}")
    expect(page.locator("#status-line")).to_contain_text("Ready", timeout=10_000)

    # Two objects, each seeded and saved on frame 0. createObject selects
    # the new object, so each canvas click prompts for the just-created one.
    page.fill("#new-object-label", "glove")
    page.click("#add-object-button")
    expect(page.locator("#object-list li")).to_have_count(1)
    page.locator("#frame-canvas").click(position={"x": 15, "y": 15})
    expect(page.locator("#accept-candidate-button")).to_be_enabled(timeout=10_000)
    page.click("#accept-candidate-button")
    expect(page.locator("#status-line")).to_contain_text("Saved", timeout=10_000)

    page.fill("#new-object-label", "tube")
    page.click("#add-object-button")
    expect(page.locator("#object-list li")).to_have_count(2)
    page.locator("#frame-canvas").click(position={"x": 45, "y": 45})
    expect(page.locator("#accept-candidate-button")).to_be_enabled(timeout=10_000)
    page.click("#accept-candidate-button")
    expect(page.locator("#status-line")).to_contain_text("Saved", timeout=10_000)

    objects = httpx.get(f"{live_server_url}/api/v1/projects/{project_id}/objects").json()[
        "objects"
    ]
    object_id_by_label = {o["label_name"]: o["object_id"] for o in objects}

    glove_item = page.locator("#object-list li", has_text="glove")
    glove_item.get_by_role("button", name="Deactivate", exact=True).click()
    expect(glove_item.get_by_role("button", name="Activate", exact=True)).to_be_visible(
        timeout=10_000
    )

    page.click("#propagate-button")
    expect(page.locator("#status-line")).to_contain_text(
        "Propagation complete", timeout=20_000
    )

    annotation = httpx.get(
        f"{live_server_url}/api/v1/projects/{project_id}/frames/9/annotations"
    ).json()
    propagated_ids = {o["object_id"] for o in annotation["objects"]}
    assert object_id_by_label["tube"] in propagated_ids
    assert object_id_by_label["glove"] not in propagated_ids


def test_propagate_requires_an_object_with_a_saved_mask(
    page: Page, live_server_url: str, ffmpeg_binaries: FfmpegBinaries
) -> None:
    """Run covers every object with a saved mask on the current frame; with
    none of those, it refuses with a status message instead of submitting
    a job that would only fail server-side."""
    project_id = create_project_via_requests(live_server_url, "propagate no object test")

    page.goto(f"{live_server_url}/projects/{project_id}")
    expect(page.locator("#status-line")).to_contain_text("Ready", timeout=10_000)

    page.click("#propagate-button")
    expect(page.locator("#status-line")).to_contain_text(
        "No object has a saved mask on this frame", timeout=10_000
    )


# Records what the page asks the browser for instead of really prompting,
# playing audio or showing OS notifications; hasFocus() -> false stands in
# for the user having switched to another tab.
_NOTICE_STUBS = """
window.__notices = [];
window.__chimeNotes = 0;
class FakeNotification {
  constructor(title, options) { window.__notices.push({ title, body: options.body }); }
  close() {}
}
FakeNotification.permission = "granted";
FakeNotification.requestPermission = () => Promise.resolve("granted");
window.Notification = FakeNotification;
class FakeAudioContext {
  constructor() { this.state = "running"; this.currentTime = 0; this.destination = {}; }
  resume() { return Promise.resolve(); }
  createOscillator() {
    return {
      frequency: {}, connect: (node) => node,
      start: () => { window.__chimeNotes += 1; }, stop: () => {},
    };
  }
  createGain() {
    const param = { setValueAtTime() {}, exponentialRampToValueAtTime() {} };
    return { gain: param, connect: (node) => node };
  }
}
window.AudioContext = FakeAudioContext;
document.hasFocus = () => false;
"""


def test_finished_propagation_chimes_and_notifies_when_the_tab_is_unfocused(
    page: Page, live_server_url: str, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_via_requests(live_server_url, "propagate notify test")
    page.add_init_script(_NOTICE_STUBS)

    page.goto(f"{live_server_url}/projects/{project_id}")
    expect(page.locator("#status-line")).to_contain_text("Ready", timeout=10_000)
    expect(page.locator("#propagation-notify")).to_be_checked()

    page.fill("#new-object-label", "pipette")
    page.click("#add-object-button")
    expect(page.locator("#object-list li")).to_have_count(1)
    page.locator("#frame-canvas").click(position={"x": 20, "y": 20})
    expect(page.locator("#accept-candidate-button")).to_be_enabled(timeout=10_000)
    page.click("#accept-candidate-button")
    expect(page.locator("#status-line")).to_contain_text("Saved", timeout=10_000)

    page.click("#propagate-button")
    expect(page.locator("#status-line")).to_contain_text(
        "Propagation complete", timeout=20_000
    )

    notices = page.evaluate("window.__notices")
    assert notices == [
        {"title": "VISTLabel — propagate notify test", "body": "Propagation complete"}
    ]
    assert page.evaluate("window.__chimeNotes") == 3  # the success chime
    assert page.title().startswith("✓ ")

    # Unticking Notify silences the next run entirely.
    page.uncheck("#propagation-notify")
    page.click("#propagate-button")  # disables the button before returning
    expect(page.locator("#propagate-button")).to_be_enabled(timeout=20_000)
    assert len(page.evaluate("window.__notices")) == 1
    assert page.evaluate("window.__chimeNotes") == 3
