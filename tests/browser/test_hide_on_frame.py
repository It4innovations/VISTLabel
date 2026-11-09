# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Browser-driven per-frame actions: Deactivate/Activate and Hide/Show an
object on one frame, the R shortcut, the review queue's Solved button, and
Delete/Restore frame."""

from __future__ import annotations

import httpx
import pytest
from playwright.sync_api import Page, expect

from tests.conftest import create_project_via_requests
from vistlabel.video.ffmpeg import FfmpegBinaries

pytestmark = pytest.mark.browser


def _frame(live_server_url: str, project_id: str, frame_index: int = 0) -> dict:
    return httpx.get(
        f"{live_server_url}/api/v1/projects/{project_id}/frames/{frame_index}/annotations",
        timeout=10,
    ).json()


def _open_with_one_object(page: Page, live_server_url: str, project_id: str) -> None:
    page.goto(f"{live_server_url}/projects/{project_id}")
    expect(page.locator("#status-line")).to_contain_text("Ready", timeout=10_000)
    page.fill("#new-object-label", "pipette")
    page.click("#add-object-button")
    expect(page.locator("#object-list li")).to_have_count(1)


def _accept_mask(page: Page) -> None:
    page.locator("#frame-canvas").click(position={"x": 20, "y": 20})
    expect(page.locator("#accept-candidate-button")).to_be_enabled(timeout=10_000)
    page.click("#accept-candidate-button")
    expect(page.locator("#status-line")).to_contain_text("Saved", timeout=10_000)


_CANVAS_PIXELS = """() => {
  const canvas = document.getElementById("frame-canvas");
  return Array.from(
    canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data
  );
}"""


def test_deactivate_then_activate_restores_the_mask_on_this_frame(
    page: Page, live_server_url: str, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_via_requests(live_server_url, "browser deactivate test")
    _open_with_one_object(page, live_server_url, project_id)
    item = page.locator("#object-list li")
    # Nothing to deactivate or hide until the object has a mask on this frame.
    expect(item.locator(".object-activation")).to_have_count(0)
    expect(item.locator(".object-visibility")).to_have_count(0)

    _accept_mask(page)
    saved_mask = _frame(live_server_url, project_id)["objects"][0]["visible_mask_rle"]

    item.get_by_role("button", name="Deactivate", exact=True).click()
    expect(page.locator("#status-line")).to_contain_text("Deactivated on frame 0", timeout=10_000)
    expect(item).to_contain_text("[inactive]")
    expect(item).to_have_class("selected inactive")
    assert _frame(live_server_url, project_id)["objects"] == []

    item.get_by_role("button", name="Activate", exact=True).click()
    expect(page.locator("#status-line")).to_contain_text("Activated on frame 0", timeout=10_000)
    expect(item).to_contain_text("[visible]")
    frame = _frame(live_server_url, project_id)
    assert frame["inactive_objects"] == []
    assert frame["objects"][0]["visible_mask_rle"] == saved_mask

    # D toggles the selected object the same way, both ways.
    page.keyboard.press("d")
    expect(item).to_contain_text("[inactive]", timeout=10_000)
    page.keyboard.press("d")
    expect(item).to_contain_text("[visible]", timeout=10_000)


def test_hide_is_view_only(
    page: Page, live_server_url: str, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_via_requests(live_server_url, "browser hide test")
    _open_with_one_object(page, live_server_url, project_id)
    item = page.locator("#object-list li")
    _accept_mask(page)
    before = _frame(live_server_url, project_id)
    shown_pixels = page.evaluate(_CANVAS_PIXELS)

    item.get_by_role("button", name="Hide", exact=True).click()
    expect(page.locator("#status-line")).to_contain_text("Hidden on frame 0 (view only)")
    expect(item).to_contain_text("[visible, hidden]")
    hidden_pixels = page.evaluate(_CANVAS_PIXELS)
    assert sum(a != b for a, b in zip(shown_pixels, hidden_pixels, strict=True)) > 0
    # Nothing saved: the mask is still there for the export.
    assert _frame(live_server_url, project_id) == before

    # Only on this frame.
    page.keyboard.press("ArrowRight")
    expect(page.locator("#frame-slider")).to_have_value("1", timeout=10_000)
    page.keyboard.press("ArrowLeft")
    expect(page.locator("#frame-slider")).to_have_value("0", timeout=10_000)
    expect(item).to_contain_text("[visible, hidden]")

    page.keyboard.press("h")
    expect(item).to_contain_text("[visible]")
    expect(item).not_to_contain_text("hidden")
    assert page.evaluate(_CANVAS_PIXELS) == shown_pixels


def test_r_marks_reviewed_and_solved_clears_a_queue_entry(
    page: Page, live_server_url: str, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_via_requests(live_server_url, "browser review test")
    for frame_index in (1, 2):
        response = httpx.patch(
            f"{live_server_url}/api/v1/projects/{project_id}/frames/{frame_index}/annotations",
            json={"expected_revision": 0, "review_status": "needs_review", "review_comment": "x"},
            timeout=10,
        )
        assert response.status_code == 200, response.text
    _open_with_one_object(page, live_server_url, project_id)

    page.keyboard.press("r")
    expect(page.locator("#status-line")).to_contain_text("Frame 0 reviewed", timeout=10_000)
    assert _frame(live_server_url, project_id)["review_status"] == "reviewed"

    page.click("#tab-queue")
    entries = page.locator("#review-queue-list .lv-queue-item")
    expect(entries).to_have_count(2)
    entries.first.get_by_role("button", name="Solved").click()
    expect(entries).to_have_count(1)
    expect(entries.first).to_contain_text("Frame 2")
    expect(page.locator("#review-queue-count")).to_have_text("1")
    assert _frame(live_server_url, project_id, 1)["review_status"] == "reviewed"


def test_mark_all_annotated_reviewed_confirms_then_updates_the_open_frame(
    page: Page, live_server_url: str, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_via_requests(live_server_url, "browser mark all test")
    _open_with_one_object(page, live_server_url, project_id)
    _accept_mask(page)
    page.click("#tab-review")
    expect(page.locator("#review-status")).to_have_value("unreviewed")

    messages: list[str] = []

    def accept(dialog) -> None:
        messages.append(dialog.message)
        dialog.accept()

    page.on("dialog", accept)
    page.click("#mark-all-reviewed-button")
    expect(page.locator("#mark-all-reviewed-status")).to_contain_text(
        "Marked 1 frame(s) reviewed", timeout=10_000
    )
    assert messages and "Mark 1 annotated frame(s) as reviewed?" in messages[0]
    # The open frame was reloaded, so the editor shows the new status and
    # can keep saving without a revision conflict.
    expect(page.locator("#review-status")).to_have_value("reviewed")
    page.keyboard.press("r")
    expect(page.locator("#status-line")).to_contain_text("Frame 0 reviewed", timeout=10_000)


def test_delete_frame_toggles_exclusion(
    page: Page, live_server_url: str, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_via_requests(live_server_url, "browser delete frame test")
    _open_with_one_object(page, live_server_url, project_id)
    expect(page.locator("#frame-excluded-badge")).to_be_hidden()

    page.click("#delete-frame-button")
    expect(page.locator("#frame-excluded-badge")).to_be_visible(timeout=10_000)
    expect(page.locator("#delete-frame-button")).to_have_text("Restore frame")
    assert _frame(live_server_url, project_id)["excluded"] is True

    page.keyboard.press("ArrowRight")
    expect(page.locator("#frame-excluded-badge")).to_be_hidden(timeout=10_000)
    page.keyboard.press("ArrowLeft")
    expect(page.locator("#frame-excluded-badge")).to_be_visible(timeout=10_000)

    page.click("#delete-frame-button")
    expect(page.locator("#delete-frame-button")).to_have_text("Delete frame", timeout=10_000)
    assert _frame(live_server_url, project_id)["excluded"] is False
