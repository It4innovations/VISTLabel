# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Browser-driven annotation workspace: point prompting,
accepting a candidate mask, brush editing, undo, and object management
through a real page -- not just the HTTP API.
"""

from __future__ import annotations

import re

import httpx
import pytest
from playwright.sync_api import Page, Response, expect

from tests.conftest import create_project_via_requests
from vistlabel.video.ffmpeg import FfmpegBinaries

pytestmark = pytest.mark.browser


def test_backend_selector_disables_unavailable_backends_with_a_reason(
    page: Page, live_server_url: str, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_via_requests(live_server_url, "backend selector test")

    page.goto(f"{live_server_url}/projects/{project_id}")
    expect(page.locator("#status-line")).to_contain_text("Ready", timeout=10_000)

    options = page.locator("#backend-select option")
    expect(options).to_have_count(4)  # fake, sam21, sam3, yolo_sam

    fake_option = page.locator('#backend-select option[value="fake"]')
    sam21_option = page.locator('#backend-select option[value="sam21"]')
    assert fake_option.get_attribute("disabled") is None
    assert sam21_option.get_attribute("disabled") is not None
    assert sam21_option.get_attribute("title")  # exact reason, not just "disabled"


def test_point_tools_enabled_for_sam3_tracker(
    page: Page, live_server_url: str, ffmpeg_binaries: FfmpegBinaries
) -> None:
    """SAM 3's tracker path accepts point and box prompts, so capability
    gating must leave the normal annotation tools usable."""
    project_id = create_project_via_requests(live_server_url, "capability gate test")

    current = httpx.get(f"{live_server_url}/api/v1/projects/{project_id}").json()
    patched = httpx.patch(
        f"{live_server_url}/api/v1/projects/{project_id}",
        json={"expected_revision": current["revision"], "selected_backend_id": "sam3"},
    )
    assert patched.status_code == 200, patched.text

    page.goto(f"{live_server_url}/projects/{project_id}")
    expect(page.locator("#status-line")).to_contain_text("Ready", timeout=10_000)

    positive = page.locator('[data-tool="point_positive"]')
    expect(positive).to_be_enabled()
    expect(page.locator('[data-tool="point_negative"]')).to_be_enabled()


def test_point_prompt_accept_and_persist_across_reload(
    page: Page, live_server_url: str, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_via_requests(live_server_url, "browser annotation test")

    page.goto(f"{live_server_url}/projects/{project_id}")
    expect(page.locator("#status-line")).to_contain_text("Ready", timeout=10_000)

    # Object names are free text -- the project ontology starts empty and
    # grows as names are used, so one must be typed before creating.
    page.fill("#new-object-label", "pipette")
    page.click("#add-object-button")
    expect(page.locator("#object-list li")).to_have_count(1)

    page.locator("#frame-canvas").click(position={"x": 20, "y": 20})
    expect(page.locator("#accept-candidate-button")).to_be_enabled(timeout=10_000)
    page.click("#accept-candidate-button")
    expect(page.locator("#status-line")).to_contain_text("Saved", timeout=10_000)

    # Reload the whole page -- the previously accepted mask must redraw
    # from the server (reopening a project restores its current masks), not
    # just remain visible because it was never cleared client-side.
    page.reload()
    expect(page.locator("#object-list li")).to_have_count(1, timeout=10_000)
    expect(page.locator("#status-line")).to_contain_text("Ready", timeout=10_000)


def test_brush_edit_autosaves_and_undo_reverts(
    page: Page, live_server_url: str, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_via_requests(live_server_url, "browser brush test")

    page.goto(f"{live_server_url}/projects/{project_id}")
    page.fill("#new-object-label", "pipette")
    page.click("#add-object-button")
    expect(page.locator("#object-list li")).to_have_count(1)

    page.click('[data-tool="brush"]')
    canvas = page.locator("#frame-canvas")
    box = canvas.bounding_box()
    assert box is not None
    start_x, start_y = box["x"] + 15, box["y"] + 15

    page.mouse.move(start_x, start_y)
    page.mouse.down()
    page.mouse.move(start_x + 10, start_y + 10)
    page.mouse.up()
    expect(page.locator("#status-line")).to_contain_text("Saved", timeout=10_000)
    expect(page.locator("#undo-button")).to_be_enabled()

    page.click("#undo-button")
    expect(page.locator("#status-line")).to_contain_text("Saved", timeout=10_000)


def test_right_drag_with_the_brush_erases(
    page: Page, live_server_url: str, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_via_requests(live_server_url, "browser right drag test")

    page.goto(f"{live_server_url}/projects/{project_id}")
    expect(page.locator("#status-line")).to_contain_text("Ready", timeout=10_000)
    page.fill("#new-object-label", "pipette")
    page.click("#add-object-button")
    expect(page.locator("#object-list li")).to_have_count(1)

    page.click('[data-tool="brush"]')
    box = page.locator("#frame-canvas").bounding_box()
    assert box is not None
    x, y = box["x"] + 15, box["y"] + 15

    def saved_source() -> str:
        annotation = page.request.get(
            f"{live_server_url}/api/v1/projects/{project_id}/frames/0/annotations"
        ).json()
        return annotation["objects"][0]["source"]

    def is_mask_save(response: Response) -> bool:
        return response.url.endswith("/annotations") and response.request.method == "PATCH"

    page.mouse.move(x, y)
    with page.expect_response(is_mask_save):
        page.mouse.down()
        page.mouse.move(x + 15, y + 15)
        page.mouse.up()
    assert saved_source() == "manual_brush"

    ring = page.locator("#brush-cursor")
    expect(ring).not_to_have_class(re.compile(r"\beraser\b"))
    with page.expect_response(is_mask_save):
        page.mouse.down(button="right")
        expect(ring).to_have_class(re.compile(r"\beraser\b"))
        page.mouse.up(button="right")
    assert saved_source() == "manual_eraser"
    expect(ring).not_to_have_class(re.compile(r"\beraser\b"))

    menu_blocked = page.evaluate(
        "() => !document.getElementById('frame-canvas').dispatchEvent("
        "new MouseEvent('contextmenu', {bubbles: true, cancelable: true}))"
    )
    assert menu_blocked


def test_wheel_with_ctrl_zooms_and_with_shift_sizes_the_brush(
    page: Page, live_server_url: str, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_via_requests(live_server_url, "browser wheel test")

    page.goto(f"{live_server_url}/projects/{project_id}")
    expect(page.locator("#status-line")).to_contain_text("Ready", timeout=10_000)
    box = page.locator("#frame-canvas").bounding_box()
    assert box is not None
    page.mouse.move(box["x"] + 20, box["y"] + 20)
    zoom = page.locator("#zoom-slider")
    radius = page.locator("#brush-radius")

    page.keyboard.down("Control")
    page.mouse.wheel(0, -100)
    expect(zoom).to_have_value("110")
    page.mouse.wheel(0, 100)
    page.mouse.wheel(0, 100)
    expect(zoom).to_have_value("90")
    page.keyboard.up("Control")
    expect(radius).to_have_value("10")

    page.keyboard.down("Shift")
    page.mouse.wheel(0, -100)
    expect(radius).to_have_value("12")
    # What a browser actually delivers for Shift+wheel: a horizontal delta.
    page.mouse.wheel(100, 0)
    expect(radius).to_have_value("10")
    page.keyboard.up("Shift")
    expect(zoom).to_have_value("90")


def test_ctrl_wheel_zooms_around_the_pointer_and_every_edge_stays_reachable(
    page: Page, live_server_url: str, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_via_requests(live_server_url, "browser zoom anchor test")

    page.goto(f"{live_server_url}/projects/{project_id}")
    expect(page.locator("#status-line")).to_contain_text("Ready", timeout=10_000)
    # Pin the viewport below the frame's size so zooming has to scroll.
    page.evaluate(
        """() => {
          const scroller = document.getElementById("canvas-scroll");
          scroller.style.flex = "none";
          scroller.style.width = "120px";
          scroller.style.height = "120px";
        }"""
    )
    page.locator("#zoom-slider").fill("300")

    scroller = page.locator("#canvas-scroll")
    canvas = page.locator("#frame-canvas")
    scroller.evaluate("el => { el.scrollLeft = 0; el.scrollTop = 0; }")
    edge = page.evaluate(
        """() => {
          const canvas = document.getElementById("frame-canvas").getBoundingClientRect();
          const scroller = document.getElementById("canvas-scroll").getBoundingClientRect();
          return canvas.left - scroller.left;
        }"""
    )
    assert edge >= 0, "the frame's left edge can't be scrolled into view"

    scroller.evaluate("el => { el.scrollLeft = 40; el.scrollTop = 40; }")
    scroller_box = scroller.bounding_box()
    assert scroller_box is not None
    pointer_x, pointer_y = scroller_box["x"] + 60, scroller_box["y"] + 60
    page.mouse.move(pointer_x, pointer_y)

    def frame_fraction_under_pointer() -> tuple[float, float]:
        box = canvas.bounding_box()
        assert box is not None
        return (pointer_x - box["x"]) / box["width"], (pointer_y - box["y"]) / box["height"]

    before = frame_fraction_under_pointer()
    page.keyboard.down("Control")
    page.mouse.wheel(0, 100)
    page.keyboard.up("Control")
    expect(page.locator("#zoom-slider")).to_have_value("290")
    after = frame_fraction_under_pointer()
    assert after == pytest.approx(before, abs=0.01)


def test_number_row_keys_pick_the_rail_tools(
    page: Page, live_server_url: str, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_via_requests(live_server_url, "browser tool keys test")

    page.goto(f"{live_server_url}/projects/{project_id}")
    expect(page.locator("#status-line")).to_contain_text("Ready", timeout=10_000)
    active = page.locator(".tool-button.active")
    tools = ("point_positive", "point_negative", "brush", "eraser", "axis_hint")
    # Physical keys 1-5 (typed with or without Shift)...
    for digit, tool in reversed(list(enumerate(tools, start=1))):
        page.keyboard.press(f"Digit{digit}")
        expect(active).to_have_attribute("data-tool", tool)
    # ...and, failing that, the characters a Czech layout types on them
    # (dispatched directly: Playwright's US layout can't type them).
    for key, tool in zip(("+", "\u011b", "\u0161", "\u010d", "\u0159"), tools, strict=True):
        page.keyboard.press("Digit3" if tool != "brush" else "Digit1")  # move off it first
        page.evaluate(
            "key => document.body.dispatchEvent("
            "new KeyboardEvent('keydown', {key, bubbles: true}))",
            key,
        )
        expect(active).to_have_attribute("data-tool", tool)


def test_brush_cursor_is_a_ring_of_the_brush_radius(
    page: Page, live_server_url: str, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_via_requests(live_server_url, "browser brush cursor test")

    page.goto(f"{live_server_url}/projects/{project_id}")
    expect(page.locator("#status-line")).to_contain_text("Ready", timeout=10_000)
    canvas = page.locator("#frame-canvas")
    ring = page.locator("#brush-cursor")
    box = canvas.bounding_box()
    assert box is not None
    centre_x, centre_y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2

    page.mouse.move(centre_x, centre_y)
    expect(ring).to_be_hidden()  # point tool: the crosshair stays

    page.keyboard.press("Digit3")
    page.mouse.move(centre_x + 1, centre_y)
    expect(ring).to_be_visible()
    expect(canvas).to_have_css("cursor", "none")

    def ring_box() -> dict:
        ring_bounds = ring.bounding_box()
        assert ring_bounds is not None
        return ring_bounds

    # Default radius 10 frame px at 100% zoom, centred on the pointer.
    assert ring_box()["width"] == pytest.approx(20, abs=1)
    assert ring_box()["x"] + ring_box()["width"] / 2 == pytest.approx(centre_x + 1, abs=1)

    # Radius 30 at 200% zoom: 120 px across on screen.
    page.locator("#brush-radius").fill("30")
    page.locator("#zoom-slider").fill("200")
    box = canvas.bounding_box()
    assert box is not None
    page.mouse.move(box["x"] + 100, box["y"] + 100)
    assert ring_box()["width"] == pytest.approx(120, abs=1.5)

    page.keyboard.press("Digit4")
    expect(ring).to_have_class("eraser")
    page.mouse.move(box["x"] - 50, box["y"] - 50)
    expect(ring).to_be_hidden()


def test_click_points_stay_the_same_size_on_screen_when_zooming(
    page: Page, live_server_url: str, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_via_requests(live_server_url, "browser point zoom test")

    page.goto(f"{live_server_url}/projects/{project_id}")
    expect(page.locator("#status-line")).to_contain_text("Ready", timeout=10_000)
    page.fill("#new-object-label", "pipette")
    page.click("#add-object-button")
    expect(page.locator("#object-list li")).to_have_count(1)
    canvas = page.locator("#frame-canvas")
    canvas.click(position={"x": 20, "y": 20})
    page.click('[data-tool="point_negative"]')
    canvas.click(position={"x": 40, "y": 40})
    points = page.locator("#point-layer .prompt-point")
    expect(points).to_have_count(2)
    expect(page.locator("#point-layer .prompt-point.negative")).to_have_count(1)

    def centre(index: int) -> tuple[float, float]:
        box = points.nth(index).bounding_box()
        assert box is not None
        return box["x"] + box["width"] / 2, box["y"] + box["height"] / 2

    sizes_at_100 = points.first.bounding_box()
    page.locator("#zoom-slider").fill("300")
    sizes_at_300 = points.first.bounding_box()
    assert sizes_at_100 is not None and sizes_at_300 is not None
    assert sizes_at_300["width"] == sizes_at_100["width"] == 10

    # Each point stays on the frame pixel it was placed on.
    canvas_box = canvas.bounding_box()
    assert canvas_box is not None
    first_x, first_y = centre(0)
    assert first_x - canvas_box["x"] == pytest.approx(60, abs=1.5)
    assert first_y - canvas_box["y"] == pytest.approx(60, abs=1.5)


def test_object_list_shows_only_objects_on_the_current_frame(
    page: Page, live_server_url: str, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_via_requests(live_server_url, "browser per-frame list test")

    page.goto(f"{live_server_url}/projects/{project_id}")
    expect(page.locator("#status-line")).to_contain_text("Ready", timeout=10_000)
    items = page.locator("#object-list li")
    for count, (name, x) in enumerate((("pipette", 20), ("glove", 50)), start=1):
        page.fill("#new-object-label", name)
        page.click("#add-object-button")
        # A new object with no mask yet stays listed while the next is made.
        expect(items).to_have_count(count)
        page.locator("#frame-canvas").click(position={"x": x, "y": x})
        expect(page.locator("#accept-candidate-button")).to_be_enabled(timeout=10_000)
        page.click("#accept-candidate-button")
        expect(page.locator("#status-line")).to_contain_text("Saved", timeout=10_000)

    toggle = page.locator("#toggle-offframe-objects")
    expect(toggle).to_be_hidden()

    # Frame 1 has no masks: only the selected object (glove) is listed, so a
    # mask for it can still be drawn here.
    page.fill("#frame-number-input", "1")
    page.locator("#frame-number-input").press("Enter")
    expect(items).to_have_count(1, timeout=10_000)
    expect(items.first).to_contain_text("glove")
    expect(toggle).to_have_text("Show 1 not on this frame")

    toggle.click()
    expect(items).to_have_count(2)
    expect(toggle).to_have_text("Hide 1 not on this frame")
    toggle.click()
    expect(items).to_have_count(1)


def test_frame_navigation_persists_across_reload(
    page: Page, live_server_url: str, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_via_requests(live_server_url, "browser frame nav test")

    page.goto(f"{live_server_url}/projects/{project_id}")
    # init()'s own loadFrame(0) would otherwise land after the jump to 3.
    expect(page.locator("#status-line")).to_contain_text("Ready", timeout=10_000)
    page.fill("#frame-number-input", "3")
    page.locator("#frame-number-input").press("Enter")
    expect(page.locator("#frame-slider")).to_have_value("3", timeout=10_000)

    # Allow the debounced ui_state autosave to fire before reloading.
    page.wait_for_timeout(500)
    page.reload()
    expect(page.locator("#frame-number-input")).to_have_value("3", timeout=10_000)


def test_bottom_bar_review_status_saves_and_follows_the_frame(
    page: Page, live_server_url: str, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_via_requests(live_server_url, "browser review bar test")
    page.goto(f"{live_server_url}/projects/{project_id}")
    expect(page.locator("#status-line")).to_contain_text("Ready", timeout=10_000)
    bar = page.locator("#bar-review-status")
    expect(bar).to_have_value("unreviewed")

    bar.select_option("needs_review")
    expect(page.locator("#status-line")).to_contain_text("Frame 0: Needs review")
    expect(bar).to_have_attribute("data-status", "needs_review")
    expect(page.locator("#review-status")).to_have_value("needs_review")
    frame = httpx.get(f"{live_server_url}/api/v1/projects/{project_id}/frames/0/annotations")
    assert frame.json()["review_status"] == "needs_review"

    # Another frame shows its own status; R on it shows in the bar too.
    page.locator("#next-frame-button").click()
    expect(page.locator("#frame-slider")).to_have_value("1")
    expect(bar).to_have_value("unreviewed")
    page.keyboard.press("r")
    expect(bar).to_have_value("reviewed")
    expect(bar).to_have_attribute("data-status", "reviewed")


def test_object_relabel_and_delete(
    page: Page, live_server_url: str, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_via_requests(live_server_url, "browser object mgmt test")

    page.goto(f"{live_server_url}/projects/{project_id}")
    expect(page.locator("#status-line")).to_contain_text("Ready", timeout=10_000)
    page.fill("#new-object-label", "pipette")
    page.click("#add-object-button")
    expect(page.locator("#object-list li")).to_have_count(1)

    page.locator("#frame-canvas").click(position={"x": 20, "y": 20})
    expect(page.locator("#accept-candidate-button")).to_be_enabled(timeout=10_000)
    page.click("#accept-candidate-button")
    expect(page.locator("#status-line")).to_contain_text("Saved", timeout=10_000)

    # Deactivated on this frame: its mask moves to inactive_objects.
    page.get_by_role("button", name="Deactivate", exact=True).click()
    expect(page.locator("#object-list li.inactive")).to_have_count(1, timeout=10_000)

    # Delete asks first; dismissing keeps the object.
    page.once("dialog", lambda dialog: dialog.dismiss())
    page.click("#object-list li >> text=Delete")
    expect(page.locator("#object-list li")).to_have_count(1)

    page.once("dialog", lambda dialog: dialog.accept())
    page.click("#object-list li >> text=Delete")
    expect(page.locator("#object-list li")).to_have_count(0, timeout=10_000)
    expect(page.locator("#status-line")).to_contain_text("Deleted", timeout=10_000)
    frame = httpx.get(
        f"{live_server_url}/api/v1/projects/{project_id}/frames/0/annotations", timeout=10
    ).json()
    assert frame["objects"] == []
    assert frame["inactive_objects"] == []


def test_undo_brings_back_a_deleted_object_and_redo_deletes_it_again(
    page: Page, live_server_url: str, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_via_requests(live_server_url, "browser undo delete test")
    frame_url = f"{live_server_url}/api/v1/projects/{project_id}/frames/0/annotations"

    page.goto(f"{live_server_url}/projects/{project_id}")
    expect(page.locator("#status-line")).to_contain_text("Ready", timeout=10_000)
    page.fill("#new-object-label", "pipette")
    page.click("#add-object-button")
    items = page.locator("#object-list li")
    expect(items).to_have_count(1)
    page.locator("#frame-canvas").click(position={"x": 20, "y": 20})
    expect(page.locator("#accept-candidate-button")).to_be_enabled(timeout=10_000)
    page.click("#accept-candidate-button")
    expect(page.locator("#status-line")).to_contain_text("Saved", timeout=10_000)
    mask = httpx.get(frame_url, timeout=10).json()["objects"][0]["visible_mask_rle"]

    page.once("dialog", lambda dialog: dialog.accept())
    items.get_by_role("button", name="Delete", exact=True).click()
    expect(items).to_have_count(0, timeout=10_000)
    expect(page.locator("#status-line")).to_contain_text("Ctrl+Z", timeout=10_000)
    assert httpx.get(frame_url, timeout=10).json()["objects"] == []

    # Moving to another frame and back doesn't lose the undo.
    page.keyboard.press("ArrowRight")
    expect(page.locator("#frame-number-input")).to_have_value("1", timeout=10_000)
    page.keyboard.press("ArrowLeft")
    expect(page.locator("#frame-number-input")).to_have_value("0", timeout=10_000)

    page.keyboard.press("Control+z")
    expect(items).to_have_count(1, timeout=10_000)
    expect(items.first).to_contain_text("pipette")
    expect(page.locator("#status-line")).to_contain_text("Restored", timeout=10_000)
    frame = httpx.get(frame_url, timeout=10).json()
    assert [o["visible_mask_rle"] for o in frame["objects"]] == [mask]
    expect(page.locator("#class-list li")).to_have_count(1)

    page.keyboard.press("Control+y")
    expect(items).to_have_count(0, timeout=10_000)
    expect(page.locator("#status-line")).to_contain_text("again", timeout=10_000)
    page.click("#undo-button")
    expect(items).to_have_count(1, timeout=10_000)

    # A whole class too.
    page.click("#class-section summary")
    page.once("dialog", lambda dialog: dialog.accept())
    page.locator("#class-list button.class-delete").click()
    expect(items).to_have_count(0, timeout=10_000)
    page.keyboard.press("Control+z")
    expect(items).to_have_count(1, timeout=10_000)
    assert len(httpx.get(frame_url, timeout=10).json()["objects"]) == 1


def test_locked_object_cannot_be_deleted(
    page: Page, live_server_url: str, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_via_requests(live_server_url, "browser object lock test")

    page.goto(f"{live_server_url}/projects/{project_id}")
    expect(page.locator("#status-line")).to_contain_text("Ready", timeout=10_000)
    page.fill("#new-object-label", "pipette")
    page.click("#add-object-button")
    item = page.locator("#object-list li")
    expect(item).to_have_count(1)
    page.locator("#frame-canvas").click(position={"x": 20, "y": 20})
    expect(page.locator("#accept-candidate-button")).to_be_enabled(timeout=10_000)
    page.click("#accept-candidate-button")
    expect(page.locator("#status-line")).to_contain_text("Saved", timeout=10_000)

    delete = item.get_by_role("button", name="Delete", exact=True)
    class_delete = page.locator("#class-list button.class-delete")
    item.get_by_role("button", name="Lock", exact=True).click()
    expect(page.locator("#object-list li.locked")).to_have_count(1)
    expect(delete).to_be_disabled()
    expect(class_delete).to_be_disabled()
    objects = httpx.get(f"{live_server_url}/api/v1/projects/{project_id}/objects", timeout=10)
    assert objects.json()["objects"][0]["locked"] is True

    # Still locked after a reload, and it can still be deactivated.
    page.reload()
    expect(delete).to_be_disabled(timeout=10_000)
    item.get_by_role("button", name="Deactivate", exact=True).click()
    expect(page.locator("#object-list li.inactive")).to_have_count(1, timeout=10_000)
    expect(delete).to_be_disabled()

    item.get_by_role("button", name="Unlock", exact=True).click()
    expect(delete).to_be_enabled()
    expect(class_delete).to_be_enabled()
    page.once("dialog", lambda dialog: dialog.accept())
    delete.click()
    expect(item).to_have_count(0, timeout=10_000)


def test_class_button_changes_the_object_class_on_every_frame(
    page: Page, live_server_url: str, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_via_requests(live_server_url, "class change test")

    page.goto(f"{live_server_url}/projects/{project_id}")
    expect(page.locator("#status-line")).to_contain_text("Ready", timeout=10_000)
    page.fill("#new-object-label", "Tube 15ml")
    page.click("#add-object-button")
    expect(page.locator("#object-list li")).to_have_count(1)
    page.locator("#frame-canvas").click(position={"x": 20, "y": 20})
    expect(page.locator("#accept-candidate-button")).to_be_enabled(timeout=10_000)
    page.click("#accept-candidate-button")
    expect(page.locator("#status-line")).to_contain_text("Saved", timeout=10_000)

    item = page.locator("#object-list li")
    # Escape leaves the class as it was.
    item.locator(".object-class").click()
    page.locator(".object-class-input").fill("nope")
    page.locator(".object-class-input").press("Escape")
    expect(item).to_contain_text("Tube 15ml")

    item.locator(".object-class").click()
    page.locator(".object-class-input").fill("Tube 50ml")
    page.locator(".object-class-input").press("Enter")
    expect(page.locator("#status-line")).to_contain_text("is now Tube 50ml", timeout=10_000)
    expect(item).to_contain_text("Tube 50ml")

    labels = httpx.get(f"{live_server_url}/api/v1/projects/{project_id}/labels").json()
    assert [label["name"] for label in labels["labels"]] == ["Tube 50ml"]
    frame = httpx.get(
        f"{live_server_url}/api/v1/projects/{project_id}/frames/0/annotations", timeout=10
    ).json()
    assert [o["label_id"] for o in frame["objects"]] == [labels["labels"][0]["label_id"]]


def test_class_order_arrows_renumber_the_export_classes(
    page: Page, live_server_url: str, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_via_requests(live_server_url, "class order test")

    page.goto(f"{live_server_url}/projects/{project_id}")
    expect(page.locator("#status-line")).to_contain_text("Ready", timeout=10_000)
    expect(page.locator("#class-section")).to_be_hidden()

    for count, name in enumerate(("QR code on white sticker", "hand"), start=1):
        page.fill("#new-object-label", name)
        page.click("#add-object-button")
        expect(page.locator("#object-list li")).to_have_count(count)

    page.click("#class-section summary")
    rows = page.locator("#class-list li")
    _expect_classes(page, "QR code on white sticker", "hand")
    expect(rows.nth(0).get_by_role("button").first).to_be_disabled()

    page.get_by_role("button", name="Move up (lower class number): hand").click()
    _expect_classes(page, "hand", "QR code on white sticker")

    labels = httpx.get(f"{live_server_url}/api/v1/projects/{project_id}/labels").json()
    # Only the order changes; each label keeps the label_id objects reference.
    assert [(label["name"], label["label_id"]) for label in labels["labels"]] == [
        ("hand", 1),
        ("QR code on white sticker", 0),
    ]

    # Deleting the last "hand" object takes the class with it.
    page.once("dialog", lambda dialog: dialog.accept())
    page.locator("#object-list li", has_text="hand").locator(".object-delete").click()
    expect(page.locator("#object-list li")).to_have_count(1)
    _expect_classes(page, "QR code on white sticker")


def _expect_classes(page: Page, *names: str) -> None:
    """The Class order list shows these names, numbered 0, 1, ... in order."""
    rows = page.locator("#class-list li")
    expect(rows).to_have_count(len(names))
    for index, name in enumerate(names):
        expect(rows.nth(index).locator(".class-number")).to_have_text(str(index))
        expect(rows.nth(index).locator(".class-name")).to_have_value(name)


def _class_project(page: Page, live_server_url: str, names: tuple[str, ...]) -> str:
    """A project with one object (and its mask on frame 0) per name."""
    project_id = create_project_via_requests(live_server_url, "class actions test")
    page.goto(f"{live_server_url}/projects/{project_id}")
    expect(page.locator("#status-line")).to_contain_text("Ready", timeout=10_000)
    for count, name in enumerate(names, start=1):
        page.fill("#new-object-label", name)
        page.click("#add-object-button")
        expect(page.locator("#object-list li")).to_have_count(count)
        page.locator("#frame-canvas").click(position={"x": 15 * count, "y": 15 * count})
        expect(page.locator("#accept-candidate-button")).to_be_enabled(timeout=10_000)
        page.click("#accept-candidate-button")
        expect(page.locator("#status-line")).to_contain_text("Saved", timeout=10_000)
    page.click("#class-section summary")
    return project_id


def _frame_label_ids(live_server_url: str, project_id: str) -> list[int]:
    frame = httpx.get(
        f"{live_server_url}/api/v1/projects/{project_id}/frames/0/annotations", timeout=10
    ).json()
    return sorted(o["label_id"] for o in frame["objects"])


def test_class_name_box_renames_the_class(
    page: Page, live_server_url: str, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = _class_project(page, live_server_url, ("Tube 15ml", "hand"))
    name = page.locator("#class-list li").nth(0).locator(".class-name")

    # Escape puts the old name back.
    name.fill("nope")
    name.press("Escape")
    expect(name).to_have_value("Tube 15ml")

    name.fill("Tube 50ml")
    name.press("Enter")
    expect(page.locator("#status-line")).to_contain_text('Renamed "Tube 15ml"', timeout=10_000)
    _expect_classes(page, "Tube 50ml", "hand")
    expect(page.locator("#object-list li").nth(0)).to_contain_text("Tube 50ml")

    # Renaming to another class's name offers a merge instead.
    page.once("dialog", lambda dialog: dialog.accept())
    name = page.locator("#class-list li").nth(0).locator(".class-name")
    name.fill("hand")
    name.press("Enter")
    expect(page.locator("#status-line")).to_contain_text(
        'Merged 1 object into "hand"', timeout=10_000
    )
    _expect_classes(page, "hand")
    labels = httpx.get(f"{live_server_url}/api/v1/projects/{project_id}/labels").json()
    assert _frame_label_ids(live_server_url, project_id) == [labels["labels"][0]["label_id"]] * 2


def test_merge_and_delete_buttons_change_the_whole_class(
    page: Page, live_server_url: str, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = _class_project(page, live_server_url, ("qr", "tube", "hand"))
    rows = page.locator("#class-list li")

    # Merge offers the other classes; Escape cancels.
    rows.nth(0).locator(".class-merge").click()
    select = rows.nth(0).locator(".class-merge-select")
    expect(select.locator("option")).to_have_text(["Merge into...", "tube", "hand"])
    select.press("Escape")
    expect(rows.nth(0).locator(".class-merge")).to_be_visible()

    page.once("dialog", lambda dialog: dialog.accept())
    rows.nth(0).locator(".class-merge").click()
    rows.nth(0).locator(".class-merge-select").select_option(label="hand")
    expect(page.locator("#status-line")).to_contain_text(
        'Merged 1 object into "hand"', timeout=10_000
    )
    _expect_classes(page, "tube", "hand")
    expect(page.locator("#object-list li", has_text="hand")).to_have_count(2)

    # Delete asks first; dismissing keeps the class.
    page.once("dialog", lambda dialog: dialog.dismiss())
    rows.nth(1).locator(".class-delete").click()
    _expect_classes(page, "tube", "hand")

    page.once("dialog", lambda dialog: dialog.accept())
    rows.nth(1).locator(".class-delete").click()
    expect(page.locator("#status-line")).to_contain_text('Deleted "hand"', timeout=10_000)
    _expect_classes(page, "tube")
    expect(page.locator("#object-list li")).to_have_count(1)
    assert len(_frame_label_ids(live_server_url, project_id)) == 1
