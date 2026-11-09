# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Browser-driven bounding-box display toggle and the export dialog's
annotation-type / dataset-format options."""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

from tests.conftest import create_project_via_requests
from vistlabel.video.ffmpeg import FfmpegBinaries

pytestmark = pytest.mark.browser

_CANVAS_PIXELS = """() => {
  const canvas = document.getElementById("frame-canvas");
  return Array.from(
    canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data
  );
}"""


def _seed_one_object(page: Page, live_server_url: str, project_id: str) -> None:
    page.goto(f"{live_server_url}/projects/{project_id}")
    expect(page.locator("#status-line")).to_contain_text("Ready", timeout=10_000)
    page.fill("#new-object-label", "pipette")
    page.click("#add-object-button")
    expect(page.locator("#object-list li")).to_have_count(1)
    page.locator("#frame-canvas").click(position={"x": 20, "y": 20})
    expect(page.locator("#accept-candidate-button")).to_be_enabled(timeout=10_000)
    page.click("#accept-candidate-button")
    expect(page.locator("#status-line")).to_contain_text("Saved", timeout=10_000)


def test_show_boxes_draws_a_sharp_tag_and_is_remembered_across_reload(
    page: Page, live_server_url: str, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_via_requests(live_server_url, "browser boxes test")
    _seed_one_object(page, live_server_url, project_id)

    expect(page.locator("#show-boxes")).not_to_be_checked()
    expect(page.locator("#box-layer .box-tag")).to_have_count(0)
    without_boxes = page.evaluate(_CANVAS_PIXELS)
    page.check("#show-boxes")
    expect(page.locator("#box-layer .box-tag")).to_have_text(["pipette"])
    expect(page.locator("#box-layer .box-outline")).to_have_count(1)
    # Boxes sit over the canvas, so the frame canvas itself is unchanged.
    assert page.evaluate(_CANVAS_PIXELS) == without_boxes

    # Zooming in keeps the tag at 12px text instead of stretching it, and
    # the outline follows the zoomed frame.
    outline = page.locator("#box-layer .box-outline")
    width_at_100 = outline.bounding_box()["width"]
    page.locator("#zoom-slider").fill("300")
    tag = page.locator("#box-layer .box-tag")
    expect(tag).to_have_css("font-size", "12px")
    assert tag.bounding_box()["height"] == pytest.approx(16, abs=0.5)
    assert outline.bounding_box()["width"] == pytest.approx(
        (width_at_100 - 2) * 3 + 2, abs=2
    )

    page.reload()
    expect(page.locator("#status-line")).to_contain_text("Ready", timeout=10_000)
    expect(page.locator("#show-boxes")).to_be_checked()
    expect(page.locator("#box-layer .box-tag")).to_have_text(["pipette"])


def test_export_dialog_runs_a_coco_bbox_export(
    page: Page, live_server_url: str, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_via_requests(live_server_url, "browser bbox export test")
    _seed_one_object(page, live_server_url, project_id)

    page.click("#open-export-button")
    expect(page.locator("#export-modal")).to_be_visible()
    # One annotated, unreviewed frame: the counts make the difference plain.
    expect(page.locator('#export-policy option[value="annotated_only"]')).to_have_text(
        "All annotated frames (reviewed or not) (1)"
    )
    expect(page.locator('#export-policy option[value="reviewed_only"]')).to_have_text(
        "Reviewed frames only (0)"
    )
    page.select_option("#export-policy", "annotated_only")

    # Multipart policy only applies to YOLO polygons; the extra-COCO checkbox
    # only applies to the Ultralytics layout.
    page.select_option("#export-annotation-type", "bbox")
    expect(page.locator("#export-multipart-field")).to_be_hidden()
    expect(page.locator("#export-coco-exact-field")).to_be_visible()
    page.select_option("#export-layout", "coco")
    expect(page.locator("#export-coco-exact-field")).to_be_hidden()

    page.click("#export-start-button")
    expect(page.locator("#export-status")).to_contain_text("Export finished", timeout=30_000)
    expect(page.locator("#export-history li").first).to_contain_text("COCO boxes")
