# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Preview and acceptance UI with deterministic text-search responses."""

import httpx
import numpy as np
import pytest
from playwright.sync_api import Page, Route, expect

from tests.conftest import create_project_via_requests
from vistlabel.annotations.png_mask import encode_mask_png_base64
from vistlabel.video.ffmpeg import FfmpegBinaries

pytestmark = pytest.mark.browser


def test_workspace_bypasses_old_unversioned_assets(
    page: Page, live_server_url: str, ffmpeg_binaries: FfmpegBinaries,
) -> None:
    project_id = create_project_via_requests(live_server_url, "cached workspace")
    _mock_text_backend(page)
    # Simulate a browser retaining JavaScript from before text search existed.
    page.route(
        "**/static/js/*.js",
        lambda route: route.fulfill(
            body="throw new Error('Loaded stale unversioned JavaScript');",
            content_type="text/javascript",
        ),
    )
    requested: list[str] = []
    page.on("request", lambda request: requested.append(request.url))
    page.goto(f"{live_server_url}/projects/{project_id}")
    expect(page.locator("#status-line")).to_have_text("Ready")
    page.select_option("#backend-select", "sam3")
    expect(page.locator("#concept-search-button")).to_be_enabled()
    expect(page.locator("#concept-search-controls")).to_have_css("display", "grid")
    assets = [url for url in requested if "/static/js/" in url or "/static/css/" in url]
    assert any("concept-search.js?v=" in url for url in assets)
    assert all("?v=" in url for url in assets)
    assert len({url.split("?v=")[1] for url in assets}) == 1


def _mock_text_backend(page: Page) -> None:
    def models(route: Route) -> None:
        data = route.fetch().json()
        backend = next(b for b in data["backends"] if b["backend_id"] == "sam3")
        backend["available"] = True
        backend["unavailable_reason"] = None
        route.fulfill(json=data)

    page.route("**/api/v1/models", models)


def _response() -> dict:
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:20, 10:20] = True
    mask[30:40, 10:20] = True
    return {
        "frame_index": 0, "backend_id": "sam3", "model_name": "sam3",
        "checkpoint_sha256": "test-checkpoint", "code_revision": "test-revision",
        "proposals": [{
            "proposal_id": "pipette:1", "frame_index": 0,
            "visible_mask_png_base64": encode_mask_png_base64(mask),
            "score": 0.9, "suggested_label_name": "pipette",
            "box_xyxy": [10, 10, 19, 39],
        }],
    }


def test_text_preview_accept_and_reload(
    page: Page, live_server_url: str, ffmpeg_binaries: FfmpegBinaries,
) -> None:
    project_id = create_project_via_requests(live_server_url, "text segmentation")
    base = f"{live_server_url}/api/v1/projects/{project_id}"
    _mock_text_backend(page)
    page.route("**/segment-concept", lambda route: route.fulfill(json=_response()))
    page.goto(f"{live_server_url}/projects/{project_id}")
    expect(page.locator("#status-line")).to_have_text("Ready")
    expect(page.locator("#concept-search-button")).to_be_disabled()
    page.select_option("#backend-select", "sam3")
    page.fill("#concept-text", "pipette")
    page.click("#concept-search-button")
    expect(page.locator("#concept-results")).to_be_visible()
    assert httpx.get(f"{base}/objects").json()["objects"] == []
    assert httpx.get(f"{base}/frames/0/annotations").json()["objects"] == []
    page.fill("#concept-label", "glass pipette")
    page.click("#concept-accept-button")
    expect(page.locator("#concept-status")).to_have_text("All matching instances accepted.")
    saved = httpx.get(f"{base}/frames/0/annotations").json()["objects"][0]
    assert saved["source"] == "sam3_text"
    assert saved["confidence"] == 0.9
    assert saved["connected_component_count"] == 2
    assert saved["backend_provenance"]["checkpoint_sha256"] == "test-checkpoint"
    assert httpx.get(f"{base}/objects").json()["objects"][0]["label_name"] == "glass pipette"
    page.reload()
    expect(page.locator("#status-line")).to_have_text("Ready")
    expect(page.locator("#object-list li")).to_have_count(1)


def test_late_text_result_is_discarded_after_frame_navigation(
    page: Page, live_server_url: str, ffmpeg_binaries: FfmpegBinaries,
) -> None:
    project_id = create_project_via_requests(live_server_url, "stale text result")
    _mock_text_backend(page)
    pending: list[Route] = []
    page.route("**/segment-concept", lambda route: pending.append(route))
    page.goto(f"{live_server_url}/projects/{project_id}")
    expect(page.locator("#status-line")).to_have_text("Ready")
    page.select_option("#backend-select", "sam3")
    page.fill("#concept-text", "pipette")
    with page.expect_request("**/segment-concept"):
        page.click("#concept-search-button")
    expect(page.locator("#concept-search-button")).to_be_disabled()
    expect(page.locator("#concept-search-button")).to_contain_text("Searching")
    expect(page.locator("#concept-search-button")).to_have_attribute("aria-busy", "true")
    expect(page.locator("#status-line")).to_contain_text("Searching this frame")
    page.click("#next-frame-button")
    expect(page.locator("#frame-number-input")).to_have_value("1")
    pending[0].fulfill(json=_response())
    expect(page.locator("#concept-results")).to_be_hidden()
    expect(page.locator("#concept-search-button")).to_be_enabled()
    expect(page.locator("#concept-search-button")).to_have_text("Find in this frame")
    expect(page.locator("#concept-search-button")).to_have_attribute("aria-busy", "false")
