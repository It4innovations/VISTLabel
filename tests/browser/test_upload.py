# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Browser-driven MP4 upload: project creation, file picker,
upload progress, and the resulting project appearing in the list -- through
a real Chromium page, not just the HTTP API.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest
from PIL import Image
from playwright.sync_api import Page, expect

from tests.conftest import TINY_TEST_VIDEO
from vistlabel.video.ffmpeg import FfmpegBinaries

pytestmark = pytest.mark.browser


def test_create_project_via_file_picker_and_see_it_listed(
    page: Page, live_server_url: str, ffmpeg_binaries: FfmpegBinaries
) -> None:
    page.goto(live_server_url)

    page.click("#create-project-button")
    page.fill("#project_name", "browser upload test")
    page.set_input_files("#video_file", str(TINY_TEST_VIDEO))
    page.click("#submit-button")

    # Successful upload reloads the page with the new project listed.
    expect(page.locator("#project-list")).to_contain_text(
        "browser upload test", timeout=10_000
    )
    expect(page.locator("#project-list")).to_contain_text("New")


def test_invalid_file_shows_an_error_without_leaving_the_page(
    page: Page, live_server_url: str, ffmpeg_binaries: FfmpegBinaries, tmp_path: Path
) -> None:
    bad_file = tmp_path / "not_a_video.mp4"
    bad_file.write_bytes(b"not an mp4 container")

    page.goto(live_server_url)
    page.click("#create-project-button")
    page.fill("#project_name", "bad upload test")
    page.set_input_files("#video_file", str(bad_file))
    page.click("#submit-button")

    expect(page.locator("#upload-error")).to_contain_text("Upload failed", timeout=10_000)
    expect(page.locator("#project-list")).not_to_contain_text("bad upload test")


def test_import_options_downscale_and_thin_frames(
    page: Page, live_server_url: str, ffmpeg_binaries: FfmpegBinaries
) -> None:
    page.goto(live_server_url)
    page.click("#create-project-button")
    page.fill("#project_name", "import options test")
    page.set_input_files("#video_file", str(TINY_TEST_VIDEO))

    expect(page.locator("#import-custom-size-field")).to_be_hidden()
    page.select_option("#import-resolution", "custom")
    expect(page.locator("#import-custom-size-field")).to_be_visible()
    page.fill("#import-max-width", "32")
    page.fill("#import-max-height", "32")

    expect(page.locator("#import-keep-every-nth-field")).to_be_hidden()
    page.select_option("#import-frame-mode", "nth")
    page.fill("#import-keep-every-nth", "2")
    page.click("#submit-button")

    # Fixture is 64x64 at 10 FPS with 10 frames.
    card = page.locator(".lv-card", has_text="import options test")
    expect(card).to_contain_text("32×32", timeout=15_000)
    expect(card).to_contain_text("5 frames")
    expect(card).to_contain_text("from 64×64")


def test_create_project_from_a_yolo_dataset(
    page: Page, live_server_url: str, ffmpeg_binaries: FfmpegBinaries, tmp_path: Path
) -> None:
    dataset = tmp_path / "dataset.zip"
    with zipfile.ZipFile(dataset, "w") as archive:
        for index in range(3):
            buffer = io.BytesIO()
            Image.new("RGB", (48, 32), (index * 60, 80, 80)).save(buffer, format="PNG")
            archive.writestr(f"images/train/frame_{index}.png", buffer.getvalue())
        archive.writestr("labels/train/frame_0.txt", "0 0.5 0.5 0.25 0.25\n")
        archive.writestr("data.yaml", "names: [pipette]\n")

    page.goto(live_server_url)
    page.click("#create-project-button")
    page.fill("#project_name", "dataset import test")
    expect(page.locator("#dataset-source-fields")).to_be_hidden()
    page.select_option("#create-source", "dataset")
    expect(page.locator("#video-source-fields")).to_be_hidden()
    # The MP4 input is required for a video project, but not here.
    page.set_input_files("#dataset_file", str(dataset))
    page.click("#submit-button")

    card = page.locator(".lv-card", has_text="dataset import test")
    expect(card).to_contain_text("48×32", timeout=15_000)
    expect(card).to_contain_text("3 frames")


def test_dataset_import_can_keep_every_nth_image(
    page: Page, live_server_url: str, ffmpeg_binaries: FfmpegBinaries, tmp_path: Path
) -> None:
    dataset = tmp_path / "dataset.zip"
    with zipfile.ZipFile(dataset, "w") as archive:
        for index in range(5):
            buffer = io.BytesIO()
            Image.new("RGB", (48, 32), (index * 40, 80, 80)).save(buffer, format="PNG")
            archive.writestr(f"images/train/frame_{index}.png", buffer.getvalue())
        archive.writestr("data.yaml", "names: [pipette]\n")

    page.goto(live_server_url)
    page.click("#create-project-button")
    page.fill("#project_name", "dataset nth test")
    page.select_option("#create-source", "dataset")
    expect(page.locator("#dataset-keep-every-nth-field")).to_be_hidden()
    page.select_option("#dataset-frame-mode", "nth")
    page.fill("#dataset-keep-every-nth", "2")
    page.set_input_files("#dataset_file", str(dataset))
    page.click("#submit-button")

    card = page.locator(".lv-card", has_text="dataset nth test")
    expect(card).to_contain_text("3 frames", timeout=15_000)
