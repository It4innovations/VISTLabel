# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from vistlabel.main import create_app
from vistlabel.settings import AppSettings
from vistlabel.video.ffmpeg import FfmpegBinaries, try_find_ffmpeg_binaries

FIXTURES_DIR = Path(__file__).parent / "fixtures"
TINY_TEST_VIDEO = FIXTURES_DIR / "tiny_test_video.mp4"


@pytest.fixture(autouse=True)
def _no_free_disk_space_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    """The upload's free-disk-space floor (20 GiB by default) is a property
    of the machine, not of the code: tests upload a few-KB fixture video,
    yet failed wholesale on a test machine with ~13 GB free. Every AppSettings
    a test builds -- here, in the browser live server, or inline -- reads
    this from the environment. The check itself is covered by
    tests/unit/test_video_upload.py, which passes its own threshold."""
    monkeypatch.setenv("VISTLABEL_UPLOAD__MINIMUM_FREE_DISK_SPACE_GB", "0")


@pytest.fixture(scope="session")
def ffmpeg_binaries() -> FfmpegBinaries:
    """Resolves ffmpeg/ffprobe via PATH, or via VISTLABEL_RUNTIME__FFMPEG_DIR
    for local development where ffmpeg is installed but not on PATH (see
    README.md). Skips (not fails) dependent tests when unavailable, since
    whether ffmpeg is installed is an environment fact, not a code
    correctness question."""
    ffmpeg_dir_env = os.environ.get("VISTLABEL_RUNTIME__FFMPEG_DIR")
    ffmpeg_dir = Path(ffmpeg_dir_env) if ffmpeg_dir_env else None
    binaries = try_find_ffmpeg_binaries(ffmpeg_dir)
    if binaries is None:
        pytest.skip("ffmpeg/ffprobe not available (set VISTLABEL_RUNTIME__FFMPEG_DIR)")
    return binaries


@pytest.fixture
def app_settings(tmp_path: Path) -> AppSettings:
    return AppSettings(workspace={"root": tmp_path / "workspace"})


@pytest.fixture
def api_client(app_settings: AppSettings) -> TestClient:
    app = create_app(app_settings)
    return TestClient(app)


def create_project_with_video(
    api_client: TestClient, project_name: str, video_path: Path = TINY_TEST_VIDEO
) -> dict:
    with video_path.open("rb") as video_file:
        response = api_client.post(
            "/api/v1/projects",
            data={"project_name": project_name},
            files={"video_file": (video_path.name, video_file, "video/mp4")},
        )
    assert response.status_code == 201, response.text
    return response.json()


def create_project_via_requests(
    base_url: str, project_name: str, video_path: Path = TINY_TEST_VIDEO
) -> str:
    """Same as :func:`create_project_with_video` but against a real running
    server (used by browser tests, which drive a live uvicorn instance
    rather than an in-process TestClient). Returns the new project_id."""
    with video_path.open("rb") as video_file:
        response = httpx.post(
            f"{base_url}/api/v1/projects",
            data={"project_name": project_name},
            files={"video_file": (video_path.name, video_file, "video/mp4")},
            timeout=30,
        )
    assert response.status_code == 201, response.text
    return str(response.json()["project_id"])
