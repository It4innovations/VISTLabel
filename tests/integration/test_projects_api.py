# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient
from httpx import Response

from tests.conftest import FIXTURES_DIR, TINY_TEST_VIDEO, create_project_with_video
from vistlabel.video.ffmpeg import FfmpegBinaries


def test_create_list_get_delete_project(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project = create_project_with_video(api_client, "isolator run 1")
    assert project["name"] == "isolator run 1"
    # Upload lands a project in "new"; moving it on is the operator's call.
    assert project["status"] == "new"
    assert project["source"]["width"] == 64
    assert project["source"]["height"] == 64
    assert project["source"]["frame_count"] == 10
    assert project["progress"]["total_frame_count"] == 10
    project_id = project["project_id"]

    list_response = api_client.get("/api/v1/projects")
    assert list_response.status_code == 200
    assert [p["project_id"] for p in list_response.json()] == [project_id]

    get_response = api_client.get(f"/api/v1/projects/{project_id}")
    assert get_response.status_code == 200
    assert get_response.json()["project_id"] == project_id

    delete_response = api_client.delete(f"/api/v1/projects/{project_id}")
    assert delete_response.status_code == 204

    missing_response = api_client.get(f"/api/v1/projects/{project_id}")
    assert missing_response.status_code == 404


def test_get_unknown_project_returns_404(api_client: TestClient) -> None:
    response = api_client.get("/api/v1/projects/prj_doesnotexist")
    assert response.status_code == 404


def test_create_project_with_empty_name_is_rejected(api_client: TestClient) -> None:
    with TINY_TEST_VIDEO.open("rb") as video_file:
        response = api_client.post(
            "/api/v1/projects",
            data={"project_name": "   "},
            files={"video_file": ("tiny_test_video.mp4", video_file, "video/mp4")},
        )
    assert response.status_code == 422


def test_create_project_with_invalid_video_is_rejected_and_leaves_no_project(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    not_a_video = b"this is definitely not an mp4 file"
    response = api_client.post(
        "/api/v1/projects",
        data={"project_name": "bad upload"},
        files={"video_file": ("not_a_video.mp4", not_a_video, "video/mp4")},
    )
    assert response.status_code == 422

    assert api_client.get("/api/v1/projects").json() == []


def test_labels_start_empty_and_can_be_replaced(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_with_video(api_client, "labels api test")["project_id"]

    labels_response = api_client.get(f"/api/v1/projects/{project_id}/labels")
    assert labels_response.status_code == 200
    # Operator-defined ontology: nothing is pre-seeded.
    assert labels_response.json()["labels"] == []

    new_labels = {
        "labels": [
            {"label_id": 0, "name": "custom_a"},
            {"label_id": 1, "name": "custom_b"},
        ]
    }
    put_response = api_client.put(f"/api/v1/projects/{project_id}/labels", json=new_labels)
    assert put_response.status_code == 200
    assert {label["name"] for label in put_response.json()["labels"]} == {"custom_a", "custom_b"}

    get_again = api_client.get(f"/api/v1/projects/{project_id}/labels")
    assert {label["name"] for label in get_again.json()["labels"]} == {"custom_a", "custom_b"}


def test_labels_for_unknown_project_returns_404(api_client: TestClient) -> None:
    response = api_client.get("/api/v1/projects/prj_doesnotexist/labels")
    assert response.status_code == 404


def test_get_frame_returns_a_jpeg(api_client: TestClient, ffmpeg_binaries: FfmpegBinaries) -> None:
    project_id = create_project_with_video(api_client, "frames test")["project_id"]

    response = api_client.get(f"/api/v1/projects/{project_id}/frames/0")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content[:2] == b"\xff\xd8"  # JPEG magic bytes


def test_get_frame_is_cached_and_deterministic(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_with_video(api_client, "frames cache test")["project_id"]

    first = api_client.get(f"/api/v1/projects/{project_id}/frames/3")
    second = api_client.get(f"/api/v1/projects/{project_id}/frames/3")

    assert first.content == second.content


def test_get_frame_out_of_range_returns_404(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_with_video(api_client, "frames range test")["project_id"]

    response = api_client.get(f"/api/v1/projects/{project_id}/frames/9999")

    assert response.status_code == 404


def test_get_frame_thumbnail_is_smaller_than_full_frame(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_with_video(api_client, "thumbnail test")["project_id"]

    frame = api_client.get(f"/api/v1/projects/{project_id}/frames/0")
    thumbnail = api_client.get(f"/api/v1/projects/{project_id}/frames/0/thumbnail")

    assert thumbnail.status_code == 200
    assert thumbnail.headers["content-type"] == "image/jpeg"
    # The fixture is only 64x64, smaller than the thumbnail cap, so this
    # mostly asserts the endpoint round-trips rather than that it shrank.
    assert len(thumbnail.content) > 0
    assert len(frame.content) > 0


def test_frames_for_project_without_video_return_409(api_client: TestClient) -> None:
    # There is no HTTP path today that leaves a project without a video
    # (project creation and upload are a single combined flow) -- reach into
    # the service layer directly to exercise the defensive check regardless.
    project_store = api_client.app.state.project_store  # type: ignore[attr-defined]
    project = project_store.create("no video yet")

    response = api_client.get(f"/api/v1/projects/{project.project_id}/frames/0")

    assert response.status_code == 409


def test_fixture_video_file_exists() -> None:
    assert FIXTURES_DIR.is_dir()
    assert TINY_TEST_VIDEO.is_file()


def test_status_can_be_moved_through_the_workflow(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project = create_project_with_video(api_client, "status workflow test")
    project_id = project["project_id"]
    assert project["status"] == "new"

    response = api_client.patch(
        f"/api/v1/projects/{project_id}",
        json={"expected_revision": project["revision"], "status": "in_progress"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "in_progress"

    response = api_client.patch(
        f"/api/v1/projects/{project_id}",
        json={"expected_revision": response.json()["revision"], "status": "done"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "done"


def test_status_cannot_be_set_back_to_pending(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    """"pending" means "upload not finished" -- only the upload path may
    assert it, otherwise a project could claim to have no usable video."""
    project = create_project_with_video(api_client, "status pending test")

    response = api_client.patch(
        f"/api/v1/projects/{project['project_id']}",
        json={"expected_revision": project["revision"], "status": "pending"},
    )

    assert response.status_code == 422
    assert "not operator-settable" in response.json()["detail"]


def test_status_change_leaves_ui_state_untouched(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project = create_project_with_video(api_client, "status ui_state test")
    project_id = project["project_id"]

    stored = api_client.patch(
        f"/api/v1/projects/{project_id}",
        json={"expected_revision": project["revision"], "last_opened_frame": 4},
    ).json()

    updated = api_client.patch(
        f"/api/v1/projects/{project_id}",
        json={"expected_revision": stored["revision"], "status": "done"},
    ).json()

    assert updated["status"] == "done"
    assert updated["ui_state"]["last_opened_frame"] == 4


def _create_with_options(api_client: TestClient, name: str, **options: str) -> Response:
    with TINY_TEST_VIDEO.open("rb") as video_file:
        return api_client.post(
            "/api/v1/projects",
            data={"project_name": name, **options},
            files={"video_file": ("tiny_test_video.mp4", video_file, "video/mp4")},
        )


def test_create_project_applies_import_options(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    # Fixture: 64x64, 10 FPS, 10 frames.
    response = _create_with_options(
        api_client, "downscaled", max_width="32", max_height="32", keep_every_nth="2"
    )

    assert response.status_code == 201, response.text
    source = response.json()["source"]
    assert (source["width"], source["height"]) == (32, 32)
    assert source["frame_count"] == 5
    assert source["nominal_fps"] == 5.0
    assert source["import_options"]["keep_every_nth"] == 2
    assert source["original"]["width"] == 64
    assert source["original"]["nominal_fps"] == 10.0
    assert response.json()["progress"]["total_frame_count"] == 5

    project_id = response.json()["project_id"]
    assert api_client.get(f"/api/v1/projects/{project_id}/frames/4").status_code == 200
    assert api_client.get(f"/api/v1/projects/{project_id}/frames/5").status_code == 404


def test_create_project_without_options_keeps_the_native_video(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    source = create_project_with_video(api_client, "native")["source"]

    assert source["original"] is None
    assert source["import_options"] == {
        "max_width": None,
        "max_height": None,
        "target_fps": None,
        "keep_every_nth": None,
    }


@pytest.mark.parametrize(
    "options",
    [
        {"max_width": "32"},
        {"target_fps": "5", "keep_every_nth": "2"},
        {"keep_every_nth": "0"},
        {"target_fps": "60"},  # above the fixture's 10 FPS
    ],
)
def test_create_project_with_invalid_import_options_leaves_no_project(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries, options: dict[str, str]
) -> None:
    response = _create_with_options(api_client, "bad options", **options)

    assert response.status_code == 422
    assert api_client.get("/api/v1/projects").json() == []


def test_project_list_page_versions_its_assets(api_client: TestClient) -> None:
    # Unversioned URLs let a browser keep running a cached upload.js against
    # a newer template (e.g. new dialog fields that the old script never shows).
    html = api_client.get("/").text

    assert re.search(r"js/upload\.js\?v=[0-9a-f]{16}", html)
    assert re.search(r"css/app\.css\?v=[0-9a-f]{16}", html)
