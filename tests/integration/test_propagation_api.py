# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

from __future__ import annotations

import base64
import io
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tests.conftest import create_project_with_video
from vistlabel.segmentation.fake import FakeBackend
from vistlabel.video.ffmpeg import FfmpegBinaries


def _create_object(api_client: TestClient, project_id: str) -> str:
    response = api_client.post(
        f"/api/v1/projects/{project_id}/objects",
        json={"label_id": 3, "label_name": "pipette"},
    )
    assert response.status_code == 201, response.text
    return response.json()["object_id"]


def _mask_png_base64(height: int = 64, width: int = 64) -> str:
    """A small solid-square mask PNG matching the tiny test video's frame
    size, for saving an annotation without going through /segment."""
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[10:30, 10:30] = 255
    buffer = io.BytesIO()
    Image.fromarray(mask).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _segment_and_save(api_client: TestClient, project_id: str, object_id: str, x: int) -> None:
    mask_png = api_client.post(
        f"/api/v1/projects/{project_id}/segment",
        json={
            "backend_id": "fake",
            "frame_index": 0,
            "object_id": object_id,
            "points": [{"x": x, "y": 10, "positive": True}],
        },
    ).json()["visible_mask_png_base64"]
    revision = api_client.get(f"/api/v1/projects/{project_id}/frames/0/annotations").json()[
        "revision"
    ]
    save_response = api_client.patch(
        f"/api/v1/projects/{project_id}/frames/0/annotations",
        json={
            "expected_revision": revision,
            "upsert_object": {
                "object_id": object_id,
                "label_id": 3,
                "visible_mask_png_base64": mask_png,
                "source": "fake_interactive",
                "backend_id": "fake",
            },
        },
    )
    assert save_response.status_code == 200, save_response.text


def _wait_for_job(
    api_client: TestClient, project_id: str, job_id: str, timeout: float = 20.0
) -> dict:
    deadline = time.monotonic() + timeout
    terminal = {"succeeded", "failed", "cancelled"}
    while time.monotonic() < deadline:
        response = api_client.get(f"/api/v1/projects/{project_id}/jobs/{job_id}")
        assert response.status_code == 200
        record = response.json()
        if record["status"] in terminal:
            return record
        time.sleep(0.02)
    raise TimeoutError(f"job {job_id} did not finish within {timeout}s")


def test_propagate_runs_as_a_background_job_and_writes_frame_annotations(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_with_video(api_client, "propagation test")["project_id"]
    object_id = _create_object(api_client, project_id)

    segment_response = api_client.post(
        f"/api/v1/projects/{project_id}/segment",
        json={
            "backend_id": "fake",
            "frame_index": 0,
            "object_id": object_id,
            "points": [{"x": 10, "y": 10, "positive": True}],
        },
    )
    assert segment_response.status_code == 200
    mask_png = segment_response.json()["visible_mask_png_base64"]
    save_response = api_client.patch(
        f"/api/v1/projects/{project_id}/frames/0/annotations",
        json={
            "expected_revision": 0,
            "upsert_object": {
                "object_id": object_id,
                "label_id": 3,
                "visible_mask_png_base64": mask_png,
                "source": "fake_interactive",
                "backend_id": "fake",
            },
        },
    )
    assert save_response.status_code == 200

    propagate_response = api_client.post(
        f"/api/v1/projects/{project_id}/propagate",
        json={
            "backend_id": "fake",
            "object_ids": [object_id],
            "start_frame": 0,
            "end_frame": 5,
            "direction": "forward",
        },
    )
    assert propagate_response.status_code == 202, propagate_response.text
    job_id = propagate_response.json()["job_id"]

    record = _wait_for_job(api_client, project_id, job_id)
    assert record["status"] == "succeeded"
    assert record["progress_fraction"] == 1.0
    assert record["job_type"] == "propagate"

    for frame_index in range(0, 6):
        annotation = api_client.get(
            f"/api/v1/projects/{project_id}/frames/{frame_index}/annotations"
        ).json()
        object_ids = [o["object_id"] for o in annotation["objects"]]
        assert object_id in object_ids, f"frame {frame_index} missing propagated object"


def test_propagate_tracks_multiple_objects_in_one_run(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    """The UI's Run button submits every active object with a saved mask on
    the start frame, not just the selected one; one job must carry all of
    them through and write each object's mask on every covered frame."""
    project_id = create_project_with_video(api_client, "propagate multi object")["project_id"]
    first_object = _create_object(api_client, project_id)
    second_object = _create_object(api_client, project_id)
    _segment_and_save(api_client, project_id, first_object, x=10)
    _segment_and_save(api_client, project_id, second_object, x=40)

    response = api_client.post(
        f"/api/v1/projects/{project_id}/propagate",
        json={
            "backend_id": "fake",
            "object_ids": [first_object, second_object],
            "start_frame": 0,
            "end_frame": 5,
            "direction": "forward",
        },
    )
    assert response.status_code == 202, response.text
    record = _wait_for_job(api_client, project_id, response.json()["job_id"])
    assert record["status"] == "succeeded"

    for frame_index in range(0, 6):
        annotation = api_client.get(
            f"/api/v1/projects/{project_id}/frames/{frame_index}/annotations"
        ).json()
        object_ids = {o["object_id"] for o in annotation["objects"]}
        assert {first_object, second_object} <= object_ids, (
            f"frame {frame_index} is missing one of the propagated objects"
        )


def test_propagate_reseeds_backend_session_from_saved_masks(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    """A mask saved directly to the frame annotation (no /segment call in
    this process, e.g. annotated before a server restart) must still be
    propagatable: start_propagation re-seeds the backend session from the
    canonical saved mask, same as tracking-run resume does."""
    project_id = create_project_with_video(api_client, "propagate reseed")["project_id"]
    object_id = _create_object(api_client, project_id)
    save_response = api_client.patch(
        f"/api/v1/projects/{project_id}/frames/0/annotations",
        json={
            "expected_revision": 0,
            "upsert_object": {
                "object_id": object_id,
                "label_id": 3,
                "visible_mask_png_base64": _mask_png_base64(),
                "source": "manual",
                "backend_id": "fake",
            },
        },
    )
    assert save_response.status_code == 200, save_response.text

    response = api_client.post(
        f"/api/v1/projects/{project_id}/propagate",
        json={
            "backend_id": "fake",
            "object_ids": [object_id],
            "start_frame": 0,
            "end_frame": 3,
            "direction": "forward",
        },
    )
    assert response.status_code == 202, response.text
    record = _wait_for_job(api_client, project_id, response.json()["job_id"])
    assert record["status"] == "succeeded", record["error"]

    for frame_index in range(0, 4):
        annotation = api_client.get(
            f"/api/v1/projects/{project_id}/frames/{frame_index}/annotations"
        ).json()
        object_ids = [o["object_id"] for o in annotation["objects"]]
        assert object_id in object_ids, f"frame {frame_index} missing propagated object"


@pytest.mark.parametrize(
    ("source", "forgets"),
    [
        ("manual_brush", True),
        ("manual_eraser", True),
        ("fake_interactive", True),
        ("fake_propagation", False),
    ],
)
def test_propagate_forgets_the_object_only_for_a_fresh_seed(
    api_client: TestClient,
    ffmpeg_binaries: FfmpegBinaries,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    forgets: bool,
) -> None:
    """A correction replaces what the tracker remembers; continuing from a
    propagated mask keeps that memory, or the run's original seed is lost."""
    forgotten: list[str] = []
    original = FakeBackend.forget_object

    def spy(self: FakeBackend, session_id: str, object_id: str) -> None:
        forgotten.append(object_id)
        original(self, session_id, object_id)

    monkeypatch.setattr(FakeBackend, "forget_object", spy)

    project_id = create_project_with_video(api_client, "propagate fresh seed")["project_id"]
    object_id = _create_object(api_client, project_id)
    save_response = api_client.patch(
        f"/api/v1/projects/{project_id}/frames/0/annotations",
        json={
            "expected_revision": 0,
            "upsert_object": {
                "object_id": object_id,
                "label_id": 3,
                "visible_mask_png_base64": _mask_png_base64(),
                "source": source,
                "backend_id": "fake",
            },
        },
    )
    assert save_response.status_code == 200, save_response.text

    response = api_client.post(
        f"/api/v1/projects/{project_id}/propagate",
        json={
            "backend_id": "fake",
            "object_ids": [object_id],
            "start_frame": 0,
            "end_frame": 1,
            "direction": "forward",
        },
    )
    assert response.status_code == 202, response.text
    record = _wait_for_job(api_client, project_id, response.json()["job_id"])
    assert record["status"] == "succeeded", record["error"]

    assert forgotten == ([object_id] if forgets else [])


def test_propagate_with_unavailable_backend_returns_409(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_with_video(api_client, "propagate unavailable backend")[
        "project_id"
    ]
    object_id = _create_object(api_client, project_id)

    response = api_client.post(
        f"/api/v1/projects/{project_id}/propagate",
        json={
            "backend_id": "sam21",
            "object_ids": [object_id],
            "start_frame": 0,
            "end_frame": 2,
            "direction": "forward",
        },
    )

    assert response.status_code == 409


def test_propagate_with_unknown_object_returns_404(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_with_video(api_client, "propagate unknown object")["project_id"]

    response = api_client.post(
        f"/api/v1/projects/{project_id}/propagate",
        json={
            "backend_id": "fake",
            "object_ids": ["obj_doesnotexist"],
            "start_frame": 0,
            "end_frame": 2,
            "direction": "forward",
        },
    )

    assert response.status_code == 404


def test_propagate_without_prior_mask_on_start_frame_fails_job_not_request(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    """FakeBackend.propagate requires segment_interactive to have run on
    start_frame first; that failure happens inside the background job, so
    the HTTP request itself still succeeds with 202 and the job ends up
    FAILED with the reason recorded -- not a 4xx/5xx at request time."""
    project_id = create_project_with_video(api_client, "propagate no prior mask")["project_id"]
    object_id = _create_object(api_client, project_id)

    response = api_client.post(
        f"/api/v1/projects/{project_id}/propagate",
        json={
            "backend_id": "fake",
            "object_ids": [object_id],
            "start_frame": 0,
            "end_frame": 2,
            "direction": "forward",
        },
    )
    assert response.status_code == 202
    job_id = response.json()["job_id"]

    record = _wait_for_job(api_client, project_id, job_id)
    assert record["status"] == "failed"
    assert record["error"]


def test_get_unknown_job_returns_404(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_with_video(api_client, "unknown job")["project_id"]

    response = api_client.get(f"/api/v1/projects/{project_id}/jobs/job_doesnotexist")

    assert response.status_code == 404


def test_cancel_after_completion_is_a_harmless_no_op(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_with_video(api_client, "cancel after completion")["project_id"]
    object_id = _create_object(api_client, project_id)
    mask_png = api_client.post(
        f"/api/v1/projects/{project_id}/segment",
        json={
            "backend_id": "fake",
            "frame_index": 0,
            "object_id": object_id,
            "points": [{"x": 5, "y": 5, "positive": True}],
        },
    ).json()["visible_mask_png_base64"]
    api_client.patch(
        f"/api/v1/projects/{project_id}/frames/0/annotations",
        json={
            "expected_revision": 0,
            "upsert_object": {
                "object_id": object_id,
                "label_id": 3,
                "visible_mask_png_base64": mask_png,
                "source": "fake_interactive",
                "backend_id": "fake",
            },
        },
    )

    job_id = api_client.post(
        f"/api/v1/projects/{project_id}/propagate",
        json={
            "backend_id": "fake",
            "object_ids": [object_id],
            "start_frame": 0,
            "end_frame": 1,
            "direction": "forward",
        },
    ).json()["job_id"]
    _wait_for_job(api_client, project_id, job_id)

    cancel_response = api_client.post(f"/api/v1/projects/{project_id}/jobs/{job_id}/cancel")

    assert cancel_response.status_code == 200
    assert cancel_response.json()["status"] == "succeeded"
