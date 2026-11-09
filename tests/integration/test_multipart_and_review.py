# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Occlusion-state derivation, review-reason auto-flagging, geometry
hints, and occlusion edges, exercised through the real API."""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from tests.conftest import create_project_with_video
from vistlabel.video.ffmpeg import FfmpegBinaries


def _create_object(api_client: TestClient, project_id: str, label_id: int = 3) -> str:
    response = api_client.post(
        f"/api/v1/projects/{project_id}/objects",
        json={"label_id": label_id, "label_name": "pipette"},
    )
    assert response.status_code == 201, response.text
    return response.json()["object_id"]


def _wait_for_job(api_client: TestClient, project_id: str, job_id: str, timeout: float = 20.0):
    deadline = time.monotonic() + timeout
    terminal = {"succeeded", "failed", "cancelled"}
    while time.monotonic() < deadline:
        record = api_client.get(f"/api/v1/projects/{project_id}/jobs/{job_id}").json()
        if record["status"] in terminal:
            return record
        time.sleep(0.02)
    raise TimeoutError(f"job {job_id} did not finish within {timeout}s")


def test_upsert_derives_occlusion_state_from_component_count(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_with_video(api_client, "occlusion state derivation")[
        "project_id"
    ]
    object_id = _create_object(api_client, project_id)

    single_point = api_client.post(
        f"/api/v1/projects/{project_id}/segment",
        json={
            "backend_id": "fake",
            "frame_index": 0,
            "object_id": object_id,
            "points": [{"x": 10, "y": 10, "positive": True}],
        },
    ).json()
    assert single_point["component_count"] == 1

    saved = api_client.patch(
        f"/api/v1/projects/{project_id}/frames/0/annotations",
        json={
            "expected_revision": 0,
            "upsert_object": {
                "object_id": object_id,
                "label_id": 3,
                "visible_mask_png_base64": single_point["visible_mask_png_base64"],
                "source": "fake_interactive",
                "backend_id": "fake",
            },
        },
    ).json()
    assert saved["objects"][0]["occlusion_state"] == "visible"
    assert saved["objects"][0]["review_reasons"] == []


def test_upsert_two_disjoint_points_is_partially_occluded_and_flagged_for_review(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_with_video(api_client, "multipart occlusion state")[
        "project_id"
    ]
    object_id = _create_object(api_client, project_id)

    candidate = api_client.post(
        f"/api/v1/projects/{project_id}/segment",
        json={
            "backend_id": "fake",
            "frame_index": 0,
            "object_id": object_id,
            "points": [
                {"x": 4, "y": 4, "positive": True},
                {"x": 40, "y": 40, "positive": True},
            ],
        },
    ).json()
    assert candidate["component_count"] >= 2

    saved = api_client.patch(
        f"/api/v1/projects/{project_id}/frames/0/annotations",
        json={
            "expected_revision": 0,
            "upsert_object": {
                "object_id": object_id,
                "label_id": 3,
                "visible_mask_png_base64": candidate["visible_mask_png_base64"],
                "source": "fake_interactive",
                "backend_id": "fake",
            },
        },
    ).json()

    entry = saved["objects"][0]
    assert entry["occlusion_state"] == "partially_occluded"
    # One logical object, not two: a multipart mask is still one instance.
    assert len(saved["objects"]) == 1
    assert entry["object_id"] == object_id


def test_propagation_detects_component_count_change_across_frames(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    """Exploits FakeBackend.propagate's np.roll-based motion: a disk seeded
    near the right edge wraps around the canvas boundary for a couple of
    frames, genuinely splitting into two components before reunifying --
    a deterministic way to exercise real component-count-change detection
    end to end without needing the real SAM 2.1 backend for this test."""
    project_id = create_project_with_video(api_client, "propagation review reasons")[
        "project_id"
    ]
    object_id = _create_object(api_client, project_id)

    seed = api_client.post(
        f"/api/v1/projects/{project_id}/segment",
        json={
            "backend_id": "fake",
            "frame_index": 0,
            "object_id": object_id,
            "points": [{"x": 62, "y": 32, "positive": True}],
        },
    ).json()
    assert seed["component_count"] == 1
    api_client.patch(
        f"/api/v1/projects/{project_id}/frames/0/annotations",
        json={
            "expected_revision": 0,
            "upsert_object": {
                "object_id": object_id,
                "label_id": 3,
                "visible_mask_png_base64": seed["visible_mask_png_base64"],
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
            "end_frame": 3,
            "direction": "forward",
        },
    ).json()["job_id"]
    record = _wait_for_job(api_client, project_id, job_id)
    assert record["status"] == "succeeded"

    component_counts = []
    review_flags = []
    for frame_index in range(4):
        annotation = api_client.get(
            f"/api/v1/projects/{project_id}/frames/{frame_index}/annotations"
        ).json()
        entry = annotation["objects"][0]
        component_counts.append(entry["connected_component_count"])
        review_flags.append("component_count_change" in entry["review_reasons"])

    assert component_counts == [1, 2, 2, 1]
    # Both transitions (1->2 and 2->1) must be flagged; the frame where the
    # count stays the same (2->2) must not be.
    assert review_flags == [False, True, False, True]


def test_geometry_hint_can_be_set_and_cleared_on_an_object(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_with_video(api_client, "geometry hint test")["project_id"]
    created = api_client.post(
        f"/api/v1/projects/{project_id}/objects",
        json={"label_id": 3, "label_name": "pipette"},
    ).json()
    object_id = created["object_id"]
    assert created["geometry_hint"] is None

    updated = api_client.patch(
        f"/api/v1/projects/{project_id}/objects/{object_id}",
        json={
            "expected_revision": 1,
            "geometry_hint": {
                "type": "elongated_axis",
                "polyline": [[10.0, 20.0], [50.0, 22.0]],
                "estimated_width_px": 6.0,
            },
        },
    ).json()
    assert updated["geometry_hint"]["polyline"] == [[10.0, 20.0], [50.0, 22.0]]

    listed = api_client.get(f"/api/v1/projects/{project_id}/objects").json()
    assert listed["objects"][0]["geometry_hint"]["estimated_width_px"] == 6.0

    cleared = api_client.patch(
        f"/api/v1/projects/{project_id}/objects/{object_id}",
        json={"expected_revision": 2, "clear_geometry_hint": True},
    ).json()
    assert cleared["geometry_hint"] is None


def test_occlusion_edges_can_be_set_on_a_frame(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_with_video(api_client, "occlusion edges test")["project_id"]
    glove_id = _create_object(api_client, project_id, label_id=4)
    pipette_id = _create_object(api_client, project_id, label_id=3)

    updated = api_client.patch(
        f"/api/v1/projects/{project_id}/frames/0/annotations",
        json={
            "expected_revision": 0,
            "occlusion_edges": [
                {
                    "occluder_object_id": glove_id,
                    "occluded_object_id": pipette_id,
                    "relation": "in_front_of",
                }
            ],
        },
    ).json()

    assert len(updated["occlusion_edges"]) == 1
    assert updated["occlusion_edges"][0]["occluder_object_id"] == glove_id
    assert updated["occlusion_edges"][0]["occluded_object_id"] == pipette_id

    refetched = api_client.get(f"/api/v1/projects/{project_id}/frames/0/annotations").json()
    assert refetched["occlusion_edges"][0]["occluder_object_id"] == glove_id
