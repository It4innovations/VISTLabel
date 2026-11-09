# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

from __future__ import annotations

from fastapi.testclient import TestClient

from tests.conftest import create_project_with_video
from vistlabel.video.ffmpeg import FfmpegBinaries


def _create_object(api_client: TestClient, project_id: str) -> str:
    response = api_client.post(
        f"/api/v1/projects/{project_id}/objects",
        json={"label_id": 3, "label_name": "pipette"},
    )
    assert response.status_code == 201, response.text
    return response.json()["object_id"]


def _save_mask(api_client: TestClient, project_id: str, frame_index: int, object_id: str) -> dict:
    candidate = api_client.post(
        f"/api/v1/projects/{project_id}/segment",
        json={
            "backend_id": "fake",
            "frame_index": frame_index,
            "object_id": object_id,
            "points": [{"x": 20, "y": 20, "positive": True}],
        },
    ).json()
    response = api_client.patch(
        f"/api/v1/projects/{project_id}/frames/{frame_index}/annotations",
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
    )
    assert response.status_code == 200, response.text
    return response.json()


def _progress(api_client: TestClient, project_id: str) -> dict:
    projects = api_client.get("/api/v1/projects").json()
    return next(p for p in projects if p["project_id"] == project_id)["progress"]


def test_project_list_counts_annotated_and_reviewed_frames(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_with_video(api_client, "progress counts")["project_id"]
    object_id = _create_object(api_client, project_id)
    assert _progress(api_client, project_id)["annotated_frame_count"] == 0

    _save_mask(api_client, project_id, 0, object_id)
    frame_1 = _save_mask(api_client, project_id, 1, object_id)
    reviewed = api_client.patch(
        f"/api/v1/projects/{project_id}/frames/1/annotations",
        json={"expected_revision": frame_1["revision"], "review_status": "reviewed"},
    )
    assert reviewed.status_code == 200, reviewed.text
    # Reviewed with nothing on it: counts as reviewed, not annotated.
    empty_reviewed = api_client.patch(
        f"/api/v1/projects/{project_id}/frames/2/annotations",
        json={"expected_revision": 0, "review_status": "reviewed"},
    )
    assert empty_reviewed.status_code == 200, empty_reviewed.text

    progress = _progress(api_client, project_id)
    assert progress["annotated_frame_count"] == 2
    assert progress["reviewed_frame_count"] == 2

    page = api_client.get("/")
    assert "2 annotated" in page.text
    assert "2 reviewed" in page.text


def test_deactivate_and_activate_round_trip_the_mask(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_with_video(api_client, "deactivate activate")["project_id"]
    object_id = _create_object(api_client, project_id)
    saved = _save_mask(api_client, project_id, 0, object_id)
    url = f"/api/v1/projects/{project_id}/frames/0/annotations"

    inactive = api_client.patch(
        url, json={"expected_revision": saved["revision"], "deactivate_object_id": object_id}
    ).json()
    assert inactive["objects"] == []
    assert [o["object_id"] for o in inactive["inactive_objects"]] == [object_id]
    assert _progress(api_client, project_id)["annotated_frame_count"] == 0
    # Deactivating twice is an error, not a silent no-op.
    again = api_client.patch(
        url, json={"expected_revision": inactive["revision"], "deactivate_object_id": object_id}
    )
    assert again.status_code == 422

    active = api_client.patch(
        url, json={"expected_revision": inactive["revision"], "activate_object_id": object_id}
    ).json()
    assert active["inactive_objects"] == []
    assert active["objects"][0]["visible_mask_rle"] == saved["objects"][0]["visible_mask_rle"]
    not_inactive = api_client.patch(
        url, json={"expected_revision": active["revision"], "activate_object_id": object_id}
    )
    assert not_inactive.status_code == 422


def test_a_new_mask_supersedes_an_inactive_one(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_with_video(api_client, "deactivate then segment")["project_id"]
    object_id = _create_object(api_client, project_id)
    saved = _save_mask(api_client, project_id, 0, object_id)
    url = f"/api/v1/projects/{project_id}/frames/0/annotations"
    inactive = api_client.patch(
        url, json={"expected_revision": saved["revision"], "deactivate_object_id": object_id}
    ).json()

    candidate = api_client.post(
        f"/api/v1/projects/{project_id}/segment",
        json={
            "backend_id": "fake",
            "frame_index": 0,
            "object_id": object_id,
            "points": [{"x": 40, "y": 40, "positive": True}],
        },
    ).json()
    resegmented = api_client.patch(
        url,
        json={
            "expected_revision": inactive["revision"],
            "upsert_object": {
                "object_id": object_id,
                "label_id": 3,
                "visible_mask_png_base64": candidate["visible_mask_png_base64"],
                "source": "fake_interactive",
            },
        },
    ).json()
    assert [o["object_id"] for o in resegmented["objects"]] == [object_id]
    assert resegmented["inactive_objects"] == []


def test_deleted_frame_leaves_counts_and_queue(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_with_video(api_client, "delete frame")["project_id"]
    object_id = _create_object(api_client, project_id)
    saved = _save_mask(api_client, project_id, 0, object_id)
    url = f"/api/v1/projects/{project_id}/frames/0/annotations"
    flagged = api_client.patch(
        url, json={"expected_revision": saved["revision"], "review_status": "needs_review"}
    ).json()
    queue_url = f"/api/v1/projects/{project_id}/review-queue"
    assert [e["frame_index"] for e in api_client.get(queue_url).json()] == [0]

    deleted = api_client.patch(
        url, json={"expected_revision": flagged["revision"], "excluded": True}
    ).json()
    assert deleted["excluded"] is True
    assert deleted["objects"], "deleting a frame keeps its masks for Restore"
    assert api_client.get(queue_url).json() == []
    assert _progress(api_client, project_id)["annotated_frame_count"] == 0

    restored = api_client.patch(
        url, json={"expected_revision": deleted["revision"], "excluded": False}
    ).json()
    assert restored["excluded"] is False
    assert _progress(api_client, project_id)["annotated_frame_count"] == 1


def test_mark_all_annotated_frames_reviewed(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_with_video(api_client, "mark all reviewed")["project_id"]
    object_id = _create_object(api_client, project_id)
    _save_mask(api_client, project_id, 0, object_id)
    frame_1 = _save_mask(api_client, project_id, 1, object_id)
    api_client.patch(
        f"/api/v1/projects/{project_id}/frames/1/annotations",
        json={"expected_revision": frame_1["revision"], "review_status": "needs_review"},
    )
    frame_2 = _save_mask(api_client, project_id, 2, object_id)
    api_client.patch(  # deleted frames are left alone
        f"/api/v1/projects/{project_id}/frames/2/annotations",
        json={"expected_revision": frame_2["revision"], "excluded": True},
    )
    api_client.patch(  # flagged but empty: not annotated, so not touched
        f"/api/v1/projects/{project_id}/frames/3/annotations",
        json={"expected_revision": 0, "review_status": "needs_review"},
    )
    url = f"/api/v1/projects/{project_id}/frames/mark-annotated-reviewed"

    preview = api_client.post(url, json={"dry_run": True}).json()
    assert preview == {"frame_count": 2, "needs_review_count": 1}
    assert _progress(api_client, project_id)["reviewed_frame_count"] == 0

    assert api_client.post(url, json={}).json() == preview
    progress = api_client.get(f"/api/v1/projects/{project_id}/progress").json()
    assert progress["reviewed_frame_count"] == 2
    assert progress["annotated_frame_count"] == 2
    statuses = {
        i: api_client.get(f"/api/v1/projects/{project_id}/frames/{i}/annotations").json()[
            "review_status"
        ]
        for i in range(4)
    }
    assert statuses == {0: "reviewed", 1: "reviewed", 2: "unreviewed", 3: "needs_review"}
    assert api_client.post(url, json={"dry_run": True}).json()["frame_count"] == 0


def test_deleting_an_object_removes_it_from_every_frame(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_with_video(api_client, "delete object")["project_id"]
    doomed = _create_object(api_client, project_id)
    kept = _create_object(api_client, project_id)
    _save_mask(api_client, project_id, 0, doomed)
    frame_1 = _save_mask(api_client, project_id, 1, doomed)
    # Frame 1: the doomed object inactive, the kept one visible, and an
    # occlusion edge between them.
    frame_1 = api_client.patch(
        f"/api/v1/projects/{project_id}/frames/1/annotations",
        json={"expected_revision": frame_1["revision"], "deactivate_object_id": doomed},
    ).json()
    candidate = api_client.post(
        f"/api/v1/projects/{project_id}/segment",
        json={
            "backend_id": "fake",
            "frame_index": 1,
            "object_id": kept,
            "points": [{"x": 50, "y": 50, "positive": True}],
        },
    ).json()
    frame_1 = api_client.patch(
        f"/api/v1/projects/{project_id}/frames/1/annotations",
        json={
            "expected_revision": frame_1["revision"],
            "upsert_object": {
                "object_id": kept,
                "label_id": 3,
                "visible_mask_png_base64": candidate["visible_mask_png_base64"],
                "source": "fake_interactive",
            },
            "occlusion_edges": [{"occluder_object_id": kept, "occluded_object_id": doomed}],
        },
    ).json()

    objects_revision = api_client.get(f"/api/v1/projects/{project_id}/objects").json()["revision"]
    deleted = api_client.request(
        "DELETE",
        f"/api/v1/projects/{project_id}/objects/{doomed}",
        params={"expected_revision": objects_revision},
    )
    assert deleted.status_code == 204, deleted.text

    frame_0 = api_client.get(f"/api/v1/projects/{project_id}/frames/0/annotations").json()
    assert frame_0["objects"] == []
    frame_1 = api_client.get(f"/api/v1/projects/{project_id}/frames/1/annotations").json()
    assert [o["object_id"] for o in frame_1["objects"]] == [kept]
    assert frame_1["inactive_objects"] == []
    assert frame_1["occlusion_edges"] == []
    assert _progress(api_client, project_id)["annotated_frame_count"] == 1


def test_removing_an_object_on_one_frame_keeps_it_on_others(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_with_video(api_client, "remove on frame")["project_id"]
    object_id = _create_object(api_client, project_id)
    _save_mask(api_client, project_id, 0, object_id)
    frame_1 = _save_mask(api_client, project_id, 1, object_id)

    inactive = api_client.patch(
        f"/api/v1/projects/{project_id}/frames/1/annotations",
        json={"expected_revision": frame_1["revision"], "remove_object_id": object_id},
    )
    assert inactive.status_code == 200, inactive.text
    assert inactive.json()["objects"] == []

    frame_0 = api_client.get(f"/api/v1/projects/{project_id}/frames/0/annotations").json()
    assert [o["object_id"] for o in frame_0["objects"]] == [object_id]
    objects = api_client.get(f"/api/v1/projects/{project_id}/objects").json()["objects"]
    assert [o["object_id"] for o in objects] == [object_id]
    assert _progress(api_client, project_id)["annotated_frame_count"] == 1
