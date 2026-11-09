# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

from __future__ import annotations

from fastapi.testclient import TestClient

from tests.conftest import create_project_with_video
from vistlabel.video.ffmpeg import FfmpegBinaries


def _create_object(api_client: TestClient, project_id: str, label_id: int = 3) -> dict:
    response = api_client.post(
        f"/api/v1/projects/{project_id}/objects",
        json={"label_id": label_id, "label_name": "pipette"},
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_object_lifecycle_create_relabel_deactivate_delete(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_with_video(api_client, "object lifecycle")["project_id"]

    listed = api_client.get(f"/api/v1/projects/{project_id}/objects").json()
    assert listed["objects"] == []
    assert listed["revision"] == 0

    created = _create_object(api_client, project_id)
    assert created["label_name"] == "pipette"
    assert created["active"] is True
    object_id = created["object_id"]

    after_create = api_client.get(f"/api/v1/projects/{project_id}/objects").json()
    assert after_create["revision"] == 1
    assert [o["object_id"] for o in after_create["objects"]] == [object_id]

    relabel_response = api_client.patch(
        f"/api/v1/projects/{project_id}/objects/{object_id}",
        json={"expected_revision": 1, "label_name": "renamed_pipette", "active": False},
    )
    assert relabel_response.status_code == 200
    assert relabel_response.json()["label_name"] == "renamed_pipette"
    assert relabel_response.json()["active"] is False

    stale_response = api_client.patch(
        f"/api/v1/projects/{project_id}/objects/{object_id}",
        json={"expected_revision": 1, "active": True},
    )
    assert stale_response.status_code == 409

    delete_response = api_client.request(
        "DELETE",
        f"/api/v1/projects/{project_id}/objects/{object_id}",
        params={"expected_revision": 2},
    )
    assert delete_response.status_code == 204
    assert api_client.get(f"/api/v1/projects/{project_id}/objects").json()["objects"] == []


def test_locked_object_cannot_be_deleted_until_unlocked(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_with_video(api_client, "object lock")["project_id"]
    created = _create_object(api_client, project_id)
    assert created["locked"] is False
    url = f"/api/v1/projects/{project_id}/objects/{created['object_id']}"

    locked = api_client.patch(url, json={"expected_revision": 1, "locked": True})
    assert locked.status_code == 200, locked.text
    assert locked.json()["locked"] is True

    # Locking doesn't stop other changes, e.g. deactivating the object.
    deactivated = api_client.patch(url, json={"expected_revision": 2, "active": False})
    assert deactivated.status_code == 200, deactivated.text
    assert deactivated.json()["locked"] is True

    refused = api_client.request("DELETE", url, params={"expected_revision": 3})
    assert refused.status_code == 409
    assert "locked" in refused.json()["detail"]
    objects = api_client.get(f"/api/v1/projects/{project_id}/objects").json()
    assert [o["object_id"] for o in objects["objects"]] == [created["object_id"]]
    assert objects["revision"] == 3

    unlocked = api_client.patch(url, json={"expected_revision": 3, "locked": False})
    assert unlocked.json()["locked"] is False
    assert api_client.request("DELETE", url, params={"expected_revision": 4}).status_code == 204


def test_update_unknown_object_returns_404(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_with_video(api_client, "unknown object")["project_id"]

    response = api_client.patch(
        f"/api/v1/projects/{project_id}/objects/obj_doesnotexist",
        json={"expected_revision": 0, "active": False},
    )

    assert response.status_code == 404


def test_segment_produces_a_mask_that_can_be_saved_and_read_back(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_with_video(api_client, "segment workflow")["project_id"]
    object_id = _create_object(api_client, project_id)["object_id"]

    segment_response = api_client.post(
        f"/api/v1/projects/{project_id}/segment",
        json={
            "backend_id": "fake",
            "frame_index": 0,
            "object_id": object_id,
            "points": [{"x": 10, "y": 10, "positive": True}],
        },
    )
    assert segment_response.status_code == 200, segment_response.text
    prediction = segment_response.json()
    assert prediction["object_id"] == object_id
    assert prediction["component_count"] >= 1
    mask_png = prediction["visible_mask_png_base64"]

    get_before = api_client.get(f"/api/v1/projects/{project_id}/frames/0/annotations")
    assert get_before.status_code == 200
    assert get_before.json()["revision"] == 0
    assert get_before.json()["objects"] == []

    patch_response = api_client.patch(
        f"/api/v1/projects/{project_id}/frames/0/annotations",
        json={
            "expected_revision": 0,
            "upsert_object": {
                "object_id": object_id,
                "label_id": 3,
                "visible_mask_png_base64": mask_png,
                "source": "fake_interactive",
                "manually_modified": False,
                "backend_id": "fake",
            },
        },
    )
    assert patch_response.status_code == 200, patch_response.text
    saved = patch_response.json()
    assert saved["revision"] == 1
    assert len(saved["objects"]) == 1
    assert saved["objects"][0]["object_id"] == object_id
    assert saved["objects"][0]["connected_component_count"] >= 1

    get_after = api_client.get(f"/api/v1/projects/{project_id}/frames/0/annotations")
    assert get_after.json() == saved


def test_patch_frame_annotation_with_stale_revision_conflicts(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_with_video(api_client, "revision conflict")["project_id"]

    first = api_client.patch(
        f"/api/v1/projects/{project_id}/frames/0/annotations",
        json={"expected_revision": 0, "review_status": "needs_review"},
    )
    assert first.status_code == 200

    second = api_client.patch(
        f"/api/v1/projects/{project_id}/frames/0/annotations",
        json={"expected_revision": 0, "review_status": "reviewed"},
    )
    assert second.status_code == 409


def test_patch_frame_annotation_rejects_malformed_mask(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_with_video(api_client, "bad mask")["project_id"]
    object_id = _create_object(api_client, project_id)["object_id"]

    response = api_client.patch(
        f"/api/v1/projects/{project_id}/frames/0/annotations",
        json={
            "expected_revision": 0,
            "upsert_object": {
                "object_id": object_id,
                "label_id": 3,
                "visible_mask_png_base64": "not-valid-base64-png",
                "source": "manual_brush",
            },
        },
    )

    assert response.status_code == 422


def test_two_disjoint_points_form_one_multipart_object(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    """Two disconnected visible regions must belong to one
    logical object_id, not be split into two objects."""
    project_id = create_project_with_video(api_client, "multipart object")["project_id"]
    object_id = _create_object(api_client, project_id, label_id=3)["object_id"]

    response = api_client.post(
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
    )

    assert response.status_code == 200
    prediction = response.json()
    assert prediction["object_id"] == object_id
    assert prediction["component_count"] >= 2


def test_segment_with_unknown_backend_returns_404(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_with_video(api_client, "unknown backend")["project_id"]
    object_id = _create_object(api_client, project_id)["object_id"]

    response = api_client.post(
        f"/api/v1/projects/{project_id}/segment",
        json={
            "backend_id": "not_a_real_backend_id",
            "frame_index": 0,
            "object_id": object_id,
            "points": [{"x": 1, "y": 1, "positive": True}],
        },
    )

    assert response.status_code == 404


def test_segment_with_unavailable_backend_returns_409(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    """sam21/sam3/yolo_sam are always registered but report themselves
    unavailable when their model is not configured (as in the test
    settings) -- a *known but unusable* backend is a different, more
    specific error than an unknown backend_id."""
    project_id = create_project_with_video(api_client, "unavailable backend")["project_id"]
    object_id = _create_object(api_client, project_id)["object_id"]

    response = api_client.post(
        f"/api/v1/projects/{project_id}/segment",
        json={
            "backend_id": "sam21",
            "frame_index": 0,
            "object_id": object_id,
            "points": [{"x": 1, "y": 1, "positive": True}],
        },
    )

    assert response.status_code == 409


def test_a_class_leaves_the_list_with_its_last_object(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_with_video(api_client, "class pruning")["project_id"]
    labels_url = f"/api/v1/projects/{project_id}/labels"
    api_client.put(
        labels_url,
        json={
            "labels": [
                {"label_id": 0, "name": "qr"},
                {"label_id": 1, "name": "hand"},
                {"label_id": 2, "name": "tube"},
            ]
        },
    )
    qr_a, qr_b, hand = (
        api_client.post(
            f"/api/v1/projects/{project_id}/objects",
            json={"label_id": label_id, "label_name": name},
        ).json()["object_id"]
        for label_id, name in ((0, "qr"), (0, "qr"), (1, "hand"))
    )

    def names() -> list[str]:
        return [label["name"] for label in api_client.get(labels_url).json()["labels"]]

    def revision() -> int:
        return api_client.get(f"/api/v1/projects/{project_id}/objects").json()["revision"]

    def delete(object_id: str) -> None:
        response = api_client.delete(
            f"/api/v1/projects/{project_id}/objects/{object_id}",
            params={"expected_revision": revision()},
        )
        assert response.status_code == 204, response.text

    delete(qr_a)
    assert names() == ["qr", "hand", "tube"]  # qr_b still uses it
    delete(qr_b)
    # tube never had an object (e.g. declared by an imported dataset): kept.
    assert names() == ["hand", "tube"]

    relabeled = api_client.patch(
        f"/api/v1/projects/{project_id}/objects/{hand}",
        json={"expected_revision": revision(), "label_id": 2, "label_name": "tube"},
    )
    assert relabeled.status_code == 200, relabeled.text
    assert names() == ["tube"]


def test_changing_an_objects_class_relabels_it_on_every_frame(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_with_video(api_client, "class change")["project_id"]
    tube = _create_object(api_client, project_id, label_id=0)["object_id"]
    other = _create_object(api_client, project_id, label_id=0)["object_id"]

    def save_mask(frame_index: int, object_id: str) -> dict:
        mask_png = api_client.post(
            f"/api/v1/projects/{project_id}/segment",
            json={
                "backend_id": "fake",
                "frame_index": frame_index,
                "object_id": object_id,
                "points": [{"x": 10, "y": 10, "positive": True}],
            },
        ).json()["visible_mask_png_base64"]
        url = f"/api/v1/projects/{project_id}/frames/{frame_index}/annotations"
        response = api_client.patch(
            url,
            json={
                "expected_revision": api_client.get(url).json()["revision"],
                "upsert_object": {
                    "object_id": object_id,
                    "label_id": 0,
                    "visible_mask_png_base64": mask_png,
                    "source": "fake_interactive",
                },
            },
        )
        assert response.status_code == 200, response.text
        return response.json()

    save_mask(0, tube)
    save_mask(0, other)
    saved = save_mask(1, tube)
    # An inactive mask comes back on Activate, so it must not keep the old class.
    deactivated = api_client.patch(
        f"/api/v1/projects/{project_id}/frames/1/annotations",
        json={"expected_revision": saved["revision"], "deactivate_object_id": tube},
    )
    assert deactivated.status_code == 200, deactivated.text

    revision = api_client.get(f"/api/v1/projects/{project_id}/objects").json()["revision"]
    relabeled = api_client.patch(
        f"/api/v1/projects/{project_id}/objects/{tube}",
        json={"expected_revision": revision, "label_id": 1, "label_name": "Tube 50ml"},
    )
    assert relabeled.status_code == 200, relabeled.text
    assert relabeled.json()["label_id"] == 1

    frame0 = api_client.get(f"/api/v1/projects/{project_id}/frames/0/annotations").json()
    assert {o["object_id"]: o["label_id"] for o in frame0["objects"]} == {tube: 1, other: 0}
    frame1 = api_client.get(f"/api/v1/projects/{project_id}/frames/1/annotations").json()
    assert [(o["object_id"], o["label_id"]) for o in frame1["inactive_objects"]] == [(tube, 1)]
