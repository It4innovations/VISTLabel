# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Whole-class operations from the Class order list: rename, merge into
another class, delete -- each reaching every object of the class and its
masks on every frame."""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from tests.conftest import create_project_with_video
from vistlabel.annotations import objects_service
from vistlabel.video.ffmpeg import FfmpegBinaries


def _project_with_classes(api_client: TestClient) -> tuple[str, dict[str, str]]:
    """Classes qr (0), tube (1), hand (2); objects qr_a and qr_b, tube, hand,
    each with a mask on frames 0 and 1 except hand (frame 0 only). qr_b is
    inactive on frame 1, and qr_a is in front of tube on frame 0."""
    project_id = create_project_with_video(api_client, "class operations")["project_id"]
    api_client.put(
        f"/api/v1/projects/{project_id}/labels",
        json={
            "labels": [
                {"label_id": 0, "name": "qr"},
                {"label_id": 1, "name": "tube"},
                {"label_id": 2, "name": "hand"},
            ]
        },
    )
    ids: dict[str, str] = {}
    for key, label_id, name in (
        ("qr_a", 0, "qr"),
        ("qr_b", 0, "qr"),
        ("tube", 1, "tube"),
        ("hand", 2, "hand"),
    ):
        response = api_client.post(
            f"/api/v1/projects/{project_id}/objects",
            json={"label_id": label_id, "label_name": name},
        )
        assert response.status_code == 201, response.text
        ids[key] = response.json()["object_id"]

    def patch_frame(frame_index: int, change: dict) -> None:
        url = f"/api/v1/projects/{project_id}/frames/{frame_index}/annotations"
        revision = api_client.get(url).json()["revision"]
        response = api_client.patch(url, json={"expected_revision": revision, **change})
        assert response.status_code == 200, response.text

    for frame_index in (0, 1):
        for key, label_id in (("qr_a", 0), ("qr_b", 0), ("tube", 1), ("hand", 2)):
            if key == "hand" and frame_index == 1:
                continue
            mask_png = api_client.post(
                f"/api/v1/projects/{project_id}/segment",
                json={
                    "backend_id": "fake",
                    "frame_index": frame_index,
                    "object_id": ids[key],
                    "points": [{"x": 10, "y": 10, "positive": True}],
                },
            ).json()["visible_mask_png_base64"]
            patch_frame(
                frame_index,
                {
                    "upsert_object": {
                        "object_id": ids[key],
                        "label_id": label_id,
                        "visible_mask_png_base64": mask_png,
                        "source": "fake_interactive",
                    }
                },
            )
    patch_frame(1, {"deactivate_object_id": ids["qr_b"]})
    patch_frame(
        0,
        {
            "occlusion_edges": [
                {"occluder_object_id": ids["qr_a"], "occluded_object_id": ids["tube"]}
            ]
        },
    )
    return project_id, ids


def _labels(api_client: TestClient, project_id: str) -> list[tuple[int, str]]:
    labels = api_client.get(f"/api/v1/projects/{project_id}/labels").json()["labels"]
    return [(label["label_id"], label["name"]) for label in labels]


def _objects(api_client: TestClient, project_id: str) -> dict[str, tuple[int, str]]:
    objects = api_client.get(f"/api/v1/projects/{project_id}/objects").json()["objects"]
    return {o["object_id"]: (o["label_id"], o["label_name"]) for o in objects}


def _frame(api_client: TestClient, project_id: str, frame_index: int) -> dict:
    return api_client.get(f"/api/v1/projects/{project_id}/frames/{frame_index}/annotations").json()


def test_rename_a_class_renames_its_objects(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id, ids = _project_with_classes(api_client)
    response = api_client.patch(
        f"/api/v1/projects/{project_id}/labels/0", json={"name": "  QR code  "}
    )
    assert response.status_code == 200, response.text
    assert response.json()["object_count"] == 2
    assert _labels(api_client, project_id) == [(0, "QR code"), (1, "tube"), (2, "hand")]
    objects = _objects(api_client, project_id)
    assert objects[ids["qr_a"]] == objects[ids["qr_b"]] == (0, "QR code")
    assert objects[ids["tube"]] == (1, "tube")


def test_rename_to_another_class_name_is_refused(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id, _ids = _project_with_classes(api_client)
    url = f"/api/v1/projects/{project_id}/labels/0"
    assert api_client.patch(url, json={"name": "tube"}).status_code == 409
    assert api_client.patch(url, json={"name": "   "}).status_code == 422
    assert (
        api_client.patch(f"/api/v1/projects/{project_id}/labels/9", json={"name": "x"}).status_code
        == 404
    )
    assert _labels(api_client, project_id) == [(0, "qr"), (1, "tube"), (2, "hand")]


def test_merge_moves_every_object_and_mask_into_the_other_class(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id, ids = _project_with_classes(api_client)
    response = api_client.post(
        f"/api/v1/projects/{project_id}/labels/0/merge", json={"into_label_id": 2}
    )
    assert response.status_code == 200, response.text
    assert (response.json()["object_count"], response.json()["frame_count"]) == (2, 2)
    # hand keeps its place; qr is gone, so hand is now class 1 in exports.
    assert _labels(api_client, project_id) == [(1, "tube"), (2, "hand")]
    objects = _objects(api_client, project_id)
    assert objects[ids["qr_a"]] == objects[ids["qr_b"]] == (2, "hand")

    frame0, frame1 = _frame(api_client, project_id, 0), _frame(api_client, project_id, 1)
    labels0 = {o["object_id"]: o["label_id"] for o in frame0["objects"]}
    assert labels0 == {ids["qr_a"]: 2, ids["qr_b"]: 2, ids["tube"]: 1, ids["hand"]: 2}
    assert [(o["object_id"], o["label_id"]) for o in frame1["inactive_objects"]] == [
        (ids["qr_b"], 2)
    ]


def test_merge_needs_another_existing_class(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id, _ids = _project_with_classes(api_client)
    url = f"/api/v1/projects/{project_id}/labels/0/merge"
    assert api_client.post(url, json={"into_label_id": 0}).status_code == 422
    assert api_client.post(url, json={"into_label_id": 9}).status_code == 404
    assert _labels(api_client, project_id) == [(0, "qr"), (1, "tube"), (2, "hand")]


def test_delete_a_class_deletes_its_objects_and_masks_everywhere(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id, ids = _project_with_classes(api_client)
    response = api_client.delete(f"/api/v1/projects/{project_id}/labels/0")
    assert response.status_code == 200, response.text
    assert (response.json()["object_count"], response.json()["frame_count"]) == (2, 2)
    assert _labels(api_client, project_id) == [(1, "tube"), (2, "hand")]
    assert set(_objects(api_client, project_id)) == {ids["tube"], ids["hand"]}

    frame0, frame1 = _frame(api_client, project_id, 0), _frame(api_client, project_id, 1)
    assert {o["object_id"] for o in frame0["objects"]} == {ids["tube"], ids["hand"]}
    assert frame0["occlusion_edges"] == []  # qr_a was the occluder
    assert {o["object_id"] for o in frame1["objects"]} == {ids["tube"]}
    assert frame1["inactive_objects"] == []

    assert api_client.delete(f"/api/v1/projects/{project_id}/labels/0").status_code == 404


def test_a_class_with_a_locked_object_cannot_be_deleted(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id, ids = _project_with_classes(api_client)
    revision = api_client.get(f"/api/v1/projects/{project_id}/objects").json()["revision"]
    api_client.patch(
        f"/api/v1/projects/{project_id}/objects/{ids['qr_b']}",
        json={"expected_revision": revision, "locked": True},
    )

    response = api_client.delete(f"/api/v1/projects/{project_id}/labels/0")
    assert response.status_code == 409
    assert "locked" in response.json()["detail"]
    # Nothing was deleted, not even the unlocked qr_a or its masks.
    assert _labels(api_client, project_id) == [(0, "qr"), (1, "tube"), (2, "hand")]
    assert {ids["qr_a"], ids["qr_b"]} <= set(_objects(api_client, project_id))
    frame0 = _frame(api_client, project_id, 0)
    assert ids["qr_a"] in {o["object_id"] for o in frame0["objects"]}
    assert len(frame0["occlusion_edges"]) == 1


def _delete_object(api_client: TestClient, project_id: str, object_id: str) -> None:
    revision = api_client.get(f"/api/v1/projects/{project_id}/objects").json()["revision"]
    response = api_client.delete(
        f"/api/v1/projects/{project_id}/objects/{object_id}",
        params={"expected_revision": revision},
    )
    assert response.status_code == 204, response.text


def _restore(api_client: TestClient, project_id: str, *object_ids: str) -> httpx.Response:
    return api_client.post(
        f"/api/v1/projects/{project_id}/objects/restore", json={"object_ids": list(object_ids)}
    )


def _frame_objects(api_client: TestClient, project_id: str, frame_index: int) -> dict:
    frame = _frame(api_client, project_id, frame_index)
    return {
        "objects": {o["object_id"]: o for o in frame["objects"]},
        "inactive": {o["object_id"] for o in frame["inactive_objects"]},
        "edges": [
            (e["occluder_object_id"], e["occluded_object_id"]) for e in frame["occlusion_edges"]
        ],
    }


def test_restore_undoes_deleting_an_object(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id, ids = _project_with_classes(api_client)
    before = [_frame_objects(api_client, project_id, i) for i in (0, 1)]

    # tube is its class's only object, so the class leaves with it.
    _delete_object(api_client, project_id, ids["tube"])
    assert _labels(api_client, project_id) == [(0, "qr"), (2, "hand")]
    assert ids["tube"] not in _frame_objects(api_client, project_id, 0)["objects"]

    response = _restore(api_client, project_id, ids["tube"])
    assert response.status_code == 200, response.text
    assert response.json()["frame_count"] == 2
    assert [o["object_id"] for o in response.json()["objects"]] == [ids["tube"]]
    # The class is back in its old place, with its old id.
    assert _labels(api_client, project_id) == [(0, "qr"), (1, "tube"), (2, "hand")]
    assert _objects(api_client, project_id)[ids["tube"]] == (1, "tube")
    after = [_frame_objects(api_client, project_id, i) for i in (0, 1)]
    assert after == before  # masks, and the qr_a-in-front-of-tube edge

    # Undone once; there is nothing left to restore.
    assert _restore(api_client, project_id, ids["tube"]).status_code == 409
    _delete_object(api_client, project_id, ids["tube"])
    assert _restore(api_client, project_id, ids["tube"]).status_code == 200
    assert _restore(api_client, project_id, ids["hand"]).status_code == 409


def test_restore_undoes_deleting_a_whole_class(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id, ids = _project_with_classes(api_client)
    before = [_frame_objects(api_client, project_id, i) for i in (0, 1)]

    deleted = api_client.delete(f"/api/v1/projects/{project_id}/labels/0").json()
    assert set(deleted["deleted_object_ids"]) == {ids["qr_a"], ids["qr_b"]}

    response = _restore(api_client, project_id, *deleted["deleted_object_ids"])
    assert response.status_code == 200, response.text
    assert _labels(api_client, project_id) == [(0, "qr"), (1, "tube"), (2, "hand")]
    after = [_frame_objects(api_client, project_id, i) for i in (0, 1)]
    assert after == before  # incl. qr_b still inactive on frame 1


def test_restore_skips_edges_to_objects_gone_since(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id, ids = _project_with_classes(api_client)
    _delete_object(api_client, project_id, ids["qr_a"])
    _delete_object(api_client, project_id, ids["tube"])

    assert _restore(api_client, project_id, ids["tube"]).status_code == 200
    assert _frame_objects(api_client, project_id, 0)["edges"] == []
    assert _restore(api_client, project_id, ids["qr_a"]).status_code == 200
    assert _frame_objects(api_client, project_id, 0)["edges"] == [(ids["qr_a"], ids["tube"])]


def test_restore_gives_a_new_id_to_a_class_whose_id_was_taken(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id, ids = _project_with_classes(api_client)
    _delete_object(api_client, project_id, ids["tube"])
    # A class made after the delete takes the freed label_id 1.
    labels = api_client.get(f"/api/v1/projects/{project_id}/labels").json()["labels"]
    api_client.put(
        f"/api/v1/projects/{project_id}/labels",
        json={"labels": [*labels, {"label_id": 1, "name": "glove"}]},
    )

    assert _restore(api_client, project_id, ids["tube"]).status_code == 200
    assert _labels(api_client, project_id) == [(0, "qr"), (3, "tube"), (2, "hand"), (1, "glove")]
    assert _objects(api_client, project_id)[ids["tube"]] == (3, "tube")
    for frame_index in (0, 1):
        tube = _frame_objects(api_client, project_id, frame_index)["objects"][ids["tube"]]
        assert tube["label_id"] == 3


def test_restore_refuses_ids_with_nothing_kept(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id, _ids = _project_with_classes(api_client)
    for object_id in ("obj_neverexisted", "../project", "..\\project", "a/b"):
        response = _restore(api_client, project_id, object_id)
        assert response.status_code == 404, object_id
        assert "nothing to undo" in response.json()["detail"]


def test_restore_after_a_delete_that_stopped_partway_and_was_retried(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_id, ids = _project_with_classes(api_client)
    before = [_frame_objects(api_client, project_id, i) for i in (0, 1)]
    real_apply = objects_service.apply_frame_annotation
    calls = 0

    def apply_then_stop(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("disk went away")
        return real_apply(*args, **kwargs)

    # Frame 0 is stripped, then the delete stops before frame 1.
    monkeypatch.setattr(objects_service, "apply_frame_annotation", apply_then_stop)
    with pytest.raises(OSError):
        _delete_object(api_client, project_id, ids["tube"])
    monkeypatch.undo()
    assert ids["tube"] not in _frame_objects(api_client, project_id, 0)["objects"]

    _delete_object(api_client, project_id, ids["tube"])  # the retry finishes it
    assert _restore(api_client, project_id, ids["tube"]).status_code == 200
    assert [_frame_objects(api_client, project_id, i) for i in (0, 1)] == before
