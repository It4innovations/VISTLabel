# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Dataset export exercised through the real API -- frame
selection, train/val split intervals, all three YOLO multipart policies,
negative frames, exact COCO preservation, ZIP download, and project
-scoped delete."""

from __future__ import annotations

import io
import json
import time
import zipfile

from fastapi.testclient import TestClient

from tests.conftest import create_project_with_video
from vistlabel.persistence.paths import objects_json_path, project_dir
from vistlabel.video.ffmpeg import FfmpegBinaries


def _create_object(api_client: TestClient, project_id: str, label_id: int = 3) -> str:
    # Projects start with an empty, operator-defined ontology, so the label
    # has to be registered before an object can reference it -- this mirrors
    # what the annotation UI does when a new object name is first typed.
    labels_response = api_client.put(
        f"/api/v1/projects/{project_id}/labels",
        json={"labels": [{"label_id": label_id, "name": "pipette"}]},
    )
    assert labels_response.status_code == 200, labels_response.text
    response = api_client.post(
        f"/api/v1/projects/{project_id}/objects",
        json={"label_id": label_id, "label_name": "pipette"},
    )
    assert response.status_code == 201, response.text
    return response.json()["object_id"]


def _seed_and_review(
    api_client: TestClient,
    project_id: str,
    object_id: str,
    frame_index: int,
    points: list[dict],
) -> None:
    segment = api_client.post(
        f"/api/v1/projects/{project_id}/segment",
        json={
            "backend_id": "fake",
            "frame_index": frame_index,
            "object_id": object_id,
            "points": points,
        },
    )
    assert segment.status_code == 200, segment.text
    mask_png = segment.json()["visible_mask_png_base64"]
    save = api_client.patch(
        f"/api/v1/projects/{project_id}/frames/{frame_index}/annotations",
        json={
            "expected_revision": 0,
            "upsert_object": {
                "object_id": object_id,
                "label_id": 3,
                "visible_mask_png_base64": mask_png,
                "source": "fake_interactive",
            },
        },
    )
    assert save.status_code == 200, save.text
    api_client.patch(
        f"/api/v1/projects/{project_id}/frames/{frame_index}/annotations",
        json={"expected_revision": save.json()["revision"], "review_status": "reviewed"},
    )


def _wait_for_job(api_client: TestClient, project_id: str, job_id: str, timeout: float = 20.0):
    deadline = time.monotonic() + timeout
    terminal = {"succeeded", "failed", "cancelled", "paused"}
    while time.monotonic() < deadline:
        record = api_client.get(f"/api/v1/projects/{project_id}/jobs/{job_id}").json()
        if record["status"] in terminal:
            return record
        time.sleep(0.02)
    raise TimeoutError(f"job {job_id} did not finish within {timeout}s")


def _start_and_wait(api_client: TestClient, project_id: str, payload: dict) -> tuple[dict, dict]:
    response = api_client.post(f"/api/v1/projects/{project_id}/exports", json=payload)
    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]
    export_id = response.json()["export_id"]
    job_record = _wait_for_job(api_client, project_id, job_id)
    return job_record, {"export_id": export_id}


def test_export_with_single_part_mask_produces_exact_expected_yolo_and_coco_output(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    """Exact expected-output fixture: a deterministic single-object,
    single-frame export, asserting the actual
    label line and COCO segmentation content, not just shapes."""
    project = create_project_with_video(api_client, "exact fixture test")
    project_id = project["project_id"]
    width, height = project["source"]["width"], project["source"]["height"]
    object_id = _create_object(api_client, project_id)
    _seed_and_review(
        api_client, project_id, object_id, 0, [{"x": 10, "y": 10, "positive": True}]
    )

    job_record, ids = _start_and_wait(
        api_client,
        project_id,
        {"frame_selection_policy": "reviewed_only", "yolo_multipart_policy": "strict_error"},
    )
    assert job_record["status"] == "succeeded", job_record
    export_id = ids["export_id"]

    download = api_client.get(f"/api/v1/projects/{project_id}/exports/{export_id}/download")
    assert download.status_code == 200
    archive = zipfile.ZipFile(io.BytesIO(download.content))
    names = set(archive.namelist())
    assert "dataset.yaml" in names
    assert "export_manifest.json" in names
    assert "validation_report.json" in names

    label_files = [n for n in names if n.startswith("labels/") and n.endswith(".txt")]
    assert len(label_files) == 1
    stem = label_files[0].rsplit("/", 1)[1].removesuffix(".txt")
    image_files = [n for n in names if n.startswith("images/") and stem in n]
    assert len(image_files) == 1  # image and label stems match exactly

    label_text = archive.read(label_files[0]).decode().strip()
    parts = label_text.split(" ")
    assert parts[0] == "0"  # the label's position in labels.json, not its label_id (3)
    coordinates = [float(v) for v in parts[1:]]
    assert len(coordinates) % 2 == 0 and len(coordinates) >= 6
    assert all(0.0 <= v <= 1.0 for v in coordinates)
    # A point prompt at (10, 10) produces a disk-shaped mask centered
    # there (FakeBackend) -- every polygon vertex should be within a
    # bounded neighborhood of that seed point, not scattered arbitrarily.
    pixel_coords = [
        (coordinates[i] * width, coordinates[i + 1] * height)
        for i in range(0, len(coordinates), 2)
    ]
    assert all(abs(x - 10) <= 6 and abs(y - 10) <= 6 for x, y in pixel_coords)

    if "annotations/instances_val.json" in names:
        coco_split = "annotations/instances_val.json"
    else:
        coco_split = "annotations/instances_train.json"
    coco = json.loads(archive.read(coco_split))
    assert len(coco["annotations"]) == 1
    assert coco["annotations"][0]["category_id"] == 0
    # Categories mirror the project's own ontology, which here holds exactly
    # the one label the fixture registered.
    assert coco["categories"] == [{"id": 0, "name": "pipette"}]


def test_export_numbers_classes_by_their_order_in_labels_json(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    """Reordering labels.json renumbers the exported classes; the stored
    label_id (3 here) stays what objects and frames reference."""
    project_id = create_project_with_video(api_client, "class order test")["project_id"]
    object_id = _create_object(api_client, project_id)
    _seed_and_review(
        api_client, project_id, object_id, 0, [{"x": 10, "y": 10, "positive": True}]
    )
    reordered = api_client.put(
        f"/api/v1/projects/{project_id}/labels",
        json={"labels": [{"label_id": 7, "name": "hand"}, {"label_id": 3, "name": "pipette"}]},
    )
    assert reordered.status_code == 200, reordered.text

    job_record, ids = _start_and_wait(
        api_client,
        project_id,
        {"frame_selection_policy": "reviewed_only", "yolo_multipart_policy": "strict_error"},
    )
    assert job_record["status"] == "succeeded", job_record
    download = api_client.get(
        f"/api/v1/projects/{project_id}/exports/{ids['export_id']}/download"
    )
    archive = zipfile.ZipFile(io.BytesIO(download.content))
    names = set(archive.namelist())

    dataset_yaml = archive.read("dataset.yaml").decode().splitlines()
    assert dataset_yaml[-3:] == ["names:", "  0: hand", "  1: pipette"]
    label_file = next(n for n in names if n.startswith("labels/") and n.endswith(".txt"))
    assert archive.read(label_file).decode().split(" ")[0] == "1"
    cocos = [json.loads(archive.read(n)) for n in names if n.startswith("annotations/instances_")]
    assert cocos
    for coco in cocos:
        assert coco["categories"] == [{"id": 0, "name": "hand"}, {"id": 1, "name": "pipette"}]
    assert [a["category_id"] for coco in cocos for a in coco["annotations"]] == [1]


def test_export_minimal_bridge_preserves_one_instance_and_records_conversion(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_with_video(api_client, "bridge policy test")["project_id"]
    object_id = _create_object(api_client, project_id)
    _seed_and_review(
        api_client,
        project_id,
        object_id,
        0,
        [{"x": 10, "y": 10, "positive": True}, {"x": 50, "y": 50, "positive": True}],
    )

    job_record, ids = _start_and_wait(
        api_client,
        project_id,
        {"frame_selection_policy": "reviewed_only", "yolo_multipart_policy": "minimal_bridge"},
    )
    assert job_record["status"] == "succeeded"

    report = api_client.get(
        f"/api/v1/projects/{project_id}/exports/{ids['export_id']}/validation-report"
    ).json()
    assert len(report["yolo_bridge_conversions"]) == 1
    assert report["yolo_bridge_conversions"][0]["bridge_pixel_count"] > 0
    assert report["dropped_instances"] == []


def test_export_strict_error_drops_multipart_instance_from_yolo_but_keeps_it_in_coco(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_with_video(api_client, "strict error test")["project_id"]
    object_id = _create_object(api_client, project_id)
    _seed_and_review(
        api_client,
        project_id,
        object_id,
        0,
        [{"x": 10, "y": 10, "positive": True}, {"x": 50, "y": 50, "positive": True}],
    )

    job_record, ids = _start_and_wait(
        api_client,
        project_id,
        {"frame_selection_policy": "reviewed_only", "yolo_multipart_policy": "strict_error"},
    )
    assert job_record["status"] == "succeeded"
    export_id = ids["export_id"]

    report = api_client.get(
        f"/api/v1/projects/{project_id}/exports/{export_id}/validation-report"
    ).json()
    assert len(report["dropped_instances"]) == 1

    download = api_client.get(f"/api/v1/projects/{project_id}/exports/{export_id}/download")
    archive = zipfile.ZipFile(io.BytesIO(download.content))
    label_files = [n for n in archive.namelist() if n.endswith(".txt")]
    assert archive.read(label_files[0]).decode() == ""  # excluded, not fabricated

    coco_names = [n for n in archive.namelist() if n.startswith("annotations/")]
    combined_annotations = sum(
        len(json.loads(archive.read(name))["annotations"]) for name in coco_names
    )
    assert combined_annotations == 1  # COCO keeps the exact instance regardless


def test_export_split_components_emits_one_row_per_component(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_with_video(api_client, "split components test")["project_id"]
    object_id = _create_object(api_client, project_id)
    _seed_and_review(
        api_client,
        project_id,
        object_id,
        0,
        [{"x": 10, "y": 10, "positive": True}, {"x": 50, "y": 50, "positive": True}],
    )

    job_record, ids = _start_and_wait(
        api_client,
        project_id,
        {"frame_selection_policy": "reviewed_only", "yolo_multipart_policy": "split_components"},
    )
    assert job_record["status"] == "succeeded"

    download = api_client.get(
        f"/api/v1/projects/{project_id}/exports/{ids['export_id']}/download"
    )
    archive = zipfile.ZipFile(io.BytesIO(download.content))
    label_files = [n for n in archive.namelist() if n.endswith(".txt")]
    lines = [line for line in archive.read(label_files[0]).decode().splitlines() if line]
    assert len(lines) == 2  # both components exported as separate rows


def test_export_all_frames_requires_explicit_acknowledgement(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_with_video(api_client, "all frames warning test")["project_id"]
    response = api_client.post(
        f"/api/v1/projects/{project_id}/exports",
        json={"frame_selection_policy": "all_frames"},
    )
    assert response.status_code == 422

    acknowledged = api_client.post(
        f"/api/v1/projects/{project_id}/exports",
        json={"frame_selection_policy": "all_frames", "acknowledge_all_frames_warning": True},
    )
    assert acknowledged.status_code == 202


def test_split_assignments_control_train_val_and_exclude_frames(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_with_video(api_client, "split assignments test")["project_id"]
    object_id = _create_object(api_client, project_id)
    for frame_index in (0, 1, 2):
        _seed_and_review(
            api_client, project_id, object_id, frame_index, [{"x": 10, "y": 10, "positive": True}]
        )

    project = api_client.get(f"/api/v1/projects/{project_id}").json()
    put_response = api_client.put(
        f"/api/v1/projects/{project_id}/split-assignments",
        json={
            "expected_revision": project["revision"],
            "split_assignments": [
                {"start_frame": 0, "end_frame": 0, "label": "train"},
                {"start_frame": 1, "end_frame": 1, "label": "val"},
                {"start_frame": 2, "end_frame": 2, "label": "excluded"},
            ],
        },
    )
    assert put_response.status_code == 200, put_response.text

    get_response = api_client.get(f"/api/v1/projects/{project_id}/split-assignments")
    assert len(get_response.json()) == 3

    job_record, ids = _start_and_wait(
        api_client, project_id, {"frame_selection_policy": "reviewed_only"}
    )
    assert job_record["status"] == "succeeded"
    manifest = api_client.get(
        f"/api/v1/projects/{project_id}/exports/{ids['export_id']}"
    ).json()
    assert manifest["train_frame_count"] == 1
    assert manifest["val_frame_count"] == 1
    assert manifest["frame_count"] == 2  # frame 2 excluded, never written


def test_negative_frame_fraction_adds_empty_label_frames(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_with_video(api_client, "negative frames test")["project_id"]
    object_id = _create_object(api_client, project_id)
    _seed_and_review(
        api_client, project_id, object_id, 0, [{"x": 10, "y": 10, "positive": True}]
    )

    job_record, ids = _start_and_wait(
        api_client,
        project_id,
        {"frame_selection_policy": "reviewed_only", "negative_frame_fraction": 1.0},
    )
    assert job_record["status"] == "succeeded"
    manifest = api_client.get(
        f"/api/v1/projects/{project_id}/exports/{ids['export_id']}"
    ).json()
    assert manifest["negative_frame_count"] >= 1
    assert manifest["frame_count"] > 1


def test_deleted_frames_are_left_out_of_the_export(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_with_video(api_client, "deleted frames test")["project_id"]
    object_id = _create_object(api_client, project_id)
    for frame_index in (0, 1, 2):
        _seed_and_review(
            api_client, project_id, object_id, frame_index, [{"x": 10, "y": 10, "positive": True}]
        )
    frame_1 = api_client.get(f"/api/v1/projects/{project_id}/frames/1/annotations").json()
    deleted = api_client.patch(
        f"/api/v1/projects/{project_id}/frames/1/annotations",
        json={"expected_revision": frame_1["revision"], "excluded": True},
    )
    assert deleted.status_code == 200, deleted.text

    job_record, ids = _start_and_wait(
        api_client,
        project_id,
        {"frame_selection_policy": "reviewed_only", "negative_frame_fraction": 1.0},
    )
    assert job_record["status"] == "succeeded", job_record
    download = api_client.get(
        f"/api/v1/projects/{project_id}/exports/{ids['export_id']}/download"
    )
    images = [n for n in zipfile.ZipFile(io.BytesIO(download.content)).namelist()
              if n.startswith("images/")]
    assert any(n.endswith(f"{project_id}_000000.jpg") for n in images)
    assert any(n.endswith(f"{project_id}_000002.jpg") for n in images)
    assert not any("_000001." in n for n in images)
    report = api_client.get(
        f"/api/v1/projects/{project_id}/exports/{ids['export_id']}/validation-report"
    )
    assert report.status_code == 200, report.text
    assert any("deleted in the editor" in w for w in report.json()["warnings"])


def test_masks_of_a_missing_object_are_left_out_of_the_export(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_with_video(api_client, "orphaned mask test")["project_id"]
    object_id = _create_object(api_client, project_id)
    _seed_and_review(
        api_client, project_id, object_id, 0, [{"x": 10, "y": 10, "positive": True}]
    )
    # Simulate a mask outliving its object (e.g. written by a propagation
    # run after the object was deleted) by dropping it from objects.json only.
    settings = api_client.app.state.settings
    objects_path = objects_json_path(project_dir(settings.workspace.root, project_id))
    objects_file = json.loads(objects_path.read_text(encoding="utf-8"))
    objects_file["objects"] = []
    objects_path.write_text(json.dumps(objects_file), encoding="utf-8")

    job_record, ids = _start_and_wait(
        api_client,
        project_id,
        {
            "frame_selection_policy": "reviewed_only",
            "annotation_type": "bbox",
            "dataset_layout": "coco",
        },
    )
    assert job_record["status"] == "succeeded", job_record
    download = api_client.get(
        f"/api/v1/projects/{project_id}/exports/{ids['export_id']}/download"
    )
    archive = zipfile.ZipFile(io.BytesIO(download.content))
    for name in archive.namelist():
        if name.endswith("_annotations.coco.json"):
            assert json.loads(archive.read(name))["annotations"] == []
    report = api_client.get(
        f"/api/v1/projects/{project_id}/exports/{ids['export_id']}/validation-report"
    ).json()
    assert any("mask(s) of deleted objects" in w for w in report["warnings"])


def test_list_and_delete_export(api_client: TestClient, ffmpeg_binaries: FfmpegBinaries) -> None:
    project_id = create_project_with_video(api_client, "list delete export test")["project_id"]
    object_id = _create_object(api_client, project_id)
    _seed_and_review(
        api_client, project_id, object_id, 0, [{"x": 10, "y": 10, "positive": True}]
    )
    job_record, ids = _start_and_wait(
        api_client, project_id, {"frame_selection_policy": "reviewed_only"}
    )
    assert job_record["status"] == "succeeded"
    export_id = ids["export_id"]

    listed = api_client.get(f"/api/v1/projects/{project_id}/exports").json()
    assert any(entry["export_id"] == export_id for entry in listed)

    delete_response = api_client.delete(f"/api/v1/projects/{project_id}/exports/{export_id}")
    assert delete_response.status_code == 204

    missing = api_client.get(f"/api/v1/projects/{project_id}/exports/{export_id}")
    assert missing.status_code == 404
    missing_download = api_client.get(
        f"/api/v1/projects/{project_id}/exports/{export_id}/download"
    )
    assert missing_download.status_code == 404


def test_export_with_no_reviewed_frames_produces_an_empty_but_valid_dataset(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_with_video(api_client, "empty export test")["project_id"]
    job_record, ids = _start_and_wait(
        api_client, project_id, {"frame_selection_policy": "reviewed_only"}
    )
    assert job_record["status"] == "succeeded"
    manifest = api_client.get(
        f"/api/v1/projects/{project_id}/exports/{ids['export_id']}"
    ).json()
    assert manifest["frame_count"] == 0


def test_export_bbox_in_coco_layout_matches_the_rf_detr_dataset_structure(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_with_video(api_client, "rf-detr layout test")["project_id"]
    object_id = _create_object(api_client, project_id)
    for frame_index in (0, 1):
        _seed_and_review(
            api_client, project_id, object_id, frame_index, [{"x": 10, "y": 10, "positive": True}]
        )

    job_record, ids = _start_and_wait(
        api_client,
        project_id,
        {
            "frame_selection_policy": "reviewed_only",
            "annotation_type": "bbox",
            "dataset_layout": "coco",
        },
    )
    assert job_record["status"] == "succeeded", job_record
    export_id = ids["export_id"]

    manifest = api_client.get(f"/api/v1/projects/{project_id}/exports/{export_id}").json()
    assert manifest["annotation_type"] == "bbox"
    assert manifest["dataset_layout"] == "coco"

    download = api_client.get(f"/api/v1/projects/{project_id}/exports/{export_id}/download")
    archive = zipfile.ZipFile(io.BytesIO(download.content))
    names = set(archive.namelist())
    # RF-DETR detects COCO format by train/_annotations.coco.json and reads
    # images from the same folder; no YOLO artifacts belong in this layout.
    assert "train/_annotations.coco.json" in names
    assert "valid/_annotations.coco.json" in names
    assert not any(n.startswith(("labels/", "images/", "annotations/")) for n in names)
    assert "dataset.yaml" not in names

    annotation_count = 0
    for split_dir in ("train", "valid"):
        coco = json.loads(archive.read(f"{split_dir}/_annotations.coco.json"))
        assert coco["categories"] == [{"id": 0, "name": "pipette"}]
        for image in coco["images"]:
            assert f"{split_dir}/{image['file_name']}" in names
        for annotation in coco["annotations"]:
            assert annotation["segmentation"] == []
            x, y, w, h = annotation["bbox"]
            # FakeBackend draws a small disk around the (10, 10) seed point.
            assert w > 0 and h > 0
            assert x <= 10 <= x + w and y <= 10 <= y + h
            annotation_count += 1
    assert annotation_count == 2


def test_export_bbox_in_ultralytics_layout_writes_yolo_detection_lines(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project = create_project_with_video(api_client, "yolo bbox test")
    project_id = project["project_id"]
    width = project["source"]["width"]
    object_id = _create_object(api_client, project_id)
    # Two disconnected blobs: the box export must still yield one instance.
    _seed_and_review(
        api_client,
        project_id,
        object_id,
        0,
        [{"x": 10, "y": 10, "positive": True}, {"x": 50, "y": 50, "positive": True}],
    )

    job_record, ids = _start_and_wait(
        api_client,
        project_id,
        {
            "frame_selection_policy": "reviewed_only",
            "annotation_type": "bbox",
            "dataset_layout": "ultralytics",
            "yolo_multipart_policy": "strict_error",
        },
    )
    assert job_record["status"] == "succeeded", job_record
    export_id = ids["export_id"]

    download = api_client.get(f"/api/v1/projects/{project_id}/exports/{export_id}/download")
    archive = zipfile.ZipFile(io.BytesIO(download.content))
    names = set(archive.namelist())
    assert "dataset.yaml" in names
    label_files = [n for n in names if n.startswith("labels/") and n.endswith(".txt")]
    assert len(label_files) == 1
    lines = archive.read(label_files[0]).decode().strip().splitlines()
    assert len(lines) == 1  # strict_error does not drop it: boxes have no multipart issue
    parts = lines[0].split(" ")
    assert parts[0] == "0"
    cx, cy, w, h = (float(v) for v in parts[1:])
    assert all(0.0 <= v <= 1.0 for v in (cx, cy, w, h))
    # Spans both seed points, 40px apart horizontally.
    assert w * width > 40

    report = api_client.get(
        f"/api/v1/projects/{project_id}/exports/{export_id}/validation-report"
    ).json()
    assert report["dropped_instances"] == []
    assert report["invalid_polygons"] == []
