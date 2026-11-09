# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Creating a project from a COCO or YOLO dataset ZIP through the real API:
images become frames, annotations become masks and objects, and a VISTLabel
export imports back with its masks intact."""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from tests.conftest import create_project_with_video
from tests.integration.test_exports_api import _create_object, _seed_and_review, _start_and_wait
from vistlabel.domain.frames import FrameAnnotation
from vistlabel.persistence.atomic import load_model
from vistlabel.persistence.paths import frame_annotation_path, project_dir
from vistlabel.persistence.rle import decode_mask
from vistlabel.video.ffmpeg import FfmpegBinaries


def _png(width: int, height: int, shade: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (shade, shade, shade)).save(buffer, format="PNG")
    return buffer.getvalue()


def _zip_bytes(members: dict[str, bytes | str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def _import(api_client: TestClient, data: bytes, **form: str) -> dict:
    response = api_client.post(
        "/api/v1/projects/import-dataset",
        data={"project_name": "imported", **form},
        files={"dataset_file": ("dataset.zip", data, "application/zip")},
    )
    assert response.status_code == 201, response.text
    return response.json()


def _frame(api_client: TestClient, project_id: str, frame_index: int) -> FrameAnnotation:
    settings = api_client.app.state.settings  # type: ignore[attr-defined]
    pdir = project_dir(settings.workspace.root, project_id)
    return load_model(frame_annotation_path(pdir, frame_index), FrameAnnotation)


def test_import_coco_segmentation(api_client: TestClient, ffmpeg_binaries: FfmpegBinaries) -> None:
    categories = [{"id": 1, "name": "pipette"}, {"id": 2, "name": "glove"}]
    square = [[4, 4, 14, 4, 14, 14, 4, 14]]
    train = {
        "images": [
            {"id": 1, "file_name": "f_1.png", "width": 48, "height": 32},
            {"id": 2, "file_name": "f_2.png", "width": 48, "height": 32},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "segmentation": square,
                "bbox": [4, 4, 10, 10],
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 2,
                "category_id": 1,
                "segmentation": [[5, 4, 15, 4, 15, 14, 5, 14]],
                "bbox": [5, 4, 10, 10],
                "iscrowd": 0,
            },
            {
                "id": 3,
                "image_id": 2,
                "category_id": 2,
                "segmentation": [],
                "bbox": [30, 10, 6, 8],
                "iscrowd": 0,
            },
        ],
        "categories": categories,
    }
    valid = {
        # Smaller than the rest: padded, not scaled.
        "images": [{"id": 1, "file_name": "f_3.png", "width": 40, "height": 30}],
        "annotations": [],
        "categories": categories,
    }
    data = _zip_bytes(
        {
            "train/_annotations.coco.json": json.dumps(train),
            "train/f_1.png": _png(48, 32, 40),
            "train/f_2.png": _png(48, 32, 80),
            "valid/_annotations.coco.json": json.dumps(valid),
            "valid/f_3.png": _png(40, 30, 120),
        }
    )

    body = _import(api_client, data, frame_rate="5")
    project, report = body["project"], body["report"]
    assert report["dataset_format"] == "coco"
    assert (report["image_count"], report["annotation_count"], report["bbox_annotation_count"]) == (
        3,
        3,
        1,
    )
    # The pipette overlaps itself across the two frames: one object, not two.
    assert report["object_count"] == 2
    assert any("padded" in warning for warning in report["warnings"])
    assert project["status"] == "new"
    assert (project["source"]["width"], project["source"]["height"]) == (48, 32)
    assert project["source"]["frame_count"] == 3
    assert project["source"]["nominal_fps"] == 5
    assert project["split_assignments"] == [
        {"start_frame": 0, "end_frame": 1, "label": "train"},
        {"start_frame": 2, "end_frame": 2, "label": "val"},
    ]
    project_id = project["project_id"]

    labels = api_client.get(f"/api/v1/projects/{project_id}/labels").json()["labels"]
    assert [(label["label_id"], label["name"]) for label in labels] == [
        (1, "pipette"),
        (2, "glove"),
    ]
    objects = api_client.get(f"/api/v1/projects/{project_id}/objects").json()["objects"]
    assert sorted(o["label_name"] for o in objects) == ["glove", "pipette"]

    first, second = _frame(api_client, project_id, 0), _frame(api_client, project_id, 1)
    assert first.objects[0].object_id == second.objects[0].object_id
    assert first.objects[0].source == "import"
    glove = second.objects[1]
    assert glove.source == "import_bbox"
    ys, xs = np.nonzero(decode_mask(glove.visible_mask_rle))
    assert (xs.min(), xs.max(), ys.min(), ys.max()) == (30, 35, 10, 17)
    assert decode_mask(glove.visible_mask_rle).sum() == 6 * 8

    # The frames really are the images, in order.
    frame = api_client.get(f"/api/v1/projects/{project_id}/frames/1")
    assert frame.status_code == 200
    pixel = Image.open(io.BytesIO(frame.content)).convert("L").getpixel((24, 16))
    assert abs(pixel - 80) <= 3


def test_import_yolo_boxes(api_client: TestClient, ffmpeg_binaries: FfmpegBinaries) -> None:
    data = _zip_bytes(
        {
            "data.yaml": "names: [pipette]\n",
            "images/train/a.png": _png(40, 20, 50),
            "labels/train/a.txt": "0 0.25 0.5 0.2 0.4\n",
            "images/train/b.png": _png(40, 20, 60),
        }
    )
    body = _import(api_client, data, dataset_format="yolo")
    assert body["report"]["dataset_format"] == "yolo"
    assert body["project"]["source"]["frame_count"] == 2
    # A single split is left to the export's own temporal split.
    assert body["project"]["split_assignments"] == []
    project_id = body["project"]["project_id"]

    (box,) = _frame(api_client, project_id, 0).objects
    assert box.source == "import_bbox"
    ys, xs = np.nonzero(decode_mask(box.visible_mask_rle))
    assert (xs.min(), xs.max(), ys.min(), ys.max()) == (6, 13, 6, 13)
    progress = api_client.get(f"/api/v1/projects/{project_id}/progress").json()
    assert progress["annotated_frame_count"] == 1


def test_vistlabel_export_imports_back_with_exact_masks(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    source_id = create_project_with_video(api_client, "export source")["project_id"]
    object_id = _create_object(api_client, source_id)
    for frame_index in (2, 3):
        _seed_and_review(
            api_client, source_id, object_id, frame_index, [{"x": 30, "y": 30, "positive": True}]
        )
    job, export = _start_and_wait(
        api_client,
        source_id,
        {"frame_selection_policy": "reviewed_only", "dataset_layout": "ultralytics"},
    )
    assert job["status"] == "succeeded", job
    exported = api_client.get(
        f"/api/v1/projects/{source_id}/exports/{export['export_id']}/download"
    )
    assert exported.status_code == 200

    body = _import(api_client, exported.content)
    # Both COCO and YOLO labels are in the export; COCO's exact masks win.
    assert body["report"]["dataset_format"] == "coco"
    assert body["report"]["object_count"] == 1
    imported_id = body["project"]["project_id"]
    for imported_index, source_index in enumerate((2, 3)):
        (original,) = _frame(api_client, source_id, source_index).objects
        (imported,) = _frame(api_client, imported_id, imported_index).objects
        assert np.array_equal(
            decode_mask(imported.visible_mask_rle), decode_mask(original.visible_mask_rle)
        )
        # The export numbers classes by position, so pipette (label_id 3,
        # the only label) comes back as class 0 under the same name.
        assert imported.label_id == 0
    assert api_client.get(f"/api/v1/projects/{imported_id}/labels").json()["labels"] == [
        {"label_id": 0, "name": "pipette", "color": None}
    ]


def test_failed_import_leaves_no_project(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    response = api_client.post(
        "/api/v1/projects/import-dataset",
        data={"project_name": "broken"},
        files={"dataset_file": ("dataset.zip", b"not a zip", "application/zip")},
    )
    assert response.status_code == 422
    assert "ZIP" in response.json()["detail"]

    response = api_client.post(
        "/api/v1/projects/import-dataset",
        data={"project_name": "no labels"},
        files={"dataset_file": ("dataset.zip", _zip_bytes({"a.png": _png(8, 8, 0)}), "x")},
    )
    assert response.status_code == 422
    assert api_client.get("/api/v1/projects").json() == []


def test_frame_rate_is_validated_before_creating_anything(api_client: TestClient) -> None:
    response = api_client.post(
        "/api/v1/projects/import-dataset",
        data={"project_name": "x", "frame_rate": "0"},
        files={"dataset_file": ("dataset.zip", b"", "application/zip")},
    )
    assert response.status_code == 422
    assert api_client.get("/api/v1/projects").json() == []


def test_keep_every_nth_keeps_images_0_n_2n_with_their_annotations(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    members: dict[str, bytes | str] = {"data.yaml": "names: [pipette]\n"}
    for index in range(5):
        members[f"images/train/f_{index}.png"] = _png(40, 20, 40 * index)
        # Box width grows with the index, so each frame shows which image it is.
        members[f"labels/train/f_{index}.txt"] = f"0 0.5 0.5 {0.05 * (index + 1)} 0.5\n"
    body = _import(api_client, _zip_bytes(members), dataset_format="yolo", keep_every_nth="2")

    report, project = body["report"], body["project"]
    assert (report["image_count"], report["source_image_count"]) == (3, 5)
    assert report["annotation_count"] == 3
    assert project["source"]["frame_count"] == 3
    assert project["source"]["import_options"]["keep_every_nth"] == 2
    project_id = project["project_id"]
    for frame_index, image_index in enumerate((0, 2, 4)):
        (box,) = _frame(api_client, project_id, frame_index).objects
        _, xs = np.nonzero(decode_mask(box.visible_mask_rle))
        assert xs.max() - xs.min() + 1 == 2 * (image_index + 1)
        frame = api_client.get(f"/api/v1/projects/{project_id}/frames/{frame_index}")
        pixel = Image.open(io.BytesIO(frame.content)).convert("L").getpixel((2, 2))
        assert abs(pixel - 40 * image_index) <= 3


def test_keep_every_nth_is_validated_before_creating_anything(api_client: TestClient) -> None:
    response = api_client.post(
        "/api/v1/projects/import-dataset",
        data={"project_name": "x", "keep_every_nth": "0"},
        files={"dataset_file": ("dataset.zip", b"", "application/zip")},
    )
    assert response.status_code == 422
    assert api_client.get("/api/v1/projects").json() == []
