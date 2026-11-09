# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""COCO / YOLO dataset import: archive access, both readers, mask
rasterization, identity linking and split recovery -- everything short of
encoding the video (tests/integration/test_dataset_import_api.py)."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pycocotools import mask as mask_utils

from vistlabel.domain.project import SplitLabel
from vistlabel.imports.archive import (
    DatasetArchive,
    DatasetImportError,
    frame_order_key,
    split_from_path,
)
from vistlabel.imports.dataset import DatasetAnnotation, DatasetImage, ImportWarnings
from vistlabel.imports.service import (
    _Instance,
    link_instances,
    parse_dataset,
    rasterize,
    split_assignments_for,
)
from vistlabel.imports.yolo_reader import label_member_for
from vistlabel.persistence.rle import encode_mask


def _png(width: int = 40, height: int = 30) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (90, 120, 150)).save(buffer, format="PNG")
    return buffer.getvalue()


def _zip(tmp_path: Path, members: dict[str, bytes | str]) -> Path:
    path = tmp_path / "dataset.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return path


def _coco(images: list[dict], annotations: list[dict], categories: list[dict]) -> str:
    return json.dumps({"images": images, "annotations": annotations, "categories": categories})


# -- archive ----------------------------------------------------------------


def test_archive_ignores_hidden_and_traversal_members(tmp_path: Path) -> None:
    path = _zip(
        tmp_path,
        {
            "images/a.png": _png(),
            "__MACOSX/images/._a.png": b"resource fork",
            "images/.hidden.png": _png(),
            "../escape.png": _png(),
        },
    )
    with DatasetArchive(path) as archive:
        assert archive.image_names == ["images/a.png"]


def test_not_a_zip_is_a_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "dataset.zip"
    path.write_bytes(b"not a zip")
    with pytest.raises(DatasetImportError, match="not a valid ZIP"):
        DatasetArchive(path)


def test_image_size_follows_exif_orientation(tmp_path: Path) -> None:
    image = Image.new("RGB", (40, 30))
    exif = image.getexif()
    exif[0x0112] = 6  # rotate 90 degrees on display
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    path = _zip(tmp_path, {"images/a.jpg": buffer.getvalue()})
    with DatasetArchive(path) as archive:
        assert archive.image_size("images/a.jpg") == (30, 40)
        assert archive.open_image("images/a.jpg").size == (30, 40)


@pytest.mark.parametrize(
    ("parts", "split"),
    [
        (["train", "_annotations.coco.json"], "train"),
        (["valid", "_annotations.coco.json"], "val"),
        (["annotations", "instances_val2017.json"], "val"),
        (["images", "test"], "test"),
        (["valves", "images"], None),
        (["data"], None),
    ],
)
def test_split_from_path(parts: list[str], split: str | None) -> None:
    assert split_from_path(parts) == split


def test_frame_order_is_natural_and_ignores_split_folders() -> None:
    members = ["valid/x_10.png", "train/x_9.png", "train/x_100.png", "train/x_11.png"]
    assert sorted(members, key=frame_order_key) == [
        "train/x_9.png",
        "valid/x_10.png",
        "train/x_11.png",
        "train/x_100.png",
    ]


# -- COCO -----------------------------------------------------------------------


def test_coco_roboflow_layout(tmp_path: Path) -> None:
    rle = mask_utils.encode(
        np.asfortranarray(np.pad(np.ones((5, 5), np.uint8), ((0, 25), (0, 35))))
    )
    categories = [
        {"id": 0, "name": "lab", "supercategory": "none"},
        {"id": 1, "name": "pipette", "supercategory": "lab"},
        {"id": 2, "name": "glove", "supercategory": "lab"},
    ]
    train = _coco(
        [{"id": 7, "file_name": "a.png", "width": 40, "height": 30}],
        [
            {
                "id": 1,
                "image_id": 7,
                "category_id": 1,
                "iscrowd": 0,
                "bbox": [1, 1, 4, 4],
                "segmentation": [[1, 1, 10, 1, 10, 10, 1, 10]],
            },
            {
                "id": 2,
                "image_id": 7,
                "category_id": 2,
                "iscrowd": 0,
                "bbox": [2, 3, 4, 5],
                "segmentation": [],
            },
            {
                "id": 3,
                "image_id": 7,
                "category_id": 2,
                "iscrowd": 1,
                "bbox": [0, 0, 1, 1],
                "segmentation": [],
            },
        ],
        categories,
    )
    valid = _coco(
        [
            {"id": 1, "file_name": "b.png", "width": 40, "height": 30},
            {"id": 2, "file_name": "missing.png", "width": 40, "height": 30},
        ],
        [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "iscrowd": 0,
                "bbox": [0, 0, 5, 5],
                "track_id": 4,
                "segmentation": {"size": rle["size"], "counts": rle["counts"].decode("ascii")},
            }
        ],
        categories,
    )
    path = _zip(
        tmp_path,
        {
            "train/_annotations.coco.json": train,
            "train/a.png": _png(),
            "valid/_annotations.coco.json": valid,
            "valid/b.png": _png(),
            "README.roboflow.txt": "exported",
        },
    )
    with DatasetArchive(path) as archive:
        dataset = parse_dataset(archive, "auto")

    assert dataset.format == "coco"
    # The unused superclass is dropped; real category ids are kept.
    assert dataset.labels == {1: "pipette", 2: "glove"}
    by_member = {image.member: image for image in dataset.images}
    assert by_member["train/a.png"].split == "train"
    assert by_member["valid/b.png"].split == "val"
    polygon, box = by_member["train/a.png"].annotations
    assert polygon.polygons == [[1, 1, 10, 1, 10, 10, 1, 10]]
    assert box.is_bbox_only and box.bbox == (2, 3, 4, 5)
    (masked,) = by_member["valid/b.png"].annotations
    assert masked.rle is not None and masked.track_id == 4
    warnings = " | ".join(dataset.warnings.as_list())
    assert "crowd" in warnings and "missing.png" in warnings


def test_coco_beside_images_folder(tmp_path: Path) -> None:
    """VISTLabel's Ultralytics export: annotations/instances_<split>.json with
    file names that resolve into images/<split>/."""
    document = _coco(
        [{"id": 0, "file_name": "p_000000.png", "width": 40, "height": 30}],
        [],
        [{"id": 3, "name": "pipette"}],
    )
    path = _zip(
        tmp_path,
        {
            "annotations/instances_train.json": document,
            "images/train/p_000000.png": _png(),
            "labels/train/p_000000.txt": "",
            "export_manifest.json": json.dumps({"export_id": "exp_x"}),
        },
    )
    with DatasetArchive(path) as archive:
        dataset = parse_dataset(archive, "auto")
    assert dataset.format == "coco"
    assert [image.member for image in dataset.images] == ["images/train/p_000000.png"]
    assert dataset.labels == {3: "pipette"}


def test_coco_requested_but_absent(tmp_path: Path) -> None:
    path = _zip(tmp_path, {"images/a.png": _png(), "labels/a.txt": "0 .5 .5 .1 .1"})
    with DatasetArchive(path) as archive, pytest.raises(DatasetImportError, match="no COCO"):
        parse_dataset(archive, "coco")


# -- YOLO -----------------------------------------------------------------------


def test_label_member_for() -> None:
    assert label_member_for("images/train/a.jpg") == "labels/train/a.txt"
    assert label_member_for("train/images/a.jpg") == "train/labels/a.txt"
    assert label_member_for("frames/a.jpg") == "frames/a.txt"


def test_yolo_ultralytics_layout(tmp_path: Path) -> None:
    path = _zip(
        tmp_path,
        {
            "data.yaml": "path: .\ntrain: images/train\nval: images/val\nnames:\n  0: pipette\n"
            "  1: glove\n",
            "images/train/a.png": _png(),
            "labels/train/a.txt": "0 0.5 0.5 0.25 0.5\n1 0.1 0.1 0.5 0.1 0.5 0.5\nnonsense\n",
            "images/val/b.png": _png(),  # no label file: a negative image
        },
    )
    with DatasetArchive(path) as archive:
        dataset = parse_dataset(archive, "auto")
    assert dataset.format == "yolo"
    assert dataset.labels == {0: "pipette", 1: "glove"}
    a, b = sorted(dataset.images, key=lambda image: image.member)
    assert (a.split, b.split) == ("train", "val")
    assert b.annotations == []
    box, polygon = a.annotations
    assert box.bbox == pytest.approx((15.0, 7.5, 10.0, 15.0))
    assert polygon.polygons == [pytest.approx([4.0, 3.0, 20.0, 3.0, 20.0, 15.0])]
    assert any("unparseable" in line for line in dataset.warnings.as_list())


def test_yolo_roboflow_layout_with_list_names(tmp_path: Path) -> None:
    path = _zip(
        tmp_path,
        {
            "data.yaml": "names: ['pipette']\nnc: 1\n",
            "train/images/a.png": _png(),
            "train/labels/a.txt": "0 0.5 0.5 0.2 0.2\n3 0.5 0.5 0.2 0.2\n",
        },
    )
    with DatasetArchive(path) as archive:
        dataset = parse_dataset(archive, "auto")
    assert dataset.labels == {0: "pipette", 3: "class_3"}
    assert dataset.images[0].split == "train"


def test_neither_coco_nor_yolo(tmp_path: Path) -> None:
    path = _zip(tmp_path, {"photos/a.png": _png()})
    with DatasetArchive(path) as archive, pytest.raises(DatasetImportError, match="COCO or YOLO"):
        parse_dataset(archive, "auto")


# -- masks, identity, splits ------------------------------------------------


def test_box_rasterizes_to_exact_pixels() -> None:
    mask = rasterize(DatasetAnnotation(label_id=0, bbox=(2.0000004, 3, 4, 5)), 40, 30)
    assert mask is not None
    ys, xs = np.nonzero(mask)
    assert (xs.min(), xs.max(), ys.min(), ys.max()) == (2, 5, 3, 7)


def test_rle_for_another_image_size_is_rejected() -> None:
    rle = {"size": [10, 10], "counts": [0, 100]}
    assert rasterize(DatasetAnnotation(label_id=0, rle=rle), 40, 30) is None
    assert rasterize(DatasetAnnotation(label_id=0, rle=rle), 10, 10).all()  # type: ignore[union-attr]


def _instance(label_id: int, x: int, track_id: int | None = None) -> _Instance:
    mask = np.zeros((30, 40), dtype=bool)
    mask[5:15, x : x + 10] = True
    return _Instance(label_id, track_id, encode_mask(mask), 1, False)


def test_iou_linking_follows_objects_and_respects_class() -> None:
    frames = [
        [_instance(0, 0), _instance(0, 25)],
        [_instance(0, 26), _instance(0, 1), _instance(1, 0)],  # swapped order, new class
        [_instance(0, 12)],  # overlaps neither enough: a new object
    ]
    numbers = link_instances(frames, ImportWarnings())
    assert numbers[0] == [0, 1]
    assert numbers[1] == [1, 0, 2]
    assert numbers[2] == [3]


def test_track_ids_win_over_iou() -> None:
    frames = [
        [_instance(0, 0, track_id=7), _instance(0, 25, track_id=8)],
        [_instance(0, 0, track_id=8), _instance(0, 25, track_id=7)],
    ]
    assert link_instances(frames, ImportWarnings()) == [[0, 1], [1, 0]]


def test_split_assignments_need_train_and_val() -> None:
    def images(*splits: str | None) -> list[DatasetImage]:
        return [DatasetImage(member=f"{i}.png", split=split) for i, split in enumerate(splits)]

    assert split_assignments_for(images("train", "train")) == []
    ranges = split_assignments_for(images("train", "train", "val", "test", "val", "train"))
    assert [(r.start_frame, r.end_frame, r.label) for r in ranges] == [
        (0, 1, SplitLabel.TRAIN),
        (2, 2, SplitLabel.VAL),
        (4, 4, SplitLabel.VAL),
        (5, 5, SplitLabel.TRAIN),
    ]
