# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Exact COCO export preserves disconnected visible components as one
instance -- RLE segmentation passes through unchanged, never a lossy
polygon approximation."""

from __future__ import annotations

import numpy as np

from vistlabel.domain.frames import FrameAnnotation, FrameObjectAnnotation
from vistlabel.domain.labels import LabelDefinition, LabelsFile
from vistlabel.exports.coco_writer import build_coco_dataset
from vistlabel.persistence.rle import decode_mask, encode_mask


def test_coco_dataset_preserves_multipart_mask_as_one_annotation() -> None:
    mask = np.zeros((20, 40), dtype=bool)
    mask[4:8, 2:8] = True
    mask[4:8, 32:38] = True  # two disconnected components, one logical object

    obj = FrameObjectAnnotation(
        object_id="obj_1",
        label_id=3,
        visible_mask_rle=encode_mask(mask),
        connected_component_count=2,
        source="test",
    )
    frame = FrameAnnotation(frame_index=0, timestamp_seconds=0.0, objects=[obj])
    labels = LabelsFile(labels=[LabelDefinition(label_id=3, name="pipette")])

    coco = build_coco_dataset(
        frame_indices=[0],
        frame_annotations={0: frame},
        labels=labels,
        image_filename_by_frame={0: "frame_000000.jpg"},
        image_width=40,
        image_height=20,
    )

    assert len(coco["images"]) == 1
    assert coco["images"][0] == {
        "id": 0,
        "file_name": "frame_000000.jpg",
        "width": 40,
        "height": 20,
    }
    assert len(coco["annotations"]) == 1  # one instance, not two
    annotation = coco["annotations"][0]
    assert annotation["category_id"] == 3
    assert annotation["area"] == int(mask.sum())

    # The segmentation round-trips to the exact original mask, disconnected
    # components and all -- the whole point of exact COCO export.
    from vistlabel.domain.frames import RleMask

    segmentation = annotation["segmentation"]
    restored = decode_mask(
        RleMask(size=tuple(segmentation["size"]), counts=segmentation["counts"])
    )
    np.testing.assert_array_equal(restored, mask)


def test_coco_dataset_includes_negative_frames_with_no_annotations() -> None:
    labels = LabelsFile(labels=[LabelDefinition(label_id=0, name="pipette")])
    coco = build_coco_dataset(
        frame_indices=[0, 1],
        frame_annotations={},  # frame 1 has no annotation file at all -- a negative
        labels=labels,
        image_filename_by_frame={0: "a.jpg", 1: "b.jpg"},
        image_width=10,
        image_height=10,
    )
    assert len(coco["images"]) == 2
    assert coco["annotations"] == []


def test_coco_dataset_lists_every_project_label_as_a_category_even_if_unused() -> None:
    labels = LabelsFile(
        labels=[LabelDefinition(label_id=0, name="a"), LabelDefinition(label_id=1, name="b")]
    )
    coco = build_coco_dataset(
        frame_indices=[],
        frame_annotations={},
        labels=labels,
        image_filename_by_frame={},
        image_width=10,
        image_height=10,
    )
    assert [c["name"] for c in coco["categories"]] == ["a", "b"]


def test_coco_bbox_mode_keeps_only_the_tight_box_around_all_components() -> None:
    mask = np.zeros((20, 40), dtype=bool)
    mask[4:8, 2:8] = True
    mask[4:8, 32:38] = True
    empty = FrameObjectAnnotation(
        object_id="obj_empty",
        label_id=3,
        visible_mask_rle=encode_mask(np.zeros((20, 40), dtype=bool)),
        connected_component_count=0,
        source="test",
    )
    obj = FrameObjectAnnotation(
        object_id="obj_1",
        label_id=3,
        visible_mask_rle=encode_mask(mask),
        connected_component_count=2,
        source="test",
    )
    frame = FrameAnnotation(frame_index=0, timestamp_seconds=0.0, objects=[obj, empty])
    labels = LabelsFile(labels=[LabelDefinition(label_id=3, name="pipette")])

    coco = build_coco_dataset(
        frame_indices=[0],
        frame_annotations={0: frame},
        labels=labels,
        image_filename_by_frame={0: "frame_000000.jpg"},
        image_width=40,
        image_height=20,
        annotation_type="bbox",
    )

    # The empty mask has no box, so it is not a detection target at all.
    assert len(coco["annotations"]) == 1
    annotation = coco["annotations"][0]
    assert annotation["segmentation"] == []
    assert annotation["bbox"] == [2.0, 4.0, 36.0, 4.0]
    assert annotation["category_id"] == 3
