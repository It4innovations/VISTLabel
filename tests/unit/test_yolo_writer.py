# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""The three YOLO multipart compatibility policies, and dataset.yaml
generation."""

from __future__ import annotations

import numpy as np

from vistlabel.domain.frames import FrameAnnotation, FrameObjectAnnotation
from vistlabel.exports.yolo_writer import (
    build_dataset_yaml,
    build_yolo_bbox_lines,
    build_yolo_label_lines,
)
from vistlabel.persistence.rle import encode_mask

_WIDTH, _HEIGHT = 40, 20


def _multipart_mask() -> np.ndarray:
    mask = np.zeros((_HEIGHT, _WIDTH), dtype=bool)
    mask[8:12, 2:8] = True
    mask[8:12, 32:38] = True
    return mask


def _single_part_mask() -> np.ndarray:
    mask = np.zeros((_HEIGHT, _WIDTH), dtype=bool)
    mask[8:12, 15:25] = True
    return mask


def _frame_with(mask: np.ndarray, component_count: int) -> FrameAnnotation:
    obj = FrameObjectAnnotation(
        object_id="obj_1",
        label_id=3,
        visible_mask_rle=encode_mask(mask),
        connected_component_count=component_count,
        source="test",
    )
    return FrameAnnotation(frame_index=0, timestamp_seconds=0.0, objects=[obj])


def test_single_part_object_yields_one_label_line() -> None:
    frame = _frame_with(_single_part_mask(), component_count=1)
    lines, conversions, dropped, invalid = build_yolo_label_lines(
        frame=frame,
        width=_WIDTH,
        height=_HEIGHT,
        multipart_policy="strict_error",
        label_name_by_id={3: "pipette"},
    )
    assert len(lines) == 1
    assert lines[0].startswith("3 ")
    assert conversions == []
    assert dropped == []
    assert invalid == []


def test_strict_error_drops_multipart_instance() -> None:
    frame = _frame_with(_multipart_mask(), component_count=2)
    lines, conversions, dropped, _invalid = build_yolo_label_lines(
        frame=frame,
        width=_WIDTH,
        height=_HEIGHT,
        multipart_policy="strict_error",
        label_name_by_id={3: "pipette"},
    )
    assert lines == []
    assert len(dropped) == 1
    assert dropped[0].object_id == "obj_1"
    assert len(conversions) == 1
    assert conversions[0].component_count == 2


def test_minimal_bridge_yields_exactly_one_label_line_with_bridge_pixel_count() -> None:
    frame = _frame_with(_multipart_mask(), component_count=2)
    lines, conversions, dropped, _invalid = build_yolo_label_lines(
        frame=frame,
        width=_WIDTH,
        height=_HEIGHT,
        multipart_policy="minimal_bridge",
        label_name_by_id={3: "pipette"},
    )
    assert len(lines) == 1
    assert dropped == []
    assert len(conversions) == 1
    assert conversions[0].bridge_pixel_count is not None
    assert conversions[0].bridge_pixel_count > 0


def test_split_components_yields_one_label_line_per_component() -> None:
    frame = _frame_with(_multipart_mask(), component_count=2)
    lines, conversions, dropped, _invalid = build_yolo_label_lines(
        frame=frame,
        width=_WIDTH,
        height=_HEIGHT,
        multipart_policy="split_components",
        label_name_by_id={3: "pipette"},
    )
    assert len(lines) == 2
    assert all(line.startswith("3 ") for line in lines)
    assert dropped == []
    assert len(conversions) == 1  # one MultipartConversion record per logical instance


def test_build_dataset_yaml_lists_all_labels_sorted_by_id() -> None:
    yaml_text = build_dataset_yaml(
        label_name_by_id={2: "culture_flask", 0: "conical_tube_50ml"}, has_val_split=True
    )
    assert "train: images/train" in yaml_text
    assert "val: images/val" in yaml_text
    assert yaml_text.index("0: conical_tube_50ml") < yaml_text.index("2: culture_flask")


def test_build_dataset_yaml_omits_val_when_no_val_split() -> None:
    yaml_text = build_dataset_yaml(label_name_by_id={0: "pipette"}, has_val_split=False)
    assert "val:" not in yaml_text


def test_bbox_lines_are_normalized_center_and_size_of_the_mask() -> None:
    frame = _frame_with(_single_part_mask(), component_count=1)
    lines, invalid = build_yolo_bbox_lines(frame=frame, width=_WIDTH, height=_HEIGHT)
    # mask[8:12, 15:25] -> x=15, y=8, w=10, h=4 in a 40x20 frame
    assert lines == ["3 0.500000 0.500000 0.250000 0.200000"]
    assert invalid == []


def test_bbox_lines_cover_all_components_of_a_multipart_instance_with_one_box() -> None:
    frame = _frame_with(_multipart_mask(), component_count=2)
    lines, invalid = build_yolo_bbox_lines(frame=frame, width=_WIDTH, height=_HEIGHT)
    # components span x 2..37 -> one 36px-wide box, not one per component
    assert lines == ["3 0.500000 0.500000 0.900000 0.200000"]
    assert invalid == []


def test_bbox_lines_skip_an_empty_mask_and_report_it() -> None:
    frame = _frame_with(np.zeros((_HEIGHT, _WIDTH), dtype=bool), component_count=0)
    lines, invalid = build_yolo_bbox_lines(frame=frame, width=_WIDTH, height=_HEIGHT)
    assert lines == []
    assert len(invalid) == 1
