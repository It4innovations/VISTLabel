# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Dataset validation checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vistlabel.validation.engine import (
    find_duplicate_annotations,
    find_missing_files,
    is_tiny_mask,
    validate_bbox_coordinates,
    validate_polygon_coordinates,
)


def test_validate_bbox_coordinates_accepts_a_valid_box() -> None:
    assert validate_bbox_coordinates([0.5, 0.5, 0.25, 0.2]) is None


def test_validate_bbox_coordinates_rejects_wrong_length_out_of_range_and_zero_size() -> None:
    assert validate_bbox_coordinates([0.5, 0.5, 0.25]) is not None
    assert validate_bbox_coordinates([0.5, 1.2, 0.25, 0.2]) is not None
    assert validate_bbox_coordinates([0.5, 0.5, 0.0, 0.2]) is not None


def test_validate_polygon_coordinates_accepts_a_valid_polygon() -> None:
    assert validate_polygon_coordinates([0.0, 0.0, 1.0, 0.0, 0.5, 1.0]) is None


def test_validate_polygon_coordinates_rejects_odd_length() -> None:
    assert validate_polygon_coordinates([0.0, 0.0, 1.0]) is not None


def test_validate_polygon_coordinates_rejects_too_few_points() -> None:
    assert validate_polygon_coordinates([0.0, 0.0, 1.0, 1.0]) is not None


def test_validate_polygon_coordinates_rejects_out_of_range_values() -> None:
    assert validate_polygon_coordinates([0.0, 0.0, 1.5, 0.0, 0.5, 1.0]) is not None
    assert validate_polygon_coordinates([-0.1, 0.0, 1.0, 0.0, 0.5, 1.0]) is not None


def test_is_tiny_mask_flags_small_areas() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[0, 0:2] = True  # area 2
    assert is_tiny_mask(mask, threshold_pixels=4) is True


def test_is_tiny_mask_false_for_normal_sized_mask() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[0:5, 0:5] = True  # area 25
    assert is_tiny_mask(mask, threshold_pixels=4) is False


def test_find_missing_files(tmp_path: Path) -> None:
    present = tmp_path / "present.txt"
    present.write_text("x", encoding="utf-8")
    missing = tmp_path / "missing.txt"
    result = find_missing_files([present, missing])
    assert result == [str(missing)]


def test_find_duplicate_annotations_detects_identical_masks_on_different_objects() -> None:
    entries = [(0, "obj_a", "same-rle"), (0, "obj_b", "same-rle"), (1, "obj_c", "other-rle")]
    duplicates = find_duplicate_annotations(entries)
    assert len(duplicates) == 1
    assert "obj_a" in duplicates[0] and "obj_b" in duplicates[0]


def test_find_duplicate_annotations_no_flag_for_distinct_masks() -> None:
    entries = [(0, "obj_a", "rle-a"), (0, "obj_b", "rle-b")]
    assert find_duplicate_annotations(entries) == []
