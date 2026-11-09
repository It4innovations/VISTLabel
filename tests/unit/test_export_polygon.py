# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Mask -> polygon conversion and the minimal_bridge multipart
compatibility transform."""

from __future__ import annotations

import numpy as np

from vistlabel.exports.polygon import bridge_mask, mask_to_contours, normalize_polygon


def _rect_mask(height: int, width: int, box: tuple[int, int, int, int]) -> np.ndarray:
    y0, y1, x0, x1 = box
    mask = np.zeros((height, width), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def test_single_component_mask_yields_one_contour() -> None:
    mask = _rect_mask(20, 20, (2, 8, 2, 8))
    contours = mask_to_contours(mask)
    assert len(contours) == 1
    assert contours[0].shape[1] == 2


def test_two_component_mask_yields_two_contours() -> None:
    mask = _rect_mask(20, 20, (2, 6, 2, 6)) | _rect_mask(20, 20, (12, 16, 12, 16))
    contours = mask_to_contours(mask)
    assert len(contours) == 2


def test_empty_mask_yields_no_contours() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    assert mask_to_contours(mask) == []


def test_normalize_polygon_scales_to_unit_range() -> None:
    contour = np.array([[0, 0], [10, 0], [10, 20], [0, 20]])
    normalized = normalize_polygon(contour, width=10, height=20)
    assert normalized == [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    assert all(0.0 <= v <= 1.0 for v in normalized)


def test_bridge_mask_merges_two_components_into_one_contour() -> None:
    mask = _rect_mask(20, 40, (8, 12, 2, 8)) | _rect_mask(20, 40, (8, 12, 32, 38))
    bridged, bridge_pixels = bridge_mask(mask)
    assert len(mask_to_contours(bridged)) == 1
    assert bridge_pixels > 0
    # The canonical mask is never mutated.
    assert mask.sum() == 4 * 6 + 4 * 6


def test_bridge_mask_merges_three_components_into_one_contour() -> None:
    mask = (
        _rect_mask(20, 60, (8, 12, 2, 6))
        | _rect_mask(20, 60, (8, 12, 27, 31))
        | _rect_mask(20, 60, (8, 12, 52, 56))
    )
    bridged, bridge_pixels = bridge_mask(mask)
    assert len(mask_to_contours(bridged)) == 1
    assert bridge_pixels > 0


def test_bridge_mask_handles_components_holding_hundreds_of_thousands_of_pixels() -> None:
    """Regression: bridging used a pairwise distance matrix over every pixel
    of both components, so a 4K frame asked for ~285 GiB and the export died."""
    mask = _rect_mask(2176, 3840, (200, 800, 300, 912)) | _rect_mask(
        2176, 3840, (1500, 1700, 2000, 2260)
    )
    bridged, bridge_pixels = bridge_mask(mask)
    assert len(mask_to_contours(bridged)) == 1
    assert bridge_pixels > 0


def test_bridge_mask_is_a_no_op_for_a_single_component() -> None:
    mask = _rect_mask(20, 20, (2, 8, 2, 8))
    bridged, bridge_pixels = bridge_mask(mask)
    assert bridge_pixels == 0
    np.testing.assert_array_equal(bridged, mask)


def test_bridge_mask_is_a_no_op_for_an_empty_mask() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    bridged, bridge_pixels = bridge_mask(mask)
    assert bridge_pixels == 0
    assert not bridged.any()
