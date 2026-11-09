# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Mask -> YOLO polygon conversion and the ``minimal_bridge``/
``split_components`` multipart compatibility transforms.

Uses OpenCV's contour tracer (``opencv-python-headless``, a base
dependency -- polygon export is core, non-GPU-gated functionality, not
something that should require the ``gpu`` extra).
"""

from __future__ import annotations

import cv2
import numpy as np

#: A component smaller than this many pixels is almost certainly noise,
#: not a real annotation instance -- reported as a "tiny mask" rather than
#: silently exported as a valid polygon.
TINY_MASK_AREA_PIXELS = 4


def mask_bbox_xywh(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Tight pixel box ``(x, y, width, height)`` around every set pixel --
    all components of a multipart mask together, since they are one
    instance -- or ``None`` for an empty mask."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def mask_to_contours(mask: np.ndarray) -> list[np.ndarray]:
    """One (N, 2) int array of (x, y) pixel coordinates per connected
    component, external boundary only (the *visible* mask is the export
    target, never a filled amodal guess)."""
    mask_u8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [contour.reshape(-1, 2) for contour in contours if len(contour) >= 3]


def normalize_polygon(contour: np.ndarray, width: int, height: int) -> list[float]:
    """Flattened ``[x1, y1, x2, y2, ...]`` in ``[0, 1]`` (the normalized
    YOLO polygon line format)."""
    normalized: list[float] = []
    for x, y in contour:
        normalized.append(float(x) / width)
        normalized.append(float(y) / height)
    return normalized


def _nearest_points(
    source_mask: np.ndarray, target_mask: np.ndarray
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Nearest (row, col) pixel pair between two disjoint binary masks.

    Deliberately *not* a pairwise distance matrix over the two pixel sets:
    a single 4K component holds hundreds of thousands of pixels, so that
    matrix reaches hundreds of GiB and the export dies allocating it. One
    distance transform costs a single frame-sized float32 array instead,
    whatever the components' pixel counts.
    """
    # distanceTransform measures each nonzero pixel's distance to the
    # nearest *zero* pixel, so the target set has to be the zeros.
    inverted_target = np.where(target_mask, 0, 255).astype(np.uint8)
    distance_to_target = cv2.distanceTransform(
        inverted_target, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )

    source_rows, source_cols = np.nonzero(source_mask)
    closest_source = int(np.argmin(distance_to_target[source_rows, source_cols]))
    source_point = (int(source_rows[closest_source]), int(source_cols[closest_source]))

    # The transform gives the distance but not which target pixel produced
    # it -- recover that with a one-against-many search, which is linear.
    target_points = np.column_stack(np.nonzero(target_mask)).astype(np.int64)
    offsets = target_points - np.array(source_point, dtype=np.int64)
    closest_target = int(np.argmin((offsets**2).sum(axis=1)))
    target_point = (int(target_points[closest_target][0]), int(target_points[closest_target][1]))

    return source_point, target_point


def bridge_mask(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """The ``minimal_bridge`` policy: connect every disconnected component
    with a minimal-width straight path so exactly one external contour
    remains. Returns ``(bridged_mask, bridge_pixel_count)``.

    Never mutates the input -- the canonical visible mask is never
    modified. The bridge is purely an export-time compatibility
    transform for the lossy YOLO TXT format; COCO export always uses the
    original, unbridged mask.
    """
    mask_u8 = mask.astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(mask_u8, connectivity=4)
    component_count = num_labels - 1  # label 0 is background
    if component_count <= 1:
        return mask.copy(), 0

    bridged = mask_u8.copy()
    bridge_pixels = 0
    # Iteratively bridge every other component to component 1's growing
    # union -- each new bridge only needs to reach *some* already-bridged
    # pixel, not component 1 specifically, so later bridges can be shorter.
    bridged_component_mask = labels == 1
    for label in range(2, num_labels):
        start, end = _nearest_points(labels == label, bridged_component_mask)
        before = int(bridged.sum())
        # cv2.line takes (x, y) = (col, row).
        cv2.line(bridged, (start[1], start[0]), (end[1], end[0]), color=1, thickness=1)
        bridge_pixels += int(bridged.sum()) - before
        bridged_component_mask = bridged_component_mask | (labels == label) | (bridged > mask_u8)

    return bridged.astype(bool), bridge_pixels
