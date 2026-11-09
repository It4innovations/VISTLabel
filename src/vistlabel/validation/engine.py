# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Export/dataset validation checks. Pure functions over
already-built export data -- the orchestrator
(:mod:`vistlabel.exports.service`) collects their results into a
:class:`~vistlabel.domain.exports.ValidationReport`, the same "pure checks,
orchestrator assembles the report" split already used for review reasons
in :mod:`vistlabel.annotations.occlusion`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vistlabel.exports.polygon import TINY_MASK_AREA_PIXELS


def validate_polygon_coordinates(polygon: list[float]) -> str | None:
    """Returns an error message if the polygon is malformed, else
    ``None``."""
    if len(polygon) < 6 or len(polygon) % 2 != 0:
        return f"polygon has {len(polygon)} coordinates (need an even number, >= 6)"
    if any(value < 0.0 or value > 1.0 for value in polygon):
        return "polygon coordinates must be normalized to [0, 1]"
    return None


def validate_bbox_coordinates(box: list[float]) -> str | None:
    """Same check as :func:`validate_polygon_coordinates` for a YOLO
    detection line's ``cx cy w h``."""
    if len(box) != 4:
        return f"box has {len(box)} coordinates (need exactly 4: cx cy w h)"
    if any(value < 0.0 or value > 1.0 for value in box):
        return "box coordinates must be normalized to [0, 1]"
    if box[2] <= 0.0 or box[3] <= 0.0:
        return "box has zero width or height"
    return None


def is_tiny_mask(mask: np.ndarray, *, threshold_pixels: int = TINY_MASK_AREA_PIXELS) -> bool:
    """Flags tiny masks -- almost certainly noise, not a real annotated
    instance."""
    return int(mask.sum()) < threshold_pixels


def find_missing_files(expected_paths: list[Path]) -> list[str]:
    """Detects missing label files: a post-write sanity check that every
    file the manifest/labels claim to have written
    actually landed on disk."""
    return [str(path) for path in expected_paths if not path.is_file()]


def find_duplicate_annotations(
    entries: list[tuple[int, str, str]],
) -> list[str]:
    """``entries`` is ``(frame_index, object_id, rle_counts)`` tuples.
    Flags two *different* objects on the same frame sharing byte-for-byte
    identical mask data -- a real duplicate-annotation data-quality issue,
    not something that should happen through any normal editing path."""
    seen: dict[tuple[int, str], str] = {}
    duplicates: list[str] = []
    for frame_index, object_id, rle_counts in entries:
        key = (frame_index, rle_counts)
        existing = seen.get(key)
        if existing is not None and existing != object_id:
            duplicates.append(
                f"frame {frame_index}: object {object_id!r} and {existing!r} have "
                "byte-identical mask data"
            )
        else:
            seen[key] = object_id
    return duplicates
