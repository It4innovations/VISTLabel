# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Requested YOLO segmentation export and its three multipart
compatibility policies. The standard Ultralytics polygon TXT format has no
way to say "these two disconnected contours are one instance" -- every
policy here is a different, explicit answer to that limitation; none of
them ever modify the canonical visible mask.

The detection variant (:func:`build_yolo_bbox_lines`) has no such
limitation: a multipart instance simply gets one box around all of its
components, so the multipart policies do not apply to it.
"""

from __future__ import annotations

from typing import Literal

from vistlabel.domain.exports import DroppedInstance, MultipartConversion
from vistlabel.domain.frames import FrameAnnotation
from vistlabel.exports.polygon import (
    bridge_mask,
    mask_bbox_xywh,
    mask_to_contours,
    normalize_polygon,
)
from vistlabel.persistence.rle import decode_mask

YoloMultipartPolicy = Literal["strict_error", "minimal_bridge", "split_components"]


def build_yolo_label_lines(
    *,
    frame: FrameAnnotation,
    width: int,
    height: int,
    multipart_policy: YoloMultipartPolicy,
    label_name_by_id: dict[int, str],
) -> tuple[list[str], list[MultipartConversion], list[DroppedInstance], list[str]]:
    """Returns ``(label_lines, multipart_conversions, dropped_instances,
    invalid_polygon_notes)`` for one frame."""
    lines: list[str] = []
    multipart_conversions: list[MultipartConversion] = []
    dropped: list[DroppedInstance] = []
    invalid_polygons: list[str] = []

    for obj in frame.objects:
        mask = decode_mask(obj.visible_mask_rle)
        label_name = label_name_by_id.get(obj.label_id, f"label_{obj.label_id}")

        if obj.connected_component_count <= 1:
            contours = mask_to_contours(mask)
            if not contours:
                invalid_polygons.append(
                    f"frame {frame.frame_index} object {obj.object_id}: mask has no "
                    "traceable contour (component_count=1 but empty/degenerate mask)"
                )
                continue
            polygon = normalize_polygon(contours[0], width, height)
            lines.append(_format_line(obj.label_id, polygon))
            continue

        # Multipart: obj.connected_component_count >= 2.
        conversion = MultipartConversion(
            frame_index=frame.frame_index,
            object_id=obj.object_id,
            label_name=label_name,
            component_count=obj.connected_component_count,
        )
        multipart_conversions.append(conversion)

        if multipart_policy == "strict_error":
            dropped.append(
                DroppedInstance(
                    frame_index=frame.frame_index,
                    object_id=obj.object_id,
                    reason=(
                        "multipart instance excluded from YOLO export under the "
                        "strict_error policy (exact mask remains in the COCO export)"
                    ),
                )
            )
        elif multipart_policy == "minimal_bridge":
            bridged, bridge_pixels = bridge_mask(mask)
            contours = mask_to_contours(bridged)
            if not contours:
                invalid_polygons.append(
                    f"frame {frame.frame_index} object {obj.object_id}: bridged mask has "
                    "no traceable contour"
                )
                continue
            polygon = normalize_polygon(contours[0], width, height)
            lines.append(_format_line(obj.label_id, polygon))
            multipart_conversions[-1] = conversion.model_copy(
                update={"bridge_pixel_count": bridge_pixels}
            )
        elif multipart_policy == "split_components":
            for contour in mask_to_contours(mask):
                polygon = normalize_polygon(contour, width, height)
                lines.append(_format_line(obj.label_id, polygon))
        else:
            raise ValueError(f"unknown multipart policy: {multipart_policy!r}")  # pragma: no cover

    return lines, multipart_conversions, dropped, invalid_polygons


def build_yolo_bbox_lines(
    *, frame: FrameAnnotation, width: int, height: int
) -> tuple[list[str], list[str]]:
    """Ultralytics detection labels (``class cx cy w h``, normalized) for one
    frame. Returns ``(label_lines, invalid_box_notes)``."""
    lines: list[str] = []
    invalid: list[str] = []
    for obj in frame.objects:
        box = mask_bbox_xywh(decode_mask(obj.visible_mask_rle))
        if box is None:
            invalid.append(
                f"frame {frame.frame_index} object {obj.object_id}: empty mask has no box"
            )
            continue
        x, y, box_width, box_height = box
        normalized = [
            (x + box_width / 2) / width,
            (y + box_height / 2) / height,
            box_width / width,
            box_height / height,
        ]
        lines.append(_format_line(obj.label_id, normalized))
    return lines, invalid


def _format_line(label_id: int, polygon: list[float]) -> str:
    coordinates = " ".join(f"{value:.6f}" for value in polygon)
    return f"{label_id} {coordinates}"


def build_dataset_yaml(
    *, label_name_by_id: dict[int, str], has_val_split: bool
) -> str:
    lines = ["path: .", "train: images/train"]
    if has_val_split:
        lines.append("val: images/val")
    lines.append("names:")
    for label_id in sorted(label_name_by_id):
        lines.append(f"  {label_id}: {label_name_by_id[label_id]}")
    return "\n".join(lines) + "\n"
