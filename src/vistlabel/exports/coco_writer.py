# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Exact COCO instance-segmentation export. Always generated in addition
to the requested YOLO output (the ``always_include_coco_exact_annotations``
default), because it is the only
format this project's canonical RLE representation maps onto without any
lossy compatibility transform: COCO's compressed-RLE segmentation format
*is* :class:`~vistlabel.domain.frames.RleMask`, so disconnected visible
components, holes, and exact binary masks all pass through unchanged.

With ``annotation_type="bbox"`` the same file becomes a plain detection
dataset: ``segmentation`` is left empty and only the mask's tight box is
kept, which is what box-only trainers such as RF-DETR consume.
"""

from __future__ import annotations

from vistlabel.domain.exports import AnnotationType
from vistlabel.domain.frames import FrameAnnotation
from vistlabel.domain.labels import LabelsFile
from vistlabel.exports.polygon import mask_bbox_xywh
from vistlabel.persistence.rle import decode_mask


def build_coco_dataset(
    *,
    frame_indices: list[int],
    frame_annotations: dict[int, FrameAnnotation],
    labels: LabelsFile,
    image_filename_by_frame: dict[int, str],
    image_width: int,
    image_height: int,
    annotation_type: AnnotationType = "segmentation",
) -> dict:
    categories = [{"id": label.label_id, "name": label.name} for label in labels.labels]
    images = []
    annotations = []
    annotation_id = 1

    for frame_index in frame_indices:
        images.append(
            {
                "id": frame_index,
                "file_name": image_filename_by_frame[frame_index],
                "width": image_width,
                "height": image_height,
            }
        )
        frame = frame_annotations.get(frame_index)
        if frame is None:
            continue
        for obj in frame.objects:
            mask = decode_mask(obj.visible_mask_rle)
            box = mask_bbox_xywh(mask)
            if annotation_type == "bbox":
                # A zero-size box is not a detection target -- trainers
                # either filter it out or fail on it.
                if box is None:
                    continue
                segmentation: dict | list = []
            else:
                # Exact pass-through: RleMask's own shape already *is*
                # COCO's compressed-RLE segmentation format.
                segmentation = {
                    "size": list(obj.visible_mask_rle.size),
                    "counts": obj.visible_mask_rle.counts,
                }
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": frame_index,
                    "category_id": obj.label_id,
                    "segmentation": segmentation,
                    "area": int(mask.sum()),
                    "bbox": [float(value) for value in box or (0, 0, 0, 0)],
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    # info/licenses are optional in COCO, but some loaders (pycocotools'
    # loadRes in certain versions) index them unconditionally.
    return {
        "info": {"description": "VISTLabel export"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
