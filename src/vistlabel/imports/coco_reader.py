# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""COCO reader: every ``*.json`` in the archive that has ``images`` and
``annotations`` lists is a COCO file -- one per split in the RF-DETR /
Roboflow layout (``train/_annotations.coco.json``), or
``annotations/instances_<split>.json`` next to ``images/<split>/``.

``segmentation`` may be polygons, uncompressed RLE or compressed RLE; an
empty one means a box-only annotation. ``track_id`` / ``instance_id`` (or
CVAT's ``attributes.track_id``) carries identity across images when present.
"""

from __future__ import annotations

import json
import posixpath
from collections import defaultdict

from vistlabel.imports.archive import DatasetArchive, DatasetImportError, split_from_path
from vistlabel.imports.dataset import (
    DatasetAnnotation,
    DatasetImage,
    ImportWarnings,
    LabelRegistry,
    ParsedDataset,
)


def _is_coco(document: object) -> bool:
    return (
        isinstance(document, dict)
        and isinstance(document.get("images"), list)
        and isinstance(document.get("annotations"), list)
    )


def find_coco_documents(archive: DatasetArchive) -> list[tuple[str, dict]]:
    documents = []
    for name in archive.names:
        if not name.lower().endswith(".json"):
            continue
        try:
            document = json.loads(archive.read_text(name))
        except (json.JSONDecodeError, DatasetImportError):
            # Not every JSON in a dataset is COCO (VISTLabel's own exports carry
            # a manifest and a validation report).
            continue
        if _is_coco(document):
            documents.append((name, document))
    return documents


def _track_id(annotation: dict) -> int | str | None:
    attributes = annotation.get("attributes")
    for value in (
        annotation.get("track_id"),
        annotation.get("instance_id"),
        attributes.get("track_id") if isinstance(attributes, dict) else None,
    ):
        if isinstance(value, int | str) and not isinstance(value, bool):
            return value
    return None


def _bbox(annotation: dict) -> tuple[float, float, float, float] | None:
    box = annotation.get("bbox")
    if not isinstance(box, list) or len(box) != 4:
        return None
    try:
        x, y, width, height = (float(value) for value in box)
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return x, y, width, height


def _parse_annotation(
    annotation: dict, label_id: int, warnings: ImportWarnings
) -> DatasetAnnotation | None:
    track_id = _track_id(annotation)
    segmentation = annotation.get("segmentation")
    if isinstance(segmentation, dict) and "counts" in segmentation and "size" in segmentation:
        return DatasetAnnotation(label_id=label_id, rle=segmentation, track_id=track_id)
    if isinstance(segmentation, list) and segmentation:
        polygons = [
            [float(value) for value in polygon]
            for polygon in segmentation
            if isinstance(polygon, list) and len(polygon) >= 6 and len(polygon) % 2 == 0
        ]
        if polygons:
            return DatasetAnnotation(label_id=label_id, polygons=polygons, track_id=track_id)
        warnings.add("annotations with no usable polygon were imported as their box")
    box = _bbox(annotation)
    if box is None:
        warnings.add("annotations with neither a segmentation nor a valid bbox were skipped")
        return None
    return DatasetAnnotation(label_id=label_id, bbox=box, track_id=track_id)


def _register_categories(
    document: dict, registry: LabelRegistry, used_category_ids: set[object]
) -> dict[object, int]:
    categories = [c for c in document.get("categories") or [] if isinstance(c, dict)]
    # Roboflow adds an unused "superclass" category (id 0) that every real
    # category names as its supercategory -- not a class anyone labelled.
    supercategories = {c.get("supercategory") for c in categories}
    label_by_category: dict[object, int] = {}
    for category in categories:
        category_id = category.get("id")
        name = str(category.get("name") or f"category_{category_id}").strip()
        if category_id not in used_category_ids and name in supercategories:
            continue
        preferred = category_id if isinstance(category_id, int) else None
        label_by_category[category_id] = registry.add(name, preferred)
    return label_by_category


def read_coco(archive: DatasetArchive) -> ParsedDataset | None:
    """``None`` if the archive holds no COCO file at all."""
    documents = find_coco_documents(archive)
    if not documents:
        return None

    registry = LabelRegistry()
    warnings = ImportWarnings()
    images: dict[str, DatasetImage] = {}

    for document_name, document in documents:
        base_dir = posixpath.dirname(document_name)
        split = split_from_path(document_name.split("/"))
        annotations_by_image: dict[object, list[dict]] = defaultdict(list)
        for annotation in document["annotations"]:
            if isinstance(annotation, dict):
                annotations_by_image[annotation.get("image_id")].append(annotation)
        label_by_category = _register_categories(
            document,
            registry,
            {a.get("category_id") for a in document["annotations"] if isinstance(a, dict)},
        )

        for image in document["images"]:
            if not isinstance(image, dict) or not isinstance(image.get("file_name"), str):
                warnings.add("image entries without a file_name were skipped")
                continue
            member = archive.resolve_image(image["file_name"], base_dir)
            if member is None:
                warnings.add(
                    "images listed in the COCO file but missing from the archive were skipped",
                    image["file_name"],
                )
                continue
            if member in images:
                warnings.add("images listed more than once were imported once", member)
                continue
            parsed = DatasetImage(member=member, split=split)
            for annotation in annotations_by_image.get(image.get("id"), []):
                if annotation.get("iscrowd"):
                    warnings.add("crowd (iscrowd=1) annotations were skipped")
                    continue
                label_id = label_by_category.get(annotation.get("category_id"))
                if label_id is None:
                    warnings.add("annotations with an unknown category_id were skipped")
                    continue
                entry = _parse_annotation(annotation, label_id, warnings)
                if entry is not None:
                    parsed.annotations.append(entry)
            images[member] = parsed

    if not images:
        raise DatasetImportError(
            "the COCO file(s) list no image that exists in the archive -- "
            "put the images in the ZIP next to the annotations"
        )
    return ParsedDataset(
        format="coco", labels=registry.names, images=list(images.values()), warnings=warnings
    )
