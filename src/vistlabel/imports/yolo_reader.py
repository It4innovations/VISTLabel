# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""YOLO reader (Ultralytics TXT labels): ``images/<split>/x.jpg`` with
``labels/<split>/x.txt`` (Ultralytics, and VISTLabel's own export), or
``<split>/images`` + ``<split>/labels`` (Roboflow). A label line with four
values is a box (``class cx cy w h``), with six or more a polygon (``class
x1 y1 x2 y2 ...``); all coordinates are normalized to ``[0, 1]``.

Class names come from ``data.yaml`` / ``dataset.yaml`` (``names`` as a list
or an ``{index: name}`` map), else ``classes.txt`` / ``*.names`` (one per
line). YOLO has no identity across images.
"""

from __future__ import annotations

import posixpath

import yaml

from vistlabel.imports.archive import DatasetArchive, DatasetImportError, split_from_path
from vistlabel.imports.dataset import DatasetAnnotation, DatasetImage, ImportWarnings, ParsedDataset


def label_member_for(image_member: str) -> str:
    """Ultralytics' own rule: the last ``images`` folder becomes ``labels``;
    an image outside any ``images`` folder keeps its label beside it."""
    parts = image_member.split("/")
    for index in range(len(parts) - 2, -1, -1):
        if parts[index] == "images":
            parts[index] = "labels"
            break
    return posixpath.splitext("/".join(parts))[0] + ".txt"


def _depth_then_name(name: str) -> tuple[int, str]:
    return name.count("/"), name


def _names_from_yaml(archive: DatasetArchive) -> dict[int, str] | None:
    candidates = sorted(
        (n for n in archive.names if n.lower().endswith((".yaml", ".yml"))), key=_depth_then_name
    )
    # data.yaml / dataset.yaml first, then any other YAML that has names.
    candidates.sort(
        key=lambda n: posixpath.basename(n).lower() not in ("data.yaml", "dataset.yaml")
    )
    for name in candidates:
        try:
            document = yaml.safe_load(archive.read_text(name))
        except (yaml.YAMLError, DatasetImportError):
            continue
        names = document.get("names") if isinstance(document, dict) else None
        if isinstance(names, list):
            return {index: str(value) for index, value in enumerate(names)}
        if isinstance(names, dict):
            try:
                return {int(key): str(value) for key, value in names.items()}
            except (TypeError, ValueError):
                continue
    return None


def _names_from_list_file(archive: DatasetArchive) -> dict[int, str] | None:
    candidates = sorted(
        (
            n
            for n in archive.names
            if posixpath.basename(n).lower() == "classes.txt" or n.lower().endswith(".names")
        ),
        key=_depth_then_name,
    )
    if not candidates:
        return None
    lines = [line.strip() for line in archive.read_text(candidates[0]).splitlines()]
    return {index: line for index, line in enumerate(line for line in lines if line)}


def _parse_label_file(
    text: str, width: int, height: int, member: str, warnings: ImportWarnings
) -> list[DatasetAnnotation]:
    annotations = []
    for line in text.splitlines():
        tokens = line.split()
        if not tokens:
            continue
        try:
            class_value = float(tokens[0])
            values = [float(token) for token in tokens[1:]]
        except ValueError:
            warnings.add("unparseable label lines were skipped", member)
            continue
        if not class_value.is_integer() or class_value < 0:
            warnings.add("label lines with an invalid class index were skipped", member)
            continue
        label_id = int(class_value)
        if len(values) == 4:
            cx, cy, box_width, box_height = values
            box = (
                (cx - box_width / 2) * width,
                (cy - box_height / 2) * height,
                box_width * width,
                box_height * height,
            )
            if box[2] <= 0 or box[3] <= 0:
                warnings.add("zero-size boxes were skipped", member)
                continue
            annotations.append(DatasetAnnotation(label_id=label_id, bbox=box))
        elif len(values) >= 6 and len(values) % 2 == 0:
            polygon = [
                value * (width if index % 2 == 0 else height) for index, value in enumerate(values)
            ]
            annotations.append(DatasetAnnotation(label_id=label_id, polygons=[polygon]))
        else:
            warnings.add("label lines that are neither a box nor a polygon were skipped", member)
    return annotations


def read_yolo(archive: DatasetArchive) -> ParsedDataset:
    labelled = {image: label_member_for(image) for image in archive.image_names}
    labelled = {image: label for image, label in labelled.items() if archive.has(label)}
    names = _names_from_yaml(archive) or _names_from_list_file(archive)
    if not labelled and names is None:
        raise DatasetImportError(
            "found no COCO annotation file and no YOLO labels (labels/<split>/<image>.txt) or "
            "data.yaml -- is this a COCO or YOLO dataset?"
        )

    warnings = ImportWarnings()
    images = []
    used_labels: set[int] = set()
    for member in archive.image_names:
        parts = member.split("/")
        # An image outside any images/ folder is part of the dataset only if
        # it has a label file -- otherwise it is probably a stray preview.
        if "images" not in parts[:-1] and member not in labelled:
            continue
        image = DatasetImage(member=member, split=split_from_path(parts[:-1]))
        if member in labelled:
            width, height = archive.image_size(member)
            image.annotations = _parse_label_file(
                archive.read_text(labelled[member]), width, height, member, warnings
            )
            used_labels.update(annotation.label_id for annotation in image.annotations)
        images.append(image)

    if not images:
        raise DatasetImportError("the YOLO dataset has no images (expected images/<split>/...)")
    labels = dict(names or {})
    if names is None:
        warnings.add("no data.yaml or classes.txt found -- classes are named class_<index>")
    for label_id in sorted(used_labels - set(labels)):
        if names is not None:
            warnings.add(
                "class indices missing from the class list were named class_<index>", str(label_id)
            )
        labels[label_id] = f"class_{label_id}"
    return ParsedDataset(format="yolo", labels=labels, images=images, warnings=warnings)
