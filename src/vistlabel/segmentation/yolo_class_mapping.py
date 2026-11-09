# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""``yolo_class_mapping.json`` loading and validation: YOLO class indices
are mapped to project labels through an explicit mapping file, and unknown
or unmapped classes are rejected.

Deliberately maps YOLO class index -> label *name*, not directly to a
project's numeric ``label_id``: the mapping file is administrator-owned and
shared across projects, while ``label_id`` values are per-project
(``labels.json``). Resolving name -> ``label_id`` for a
specific project happens at the API/job layer (mirrors how
:class:`~vistlabel.segmentation.base.ConceptProposal` carries a
``suggested_label_name`` rather than a resolved ``label_id``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


class ClassMappingError(ValueError):
    """Raised for a malformed or empty ``yolo_class_mapping.json``."""


@dataclass(frozen=True)
class YoloClassMapping:
    schema_version: int
    #: YOLO class index -> label name.
    model_classes: dict[int, str]


def load_class_mapping(path: Path) -> YoloClassMapping:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ClassMappingError(f"{path}: not valid JSON: {exc}") from exc

    if not isinstance(raw, dict):
        raise ClassMappingError(f"{path}: expected a JSON object at the top level")
    schema_version = raw.get("schema_version")
    if schema_version != 1:
        raise ClassMappingError(f"{path}: unsupported schema_version {schema_version!r}")
    raw_classes = raw.get("model_classes")
    if not isinstance(raw_classes, dict) or not raw_classes:
        raise ClassMappingError(f"{path}: 'model_classes' must be a non-empty object")

    model_classes: dict[int, str] = {}
    for key, value in raw_classes.items():
        try:
            class_id = int(key)
        except (TypeError, ValueError) as exc:
            raise ClassMappingError(f"{path}: model_classes key {key!r} is not an integer") from exc
        if not isinstance(value, str) or not value:
            raise ClassMappingError(f"{path}: model_classes[{key!r}] must be a non-empty string")
        model_classes[class_id] = value

    names = list(model_classes.values())
    if len(names) != len(set(names)):
        raise ClassMappingError(f"{path}: model_classes values must be unique label names")

    return YoloClassMapping(schema_version=schema_version, model_classes=model_classes)
