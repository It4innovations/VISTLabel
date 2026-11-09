# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Project label ontology (``labels.json``).

The ontology is per project and operator-defined: a project is created with
no labels at all and grows one entry per distinct object name typed in the
annotation UI. ``label_id`` is a label's stable identity: objects and frame
annotations store it, so it is allocated in order of first use and never
reassigned while in use (a label leaves the list when its last object is
deleted or relabeled, see ``api/objects.py``). The *exported* class number
is something else -- the label's position in ``labels`` -- so the operator
can reorder classes (``PUT .../labels`` with the same entries in a new
order) without rewriting a single annotation. (The backend protocol still
calls this value ``class_id`` -- a residual naming inconsistency with
``label_id``.)
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class LabelDefinition(BaseModel):
    label_id: int
    name: str
    color: str | None = None


class LabelsFile(BaseModel):
    schema_version: int = 1
    labels: list[LabelDefinition] = Field(default_factory=list)

    @field_validator("labels")
    @classmethod
    def _unique_ids_and_names(cls, labels: list[LabelDefinition]) -> list[LabelDefinition]:
        ids = [label.label_id for label in labels]
        if len(ids) != len(set(ids)):
            raise ValueError("label_id values must be unique")
        names = [label.name for label in labels]
        if len(names) != len(set(names)):
            raise ValueError("label names must be unique")
        return labels


#: Projects start with an empty ontology: the operator names objects freely in
#: the annotation UI and each new name is registered here on first use, so the
#: vocabulary is not capped by a built-in enumeration. ``config/
#: yolo_class_mapping.example.json`` ships example laboratory names as a
#: starting point for the YOLO class mapping, which is a separate, model-side
#: numbering that only has to agree with whatever ``label_id`` values a given
#: project ends up assigning.
DEFAULT_LABELS: tuple[LabelDefinition, ...] = ()


def default_labels_file() -> LabelsFile:
    return LabelsFile(labels=list(DEFAULT_LABELS))
