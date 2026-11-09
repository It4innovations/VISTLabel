# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Canonical logical-object model: one object_id, one class, one visible
mask that may span multiple disconnected components.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from vistlabel.domain.frames import FrameObjectAnnotation, OcclusionEdge, OcclusionState
from vistlabel.domain.labels import LabelDefinition


class AnnotationMode(StrEnum):
    MODAL_VISIBLE = "modal_visible"
    # Amodal support, if added, is a separate, independent layer; it must
    # never silently replace the visible mask.


class GeometryHint(BaseModel):
    """Optional metadata for association/review -- never part of the mask."""

    type: Literal["elongated_axis"] = "elongated_axis"
    polyline: list[tuple[float, float]]
    estimated_width_px: float | None = None


class ObjectRecord(BaseModel):
    object_id: str
    label_id: int
    label_name: str
    annotation_mode: AnnotationMode = AnnotationMode.MODAL_VISIBLE
    supports_multiple_components: bool = True
    occlusion_state: OcclusionState = OcclusionState.VISIBLE
    component_tracklets: list[str] = Field(default_factory=list)
    geometry_hint: GeometryHint | None = None
    active: bool = True
    #: Locked objects can't be deleted (alone or with their class) until
    #: unlocked; everything else, e.g. (de)activating, still works.
    locked: bool = False


class ObjectsFile(BaseModel):
    schema_version: int = 1
    #: Optimistic-concurrency counter. Additive with a default -- older
    #: objects.json files (schema_version 1, revision absent) remain
    #: readable without a migration.
    revision: int = 0
    objects: list[ObjectRecord] = Field(default_factory=list)


class DeletedObjectFrame(BaseModel):
    """What deleting an object took off one frame."""

    frame_index: int
    timestamp_seconds: float
    objects: list[FrameObjectAnnotation] = Field(default_factory=list)
    inactive_objects: list[FrameObjectAnnotation] = Field(default_factory=list)
    occlusion_edges: list[OcclusionEdge] = Field(default_factory=list)


class DeletedObject(BaseModel):
    """A deleted object, kept so Undo can restore it as it was: the record,
    its masks and occlusion edges on every frame, and its class with its
    place in the class order (the class leaves labels.json with its last
    object). ``annotations/deleted_objects/<object_id>.json``."""

    schema_version: int = 1
    record: ObjectRecord
    label: LabelDefinition | None = None
    label_position: int | None = None
    frames: list[DeletedObjectFrame] = Field(default_factory=list)
