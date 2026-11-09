# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Export manifest / validation report schemas, populated by the export
service.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class ExportStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class FrameSelectionPolicy(StrEnum):
    """Supported frame-inclusion policies."""

    REVIEWED_ONLY = "reviewed_only"
    MANUALLY_EDITED_ONLY = "manually_edited_only"
    ANNOTATED_ONLY = "annotated_only"
    EVERY_NTH_REVIEWED = "every_nth_reviewed"
    #: Same stride over the frames ``annotated_only`` would export.
    EVERY_NTH_ANNOTATED = "every_nth_annotated"
    SELECTED_INTERVALS = "selected_intervals"
    #: All frames, with an explicit warning -- the warning is
    #: enforced by the API layer requiring ``acknowledge_all_frames_warning``
    #: on the request, not just by this enum value existing.
    ALL_FRAMES = "all_frames"


#: What each exported instance is written as: the exact mask (YOLO polygon /
#: COCO RLE) or only its tight bounding box, derived from that same mask.
#: The canonical annotation is always the mask either way.
AnnotationType = Literal["segmentation", "bbox"]

#: On-disk dataset structure. ``ultralytics`` is images/ + labels/ +
#: dataset.yaml (plus optional annotations/instances_*.json); ``coco`` is the
#: Roboflow COCO layout RF-DETR trains from directly: ``train/`` and
#: ``valid/``, each holding its images next to an ``_annotations.coco.json``.
DatasetLayout = Literal["ultralytics", "coco"]


class ExportManifest(BaseModel):
    schema_version: int = 1
    export_id: str
    project_id: str
    source_video_sha256: str | None = None
    created_at: datetime
    status: ExportStatus = ExportStatus.PENDING
    job_id: str | None = None
    frame_selection_policy: FrameSelectionPolicy
    yolo_multipart_policy: Literal["strict_error", "minimal_bridge", "split_components"]
    always_include_coco_exact_annotations: bool = True
    negative_frame_fraction: float = 0.0
    image_format: Literal["jpg", "png"] = "jpg"
    #: Defaults keep manifests written before these options existed loadable;
    #: those exports were always ultralytics-layout segmentation.
    annotation_type: AnnotationType = "segmentation"
    dataset_layout: DatasetLayout = "ultralytics"
    frame_count: int = 0
    train_frame_count: int = 0
    val_frame_count: int = 0
    negative_frame_count: int = 0
    error: str | None = None


class MultipartConversion(BaseModel):
    frame_index: int
    object_id: str
    label_name: str
    component_count: int
    #: Only meaningful for the minimal_bridge policy.
    bridge_pixel_count: int | None = None


class DroppedInstance(BaseModel):
    frame_index: int
    object_id: str
    reason: str


class ValidationReport(BaseModel):
    schema_version: int = 1
    export_id: str
    generated_at: datetime
    image_count: int = 0
    object_count_by_class: dict[str, int] = Field(default_factory=dict)
    negative_frame_count: int = 0
    train_frame_count: int = 0
    val_frame_count: int = 0
    missing_or_malformed_labels: list[str] = Field(default_factory=list)
    invalid_polygons: list[str] = Field(default_factory=list)
    tiny_masks: list[str] = Field(default_factory=list)
    multipart_instances: list[MultipartConversion] = Field(default_factory=list)
    yolo_bridge_conversions: list[MultipartConversion] = Field(default_factory=list)
    split_component_conversions: list[MultipartConversion] = Field(default_factory=list)
    dropped_instances: list[DroppedInstance] = Field(default_factory=list)
    frame_leakage_risks: list[str] = Field(default_factory=list)
    #: backend_id -> number of frame/object annotations produced by it.
    model_provenance_summary: dict[str, int] = Field(default_factory=dict)
    source_project_id: str = ""
    source_video_sha256: str | None = None
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
