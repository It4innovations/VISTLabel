# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Frame annotation schema and shared occlusion vocabulary."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import AliasChoices, BaseModel, Field


class OcclusionState(StrEnum):
    VISIBLE = "visible"
    PARTIALLY_OCCLUDED = "partially_occluded"
    FULLY_OCCLUDED = "fully_occluded"


class ReviewStatus(StrEnum):
    UNREVIEWED = "unreviewed"
    NEEDS_REVIEW = "needs_review"
    REVIEWED = "reviewed"


class ReviewReason(StrEnum):
    """Review-queue reasons. Machine-set wherever the signal is derivable
    from the mask/score itself (see ``annotations/occlusion.py``); backend
    disagreement comes from the YOLO+SAM reconciliation and manual review
    sampling from the configured sampling stride. The vocabulary is kept
    stable and additive rather than growing ad hoc."""

    LOW_MODEL_SCORE = "low_model_score"
    LARGE_AREA_CHANGE = "large_area_change"
    ABRUPT_CENTROID_DISPLACEMENT = "abrupt_centroid_displacement"
    COMPONENT_COUNT_CHANGE = "component_count_change"
    OBJECT_OVERLAP_CONFLICT = "object_overlap_conflict"
    IDENTITY_AMBIGUITY = "identity_ambiguity"
    DISAPPEARANCE = "disappearance"
    REAPPEARANCE = "reappearance"
    COMPLETE_OCCLUSION = "complete_occlusion"
    BACKEND_DISAGREEMENT = "backend_disagreement"
    MASK_TOUCHES_FRAME_BORDER = "mask_touches_frame_border"
    MANUAL_REVIEW_SAMPLING = "manual_review_sampling"


class RleMask(BaseModel):
    """Canonical COCO-compatible compressed RLE.

    ``size`` is ``(height, width)``; ``counts`` is the ASCII-compressed COCO
    RLE string. Never store masks as YAML, and never store only a YOLO
    polygon as the canonical representation.
    """

    size: tuple[int, int]
    counts: str


class BackendProvenance(BaseModel):
    """Provenance every model-generated or modified annotation retains."""

    backend_id: str
    model_name: str
    checkpoint_sha256: str | None = None
    code_revision: str | None = None
    operation_id: str


class FrameObjectAnnotation(BaseModel):
    object_id: str
    label_id: int
    visible_mask_rle: RleMask
    connected_component_count: int
    occlusion_state: OcclusionState = OcclusionState.VISIBLE
    source: str
    confidence: float | None = None
    manually_modified: bool = False
    component_tracklet_ids: list[str] = Field(default_factory=list)
    backend_provenance: BackendProvenance | None = None
    #: Machine-detected review triggers for *this object on this frame*.
    #: Additive field -- absent on older frame JSON, defaults to empty, no
    #: migration needed.
    review_reasons: list[ReviewReason] = Field(default_factory=list)


class OcclusionEdge(BaseModel):
    occluder_object_id: str
    occluded_object_id: str
    relation: Literal["in_front_of"] = "in_front_of"


class FrameAnnotation(BaseModel):
    schema_version: int = 1
    frame_index: int
    timestamp_seconds: float
    revision: int = 0
    review_status: ReviewStatus = ReviewStatus.UNREVIEWED
    #: Free-text review reason/comment for the frame. Additive field --
    #: defaults to None so existing frame JSON remains readable without a
    #: migration.
    review_comment: str | None = None
    objects: list[FrameObjectAnnotation] = Field(default_factory=list)
    occlusion_edges: list[OcclusionEdge] = Field(default_factory=list)
    #: Objects deactivated on this frame ("Deactivate/Activate" in the
    #: sidebar), with the mask they had, so Activate can restore it. Not
    #: part of ``objects``, so nothing exports, renders or queues them;
    #: propagation leaves them alone like a manual edit. Additive field,
    #: defaults to empty. Briefly stored as ``hidden_objects``, still read.
    inactive_objects: list[FrameObjectAnnotation] = Field(
        default_factory=list,
        validation_alias=AliasChoices("inactive_objects", "hidden_objects"),
    )
    #: "Delete frame": left out of every export (positives and sampled
    #: negatives) and the review queue. The video frame itself stays --
    #: frame indices are positions in the video -- and so do the masks, so
    #: restoring the frame is lossless. Additive field, defaults to False.
    excluded: bool = False

    @property
    def is_annotated(self) -> bool:
        """At least one non-empty mask -- an erased (empty) one alone
        doesn't count -- on a frame not deleted from the dataset. The one
        definition behind the overview's count, the "annotated" export
        policy and "mark all annotated frames reviewed"."""
        return not self.excluded and any(
            obj.connected_component_count > 0 for obj in self.objects
        )
