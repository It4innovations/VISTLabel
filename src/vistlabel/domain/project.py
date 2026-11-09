# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Project state schema (``project.json``)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class ProjectStatus(StrEnum):
    """Workflow state of a project.

    ``pending`` is the only machine-owned value: it marks a project whose
    source video is not finalized yet and is never settable through the API.
    Everything from ``new`` on is an operator-set label -- deliberately
    independent of per-frame ``review_status``, which tracks a different
    granularity (one frame's QA, not the project's workflow stage).
    """

    #: created, source video not yet finalized
    PENDING = "pending"
    #: video finalized, annotation not started -- where upload lands a project
    NEW = "new"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    ARCHIVED = "archived"


#: Values an operator may set; ``pending`` is excluded because it describes
#: an upload that has not finished, which the API must not let anyone fake.
OPERATOR_SETTABLE_STATUSES: frozenset[ProjectStatus] = frozenset(
    {
        ProjectStatus.NEW,
        ProjectStatus.IN_PROGRESS,
        ProjectStatus.DONE,
        ProjectStatus.ARCHIVED,
    }
)


class VideoImportOptions(BaseModel):
    """Operator-chosen re-encode applied once at upload, before frame
    indexing. Everything ``None`` means "keep the native video" (it may
    still be re-encoded if it has a variable frame rate)."""

    #: Fit within this box, keeping the aspect ratio; never upscales.
    max_width: int | None = Field(default=None, ge=16, le=16384)
    max_height: int | None = Field(default=None, ge=16, le=16384)
    #: Resample to this constant frame rate; never above the source rate.
    target_fps: float | None = Field(default=None, gt=0, le=240)
    #: Keep frames 0, n, 2n, ... (the frame rate drops by the same factor).
    keep_every_nth: int | None = Field(default=None, ge=1, le=1000)

    @model_validator(mode="after")
    def _check_combinations(self) -> VideoImportOptions:
        if (self.max_width is None) != (self.max_height is None):
            raise ValueError("max_width and max_height must be given together")
        if self.target_fps is not None and self.keep_every_nth is not None:
            raise ValueError("give either target_fps or keep_every_nth, not both")
        return self

    @property
    def resizes(self) -> bool:
        return self.max_width is not None

    @property
    def changes_frame_rate(self) -> bool:
        return self.target_fps is not None or (self.keep_every_nth or 1) > 1


class OriginalVideoInfo(BaseModel):
    """The uploaded file as probed, before any import re-encode."""

    width: int
    height: int
    nominal_fps: float
    frame_count_hint: int
    sha256: str
    size_bytes: int


class SourceVideoInfo(BaseModel):
    original_filename: str
    stored_path: str
    sha256: str
    size_bytes: int
    video_codec: str
    pixel_format: str
    width: int
    height: int
    nominal_fps: float
    duration_seconds: float
    frame_count: int
    variable_frame_rate: bool
    #: Additive (absent on older project.json): what the operator asked for
    #: at upload, and the uploaded file's own properties when the stored
    #: video is a re-encode of it (``None`` when stored as uploaded).
    import_options: VideoImportOptions = Field(default_factory=VideoImportOptions)
    original: OriginalVideoInfo | None = None


class UiState(BaseModel):
    last_opened_frame: int = 0
    selected_object_id: str | None = None
    selected_backend_id: str = "fake"
    zoom: float = 1.0
    pan_x: float = 0.0
    pan_y: float = 0.0


class ProjectProgress(BaseModel):
    reviewed_frame_count: int = 0
    annotated_frame_count: int = 0
    total_frame_count: int = 0


class SplitLabel(StrEnum):
    """Label for a user-marked train / validation / excluded frame
    interval. Unassigned frames default to no label at export time (the
    frame-selection policy decides what to do with them), never silently
    to "train"."""

    TRAIN = "train"
    VAL = "val"
    EXCLUDED = "excluded"


class FrameRangeAssignment(BaseModel):
    #: Inclusive on both ends, matching frame_index semantics elsewhere in
    #: the codebase (e.g. propagation's start_frame/end_frame).
    start_frame: int
    end_frame: int
    label: SplitLabel


class ProjectState(BaseModel):
    schema_version: int = 1
    project_id: str
    name: str
    status: ProjectStatus = ProjectStatus.PENDING
    revision: int = 0
    created_at: datetime
    modified_at: datetime
    source: SourceVideoInfo | None = None
    ui_state: UiState = Field(default_factory=UiState)
    progress: ProjectProgress = Field(default_factory=ProjectProgress)
    #: User-marked train/validation/excluded frame intervals. Additive
    #: field: absent on older project.json, defaults to empty, no migration
    #: needed.
    split_assignments: list[FrameRangeAssignment] = Field(default_factory=list)
