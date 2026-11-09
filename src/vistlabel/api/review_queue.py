# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Review queue listing: each entry links directly to the relevant frame
and object.

A full scan of every persisted frame annotation, not an index -- consistent
with this project's filesystem-only, no-database persistence model at its
current scale.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

from vistlabel.api.deps import require_project_with_video
from vistlabel.domain.frames import FrameAnnotation, ReviewReason, ReviewStatus
from vistlabel.persistence.atomic import load_model_migrated
from vistlabel.persistence.paths import annotated_frame_paths

router = APIRouter(prefix="/api/v1/projects", tags=["review-queue"])


class ReviewQueueEntry(BaseModel):
    frame_index: int
    #: None for a frame flagged needs_review with no per-object reason
    #: (e.g. a manual review_comment with nothing machine-detected).
    object_id: str | None
    label_id: int | None
    review_reasons: list[ReviewReason]
    review_comment: str | None


@router.get("/{project_id}/review-queue")
def get_review_queue(project_id: str, request: Request) -> list[ReviewQueueEntry]:
    _project, pdir, _timestamps = require_project_with_video(request, project_id)
    entries: list[ReviewQueueEntry] = []
    for path in annotated_frame_paths(pdir):
        frame = load_model_migrated(path, FrameAnnotation, "frame_annotation")
        if frame.review_status != ReviewStatus.NEEDS_REVIEW or frame.excluded:
            continue
        flagged_objects = [o for o in frame.objects if o.review_reasons]
        if not flagged_objects:
            entries.append(
                ReviewQueueEntry(
                    frame_index=frame.frame_index,
                    object_id=None,
                    label_id=None,
                    review_reasons=[],
                    review_comment=frame.review_comment,
                )
            )
            continue
        for entry in flagged_objects:
            entries.append(
                ReviewQueueEntry(
                    frame_index=frame.frame_index,
                    object_id=entry.object_id,
                    label_id=entry.label_id,
                    review_reasons=entry.review_reasons,
                    review_comment=frame.review_comment,
                )
            )
    entries.sort(key=lambda entry: entry.frame_index)
    return entries
