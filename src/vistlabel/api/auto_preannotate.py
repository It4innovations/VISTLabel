# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Automatic class-aware pre-annotation endpoint.

Runs ``backend.auto_preannotate`` (the YOLO-boxes-into-SAM orchestration)
as a cancellable background job, mirroring ``api/propagation.py``. The one
thing specific to this endpoint: a backend detection only carries a *label
name* in ``backend_metadata`` (the class mapping file is name-based,
administrator-owned and shared across projects), so this job resolves it
against *this* project's own ``labels.json`` and silently skips -- never
guesses -- any detection whose resolved name isn't a configured project
label: unknown or unmapped classes are rejected.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from vistlabel.annotations import objects_service
from vistlabel.annotations.frames_service import apply_frame_annotation
from vistlabel.annotations.occlusion import compute_occlusion_state, frame_local_review_reasons
from vistlabel.api.deps import require_frame_index, require_project_with_video
from vistlabel.domain.frames import (
    FrameAnnotation,
    FrameObjectAnnotation,
    ReviewReason,
    ReviewStatus,
)
from vistlabel.jobs.manager import JobContext
from vistlabel.persistence.paths import source_video_path
from vistlabel.persistence.rle import encode_mask
from vistlabel.segmentation.base import MaskPrediction
from vistlabel.segmentation.registry import UnknownBackendError

router = APIRouter(prefix="/api/v1/projects", tags=["auto-preannotate"])


class AutoPreannotateRequest(BaseModel):
    backend_id: str = "yolo_sam"
    frame_indices: list[int]
    #: Backend-specific class ids to look for (YOLO class indices for
    #: ``yolo_sam``); empty means "everything the backend's
    #: configured mapping knows about".
    requested_class_ids: list[int] = []


class JobIdResponse(BaseModel):
    job_id: str


@router.post("/{project_id}/auto-preannotate", status_code=202)
def start_auto_preannotate(
    project_id: str, payload: AutoPreannotateRequest, request: Request
) -> JobIdResponse:
    _project, pdir, timestamps = require_project_with_video(request, project_id)
    for frame_index in payload.frame_indices:
        require_frame_index(frame_index, timestamps)

    try:
        backend = request.app.state.registry.get(payload.backend_id)
    except UnknownBackendError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    descriptor = backend.descriptor()
    if not descriptor.available:
        raise HTTPException(
            status_code=409,
            detail=(
                f"backend {payload.backend_id!r} is unavailable: {descriptor.unavailable_reason}"
            ),
        )

    session_id = request.app.state.backend_sessions.get_or_open(
        backend, project_id, source_video_path(pdir)
    )
    total_keyframes = max(len(payload.frame_indices), 1)
    job_manager = request.app.state.job_manager
    project_store = request.app.state.project_store

    def work(context: JobContext) -> None:
        labels_file = project_store.get_labels(project_id)
        label_id_by_name = {label.name: label.label_id for label in labels_file.labels}
        canonical_object_id_by_backend_id: dict[str, str] = {}
        completed_keyframes: set[int] = set()

        for prediction in backend.auto_preannotate(
            session_id, payload.frame_indices, payload.requested_class_ids
        ):
            context.raise_if_cancelled()
            metadata = prediction.backend_metadata
            label_name = metadata.get("label_name")
            label_id = label_id_by_name.get(label_name) if isinstance(label_name, str) else None
            if label_id is None:
                # A class this project's ontology doesn't have is
                # rejected, never silently guessed at.
                continue

            backend_object_id = prediction.object_id
            canonical_object_id = canonical_object_id_by_backend_id.get(backend_object_id)
            if canonical_object_id is None:
                record = objects_service.create_object(pdir, label_id, label_name)
                canonical_object_id = record.object_id
                canonical_object_id_by_backend_id[backend_object_id] = canonical_object_id

            raw_review_reasons = metadata.get("review_reasons", [])
            extra_review_reasons = (
                [ReviewReason(value) for value in raw_review_reasons]
                if isinstance(raw_review_reasons, list)
                else []
            )
            review_reasons = (
                frame_local_review_reasons(prediction.visible_mask, prediction.score)
                + extra_review_reasons
            )

            def _mutate(
                frame: FrameAnnotation,
                prediction: MaskPrediction = prediction,
                canonical_object_id: str = canonical_object_id,
                label_id: int = label_id,
                review_reasons: list[ReviewReason] = review_reasons,
            ) -> FrameAnnotation:
                entry = FrameObjectAnnotation(
                    object_id=canonical_object_id,
                    label_id=label_id,
                    visible_mask_rle=encode_mask(prediction.visible_mask),
                    connected_component_count=prediction.component_count,
                    occlusion_state=compute_occlusion_state(prediction.component_count),
                    source=f"{payload.backend_id}_auto_preannotate",
                    confidence=prediction.score,
                    manually_modified=False,
                    review_reasons=review_reasons,
                )
                objects = [o for o in frame.objects if o.object_id != entry.object_id]
                objects.append(entry)
                changes: dict[str, object] = {"objects": objects}
                if review_reasons and frame.review_status == ReviewStatus.UNREVIEWED:
                    changes["review_status"] = ReviewStatus.NEEDS_REVIEW
                return frame.model_copy(update=changes)

            apply_frame_annotation(
                pdir, prediction.frame_index, timestamps[prediction.frame_index], _mutate
            )
            completed_keyframes.add(prediction.frame_index)
            job_manager.report_progress(
                project_id, context.job_id, min(1.0, len(completed_keyframes) / total_keyframes)
            )

    job_id = job_manager.submit(project_id, "auto_preannotate", work)
    return JobIdResponse(job_id=job_id)
