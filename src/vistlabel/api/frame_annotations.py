# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Per-frame annotation endpoints, including the autosave flow.

A single ``PATCH`` accepts review-state changes and/or an object mask
upsert/removal in one optimistic-revision-checked request -- there is only
one endpoint for this resource, so the richer brush/eraser semantics live
in the request body rather than as separate routes (matching the precedent
set by the labels endpoint).
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from vistlabel.annotations.errors import RevisionConflictError
from vistlabel.annotations.frames_service import (
    apply_frame_annotation,
    get_frame_annotation,
    save_frame_annotation,
)
from vistlabel.annotations.occlusion import compute_occlusion_state, frame_local_review_reasons
from vistlabel.annotations.png_mask import (
    MaskDecodeError,
    decode_mask_png_base64,
    encode_mask_png_bytes,
)
from vistlabel.api.deps import require_frame_index, require_project_with_video
from vistlabel.domain.frames import (
    BackendProvenance,
    FrameAnnotation,
    FrameObjectAnnotation,
    OcclusionEdge,
    ReviewStatus,
)
from vistlabel.persistence.atomic import load_model_migrated
from vistlabel.persistence.jsonl import append_jsonl
from vistlabel.persistence.paths import annotated_frame_paths, events_log_path
from vistlabel.persistence.rle import count_connected_components, decode_mask, encode_mask

router = APIRouter(prefix="/api/v1/projects", tags=["frame-annotations"])


class FrameObjectMaskUpsert(BaseModel):
    object_id: str
    label_id: int
    visible_mask_png_base64: str
    source: str
    confidence: float | None = None
    manually_modified: bool = False
    backend_id: str | None = None
    model_name: str | None = None
    checkpoint_sha256: str | None = None
    code_revision: str | None = None


class FrameAnnotationPatchRequest(BaseModel):
    expected_revision: int
    review_status: ReviewStatus | None = None
    review_comment: str | None = None
    upsert_object: FrameObjectMaskUpsert | None = None
    remove_object_id: str | None = None
    #: Move an object's mask on this frame into ``inactive_objects`` / back.
    #: Unlike ``remove_object_id`` the mask is kept, so Deactivate is
    #: undoable.
    deactivate_object_id: str | None = None
    activate_object_id: str | None = None
    #: "Delete frame" (True) / restore it (False); see FrameAnnotation.excluded.
    excluded: bool | None = None
    #: Full replace, not a diff -- "Mark Occluder" sends the complete list
    #: for the frame.
    occlusion_edges: list[OcclusionEdge] | None = None


class MarkAnnotatedReviewedRequest(BaseModel):
    #: Count what would change without writing -- the UI confirms first.
    dry_run: bool = False


class MarkAnnotatedReviewedResponse(BaseModel):
    #: Annotated frames not yet reviewed: changed, or would be on a dry run.
    frame_count: int
    #: Of those, how many are flagged needs_review (in the review queue).
    needs_review_count: int


@router.post("/{project_id}/frames/mark-annotated-reviewed")
def mark_annotated_frames_reviewed(
    project_id: str, payload: MarkAnnotatedReviewedRequest, request: Request
) -> MarkAnnotatedReviewedResponse:
    """Review tab's "Mark all annotated frames reviewed": every frame that
    :attr:`FrameAnnotation.is_annotated` and isn't reviewed yet, including
    ones in the review queue (the dry run reports how many of those)."""
    _project, pdir, _timestamps = require_project_with_video(request, project_id)
    targets: list[FrameAnnotation] = []
    for path in annotated_frame_paths(pdir):
        frame = load_model_migrated(path, FrameAnnotation, "frame_annotation")
        if frame.is_annotated and frame.review_status is not ReviewStatus.REVIEWED:
            targets.append(frame)
    response = MarkAnnotatedReviewedResponse(
        frame_count=len(targets),
        needs_review_count=sum(
            1 for frame in targets if frame.review_status is ReviewStatus.NEEDS_REVIEW
        ),
    )
    if payload.dry_run or not targets:
        return response

    for frame in targets:
        apply_frame_annotation(
            pdir,
            frame.frame_index,
            frame.timestamp_seconds,
            lambda current: current.model_copy(update={"review_status": ReviewStatus.REVIEWED}),
        )
    append_jsonl(
        events_log_path(pdir),
        {
            "event": "annotated_frames_marked_reviewed",
            "frame_indices": [frame.frame_index for frame in targets],
        },
    )
    return response


@router.get("/{project_id}/frames/{frame_index}/annotations")
def get_frame_annotations(project_id: str, frame_index: int, request: Request) -> FrameAnnotation:
    _project, pdir, timestamps = require_project_with_video(request, project_id)
    timestamp = require_frame_index(frame_index, timestamps)
    return get_frame_annotation(pdir, frame_index, timestamp)


@router.get("/{project_id}/frames/{frame_index}/objects/{object_id}/mask.png")
def get_object_mask_png(
    project_id: str, frame_index: int, object_id: str, request: Request
) -> Response:
    """The canonical stored representation is COCO RLE, which the browser
    cannot decode directly; this re-exposes a saved mask as a renderable
    PNG so a reopened project can redraw previously accepted masks, not
    only ones created in the current browser session."""
    _project, pdir, timestamps = require_project_with_video(request, project_id)
    timestamp = require_frame_index(frame_index, timestamps)
    annotation = get_frame_annotation(pdir, frame_index, timestamp)
    entry = next((o for o in annotation.objects if o.object_id == object_id), None)
    if entry is None:
        raise HTTPException(
            status_code=404, detail=f"no mask for object {object_id!r} on this frame"
        )
    mask = decode_mask(entry.visible_mask_rle)
    return Response(content=encode_mask_png_bytes(mask), media_type="image/png")


@router.patch("/{project_id}/frames/{frame_index}/annotations")
def patch_frame_annotations(
    project_id: str, frame_index: int, payload: FrameAnnotationPatchRequest, request: Request
) -> FrameAnnotation:
    project, pdir, timestamps = require_project_with_video(request, project_id)
    timestamp = require_frame_index(frame_index, timestamps)
    assert project.source is not None  # guaranteed by require_project_with_video
    expected_shape = (project.source.height, project.source.width)

    def _mutate(frame: FrameAnnotation) -> FrameAnnotation:
        changes: dict[str, object] = {}
        if payload.review_status is not None:
            changes["review_status"] = payload.review_status
        if payload.review_comment is not None:
            changes["review_comment"] = payload.review_comment
        if payload.occlusion_edges is not None:
            changes["occlusion_edges"] = payload.occlusion_edges
        if payload.excluded is not None:
            changes["excluded"] = payload.excluded

        objects = list(frame.objects)
        inactive = list(frame.inactive_objects)
        if payload.upsert_object is not None:
            update = payload.upsert_object
            mask = decode_mask_png_base64(
                update.visible_mask_png_base64, expected_shape=expected_shape
            )
            component_count = count_connected_components(mask)
            review_reasons = frame_local_review_reasons(mask, update.confidence)
            provenance = None
            if update.backend_id is not None:
                provenance = BackendProvenance(
                    backend_id=update.backend_id,
                    model_name=update.model_name or update.backend_id,
                    checkpoint_sha256=update.checkpoint_sha256,
                    code_revision=update.code_revision,
                    operation_id=uuid.uuid4().hex,
                )
            new_entry = FrameObjectAnnotation(
                object_id=update.object_id,
                label_id=update.label_id,
                visible_mask_rle=encode_mask(mask),
                connected_component_count=component_count,
                # Derived from the mask topology, not settable
                # by the client -- otherwise it can silently drift from what
                # the mask actually shows.
                occlusion_state=compute_occlusion_state(component_count),
                source=update.source,
                confidence=update.confidence,
                manually_modified=update.manually_modified,
                backend_provenance=provenance,
                review_reasons=review_reasons,
            )
            objects = [o for o in objects if o.object_id != new_entry.object_id]
            objects.append(new_entry)
            if review_reasons and payload.review_status is None:
                changes["review_status"] = ReviewStatus.NEEDS_REVIEW
        if payload.remove_object_id is not None:
            objects = [o for o in objects if o.object_id != payload.remove_object_id]
            inactive = [o for o in inactive if o.object_id != payload.remove_object_id]
        if payload.deactivate_object_id is not None:
            object_id = payload.deactivate_object_id
            entry = next((o for o in objects if o.object_id == object_id), None)
            if entry is None:
                raise HTTPException(
                    status_code=422,
                    detail=f"object {object_id!r} has no mask on this frame to deactivate",
                )
            objects = [o for o in objects if o is not entry]
            inactive = [o for o in inactive if o.object_id != object_id] + [entry]
        if payload.activate_object_id is not None:
            object_id = payload.activate_object_id
            entry = next((o for o in inactive if o.object_id == object_id), None)
            if entry is None:
                raise HTTPException(
                    status_code=422,
                    detail=f"object {object_id!r} is not deactivated on this frame",
                )
            inactive = [o for o in inactive if o is not entry]
            objects = [o for o in objects if o.object_id != object_id] + [entry]
        changes["objects"] = objects
        changes["inactive_objects"] = inactive

        return frame.model_copy(update=changes)

    try:
        updated = save_frame_annotation(
            pdir, frame_index, timestamp, payload.expected_revision, _mutate
        )
    except RevisionConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except MaskDecodeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    append_jsonl(
        events_log_path(pdir),
        {
            "event": "frame_annotation_updated",
            "frame_index": frame_index,
            "revision": updated.revision,
        },
    )
    return updated
