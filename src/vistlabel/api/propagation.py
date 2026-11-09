# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Propagation job endpoints.

Runs ``backend.propagate`` as a cancellable, pausable, chunked background
job: each chunk (``tracking.default_chunk_frames`` frames by default) is
its own checkpoint -- a :class:`~vistlabel.domain.tracking.TrackingRunManifest`
records the last *fully completed* chunk, so cancelling, pausing, or a
process crash never loses already-committed chunks, and resuming restarts
from the last checkpoint rather than from ``start_frame``. Results are
persisted incrementally by chunk, so a restart can identify incomplete
work and offer safe resume or rollback.

Automatic review-reason detection uses each object's frame-to-frame
history within the propagation run: frame-local checks, temporal ones,
abrupt centroid displacement and periodic review sampling. Manually
modified masks are protected from silent overwrite by default --
overwriting them requires an explicit ``overwrite_manual_masks: true`` on
the request.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from vistlabel.annotations.frames_service import apply_frame_annotation, get_frame_annotation
from vistlabel.annotations.objects_service import get_objects
from vistlabel.annotations.occlusion import (
    PreviousFrameContext,
    compute_occlusion_state,
    frame_local_review_reasons,
    previous_frame_context,
    review_sampling_reasons,
    temporal_review_reasons,
)
from vistlabel.api.deps import require_frame_index, require_project_with_video
from vistlabel.domain.frames import (
    FrameAnnotation,
    FrameObjectAnnotation,
    ReviewReason,
    ReviewStatus,
)
from vistlabel.domain.jobs import JobRecord
from vistlabel.domain.tracking import TrackingRunManifest, TrackingRunStatus
from vistlabel.jobs.manager import JobCancelled, JobContext, JobManager, JobPaused
from vistlabel.persistence.paths import source_video_path
from vistlabel.persistence.rle import decode_mask, encode_mask
from vistlabel.segmentation.base import MaskPrediction, VideoSegmentationBackend
from vistlabel.segmentation.registry import UnknownBackendError
from vistlabel.tracking import manifests

router = APIRouter(prefix="/api/v1/projects", tags=["propagation"])


class PropagateRequest(BaseModel):
    backend_id: str = "fake"
    object_ids: list[str]
    start_frame: int
    end_frame: int
    direction: Literal["forward", "backward"] = "forward"
    #: Overrides ``tracking.default_chunk_frames`` for this run.
    chunk_frames: int | None = None
    #: Propagation never overwrites a manually_modified mask
    #: unless the caller opts in explicitly.
    overwrite_manual_masks: bool = False


class RunJobResponse(BaseModel):
    job_id: str
    run_id: str


def _chunk_ranges(
    start_frame: int, end_frame: int, direction: str, chunk_frames: int
) -> Iterator[tuple[int, int]]:
    """Consecutive chunks *share* their boundary frame -- chunk N+1 starts
    exactly where chunk N ended, not one step past it. ``backend.propagate``
    requires its ``start_frame`` to already have a seeded mask in the
    backend's session (``segment_interactive`` must be called first); the
    only frame guaranteed to satisfy that for chunk N+1 is
    chunk N's own last output. ``chunk_frames`` counts new frames of
    progress per chunk, not counting the reused seed."""
    advance = max(1, chunk_frames)
    current = start_frame
    if current == end_frame:
        yield current, end_frame
        return
    if direction == "forward":
        while current < end_frame:
            chunk_end = min(current + advance, end_frame)
            yield current, chunk_end
            current = chunk_end
    else:
        while current > end_frame:
            chunk_end = max(current - advance, end_frame)
            yield current, chunk_end
            current = chunk_end


def _run_chunked_propagation(
    *,
    context: JobContext,
    pdir: Path,
    timestamps: list[float],
    backend: VideoSegmentationBackend,
    session_id: str,
    backend_id: str,
    object_ids: list[str],
    label_by_object: dict[str, int],
    manifest_run_id: str,
    resume_start_frame: int,
    end_frame: int,
    direction: str,
    chunk_frames: int,
    overwrite_manual_masks: bool,
    review_sampling_stride: int | None,
    job_manager: JobManager,
    project_id: str,
) -> None:
    # Progress denominator counts *predictions*, not frames: the backend
    # yields one prediction per object per frame, and completed_frames
    # increments per prediction -- dividing by frames alone would hit 100%
    # after 1/N of a run tracking N objects and then sit at 100% for the
    # rest. Chunk-boundary frames are re-emitted once per chunk, so the
    # ratio can slightly overshoot; min(1.0, ...) at the report site caps
    # that.
    total_predictions = max(abs(end_frame - resume_start_frame) + 1, 1) * max(len(object_ids), 1)
    completed_frames = 0
    manual_protected = manifests.get_run(pdir, manifest_run_id).manual_masks_protected_count

    try:
        for chunk_start, chunk_end in _chunk_ranges(
            resume_start_frame, end_frame, direction, chunk_frames
        ):
            context.raise_if_paused()
            context.raise_if_cancelled()
            previous_by_object: dict[str, PreviousFrameContext] = {}

            for prediction in backend.propagate(
                session_id, object_ids, chunk_start, chunk_end, direction
            ):
                context.raise_if_cancelled()
                label_id = label_by_object[prediction.object_id]
                previous = previous_by_object.get(prediction.object_id)

                existing = get_frame_annotation(
                    pdir, prediction.frame_index, timestamps[prediction.frame_index]
                )
                existing_entry = next(
                    (o for o in existing.objects if o.object_id == prediction.object_id), None
                )
                # Deactivating an object on a frame is a manual edit too.
                inactive_here = any(
                    o.object_id == prediction.object_id for o in existing.inactive_objects
                )
                manual_here = existing_entry is not None and existing_entry.manually_modified
                if (inactive_here or manual_here) and not overwrite_manual_masks:
                    manual_protected += 1
                    completed_frames += 1
                    continue

                review_reasons = frame_local_review_reasons(
                    prediction.visible_mask, prediction.score
                )
                if previous is not None:
                    review_reasons = review_reasons + temporal_review_reasons(
                        mask=prediction.visible_mask,
                        component_count=prediction.component_count,
                        previous=previous,
                    )
                review_reasons = review_reasons + review_sampling_reasons(
                    prediction.frame_index, review_sampling_stride
                )
                previous_by_object[prediction.object_id] = previous_frame_context(
                    prediction.visible_mask, prediction.component_count
                )

                def _mutate(
                    frame: FrameAnnotation,
                    prediction: MaskPrediction = prediction,
                    label_id: int = label_id,
                    review_reasons: list[ReviewReason] = review_reasons,
                ) -> FrameAnnotation:
                    entry = FrameObjectAnnotation(
                        object_id=prediction.object_id,
                        label_id=label_id,
                        visible_mask_rle=encode_mask(prediction.visible_mask),
                        connected_component_count=prediction.component_count,
                        occlusion_state=compute_occlusion_state(prediction.component_count),
                        source=f"{backend_id}_propagation",
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
                completed_frames += 1
                job_manager.report_progress(
                    project_id, context.job_id, min(1.0, completed_frames / total_predictions)
                )

            # Chunk fully processed -- this is the checkpoint: results are
            # persisted incrementally by chunk. A crash mid-chunk
            # simply redoes this one chunk on resume; nothing before it is
            # ever lost.
            manifests.update_run(
                pdir,
                manifest_run_id,
                last_completed_frame=chunk_end,
                manual_masks_protected_count=manual_protected,
            )
    except JobPaused:
        manifests.update_run(pdir, manifest_run_id, status=TrackingRunStatus.PAUSED)
        raise
    except JobCancelled:
        manifests.update_run(pdir, manifest_run_id, status=TrackingRunStatus.CANCELLED)
        raise
    except Exception as exc:
        manifests.update_run(
            pdir, manifest_run_id, status=TrackingRunStatus.FAILED, error=str(exc)
        )
        raise

    manifests.update_run(pdir, manifest_run_id, status=TrackingRunStatus.COMPLETED)


def _resolve_backend_and_session(
    request: Request, project_id: str, payload: PropagateRequest
) -> tuple[Path, list[float], VideoSegmentationBackend, dict[str, int], str]:
    _project, pdir, timestamps = require_project_with_video(request, project_id)
    require_frame_index(payload.start_frame, timestamps)
    require_frame_index(payload.end_frame, timestamps)

    try:
        backend = request.app.state.registry.get(payload.backend_id)
    except UnknownBackendError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    descriptor = backend.descriptor()
    if not descriptor.available:
        raise HTTPException(
            status_code=409,
            detail=(
                f"backend {payload.backend_id!r} is unavailable: "
                f"{descriptor.unavailable_reason}"
            ),
        )

    label_by_object = {o.object_id: o.label_id for o in get_objects(pdir).objects}
    for object_id in payload.object_ids:
        if object_id not in label_by_object:
            raise HTTPException(status_code=404, detail=f"unknown object_id: {object_id}")

    session_id = request.app.state.backend_sessions.get_or_open(
        backend, project_id, source_video_path(pdir)
    )
    return pdir, timestamps, backend, label_by_object, session_id


@router.post("/{project_id}/propagate", status_code=202)
def start_propagation(
    project_id: str, payload: PropagateRequest, request: Request
) -> RunJobResponse:
    pdir, timestamps, backend, label_by_object, session_id = _resolve_backend_and_session(
        request, project_id, payload
    )
    # Re-seed the backend session from the canonical saved masks at
    # start_frame, same as resume does: the in-memory session only knows
    # objects segmented interactively *in this process*, but a multi-object
    # run over the object list legitimately includes objects seeded in an
    # earlier session (or before a server restart). Objects without a saved
    # mask here keep whatever live seed the session already has; the
    # backend still rejects genuinely unseeded ones with a typed error.
    # A mask propagation didn't produce (a hand correction, an accepted
    # click) is a fresh seed: the tracker forgets the object first, or what
    # it remembers ignores or outvotes that mask. A propagated mask keeps
    # the memory, so continuing a run doesn't lose its original seed.
    existing = get_frame_annotation(pdir, payload.start_frame, timestamps[payload.start_frame])
    saved_by_object = {o.object_id: o for o in existing.objects}
    for object_id in payload.object_ids:
        entry = saved_by_object.get(object_id)
        if entry is not None:
            if not entry.source.endswith("_propagation"):
                backend.forget_object(session_id, object_id)
            backend.replace_object_mask(
                session_id, payload.start_frame, object_id, decode_mask(entry.visible_mask_rle)
            )
    settings = request.app.state.settings
    chunk_frames = payload.chunk_frames or settings.tracking.default_chunk_frames
    review_sampling_stride = settings.tracking.review_sampling_stride
    job_manager = request.app.state.job_manager
    run_id = manifests.new_run_id()

    def work(context: JobContext) -> None:
        manifests.create_run(
            pdir,
            run_id=run_id,
            job_id=context.job_id,
            backend_id=payload.backend_id,
            object_ids=payload.object_ids,
            start_frame=payload.start_frame,
            end_frame=payload.end_frame,
            direction=payload.direction,
            chunk_frames=chunk_frames,
            overwrite_manual_masks=payload.overwrite_manual_masks,
        )
        _run_chunked_propagation(
            context=context,
            pdir=pdir,
            timestamps=timestamps,
            backend=backend,
            session_id=session_id,
            backend_id=payload.backend_id,
            object_ids=payload.object_ids,
            label_by_object=label_by_object,
            manifest_run_id=run_id,
            resume_start_frame=payload.start_frame,
            end_frame=payload.end_frame,
            direction=payload.direction,
            chunk_frames=chunk_frames,
            overwrite_manual_masks=payload.overwrite_manual_masks,
            review_sampling_stride=review_sampling_stride,
            job_manager=job_manager,
            project_id=project_id,
        )

    job_id = job_manager.submit(project_id, "propagate", work)
    return RunJobResponse(job_id=job_id, run_id=run_id)


@router.get("/{project_id}/tracking-runs")
def list_tracking_runs(project_id: str, request: Request) -> list[TrackingRunManifest]:
    _project, pdir, _timestamps = require_project_with_video(request, project_id)
    return manifests.list_runs(pdir)


@router.get("/{project_id}/tracking-runs/{run_id}")
def get_tracking_run(project_id: str, run_id: str, request: Request) -> TrackingRunManifest:
    _project, pdir, _timestamps = require_project_with_video(request, project_id)
    try:
        return manifests.get_run(pdir, run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"no such tracking run: {run_id}") from exc


@router.post("/{project_id}/tracking-runs/{run_id}/pause")
def pause_tracking_run(project_id: str, run_id: str, request: Request) -> TrackingRunManifest:
    _project, pdir, _timestamps = require_project_with_video(request, project_id)
    manifest = get_tracking_run(project_id, run_id, request)
    if manifest.status is not TrackingRunStatus.RUNNING:
        raise HTTPException(
            status_code=409, detail=f"tracking run {run_id} is not running (status: "
            f"{manifest.status.value})"
        )
    if not request.app.state.job_manager.pause(manifest.job_id):
        raise HTTPException(status_code=409, detail=f"job {manifest.job_id} is not running")
    return manifest


@router.post("/{project_id}/tracking-runs/{run_id}/resume", status_code=202)
def resume_tracking_run(project_id: str, run_id: str, request: Request) -> RunJobResponse:
    _project, pdir, timestamps = require_project_with_video(request, project_id)
    manifest = get_tracking_run(project_id, run_id, request)
    if manifest.status not in (TrackingRunStatus.PAUSED, TrackingRunStatus.INTERRUPTED):
        raise HTTPException(
            status_code=409,
            detail=f"tracking run {run_id} cannot be resumed from status {manifest.status.value}",
        )

    payload = PropagateRequest(
        backend_id=manifest.backend_id,
        object_ids=manifest.object_ids,
        start_frame=manifest.start_frame,
        end_frame=manifest.end_frame,
        direction=manifest.direction,
        chunk_frames=manifest.chunk_frames,
        overwrite_manual_masks=manifest.overwrite_manual_masks,
    )
    pdir, timestamps, backend, label_by_object, session_id = _resolve_backend_and_session(
        request, project_id, payload
    )
    resume_start_frame = (
        manifest.last_completed_frame
        if manifest.last_completed_frame is not None
        else manifest.start_frame
    )
    # The backend's own session may be brand new (a real process restart
    # after a crash has no in-memory state at all) -- re-seed it from the
    # canonical, already-persisted mask at the resume frame before
    # propagating again, exactly like the original interactive
    # segment_interactive call that started the run in the first place.
    # Harmless and idempotent when the session already has it (e.g.
    # resuming a same-process pause).
    for object_id in manifest.object_ids:
        existing = get_frame_annotation(pdir, resume_start_frame, timestamps[resume_start_frame])
        entry = next((o for o in existing.objects if o.object_id == object_id), None)
        if entry is None:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"cannot resume: object {object_id!r} has no saved mask on frame "
                    f"{resume_start_frame} to reseed the backend session from"
                ),
            )
        backend.replace_object_mask(
            session_id, resume_start_frame, object_id, decode_mask(entry.visible_mask_rle)
        )
    job_manager = request.app.state.job_manager

    def work(context: JobContext) -> None:
        manifests.update_run(
            pdir, run_id, job_id=context.job_id, status=TrackingRunStatus.RUNNING
        )
        _run_chunked_propagation(
            context=context,
            pdir=pdir,
            timestamps=timestamps,
            backend=backend,
            session_id=session_id,
            backend_id=manifest.backend_id,
            object_ids=manifest.object_ids,
            label_by_object=label_by_object,
            manifest_run_id=run_id,
            resume_start_frame=resume_start_frame,
            end_frame=manifest.end_frame,
            direction=manifest.direction,
            chunk_frames=manifest.chunk_frames,
            overwrite_manual_masks=manifest.overwrite_manual_masks,
            review_sampling_stride=request.app.state.settings.tracking.review_sampling_stride,
            job_manager=job_manager,
            project_id=project_id,
        )

    job_id = job_manager.submit(project_id, "propagate", work)
    return RunJobResponse(job_id=job_id, run_id=run_id)


@router.post("/{project_id}/tracking-runs/{run_id}/cancel")
def cancel_tracking_run(project_id: str, run_id: str, request: Request) -> TrackingRunManifest:
    """Rollback: for a paused or crash-interrupted run with
    no live job to cancel via the generic ``jobs/{job_id}/cancel``
    endpoint, this explicitly abandons it -- already-committed chunks stay
    exactly as they are; nothing is undone, resuming just stops being an
    option."""
    _project, pdir, _timestamps = require_project_with_video(request, project_id)
    manifest = get_tracking_run(project_id, run_id, request)
    if manifest.status not in (TrackingRunStatus.PAUSED, TrackingRunStatus.INTERRUPTED):
        raise HTTPException(
            status_code=409,
            detail=(
                f"tracking run {run_id} has status {manifest.status.value}; "
                "cancel the live job instead"
            ),
        )
    return manifests.update_run(pdir, run_id, status=TrackingRunStatus.CANCELLED)


@router.get("/{project_id}/jobs/{job_id}")
def get_job(project_id: str, job_id: str, request: Request) -> JobRecord:
    try:
        return request.app.state.job_manager.get(project_id, job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"no such job: {job_id}") from exc


@router.post("/{project_id}/jobs/{job_id}/cancel")
def cancel_job(project_id: str, job_id: str, request: Request) -> JobRecord:
    request.app.state.job_manager.cancel(job_id)
    return get_job(project_id, job_id, request)
