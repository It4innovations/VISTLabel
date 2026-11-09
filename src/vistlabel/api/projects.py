# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Project management and video/frame endpoints.

`POST /api/v1/projects` is the single combined create+upload endpoint
(`multipart/form-data`, fields `project_name` + `video_file`). By design
there is no separate "select/import existing server video" path and no
alternate creation route. If validation of the uploaded file fails, the
just-created project is deleted rather than left around half-initialized;
there is no "retry upload against an existing project" endpoint.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, HTTPException, Request, Response, UploadFile
from pydantic import BaseModel, ValidationError

from vistlabel.annotations.frames_service import count_frame_progress
from vistlabel.api.deps import require_frame_index, require_project_with_video
from vistlabel.domain.labels import LabelsFile
from vistlabel.domain.project import (
    OPERATOR_SETTABLE_STATUSES,
    FrameRangeAssignment,
    ProjectProgress,
    ProjectState,
    ProjectStatus,
    SourceVideoInfo,
    VideoImportOptions,
)
from vistlabel.persistence.paths import project_dir, source_video_path
from vistlabel.projects.errors import ProjectNotFoundError, RevisionConflictError
from vistlabel.video.ffmpeg import FfmpegNotFoundError, find_ffmpeg_binaries
from vistlabel.video.frames import (
    VideoFrameError,
    get_or_decode_frame_jpeg,
    get_or_decode_thumbnail_jpeg,
)
from vistlabel.video.upload import UploadRejectedError, finalize_video, save_upload_streaming
from vistlabel.video.validation import VideoValidationError

router = APIRouter(prefix="/api/v1/projects", tags=["projects"])


def _describe_option_error(error: Any) -> str:
    message = str(error["msg"]).removeprefix("Value error, ")
    return f"{error['loc'][0]}: {message}" if error["loc"] else message


@router.post("", status_code=201)
async def create_project(
    request: Request,
    project_name: Annotated[str, Form()],
    video_file: Annotated[UploadFile, File()],
    max_width: Annotated[int | None, Form()] = None,
    max_height: Annotated[int | None, Form()] = None,
    target_fps: Annotated[float | None, Form()] = None,
    keep_every_nth: Annotated[int | None, Form()] = None,
) -> ProjectState:
    """Optional ``max_width``/``max_height`` (fit within, never upscale) and
    ``target_fps`` *or* ``keep_every_nth`` re-encode the video once, before
    frame indexing; omitted means keep the native video."""
    settings = request.app.state.settings
    store = request.app.state.project_store

    # Checked before the project exists, so a bad option leaves nothing behind.
    try:
        import_options = VideoImportOptions(
            max_width=max_width,
            max_height=max_height,
            target_fps=target_fps,
            keep_every_nth=keep_every_nth,
        )
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail="; ".join(_describe_option_error(error) for error in exc.errors()),
        ) from exc

    try:
        project = store.create(project_name)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    pdir = project_dir(settings.workspace.root, project.project_id)
    part_path = pdir / "source" / "video.mp4.part"

    try:
        size_bytes, sha256_hex = await save_upload_streaming(
            video_file,
            part_path,
            chunk_size_bytes=settings.upload.chunk_size_mb * 1024 * 1024,
            max_size_bytes=int(settings.upload.maximum_file_size_gb * 1024**3),
            min_free_disk_bytes=int(settings.upload.minimum_free_disk_space_gb * 1024**3),
        )
    except UploadRejectedError as exc:
        store.delete(project.project_id)
        raise HTTPException(status_code=422, detail=exc.reason) from exc

    try:
        binaries = find_ffmpeg_binaries(settings.runtime.ffmpeg_dir)
    except FfmpegNotFoundError as exc:
        store.delete(project.project_id)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        finalized = await asyncio.to_thread(
            finalize_video,
            pdir,
            part_path,
            binaries,
            size_bytes=size_bytes,
            sha256=sha256_hex,
            normalize_vfr=settings.upload.normalize_variable_frame_rate,
            normalized_frame_rate=settings.upload.normalized_frame_rate,
            import_options=import_options,
        )
    except (VideoValidationError, VideoFrameError) as exc:
        store.delete(project.project_id)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    original_filename = Path(video_file.filename).name if video_file.filename else "video.mp4"

    def _attach_video(state: ProjectState) -> ProjectState:
        return state.model_copy(
            update={
                # A freshly uploaded project is annotatable but untouched;
                # the operator moves it on to in_progress/done themselves.
                "status": ProjectStatus.NEW,
                "source": SourceVideoInfo(
                    original_filename=original_filename,
                    stored_path=finalized.stored_path,
                    sha256=finalized.sha256,
                    size_bytes=finalized.size_bytes,
                    video_codec=finalized.probed.video_codec,
                    pixel_format=finalized.probed.pixel_format,
                    width=finalized.probed.width,
                    height=finalized.probed.height,
                    nominal_fps=finalized.probed.nominal_fps,
                    duration_seconds=finalized.probed.duration_seconds,
                    frame_count=finalized.frame_count,
                    variable_frame_rate=finalized.variable_frame_rate,
                    import_options=import_options,
                    original=finalized.original,
                ),
                "progress": state.progress.model_copy(
                    update={"total_frame_count": finalized.frame_count}
                ),
            }
        )

    return store.update(
        project.project_id, expected_revision=project.revision, mutate=_attach_video
    )


@router.get("")
def list_projects(request: Request) -> list[ProjectState]:
    return request.app.state.project_store.list_projects()


@router.get("/{project_id}")
def get_project(project_id: str, request: Request) -> ProjectState:
    try:
        return request.app.state.project_store.get(project_id)
    except ProjectNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{project_id}/progress")
def get_project_progress(project_id: str, request: Request) -> ProjectProgress:
    """Live annotated/reviewed counts for one project (what the overview
    shows), e.g. for the export dialog's per-policy frame counts."""
    project, pdir, _timestamps = require_project_with_video(request, project_id)
    annotated, reviewed = count_frame_progress(pdir)
    return project.progress.model_copy(
        update={"annotated_frame_count": annotated, "reviewed_frame_count": reviewed}
    )


@router.delete("/{project_id}", status_code=204)
def delete_project(project_id: str, request: Request) -> Response:
    try:
        request.app.state.project_store.delete(project_id)
    except ProjectNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    # Explicit session lifecycle: a deleted
    # project must not leave an open backend session referencing a video
    # path that no longer exists.
    request.app.state.backend_sessions.close_project_sessions(
        project_id, request.app.state.registry
    )
    return Response(status_code=204)


class UpdateProjectRequest(BaseModel):
    """Lets the browser persist ``ui_state`` (last opened frame, selected
    object/backend, zoom/pan) through the same optimistic-revision path as
    everything else, so that closing and reopening the project restores the
    last frame and current masks, and lets an operator move the project
    through its workflow states."""

    expected_revision: int
    #: Operator-set workflow state. ``pending`` is rejected: it means "upload
    #: not finished", which only the upload path may assert.
    status: ProjectStatus | None = None
    last_opened_frame: int | None = None
    selected_object_id: str | None = None
    selected_backend_id: str | None = None
    zoom: float | None = None
    pan_x: float | None = None
    pan_y: float | None = None


@router.patch("/{project_id}")
def patch_project(
    project_id: str, payload: UpdateProjectRequest, request: Request
) -> ProjectState:
    store = request.app.state.project_store
    if payload.status is not None and payload.status not in OPERATOR_SETTABLE_STATUSES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"status {payload.status.value!r} is not operator-settable; "
                f"allowed: {sorted(s.value for s in OPERATOR_SETTABLE_STATUSES)}"
            ),
        )
    changes = {
        key: value
        for key, value in payload.model_dump(exclude={"expected_revision", "status"}).items()
        if value is not None
    }

    def _mutate(state: ProjectState) -> ProjectState:
        updated = state
        if changes:
            updated = updated.model_copy(
                update={"ui_state": updated.ui_state.model_copy(update=changes)}
            )
        if payload.status is not None:
            updated = updated.model_copy(update={"status": payload.status})
        return updated

    try:
        return store.update(project_id, payload.expected_revision, _mutate)
    except ProjectNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RevisionConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/{project_id}/labels")
def get_labels(project_id: str, request: Request) -> LabelsFile:
    try:
        return request.app.state.project_store.get_labels(project_id)
    except ProjectNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.put("/{project_id}/labels")
def put_labels(project_id: str, payload: LabelsFile, request: Request) -> LabelsFile:
    try:
        return request.app.state.project_store.set_labels(project_id, payload.labels)
    except ProjectNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


class PutSplitAssignmentsRequest(BaseModel):
    expected_revision: int
    split_assignments: list[FrameRangeAssignment]


@router.get("/{project_id}/split-assignments")
def get_split_assignments(project_id: str, request: Request) -> list[FrameRangeAssignment]:
    try:
        return request.app.state.project_store.get(project_id).split_assignments
    except ProjectNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.put("/{project_id}/split-assignments")
def put_split_assignments(
    project_id: str, payload: PutSplitAssignmentsRequest, request: Request
) -> ProjectState:
    """Lets the user mark frame intervals as train / validation /
    excluded, following the same pattern as ``PUT .../labels`` and
    ``occlusion_edges``."""

    def _mutate(state: ProjectState) -> ProjectState:
        return state.model_copy(update={"split_assignments": payload.split_assignments})

    store = request.app.state.project_store
    try:
        return store.update(project_id, payload.expected_revision, _mutate)
    except ProjectNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RevisionConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/{project_id}/frames/{frame_index}")
def get_frame(project_id: str, frame_index: int, request: Request) -> Response:
    _project, pdir, timestamps = require_project_with_video(request, project_id)
    require_frame_index(frame_index, timestamps)
    settings = request.app.state.settings
    cache_path = pdir / "cache" / "frames" / f"{frame_index:06d}.jpg"
    try:
        data = get_or_decode_frame_jpeg(
            cache_path,
            source_video_path(pdir),
            frame_index,
            timestamps,
            jpeg_quality=settings.export.jpeg_quality,
        )
    except VideoFrameError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return Response(content=data, media_type="image/jpeg")


@router.get("/{project_id}/frames/{frame_index}/thumbnail")
def get_frame_thumbnail(project_id: str, frame_index: int, request: Request) -> Response:
    _project, pdir, timestamps = require_project_with_video(request, project_id)
    require_frame_index(frame_index, timestamps)
    cache_path = pdir / "cache" / "thumbnails" / f"{frame_index:06d}.jpg"
    try:
        data = get_or_decode_thumbnail_jpeg(
            cache_path, source_video_path(pdir), frame_index, timestamps
        )
    except VideoFrameError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return Response(content=data, media_type="image/jpeg")
