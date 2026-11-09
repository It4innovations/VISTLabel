# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Create a project from a COCO or YOLO dataset ZIP.

A deliberate second creation route next to the MP4 upload: still a
browser upload, never a
server-side path, and it still ends in one finalized ``source/video.mp4``
-- stitched from the dataset's images -- so everything downstream works
unchanged. Like MP4 upload, any failure deletes the just-created project.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

from vistlabel.domain.project import (
    ProjectState,
    ProjectStatus,
    SourceVideoInfo,
    VideoImportOptions,
)
from vistlabel.imports.archive import DatasetImportError
from vistlabel.imports.service import DatasetImportReport, RequestedFormat, import_dataset
from vistlabel.persistence.paths import project_dir
from vistlabel.video.ffmpeg import FfmpegNotFoundError, find_ffmpeg_binaries
from vistlabel.video.frames import VideoFrameError
from vistlabel.video.upload import UploadRejectedError, save_upload_streaming
from vistlabel.video.validation import VideoValidationError

router = APIRouter(prefix="/api/v1/projects", tags=["projects"])

MAX_FRAME_RATE = 240.0


class ImportDatasetResponse(BaseModel):
    project: ProjectState
    report: DatasetImportReport


@router.post("/import-dataset", status_code=201)
async def import_dataset_project(
    request: Request,
    project_name: Annotated[str, Form()],
    dataset_file: Annotated[UploadFile, File()],
    dataset_format: Annotated[RequestedFormat, Form()] = "auto",
    frame_rate: Annotated[float, Form()] = 10.0,
    keep_every_nth: Annotated[int | None, Form()] = None,
) -> ImportDatasetResponse:
    """``dataset_format`` ``auto`` prefers COCO when the archive has both
    (a VISTLabel Ultralytics export does): its masks are exact, YOLO
    polygons are not. ``frame_rate`` only sets playback speed -- every
    kept image is one frame either way. ``keep_every_nth`` keeps images
    0, n, 2n, ... in filename order, like the MP4 upload's frame option."""
    settings = request.app.state.settings
    store = request.app.state.project_store
    if not 0 < frame_rate <= MAX_FRAME_RATE:
        raise HTTPException(
            status_code=422, detail=f"frame_rate must be in (0, {MAX_FRAME_RATE:g}]"
        )
    try:
        import_options = VideoImportOptions(keep_every_nth=keep_every_nth)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        binaries = find_ffmpeg_binaries(settings.runtime.ffmpeg_dir)
    except FfmpegNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        project = store.create(project_name)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    pdir = project_dir(settings.workspace.root, project.project_id)
    archive_path = pdir / "source" / "dataset.zip.part"
    try:
        await save_upload_streaming(
            dataset_file,
            archive_path,
            chunk_size_bytes=settings.upload.chunk_size_mb * 1024 * 1024,
            max_size_bytes=int(settings.upload.maximum_file_size_gb * 1024**3),
            min_free_disk_bytes=int(settings.upload.minimum_free_disk_space_gb * 1024**3),
        )
        result = await asyncio.to_thread(
            import_dataset,
            pdir,
            archive_path,
            binaries,
            requested_format=dataset_format,
            frame_rate=frame_rate,
            keep_every_nth=import_options.keep_every_nth or 1,
        )
    except UploadRejectedError as exc:
        store.delete(project.project_id)
        raise HTTPException(status_code=422, detail=exc.reason) from exc
    except (DatasetImportError, VideoValidationError, VideoFrameError) as exc:
        store.delete(project.project_id)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except BaseException:
        store.delete(project.project_id)
        raise
    # The video now holds every image; keeping the ZIP would double the disk use.
    archive_path.unlink(missing_ok=True)

    finalized = result.finalized
    original_filename = Path(dataset_file.filename).name if dataset_file.filename else "dataset.zip"

    def _attach_video(state: ProjectState) -> ProjectState:
        return state.model_copy(
            update={
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
                ),
                "progress": state.progress.model_copy(
                    update={"total_frame_count": finalized.frame_count}
                ),
                "split_assignments": result.split_assignments,
            }
        )

    updated = store.update(
        project.project_id, expected_revision=project.revision, mutate=_attach_video
    )
    return ImportDatasetResponse(project=updated, report=result.report)
