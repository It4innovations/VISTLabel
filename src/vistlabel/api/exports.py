# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Dataset export endpoints. Runs as a background job (mirrors
``api/propagation.py``'s pattern) since a full-project export decodes and
writes every selected frame.
"""

from __future__ import annotations

import uuid
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from vistlabel.api.deps import require_frame_index, require_project_with_video
from vistlabel.domain.exports import (
    AnnotationType,
    DatasetLayout,
    ExportManifest,
    FrameSelectionPolicy,
    ValidationReport,
)
from vistlabel.exports.service import ExportRequest, delete_export, run_export
from vistlabel.jobs.manager import JobContext
from vistlabel.persistence.atomic import load_model
from vistlabel.persistence.paths import (
    export_manifest_path,
    export_zip_path,
    exports_dir,
    source_video_path,
    validation_report_path,
)

router = APIRouter(prefix="/api/v1/projects", tags=["exports"])


class StartExportRequest(BaseModel):
    frame_selection_policy: FrameSelectionPolicy
    every_nth: int = 1
    selected_intervals: list[tuple[int, int]] = []
    negative_frame_fraction: float = 0.0
    yolo_multipart_policy: Literal["strict_error", "minimal_bridge", "split_components"] | None = (
        None
    )
    image_format: Literal["jpg", "png"] | None = None
    always_include_coco_exact_annotations: bool | None = None
    annotation_type: AnnotationType | None = None
    dataset_layout: DatasetLayout | None = None
    #: Exporting all frames requires an explicit warning: the request
    #: must set this to true when frame_selection_policy is all_frames, or
    #: the request is rejected rather than silently exporting everything.
    acknowledge_all_frames_warning: bool = False


class ExportJobResponse(BaseModel):
    job_id: str
    export_id: str


@router.post("/{project_id}/exports", status_code=202)
def start_export(
    project_id: str, payload: StartExportRequest, request: Request
) -> ExportJobResponse:
    project, pdir, timestamps = require_project_with_video(request, project_id)
    assert project.source is not None
    source = project.source
    for start, end in payload.selected_intervals:
        require_frame_index(start, timestamps)
        require_frame_index(end, timestamps)
    if (
        payload.frame_selection_policy is FrameSelectionPolicy.ALL_FRAMES
        and not payload.acknowledge_all_frames_warning
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                "frame_selection_policy 'all_frames' requires acknowledge_all_frames_warning: true"
            ),
        )

    settings = request.app.state.settings
    export_settings = settings.export
    export_id = f"exp_{uuid.uuid4().hex[:12]}"
    export_request = ExportRequest(
        export_id=export_id,
        project_id=project_id,
        frame_selection_policy=payload.frame_selection_policy,
        every_nth=payload.every_nth,
        selected_intervals=payload.selected_intervals,
        negative_frame_fraction=payload.negative_frame_fraction,
        yolo_multipart_policy=(
            payload.yolo_multipart_policy or export_settings.yolo_multipart_policy
        ),
        always_include_coco_exact_annotations=(
            payload.always_include_coco_exact_annotations
            if payload.always_include_coco_exact_annotations is not None
            else export_settings.always_include_coco_exact_annotations
        ),
        image_format=payload.image_format or export_settings.image_format,
        jpeg_quality=export_settings.jpeg_quality,
        annotation_type=payload.annotation_type or export_settings.annotation_type,
        dataset_layout=payload.dataset_layout or export_settings.dataset_layout,
        image_width=source.width,
        image_height=source.height,
    )
    labels = request.app.state.project_store.get_labels(project_id)
    split_assignments = project.split_assignments
    video_path = source_video_path(pdir)
    source_video_sha256 = source.sha256
    job_manager = request.app.state.job_manager

    def work(context: JobContext) -> None:
        run_export(
            context=context,
            project_dir=pdir,
            video_path=video_path,
            timestamps=timestamps,
            request=export_request,
            split_assignments=split_assignments,
            labels=labels,
            source_video_sha256=source_video_sha256,
            report_progress=lambda fraction: job_manager.report_progress(
                project_id, context.job_id, fraction
            ),
        )

    job_id = job_manager.submit(project_id, "export", work)
    return ExportJobResponse(job_id=job_id, export_id=export_id)


@router.get("/{project_id}/exports")
def list_exports(project_id: str, request: Request) -> list[ExportManifest]:
    _project, pdir, _timestamps = require_project_with_video(request, project_id)
    directory = exports_dir(pdir)
    if not directory.is_dir():
        return []
    manifests = []
    for entry in sorted(directory.iterdir()):
        manifest_path = export_manifest_path(pdir, entry.name)
        if manifest_path.is_file():
            manifests.append(load_model(manifest_path, ExportManifest))
    return manifests


@router.get("/{project_id}/exports/{export_id}")
def get_export(project_id: str, export_id: str, request: Request) -> ExportManifest:
    _project, pdir, _timestamps = require_project_with_video(request, project_id)
    try:
        return load_model(export_manifest_path(pdir, export_id), ExportManifest)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"no such export: {export_id}") from exc


@router.get("/{project_id}/exports/{export_id}/validation-report")
def get_export_validation_report(
    project_id: str, export_id: str, request: Request
) -> ValidationReport:
    _project, pdir, _timestamps = require_project_with_video(request, project_id)
    try:
        return load_model(validation_report_path(pdir, export_id), ValidationReport)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail=f"no validation report yet for export: {export_id}"
        ) from exc


@router.get("/{project_id}/exports/{export_id}/download")
def download_export(project_id: str, export_id: str, request: Request) -> FileResponse:
    _project, pdir, _timestamps = require_project_with_video(request, project_id)
    zip_path = export_zip_path(pdir, export_id)
    if not zip_path.is_file():
        raise HTTPException(
            status_code=404, detail=f"no completed export ZIP for export: {export_id}"
        )
    return FileResponse(
        zip_path, media_type="application/zip", filename=f"{project_id}_{export_id}.zip"
    )


@router.delete("/{project_id}/exports/{export_id}", status_code=204)
def delete_export_endpoint(project_id: str, export_id: str, request: Request) -> None:
    """Server-side export caches are retained and deleted through
    project-scoped APIs only -- there is no other way to remove an export
    directory."""
    _project, pdir, _timestamps = require_project_with_video(request, project_id)
    delete_export(pdir, export_id)
