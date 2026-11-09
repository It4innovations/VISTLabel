# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Health, version, backend-listing and operational-metrics endpoints."""

from __future__ import annotations

import time

from fastapi import APIRouter, Request
from pydantic import BaseModel

from vistlabel import __version__
from vistlabel.diagnostics.system import SystemDiagnosticsReport, run_system_diagnostics
from vistlabel.domain.jobs import JobRecord, JobStatus
from vistlabel.persistence.atomic import load_model
from vistlabel.persistence.paths import jobs_dir, projects_root
from vistlabel.segmentation.base import BackendDescriptor

router = APIRouter(prefix="/api/v1", tags=["system"])


class VersionResponse(BaseModel):
    version: str


class BackendDescriptorResponse(BaseModel):
    backend_id: str
    display_name: str
    capabilities: list[str]
    available: bool
    unavailable_reason: str | None
    model_name: str
    checkpoint_sha256: str | None
    code_revision: str | None

    @classmethod
    def from_descriptor(cls, descriptor: BackendDescriptor) -> BackendDescriptorResponse:
        return cls(
            backend_id=descriptor.backend_id,
            display_name=descriptor.display_name,
            capabilities=sorted(capability.value for capability in descriptor.capabilities),
            available=descriptor.available,
            unavailable_reason=descriptor.unavailable_reason,
            model_name=descriptor.model_name,
            checkpoint_sha256=descriptor.checkpoint_sha256,
            code_revision=descriptor.code_revision,
        )


class ModelsResponse(BaseModel):
    backends: list[BackendDescriptorResponse]
    default_backend_id: str


@router.get("/health")
def health(request: Request) -> SystemDiagnosticsReport:
    return run_system_diagnostics(request.app.state.settings)


@router.get("/version")
def version() -> VersionResponse:
    return VersionResponse(version=__version__)


@router.get("/models")
def models(request: Request) -> ModelsResponse:
    registry = request.app.state.registry
    settings = request.app.state.settings
    return ModelsResponse(
        backends=[
            BackendDescriptorResponse.from_descriptor(descriptor)
            for descriptor in registry.list_descriptors()
        ],
        default_backend_id=settings.runtime.default_backend,
    )


class MetricsResponse(BaseModel):
    """Operational metrics as a plain JSON snapshot, not a
    Prometheus-format `/metrics` scrape target -- this application has no
    existing metrics-aggregation dependency,
    and one shared secret-free JSON endpoint is enough operational
    visibility for an internal tool of this scale; a real production
    deployment wanting Prometheus can scrape/transform this externally
    without VISTLabel needing to depend on a metrics client library itself.
    """

    uptime_seconds: float
    project_count: int
    total_annotated_frame_count: int
    jobs_by_status: dict[str, int]
    backend_availability: dict[str, bool]


@router.get("/metrics")
def metrics(request: Request) -> MetricsResponse:
    settings = request.app.state.settings
    project_store = request.app.state.project_store
    registry = request.app.state.registry
    projects = project_store.list_projects()

    jobs_by_status: dict[str, int] = {status.value: 0 for status in JobStatus}
    root = projects_root(settings.workspace.root)
    if root.is_dir():
        for project_entry in root.iterdir():
            directory = jobs_dir(project_entry)
            if not directory.is_dir():
                continue
            for job_file in directory.glob("*.json"):
                try:
                    record = load_model(job_file, JobRecord)
                except (OSError, ValueError):
                    continue
                jobs_by_status[record.status.value] = jobs_by_status.get(record.status.value, 0) + 1

    return MetricsResponse(
        uptime_seconds=round(time.monotonic() - request.app.state.started_at_monotonic, 1),
        project_count=len(projects),
        total_annotated_frame_count=sum(p.progress.annotated_frame_count for p in projects),
        jobs_by_status=jobs_by_status,
        backend_availability={
            descriptor.backend_id: descriptor.available
            for descriptor in registry.list_descriptors()
        },
    )
