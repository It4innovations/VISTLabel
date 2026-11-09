# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Persistence for :class:`TrackingRunManifest`: create, update, list, and
the crash-recovery scan.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from vistlabel.domain.tracking import TrackingRunManifest, TrackingRunStatus
from vistlabel.persistence.atomic import load_model, save_model
from vistlabel.persistence.paths import tracking_run_path, tracking_runs_dir


def new_run_id() -> str:
    import uuid

    return f"run_{uuid.uuid4().hex[:12]}"


def create_run(
    project_dir: Path,
    *,
    run_id: str,
    job_id: str,
    backend_id: str,
    object_ids: list[str],
    start_frame: int,
    end_frame: int,
    direction: str,
    chunk_frames: int,
    overwrite_manual_masks: bool,
) -> TrackingRunManifest:
    now = datetime.now(UTC)
    manifest = TrackingRunManifest(
        run_id=run_id,
        job_id=job_id,
        backend_id=backend_id,
        object_ids=object_ids,
        start_frame=start_frame,
        end_frame=end_frame,
        direction=direction,  # type: ignore[arg-type]
        chunk_frames=chunk_frames,
        overwrite_manual_masks=overwrite_manual_masks,
        created_at=now,
        updated_at=now,
    )
    save_model(tracking_run_path(project_dir, manifest.run_id), manifest)
    return manifest


def get_run(project_dir: Path, run_id: str) -> TrackingRunManifest:
    return load_model(tracking_run_path(project_dir, run_id), TrackingRunManifest)


def update_run(
    project_dir: Path,
    run_id: str,
    **changes: object,
) -> TrackingRunManifest:
    current = get_run(project_dir, run_id)
    updated = current.model_copy(update={**changes, "updated_at": datetime.now(UTC)})
    save_model(tracking_run_path(project_dir, run_id), updated)
    return updated


def list_runs(project_dir: Path) -> list[TrackingRunManifest]:
    runs_dir = tracking_runs_dir(project_dir)
    if not runs_dir.is_dir():
        return []
    return [load_model(path, TrackingRunManifest) for path in sorted(runs_dir.glob("*.json"))]


def recover_interrupted_runs(project_dir: Path) -> list[str]:
    """Called at application startup, so a restart can identify incomplete
    work and offer safe resume or rollback. A manifest still ``running``
    here cannot actually be running -- the process that would be running
    it just started -- so it is *marked* interrupted, never
    silently resumed; resuming is a distinct, explicit API call."""
    recovered: list[str] = []
    for manifest in list_runs(project_dir):
        if manifest.status is TrackingRunStatus.RUNNING:
            update_run(project_dir, manifest.run_id, status=TrackingRunStatus.INTERRUPTED)
            recovered.append(manifest.run_id)
    return recovered
