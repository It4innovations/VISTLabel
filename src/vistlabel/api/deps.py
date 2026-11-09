# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Small request-scoped helpers shared by the project/frame/object/segment
routers -- not a generic framework, just the handful of lookups every one
of them needs (locate the project, make sure it has a video).
"""

from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException, Request

from vistlabel.domain.project import ProjectState
from vistlabel.persistence.paths import project_dir
from vistlabel.projects.errors import ProjectNotFoundError
from vistlabel.video.frames import load_frame_timestamps


def require_project(request: Request, project_id: str) -> ProjectState:
    try:
        return request.app.state.project_store.get(project_id)
    except ProjectNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def require_project_with_video(
    request: Request, project_id: str
) -> tuple[ProjectState, Path, list[float]]:
    """Returns (project, project_dir, frame_timestamps); 409 if no video is
    attached yet (a project is only usable once it has a video)."""
    project = require_project(request, project_id)
    if project.source is None:
        raise HTTPException(status_code=409, detail="project has no finalized video yet")

    settings = request.app.state.settings
    pdir = project_dir(settings.workspace.root, project_id)
    timestamps = load_frame_timestamps(pdir / "cache" / "frame_timestamps.json")
    return project, pdir, timestamps


def require_frame_index(frame_index: int, timestamps: list[float]) -> float:
    if not 0 <= frame_index < len(timestamps):
        raise HTTPException(
            status_code=404, detail=f"frame_index out of range [0, {len(timestamps)})"
        )
    return timestamps[frame_index]
