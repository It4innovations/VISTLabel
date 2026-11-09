# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Server-rendered HTML pages (Jinja2 templates, no React SPA)."""

from __future__ import annotations

import hashlib
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from vistlabel.projects.errors import ProjectNotFoundError

router = APIRouter(tags=["pages"])

templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))


def asset_version() -> str:
    """Invalidate every page's module graph when any JS or CSS changes."""
    static_dir = Path(__file__).parent.parent / "static"
    digest = hashlib.sha256()
    for path in sorted(static_dir.rglob("*")):
        if path.suffix in {".js", ".css"}:
            digest.update(path.relative_to(static_dir).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


@router.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    projects = request.app.state.project_store.list_projects()
    return templates.TemplateResponse(
        request, "index.html",
        {"projects": projects, "asset_version": asset_version()},
    )


@router.get("/projects/{project_id}", response_class=HTMLResponse)
def annotation_workspace(project_id: str, request: Request) -> HTMLResponse:
    try:
        project = request.app.state.project_store.get(project_id)
    except ProjectNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if project.source is None:
        raise HTTPException(status_code=409, detail="project has no finalized video yet")
    return templates.TemplateResponse(
        request, "annotation.html",
        {"project": project, "asset_version": asset_version()},
    )


def _safe_next(target: str) -> str:
    """Only a same-site path, never ``//host`` or an absolute URL."""
    return target if target.startswith("/") and not target.startswith("//") else "/"


@router.get("/login", response_class=HTMLResponse, response_model=None)
def login_page(request: Request, next: str = "/") -> Response:  # noqa: A002
    target = _safe_next(next)
    if getattr(request.state, "session", None) is not None:
        return RedirectResponse(target, status_code=303)
    if not request.app.state.accounts.is_configured():
        return RedirectResponse("/setup", status_code=303)
    return templates.TemplateResponse(
        request, "login.html", {"next": target, "asset_version": asset_version()}
    )


@router.get("/setup", response_class=HTMLResponse, response_model=None)
def setup_page(request: Request) -> Response:
    if request.app.state.accounts.is_configured():
        return RedirectResponse("/login", status_code=303)
    return templates.TemplateResponse(
        request, "setup_required.html", {"asset_version": asset_version()}
    )
