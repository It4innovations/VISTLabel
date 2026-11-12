# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Local Help pages.

``/help`` lists the documentation (the keyboard shortcuts are a section of
the user guide, so they are not repeated there);
``/help/<name>`` renders one of the listed Markdown files from the ``docs``
directory (the repository's, or the copy packaged into a wheel) with the
local converter in :mod:`vistlabel.help.markdown`, and
``/help/images/<file>`` serves the pictures those pages show from the
``docs/images`` folder. Only the pages in :data:`HELP_INDEX` and plain image
files (no sub-directories, extensions in
:data:`~vistlabel.help.markdown.IMAGE_TYPES`) are served, so no other file is
reachable by name, and nothing here contacts the network.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse

from vistlabel.api.pages import asset_version, templates
from vistlabel.help.markdown import IMAGE_NAME, IMAGE_TYPES, Rendered, render_markdown

router = APIRouter(tags=["help"])


@dataclass(frozen=True)
class HelpPage:
    name: str
    title: str
    summary: str


#: The Help index, in display order; also the allow-list of served pages.
HELP_INDEX: tuple[HelpPage, ...] = (
    HelpPage(
        "user-guide",
        "User guide",
        "Projects, the annotation workspace, SAM prompts, brush and eraser, "
        "propagation, review, splits and export, with every keyboard shortcut.",
    ),
)
_PAGES = {page.name: page for page in HELP_INDEX}


def docs_directories() -> list[Path]:
    package_docs = Path(__file__).resolve().parent.parent / "docs"
    repo_docs = Path(__file__).resolve().parents[3] / "docs"
    return [d for d in (package_docs, repo_docs) if d.is_dir()]


def find_doc(name: str) -> Path | None:
    if name not in _PAGES:
        return None
    for directory in docs_directories():
        candidate = directory / f"{name}.md"
        if candidate.is_file():
            return candidate
    return None


def find_image(filename: str) -> Path | None:
    if not IMAGE_NAME.match(filename):
        return None
    for directory in docs_directories():
        candidate = directory / "images" / filename
        if candidate.is_file():
            return candidate
    return None


#: An SVG opened on its own must not run script or load anything.
_IMAGE_HEADERS = {
    "Cache-Control": "private, max-age=3600",
    "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; sandbox",
    "X-Content-Type-Options": "nosniff",
}


@lru_cache(maxsize=32)
def _render_cached(path: Path, mtime_ns: int) -> Rendered:
    # Keyed on the modification time, so an edited doc shows on reload.
    return render_markdown(path.read_text(encoding="utf-8"), served=_PAGES.keys())


def _index_entries() -> list[dict[str, Any]]:
    return [
        {
            "name": page.name,
            "title": page.title,
            "summary": page.summary,
            "available": find_doc(page.name) is not None,
        }
        for page in HELP_INDEX
    ]


def _respond(
    request: Request, doc: dict[str, Any] | None, status_code: int = 200
) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "help.html",
        {
            "index": _index_entries(),
            "doc": doc,
            "asset_version": asset_version(),
        },
        status_code=status_code,
    )


@router.get("/help", response_class=HTMLResponse)
def help_index(request: Request) -> HTMLResponse:
    return _respond(request, None)


@router.get("/help/images/{filename}")
def help_image(filename: str) -> FileResponse:
    path = find_image(filename)
    if path is None:
        raise HTTPException(status_code=404, detail="Not Found")
    return FileResponse(
        path, media_type=IMAGE_TYPES[path.suffix.lower()], headers=_IMAGE_HEADERS
    )


@router.get("/help/{name}", response_class=HTMLResponse)
def help_page(name: str, request: Request) -> HTMLResponse:
    path = find_doc(name)
    if path is None:
        missing = {"name": name, "title": "Not found", "html": "", "headings": [], "missing": True}
        return _respond(request, missing, status_code=404)
    rendered = _render_cached(path, path.stat().st_mtime_ns)
    return _respond(
        request,
        {
            "name": name,
            "title": rendered.title or _PAGES[name].title,
            "html": rendered.html,
            "headings": [h for h in rendered.headings if 2 <= h.level <= 3],
            "missing": False,
        },
    )
