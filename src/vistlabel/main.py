# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""FastAPI application factory and the ``vistlabel`` CLI entry point."""

from __future__ import annotations

import argparse
import mimetypes
import sys
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from vistlabel import __version__
from vistlabel.api.auth import ApiKeyAuthMiddleware
from vistlabel.api.auto_preannotate import router as auto_preannotate_router
from vistlabel.api.dataset_imports import router as dataset_imports_router
from vistlabel.api.exports import router as exports_router
from vistlabel.api.frame_annotations import router as frame_annotations_router
from vistlabel.api.health import router as health_router
from vistlabel.api.help import router as help_router
from vistlabel.api.objects import router as objects_router
from vistlabel.api.pages import router as pages_router
from vistlabel.api.projects import router as projects_router
from vistlabel.api.propagation import router as propagation_router
from vistlabel.api.review_queue import router as review_queue_router
from vistlabel.api.segmentation import router as segmentation_router
from vistlabel.jobs.manager import JobManager
from vistlabel.logging_config import configure_logging
from vistlabel.projects.store import ProjectStore
from vistlabel.segmentation.registry import build_registry
from vistlabel.segmentation.sessions import BackendSessionManager
from vistlabel.settings import AppSettings, load_settings

_STATIC_DIR = Path(__file__).parent / "static"

# StaticFiles derives Content-Type from the stdlib ``mimetypes`` module, which
# on Windows seeds itself from the registry -- where HKCR\.js is routinely
# "text/plain" (some SDK and antivirus installers rewrite it). Browsers apply
# strict MIME checking to ES modules and refuse to execute such a response,
# which silently kills the entire frontend. Pin the correct types explicitly
# rather than depending on per-machine registry state.
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/css", ".css")


def create_app(settings: AppSettings | None = None) -> FastAPI:
    settings = settings or load_settings()
    configure_logging(settings.logging)

    app = FastAPI(title="VISTLabel", version=__version__)
    app.state.started_at_monotonic = time.monotonic()
    app.state.settings = settings
    app.state.registry = build_registry(settings)
    app.state.project_store = ProjectStore(settings.workspace.root)
    app.state.project_store.recover_incomplete_projects()
    app.state.project_store.recover_interrupted_tracking_runs()
    app.state.backend_sessions = BackendSessionManager(
        max_sessions=settings.tracking.max_open_backend_sessions
    )
    app.state.job_manager = JobManager(
        settings.workspace.root, max_workers=settings.runtime.maximum_concurrent_gpu_jobs
    )
    app.state.job_manager.recover_interrupted_jobs()
    app.add_middleware(ApiKeyAuthMiddleware, settings=settings.auth)
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
    app.include_router(health_router)
    # Before projects_router, so /projects/import-dataset is never read as
    # a project_id.
    app.include_router(dataset_imports_router)
    app.include_router(projects_router)
    app.include_router(frame_annotations_router)
    app.include_router(objects_router)
    app.include_router(segmentation_router)
    app.include_router(propagation_router)
    app.include_router(review_queue_router)
    app.include_router(auto_preannotate_router)
    app.include_router(exports_router)
    app.include_router(help_router)
    app.include_router(pages_router)
    return app


app = create_app()


def _serve(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    fastapi_app = create_app(settings)
    uvicorn.run(fastapi_app, host=args.host, port=args.port)
    return 0


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vistlabel")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve_parser = subparsers.add_parser("serve", help="run the VISTLabel FastAPI application")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=9080)
    serve_parser.add_argument("--config", type=Path, default=None, help="path to config/app.yaml")
    serve_parser.set_defaults(func=_serve)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(cli())
