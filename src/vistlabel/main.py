# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""FastAPI application factory and the ``vistlabel`` CLI entry point."""

from __future__ import annotations

import argparse
import getpass
import mimetypes
import os
import sys
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from vistlabel import __version__
from vistlabel.api.auth import router as auth_router
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
from vistlabel.auth.account import AccountStore
from vistlabel.auth.middleware import AuthMiddleware
from vistlabel.auth.passwords import WeakPasswordError
from vistlabel.auth.sessions import LoginRateLimiter, SessionStore
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
    app.state.accounts = AccountStore(settings.workspace.root)
    app.state.sessions = SessionStore(
        settings.workspace.root, timeout_minutes=settings.security.session_timeout_minutes
    )
    app.state.login_limiter = LoginRateLimiter(
        limit=settings.security.login_rate_limit,
        window_seconds=settings.security.login_rate_window_seconds,
    )
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
    app.add_middleware(AuthMiddleware, sessions=app.state.sessions, accounts=app.state.accounts)
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
    app.include_router(health_router)
    app.include_router(auth_router)
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


def _version(_: argparse.Namespace) -> int:
    print(__version__)
    return 0


#: Read instead of prompting when there is no terminal (scripts, batch jobs).
PASSWORD_ENV_VAR = "VISTLABEL_PASSWORD"


def _auth_set_password(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    store = AccountStore(settings.workspace.root)
    password = os.environ.get(PASSWORD_ENV_VAR)
    if not password:
        if not sys.stdin.isatty():
            print(
                f"no terminal available; set {PASSWORD_ENV_VAR} in the environment instead",
                file=sys.stderr,
            )
            return 2
        password = getpass.getpass("New password: ")
        confirm = getpass.getpass("Repeat password: ")
        if password != confirm:
            print("passwords do not match", file=sys.stderr)
            return 2
    try:
        account = store.set_password(args.username, password)
    except (WeakPasswordError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    # Any existing sessions belong to the old credential; drop them.
    SessionStore(settings.workspace.root).revoke_all()
    print(f"account {account.username!r} updated at {store.path}")
    return 0


def _auth_status(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    account = AccountStore(settings.workspace.root).load()
    if account is None:
        print("no account configured; run: vistlabel auth set-password --username <name>")
        return 1
    sessions = SessionStore(settings.workspace.root)
    print(f"username: {account.username}")
    print(f"algorithm: {account.algorithm}")
    print(f"updated_at: {account.updated_at}")
    print(f"active sessions: {sessions.active_count()}")
    return 0


def _auth_revoke_sessions(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    count = SessionStore(settings.workspace.root).revoke_all()
    print(f"revoked {count} session(s)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vistlabel")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve_parser = subparsers.add_parser("serve", help="run the VISTLabel FastAPI application")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=9080)
    serve_parser.add_argument("--config", type=Path, default=None, help="path to config/app.yaml")
    serve_parser.set_defaults(func=_serve)

    version_parser = subparsers.add_parser("version", help="print the installed version")
    version_parser.set_defaults(func=_version)

    auth_parser = subparsers.add_parser("auth", help="manage the single application account")
    auth_sub = auth_parser.add_subparsers(dest="auth_command", required=True)
    set_pw = auth_sub.add_parser(
        "set-password",
        help=f"create the account or change its password (prompts, or reads {PASSWORD_ENV_VAR})",
    )
    set_pw.add_argument("--username", default="vistlabel")
    set_pw.add_argument("--config", type=Path, default=None, help="path to config/app.yaml")
    set_pw.set_defaults(func=_auth_set_password)
    status = auth_sub.add_parser("status", help="show the configured account")
    status.add_argument("--config", type=Path, default=None, help="path to config/app.yaml")
    status.set_defaults(func=_auth_status)
    revoke = auth_sub.add_parser("revoke-sessions", help="sign out every browser session")
    revoke.add_argument("--config", type=Path, default=None, help="path to config/app.yaml")
    revoke.set_defaults(func=_auth_revoke_sessions)
    return parser


def cli(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(cli())
