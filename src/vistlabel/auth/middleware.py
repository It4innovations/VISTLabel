# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Authentication and same-origin enforcement middleware.

* Every request outside the public allow-list needs a valid session cookie.
  API paths get a ``401`` JSON response; page paths are redirected to
  ``/login`` (or ``/setup`` while no account exists).
* State-changing API requests (``POST``/``PUT``/``PATCH``/``DELETE``) must
  carry ``X-Requested-With: VISTLabel`` and, when the browser sends an
  ``Origin``, ``Sec-Fetch-Site`` or ``Referer`` header, be same-origin.
  Together with ``SameSite=Lax`` cookies this blocks cross-site request
  forgery without a token round-trip.
"""

from __future__ import annotations

from urllib.parse import quote, urlsplit

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse, Response
from starlette.types import ASGIApp

from vistlabel.auth.account import AccountStore
from vistlabel.auth.sessions import SESSION_COOKIE_NAME, SessionStore

CSRF_HEADER = "x-requested-with"
CSRF_HEADER_VALUE = "VISTLabel"

#: Reachable without a session. Health and version carry no project data and
#: must answer an uptime probe that has no credential.
PUBLIC_PATHS: frozenset[str] = frozenset(
    {
        "/login",
        "/setup",
        "/api/v1/health",
        "/api/v1/version",
        "/api/v1/auth/login",
        "/api/v1/auth/status",
    }
)
PUBLIC_PREFIXES: tuple[str, ...] = ("/static/",)
UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def _is_public(path: str) -> bool:
    return path in PUBLIC_PATHS or path.startswith(PUBLIC_PREFIXES)


def _same_origin(request: Request) -> bool:
    host = request.headers.get("host", "")
    origin = request.headers.get("origin")
    if origin:
        return urlsplit(origin).netloc == host
    fetch_site = request.headers.get("sec-fetch-site")
    if fetch_site:
        return fetch_site in ("same-origin", "none")
    referer = request.headers.get("referer")
    if referer:
        return urlsplit(referer).netloc == host
    return True


def csrf_violation(request: Request) -> str | None:
    if request.method not in UNSAFE_METHODS:
        return None
    if request.headers.get(CSRF_HEADER) != CSRF_HEADER_VALUE:
        return f"missing {CSRF_HEADER}: {CSRF_HEADER_VALUE} header"
    if not _same_origin(request):
        return "cross-origin request rejected"
    return None


class AuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, *, sessions: SessionStore, accounts: AccountStore) -> None:
        super().__init__(app)
        self._sessions = sessions
        self._accounts = accounts

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        path = request.url.path
        is_api = path.startswith("/api/")
        session = self._sessions.resolve(request.cookies.get(SESSION_COOKIE_NAME))
        request.state.session = session

        if is_api:
            violation = csrf_violation(request)
            if violation is not None:
                return JSONResponse(
                    {"detail": violation, "code": "CSRF_REJECTED"}, status_code=403
                )

        if _is_public(path) or session is not None:
            return await call_next(request)

        if is_api:
            return JSONResponse(
                {"detail": "authentication required: sign in at /login", "code": "AUTH_REQUIRED"},
                status_code=401,
            )
        if not self._accounts.is_configured():
            return RedirectResponse("/setup", status_code=303)
        target = path
        if request.url.query:
            target += "?" + request.url.query
        return RedirectResponse(f"/login?next={quote(target, safe='')}", status_code=303)
