# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Authentication integration point. Disabled by default -- see
:class:`~vistlabel.settings.AuthSettings`.

A single shared-secret `Authorization: Bearer <key>` check, not a
user/session system: binding to localhost (the default) is the real
protection for a single-workstation install. This exists for the
"reachable from more than just its own host, no reverse-proxy/SSO in
front of it yet" case -- production deployment beyond a trusted network
should still prefer a real reverse-proxy-terminated auth scheme.
"""

from __future__ import annotations

import hmac

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from vistlabel.settings import AuthSettings

#: Always reachable without a key -- an uptime/monitoring probe must not
#: itself need a credential, and this endpoint returns no project data.
_EXEMPT_PATHS = frozenset({"/api/v1/health"})


def _key_is_valid(candidate: str, configured_keys: list[str]) -> bool:
    # Constant-time comparison against every configured key -- a
    # request's failure latency must not leak which prefix matched.
    return any(hmac.compare_digest(candidate, key) for key in configured_keys)


class ApiKeyAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, settings: AuthSettings) -> None:
        super().__init__(app)
        self._settings = settings

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if not self._settings.enabled or request.url.path in _EXEMPT_PATHS:
            return await call_next(request)

        header = request.headers.get("authorization", "")
        scheme, _, candidate = header.partition(" ")
        if scheme.lower() != "bearer" or not candidate:
            return JSONResponse(
                {"detail": "missing or malformed Authorization: Bearer <api-key> header"},
                status_code=401,
            )
        if not _key_is_valid(candidate, self._settings.api_keys):
            return JSONResponse({"detail": "invalid API key"}, status_code=401)
        return await call_next(request)
