# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Sign-in, sign-out and session-status endpoints.

The account itself is created on the server (``vistlabel auth
set-password``); see :mod:`vistlabel.auth`. Requests are checked by
:class:`~vistlabel.auth.middleware.AuthMiddleware`.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

from vistlabel.auth.sessions import SESSION_COOKIE_NAME

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])
logger = logging.getLogger(__name__)


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=1024)


class AuthStatus(BaseModel):
    authenticated: bool
    username: str | None = None
    account_configured: bool
    session_expires_at: str | None = None


def _client_key(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _set_cookie(request: Request, response: Response, token: str) -> None:
    security = request.app.state.settings.security
    response.set_cookie(
        SESSION_COOKIE_NAME,
        token,
        max_age=security.session_timeout_minutes * 60,
        httponly=True,
        samesite="lax",
        secure=security.secure_cookies,
        path="/",
    )


@router.post("/login")
def login(payload: LoginRequest, request: Request, response: Response) -> AuthStatus:
    accounts = request.app.state.accounts
    sessions = request.app.state.sessions
    limiter = request.app.state.login_limiter
    key = _client_key(request)

    if not accounts.is_configured():
        raise HTTPException(
            status_code=503,
            detail="no account exists yet: run `vistlabel auth set-password` on the server",
        )
    if limiter.is_blocked(key):
        retry = limiter.retry_after_seconds(key)
        raise HTTPException(
            status_code=429,
            detail=f"too many failed sign-in attempts: wait {retry} s and try again",
            headers={"Retry-After": str(retry)},
        )
    if not accounts.verify(payload.username, payload.password):
        limiter.record_failure(key)
        # The password is never logged, not even on failure.
        logger.warning("login failed", extra={"client": key, "username": payload.username})
        raise HTTPException(status_code=401, detail="invalid username or password")
    limiter.reset(key)
    token, session = sessions.create(payload.username.strip())
    _set_cookie(request, response, token)
    logger.info("login succeeded", extra={"client": key, "username": session.username})
    return AuthStatus(
        authenticated=True,
        username=session.username,
        account_configured=True,
        session_expires_at=session.expires_at,
    )


@router.post("/logout")
def logout(request: Request, response: Response) -> AuthStatus:
    request.app.state.sessions.revoke(request.cookies.get(SESSION_COOKIE_NAME))
    response.delete_cookie(SESSION_COOKIE_NAME, path="/")
    return AuthStatus(
        authenticated=False, account_configured=request.app.state.accounts.is_configured()
    )


@router.get("/status")
def status(request: Request) -> AuthStatus:
    session = getattr(request.state, "session", None)
    return AuthStatus(
        authenticated=session is not None,
        username=session.username if session else None,
        account_configured=request.app.state.accounts.is_configured(),
        session_expires_at=session.expires_at if session else None,
    )
