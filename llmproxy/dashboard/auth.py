"""Unauthenticated endpoints used by the dashboard login gate."""

from __future__ import annotations

from fastapi import APIRouter, Request

from llmproxy.core.security import (
    ADMIN_TOKEN_ENV,
    admin_token_from_env,
    is_admin_request,
)

router = APIRouter(prefix="/api", tags=["auth"])


@router.get("/auth/status")
async def auth_status(request: Request):
    """Report whether the supplied admin token is valid.

    This endpoint never returns credential material. It is intentionally
    public so the dashboard can decide between showing the login gate and
    loading data after a reload.
    """
    return {
        "authenticated": is_admin_request(request),
        "required": True,
        "configured_via_env": admin_token_from_env(),
        "token_env": ADMIN_TOKEN_ENV,
    }
