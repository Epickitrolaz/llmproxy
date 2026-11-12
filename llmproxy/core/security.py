"""Dashboard security helpers: admin authentication, CORS, and secret redaction."""

from __future__ import annotations

import hmac
import logging
import os
import secrets
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from fastapi import HTTPException, Request, status

from llmproxy.core.manager import plugin_manager

logger = logging.getLogger(__name__)

ADMIN_TOKEN_ENV = "LLMPROXY_ADMIN_TOKEN"
CORS_ORIGINS_ENV = "LLMPROXY_CORS_ORIGINS"
ADMIN_TOKEN_FILENAME = "admin_token"

# Fragments that mark a configuration value as credential material. Keys that
# contain any of these are masked before an account is returned by the
# dashboard API.
SENSITIVE_CONFIG_HINTS = (
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "cookie",
    "password",
    "refresh",
    "secret",
    "session",
    "token",
)

_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}

_token_lock = threading.Lock()


def _admin_token_file() -> Path:
    return Path(plugin_manager.data_dir) / ADMIN_TOKEN_FILENAME


def _read_admin_token_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""
    except OSError as exc:
        logger.warning("Failed to read admin token from %s: %s", path, exc)
        return ""


def get_admin_token() -> str:
    """Return the configured admin token.

    ``LLMPROXY_ADMIN_TOKEN`` wins. When it is not set a random token is
    generated once and persisted under ``data/admin_token`` (mode 0600) so a
    local dashboard can still authenticate without environment configuration.
    """
    env_token = (os.environ.get(ADMIN_TOKEN_ENV) or "").strip()
    if env_token:
        return env_token

    path = _admin_token_file()
    token = _read_admin_token_file(path)
    if token:
        return token

    with _token_lock:
        token = _read_admin_token_file(path)
        if token:
            return token
        token = secrets.token_urlsafe(32)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(token, encoding="utf-8")
            os.chmod(path, 0o600)
            logger.warning(
                "No %s configured. Generated a dashboard admin token at %s. "
                "Set %s before exposing the dashboard on a public interface.",
                ADMIN_TOKEN_ENV,
                path,
                ADMIN_TOKEN_ENV,
            )
        except OSError as exc:
            logger.warning("Failed to persist generated admin token to %s: %s", path, exc)
        return token


def admin_token_from_env() -> bool:
    """Return True when the admin token is explicitly configured via env."""
    return bool((os.environ.get(ADMIN_TOKEN_ENV) or "").strip())


def _supplied_admin_token(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return (request.headers.get("x-admin-token") or "").strip()


def is_admin_request(request: Request) -> bool:
    """Return True when the request carries the valid dashboard admin token."""
    supplied = _supplied_admin_token(request)
    if not supplied:
        return False
    expected = get_admin_token()
    return hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8"))


def _is_loopback_request(request: Request) -> bool:
    client = request.client
    if client is None:
        return False
    host = client.host or ""
    return host in _LOOPBACK_HOSTS or host.startswith("127.")


def require_admin(request: Request) -> None:
    """FastAPI dependency guarding every ``/api`` dashboard route."""
    if not admin_token_from_env() and not _is_loopback_request(request):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "admin_token_not_configured",
                "message": (
                    f"Dashboard access from a non-loopback address requires "
                    f"{ADMIN_TOKEN_ENV} to be set."
                ),
            },
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not is_admin_request(request):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "admin_auth_required",
                "message": "A valid dashboard admin token is required.",
            },
            headers={"WWW-Authenticate": "Bearer"},
        )


def cors_origins() -> List[str]:
    """Return the configured CORS origin allowlist (empty means disabled)."""
    raw = os.environ.get(CORS_ORIGINS_ENV) or ""
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def cors_allow_credentials(origins: Iterable[str]) -> bool:
    """Credentials are only safe with an explicit origin allowlist."""
    return "*" not in list(origins)


def mask_secret(value: str) -> str:
    """Mask a credential while keeping a short suffix for identification."""
    if not value:
        return ""
    if len(value) <= 4:
        return "*" * len(value)
    return f"{'*' * 8}{value[-4:]}"


def _is_sensitive_config_key(key: str) -> bool:
    normalized = str(key).lower().replace("-", "_")
    return any(hint in normalized for hint in SENSITIVE_CONFIG_HINTS)


def redact_extra_config(value: Any) -> Any:
    """Recursively mask sensitive values in an account ``extra_config``."""
    if isinstance(value, dict):
        redacted: Dict[str, Any] = {}
        for key, item in value.items():
            if _is_sensitive_config_key(key) and isinstance(item, str) and item:
                redacted[key] = mask_secret(item)
            else:
                redacted[key] = redact_extra_config(item)
        return redacted
    if isinstance(value, list):
        return [redact_extra_config(item) for item in value]
    return value


def redact_account(account: Any) -> Any:
    """Return a copy of an account with its credentials masked."""
    redacted = account.model_copy(deep=True)
    redacted.api_key = mask_secret(redacted.api_key or "")
    redacted.extra_config = redact_extra_config(redacted.extra_config or {})
    return redacted


def redact_api_key(api_key: Any) -> Any:
    """Return a copy of a local API key with the literal secret masked."""
    redacted = api_key.model_copy(deep=True)
    redacted.key = mask_secret(redacted.key or "")
    return redacted


__all__ = [
    "ADMIN_TOKEN_ENV",
    "CORS_ORIGINS_ENV",
    "SENSITIVE_CONFIG_HINTS",
    "admin_token_from_env",
    "cors_allow_credentials",
    "cors_origins",
    "get_admin_token",
    "is_admin_request",
    "mask_secret",
    "redact_account",
    "redact_api_key",
    "redact_extra_config",
    "require_admin",
]
