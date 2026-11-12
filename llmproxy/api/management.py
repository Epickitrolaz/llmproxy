"""
CLIProxyAPI-compatible management API.

T3 Code and OpenCode can use this router as a read-only usage hub. The two
endpoints consumed by T3 Code are:

  GET  /v0/management/auth-files
  POST /v0/management/api-call

The legacy quota-scheduler endpoint is also exposed for older clients.
"""

from __future__ import annotations

import asyncio
import json
import re
import secrets
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from llmproxy.core.antigravity_oauth import antigravity_oauth
from llmproxy.core.codex_oauth import codex_oauth
from llmproxy.core.manager import plugin_manager
from llmproxy.schemas import Account

router = APIRouter(prefix="/v0/management", tags=["management"])

CODEX_BASE = "https://chatgpt.com/backend-api/wham"
ANTIGRAVITY_SUMMARY_URL = "https://daily-cloudcode-pa.googleapis.com/v1internal:retrieveUserQuotaSummary"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"
CLAUDE_USAGE_URL = "https://api.anthropic.com/api/oauth/usage"

# Only these exact upstream requests may be performed with an account token.
_ALLOWED_CALLS: Dict[str, set[Tuple[str, str]]] = {
    "codex": {
        ("GET", f"{CODEX_BASE}/usage"),
        ("GET", f"{CODEX_BASE}/rate-limit-reset-credits"),
        ("POST", f"{CODEX_BASE}/rate-limit-reset-credits/consume"),
    },
    "antigravity": {
        ("POST", ANTIGRAVITY_SUMMARY_URL),
        ("GET", GOOGLE_USERINFO_URL),
    },
    "claude": {
        ("GET", CLAUDE_USAGE_URL),
    },
    "opencode": {
        ("GET", "https://opencode.ai/zen/go/v1/usage"),
        ("GET", "https://opencode.ai/zen/v1/usage"),
    },
}

_PROVIDER_KINDS = {
    "codex": "codex",
    "antigravity": "antigravity",
    "claude_code": "claude",
    "opencode_free": "opencode",
}

_BLOCKED_HEADERS = {
    "host",
    "content-length",
    "transfer-encoding",
    "connection",
    "proxy-authorization",
}


class ManagementApiCall(BaseModel):
    auth_index: str
    method: str = "GET"
    url: str
    header: Optional[Dict[str, str]] = None
    data: Optional[Any] = None


class ResetQuotaRequest(BaseModel):
    auth_index: Optional[str] = None


def _provider_kind(provider_id: str) -> Optional[str]:
    return _PROVIDER_KINDS.get(provider_id)


def _provider_enabled(provider_id: str) -> bool:
    plugin = plugin_manager.get_plugin(provider_id)
    cfg = plugin_manager.configs.get(provider_id, {})
    if plugin is None:
        return bool(cfg.get("enabled", True))
    return bool(cfg.get("enabled", plugin.metadata.enabled))


def _authorize(request: Request) -> None:
    supplied = ""
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        supplied = auth[7:].strip()
    if not supplied:
        supplied = (request.headers.get("x-management-key") or "").strip()

    expected = plugin_manager.get_management_key()
    if supplied and secrets.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8")):
        return

    # Local API keys are also accepted so the hub can be configured without
    # copying a second secret from the dashboard.
    api_key = plugin_manager.get_api_key_by_secret(supplied) if supplied else None
    if api_key and api_key.is_valid():
        return

    raise HTTPException(
        status_code=401,
        detail={"error": "invalid_management_key"},
    )


def _account_email(account: Account) -> str:
    email = str(account.extra_config.get("email") or "").strip()
    if email:
        return email
    match = re.search(r"[\w.+-]+@[\w.-]+", account.name or "")
    return match.group(0) if match else ""


def _find_account(auth_index: str) -> Optional[Account]:
    for account in plugin_manager.get_accounts():
        if account.id == auth_index:
            return account
    return None


def _normalize_url(url: str) -> str:
    return url.strip().rstrip("/")


def _build_headers(account: Account, provider: str, supplied: Optional[Dict[str, str]]) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    for name, value in (supplied or {}).items():
        if name.lower() in _BLOCKED_HEADERS:
            continue
        headers[name] = str(value).replace("$TOKEN$", account.api_key or "")

    if not any(name.lower() == "authorization" for name in headers):
        headers["Authorization"] = f"Bearer {account.api_key or ''}"

    if provider == "codex":
        account_id = str(account.extra_config.get("account_id") or "").strip()
        if account_id and not any(name.lower() == "chatgpt-account-id" for name in headers):
            headers["ChatGPT-Account-Id"] = account_id
        if not any(name.lower() == "user-agent" for name in headers):
            headers["User-Agent"] = "codex/cli/0.167.0 (linux; x86_64)"
    elif provider == "antigravity":
        if not any(name.lower() == "user-agent" for name in headers):
            headers["User-Agent"] = (
                "antigravity/hub/2.24.0 (aidev_client; os_type=linux; arch=x64; cl=963137146)"
            )
    elif provider == "claude":
        if not any(name.lower() == "user-agent" for name in headers):
            headers["User-Agent"] = "claude-code/2.1.273"

    return headers


async def _refresh_account_if_needed(account: Account, provider: str, force: bool = False) -> None:
    if not account.extra_config.get("refresh_token"):
        return
    expires_at = float(account.extra_config.get("expires_at") or 0)
    if not force and account.api_key and (expires_at <= 0 or expires_at - time.time() >= 120):
        return

    if provider == "codex":
        refreshed = await codex_oauth.refresh_account_token(account)
    elif provider == "antigravity":
        refreshed = await antigravity_oauth.refresh_account_token(account)
    else:
        refreshed = False

    if refreshed:
        plugin_manager._save_accounts()


async def _execute_api_call(account: Account, provider: str, call: ManagementApiCall) -> Dict[str, Any]:
    method = call.method.upper().strip()
    url = _normalize_url(call.url)

    if method not in ("GET", "POST"):
        return {"status_code": 405, "body": json.dumps({"error": "method_not_allowed"})}
    if method == "GET" and call.data is not None:
        return {"status_code": 405, "body": json.dumps({"error": "method_not_allowed"})}
    if (method, url) not in _ALLOWED_CALLS.get(provider, set()):
        return {"status_code": 403, "body": json.dumps({"error": "unsupported_request"})}
    if not account.api_key:
        return {"status_code": 401, "body": json.dumps({"error": "account_has_no_token"})}

    await _refresh_account_if_needed(account, provider)

    proxy_url, _ = plugin_manager.resolve_request_proxy(
        variables={}, account=account, provider_id=account.provider_id
    )

    async def perform() -> httpx.Response:
        headers = _build_headers(account, provider, call.header)
        async with httpx.AsyncClient(
            timeout=15.0,
            proxy=proxy_url,
            follow_redirects=False,
        ) as client:
            if method == "GET":
                return await client.get(url, headers=headers)
            return await client.post(url, headers=headers, json=call.data)

    try:
        response = await perform()
        if response.status_code == 401 and account.extra_config.get("refresh_token"):
            await _refresh_account_if_needed(account, provider, force=True)
            response = await perform()
        return {"status_code": response.status_code, "body": response.text}
    except Exception as exc:
        return {
            "status_code": 502,
            "body": json.dumps({"error": f"Upstream request failed: {exc}"}),
        }


@router.get("/auth-files")
async def list_auth_files(request: Request):
    """List pooled accounts in the shape expected by CLIProxyAPI management clients."""
    _authorize(request)

    files: List[Dict[str, Any]] = []
    for account in plugin_manager.get_accounts():
        provider = _provider_kind(account.provider_id)
        if not provider or not _provider_enabled(account.provider_id):
            continue

        entry: Dict[str, Any] = {
            "id": account.id,
            "auth_index": account.id,
            "provider": provider,
            "disabled": not account.enabled,
        }

        email = _account_email(account)
        if email:
            entry["email"] = email

        if provider == "antigravity":
            project_id = str(account.extra_config.get("project_id") or "").strip()
            if project_id:
                entry["project_id"] = project_id

        if provider == "codex":
            account_id = str(account.extra_config.get("account_id") or "").strip()
            plan_type = str(account.extra_config.get("plan_type") or "").strip()
            id_token: Dict[str, Any] = {}
            if account_id:
                id_token["chatgpt_account_id"] = account_id
            if plan_type:
                id_token["chatgpt_plan_type"] = plan_type
                id_token["plan_type"] = plan_type
            if id_token:
                entry["id_token"] = id_token

        files.append(entry)

    return {"files": files}


@router.post("/api-call")
async def management_api_call(request: Request, call: ManagementApiCall):
    """Issue one allow-listed upstream usage request with an account's credentials."""
    _authorize(request)

    account = _find_account(call.auth_index)
    if not account:
        return {"status_code": 404, "body": json.dumps({"error": "unknown_account"})}

    provider = _provider_kind(account.provider_id)
    if not provider:
        return {"status_code": 404, "body": json.dumps({"error": "unsupported_provider"})}

    return await _execute_api_call(account, provider, call)


@router.post("/reset-quota")
async def reset_quota(request: Request, body: ResetQuotaRequest):
    """Clear an account's local cooldown after a reset credit is redeemed."""
    _authorize(request)

    if body.auth_index:
        account = _find_account(body.auth_index)
        if not account:
            raise HTTPException(status_code=404, detail={"error": "unknown_account"})
        plugin_manager.reset_account_cooldown(account.id)

    return {"status": "ok"}


@router.get("/quota-scheduler/status")
async def quota_scheduler_status(request: Request):
    """
    Legacy CLIProxyAPI quota status shape.

    Current T3 Code releases use auth-files + api-call, but older clients still
    read this endpoint.
    """
    _authorize(request)

    codex_accounts = [
        account
        for account in plugin_manager.get_accounts()
        if (
            _provider_kind(account.provider_id) == "codex"
            and account.enabled
            and _provider_enabled(account.provider_id)
        )
    ]
    if not codex_accounts:
        return {"accounts": {}}

    # Imported lazily to avoid a module import cycle with the dashboard router.
    from llmproxy.dashboard.router import fetch_codex_quota

    async def build(account: Account) -> Optional[Tuple[str, Dict[str, Any]]]:
        quota = await fetch_codex_quota(account)
        if quota.get("error"):
            return None

        windows: Dict[str, Any] = {}
        for group in quota.get("groups", []):
            for bucket in group.get("buckets", []):
                name = str(bucket.get("displayName") or "").lower()
                if "5-hour" in name or "primary" in name:
                    key = "five_hour"
                elif "weekly" in name or "secondary" in name:
                    key = "weekly"
                else:
                    continue
                used = float(bucket.get("usedPercent") or 0)
                windows[key] = {
                    "used_percent": used,
                    "reset_at": bucket.get("resetTime"),
                    "known": True,
                    "hard_limited": used >= 100,
                }

        if not windows:
            return None

        return account.id, {
            "provider": "codex",
            "plan": quota.get("plan_type") or "standard",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            **windows,
        }

    results = await asyncio.gather(*(build(account) for account in codex_accounts), return_exceptions=True)
    return {"accounts": {item[0]: item[1] for item in results if isinstance(item, tuple)}}
