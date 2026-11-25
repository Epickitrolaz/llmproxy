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

from llmproxy.core.interface import BasePlugin
from llmproxy.core.manager import plugin_manager
from llmproxy.core.usage import stored_usage_limits
from llmproxy.schemas import Account

router = APIRouter(prefix="/v0/management", tags=["management"])

CODEX_BASE = "https://chatgpt.com/backend-api/wham"
ANTIGRAVITY_SUMMARY_URL = "https://daily-cloudcode-pa.googleapis.com/v1internal:retrieveUserQuotaSummary"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"
CLAUDE_USAGE_URL = "https://api.anthropic.com/api/oauth/usage"
GROK_USAGE_URL = "https://cli-chat-proxy.grok.com/v1/billing?format=credits"

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
    "xai": {
        ("GET", GROK_USAGE_URL),
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
    "opencode_go": "opencode",
    "xai": "xai",
    "grok": "xai",
    "grok_cli": "xai",
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


# ---------------------------------------------------------------------
# Tier grouping and pooled usage (mirrors the dashboard Status tab)
# ---------------------------------------------------------------------

_TIER_LABELS = {
    "free-tier": "Free",
    "g1-pro-tier": "Google AI Pro",
    "g1-ultra-tier": "Google AI Ultra",
    "free": "Free",
    "plus": "Plus",
    "pro": "Pro",
    "team": "Team",
    "business": "Business",
    "enterprise": "Enterprise",
    "standard": "Standard",
}
_PRO_TIER_RE = re.compile(r"pro|ultra|plus|team|business|enterprise", re.IGNORECASE)


def _account_tier(account: Account, quota: Optional[Dict[str, Any]]) -> Tuple[str, str, bool]:
    """Return ``(tier_key, tier_label, is_paid)`` for an account and its quota."""
    extra = account.extra_config if isinstance(account.extra_config, dict) else {}
    q = quota or {}
    provider_id = account.provider_id

    if provider_id == "antigravity":
        raw = str(q.get("tier") or extra.get("tier") or "free-tier")
        label = str(
            q.get("tier_name")
            or extra.get("tier_name")
            or _TIER_LABELS.get(raw)
            or raw
        )
        return raw.lower(), label, bool(_PRO_TIER_RE.search(raw))
    if provider_id == "codex":
        raw = str(q.get("plan_type") or extra.get("plan_type") or "standard")
        return raw.lower(), _TIER_LABELS.get(raw, raw), bool(_PRO_TIER_RE.search(raw))
    if provider_id == "openrouter":
        is_free = q.get("is_free_tier") is True
        return ("free" if is_free else "paid"), ("Free" if is_free else "Paid"), not is_free
    if provider_id == "opencode_go":
        return "go", "Go", True

    raw = str(extra.get("tier") or q.get("tier") or extra.get("plan_type") or q.get("plan_type") or "standard")
    return raw.lower(), _TIER_LABELS.get(raw, raw), False


def _quota_has_usage(quota: Optional[Dict[str, Any]]) -> bool:
    """True when a quota snapshot has something the dashboard can render."""
    if not isinstance(quota, dict) or quota.get("error"):
        return False
    if isinstance(quota.get("available_credits"), (int, float)):
        return True
    if isinstance(quota.get("balance"), dict):
        return True
    if isinstance(quota.get("summary"), list) and quota.get("summary"):
        return True
    groups = quota.get("groups")
    return isinstance(groups, list) and any(
        isinstance(group, dict) and group.get("buckets") for group in groups
    )


def _remaining_fraction(bucket: Dict[str, Any]) -> float:
    value = bucket.get("remainingFraction")
    if isinstance(value, (int, float)):
        return min(1.0, max(0.0, float(value)))
    used = bucket.get("usedPercent")
    if isinstance(used, (int, float)):
        return min(1.0, max(0.0, 1.0 - float(used) / 100.0))
    return 1.0


def _account_status(account: Account, quota: Optional[Dict[str, Any]]) -> str:
    if isinstance(quota, dict) and quota.get("error"):
        return "no_data"
    until = float(getattr(account, "rate_limited_until", 0) or 0)
    if until > time.time():
        return "cooldown"
    if not account.enabled:
        return "disabled"
    return "active"


async def _fetch_account_quota(account: Account) -> Optional[Dict[str, Any]]:
    """Resolve a normalized quota snapshot for any provider."""
    if account.disable_usage_tracking:
        return None

    # Imported lazily to avoid a module import cycle with the dashboard router.
    from llmproxy.dashboard import router as dashboard_router

    try:
        if account.provider_id == "antigravity":
            return await dashboard_router.fetch_antigravity_quota(account)
        if account.provider_id == "codex":
            return await dashboard_router.fetch_codex_quota(account)
        if account.provider_id == "openrouter":
            return await dashboard_router.fetch_openrouter_credits(account)
        if account.provider_id == "opencode_go":
            return await dashboard_router.fetch_opencode_go_quota(account)
    except Exception as exc:  # pragma: no cover - defensive
        return {"error": str(exc)}

    plugin = plugin_manager.get_plugin(account.provider_id)
    if plugin is None or not getattr(plugin.metadata, "supports_usage_meter", False):
        return None

    stored = stored_usage_limits(account)
    if stored is not None:
        return stored

    implementation = getattr(type(plugin), "fetch_usage_quota", None)
    if implementation is None or implementation is BasePlugin.fetch_usage_quota:
        return None

    cfg = plugin_manager.configs.get(account.provider_id) or {}
    variables = cfg.get("variables") if isinstance(cfg, dict) else {}
    try:
        quota = await plugin.fetch_usage_quota(
            account, variables if isinstance(variables, dict) else {}
        )
    except Exception as exc:  # pragma: no cover - defensive
        return {"error": str(exc)}
    return quota if isinstance(quota, dict) else None


def _provider_name(provider_id: str) -> str:
    plugin = plugin_manager.get_plugin(provider_id)
    return plugin.metadata.name if plugin is not None else provider_id


def _provider_pinned(provider_id: str) -> bool:
    plugin = plugin_manager.get_plugin(provider_id)
    cfg = plugin_manager.configs.get(provider_id) or {}
    pinned = cfg.get("pinned") if isinstance(cfg, dict) else None
    if pinned is None:
        pinned = getattr(getattr(plugin, "metadata", None), "pinned", False)
    return bool(pinned)


def _build_usage_pools(
    accounts: List[Account],
    quotas: Dict[str, Optional[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Group accounts by provider and subscription tier, aggregating windows."""
    pools: Dict[str, Dict[str, Any]] = {}

    for account in accounts:
        quota = quotas.get(account.id)
        if quota is None:
            continue

        tier_key, tier_label, paid = _account_tier(account, quota)
        key = f"{account.provider_id}::{tier_key}"
        pool = pools.get(key)
        if pool is None:
            pool = {
                "id": key,
                "provider": _provider_kind(account.provider_id) or account.provider_id,
                "provider_id": account.provider_id,
                "provider_name": _provider_name(account.provider_id),
                "tier": tier_key,
                "tier_label": tier_label,
                "plan": tier_label,
                "paid": paid,
                "pinned": _provider_pinned(account.provider_id),
                "accounts": [],
                "renderable": False,
                "groups": {},
                "credits": None,
                "balances": {},
                "summaries": {},
            }
            pools[key] = pool

        pool["accounts"].append(account)
        if _quota_has_usage(quota):
            pool["renderable"] = True
        if isinstance(quota, dict) and isinstance(quota.get("available_credits"), (int, float)):
            pool["credits"] = (pool["credits"] or 0.0) + float(quota["available_credits"])
        balance = quota.get("balance") if isinstance(quota, dict) else None
        if isinstance(balance, dict) and isinstance(balance.get("value"), (int, float)):
            label = str(balance.get("label") or "available balance")
            pool["balances"][label] = pool["balances"].get(label, 0.0) + float(balance["value"])
        summary = quota.get("summary") if isinstance(quota, dict) else None
        if isinstance(summary, list):
            for item in summary:
                if not isinstance(item, dict) or not isinstance(item.get("value"), (int, float)):
                    continue
                label = str(item.get("label") or "")
                if label:
                    pool["summaries"][label] = pool["summaries"].get(label, 0.0) + float(item["value"])

        for group in (quota.get("groups") or []) if isinstance(quota, dict) else []:
            if not isinstance(group, dict) or not isinstance(group.get("buckets"), list):
                continue
            group_name = str(group.get("displayName") or "Usage")
            buckets = pool["groups"].setdefault(group_name, {})
            for bucket in group["buckets"]:
                if not isinstance(bucket, dict):
                    continue
                bucket_id = str(bucket.get("bucketId") or bucket.get("displayName") or "limit")
                entry = buckets.setdefault(
                    bucket_id,
                    {
                        "bucketId": bucket_id,
                        "displayName": str(bucket.get("displayName") or bucket_id),
                        "fractions": [],
                        "resets": [],
                        "count": 0,
                    },
                )
                entry["fractions"].append(_remaining_fraction(bucket))
                if bucket.get("resetTime"):
                    entry["resets"].append(str(bucket["resetTime"]))
                entry["count"] += 1

    result: List[Dict[str, Any]] = []
    for pool in pools.values():
        if not pool["renderable"]:
            continue

        groups: List[Dict[str, Any]] = []
        windows: List[Dict[str, Any]] = []
        all_fractions: List[float] = []
        for group_name, buckets in pool["groups"].items():
            rendered: List[Dict[str, Any]] = []
            for entry in buckets.values():
                fractions = entry["fractions"]
                if not fractions:
                    continue
                remaining = sum(fractions) / len(fractions)
                all_fractions.append(remaining)
                resets = sorted(
                    (value for value in entry["resets"] if value),
                )
                bucket_payload = {
                    "bucketId": entry["bucketId"],
                    "displayName": entry["displayName"],
                    "usedPercent": round((1.0 - remaining) * 100.0, 2),
                    "remainingFraction": round(remaining, 4),
                    "resetTime": resets[0] if resets else None,
                    "accountCount": entry["count"],
                }
                rendered.append(bucket_payload)
                windows.append(
                    {
                        "id": entry["bucketId"],
                        "label": entry["displayName"],
                        "group": group_name,
                        **{key: bucket_payload[key] for key in ("usedPercent", "remainingFraction", "resetTime")},
                        "accounts": entry["count"],
                    }
                )
            if rendered:
                groups.append({"displayName": group_name, "buckets": rendered})

        pool_accounts = [
            {
                "id": account.id,
                "auth_index": account.id,
                "name": account.name,
                "email": _account_email(account),
                "enabled": account.enabled,
                "disabled": not account.enabled,
                "status": _account_status(account, quotas.get(account.id)),
                "error": (quotas.get(account.id) or {}).get("error")
                if isinstance(quotas.get(account.id), dict)
                else None,
            }
            for account in pool["accounts"]
        ]

        remaining_fraction = (
            sum(all_fractions) / len(all_fractions) if all_fractions else None
        )
        result.append(
            {
                "id": pool["id"],
                "provider": pool["provider"],
                "provider_id": pool["provider_id"],
                "provider_name": pool["provider_name"],
                "tier": pool["tier"],
                "tier_label": pool["tier_label"],
                "plan": pool["plan"],
                "paid": pool["paid"],
                "pinned": pool["pinned"],
                "account_count": len(pool_accounts),
                "active_accounts": sum(1 for item in pool_accounts if item["status"] == "active"),
                "cooldown_accounts": sum(1 for item in pool_accounts if item["status"] == "cooldown"),
                "error_accounts": sum(1 for item in pool_accounts if item["status"] == "no_data"),
                "remaining_fraction": round(remaining_fraction, 4)
                if remaining_fraction is not None
                else None,
                "used_percent": round((1.0 - remaining_fraction) * 100.0, 2)
                if remaining_fraction is not None
                else None,
                "credits": round(pool["credits"], 4) if pool["credits"] is not None else None,
                "balances": [
                    {"label": label, "value": round(value, 4)}
                    for label, value in pool["balances"].items()
                ],
                "summaries": [
                    {"label": label, "value": round(value, 4)}
                    for label, value in pool["summaries"].items()
                ],
                "groups": groups,
                "windows": windows,
                "accounts": pool_accounts,
            }
        )

    result.sort(
        key=lambda item: (
            not item["pinned"],
            not item["paid"],
            item["provider_name"].lower(),
            item["tier_label"].lower(),
        )
    )
    return result


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
    elif provider == "opencode":
        if not any(name.lower() == "user-agent" for name in headers):
            headers["User-Agent"] = "llmproxy/1.0"

    return headers


async def _refresh_account_if_needed(account: Account, provider: str, force: bool = False) -> None:
    if not account.extra_config.get("refresh_token"):
        return
    expires_at = float(account.extra_config.get("expires_at") or 0)
    if not force and account.api_key and (expires_at <= 0 or expires_at - time.time() >= 120):
        return

    plugin = plugin_manager.plugins.get(account.provider_id)
    if plugin is None:
        return
    try:
        refreshed = bool(await plugin.refresh_account(account))
    except Exception:
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

    # CLIProxyAPI clients (including T3 Code) send ``data`` as a JSON-encoded
    # string. Forwarding it verbatim would post a quoted JSON string upstream,
    # so decode it back into the object the provider expects.
    body_data: Any = call.data
    if isinstance(body_data, str):
        raw = body_data.strip()
        if raw:
            try:
                body_data = json.loads(raw)
            except (TypeError, ValueError):
                body_data = raw

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
            return await client.post(url, headers=headers, json=body_data)

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
        if not _provider_enabled(account.provider_id):
            continue

        # Known CLIProxyAPI provider kinds keep their canonical name so T3 Code
        # can probe them. Other LLMProxy providers are still listed with their
        # own id, plus tier/pool metadata, so clients that understand them can
        # group and render their usage too.
        provider = _provider_kind(account.provider_id) or account.provider_id
        quota = stored_usage_limits(account)
        tier_key, tier_label, _paid = _account_tier(account, quota)

        entry: Dict[str, Any] = {
            "id": account.id,
            "auth_index": account.id,
            "name": account.name,
            "provider": provider,
            "provider_id": account.provider_id,
            "disabled": not account.enabled,
            "tier": tier_key,
            "tier_label": tier_label,
            "pool_id": f"{account.provider_id}::{tier_key}",
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


@router.get("/auth-files/models")
async def list_auth_file_models(request: Request, name: str = ""):
    """Return the model IDs routed through one account.

    T3 Code probes ``auth-files/models?name=<account id>`` after listing auth
    files, so the response must be ``{"models": [{"id": ...}]}``.
    """
    _authorize(request)

    account = _find_account(name)
    if account is None and name:
        account = next(
            (
                candidate
                for candidate in plugin_manager.get_accounts()
                if candidate.name == name or _account_email(candidate) == name
            ),
            None,
        )
    if account is None:
        raise HTTPException(status_code=404, detail={"error": "unknown_account"})

    seen: set[str] = set()
    models: List[Dict[str, str]] = []
    provider_models = plugin_manager.get_plugin_models(account.provider_id)
    for model in provider_models:
        if not getattr(model, "enabled", True) or model.id in seen:
            continue
        seen.add(model.id)
        models.append({"id": model.id, "name": model.name})

    # Virtual load-balancing rules that route to this provider's models.
    target_ids = {model.id for model in provider_models}
    for rule in plugin_manager.get_load_balancing_rules():
        if not rule.enabled or rule.model_id in seen:
            continue
        if any(target in target_ids for target in rule.targets):
            seen.add(rule.model_id)
            models.append({"id": rule.model_id, "name": rule.name})

    return {"models": models}


@router.get("/usage")
async def grouped_usage(request: Request, provider: Optional[str] = None):
    """Return every provider's usage windows grouped by subscription tier.

    This mirrors the dashboard Status tab: accounts sharing a provider and plan
    collapse into one pool with combined remaining capacity, while a different
    tier (for example a Google free plan next to Google AI Pro) stays separate.
    """
    _authorize(request)

    accounts = [
        account
        for account in plugin_manager.get_accounts()
        if _provider_enabled(account.provider_id)
        and (provider is None or account.provider_id == provider)
    ]

    quotas_list = await asyncio.gather(
        *(_fetch_account_quota(account) for account in accounts),
        return_exceptions=True,
    )
    quotas: Dict[str, Optional[Dict[str, Any]]] = {}
    for account, quota in zip(accounts, quotas_list):
        if isinstance(quota, Exception):
            quotas[account.id] = {"error": str(quota)}
        else:
            quotas[account.id] = quota

    pools = _build_usage_pools(accounts, quotas)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "providers": sorted({pool["provider_id"] for pool in pools}),
        "pools": pools,
    }


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
