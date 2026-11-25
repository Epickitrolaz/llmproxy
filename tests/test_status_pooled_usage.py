"""Tests for the pooled usage/tier view backing the redesigned Status tab.

Accounts are grouped by provider *and* subscription tier in the dashboard, so the
backend must report the real plan for each account instead of the stale
``currentTier`` value that Antigravity keeps on every login.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

from llmproxy.schemas import Account

TEMPLATES = Path(__file__).resolve().parent.parent / "templates"


def _load_antigravity_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "test_antigravity_template", TEMPLATES / "antigravity.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _antigravity_account(**extra):
    config = {
        "refresh_token": "refresh-token",
        "project_id": "test-project",
        "email": "user@example.com",
        "tier": "free-tier",
        "expires_at": 9999999999,
    }
    config.update(extra)
    return Account(
        id="acc_google_test",
        provider_id="antigravity",
        name="Google (user@example.com)",
        api_key="access-token",
        base_url="https://daily-cloudcode-pa.googleapis.com",
        extra_config=config,
    )


def _install_httpx_post(monkeypatch, handler):
    calls: list[str] = []

    async def fake_post(self, url, headers=None, json=None):
        calls.append(url)
        return handler(url)

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    return calls


def test_antigravity_quota_uses_paid_tier_when_current_tier_is_free(monkeypatch):
    def handler(url: str):
        if url.endswith("retrieveUserQuotaSummary"):
            return httpx.Response(
                200,
                json={"groups": [{"displayName": "Gemini Models", "buckets": []}]},
            )
        if url.endswith("loadCodeAssist"):
            return httpx.Response(
                200,
                json={
                    "currentTier": {"id": "free-tier", "name": "Antigravity"},
                    "paidTier": {"id": "g1-pro-tier", "name": "Google AI Pro"},
                },
            )
        raise AssertionError(f"unexpected URL {url}")

    calls = _install_httpx_post(monkeypatch, handler)
    from llmproxy.dashboard.router import fetch_antigravity_quota

    account = _antigravity_account()
    result = asyncio.run(fetch_antigravity_quota(account))

    assert result["tier"] == "g1-pro-tier"
    assert result["tier_name"] == "Google AI Pro"
    assert account.extra_config["tier"] == "g1-pro-tier"
    assert account.extra_config["tier_name"] == "Google AI Pro"
    assert account.extra_config["tier_checked_at"] > 0
    assert any(url.endswith("loadCodeAssist") for url in calls)


def test_antigravity_quota_keeps_free_tier_when_no_paid_tier(monkeypatch):
    def handler(url: str):
        if url.endswith("retrieveUserQuotaSummary"):
            return httpx.Response(200, json={"groups": []})
        if url.endswith("loadCodeAssist"):
            return httpx.Response(
                200, json={"currentTier": {"id": "free-tier", "name": "Antigravity"}}
            )
        raise AssertionError(f"unexpected URL {url}")

    _install_httpx_post(monkeypatch, handler)
    from llmproxy.dashboard.router import fetch_antigravity_quota

    account = _antigravity_account(tier="", tier_name="")
    result = asyncio.run(fetch_antigravity_quota(account))

    assert result["tier"] == "free-tier"
    assert account.extra_config["tier"] == "free-tier"


def test_antigravity_quota_reuses_cached_tier(monkeypatch):
    import time

    def handler(url: str):
        if url.endswith("retrieveUserQuotaSummary"):
            return httpx.Response(200, json={"groups": []})
        raise AssertionError("loadCodeAssist should not be called for a cached tier")

    calls = _install_httpx_post(monkeypatch, handler)
    from llmproxy.dashboard.router import fetch_antigravity_quota

    account = _antigravity_account(
        tier="g1-pro-tier",
        tier_name="Google AI Pro",
        tier_checked_at=time.time(),
    )
    result = asyncio.run(fetch_antigravity_quota(account))

    assert result["tier"] == "g1-pro-tier"
    assert not any(url.endswith("loadCodeAssist") for url in calls)


def test_antigravity_status_exposes_tier_and_tier_name(monkeypatch):
    from llmproxy.dashboard import router as dashboard_router
    from llmproxy.core.manager import plugin_manager

    account = _antigravity_account(
        tier="g1-pro-tier",
        tier_name="Google AI Pro",
        tier_checked_at=10**10,
    )

    async def fake_quota(acc):
        return {"account_id": acc.id, "tier": acc.extra_config.get("tier"), "groups": []}

    monkeypatch.setattr(dashboard_router, "fetch_antigravity_quota", fake_quota)
    monkeypatch.setattr(
        plugin_manager, "get_accounts", lambda: [account]
    )

    payload = asyncio.run(dashboard_router.get_antigravity_all_status())
    item = payload["accounts"][0]
    assert item["account_id"] == "acc_google_test"
    assert item["tier"] == "g1-pro-tier"
    assert item["tier_name"] == "Google AI Pro"


def test_provider_usage_status_payload_includes_provider_id(monkeypatch):
    from llmproxy.core.manager import plugin_manager
    from llmproxy.dashboard import router as dashboard_router

    account = Account(
        id="acc_groq",
        provider_id="groqcloud",
        name="Groq",
        api_key="live-key",
        extra_config={
            "usage_limits": {
                "checked_at": 10**10,
                "source": "response-headers",
                "groups": [
                    {
                        "displayName": "Rate limits",
                        "buckets": [
                            {
                                "bucketId": "requests",
                                "displayName": "Requests",
                                "usedPercent": 5.0,
                                "remainingFraction": 0.95,
                            }
                        ],
                    }
                ],
            }
        },
    )

    class _Plugin:
        class metadata:  # noqa: N801 - mirror BasePlugin metadata attribute
            supports_usage_meter = True

    monkeypatch.setattr(plugin_manager, "get_accounts", lambda: [account])
    monkeypatch.setattr(plugin_manager, "get_plugin", lambda pid: _Plugin())

    payload = asyncio.run(dashboard_router.get_provider_usage_status())
    assert payload["accounts"][0]["provider_id"] == "groqcloud"
    assert payload["accounts"][0]["quota"]["groups"][0]["buckets"][0]["usedPercent"] == 5.0


# =========================================================================
# Dashboard wiring: tier-grouped pools + graph view
# =========================================================================

ROOT = Path(__file__).resolve().parent.parent


def _frontend_files():
    app_js = (ROOT / "llmproxy" / "static" / "app.js").read_text(encoding="utf-8")
    index_html = (ROOT / "llmproxy" / "static" / "index.html").read_text(encoding="utf-8")
    style_css = (ROOT / "llmproxy" / "static" / "style.css").read_text(encoding="utf-8")
    return app_js, index_html, style_css


def test_status_view_groups_pools_by_provider_and_tier():
    app_js, _, _ = _frontend_files()

    assert "function buildUsagePools(" in app_js
    assert "function resolveAccountTier(" in app_js
    # One pool per provider+tier combination, never per account.
    assert "const key = `${acc.provider_id}::${tier.key}`" in app_js
    # Tier labels separate plans (e.g. Google AI Pro vs Free).
    assert '"g1-pro-tier": "Google AI Pro"' in app_js
    assert 'isFree ? "Free" : "Paid"' in app_js
    # Combined remaining capacity is the mean across pooled accounts.
    assert "fractions.reduce((sum, value) => sum + value, 0) / entry.fractions.length" in app_js


def test_status_tab_has_graph_view_with_provider_filtering():
    app_js, index_html, style_css = _frontend_files()

    assert 'id="status-view-overview"' in index_html
    assert 'id="status-view-graph"' in index_html
    assert 'id="btn-status-view-graph"' in index_html
    assert 'id="graph-provider-filters"' in index_html
    assert 'id="graph-metric-tabs"' in index_html
    assert 'id="graph-range-tabs"' in index_html
    assert 'id="graph-timeseries-container"' in index_html

    assert "function setStatusView(" in app_js
    assert "function renderGraphTimeseries(" in app_js
    assert "function renderGraphProviderFilters(" in app_js
    assert "state.graphProviders" in app_js
    assert ".status-view-switcher" in style_css
    assert ".graph-svg" in style_css
    assert ".pool-tier-badge" in style_css
