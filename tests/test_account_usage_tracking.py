"""Tests for per-account usage tracking opt-out and OpenRouter credits."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from llmproxy.core.manager import plugin_manager
from llmproxy.dashboard.router import fetch_openrouter_credits
from llmproxy.schemas import Account, AccountCreate


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        return self._payload


class _FakeAsyncClient:
    """Minimal httpx.AsyncClient stand-in for OpenRouter calls."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url: str, headers: dict | None = None):
        if url.endswith("/key"):
            return _FakeResponse(
                200,
                {
                    "data": {
                        "label": "test-key",
                        "limit": 20.0,
                        "limit_remaining": 12.5,
                        "usage": 7.5,
                        "is_free_tier": False,
                    }
                },
            )
        if url.endswith("/credits"):
            return _FakeResponse(
                200,
                {"data": {"total_credits": 20.0, "total_usage": 7.5}},
            )
        return _FakeResponse(404, {})


@pytest.fixture
def isolated_accounts(tmp_path, monkeypatch):
    monkeypatch.setattr(plugin_manager, "accounts_file", tmp_path / "accounts.json")
    original = dict(plugin_manager.accounts)
    plugin_manager.accounts.clear()
    yield
    plugin_manager.accounts.clear()
    plugin_manager.accounts.update(original)


def test_account_schema_round_trips_disable_usage_tracking():
    account = Account(
        provider_id="openai_compatible",
        name="Local vLLM",
        api_key="key",
        disable_usage_tracking=True,
    )
    created = AccountCreate(
        provider_id="anthropic_proxy",
        name="Self-hosted Claude",
        api_key="key",
        disable_usage_tracking=True,
    )
    assert account.disable_usage_tracking is True
    assert created.disable_usage_tracking is True
    assert Account.model_validate(account.model_dump()).disable_usage_tracking is True


def test_disabled_account_skips_counters_and_cooldown(isolated_accounts):
    account = plugin_manager.add_account(
        AccountCreate(
            provider_id="openai_compatible",
            name="Unlimited endpoint",
            api_key="key",
            disable_usage_tracking=True,
        )
    )

    plugin_manager.mark_account_result(account.id, success=True)
    plugin_manager.mark_account_result(
        account.id, success=False, is_rate_limit=True, cooldown_seconds=60
    )

    assert account.success_count == 0
    assert account.error_count == 0
    assert account.rate_limited_until is None
    assert account.is_available() is True


def test_disabling_tracking_clears_existing_cooldown(isolated_accounts):
    account = plugin_manager.add_account(
        AccountCreate(
            provider_id="anthropic_proxy",
            name="Metered Claude",
            api_key="key",
        )
    )
    plugin_manager.mark_account_result(
        account.id, success=False, is_rate_limit=True, cooldown_seconds=60
    )
    assert account.rate_limited_until is not None

    from llmproxy.schemas import AccountUpdate

    plugin_manager.update_account(account.id, AccountUpdate(disable_usage_tracking=True))

    assert account.disable_usage_tracking is True
    assert account.rate_limited_until is None
    assert account.is_available() is True


def test_tracking_enabled_account_still_records(isolated_accounts):
    account = plugin_manager.add_account(
        AccountCreate(
            provider_id="openai_compatible",
            name="Metered endpoint",
            api_key="key",
        )
    )

    plugin_manager.mark_account_result(account.id, success=True)
    plugin_manager.mark_account_result(
        account.id, success=False, is_rate_limit=True, cooldown_seconds=60
    )

    assert account.success_count == 1
    assert account.error_count == 1
    assert account.rate_limited_until is not None


def test_openrouter_credits_prefers_key_limit(monkeypatch):
    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    account = Account(
        provider_id="openrouter",
        name="OpenRouter",
        api_key="sk-or-test",
    )

    result = asyncio.run(fetch_openrouter_credits(account))

    assert result["available_credits"] == pytest.approx(12.5)
    assert result["limit"] == pytest.approx(20.0)
    assert result["total_usage"] == pytest.approx(7.5)
    assert result["source"] == "key"


class _FakeCreditsOnlyClient(_FakeAsyncClient):
    async def get(self, url: str, headers: dict | None = None):
        if url.endswith("/key"):
            return _FakeResponse(
                200,
                {"data": {"limit": None, "limit_remaining": None, "usage": 7.5}},
            )
        if url.endswith("/credits"):
            return _FakeResponse(
                200,
                {"data": {"total_credits": 20.0, "total_usage": 7.5}},
            )
        return _FakeResponse(404, {})


def test_openrouter_credits_falls_back_to_credit_totals(monkeypatch):
    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", _FakeCreditsOnlyClient)
    account = Account(
        provider_id="openrouter",
        name="OpenRouter",
        api_key="sk-or-test",
    )

    result = asyncio.run(fetch_openrouter_credits(account))

    assert result["available_credits"] == pytest.approx(12.5)
    assert result["total_credits"] == pytest.approx(20.0)
    assert result["source"] == "credits"


def test_openrouter_credits_reports_missing_key(monkeypatch):
    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    monkeypatch.setattr(plugin_manager, "configs", {})
    account = Account(provider_id="openrouter", name="No Key", api_key="")

    result = asyncio.run(fetch_openrouter_credits(account))

    assert "error" in result


def test_dashboard_exposes_toggle_and_credit_rendering():
    root = Path(__file__).resolve().parent.parent
    html = (root / "llmproxy" / "static" / "index.html").read_text(encoding="utf-8")
    js = (root / "llmproxy" / "static" / "app.js").read_text(encoding="utf-8")

    assert "account-usage-tracking-toggle" in html
    assert "account-usage-tracking-group" in html
    assert "disable_usage_tracking" in js
    assert "credits-panel" in js
    assert "/api/openrouter/status" in js
