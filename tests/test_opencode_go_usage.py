"""Tests for OpenCode Go live usage tracking."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from llmproxy.core.manager import plugin_manager
from llmproxy.dashboard.router import fetch_opencode_go_quota
from llmproxy.main import app
from llmproxy.schemas import Account

ADMIN_HEADERS = {"X-Admin-Token": "test-admin-token"}


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> dict:
        if self._payload is None:
            raise ValueError("no json payload")
        return self._payload


class _FakeAsyncClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url: str, headers: dict | None = None):
        return _FakeResponse(
            200,
            {
                "usage": {
                    "rolling": {"status": "ok", "percent": 3, "resetsAt": "2026-11-17T11:34:03.000Z"},
                    "weekly": {"status": "ok", "percent": 13, "resetsAt": "2026-11-23T00:00:00.000Z"},
                    "monthly": {"status": "ok", "percent": 41, "resetsAt": "2026-12-12T06:44:01.000Z"},
                }
            },
        )


@pytest.fixture
def opencode_go_account():
    return Account(
        provider_id="opencode_go",
        name="OpenCode Go",
        api_key="go-test-key",
        base_url="https://opencode.ai/inference/go/openai/v1",
        extra_config={"email": "test@example.com"},
    )


def test_fetch_opencode_go_quota_normalizes_usage_windows(monkeypatch, opencode_go_account):
    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)

    result = asyncio.run(fetch_opencode_go_quota(opencode_go_account))

    assert "error" not in result
    buckets = result["groups"][0]["buckets"]
    assert [bucket["bucketId"] for bucket in buckets] == [
        "opencode_go_rolling",
        "opencode_go_weekly",
        "opencode_go_monthly",
    ]
    assert buckets[0]["displayName"] == "Rolling Limit"
    assert buckets[0]["usedPercent"] == 3
    assert buckets[0]["remainingFraction"] == pytest.approx(0.97)
    assert buckets[0]["resetTime"] == "2026-11-17T11:34:03.000Z"
    assert result["email"] == "test@example.com"


def test_fetch_opencode_go_quota_reports_missing_key(monkeypatch, opencode_go_account):
    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    opencode_go_account.api_key = ""

    result = asyncio.run(fetch_opencode_go_quota(opencode_go_account))

    assert "error" in result
    assert "API key" in result["error"]


def test_fetch_opencode_go_quota_reports_upstream_error(monkeypatch, opencode_go_account):
    import httpx

    class ErrorClient(_FakeAsyncClient):
        async def get(self, url: str, headers: dict | None = None):
            return _FakeResponse(403, text='{"error":"no Go subscription"}')

    monkeypatch.setattr(httpx, "AsyncClient", ErrorClient)

    result = asyncio.run(fetch_opencode_go_quota(opencode_go_account))

    assert "error" in result
    assert "403" in result["error"]


def test_opencode_go_status_endpoint(monkeypatch, opencode_go_account):
    monkeypatch.setattr(plugin_manager, "accounts", {opencode_go_account.id: opencode_go_account})
    monkeypatch.setattr(plugin_manager, "configs", {})

    async def fake_quota(account: Account):
        return {
            "account_id": account.id,
            "account_name": account.name,
            "groups": [
                {
                    "displayName": "OpenCode Go Usage",
                    "buckets": [
                        {
                            "bucketId": "opencode_go_weekly",
                            "displayName": "Weekly Limit",
                            "usedPercent": 13,
                            "remainingFraction": 0.87,
                            "resetTime": "2026-11-23T00:00:00.000Z",
                        }
                    ],
                }
            ],
        }

    monkeypatch.setattr(
        "llmproxy.dashboard.router.fetch_opencode_go_quota",
        fake_quota,
    )

    client = TestClient(app, headers=ADMIN_HEADERS)
    response = client.get("/api/opencode-go/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["accounts"][0]["account_id"] == opencode_go_account.id
    assert payload["accounts"][0]["quota"]["groups"][0]["buckets"][0]["usedPercent"] == 13


def test_management_hub_exposes_and_queries_opencode_go(monkeypatch, opencode_go_account):
    monkeypatch.setattr(plugin_manager, "accounts", {opencode_go_account.id: opencode_go_account})
    monkeypatch.setattr(plugin_manager, "configs", {})

    client = TestClient(app, headers=ADMIN_HEADERS)
    key = client.get("/api/management").json()["management_key"]

    listed = client.get(
        "/v0/management/auth-files",
        headers={"Authorization": f"Bearer {key}"},
    )
    assert listed.status_code == 200
    entry = next(
        item for item in listed.json()["files"] if item["id"] == opencode_go_account.id
    )
    assert entry["provider"] == "opencode"

    captured = {}

    class MockResponse:
        status_code = 200
        text = '{"usage":{"rolling":{"percent":3}}}'

    async def mock_get(self, url, headers=None):
        captured["url"] = url
        captured["headers"] = headers
        return MockResponse()

    monkeypatch.setattr("httpx.AsyncClient.get", mock_get)

    called = client.post(
        "/v0/management/api-call",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "auth_index": opencode_go_account.id,
            "method": "GET",
            "url": "https://opencode.ai/zen/go/v1/usage",
        },
    )

    assert called.status_code == 200
    assert called.json()["status_code"] == 200
    assert captured["url"] == "https://opencode.ai/zen/go/v1/usage"
    assert captured["headers"]["Authorization"] == "Bearer go-test-key"
    assert captured["headers"]["User-Agent"] == "llmproxy/1.0"


def test_dashboard_fetches_opencode_go_status():
    root = Path(__file__).resolve().parent.parent
    js = (root / "llmproxy" / "static" / "app.js").read_text(encoding="utf-8")

    assert 'a.provider_id === "opencode_go"' in js
    assert "/api/opencode-go/status" in js
