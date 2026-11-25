"""CLIProxyAPI usage hub tests: T3 Code contract + tier-grouped usage.

T3 Code reads ``GET /v0/management/auth-files``, probes each account through
``POST /v0/management/api-call`` (sending ``data`` as a JSON-encoded string),
and asks ``GET /v0/management/auth-files/models?name=`` for routed models.
The hub additionally exposes ``GET /v0/management/usage`` with the same pooled
provider/tier grouping the dashboard Status tab renders.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from llmproxy.core.manager import plugin_manager
from llmproxy.main import app
from llmproxy.schemas import Account

ADMIN_HEADERS = {"X-Admin-Token": "test-admin-token"}


@pytest.fixture
def hub(monkeypatch):
    """Isolated account/config view plus an authenticated hub client."""
    accounts = {}
    monkeypatch.setattr(plugin_manager, "accounts", accounts)
    monkeypatch.setattr(plugin_manager, "configs", {})
    client = TestClient(app, headers=ADMIN_HEADERS)
    key = client.get("/api/management").json()["management_key"]
    return client, accounts, {"Authorization": f"Bearer {key}"}


def _add(accounts, **kwargs):
    account = Account(**kwargs)
    accounts[account.id] = account
    return account


def test_auth_files_lists_all_providers_with_tier_and_pool(monkeypatch, hub):
    client, accounts, headers = hub
    monkeypatch.setattr(plugin_manager, "get_plugins", lambda: {}, raising=False)

    _add(
        accounts,
        id="acc_codex",
        provider_id="codex",
        name="Codex (a@example.com)",
        api_key="token",
        extra_config={
            "account_id": "chatgpt-1",
            "plan_type": "plus",
            "email": "a@example.com",
        },
    )
    _add(
        accounts,
        id="acc_ag_pro",
        provider_id="antigravity",
        name="Google (pro@example.com)",
        api_key="token",
        extra_config={
            "email": "pro@example.com",
            "project_id": "proj-1",
            "tier": "g1-pro-tier",
            "tier_name": "Google AI Pro",
        },
    )
    _add(
        accounts,
        id="acc_ag_free",
        provider_id="antigravity",
        name="Google (free@example.com)",
        api_key="token",
        extra_config={"email": "free@example.com", "tier": "free-tier"},
    )
    _add(
        accounts,
        id="acc_nano",
        provider_id="nanogpt",
        name="NanoGPT",
        api_key="token",
    )

    response = client.get("/v0/management/auth-files", headers=headers)
    assert response.status_code == 200
    files = {item["id"]: item for item in response.json()["files"]}

    # Every configured provider is listed, even when T3 has no built-in driver.
    assert {"acc_codex", "acc_ag_pro", "acc_ag_free", "acc_nano"} <= set(files)
    assert files["acc_nano"]["provider"] == "nanogpt"

    # Tier separation matches the Status tab (pro and free stay in own pools).
    assert files["acc_ag_pro"]["tier"] == "g1-pro-tier"
    assert files["acc_ag_pro"]["tier_label"] == "Google AI Pro"
    assert files["acc_ag_pro"]["pool_id"] == "antigravity::g1-pro-tier"
    assert files["acc_ag_free"]["pool_id"] == "antigravity::free-tier"
    assert files["acc_ag_pro"]["project_id"] == "proj-1"
    assert files["acc_ag_pro"]["email"] == "pro@example.com"

    # Codex keeps the id_token fields T3 Code builds its probe headers from.
    assert files["acc_codex"]["id_token"]["chatgpt_account_id"] == "chatgpt-1"
    assert files["acc_codex"]["id_token"]["plan_type"] == "plus"


def test_api_call_decodes_json_string_body_from_t3(monkeypatch, hub):
    client, accounts, headers = hub
    account = _add(
        accounts,
        id="acc_ag",
        provider_id="antigravity",
        name="Google (pro@example.com)",
        api_key="access-token",
        extra_config={"project_id": "proj-1", "expires_at": 10**12},
    )

    captured = {}

    class MockResponse:
        status_code = 200
        text = '{"groups":[]}'

    async def mock_post(self, url, headers=None, json=None):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return MockResponse()

    monkeypatch.setattr("httpx.AsyncClient.post", mock_post)

    response = client.post(
        "/v0/management/api-call",
        headers=headers,
        json={
            "auth_index": account.id,
            "method": "POST",
            "url": "https://daily-cloudcode-pa.googleapis.com/v1internal:retrieveUserQuotaSummary",
            "header": {
                "Authorization": "Bearer $TOKEN$",
                "Content-Type": "application/json",
            },
            # T3 Code encodes the body as a JSON string.
            "data": json.dumps({"project": "proj-1"}),
        },
    )

    assert response.status_code == 200
    assert response.json()["status_code"] == 200
    assert captured["json"] == {"project": "proj-1"}
    assert captured["headers"]["Authorization"] == "Bearer access-token"


def test_api_call_still_accepts_object_bodies(monkeypatch, hub):
    client, accounts, headers = hub
    account = _add(
        accounts,
        id="acc_ag2",
        provider_id="antigravity",
        name="Google two",
        api_key="access-token",
        extra_config={"expires_at": 10**12},
    )

    captured = {}

    class MockResponse:
        status_code = 200
        text = "{}"

    async def mock_post(self, url, headers=None, json=None):
        captured["json"] = json
        return MockResponse()

    monkeypatch.setattr("httpx.AsyncClient.post", mock_post)

    client.post(
        "/v0/management/api-call",
        headers=headers,
        json={
            "auth_index": account.id,
            "method": "POST",
            "url": "https://daily-cloudcode-pa.googleapis.com/v1internal:retrieveUserQuotaSummary",
            "data": {"project": "proj-2"},
        },
    )
    assert captured["json"] == {"project": "proj-2"}


def test_auth_file_models_returns_provider_and_rule_models(hub):
    client, accounts, headers = hub
    _add(
        accounts,
        id="acc_ag_models",
        provider_id="antigravity",
        name="Google models",
        api_key="token",
        extra_config={"tier": "g1-pro-tier"},
    )

    response = client.get(
        "/v0/management/auth-files/models",
        params={"name": "acc_ag_models"},
        headers=headers,
    )
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload["models"], list)
    assert payload["models"], "antigravity should expose enabled models"
    assert all(isinstance(item["id"], str) and item["id"] for item in payload["models"])


def test_auth_file_models_unknown_account_is_404(hub):
    client, _accounts, headers = hub
    response = client.get(
        "/v0/management/auth-files/models",
        params={"name": "missing"},
        headers=headers,
    )
    assert response.status_code == 404


def _quota(remaining: float, tier: str, tier_name: str) -> dict:
    return {
        "tier": tier,
        "tier_name": tier_name,
        "groups": [
            {
                "displayName": "Gemini Models",
                "buckets": [
                    {
                        "bucketId": "gemini-weekly",
                        "displayName": "Weekly Limit Remaining",
                        "remainingFraction": remaining,
                        "resetTime": "2026-12-01T00:00:00Z",
                    }
                ],
            }
        ],
    }


def test_grouped_usage_pools_by_provider_and_tier(monkeypatch, hub):
    client, accounts, headers = hub
    for index, tier in enumerate(("g1-pro-tier", "g1-pro-tier", "free-tier")):
        _add(
            accounts,
            id=f"acc_ag_{index}",
            provider_id="antigravity",
            name=f"Google {index}",
            api_key="token",
            extra_config={"tier": tier, "tier_name": "Google AI Pro" if tier == "g1-pro-tier" else "Free"},
        )

    quotas = {
        "acc_ag_0": _quota(0.5, "g1-pro-tier", "Google AI Pro"),
        "acc_ag_1": _quota(1.0, "g1-pro-tier", "Google AI Pro"),
        "acc_ag_2": _quota(0.2, "free-tier", "Free"),
    }

    async def fake_quota(account):
        return quotas[account.id]

    monkeypatch.setattr("llmproxy.api.management._fetch_account_quota", fake_quota)

    response = client.get("/v0/management/usage", headers=headers)
    assert response.status_code == 200
    pools = response.json()["pools"]
    assert len(pools) == 2

    pro = next(pool for pool in pools if pool["tier"] == "g1-pro-tier")
    free = next(pool for pool in pools if pool["tier"] == "free-tier")

    assert pro["account_count"] == 2
    assert pro["provider_id"] == "antigravity"
    assert pro["remaining_fraction"] == pytest.approx(0.75)
    assert pro["used_percent"] == pytest.approx(25.0)
    bucket = pro["groups"][0]["buckets"][0]
    assert bucket["remainingFraction"] == pytest.approx(0.75)
    assert bucket["accountCount"] == 2
    assert [item["id"] for item in pro["accounts"]] == ["acc_ag_0", "acc_ag_1"]

    assert free["account_count"] == 1
    assert free["remaining_fraction"] == pytest.approx(0.2)


def test_grouped_usage_includes_error_accounts_in_roster(monkeypatch, hub):
    client, accounts, headers = hub
    for index in range(2):
        _add(
            accounts,
            id=f"acc_ag_{index}",
            provider_id="antigravity",
            name=f"Google {index}",
            api_key="token",
            extra_config={"tier": "g1-pro-tier"},
        )

    async def fake_quota(account):
        if account.id == "acc_ag_1":
            return {"error": "Upstream error (401) from quota"}
        return _quota(0.4, "g1-pro-tier", "Google AI Pro")

    monkeypatch.setattr("llmproxy.api.management._fetch_account_quota", fake_quota)

    response = client.get("/v0/management/usage", headers=headers)
    pool = next(pool for pool in response.json()["pools"] if pool["tier"] == "g1-pro-tier")
    assert pool["account_count"] == 2
    assert pool["error_accounts"] == 1
    statuses = {item["id"]: item["status"] for item in pool["accounts"]}
    assert statuses["acc_ag_1"] == "no_data"
    # Only the reporting account contributes to the combined window.
    assert pool["remaining_fraction"] == pytest.approx(0.4)


def test_grouped_usage_provider_filter(monkeypatch, hub):
    client, accounts, headers = hub
    _add(accounts, id="acc_ag_filter", provider_id="antigravity", name="Google", api_key="t",
         extra_config={"tier": "free-tier"})
    _add(accounts, id="acc_cx_filter", provider_id="codex", name="Codex", api_key="t",
         extra_config={"plan_type": "free"})

    async def fake_quota(account):
        return {
            "plan_type": "free",
            "tier": "free-tier",
            "groups": [
                {
                    "displayName": "Usage",
                    "buckets": [
                        {"bucketId": "w", "displayName": "Weekly", "remainingFraction": 1.0}
                    ],
                }
            ],
        }

    monkeypatch.setattr("llmproxy.api.management._fetch_account_quota", fake_quota)

    response = client.get("/v0/management/usage", params={"provider": "antigravity"}, headers=headers)
    providers = {pool["provider_id"] for pool in response.json()["pools"]}
    assert providers == {"antigravity"}
