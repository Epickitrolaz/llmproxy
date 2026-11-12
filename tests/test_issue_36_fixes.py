"""Tests for GitHub issue #36 fixes."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from llmproxy.core.manager import plugin_manager
from llmproxy.main import app
from llmproxy.schemas import (
    AccountCreate,
    ChatChoice,
    ChatChoiceDelta,
    ChatCompletionChunk,
    ChatCompletionResponse,
    ChatMessage,
    CompletionUsage,
)

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_state():
    orig_accounts = dict(plugin_manager.accounts)
    orig_keys = dict(plugin_manager.api_keys)
    orig_configs = json.loads(json.dumps(plugin_manager.configs))
    plugin_manager.accounts.clear()
    plugin_manager.api_keys.clear()
    yield
    plugin_manager.accounts = orig_accounts
    plugin_manager.api_keys = orig_keys
    plugin_manager.configs = orig_configs
    plugin_manager._save_accounts()


def _setup_mock_accounts(count: int = 3):
    for account in list(plugin_manager.get_accounts("mock_provider")):
        plugin_manager.delete_account(account.id)

    accounts = [
        plugin_manager.add_account(
            AccountCreate(
                provider_id="mock_provider",
                name=f"Mock Account {index}",
                api_key=f"key-{index}",
                priority=index,
            )
        )
        for index in range(1, count + 1)
    ]
    plugin_manager.update_plugin_config(
        "mock_provider",
        enabled=True,
        variables={
            "load_balancing_strategy": "fallback",
            "account_switch_cooldown": 60,
            "account_switch_wait": 0,
        },
    )
    return plugin_manager.get_plugin("mock_provider"), accounts


def _mock_response(model: str) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id="chatcmpl-test",
        model=model,
        choices=[ChatChoice(index=0, message=ChatMessage(role="assistant", content="OK"), finish_reason="stop")],
        usage=CompletionUsage(prompt_tokens=2, completion_tokens=1, total_tokens=3),
    )


# =========================================================================
# Account failover
# =========================================================================


def test_non_streaming_failover_tries_every_account(monkeypatch):
    plugin, accounts = _setup_mock_accounts(3)
    attempted = []

    async def fake_chat(request, model_config, variables, account=None):
        attempted.append(account.id)
        if account.id != accounts[-1].id:
            raise RuntimeError("HTTP 429: rate limit / quota exceeded")
        return _mock_response(request.model)

    monkeypatch.setattr(plugin, "chat_completion", fake_chat)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "mock-gpt-4o", "messages": [{"role": "user", "content": "ping"}]},
    )

    assert response.status_code == 200
    assert attempted == [account.id for account in accounts]
    assert accounts[-1].success_count >= 1


def test_non_streaming_all_accounts_limited_returns_usage_limit(monkeypatch):
    plugin, _accounts = _setup_mock_accounts(2)

    async def fake_chat(request, model_config, variables, account=None):
        raise RuntimeError("HTTP 429: rate limit / quota exceeded")

    monkeypatch.setattr(plugin, "chat_completion", fake_chat)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "mock-gpt-4o", "messages": [{"role": "user", "content": "ping"}]},
    )

    assert response.status_code == 429
    assert response.json()["detail"]["error"]["code"] == "usage_limit_reached"


def test_streaming_failover_before_first_content(monkeypatch):
    plugin, accounts = _setup_mock_accounts(3)
    attempted = []

    async def fake_chat(request, model_config, variables, account=None):
        attempted.append(account.id)

        if account.id != accounts[-1].id:
            async def failing_stream():
                yield ChatCompletionChunk(
                    id="chunk-fail",
                    model=request.model,
                    choices=[ChatChoiceDelta(index=0, delta={"role": "assistant", "content": ""})],
                )
                raise RuntimeError("HTTP 429: rate limit / quota exceeded")

            return failing_stream()

        async def working_stream():
            yield ChatCompletionChunk(
                id="chunk-ok",
                model=request.model,
                choices=[ChatChoiceDelta(index=0, delta={"role": "assistant", "content": ""})],
            )
            yield ChatCompletionChunk(
                id="chunk-ok",
                model=request.model,
                choices=[ChatChoiceDelta(index=0, delta={"content": "OK"})],
            )
            yield ChatCompletionChunk(
                id="chunk-ok",
                model=request.model,
                choices=[ChatChoiceDelta(index=0, delta={}, finish_reason="stop")],
            )

        return working_stream()

    monkeypatch.setattr(plugin, "chat_completion", fake_chat)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-gpt-4o",
            "messages": [{"role": "user", "content": "ping"}],
            "stream": True,
        },
    )

    assert response.status_code == 200
    assert attempted == [account.id for account in accounts]
    assert '"content":"OK"' in response.text
    assert '"error"' not in response.text


def test_streaming_include_usage_emits_terminal_usage_chunk(monkeypatch):
    plugin, _accounts = _setup_mock_accounts(1)

    async def fake_chat(request, model_config, variables, account=None):
        async def working_stream():
            yield ChatCompletionChunk(
                id="chunk-usage",
                model=request.model,
                choices=[ChatChoiceDelta(index=0, delta={"content": "Hello"})],
            )
            yield ChatCompletionChunk(
                id="chunk-usage",
                model=request.model,
                choices=[ChatChoiceDelta(index=0, delta={}, finish_reason="stop")],
            )

        return working_stream()

    monkeypatch.setattr(plugin, "chat_completion", fake_chat)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-gpt-4o",
            "messages": [{"role": "user", "content": "ping"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )

    assert response.status_code == 200
    assert '"choices":[]' in response.text
    assert '"usage":{"prompt_tokens":' in response.text


# =========================================================================
# CLIProxyAPI-compatible usage hub
# =========================================================================


def test_management_api_lists_accounts_and_replaces_token(monkeypatch):
    key = client.get("/api/management").json()["management_key"]
    account = plugin_manager.add_account(
        AccountCreate(
            provider_id="codex",
            name="Codex (test@example.com)",
            api_key="account-token",
            extra_config={"account_id": "chatgpt-account", "plan_type": "plus"},
        )
    )

    response = client.get(
        "/v0/management/auth-files",
        headers={"Authorization": f"Bearer {key}"},
    )
    assert response.status_code == 200
    entry = next(item for item in response.json()["files"] if item["id"] == account.id)
    assert entry["provider"] == "codex"
    assert entry["email"] == "test@example.com"
    assert entry["id_token"]["chatgpt_account_id"] == "chatgpt-account"

    captured = {}

    class MockResponse:
        status_code = 200
        text = '{"plan_type":"plus","rate_limit":{}}'

    async def mock_get(self, url, headers=None):
        captured["url"] = url
        captured["headers"] = headers
        return MockResponse()

    monkeypatch.setattr("httpx.AsyncClient.get", mock_get)

    response = client.post(
        "/v0/management/api-call",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "auth_index": account.id,
            "method": "GET",
            "url": "https://chatgpt.com/backend-api/wham/usage",
            "header": {"Authorization": "Bearer $TOKEN$", "ChatGPT-Account-Id": "chatgpt-account"},
        },
    )

    assert response.status_code == 200
    assert response.json()["status_code"] == 200
    assert captured["headers"]["Authorization"] == "Bearer account-token"
    assert captured["headers"]["ChatGPT-Account-Id"] == "chatgpt-account"


def test_management_api_rejects_unauthorized_and_unlisted_calls():
    key = client.get("/api/management").json()["management_key"]
    account = plugin_manager.add_account(
        AccountCreate(provider_id="codex", name="Codex test", api_key="account-token")
    )

    assert client.get("/v0/management/auth-files").status_code == 401
    assert client.get(
        "/v0/management/auth-files",
        headers={"Authorization": "Bearer wrong"},
    ).status_code == 401

    response = client.post(
        "/v0/management/api-call",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "auth_index": account.id,
            "method": "GET",
            "url": "http://127.0.0.1:22",
        },
    )
    assert response.status_code == 200
    assert response.json()["status_code"] == 403


def test_legacy_quota_scheduler_status_shape(monkeypatch):
    key = client.get("/api/management").json()["management_key"]
    account = plugin_manager.add_account(
        AccountCreate(
            provider_id="codex",
            name="Codex (test@example.com)",
            api_key="account-token",
            extra_config={"account_id": "chatgpt-account", "plan_type": "plus"},
        )
    )

    async def fake_quota(acc):
        return {
            "plan_type": "plus",
            "groups": [
                {
                    "displayName": "Codex Usage",
                    "buckets": [
                        {
                            "displayName": "5-Hour Limit",
                            "usedPercent": 25,
                            "resetTime": "2026-01-01T00:00:00+00:00",
                        }
                    ],
                }
            ],
        }

    monkeypatch.setattr("llmproxy.dashboard.router.fetch_codex_quota", fake_quota)

    response = client.get(
        "/v0/management/quota-scheduler/status",
        headers={"Authorization": f"Bearer {key}"},
    )
    assert response.status_code == 200
    data = response.json()["accounts"][account.id]
    assert data["provider"] == "codex"
    assert data["five_hour"]["used_percent"] == 25


# =========================================================================
# Template and status UI fixes
# =========================================================================


def test_codex_template_reasoning_options_are_none_to_max():
    code = Path("templates/codex.py").read_text(encoding="utf-8")
    assert 'options=["none", "low", "medium", "high", "xhigh", "max"]' in code


def test_status_ui_shows_remaining_quota_and_uses_3s_refresh():
    app_js = Path("llmproxy/static/app.js").read_text(encoding="utf-8")
    index_html = Path("llmproxy/static/index.html").read_text(encoding="utf-8")

    assert "b.remainingFraction * 100" in app_js
    assert "100 - b.usedPercent" in app_js
    assert "}, 3000);" in app_js
    assert "Auto-refresh (3s)" in index_html


def test_status_endpoints_fetch_accounts_concurrently(monkeypatch):
    import asyncio
    import time

    accounts = [
        plugin_manager.add_account(
            AccountCreate(
                provider_id="codex",
                name=f"Codex concurrent {index}",
                api_key="account-token",
                extra_config={"account_id": f"chatgpt-{index}"},
            )
        )
        for index in range(6)
    ]

    async def slow_status(acc):
        await asyncio.sleep(0.1)
        return {"quota": {"groups": []}}

    monkeypatch.setattr("llmproxy.dashboard.router.fetch_codex_status", slow_status)

    started = time.perf_counter()
    response = client.get("/api/codex/status")
    elapsed = time.perf_counter() - started

    assert response.status_code == 200
    assert len(response.json()["accounts"]) == len(accounts)
    # Serial requests would take ~0.6s; concurrent requests should finish much sooner.
    assert elapsed < 0.5
