"""Cache telemetry must reach request logs and dashboard stats for every provider.

OpenAI-style providers report cached tokens through
``prompt_tokens_details.cached_tokens`` while Anthropic reports
``cache_read_input_tokens``. Both shapes must produce the same dashboard data.
"""

from __future__ import annotations

import json

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
    ChatCompletionRequest,
    PluginModelConfig,
)
from templates.antigravity import AntigravityPlugin
from templates.anthropic_proxy import AnthropicProxyPlugin
from templates.codex import CodexPlugin
from templates.openai_compatible import OpenAICompatiblePlugin
from templates.opencode_free import OpenCodeFreePlugin
from templates.openrouter import OpenRouterPlugin

client = TestClient(app, headers={"X-Admin-Token": "test-admin-token"})


@pytest.fixture(autouse=True)
def clean_state():
    orig_logs = list(plugin_manager.logs)
    orig_accounts = dict(plugin_manager.accounts)
    orig_configs = json.loads(json.dumps(plugin_manager.configs))
    plugin_manager.accounts.clear()
    plugin_manager.logs.clear()
    yield
    plugin_manager.logs = orig_logs
    plugin_manager.accounts = orig_accounts
    plugin_manager.configs = orig_configs
    plugin_manager._save_accounts()


def _setup_mock_provider():
    for account in list(plugin_manager.get_accounts("mock_provider")):
        plugin_manager.delete_account(account.id)
    account = plugin_manager.add_account(
        AccountCreate(provider_id="mock_provider", name="Cache Test", api_key="key-1")
    )
    plugin_manager.update_plugin_config("mock_provider", enabled=True, variables={})
    return plugin_manager.get_plugin("mock_provider"), account


def _openai_style_usage(prompt_tokens: int = 1000, cached_tokens: int = 800):
    return CompletionUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=20,
        total_tokens=prompt_tokens + 20,
        prompt_tokens_details={"cached_tokens": cached_tokens},
    )


def test_openai_style_cache_usage_is_logged(monkeypatch):
    plugin, _account = _setup_mock_provider()

    async def fake_chat(request, model_config, variables, account=None):
        return ChatCompletionResponse(
            id="chatcmpl-cache",
            model=request.model,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="OK"),
                    finish_reason="stop",
                )
            ],
            usage=_openai_style_usage(),
        )

    monkeypatch.setattr(plugin, "chat_completion", fake_chat)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "mock-gpt-4o", "messages": [{"role": "user", "content": "ping"}]},
    )
    assert response.status_code == 200
    assert response.json()["usage"]["cache_read_input_tokens"] == 800

    log = client.get("/api/logs").json()[0]
    assert log["prompt_tokens"] == 1000
    assert log["cache_read_input_tokens"] == 800
    assert log["cache_creation_input_tokens"] == 0

    stats = client.get("/api/stats").json()
    assert stats["cache_read_input_tokens"] == 800
    assert stats["cache_hit_rate"] == 80.0


def test_openai_style_cache_usage_is_logged_from_streams(monkeypatch):
    plugin, _account = _setup_mock_provider()

    async def fake_chat(request, model_config, variables, account=None):
        async def stream():
            yield ChatCompletionChunk(
                id="chunk-cache",
                model=request.model,
                choices=[ChatChoiceDelta(index=0, delta={"content": "OK"})],
            )
            yield ChatCompletionChunk(
                id="chunk-cache",
                model=request.model,
                choices=[ChatChoiceDelta(index=0, delta={}, finish_reason="stop")],
                usage=_openai_style_usage(),
            )

        return stream()

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
    # Usage is only forwarded when the client opts in.
    assert "cache_read_input_tokens" not in response.text

    log = client.get("/api/logs").json()[0]
    assert log["cache_read_input_tokens"] == 800

    stats = client.get("/api/stats").json()
    assert stats["cache_hit_rate"] == 80.0


def test_anthropic_native_counters_match_openai_semantics():
    plugin = AnthropicProxyPlugin()
    usage = plugin._build_usage(
        {
            "input_tokens": 120,
            "output_tokens": 30,
            "cache_read_input_tokens": 700,
            "cache_creation_input_tokens": 180,
        }
    )
    assert usage.prompt_tokens == 1000
    assert usage.cache_read_input_tokens == 700
    assert usage.cache_creation_input_tokens == 180


def test_antigravity_usage_maps_cached_content_tokens():
    usage = AntigravityPlugin._build_usage(
        {
            "promptTokenCount": 1000,
            "candidatesTokenCount": 20,
            "totalTokenCount": 1020,
            "cachedContentTokenCount": 800,
        }
    )
    assert usage.prompt_tokens == 1000
    assert usage.cache_read_input_tokens == 800


def test_codex_usage_maps_cached_input_tokens():
    usage = CodexPlugin._build_usage(
        {
            "input_tokens": 1000,
            "output_tokens": 20,
            "total_tokens": 1020,
            "input_tokens_details": {"cached_tokens": 800},
        }
    )
    assert usage.prompt_tokens == 1000
    assert usage.cache_read_input_tokens == 800


@pytest.mark.parametrize(
    "plugin,payload",
    [
        (
            OpenAICompatiblePlugin(),
            {
                "id": "chatcmpl-oai",
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "OK"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1000,
                    "completion_tokens": 20,
                    "total_tokens": 1020,
                    "prompt_tokens_details": {"cached_tokens": 800},
                },
            },
        ),
        (
            OpenRouterPlugin(),
            {
                "id": "chatcmpl-openrouter",
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "OK"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1000,
                    "completion_tokens": 20,
                    "total_tokens": 1020,
                    "prompt_tokens_details": {"cached_tokens": 800},
                },
            },
        ),
        (
            OpenCodeFreePlugin(),
            {
                "id": "chatcmpl-opencode",
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "OK"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1000,
                    "completion_tokens": 20,
                    "total_tokens": 1020,
                    "prompt_tokens_details": {"cached_tokens": 800},
                },
            },
        ),
    ],
)
def test_openai_compatible_templates_preserve_cache_details(plugin, payload):
    response = ChatCompletionResponse.model_validate(payload)
    assert response.usage.prompt_tokens_details == {"cached_tokens": 800}
    assert response.usage.cache_read_input_tokens == 800


def test_streaming_payloads_request_usage_for_cache_accounting():
    request = ChatCompletionRequest(
        model="test-model",
        messages=[ChatMessage(role="user", content="hi")],
        stream=True,
    )
    model = PluginModelConfig(id="test-model", name="Test", upstream_model="test-model")

    from templates.opencode_go import OpenCodeGoPlugin

    opencode_body = OpenCodeFreePlugin()._build_upstream_payload(request, model)
    assert opencode_body["stream_options"]["include_usage"] is True

    go_body = OpenCodeGoPlugin()._build_body(request, model, {})
    assert go_body["stream_options"]["include_usage"] is True

    openrouter_body = OpenRouterPlugin()._build_body(request, model, {})
    assert openrouter_body["stream_options"]["include_usage"] is True
