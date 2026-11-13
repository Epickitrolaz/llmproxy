"""Regression tests for rule status attribution, token totals, and UI fixes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from llmproxy.core.manager import plugin_manager
from llmproxy.main import app
from llmproxy.schemas import (
    ChatChoice,
    ChatChoiceDelta,
    ChatCompletionChunk,
    ChatCompletionResponse,
    ChatMessage,
    CompletionUsage,
    LoadBalancingRuleCreate,
)
from templates.anthropic_proxy import AnthropicProxyPlugin

client = TestClient(app, headers={"X-Admin-Token": "test-admin-token"})


@pytest.fixture(autouse=True)
def clean_state():
    original_logs = list(plugin_manager.logs)
    original_rules = dict(plugin_manager.rules)
    original_requests = plugin_manager.total_requests_count
    original_configs = json.loads(json.dumps(plugin_manager.configs))
    plugin_manager.logs.clear()
    plugin_manager.rules.clear()
    plugin_manager.total_requests_count = 0
    yield
    plugin_manager.logs = original_logs
    plugin_manager.rules = original_rules
    plugin_manager.total_requests_count = original_requests
    plugin_manager.configs = original_configs
    plugin_manager._save_rules()
    plugin_manager._save_logs()
    plugin_manager._save_configs()


def test_rule_logs_downstream_model_and_anthropic_token_totals(monkeypatch):
    plugin = plugin_manager.get_plugin("mock_provider")
    if plugin is None:
        pytest.skip("mock provider is not loaded")

    plugin_manager.update_plugin_config("mock_provider", enabled=True, variables={})
    plugin_manager.add_load_balancing_rule(
        LoadBalancingRuleCreate(
            model_id="orchestrator-free",
            name="Orchestrator Free",
            strategy="round_robin",
            targets=["mock-gpt-4o"],
        )
    )

    # Anthropic-style raw usage: input_tokens excludes both cache buckets.
    usage = AnthropicProxyPlugin()._build_usage(
        {
            "input_tokens": 120,
            "output_tokens": 30,
            "cache_read_input_tokens": 700,
            "cache_creation_input_tokens": 180,
        }
    )

    async def fake_chat_completion(request, model_config, variables, account=None):
        return ChatCompletionResponse(
            id="chatcmpl-rule-test",
            model=model_config.id,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="OK"),
                    finish_reason="stop",
                )
            ],
            usage=usage,
        )

    monkeypatch.setattr(plugin, "chat_completion", fake_chat_completion)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "orchestrator-free",
            "messages": [{"role": "user", "content": "ping"}],
        },
    )
    assert response.status_code == 200

    log = client.get("/api/logs", params={"limit": 1}).json()[0]
    assert log["model"] == "mock-gpt-4o"
    assert log["plugin_id"] == "mock_provider"
    assert log["prompt_tokens"] == 1000  # 120 raw + 700 cached read + 180 written
    assert log["completion_tokens"] == 30
    assert log["cache_read_input_tokens"] == 700
    assert log["cache_creation_input_tokens"] == 180

    stats = client.get("/api/stats").json()
    assert stats["cache_hit_rate"] == 70.0


def test_missing_completion_usage_preserves_cache_counters(monkeypatch):
    plugin = plugin_manager.get_plugin("mock_provider")
    if plugin is None:
        pytest.skip("mock provider is not loaded")
    plugin_manager.update_plugin_config("mock_provider", enabled=True, variables={})

    async def fake_chat_completion(request, model_config, variables, account=None):
        return ChatCompletionResponse(
            id="chatcmpl-missing-completion",
            model=model_config.id,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="OK"),
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=1000,
                completion_tokens=0,
                total_tokens=1000,
                prompt_tokens_details={"cached_tokens": 800},
            ),
        )

    monkeypatch.setattr(plugin, "chat_completion", fake_chat_completion)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "mock-gpt-4o", "messages": [{"role": "user", "content": "ping"}]},
    )
    assert response.status_code == 200

    log = client.get("/api/logs", params={"limit": 1}).json()[0]
    assert log["prompt_tokens"] == 1000
    assert log["completion_tokens"] > 0
    assert log["cache_read_input_tokens"] == 800


def test_streaming_usage_keeps_cache_counters_from_earlier_chunks(monkeypatch):
    plugin = plugin_manager.get_plugin("mock_provider")
    if plugin is None:
        pytest.skip("mock provider is not loaded")
    plugin_manager.update_plugin_config("mock_provider", enabled=True, variables={})

    async def fake_chat_completion(request, model_config, variables, account=None):
        async def stream():
            yield ChatCompletionChunk(
                id="chatcmpl-stream-cache",
                model=model_config.id,
                choices=[ChatChoiceDelta(index=0, delta={"content": "OK"})],
                usage=CompletionUsage(
                    prompt_tokens=1000,
                    completion_tokens=0,
                    total_tokens=1000,
                    prompt_tokens_details={"cached_tokens": 800},
                ),
            )
            yield ChatCompletionChunk(
                id="chatcmpl-stream-cache",
                model=model_config.id,
                choices=[ChatChoiceDelta(index=0, delta={}, finish_reason="stop")],
                usage=CompletionUsage(
                    prompt_tokens=1000,
                    completion_tokens=20,
                    total_tokens=1020,
                ),
            )

        return stream()

    monkeypatch.setattr(plugin, "chat_completion", fake_chat_completion)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-gpt-4o",
            "messages": [{"role": "user", "content": "ping"}],
            "stream": True,
        },
    )
    assert response.status_code == 200

    log = client.get("/api/logs", params={"limit": 1}).json()[0]
    assert log["prompt_tokens"] == 1000
    assert log["completion_tokens"] == 20
    assert log["cache_read_input_tokens"] == 800


def test_providers_tab_has_a_fifth_status_banner():
    html = Path("llmproxy/static/index.html").read_text(encoding="utf-8")
    provider_section = html.split('id="tab-providers"', 1)[1].split("<!-- TAB 3", 1)[0]
    assert 'id="stat-provider-models"' in provider_section
    assert provider_section.count('class="stat-card"') == 5


def test_switch_toggle_cannot_shrink_outside_its_track():
    css = Path("llmproxy/static/style.css").read_text(encoding="utf-8")
    switch_block = css.split(".switch {", 1)[1].split("}", 1)[0]
    assert "flex: 0 0 36px" in switch_block
