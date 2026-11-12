"""Automatic Anthropic prompt-cache breakpoint tests (issue #49)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from llmproxy.api.router import _usage_cache_tokens
from llmproxy.core.manager import plugin_manager
from llmproxy.main import app
from llmproxy.schemas import (
    ChatCompletionRequest,
    ChatMessage,
    CompletionUsage,
    PluginModelConfig,
    RequestLogEntry,
)
from templates.anthropic_proxy import AnthropicProxyPlugin

LARGE_TEXT = "stable prefix " * 1000
MEDIUM_TEXT = "prefix " * 500


def _model(model_id: str = "claude-sonnet-5-5") -> PluginModelConfig:
    plugin = AnthropicProxyPlugin()
    return next(m for m in plugin.metadata.models if m.id == model_id)


def _count_breakpoints(value) -> int:
    if isinstance(value, dict):
        count = 1 if value.get("cache_control") else 0
        return count + sum(_count_breakpoints(item) for item in value.values())
    if isinstance(value, list):
        return sum(_count_breakpoints(item) for item in value)
    return 0


def _large_request(model_id: str = "claude-sonnet-5-5", **kwargs) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=model_id,
        messages=[
            ChatMessage(role="system", content=LARGE_TEXT),
            ChatMessage(role="user", content=LARGE_TEXT),
        ],
        reasoning_effort="none",
        **kwargs,
    )


def test_auto_cache_adds_system_and_final_message_breakpoints():
    plugin = AnthropicProxyPlugin()
    payload = plugin._build_payload(_large_request(), _model(), {})

    assert payload["system"][-1]["cache_control"] == {"type": "ephemeral"}
    assert payload["messages"][-1]["content"][-1]["cache_control"] == {
        "type": "ephemeral"
    }
    assert _count_breakpoints(payload) == 2


def test_auto_cache_works_for_streaming_requests():
    plugin = AnthropicProxyPlugin()
    payload = plugin._build_payload(_large_request(stream=True), _model(), {})

    assert payload["stream"] is True
    assert _count_breakpoints(payload) == 2


def test_auto_cache_skips_prompts_below_minimum_cacheable_size():
    plugin = AnthropicProxyPlugin()
    request = ChatCompletionRequest(
        model="claude-sonnet-5-5",
        messages=[ChatMessage(role="user", content="short prompt")],
        reasoning_effort="none",
    )
    payload = plugin._build_payload(request, _model(), {})

    assert _count_breakpoints(payload) == 0


def test_auto_cache_uses_model_specific_minimum():
    plugin = AnthropicProxyPlugin()
    request = ChatCompletionRequest(
        model="claude-haiku-5-5",
        messages=[
            ChatMessage(role="system", content=MEDIUM_TEXT),
            ChatMessage(role="user", content=MEDIUM_TEXT),
        ],
        reasoning_effort="none",
    )

    sonnet_payload = plugin._build_payload(request, _model("claude-sonnet-5-5"), {})
    haiku_payload = plugin._build_payload(request, _model("claude-haiku-5-5"), {})

    assert _count_breakpoints(sonnet_payload) == 2
    assert _count_breakpoints(haiku_payload) == 0


def test_auto_cache_does_not_override_explicit_client_breakpoints():
    plugin = AnthropicProxyPlugin()
    explicit = {"type": "ephemeral"}
    request = ChatCompletionRequest(
        model="claude-sonnet-5-5",
        messages=[
            ChatMessage(
                role="system",
                content=[{"type": "text", "text": LARGE_TEXT, "cache_control": explicit}],
            ),
            ChatMessage(role="user", content=LARGE_TEXT),
        ],
        reasoning_effort="none",
    )
    payload = plugin._build_payload(request, _model(), {})

    assert payload["system"][0]["cache_control"] == explicit
    assert payload["messages"][-1]["content"][-1].get("cache_control") is None
    assert _count_breakpoints(payload) == 1


def test_auto_cache_does_not_override_extra_body_hint():
    plugin = AnthropicProxyPlugin()
    request = _large_request(extra_body={"cache_control": {"type": "ephemeral"}})
    payload = plugin._build_payload(request, _model(), {})

    assert payload["system"][-1].get("cache_control") is None
    assert payload["messages"][-1]["content"][-1]["cache_control"] == {
        "type": "ephemeral"
    }
    assert _count_breakpoints(payload) == 1


def test_auto_cache_can_be_disabled():
    plugin = AnthropicProxyPlugin()
    payload = plugin._build_payload(
        _large_request(),
        _model(),
        {"auto_prompt_cache": False},
    )
    assert _count_breakpoints(payload) == 0

    # String values from the dashboard editor are accepted too.
    payload_str = plugin._build_payload(
        _large_request(),
        _model(),
        {"auto_prompt_cache": "false"},
    )
    assert _count_breakpoints(payload_str) == 0


def test_auto_cache_never_exceeds_anthropic_breakpoint_limit():
    plugin = AnthropicProxyPlugin()
    payload = plugin._build_payload(_large_request(), _model(), {})
    assert _count_breakpoints(payload) <= 4


def test_usage_cache_tokens_handles_missing_and_present_counters():
    assert _usage_cache_tokens(None) == (0, 0)
    assert _usage_cache_tokens(CompletionUsage(prompt_tokens=10)) == (0, 0)

    usage = CompletionUsage(
        prompt_tokens=10,
        completion_tokens=2,
        total_tokens=12,
        cache_read_input_tokens=7,
        cache_creation_input_tokens=3,
    )
    assert _usage_cache_tokens(usage) == (7, 3)


def test_request_logs_persist_cache_counters_and_stats_show_hit_rate():
    original_logs = list(plugin_manager.logs)
    original_requests = plugin_manager.total_requests_count
    try:
        plugin_manager.logs = [
            RequestLogEntry(
                id="cache-log-1",
                model="claude-opus-5-5",
                status="success",
                prompt_tokens=100,
                completion_tokens=20,
                cache_read_input_tokens=800,
                cache_creation_input_tokens=100,
            ),
            RequestLogEntry(
                id="cache-log-2",
                model="gpt-4o",
                status="success",
                prompt_tokens=100,
                completion_tokens=20,
            ),
        ]
        plugin_manager.total_requests_count = 2

        client = TestClient(app)
        logs = client.get("/api/logs").json()
        by_id = {log["id"]: log for log in logs}
        assert by_id["cache-log-1"]["cache_read_input_tokens"] == 800
        assert by_id["cache-log-1"]["cache_creation_input_tokens"] == 100

        stats = client.get("/api/stats").json()
        # 800 cached out of (100 + 800 + 100) + 100 = 1100 cacheable prompt tokens.
        assert stats["cache_read_input_tokens"] == 800
        assert stats["cache_creation_input_tokens"] == 100
        assert stats["cache_hit_rate"] == 72.7
    finally:
        plugin_manager.logs = original_logs
        plugin_manager.total_requests_count = original_requests
