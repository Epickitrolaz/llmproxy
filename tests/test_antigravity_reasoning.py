"""
Tests for Google Antigravity Provider Plugin:
- Reasoning tags removed
- Reasoning effort mapped to Google's model variants (-high, -medium, -low)
- Default strategy is fallback
- Switch delay defaults to 60s
"""

import json
import pytest
from unittest.mock import patch

from llmproxy.core.manager import plugin_manager
from llmproxy.schemas import (
    ChatMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    PluginModelConfig,
)
from templates.antigravity import AntigravityPlugin


@pytest.fixture(autouse=True)
def setup_antigravity():
    orig_accounts = dict(plugin_manager.accounts)
    plugin_manager.accounts.clear()
    plugin_manager.update_plugin_config(
        "antigravity",
        enabled=True,
        variables={
            "api_key": "test-key",
            "timeout": 60,
            "load_balancing_strategy": "fallback",
            "account_switch_wait": 60,
            "reasoning_effort": "high",
        },
    )
    yield
    plugin_manager.accounts = orig_accounts


def test_antigravity_metadata_settings_and_defaults():
    """Verify reasoning_tags are removed and default strategy/switch delay are set."""
    plugin = AntigravityPlugin()
    var_names = {v.name: v for v in plugin.metadata.variables}

    # reasoning_tags and reasoning_tag_name must NOT exist
    assert "reasoning_tags" not in var_names
    assert "reasoning_tag_name" not in var_names

    # reasoning_effort must be declared and default to high
    assert "reasoning_effort" in var_names
    assert var_names["reasoning_effort"].type == "select"
    assert var_names["reasoning_effort"].default == "high"
    assert "high" in var_names["reasoning_effort"].options
    assert "medium" in var_names["reasoning_effort"].options
    assert "low" in var_names["reasoning_effort"].options
    assert "none" in var_names["reasoning_effort"].options

    # Strategy should default to fallback
    assert var_names["load_balancing_strategy"].default == "fallback"

    # Switch wait delay should default to 60
    assert var_names["account_switch_wait"].default == 60


def test_antigravity_models_exposed_without_suffix():
    """Verify that models are exposed without the -high/-medium/-low suffix by default."""
    plugin = AntigravityPlugin()
    model_ids = {m.id for m in plugin.metadata.models}

    # Default exposed models without suffix
    assert "gemini-3.8-flash" in model_ids
    assert "gemini-3.1-pro" in model_ids
    assert "claude-sonnet-4-6" in model_ids
    assert "claude-opus-4-6-thinking" in model_ids

    # Suffix variants should NOT be exposed in the default models list
    assert "gemini-3.8-flash-high" not in model_ids
    assert "gemini-3.8-flash-medium" not in model_ids
    assert "gemini-3.8-flash-low" not in model_ids
    assert "gemini-3.1-pro-high" not in model_ids


def test_reasoning_effort_maps_to_google_upstream_model_variants():
    """Verify that reasoning effort routes to Google's -high, -medium, -low models under the hood."""
    plugin = AntigravityPlugin()
    model_cfg = PluginModelConfig(
        id="gemini-3.8-flash",
        name="Gemini 3.8 Flash",
        upstream_model="gemini-3.8-flash",
    )

    # 1. Default (no effort specified) -> maps to -high
    req_default = ChatCompletionRequest(
        model="gemini-3.8-flash",
        messages=[ChatMessage(role="user", content="hello")],
    )
    actual_model, level, enable = plugin._parse_reasoning_level(req_default, model_cfg, {})
    assert actual_model == "gemini-3.8-flash-high"
    assert level == "high"
    assert enable is True

    # 2. Explicit reasoning_effort: "high"
    req_high = ChatCompletionRequest(
        model="gemini-3.8-flash",
        messages=[ChatMessage(role="user", content="hello")],
        reasoning_effort="high",
    )
    actual_model, level, enable = plugin._parse_reasoning_level(req_high, model_cfg, {})
    assert actual_model == "gemini-3.8-flash-high"
    assert level == "high"
    assert enable is True

    # 3. Explicit reasoning_effort: "medium"
    req_med = ChatCompletionRequest(
        model="gemini-3.8-flash",
        messages=[ChatMessage(role="user", content="hello")],
        reasoning_effort="medium",
    )
    actual_model, level, enable = plugin._parse_reasoning_level(req_med, model_cfg, {})
    assert actual_model == "gemini-3.8-flash-medium"
    assert level == "medium"
    assert enable is True

    # 4. Explicit reasoning_effort: "low"
    req_low = ChatCompletionRequest(
        model="gemini-3.8-flash",
        messages=[ChatMessage(role="user", content="hello")],
        reasoning_effort="low",
    )
    actual_model, level, enable = plugin._parse_reasoning_level(req_low, model_cfg, {})
    assert actual_model == "gemini-3.8-flash-low"
    assert level == "low"
    assert enable is True

    # 5. Explicit reasoning_effort: "none"
    req_none = ChatCompletionRequest(
        model="gemini-3.8-flash",
        messages=[ChatMessage(role="user", content="hello")],
        reasoning_effort="none",
    )
    actual_model, level, enable = plugin._parse_reasoning_level(req_none, model_cfg, {})
    assert actual_model == "gemini-3.8-flash"
    assert level == "none"
    assert enable is False

    # 6. Gemini 3.1 Pro with medium effort
    pro_cfg = PluginModelConfig(id="gemini-3.1-pro", name="Gemini 3.1 Pro", upstream_model="gemini-3.1-pro")
    req_pro = ChatCompletionRequest(
        model="gemini-3.1-pro",
        messages=[ChatMessage(role="user", content="hello")],
        reasoning_effort="medium",
    )
    actual_model, level, enable = plugin._parse_reasoning_level(req_pro, pro_cfg, {})
    assert actual_model == "gemini-3.1-pro-medium"
    assert level == "medium"
    assert enable is True

    # 7. Via extra_body
    req_extra = ChatCompletionRequest(
        model="gemini-3.8-flash",
        messages=[ChatMessage(role="user", content="hello")],
        extra_body={"reasoning_effort": "low"},
    )
    actual_model, level, enable = plugin._parse_reasoning_level(req_extra, model_cfg, {})
    assert actual_model == "gemini-3.8-flash-low"
    assert level == "low"
    assert enable is True


def test_antigravity_sanitizes_reasoning_fields():
    """Verify reasoning config fields are stripped from direct upstream Google payloads."""
    plugin = AntigravityPlugin()
    payload = {
        "model": "gemini-3.8-flash-high",
        "messages": [{"role": "user", "content": "hello"}],
        "reasoning_tags": True,
        "reasoning_tag": "think",
        "reasoning_tag_name": "think",
        "thinking": True,
        "reasoning_effort": "high",
    }
    cleaned = plugin.sanitize_payload(payload)
    assert "reasoning_tags" not in cleaned
    assert "reasoning_tag" not in cleaned
    assert "reasoning_tag_name" not in cleaned
    assert "thinking" not in cleaned
    assert "reasoning_effort" not in cleaned
    assert cleaned["model"] == "gemini-3.8-flash-high"


def test_mock_response_with_reasoning_content_no_tags():
    """Verify _mock_response delivers thoughts via reasoning_content and never tags."""
    plugin = AntigravityPlugin()
    model_cfg = PluginModelConfig(id="gemini-3.8-flash", name="Gemini 3.8 Flash")

    # Default without mock_thought: normal simulated response, no think tags
    req_normal = ChatCompletionRequest(
        model="gemini-3.8-flash",
        messages=[ChatMessage(role="user", content="hello")],
    )
    resp_normal = plugin._mock_response(req_normal, model_cfg)
    assert "<think>" not in resp_normal.choices[0].message.content
    assert getattr(resp_normal.choices[0].message, "reasoning_content", None) is None

    # With mock_thought: reasoning_content is set, content does NOT duplicate reasoning
    req_thought = ChatCompletionRequest(
        model="gemini-3.8-flash",
        messages=[ChatMessage(role="user", content="solve 2+2")],
        extra_body={"mock_thought": True},
    )
    resp_thought = plugin._mock_response(req_thought, model_cfg)
    content = resp_thought.choices[0].message.content
    assert "<think>" not in content
    assert "</think>" not in content
    assert "Simulated reasoning" not in content
    assert resp_thought.choices[0].message.reasoning_content == "Simulated reasoning for gemini-3.8-flash."


@pytest.mark.anyio
async def test_mock_stream_reasoning_content_no_tags():
    """Verify _mock_stream emits reasoning_content and no think tags in content."""
    plugin = AntigravityPlugin()
    model_cfg = PluginModelConfig(id="gemini-3.8-flash", name="Gemini 3.8 Flash")

    req_stream = ChatCompletionRequest(
        model="gemini-3.8-flash",
        messages=[ChatMessage(role="user", content="solve 2+2")],
        stream=True,
        extra_body={"mock_thought": True},
    )

    stream_gen = await plugin.chat_completion(
        req_stream,
        model_cfg,
        variables={"api_key": "test-key"},
    )

    chunks = []
    async for chunk in stream_gen:
        chunks.append(chunk)

    contents = [c.choices[0].delta.get("content", "") for c in chunks if c.choices[0].delta.get("content")]
    full_text = "".join(contents)

    assert "<think>" not in full_text
    assert "</think>" not in full_text
    assert "Simulated reasoning." not in full_text

    thought_deltas = [
        c.choices[0].delta for c in chunks
        if c.choices[0].delta.get("reasoning_content")
    ]
    assert len(thought_deltas) > 0
    assert thought_deltas[0]["reasoning_content"] == "Simulated reasoning."


@pytest.mark.anyio
async def test_upstream_non_streaming_reasoning_content():
    """Verify authentic upstream SSE thought parts map directly to reasoning_content."""
    plugin = AntigravityPlugin()
    model_cfg = PluginModelConfig(id="gemini-3.8-flash", name="Gemini 3.8 Flash")

    sse_lines = [
        b'data: {"response": {"candidates": [{"content": {"role": "model", "parts": [{"thought": true, "text": "I am analyzing the problem step by step."}]}}]}}\r\n',
        b'data: {"response": {"candidates": [{"content": {"role": "model", "parts": [{"thought": true, "text": " Everything looks clear."}]}}]}}\r\n',
        b'data: {"response": {"candidates": [{"content": {"role": "model", "parts": [{"text": "The answer is 42."}]}}], "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 20, "totalTokenCount": 30}}}\r\n',
        b'data: [DONE]\r\n',
    ]

    class FakeStreamResponse:
        status_code = 200
        async def aiter_lines(self):
            for l in sse_lines:
                yield l.decode("utf-8").strip()

    class FakeClient:
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
        def stream(self, method, url, **kwargs):
            class FakeStreamContext:
                async def __aenter__(self):
                    return FakeStreamResponse()
                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    pass
            return FakeStreamContext()

    req = ChatCompletionRequest(
        model="gemini-3.8-flash",
        messages=[ChatMessage(role="user", content="What is the answer?")],
        stream=False,
    )

    with patch("plugins.antigravity.httpx.AsyncClient", return_value=FakeClient()):
        resp = await plugin.chat_completion(
            req,
            model_cfg,
            variables={"api_key": "real-token-123"},
        )

        assert isinstance(resp, ChatCompletionResponse)
        content = resp.choices[0].message.content
        assert "<think>" not in content
        assert "analyzing" not in content
        assert content == "The answer is 42."
        assert resp.choices[0].message.reasoning_content == "I am analyzing the problem step by step. Everything looks clear."
        # Exposed model matches client's request
        assert resp.model == "gemini-3.8-flash"


@pytest.mark.anyio
async def test_upstream_streaming_reasoning_content():
    """Verify streaming correctly emits thought deltas via reasoning_content and answer via content."""
    plugin = AntigravityPlugin()
    model_cfg = PluginModelConfig(id="gemini-3.8-flash", name="Gemini 3.8 Flash")

    sse_lines = [
        b'data: {"response": {"candidates": [{"content": {"role": "model", "parts": [{"thought": true, "text": "Step 1: Calculate sum."}]}}]}}\r\n',
        b'data: {"response": {"candidates": [{"content": {"role": "model", "parts": [{"text": "Sum is 10."}]}}]}}\r\n',
        b'data: [DONE]\r\n',
    ]

    class FakeStreamResponse:
        status_code = 200
        async def aiter_lines(self):
            for l in sse_lines:
                yield l.decode("utf-8").strip()

    class FakeClient:
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
        def stream(self, method, url, **kwargs):
            class FakeStreamContext:
                async def __aenter__(self):
                    return FakeStreamResponse()
                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    pass
            return FakeStreamContext()

    req = ChatCompletionRequest(
        model="gemini-3.8-flash",
        messages=[ChatMessage(role="user", content="Calculate sum")],
        stream=True,
    )

    with patch("plugins.antigravity.httpx.AsyncClient", return_value=FakeClient()):
        stream_gen = await plugin.chat_completion(
            req,
            model_cfg,
            variables={"api_key": "real-token-123"},
        )

        chunks = []
        async for chunk in stream_gen:
            chunks.append(chunk)

        content_deltas = [c.choices[0].delta.get("content", "") for c in chunks if "content" in c.choices[0].delta]
        full_streamed = "".join(content_deltas)

        assert "<think>" not in full_streamed
        assert "</think>" not in full_streamed
        assert "Step 1: Calculate sum." not in full_streamed
        assert full_streamed == "Sum is 10."

        thought_chunks = [c for c in chunks if c.choices[0].delta.get("reasoning_content")]
        assert len(thought_chunks) == 1
        assert thought_chunks[0].choices[0].delta["reasoning_content"] == "Step 1: Calculate sum."


def test_build_upstream_request_sanitizes_prior_reasoning_tags():
    """Verify _build_upstream_request strips reasoning tags from previous assistant messages to avoid context poisoning."""
    plugin = AntigravityPlugin()
    model_cfg = PluginModelConfig(id="gemini-3.8-flash", name="Gemini 3.8 Flash")

    messages = [
        ChatMessage(role="user", content="First question"),
        ChatMessage(
            role="assistant",
            content="<think>\nAnalyzing first question step-by-step.\n</think>\nFirst answer text.",
        ),
        ChatMessage(role="user", content="Second question"),
        ChatMessage(
            role="assistant",
            content="<thought>\nThinking before tool invocation.\n</thought>\n",
            tool_calls=[{"id": "call_abc", "type": "function", "function": {"name": "test_tool", "arguments": "{}"}}],
        ),
        ChatMessage(role="tool", tool_call_id="call_abc", content="result"),
        ChatMessage(role="user", content="Third question"),
        ChatMessage(
            role="assistant",
            content="<think>\nUnclosed thought block without end tag",
        ),
    ]

    req = ChatCompletionRequest(model="gemini-3.8-flash", messages=messages)
    headers, body = plugin._build_upstream_request(req, model_cfg, account=None)

    contents = body["request"]["contents"]
    model_turns = [c for c in contents if c.get("role") == "model"]
    assert len(model_turns) == 3

    # Turn 1: Should only have "First answer text."
    turn1_texts = [p["text"] for p in model_turns[0]["parts"] if "text" in p]
    assert len(turn1_texts) == 1
    assert turn1_texts[0] == "First answer text."
    assert "<think>" not in turn1_texts[0]

    # Turn 2: Thought before tool call was stripped; only functionCall remains
    turn2_parts = model_turns[1]["parts"]
    assert any("functionCall" in p for p in turn2_parts)
    assert not any("<thought>" in p.get("text", "") for p in turn2_parts)

    # Turn 3: Unclosed think block was completely stripped, falling back to empty string
    turn3_texts = [p["text"] for p in model_turns[2]["parts"] if "text" in p]
    assert len(turn3_texts) == 1
    assert turn3_texts[0] == ""
