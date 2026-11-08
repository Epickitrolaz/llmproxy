"""
Tests for Reasoning Tags support in Google Antigravity Provider Plugin.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from llmproxy.main import app
from llmproxy.core.manager import plugin_manager
from llmproxy.schemas import (
    Account,
    ChatMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    PluginModelConfig,
)
from plugins.antigravity import AntigravityPlugin


@pytest.fixture(autouse=True)
def setup_antigravity():
    plugin_manager.discover_and_load_all()
    plugin_manager.update_plugin_config(
        "antigravity",
        enabled=True,
        variables={
            "api_key": "test-key",
            "timeout": 60,
            "reasoning_tags": True,
            "reasoning_tag_name": "think",
        },
    )


def test_antigravity_metadata_has_reasoning_variables():
    """Verify that Antigravity plugin declares reasoning_tags and reasoning_tag_name."""
    plugin = AntigravityPlugin()
    var_names = {v.name: v for v in plugin.metadata.variables}

    assert "reasoning_tags" in var_names
    assert var_names["reasoning_tags"].type == "boolean"
    assert var_names["reasoning_tags"].default is True

    assert "reasoning_tag_name" in var_names
    assert var_names["reasoning_tag_name"].type == "select"
    assert "think" in var_names["reasoning_tag_name"].options
    assert "thought" in var_names["reasoning_tag_name"].options


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


def test_mock_response_with_reasoning_tags():
    """Verify _mock_response wraps simulated reasoning when mock_thought is enabled."""
    plugin = AntigravityPlugin()
    model_cfg = PluginModelConfig(id="gemini-3.8-flash-high", name="Gemini 3.8 Flash High")

    # Default without mock_thought: normal simulated response, no think tags
    req_normal = ChatCompletionRequest(
        model="gemini-3.8-flash-high",
        messages=[ChatMessage(role="user", content="hello")],
    )
    resp_normal = plugin._mock_response(req_normal, model_cfg)
    assert "<think>" not in resp_normal.choices[0].message.content
    assert getattr(resp_normal.choices[0].message, "reasoning_content", None) is None

    # With mock_thought: wrapped in <think> tags
    req_thought = ChatCompletionRequest(
        model="gemini-3.8-flash-high",
        messages=[ChatMessage(role="user", content="solve 2+2")],
        extra_body={"mock_thought": True},
    )
    resp_thought = plugin._mock_response(req_thought, model_cfg)
    content = resp_thought.choices[0].message.content
    assert "<think>" in content
    assert "</think>" in content
    assert "Simulated reasoning for gemini-3.8-flash-high." in content
    assert resp_thought.choices[0].message.reasoning_content == "Simulated reasoning for gemini-3.8-flash-high."

    # With custom tag_name 'thought'
    resp_custom_tag = plugin._mock_response(
        req_thought, model_cfg, enable_reasoning_tags=True, tag_name="thought"
    )
    content_custom = resp_custom_tag.choices[0].message.content
    assert "<thought>" in content_custom
    assert "</thought>" in content_custom
    assert "<think>" not in content_custom


@pytest.mark.anyio
async def test_mock_stream_with_reasoning_tags():
    """Verify _mock_stream emits reasoning tag chunks when mock_thought is enabled."""
    plugin = AntigravityPlugin()
    model_cfg = PluginModelConfig(id="gemini-3.8-flash-high", name="Gemini 3.8 Flash High")

    req_stream = ChatCompletionRequest(
        model="gemini-3.8-flash-high",
        messages=[ChatMessage(role="user", content="solve 2+2")],
        stream=True,
        extra_body={"mock_thought": True},
    )

    stream_gen = await plugin.chat_completion(
        req_stream,
        model_cfg,
        variables={"api_key": "test-key", "reasoning_tags": True, "reasoning_tag_name": "think"},
    )

    chunks = []
    async for chunk in stream_gen:
        chunks.append(chunk)

    contents = [c.choices[0].delta.get("content", "") for c in chunks if c.choices[0].delta.get("content")]
    full_text = "".join(contents)

    assert "<think>" in full_text
    assert "</think>" in full_text
    assert "Simulated reasoning." in full_text

    # Verify reasoning_content is present on the thought chunk delta
    thought_deltas = [
        c.choices[0].delta for c in chunks
        if c.choices[0].delta.get("reasoning_content")
    ]
    assert len(thought_deltas) > 0
    assert thought_deltas[0]["reasoning_content"] == "Simulated reasoning."


@pytest.mark.anyio
async def test_upstream_non_streaming_reasoning_tags():
    """Simulate authentic Google streamGenerateContent SSE lines with thought parts for non-streaming call."""
    plugin = AntigravityPlugin()
    model_cfg = PluginModelConfig(id="gemini-3.8-flash-high", name="Gemini 3.8 Flash High")

    # Mock SSE stream from Google containing both thought and response text
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
        model="gemini-3.8-flash-high",
        messages=[ChatMessage(role="user", content="What is the answer?")],
        stream=False,
    )

    with patch("plugins.antigravity.httpx.AsyncClient", return_value=FakeClient()):
        # Call with real api_key (not test mode)
        resp = await plugin.chat_completion(
            req,
            model_cfg,
            variables={"api_key": "real-token-123", "reasoning_tags": True, "reasoning_tag_name": "think"},
        )

        assert isinstance(resp, ChatCompletionResponse)
        content = resp.choices[0].message.content
        assert content.startswith("<think>\nI am analyzing the problem step by step. Everything looks clear.\n</think>\n")
        assert "The answer is 42." in content
        assert resp.choices[0].message.reasoning_content == "I am analyzing the problem step by step. Everything looks clear."


@pytest.mark.anyio
async def test_upstream_streaming_reasoning_tags():
    """Verify streaming correctly opens, emits thought deltas with reasoning_content, and closes tags."""
    plugin = AntigravityPlugin()
    model_cfg = PluginModelConfig(id="gemini-3.8-flash-high", name="Gemini 3.8 Flash High")

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
        model="gemini-3.8-flash-high",
        messages=[ChatMessage(role="user", content="Calculate sum")],
        stream=True,
    )

    with patch("plugins.antigravity.httpx.AsyncClient", return_value=FakeClient()):
        stream_gen = await plugin.chat_completion(
            req,
            model_cfg,
            variables={"api_key": "real-token-123", "reasoning_tags": True, "reasoning_tag_name": "think"},
        )

        chunks = []
        async for chunk in stream_gen:
            chunks.append(chunk)

        content_deltas = [c.choices[0].delta.get("content", "") for c in chunks if "content" in c.choices[0].delta]
        full_streamed = "".join(content_deltas)

        assert "<think>\n" in full_streamed
        assert "</think>\n" in full_streamed
        assert "Step 1: Calculate sum." in full_streamed
        assert "Sum is 10." in full_streamed

        # Check reasoning_content in delta
        thought_chunks = [c for c in chunks if c.choices[0].delta.get("reasoning_content")]
        assert len(thought_chunks) == 1
        assert thought_chunks[0].choices[0].delta["reasoning_content"] == "Step 1: Calculate sum."


@pytest.mark.anyio
async def test_upstream_streaming_thought_followed_by_tool_call():
    """Verify that when a thought is followed by a tool call, the think tag closes before the tool call."""
    plugin = AntigravityPlugin()
    model_cfg = PluginModelConfig(id="gemini-3.8-flash-high", name="Gemini 3.8 Flash High")

    sse_lines = [
        b'data: {"response": {"candidates": [{"content": {"role": "model", "parts": [{"thought": true, "text": "I should call get_weather now."}]}}]}}\r\n',
        b'data: {"response": {"candidates": [{"content": {"role": "model", "parts": [{"functionCall": {"name": "get_weather", "args": {"location": "London"}, "id": "call_123"}}]}}]}}\r\n',
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
        model="gemini-3.8-flash-high",
        messages=[ChatMessage(role="user", content="Weather in London")],
        stream=True,
    )

    with patch("plugins.antigravity.httpx.AsyncClient", return_value=FakeClient()):
        stream_gen = await plugin.chat_completion(
            req,
            model_cfg,
            variables={"api_key": "real-token-123", "reasoning_tags": True, "reasoning_tag_name": "think"},
        )

        chunks = []
        async for chunk in stream_gen:
            chunks.append(chunk)

        # Confirm close tag was yielded before tool_call chunk
        content_items = []
        tool_call_seen = False
        close_tag_before_tool = False
        for c in chunks:
            delta = c.choices[0].delta
            if "content" in delta and delta["content"]:
                content_items.append(delta["content"])
                if "</think>" in delta["content"] and not tool_call_seen:
                    close_tag_before_tool = True
            if "tool_calls" in delta and delta["tool_calls"]:
                tool_call_seen = True

        assert close_tag_before_tool is True
        assert tool_call_seen is True


@pytest.mark.anyio
async def test_upstream_disabled_reasoning_tags():
    """Verify that when reasoning_tags is False, thoughts are not wrapped in tags."""
    plugin = AntigravityPlugin()
    model_cfg = PluginModelConfig(id="gemini-3.8-flash-high", name="Gemini 3.8 Flash High")

    sse_lines = [
        b'data: {"response": {"candidates": [{"content": {"role": "model", "parts": [{"thought": true, "text": "Raw thought text."}]}}]}}\r\n',
        b'data: {"response": {"candidates": [{"content": {"role": "model", "parts": [{"text": "Final answer."}]}}]}}\r\n',
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
        model="gemini-3.8-flash-high",
        messages=[ChatMessage(role="user", content="test")],
        stream=False,
    )

    with patch("plugins.antigravity.httpx.AsyncClient", return_value=FakeClient()):
        resp = await plugin.chat_completion(
            req,
            model_cfg,
            variables={"api_key": "real-token-123", "reasoning_tags": False},
        )

        content = resp.choices[0].message.content
        assert "<think>" not in content
        assert "</think>" not in content
        assert "Raw thought text.Final answer." == content


def test_build_upstream_request_sanitizes_prior_reasoning_tags():
    """Verify _build_upstream_request strips reasoning tags from previous assistant messages to avoid context poisoning."""
    plugin = AntigravityPlugin()
    model_cfg = PluginModelConfig(id="gemini-3.8-flash-high", name="Gemini 3.8 Flash High")

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

    req = ChatCompletionRequest(model="gemini-3.8-flash-high", messages=messages)
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


@pytest.mark.anyio
async def test_upstream_streaming_preserves_thought_across_whitespace_chunks():
    """Verify that whitespace/newline chunks between thought parts do not cause premature closing and reopening."""
    plugin = AntigravityPlugin()
    model_cfg = PluginModelConfig(id="gemini-3.8-flash-high", name="Gemini 3.8 Flash High")

    sse_lines = [
        b'data: {"response": {"candidates": [{"content": {"role": "model", "parts": [{"thought": true, "text": "Thought step 1."}]}}]}}\r\n',
        b'data: {"response": {"candidates": [{"content": {"role": "model", "parts": [{"text": "\\n\\n"}]}}]}}\r\n',
        b'data: {"response": {"candidates": [{"content": {"role": "model", "parts": [{"thought": true, "text": "Thought step 2."}]}}]}}\r\n',
        b'data: {"response": {"candidates": [{"content": {"role": "model", "parts": [{"text": "Final visible answer."}]}}]}}\r\n',
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
        model="gemini-3.8-flash-high",
        messages=[ChatMessage(role="user", content="solve")],
        stream=True,
    )

    with patch("plugins.antigravity.httpx.AsyncClient", return_value=FakeClient()):
        stream_gen = await plugin.chat_completion(
            req,
            model_cfg,
            variables={"api_key": "real-token-123", "reasoning_tags": True, "reasoning_tag_name": "think"},
        )

        chunks = []
        async for chunk in stream_gen:
            chunks.append(chunk)

        content_deltas = [c.choices[0].delta.get("content", "") for c in chunks if "content" in c.choices[0].delta]
        full_streamed = "".join(content_deltas)

        # There should be exactly ONE <think> and ONE </think> tag
        assert full_streamed.count("<think>") == 1
        assert full_streamed.count("</think>") == 1
        assert "Thought step 1." in full_streamed
        assert "Thought step 2." in full_streamed
        assert "Final visible answer." in full_streamed


@pytest.mark.anyio
async def test_upstream_streaming_handles_raw_tag_leakage():
    """Verify that if upstream thought text already has raw think tags, they are not duplicated."""
    plugin = AntigravityPlugin()
    model_cfg = PluginModelConfig(id="gemini-3.8-flash-high", name="Gemini 3.8 Flash High")

    sse_lines = [
        b'data: {"response": {"candidates": [{"content": {"role": "model", "parts": [{"thought": true, "text": "<think>\\nInner thought\\n</think>"}]}}]}}\r\n',
        b'data: {"response": {"candidates": [{"content": {"role": "model", "parts": [{"text": "Answer."}]}}]}}\r\n',
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
        model="gemini-3.8-flash-high",
        messages=[ChatMessage(role="user", content="solve")],
        stream=True,
    )

    with patch("plugins.antigravity.httpx.AsyncClient", return_value=FakeClient()):
        stream_gen = await plugin.chat_completion(
            req,
            model_cfg,
            variables={"api_key": "real-token-123", "reasoning_tags": True, "reasoning_tag_name": "think"},
        )

        chunks = []
        async for chunk in stream_gen:
            chunks.append(chunk)

        content_deltas = [c.choices[0].delta.get("content", "") for c in chunks if "content" in c.choices[0].delta]
        full_streamed = "".join(content_deltas)

        # No double <think><think> tags
        assert full_streamed.count("<think>") == 1
        assert full_streamed.count("</think>") == 1
        assert "Inner thought" in full_streamed
        assert "Answer." in full_streamed

