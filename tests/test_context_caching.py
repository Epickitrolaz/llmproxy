"""Cross-provider tests for prompt/context caching support."""

from __future__ import annotations

import asyncio

import pytest

from llmproxy.schemas import (
    ChatCompletionRequest,
    ChatMessage,
    ChatCompletionResponse,
    PluginModelConfig,
)
from templates.antigravity import AntigravityPlugin
from templates.anthropic_proxy import AnthropicProxyPlugin
from templates.codex import CodexPlugin
from templates.mock_provider import MockProviderPlugin
from templates.openai_compatible import OpenAICompatiblePlugin
from templates.opencode_free import OpenCodeFreePlugin
from templates.openrouter import OpenRouterPlugin
import templates.openai_compatible as openai_module


CACHE_CONTROL = {"type": "ephemeral"}


def test_anthropic_usage_includes_cache_counters():
    plugin = AnthropicProxyPlugin()
    usage = plugin._build_usage(
        {
            "input_tokens": 120,
            "output_tokens": 30,
            "cache_read_input_tokens": 90,
            "cache_creation_input_tokens": 10,
        }
    )
    assert usage.prompt_tokens == 220  # 120 input + 90 cache read + 10 cache write
    assert usage.completion_tokens == 30
    assert usage.prompt_tokens_details == {"cached_tokens": 90}
    assert usage.cache_read_input_tokens == 90
    assert usage.cache_creation_input_tokens == 10

    previous = plugin._build_usage(
        {"input_tokens": 120, "cache_read_input_tokens": 90}
    )
    merged = plugin._build_usage(
        {"output_tokens": 30},
        default_prompt=previous.prompt_tokens,
        previous=previous,
    )
    assert merged.prompt_tokens == 210  # 120 input + 90 cache read
    assert merged.completion_tokens == 30
    assert merged.prompt_tokens_details == {"cached_tokens": 90}


def test_codex_forwards_prompt_cache_key_and_maps_usage():
    plugin = CodexPlugin()
    model = PluginModelConfig(id="gpt-6-luna", name="GPT-6 Luna", upstream_model="gpt-6-luna")
    request = ChatCompletionRequest(
        model="gpt-6-luna",
        messages=[ChatMessage(role="user", content="hi")],
        extra_body={"prompt_cache_key": "cache-key-1", "prompt_cache_retention": "24h"},
    )
    _, _, payload, _ = plugin._build_upstream_request(request, model, None, {})
    assert payload["prompt_cache_key"] == "cache-key-1"
    assert payload["prompt_cache_retention"] == "24h"

    usage = plugin._build_usage(
        {
            "input_tokens": 200,
            "output_tokens": 40,
            "total_tokens": 240,
            "input_tokens_details": {"cached_tokens": 160},
            "output_tokens_details": {"reasoning_tokens": 12},
        }
    )
    assert usage.prompt_tokens == 200
    assert usage.prompt_tokens_details == {"cached_tokens": 160}
    assert usage.cache_read_input_tokens == 160
    assert usage.completion_tokens_details == {"reasoning_tokens": 12}


def test_antigravity_forwards_cached_content_and_cache_control():
    plugin = AntigravityPlugin()
    model = PluginModelConfig(
        id="claude-sonnet-4-6",
        name="Claude Sonnet 4.6",
        upstream_model="claude-sonnet-4-6",
    )
    request = ChatCompletionRequest(
        model="claude-sonnet-4-6",
        messages=[
            ChatMessage(
                role="system",
                content=[
                    {
                        "type": "text",
                        "text": "Stable system prefix",
                        "cache_control": CACHE_CONTROL,
                    }
                ],
            ),
            ChatMessage(
                role="user",
                content=[
                    {
                        "type": "text",
                        "text": "Stable user prefix",
                        "cache_control": CACHE_CONTROL,
                    }
                ],
            ),
        ],
        extra_body={"cached_content": "cachedContents/abc123"},
    )
    _, body = plugin._build_upstream_request(
        request,
        model,
        account=None,
        actual_model="claude-sonnet-4-6-thinking",
        enable_reasoning=True,
    )
    inner = body["request"]
    assert inner["cachedContent"] == "cachedContents/abc123"
    assert inner["systemInstruction"]["parts"][0]["cache_control"] == CACHE_CONTROL
    assert inner["contents"][0]["parts"][0]["cache_control"] == CACHE_CONTROL

    gemini_model = PluginModelConfig(
        id="gemini-3.8-flash",
        name="Gemini 3.8 Flash",
        upstream_model="gemini-3.8-flash",
    )
    gemini_request = ChatCompletionRequest(
        model="gemini-3.8-flash",
        messages=[
            ChatMessage(
                role="user",
                content=[
                    {
                        "type": "text",
                        "text": "Stable user prefix",
                        "cache_control": CACHE_CONTROL,
                    }
                ],
            )
        ],
        extra_body={"cached_content": "cachedContents/gemini"},
    )
    _, gemini_body = plugin._build_upstream_request(
        gemini_request,
        gemini_model,
        account=None,
        actual_model="gemini-3.8-flash-high",
        enable_reasoning=False,
    )
    gemini_inner = gemini_body["request"]
    assert gemini_inner["cachedContent"] == "cachedContents/gemini"
    assert "cache_control" not in gemini_inner["contents"][0]["parts"][0]

    usage = plugin._build_usage(
        {
            "promptTokenCount": 500,
            "candidatesTokenCount": 100,
            "totalTokenCount": 600,
            "cachedContentTokenCount": 400,
        }
    )
    assert usage.prompt_tokens_details == {"cached_tokens": 400}
    assert usage.cache_read_input_tokens == 400


def test_openai_compatible_forwards_cache_hints():
    plugin = OpenAICompatiblePlugin()
    body = {}
    request = ChatCompletionRequest(
        model="default-model",
        messages=[ChatMessage(role="user", content="hi")],
        extra_body={"prompt_cache_key": "openai-cache-key", "prompt_cache_retention": "24h"},
    )
    plugin._apply_cache_fields(body, request)
    assert body["prompt_cache_key"] == "openai-cache-key"
    assert body["prompt_cache_retention"] == "24h"


def test_openai_compatible_stream_requests_usage(monkeypatch):
    captured = {}

    def fake_stream(
        url, headers, body, model_config, request, account=None, proxy=None
    ):
        captured["body"] = body

        async def empty():
            if False:  # pragma: no cover - keeps this an async generator
                yield None

        return empty()

    plugin = OpenAICompatiblePlugin()
    monkeypatch.setattr(plugin, "_stream_request", fake_stream)
    model = PluginModelConfig(
        id="default-model", name="Default", upstream_model="default-model"
    )
    request = ChatCompletionRequest(
        model="default-model",
        messages=[ChatMessage(role="user", content="hi")],
        stream=True,
    )
    asyncio.run(plugin.chat_completion(request, model, {"api_key": "sk-live-123"}))
    assert captured["body"]["stream_options"]["include_usage"] is True


def test_openai_compatible_request_body_contains_cache_hints(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "id": "chatcmpl-cache",
                "object": "chat.completion",
                "created": 1,
                "model": "default-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                    "prompt_tokens_details": {"cached_tokens": 8},
                },
            }

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, headers=None, json=None):
            captured["body"] = json
            return FakeResponse()

    monkeypatch.setattr(openai_module.httpx, "AsyncClient", FakeClient)

    plugin = OpenAICompatiblePlugin()
    model = PluginModelConfig(id="default-model", name="Default", upstream_model="default-model")
    request = ChatCompletionRequest(
        model="default-model",
        messages=[ChatMessage(role="user", content="hi")],
        extra_body={"prompt_cache_key": "body-cache-key"},
    )
    response = asyncio.run(
        plugin.chat_completion(request, model, {"api_key": "sk-live-123"})
    )
    assert captured["body"]["prompt_cache_key"] == "body-cache-key"
    assert response.usage.prompt_tokens_details == {"cached_tokens": 8}


def test_openrouter_preserves_block_and_top_level_cache_hints():
    plugin = OpenRouterPlugin()
    model = PluginModelConfig(
        id="nvidia/nemotron-3-ultra-550b-a55b:free",
        name="Nemotron 3 Ultra (free)",
        upstream_model="nvidia/nemotron-3-ultra-550b-a55b:free",
    )
    request = ChatCompletionRequest(
        model=model.id,
        messages=[
            ChatMessage(
                role="user",
                content=[
                    {"type": "text", "text": "stable", "cache_control": CACHE_CONTROL}
                ],
            )
        ],
        extra_body={"prompt_cache_key": "or-cache-key", "cache_control": CACHE_CONTROL},
    )
    body = plugin._build_body(request, model, {})
    assert body["prompt_cache_key"] == "or-cache-key"
    assert body["messages"][0]["content"][0]["cache_control"] == CACHE_CONTROL

    # A string message plus a top-level breakpoint is normalized to a block.
    string_request = ChatCompletionRequest(
        model=model.id,
        messages=[ChatMessage(role="user", content="stable prefix")],
        extra_body={"cache_control": CACHE_CONTROL},
    )
    string_body = plugin._build_body(string_request, model, {})
    assert string_body["messages"][0]["content"][0]["cache_control"] == CACHE_CONTROL


def test_opencode_free_forwards_cache_hints_and_blocks():
    plugin = OpenCodeFreePlugin()
    model = PluginModelConfig(
        id="opencode/free-model",
        name="Free Model",
        upstream_model="opencode/free-model",
    )
    request = ChatCompletionRequest(
        model=model.id,
        messages=[
            ChatMessage(
                role="user",
                content=[
                    {"type": "text", "text": "stable", "cache_control": CACHE_CONTROL}
                ],
            )
        ],
        extra_body={"prompt_cache_key": "oc-cache-key"},
    )
    payload = plugin._build_upstream_payload(request, model)
    plugin._apply_cache_fields(payload, request)
    assert payload["prompt_cache_key"] == "oc-cache-key"
    assert payload["messages"][0]["content"][0]["cache_control"] == CACHE_CONTROL


def test_openai_style_usage_details_survive_validation():
    response = ChatCompletionResponse.model_validate(
        {
            "id": "chatcmpl-cache",
            "object": "chat.completion",
            "created": 1,
            "model": "openrouter-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 5,
                "total_tokens": 55,
                "prompt_tokens_details": {"cached_tokens": 40},
            },
        }
    )
    assert response.usage.prompt_tokens_details == {"cached_tokens": 40}
    assert response.usage.cache_read_input_tokens == 40


def test_mock_provider_reports_simulated_cache_hits():
    plugin = MockProviderPlugin()
    model = plugin.metadata.models[0]
    request = ChatCompletionRequest(
        model=model.id,
        messages=[
            ChatMessage(
                role="user",
                content=[
                    {
                        "type": "text",
                        "text": "cache this prompt",
                        "cache_control": CACHE_CONTROL,
                    }
                ],
            )
        ],
    )
    response = asyncio.run(plugin.chat_completion(request, model, {}))
    assert "Context cache" in response.choices[0].message.content
    assert response.usage.prompt_tokens_details["cached_tokens"] > 0
    assert response.usage.cache_read_input_tokens > 0
