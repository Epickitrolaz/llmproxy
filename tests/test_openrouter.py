"""Tests for the OpenRouter provider template."""

from __future__ import annotations

import asyncio

import pytest

from llmproxy.schemas import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    PluginModelConfig,
)
from templates import openrouter as openrouter_module
from templates.openrouter import OpenRouterPlugin


FREE_MODEL_IDS = {
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "baidu/ernie-5.5-flash-preview:free",
    "thinkingmachines/inkling-2-small:free",
    "nvidia/nemotron-3.5-super-vl:free",
    "google/gemma-4-31b-it:free",
    "cohere/north-mini-code:free",
}


def _model(model_id: str = "nvidia/nemotron-3-ultra-550b-a55b:free") -> PluginModelConfig:
    plugin = OpenRouterPlugin()
    return next(m for m in plugin.metadata.models if m.id == model_id)


def test_metadata_preloads_free_models():
    plugin = OpenRouterPlugin()
    assert plugin.metadata.id == "openrouter"
    model_ids = {m.id for m in plugin.metadata.models}
    assert model_ids == FREE_MODEL_IDS
    assert all(model_id.endswith(":free") for model_id in model_ids)

    variables = {v.name: v for v in plugin.metadata.variables}
    assert variables["base_url"].default == "https://openrouter.ai/api/v1"
    assert variables["reasoning_effort"].options == ["default", "none", "low", "medium", "high"]


def test_body_preserves_free_suffix_and_maps_reasoning():
    plugin = OpenRouterPlugin()
    request = ChatCompletionRequest(
        model="nvidia/nemotron-3-ultra-550b-a55b:free",
        messages=[ChatMessage(role="user", content="hi")],
        temperature=0.4,
        max_tokens=1024,
        reasoning_effort="high",
    )
    body = plugin._build_body(request, _model(), {"reasoning_effort": "low"})
    assert body["model"] == "nvidia/nemotron-3-ultra-550b-a55b:free"
    assert body["reasoning"] == {"effort": "high"}
    assert body["temperature"] == 0.4
    assert body["max_tokens"] == 1024

    request_none = ChatCompletionRequest(
        model="google/gemma-4-31b-it:free",
        messages=[ChatMessage(role="user", content="hi")],
        reasoning_effort="none",
    )
    body_none = plugin._build_body(request_none, _model("google/gemma-4-31b-it:free"), {"reasoning_effort": "high"})
    assert "reasoning" not in body_none
    assert body_none["model"] == "google/gemma-4-31b-it:free"


def test_headers_include_attribution():
    plugin = OpenRouterPlugin()
    headers = plugin._build_headers(
        "sk-or-test",
        {"site_url": "https://example.com", "app_name": "LLMProxy Test"},
    )
    assert headers["Authorization"] == "Bearer sk-or-test"
    assert headers["HTTP-Referer"] == "https://example.com"
    assert headers["X-Title"] == "LLMProxy Test"


def test_test_mode_returns_simulated_response_and_stream():
    plugin = OpenRouterPlugin()
    model = _model()
    request = ChatCompletionRequest(
        model=model.id,
        messages=[ChatMessage(role="user", content="hi")],
        stream=False,
    )
    response = asyncio.run(plugin.chat_completion(request, model, {"api_key": "test-key"}))
    assert isinstance(response, ChatCompletionResponse)
    assert "OpenRouter" in response.choices[0].message.content

    stream_request = ChatCompletionRequest(
        model=model.id,
        messages=[ChatMessage(role="user", content="hi")],
        stream=True,
    )

    async def collect():
        stream = await plugin.chat_completion(stream_request, model, {"api_key": "test-key"})
        return [chunk async for chunk in stream]

    chunks = asyncio.run(collect())
    assert chunks
    assert all(isinstance(chunk, ChatCompletionChunk) for chunk in chunks)
    assert chunks[-1].choices[0].finish_reason == "stop"


def test_real_openrouter_request_is_forwarded(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "id": "chatcmpl-or-1",
                "object": "chat.completion",
                "created": 1,
                "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello via OpenRouter."},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
            }

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, headers=None, json=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            return FakeResponse()

    monkeypatch.setattr(openrouter_module.httpx, "AsyncClient", FakeClient)

    plugin = OpenRouterPlugin()
    model = _model()
    request = ChatCompletionRequest(
        model=model.id,
        messages=[ChatMessage(role="user", content="hi")],
        reasoning_effort="high",
    )
    response = asyncio.run(
        plugin.chat_completion(
            request,
            model,
            {
                "api_key": "sk-or-live-123",
                "site_url": "https://example.com",
                "app_name": "LLMProxy",
            },
        )
    )

    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer sk-or-live-123"
    assert captured["headers"]["HTTP-Referer"] == "https://example.com"
    assert captured["json"]["model"] == model.id
    assert captured["json"]["reasoning"] == {"effort": "high"}
    assert response.choices[0].message.content == "Hello via OpenRouter."
    assert response.usage.total_tokens == 7
