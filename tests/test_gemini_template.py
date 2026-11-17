"""Tests for the built-in Google Gemini provider template."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from llmproxy.core.manager import plugin_manager
from llmproxy.dashboard.loader import load_plugin_from_file
from llmproxy.main import app
from llmproxy.schemas import (
    Account,
    AccountCreate,
    ChatCompletionRequest,
    ChatMessage,
    PluginModelConfig,
)
from templates.gemini import GeminiPlugin

ROOT = Path(__file__).resolve().parent.parent
ADMIN_HEADERS = {"X-Admin-Token": "test-admin-token"}


def _model(model_id: str = "gm/gemini-3.8-flash") -> PluginModelConfig:
    return PluginModelConfig(
        id=model_id,
        name="Gemini Test",
        upstream_model=model_id.split("/", 1)[1],
        max_tokens=4096,
    )


def test_gemini_template_metadata_and_prefixes():
    plugin = load_plugin_from_file(ROOT / "templates" / "gemini.py")
    assert plugin.metadata.id == "gemini"
    assert plugin.metadata.models
    for model in plugin.metadata.models:
        assert model.id.startswith("gm/")
        assert model.upstream_model
        assert not model.upstream_model.startswith("gm/")
        assert model.supports_vision

    by_id = {model.id: model for model in plugin.metadata.models}
    assert by_id["gm/gemini-4-argon"].reasoning_levels == ["low", "medium", "high"]
    assert by_id["gm/gemini-3.8-flash"].reasoning_levels == [
        "minimal",
        "low",
        "medium",
        "high",
    ]
    assert by_id["gm/gemini-2.5-flash"].reasoning_levels == [
        "none",
        "low",
        "medium",
        "high",
    ]
    assert by_id["gm/gemma-4-31b-it"].reasoning_levels == []


def test_gemini_payload_translates_messages_tools_cache_and_thinking():
    plugin = GeminiPlugin()
    request = ChatCompletionRequest(
        model="gm/gemini-3.8-flash",
        messages=[
            ChatMessage(role="system", content="Stable system prompt"),
            ChatMessage(role="user", content="What is this?"),
            ChatMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"q": "x"}'},
                    }
                ],
            ),
            ChatMessage(
                role="tool",
                tool_call_id="call_1",
                name="lookup",
                content='{"answer": 42}',
            ),
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look something up",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "q": {"type": "string", "enum": ["x", "y"]},
                        },
                        "required": ["q", "missing"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        tool_choice="auto",
        reasoning_effort="medium",
        extra_body={"cached_content": "cachedContents/abc"},
    )

    payload, name_map = plugin._build_payload(request, _model(), {})
    assert payload["cachedContent"] == "cachedContents/abc"
    assert payload["systemInstruction"] == {
        "role": "user",
        "parts": [{"text": "Stable system prompt"}],
    }
    assert payload["generationConfig"]["thinkingConfig"] == {
        "thinkingLevel": "medium",
        "includeThoughts": True,
    }
    assert payload["tools"][0]["functionDeclarations"][0]["name"] == "lookup"
    parameters = payload["tools"][0]["functionDeclarations"][0]["parameters"]
    assert parameters["type"] == "OBJECT"
    assert parameters["required"] == ["q"]
    assert "additionalProperties" not in parameters
    assert payload["toolConfig"]["functionCallingConfig"]["mode"] == "AUTO"

    contents = payload["contents"]
    assert contents[0]["role"] == "user"
    assert contents[0]["parts"] == [{"text": "What is this?"}]
    assert contents[1]["role"] == "model"
    function_call = contents[1]["parts"][0]["functionCall"]
    assert function_call["name"] == "lookup"
    assert function_call["args"] == {"q": "x"}
    function_response = contents[2]["parts"][0]["functionResponse"]
    assert function_response["id"] == function_call["id"]
    assert function_response["response"] == {"result": {"answer": 42}}
    assert name_map == {"lookup": "lookup"}


def test_gemini_url_and_auth_styles():
    plugin = GeminiPlugin()
    variables = {"base_url": "https://generativelanguage.googleapis.com", "api_version": "v1beta"}

    assert plugin._build_url(variables, None, "gemini-3.8-flash", False, "AIza-key") == (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-3.8-flash:generateContent"
    )
    stream_url = plugin._build_url(variables, None, "gemini-3.8-flash", True, "AIza-key")
    assert stream_url.endswith(":streamGenerateContent?alt=sse")

    query_url = plugin._build_url(
        {**variables, "auth_style": "query"}, None, "gemini-3.8-flash", False, "a b"
    )
    assert query_url.endswith(":generateContent?key=a%20b")

    assert plugin._build_headers("AIza-key", {}, None, False) == {
        "Content-Type": "application/json",
        "x-goog-api-key": "AIza-key",
    }
    assert plugin._build_headers("ya29.token", {}, None, False) == {
        "Content-Type": "application/json",
        "Authorization": "Bearer ya29.token",
    }


def test_gemini_usage_maps_cache_and_reasoning_tokens():
    usage = GeminiPlugin()._build_usage(
        {
            "promptTokenCount": 1000,
            "candidatesTokenCount": 100,
            "thoughtsTokenCount": 20,
            "totalTokenCount": 1120,
            "cachedContentTokenCount": 800,
        }
    )
    assert usage.prompt_tokens == 1000
    assert usage.completion_tokens == 120
    assert usage.total_tokens == 1120
    assert usage.cache_read_input_tokens == 800
    assert usage.prompt_tokens_details == {"cached_tokens": 800}
    assert usage.completion_tokens_details == {"reasoning_tokens": 20}


def test_gemini_non_streaming_uses_usage_metadata(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "responseId": "resp-1",
                "modelVersion": "gemini-3.8-flash",
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [
                                {"text": "thinking", "thought": True},
                                {"text": "Hello there"},
                            ],
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 1000,
                    "candidatesTokenCount": 20,
                    "totalTokenCount": 1020,
                    "cachedContentTokenCount": 800,
                    "thoughtsTokenCount": 5,
                },
            },
        )

    real_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs.pop("proxy", None)
        return real_client(*args, transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)

    plugin = GeminiPlugin()
    request = ChatCompletionRequest(
        model="gm/gemini-3.8-flash",
        messages=[ChatMessage(role="user", content="hi")],
    )
    account = Account(
        id="acc_gm", provider_id="gemini", name="Gemini", api_key="AIza-real-key"
    )
    response = asyncio.run(
        plugin.chat_completion(request, _model(), {"timeout": 5}, account)
    )

    assert captured["url"] == (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-3.8-flash:generateContent"
    )
    assert captured["headers"]["x-goog-api-key"] == "AIza-real-key"
    assert captured["body"]["contents"][0]["parts"] == [{"text": "hi"}]
    assert response.choices[0].message.content == "Hello there"
    assert response.choices[0].message.reasoning_content == "thinking"
    assert response.choices[0].finish_reason == "stop"
    assert response.usage.prompt_tokens == 1000
    assert response.usage.completion_tokens == 25
    assert response.usage.cache_read_input_tokens == 800


def test_gemini_streaming_translates_sse_with_usage(monkeypatch):
    sse = "\n\n".join(
        [
            'data: {"candidates":[{"content":{"role":"model","parts":'
            '[{"text":"Hello"}]}}]}',
            'data: {"candidates":[{"content":{"role":"model","parts":'
            '[{"text":" world"}]},"finishReason":"STOP"}],'
            '"usageMetadata":{"promptTokenCount":1000,"candidatesTokenCount":20,'
            '"totalTokenCount":1020,"cachedContentTokenCount":800}}',
            "data: [DONE]",
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Accept"] == "text/event-stream"
        return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})

    real_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs.pop("proxy", None)
        return real_client(*args, transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)

    plugin = GeminiPlugin()
    request = ChatCompletionRequest(
        model="gm/gemini-3.8-flash",
        messages=[ChatMessage(role="user", content="hi")],
        stream=True,
    )
    account = Account(
        id="acc_gm", provider_id="gemini", name="Gemini", api_key="AIza-real-key"
    )

    async def collect():
        iterator = await plugin.chat_completion(
            request, _model(), {"timeout": 5}, account
        )
        return [chunk async for chunk in iterator]

    chunks = asyncio.run(collect())
    assert chunks[0].choices[0].delta == {"role": "assistant", "content": ""}
    assert "".join(
        chunk.choices[0].delta.get("content", "") for chunk in chunks[:-1]
    ) == "Hello world"
    assert chunks[-1].choices[0].finish_reason == "stop"
    assert chunks[-1].usage.cache_read_input_tokens == 800
    assert chunks[-1].usage.completion_tokens == 20


def test_gemini_streaming_translates_tool_calls(monkeypatch):
    sse = 'data: {"candidates":[{"content":{"role":"model","parts":[' \
        '{"functionCall":{"name":"lookup","args":{"q":"x"},"id":"call_1"},' \
        '"thoughtSignature":"sig-1"},' \
        '{"functionCall":{"name":"lookup","args":{"q":"y"},"id":"call_2"}}]}}]}\n\n' \
        'data: {"candidates":[{"finishReason":"STOP"}],' \
        '"usageMetadata":{"promptTokenCount":12,"candidatesTokenCount":4,' \
        '"totalTokenCount":16}}'

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})

    real_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs.pop("proxy", None)
        return real_client(*args, transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)

    plugin = GeminiPlugin()
    request = ChatCompletionRequest(
        model="gm/gemini-3.8-flash",
        messages=[ChatMessage(role="user", content="hi")],
        stream=True,
    )
    account = Account(
        id="acc_gm", provider_id="gemini", name="Gemini", api_key="AIza-real-key"
    )

    async def collect():
        iterator = await plugin.chat_completion(
            request, _model(), {"timeout": 5}, account
        )
        return [chunk async for chunk in iterator]

    chunks = asyncio.run(collect())
    tool_chunks = [
        chunk
        for chunk in chunks
        if chunk.choices and chunk.choices[0].delta.get("tool_calls")
    ]
    assert [chunk.choices[0].delta["tool_calls"][0]["index"] for chunk in tool_chunks] == [0, 1]
    assert tool_chunks[0].choices[0].delta["tool_calls"][0]["id"] == "call_1"
    assert chunks[-1].choices[0].finish_reason == "tool_calls"
    assert chunks[-1].usage.prompt_tokens == 12


@pytest.fixture
def gemini_state():
    original_logs = list(plugin_manager.logs)
    original_accounts = dict(plugin_manager.accounts)
    original_configs = json.loads(json.dumps(plugin_manager.configs))
    plugin_manager.logs.clear()
    yield
    plugin_manager.logs = original_logs
    plugin_manager.accounts = original_accounts
    plugin_manager.configs = original_configs
    plugin_manager._save_accounts()


def test_gemini_usage_reaches_dashboard_cache_stats(monkeypatch, gemini_state):
    plugin_manager.discover_and_load_all()
    plugin = plugin_manager.get_plugin("gemini")
    assert plugin is not None, "gemini template must be available to the plugin manager"

    plugin_manager.update_plugin_config(
        "gemini",
        enabled=True,
        variables={"base_url": "https://generativelanguage.googleapis.com"},
    )
    account = plugin_manager.add_account(
        AccountCreate(
            provider_id="gemini", name="Gemini Cache Test", api_key="AIza-real-key"
        )
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [{"text": "OK"}],
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 1000,
                    "candidatesTokenCount": 20,
                    "totalTokenCount": 1020,
                    "cachedContentTokenCount": 800,
                },
            },
        )

    real_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs.pop("proxy", None)
        return real_client(*args, transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)

    client = TestClient(app, headers=ADMIN_HEADERS)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gm/gemini-3.8-flash",
            "messages": [{"role": "user", "content": "ping"}],
        },
    )
    assert response.status_code == 200
    assert response.json()["usage"]["cache_read_input_tokens"] == 800

    log = client.get("/api/logs").json()[0]
    assert log["plugin_id"] == "gemini"
    assert log["prompt_tokens"] == 1000
    assert log["cache_read_input_tokens"] == 800

    stats = client.get("/api/stats").json()
    assert stats["cache_read_input_tokens"] == 800
    assert stats["cache_hit_rate"] == 80.0

    plugin_manager.delete_account(account.id)
