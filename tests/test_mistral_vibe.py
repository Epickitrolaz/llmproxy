"""Tests for the Mistral Vibe provider template."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from llmproxy.api.router import _usage_cache_tokens
from llmproxy.schemas import (
    Account,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    PluginModelConfig,
)
from templates import mistral_vibe as mistral_module
from templates.mistral_vibe import (
    DEFAULT_BROWSER_API_URL,
    DEFAULT_CONSOLE_URL,
    MODEL_MEDIUM,
    MODEL_SMALL,
    MistralVibePlugin,
)

MEDIUM = MODEL_MEDIUM
SMALL = MODEL_SMALL


def _model(model_id: str = MEDIUM) -> PluginModelConfig:
    plugin = MistralVibePlugin()
    return next(model for model in plugin.metadata.models if model.id == model_id)


def _request(model_id: str = MEDIUM, **kwargs) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=model_id,
        messages=[ChatMessage(role="user", content="hi")],
        **kwargs,
    )


def _account(api_key: str = "vibe-real-key", **extra) -> Account:
    return Account(
        id=extra.pop("id", "acc_mistral_vibe_test"),
        provider_id="mistral_vibe",
        name=extra.pop("name", "Mistral Vibe"),
        api_key=api_key,
        **extra,
    )


class FakeResponse:
    def __init__(
        self,
        status_code: int = 200,
        payload: dict | None = None,
        *,
        headers: dict | None = None,
        text: str | None = None,
    ):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self.text = text if text is not None else json.dumps(payload or {})

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON")
        return self._payload


class FakeStream:
    def __init__(
        self,
        lines: list[str],
        *,
        status_code: int = 200,
        headers: dict | None = None,
        body: bytes = b"",
    ):
        self._lines = lines
        self.status_code = status_code
        self.headers = headers or {}
        self._body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aread(self):
        return self._body


class FakeClient:
    def __init__(self, handler, calls: list[dict]):
        self._handler = handler
        self.calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, url, headers=None, json=None, data=None):
        self.calls.append({"method": "POST", "url": url, "headers": headers, "json": json})
        return self._handler("POST", url, headers, json)

    async def get(self, url, headers=None):
        self.calls.append({"method": "GET", "url": url, "headers": headers})
        return self._handler("GET", url, headers, None)

    def stream(self, method, url, headers=None, json=None):
        self.calls.append(
            {"method": "STREAM", "url": url, "headers": headers, "json": json}
        )
        return self._handler("STREAM", url, headers, json)


def _install(monkeypatch, handler):
    calls: list[dict] = []
    monkeypatch.setattr(
        mistral_module.httpx,
        "AsyncClient",
        lambda *args, **kwargs: FakeClient(handler, calls),
    )
    return calls


def _chat_payload() -> dict:
    return {
        "id": "gen-vibe-1",
        "object": "chat.completion",
        "created": 1,
        "model": "mistral-vibe-cli-latest",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": [{"type": "text", "text": "plan the answer"}],
                        },
                        {"type": "text", "text": "Hello from Vibe."},
                    ],
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_tokens_details": {"cached_tokens": 40},
            "completion_tokens_details": {"reasoning_tokens": 7},
        },
    }


def _rate_headers() -> dict:
    return {
        "x-ratelimit-limit-requests": "10",
        "x-ratelimit-remaining-requests": "4",
        "x-ratelimit-reset-requests": "60s",
    }


def _whoami() -> dict:
    return {
        "plan_type": "chat",
        "plan_name": "FREE",
        "customer_id": "cust_123456",
        "organization_kind": "personal",
        "api_base": "https://api.mistral.ai",
    }


def test_metadata_exposes_vibe_models_and_oauth():
    plugin = MistralVibePlugin()
    meta = plugin.metadata
    assert meta.id == "mistral_vibe"
    assert meta.filename == "mistral_vibe.py"
    assert meta.oauth_login is True
    assert meta.oauth_local_import is True
    assert meta.supports_usage_meter is True

    models = {model.id: model for model in meta.models}
    assert set(models) == {MEDIUM, SMALL}
    for model in models.values():
        assert model.upstream_model
        assert not model.upstream_model.startswith("mv/")
        assert model.reasoning_levels == ["none", "high"]
        assert model.supports_vision is True
        assert model.context_window == 262_144
        assert model.max_tokens == 262_144

    assert models[MEDIUM].upstream_model == "mistral-vibe-cli-latest"
    assert models[SMALL].upstream_model == "mistral-vibe-cli-fast"
    assert not any(variable.name == "api_key" for variable in meta.variables)


def test_body_forwards_reasoning_tools_cache_and_drops_stream_options():
    plugin = MistralVibePlugin()
    request = _request(
        temperature=0.2,
        max_tokens=1024,
        reasoning_effort="medium",
        tools=[
            {
                "type": "function",
                "function": {"name": "lookup", "parameters": {"type": "object"}},
            }
        ],
        tool_choice="auto",
        parallel_tool_calls=True,
        stream=True,
        stream_options={"include_usage": True},
        extra_body={
            "prompt_cache_key": "session-42",
            "top_k": 5,
            "metadata": {"trace": "yes"},
            "reasoning_effort": "medium",
            "stream_options": {"include_usage": True},
        },
    )
    body = plugin._build_body(request, _model(), {})
    assert body["model"] == "mistral-vibe-cli-latest"
    assert body["reasoning_effort"] == "high"
    assert body["prompt_cache_key"] == "session-42"
    assert body["top_k"] == 5
    assert body["metadata"] == {"trace": "yes"}
    assert body["tools"][0]["function"]["name"] == "lookup"
    assert body["tool_choice"] == "auto"
    assert body["parallel_tool_calls"] is True
    assert "stream_options" not in body

    none_body = plugin._build_body(
        _request(reasoning_effort="off"), _model(), {}
    )
    assert none_body["reasoning_effort"] == "none"

    no_reasoning = _model().model_copy(update={"reasoning_levels": []})
    plain = plugin._build_body(_request(reasoning_effort="high"), no_reasoning, {})
    assert "reasoning_effort" not in plain

    default = plugin._build_body(_request(), _model(), {"reasoning_effort": "high"})
    assert default["reasoning_effort"] == "high"


def test_message_translation_images_and_reasoning():
    plugin = MistralVibePlugin()
    messages = [
        ChatMessage(role="system", content="system"),
        ChatMessage(
            role="user",
            content=[
                {"type": "text", "text": "describe"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AAAA"},
                },
            ],
        ),
        ChatMessage(
            role="assistant",
            content="answer",
            reasoning_content="thought",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "f", "arguments": "{}"},
                }
            ],
        ),
        ChatMessage(
            role="tool",
            content="tool result",
            tool_call_id="call_1",
            name="f",
        ),
    ]
    converted = plugin._convert_messages(messages)
    assert converted[0] == {"role": "system", "content": "system"}

    user_content = converted[1]["content"]
    assert user_content[0] == {"type": "text", "text": "describe"}
    assert user_content[1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,AAAA"},
    }

    assistant = converted[2]
    assert "reasoning_content" not in assistant
    assert assistant["content"][0] == {
        "type": "thinking",
        "thinking": [{"type": "text", "text": "thought"}],
    }
    assert assistant["content"][1] == {"type": "text", "text": "answer"}

    tool = converted[3]
    assert tool["role"] == "tool"
    assert tool["tool_call_id"] == "call_1"
    assert tool["name"] == "f"
    assert tool["content"] == "tool result"


def test_only_the_most_recent_eight_images_are_sent():
    plugin = MistralVibePlugin()
    images = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{index}"},
        }
        for index in range(10)
    ]
    messages = [ChatMessage(role="user", content=images)]
    converted = plugin._convert_messages(messages)
    parts = converted[0]["content"]
    kept = [part for part in parts if part.get("type") == "image_url"]
    dropped = [part for part in parts if part.get("type") == "text"]
    assert len(kept) == 8
    assert [part["image_url"]["url"] for part in kept] == [
        f"data:image/png;base64,{index}" for index in range(2, 10)
    ]
    assert len(dropped) == 2
    assert all("Image removed" in part["text"] for part in dropped)


def test_non_streaming_maps_reasoning_cache_and_rate_limits(monkeypatch):
    def handler(method, url, headers, payload):
        assert method == "POST"
        assert url == "https://api.mistral.ai/v1/chat/completions"
        return FakeResponse(200, _chat_payload(), headers=_rate_headers())

    calls = _install(monkeypatch, handler)
    plugin = MistralVibePlugin()
    account = _account()

    response = asyncio.run(
        plugin.chat_completion(_request(), _model(), {}, account=account)
    )

    assert isinstance(response, ChatCompletionResponse)
    message = response.choices[0].message
    assert message.content == "Hello from Vibe."
    assert message.reasoning_content == "plan the answer"
    assert response.choices[0].finish_reason == "stop"
    assert response.usage is not None
    assert response.usage.prompt_tokens == 100
    assert response.usage.completion_tokens == 20
    assert response.usage.total_tokens == 120
    assert response.usage.cache_read_input_tokens == 40
    assert response.usage.completion_tokens_details == {"reasoning_tokens": 7}
    assert _usage_cache_tokens(response.usage) == (40, 0)

    assert calls[0]["headers"]["Authorization"] == "Bearer vibe-real-key"
    assert account.extra_config["usage_limits"]["groups"][0]["buckets"]


def test_non_streaming_error_is_raised_not_simulated(monkeypatch):
    _install(
        monkeypatch,
        lambda method, url, headers, payload: FakeResponse(
            429, {"message": "rate limit exceeded"}
        ),
    )
    plugin = MistralVibePlugin()
    with pytest.raises(RuntimeError, match="429"):
        asyncio.run(
            plugin.chat_completion(_request(), _model(), {}, account=_account())
        )


def test_real_account_without_key_raises_and_default_still_simulates():
    plugin = MistralVibePlugin()
    real = Account(
        id="acc_real",
        provider_id="mistral_vibe",
        name="Real Vibe",
        api_key="",
    )
    with pytest.raises(ValueError, match="no API key configured"):
        asyncio.run(plugin.chat_completion(_request(), _model(), {}, account=real))

    default = Account(
        id="default-mistral_vibe",
        provider_id="mistral_vibe",
        name="Default",
        api_key="",
    )
    response = asyncio.run(
        plugin.chat_completion(_request(), _model(), {}, account=default)
    )
    assert isinstance(response, ChatCompletionResponse)
    assert "Simulated response from Mistral Vibe" in response.choices[0].message.content

    test_key = _account(api_key="test-key")
    response = asyncio.run(
        plugin.chat_completion(_request(), _model(), {}, account=test_key)
    )
    assert "Simulated response" in response.choices[0].message.content


def _stream_lines() -> list[str]:
    return [
        "data: "
        + json.dumps(
            {
                "id": "gen-stream",
                "model": "mistral-vibe-cli-latest",
                "created": 1,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": ""},
                        "finish_reason": None,
                    }
                ],
            }
        ),
        "data: "
        + json.dumps(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": [
                                {
                                    "type": "thinking",
                                    "thinking": [{"type": "text", "text": "plan"}],
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
        ),
        "data: "
        + json.dumps(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": [{"type": "text", "text": "Hi there"}]
                        },
                        "finish_reason": None,
                    }
                ]
            }
        ),
        "data: "
        + json.dumps(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "f", "arguments": "{}"},
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        ),
        "data: "
        + json.dumps(
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 50,
                    "completion_tokens": 10,
                    "total_tokens": 60,
                    "prompt_tokens_details": {"cached_tokens": 20},
                },
            }
        ),
        "data: [DONE]",
    ]


def test_streaming_reasoning_tool_calls_and_usage(monkeypatch):
    def handler(method, url, headers, payload):
        assert method == "STREAM"
        return FakeStream(_stream_lines(), headers=_rate_headers())

    calls = _install(monkeypatch, handler)
    plugin = MistralVibePlugin()
    account = _account()

    async def collect():
        stream = await plugin.chat_completion(
            _request(stream=True), _model(), {}, account=account
        )
        return [chunk async for chunk in stream]

    chunks = asyncio.run(collect())
    assert all(isinstance(chunk, ChatCompletionChunk) for chunk in chunks)

    first = chunks[0]
    assert first.choices[0].delta["role"] == "assistant"

    reasoning = "".join(
        chunk.choices[0].delta.get("reasoning_content", "")
        for chunk in chunks
        if chunk.choices
    )
    text = "".join(
        chunk.choices[0].delta.get("content", "")
        for chunk in chunks
        if chunk.choices
    )
    assert reasoning == "plan"
    assert text == "Hi there"
    assert any(
        chunk.choices and chunk.choices[0].delta.get("tool_calls")
        for chunk in chunks
    )
    assert any(
        chunk.choices and chunk.choices[0].finish_reason == "tool_calls"
        for chunk in chunks
    )

    usage = next(chunk.usage for chunk in chunks if chunk.usage is not None)
    assert usage.prompt_tokens == 50
    assert usage.completion_tokens == 10
    assert usage.cache_read_input_tokens == 20

    assert "stream_options" not in calls[0]["json"]
    assert account.extra_config["usage_limits"]["groups"][0]["buckets"]


def test_streaming_upstream_error_is_raised(monkeypatch):
    def handler(method, url, headers, payload):
        return FakeStream([], status_code=400, body=b'{"message":"bad request"}')

    _install(monkeypatch, handler)
    plugin = MistralVibePlugin()

    async def run():
        stream = await plugin.chat_completion(
            _request(stream=True), _model(), {}, account=_account()
        )
        async for _ in stream:
            pass

    with pytest.raises(RuntimeError, match="400"):
        asyncio.run(run())


def _oauth_handler(exchange_status: int = 200):
    state = {"polls": 0}

    def handler(method, url, headers, payload):
        if method == "POST" and url == f"{DEFAULT_BROWSER_API_URL}/vibe/sign-in":
            return FakeResponse(
                200,
                {
                    "process_id": "proc-1",
                    "sign_in_url": "https://console.mistral.ai/codestral/cli/authenticate?process_id=proc-1",
                    "poll_url": "https://console.mistral.ai/api/vibe/sign-in/poll/token-1",
                    "expires_at": "2099-01-01T00:00:00+00:00",
                },
            )
        if method == "GET" and "/vibe/sign-in/poll/" in url:
            state["polls"] += 1
            if state["polls"] == 1:
                return FakeResponse(200, {"status": "pending"})
            return FakeResponse(
                200, {"status": "completed", "exchange_token": "exchange-1"}
            )
        if method == "POST" and url.endswith("/vibe/sign-in/proc-1/exchange"):
            if exchange_status >= 400:
                return FakeResponse(
                    exchange_status,
                    {"error_code": 9004, "message": "not eligible"},
                )
            return FakeResponse(200, {"api_key": "vibe-oauth-key"})
        if method == "GET" and url.endswith("/api/vibe/whoami"):
            return FakeResponse(200, _whoami())
        raise AssertionError(f"unexpected request {method} {url}")

    return handler


def test_oauth_start_and_status_complete(monkeypatch):
    calls = _install(monkeypatch, _oauth_handler())
    plugin = MistralVibePlugin()

    flow = asyncio.run(plugin.oauth_start())
    assert flow["mode"] == "device"
    assert flow["verification_uri_complete"].startswith(
        "https://console.mistral.ai/codestral/cli/authenticate"
    )
    assert flow["flow_id"]
    assert "Free plan" in flow["instructions"]

    start_call = calls[0]
    assert start_call["json"]["code_challenge_method"] == "S256"
    assert start_call["json"]["code_challenge"]

    pending = asyncio.run(plugin.oauth_status(flow["flow_id"]))
    assert pending["status"] == "pending"

    complete = asyncio.run(plugin.oauth_status(flow["flow_id"]))
    assert complete["status"] == "complete"
    account = complete["account"]
    assert isinstance(account, Account)
    assert account.provider_id == "mistral_vibe"
    assert account.api_key == "vibe-oauth-key"
    assert account.name == "Mistral Vibe (Free) · 123456"
    assert account.extra_config["user_plan"] == "Free"
    assert account.extra_config["auth_type"] == "vibe-browser"

    again = asyncio.run(plugin.oauth_status(flow["flow_id"]))
    assert again["status"] == "complete"
    assert again["account"].api_key == "vibe-oauth-key"


def test_oauth_denied_and_unknown_flow(monkeypatch):
    def handler(method, url, headers, payload):
        if method == "POST" and url.endswith("/vibe/sign-in"):
            return FakeResponse(
                200,
                {
                    "process_id": "proc-2",
                    "sign_in_url": "https://console.mistral.ai/x",
                    "poll_url": "https://console.mistral.ai/api/vibe/sign-in/poll/token-2",
                    "expires_at": "2099-01-01T00:00:00+00:00",
                },
            )
        if method == "GET":
            return FakeResponse(200, {"status": "denied"})
        raise AssertionError(f"unexpected request {method} {url}")

    _install(monkeypatch, handler)
    plugin = MistralVibePlugin()
    flow = asyncio.run(plugin.oauth_start())
    denied = asyncio.run(plugin.oauth_status(flow["flow_id"]))
    assert denied["status"] == "error"
    assert "denied" in denied["error"].lower()

    unknown = asyncio.run(plugin.oauth_status("missing-flow"))
    assert unknown["status"] == "error"


def test_oauth_rejects_unexpected_host(monkeypatch):
    def handler(method, url, headers, payload):
        return FakeResponse(
            200,
            {
                "process_id": "proc-3",
                "sign_in_url": "https://evil.example/x",
                "poll_url": "https://console.mistral.ai/api/vibe/sign-in/poll/token-3",
                "expires_at": "2099-01-01T00:00:00+00:00",
            },
        )

    _install(monkeypatch, handler)
    plugin = MistralVibePlugin()
    with pytest.raises(RuntimeError, match="browser host"):
        asyncio.run(plugin.oauth_start())


def test_oauth_exchange_workspace_error(monkeypatch):
    _install(monkeypatch, _oauth_handler(exchange_status=403))
    plugin = MistralVibePlugin()
    flow = asyncio.run(plugin.oauth_start())
    asyncio.run(plugin.oauth_status(flow["flow_id"]))  # pending
    result = asyncio.run(plugin.oauth_status(flow["flow_id"]))
    assert result["status"] == "error"
    assert "workspace" in result["error"].lower()


def test_oauth_callback_is_not_used():
    plugin = MistralVibePlugin()
    result = asyncio.run(plugin.oauth_callback("flow", "http://localhost/callback"))
    assert result["status"] == "error"
    assert "browser poll" in result["error"]


def test_oauth_import_local_from_environment(monkeypatch):
    _install(
        monkeypatch,
        lambda method, url, headers, payload: FakeResponse(200, _whoami()),
    )
    monkeypatch.setenv("MISTRAL_API_KEY", "vibe-import-key")
    plugin = MistralVibePlugin()
    account = asyncio.run(plugin.oauth_import_local())
    assert account is not None
    assert account.api_key == "vibe-import-key"
    assert account.extra_config["imported_from"] == "MISTRAL_API_KEY environment variable"


def test_oauth_import_local_from_vibe_home(monkeypatch, tmp_path: Path):
    _install(
        monkeypatch,
        lambda method, url, headers, payload: FakeResponse(200, _whoami()),
    )
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.setenv("VIBE_HOME", str(tmp_path))
    (tmp_path / ".env").write_text(
        "export MISTRAL_API_KEY='vibe-file-key'\n", encoding="utf-8"
    )
    plugin = MistralVibePlugin()
    account = asyncio.run(plugin.oauth_import_local())
    assert account is not None
    assert account.api_key == "vibe-file-key"
    assert account.extra_config["imported_from"] == str(tmp_path / ".env")


def test_oauth_import_local_returns_none_without_credentials(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.setenv("VIBE_HOME", str(tmp_path))
    plugin = MistralVibePlugin()
    assert asyncio.run(plugin.oauth_import_local()) is None


def test_refresh_account_adopts_tenant_base_and_plan(monkeypatch):
    info = {
        **_whoami(),
        "api_base": "https://api.eu.mistral.ai",
    }
    _install(
        monkeypatch,
        lambda method, url, headers, payload: FakeResponse(200, info),
    )
    plugin = MistralVibePlugin()
    account = _account(base_url=None)
    changed = asyncio.run(plugin.refresh_account(account))
    assert changed is True
    assert account.base_url == "https://api.eu.mistral.ai/v1"
    assert account.extra_config["user_plan"] == "Free"
    assert asyncio.run(plugin.refresh_account(account)) is False
