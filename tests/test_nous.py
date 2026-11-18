"""Tests for the Nous Portal free-tier provider template."""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from llmproxy.schemas import (
    Account,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    PluginModelConfig,
)
from templates import nous as nous_module
from templates.nous import NousFreeTierChallenge, NousPortalPlugin


@pytest.fixture(autouse=True)
def _no_account_writes(monkeypatch):
    monkeypatch.setattr(NousPortalPlugin, "_persist_accounts", staticmethod(lambda: None))


def _model() -> PluginModelConfig:
    plugin = NousPortalPlugin()
    return next(m for m in plugin.metadata.models if m.id == "nous/welcome")


def _request(**kwargs) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="nous/welcome",
        messages=[ChatMessage(role="user", content="hi")],
        **kwargs,
    )


def _account(**extra) -> Account:
    return Account(
        id="acc_nous_test",
        provider_id="nous",
        name="Nous Free Tier",
        extra_config=dict(extra),
    )


class FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = "", headers=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text or json.dumps(self._payload)
        self.headers = headers or {}

    def json(self):
        return self._payload


class FakeClient:
    """Routes POSTs by URL path and records every call."""

    def __init__(self, handler, calls):
        self._handler = handler
        self.calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, url, headers=None, json=None):
        self.calls.append({"url": url, "headers": headers, "json": json})
        return self._handler(url, json)

    def stream(self, method, url, headers=None, json=None):
        self.calls.append({"url": url, "headers": headers, "json": json})
        return self._handler(url, json)


def _install(monkeypatch, handler):
    calls: list[dict] = []
    monkeypatch.setattr(
        nous_module.httpx,
        "AsyncClient",
        lambda *args, **kwargs: FakeClient(handler, calls),
    )
    return calls


def _portal_ok(url: str, payload: dict | None):
    if url.endswith("/api/anonymous/create"):
        return FakeResponse(
            201,
            {"user_id": "nas_user:test", "org_id": "nas_organisation:test", "token": "anon_test_123"},
        )
    if url.endswith("/api/anonymous/token"):
        return FakeResponse(
            200,
            {
                "access_token": "jwt-test-token",
                "token_type": "Bearer",
                "expires_in": 900,
                "inference_base_url": "https://welcome-api.nousresearch.com/v1",
            },
        )
    if url.endswith("/chat/completions"):
        return FakeResponse(
            200,
            {
                "id": "gen-test",
                "object": "chat.completion",
                "created": 1,
                "model": "nous/welcome",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello from the free tier."},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 14,
                    "completion_tokens": 8,
                    "total_tokens": 22,
                    "prompt_tokens_details": {"cached_tokens": 0},
                },
            },
        )
    raise AssertionError(f"unexpected URL {url}")


def test_metadata_exposes_only_the_free_tier_model():
    plugin = NousPortalPlugin()
    assert plugin.metadata.id == "nous"
    assert plugin.metadata.filename == "nous.py"
    assert plugin.metadata.supports_usage_meter is True
    assert [m.id for m in plugin.metadata.models] == ["nous/welcome"]
    model = plugin.metadata.models[0]
    assert model.upstream_model == "nous/welcome"
    assert model.reasoning_levels == ["low", "high", "max"]
    assert model.supports_vision is False

    variables = {v.name: v for v in plugin.metadata.variables}
    assert variables["portal_url"].default == "https://portal.nousresearch.com"
    assert variables["welcome_base_url"].default == "https://welcome-api.nousresearch.com/v1"
    assert "api_key" not in variables


def test_body_forwards_reasoning_tools_and_cache_hints():
    plugin = NousPortalPlugin()
    request = _request(
        temperature=0.3,
        max_tokens=2048,
        reasoning_effort="high",
        tools=[
            {
                "type": "function",
                "function": {"name": "lookup", "parameters": {"type": "object"}},
            }
        ],
        tool_choice="auto",
        extra_body={"prompt_cache_key": "hermes-session-1"},
    )
    body = plugin._build_body(request, _model(), {"reasoning_effort": "low"})
    assert body["model"] == "nous/welcome"
    assert body["reasoning_effort"] == "high"
    assert body["temperature"] == 0.3
    assert body["max_tokens"] == 2048
    assert body["tools"][0]["function"]["name"] == "lookup"
    assert body["tool_choice"] == "auto"
    assert body["prompt_cache_key"] == "hermes-session-1"


def test_body_drops_unsupported_reasoning_levels():
    plugin = NousPortalPlugin()
    body = plugin._build_body(_request(reasoning_effort="medium"), _model(), {})
    assert "reasoning_effort" not in body

    body = plugin._build_body(_request(reasoning_effort="max"), _model(), {})
    assert body["reasoning_effort"] == "max"


def test_stream_body_always_requests_usage():
    plugin = NousPortalPlugin()
    body = plugin._build_body(_request(stream=True), _model(), {})
    assert body["stream"] is True
    assert body["stream_options"]["include_usage"] is True


def test_no_account_uses_simulation():
    plugin = NousPortalPlugin()
    response = asyncio.run(plugin.chat_completion(_request(), _model(), {}))
    assert isinstance(response, ChatCompletionResponse)
    assert "Nous Portal Free Tier" in response.choices[0].message.content


def test_first_request_mints_exchanges_and_calls_welcome(monkeypatch):
    calls = _install(monkeypatch, _portal_ok)
    plugin = NousPortalPlugin()
    account = _account()

    response = asyncio.run(
        plugin.chat_completion(
            _request(reasoning_effort="low", max_tokens=100),
            _model(),
            {},
            account=account,
        )
    )

    assert [call["url"] for call in calls] == [
        "https://portal.nousresearch.com/api/anonymous/create",
        "https://portal.nousresearch.com/api/anonymous/token",
        "https://welcome-api.nousresearch.com/v1/chat/completions",
    ]
    assert calls[0]["headers"]["user-agent"].startswith("hermes-agent/")
    assert calls[1]["json"]["token"] == "anon_test_123"
    assert calls[1]["json"]["purpose"] == "inference"
    assert calls[2]["headers"]["Authorization"] == "Bearer jwt-test-token"
    assert calls[2]["json"]["model"] == "nous/welcome"
    assert calls[2]["json"]["reasoning_effort"] == "low"

    assert response.choices[0].message.content == "Hello from the free tier."
    assert response.usage.total_tokens == 22
    assert account.extra_config["anon_token"] == "anon_test_123"
    assert account.extra_config["access_token"] == "jwt-test-token"
    assert account.extra_config["access_token_expires_at"] > time.time()


def test_cached_token_skips_mint_and_exchange(monkeypatch):
    calls = _install(monkeypatch, _portal_ok)
    plugin = NousPortalPlugin()
    account = _account(
        anon_token="anon_test_123",
        access_token="jwt-cached",
        access_token_expires_at=time.time() + 600,
        inference_base_url="https://welcome-api.nousresearch.com/v1",
    )

    asyncio.run(plugin.chat_completion(_request(), _model(), {}, account=account))

    assert [call["url"] for call in calls] == [
        "https://welcome-api.nousresearch.com/v1/chat/completions"
    ]
    assert calls[0]["headers"]["Authorization"] == "Bearer jwt-cached"


def test_browser_challenge_raises_url_instead_of_simulating(monkeypatch):
    challenge_url = "https://portal.nousresearch.com/challenge?code=abc123"

    def handler(url, payload):
        if url.endswith("/api/anonymous/token"):
            return FakeResponse(
                428,
                {
                    "error": "challenge_required",
                    "message": "Nous needs to run a quick check.",
                    "challenges": [
                        {
                            "type": "browser",
                            "url": challenge_url,
                            "required": True,
                            "expires_in": 600,
                            "interval": 2,
                        }
                    ],
                },
            )
        raise AssertionError(f"unexpected URL {url}")

    _install(monkeypatch, handler)
    plugin = NousPortalPlugin()
    account = _account(anon_token="anon_test_123")

    with pytest.raises(NousFreeTierChallenge) as excinfo:
        asyncio.run(plugin.chat_completion(_request(), _model(), {}, account=account))

    assert challenge_url in str(excinfo.value)
    assert account.extra_config["challenge_url"] == challenge_url
    assert account.extra_config["challenge_expires_at"] > time.time()


def test_401_refreshes_token_once_and_retries(monkeypatch):
    exchanges = {"count": 0}

    def handler(url, payload):
        if url.endswith("/chat/completions"):
            if exchanges["count"] == 1:
                return FakeResponse(401, {"error": "expired token"}, text='{"error":"expired token"}')
            return _portal_ok(url, payload)
        if url.endswith("/api/anonymous/token"):
            exchanges["count"] += 1
            payload = {
                "access_token": f"jwt-{exchanges['count']}",
                "expires_in": 900,
                "inference_base_url": "https://welcome-api.nousresearch.com/v1",
            }
            return FakeResponse(200, payload)
        raise AssertionError(f"unexpected URL {url}")

    calls = _install(monkeypatch, handler)
    plugin = NousPortalPlugin()
    account = _account(anon_token="anon_test_123")

    response = asyncio.run(plugin.chat_completion(_request(), _model(), {}, account=account))

    assert response.usage.total_tokens == 22
    assert exchanges["count"] == 2
    chat_calls = [call for call in calls if call["url"].endswith("/chat/completions")]
    assert chat_calls[-1]["headers"]["Authorization"] == "Bearer jwt-2"


def test_dead_credential_is_replaced_once(monkeypatch):
    state = {"exchanges": 0, "mints": 0}

    def handler(url, payload):
        if url.endswith("/api/anonymous/create"):
            state["mints"] += 1
            return FakeResponse(
                201,
                {"user_id": "nas_user:new", "org_id": "nas_organisation:new", "token": "anon_new"},
            )
        if url.endswith("/api/anonymous/token"):
            state["exchanges"] += 1
            if state["exchanges"] == 1:
                return FakeResponse(404, {"error": "unknown_token"})
            return FakeResponse(
                200,
                {
                    "access_token": "jwt-new",
                    "expires_in": 900,
                    "inference_base_url": "https://welcome-api.nousresearch.com/v1",
                },
            )
        return _portal_ok(url, payload)

    calls = _install(monkeypatch, handler)
    plugin = NousPortalPlugin()
    account = _account(anon_token="anon_dead")

    response = asyncio.run(plugin.chat_completion(_request(), _model(), {}, account=account))

    assert response.usage.total_tokens == 22
    assert state == {"exchanges": 2, "mints": 1}
    assert account.extra_config["anon_token"] == "anon_new"
    assert account.extra_config["access_token"] == "jwt-new"
    assert [call["url"] for call in calls] == [
        "https://portal.nousresearch.com/api/anonymous/token",
        "https://portal.nousresearch.com/api/anonymous/create",
        "https://portal.nousresearch.com/api/anonymous/token",
        "https://welcome-api.nousresearch.com/v1/chat/completions",
    ]


def test_upstream_rate_limit_propagates(monkeypatch):
    def handler(url, payload):
        if url.endswith("/chat/completions"):
            return FakeResponse(429, {"error": "rate limit"}, text='{"error":"rate limit"}')
        return _portal_ok(url, payload)

    _install(monkeypatch, handler)
    plugin = NousPortalPlugin()
    account = _account(
        anon_token="anon_test_123",
        access_token="jwt-cached",
        access_token_expires_at=time.time() + 600,
        inference_base_url="https://welcome-api.nousresearch.com/v1",
    )

    with pytest.raises(RuntimeError, match="429"):
        asyncio.run(plugin.chat_completion(_request(), _model(), {}, account=account))


def test_streaming_forwards_chunks_and_usage(monkeypatch):
    payloads = [
        {
            "id": "gen-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "nous/welcome",
            "choices": [{"index": 0, "delta": {"content": "Hi", "role": "assistant"}, "finish_reason": None}],
        },
        {
            "id": "gen-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "nous/welcome",
            "choices": [{"index": 0, "delta": {"content": " there"}, "finish_reason": None}],
        },
        {
            "id": "gen-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "nous/welcome",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 2,
                "total_tokens": 14,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
        },
    ]

    class FakeStreamResponse:
        status_code = 200
        headers = {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def aiter_lines(self):
            for payload in payloads:
                yield f"data: {json.dumps(payload)}"
            yield "data: [DONE]"

    class StreamClient(FakeClient):
        def stream(self, method, url, headers=None, json=None):
            self.calls.append({"url": url, "headers": headers, "json": json})
            return FakeStreamResponse()

    calls: list[dict] = []
    monkeypatch.setattr(
        nous_module.httpx,
        "AsyncClient",
        lambda *args, **kwargs: StreamClient(_portal_ok, calls),
    )

    plugin = NousPortalPlugin()
    account = _account(
        anon_token="anon_test_123",
        access_token="jwt-cached",
        access_token_expires_at=time.time() + 600,
        inference_base_url="https://welcome-api.nousresearch.com/v1",
    )

    async def collect():
        stream = await plugin.chat_completion(_request(stream=True), _model(), {}, account=account)
        return [chunk async for chunk in stream]

    chunks = asyncio.run(collect())
    assert all(isinstance(chunk, ChatCompletionChunk) for chunk in chunks)
    assert chunks[0].choices[0].delta["role"] == "assistant"
    assert chunks[-1].choices[0].finish_reason == "stop"
    assert chunks[-1].usage.total_tokens == 14
