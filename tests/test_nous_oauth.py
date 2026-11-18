"""Tests for the Nous Portal OAuth provider template."""

from __future__ import annotations

import asyncio
import base64
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
from templates import nous_oauth as nous_module
from templates.nous_oauth import NousOAuthLoginRequired, NousOAuthPlugin

FREE_MODEL_IDS = {
    "np/upstage/solar-mini4:free",
    "np/tencent/hy4-flash:free",
    "np/stepfun/step-3.7-flash:free",
    "np/qwen/qwen4-max:free",
    "np/meituan/longcat-2.5:free",
}


@pytest.fixture(autouse=True)
def _no_account_writes(monkeypatch):
    monkeypatch.setattr(NousOAuthPlugin, "_persist_accounts", staticmethod(lambda: None))


def _jwt(*, ttl: int = 900, scope: str = "inference:invoke") -> str:
    def encode(data: dict) -> str:
        raw = json.dumps(data).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    header = encode({"alg": "RS256", "typ": "JWT"})
    claims = encode({"exp": int(time.time()) + ttl, "scope": scope, "aud": "hermes-cli"})
    return f"{header}.{claims}.signature"


def _model(model_id: str = "np/qwen/qwen4-max:free") -> PluginModelConfig:
    plugin = NousOAuthPlugin()
    return next(m for m in plugin.metadata.models if m.id == model_id)


def _request(model_id: str = "np/qwen/qwen4-max:free", **kwargs) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=model_id,
        messages=[ChatMessage(role="user", content="hi")],
        **kwargs,
    )


def _account(**extra) -> Account:
    config = {"refresh_token": "rt_initial"}
    config.update(extra)
    return Account(
        id="acc_nous_oauth_test",
        provider_id="nous_oauth",
        name="Nous Portal (OAuth)",
        extra_config=config,
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
    def __init__(self, handler, calls):
        self._handler = handler
        self.calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, url, headers=None, data=None, json=None):
        self.calls.append({"url": url, "headers": headers, "data": data, "json": json})
        return self._handler(url, data, json)

    def stream(self, method, url, headers=None, json=None):
        self.calls.append({"url": url, "headers": headers, "json": json})
        return self._handler(url, None, json)


def _install(monkeypatch, handler):
    calls: list[dict] = []
    monkeypatch.setattr(
        nous_module.httpx,
        "AsyncClient",
        lambda *args, **kwargs: FakeClient(handler, calls),
    )
    return calls


def _refresh_payload(token: str, refresh_token: str = "rt_rotated") -> dict:
    return {
        "access_token": token,
        "refresh_token": refresh_token,
        "token_type": "Bearer",
        "expires_in": 900,
        "scope": "inference:invoke",
        "inference_base_url": "https://inference-api.nousresearch.com/v1",
    }


def _chat_payload() -> dict:
    return {
        "id": "gen-test",
        "object": "chat.completion",
        "created": 1,
        "model": "qwen/qwen4-max:free",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello from the plan."},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_tokens": 18,
            "prompt_tokens_details": {"cached_tokens": 5},
        },
    }


def test_metadata_exposes_free_models_and_prefix():
    plugin = NousOAuthPlugin()
    assert plugin.metadata.id == "nous_oauth"
    assert plugin.metadata.filename == "nous_oauth.py"
    assert {m.id for m in plugin.metadata.models} == FREE_MODEL_IDS
    assert all(m.upstream_model and not m.upstream_model.startswith("np/") for m in plugin.metadata.models)
    assert all(m.id.endswith(":free") for m in plugin.metadata.models)

    variables = {v.name: v for v in plugin.metadata.variables}
    assert variables["portal_url"].default == "https://portal.nousresearch.com"
    assert variables["base_url"].default == "https://inference-api.nousresearch.com/v1"
    assert variables["client_id"].default == "hermes-cli"
    assert "api_key" not in variables


def test_body_forwards_reasoning_tools_and_cache_hints():
    plugin = NousOAuthPlugin()
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
        extra_body={"prompt_cache_key": "session-1"},
    )
    body = plugin._build_body(request, _model(), {"reasoning_effort": "low"})
    assert body["model"] == "qwen/qwen4-max:free"
    assert body["reasoning_effort"] == "high"
    assert body["temperature"] == 0.3
    assert body["max_tokens"] == 2048
    assert body["tools"][0]["function"]["name"] == "lookup"
    assert body["tool_choice"] == "auto"
    assert body["prompt_cache_key"] == "session-1"


def test_body_drops_unsupported_reasoning_levels():
    plugin = NousOAuthPlugin()
    body = plugin._build_body(_request(reasoning_effort="medium"), _model(), {})
    assert "reasoning_effort" not in body

    body = plugin._build_body(
        _request("np/upstage/solar-mini4:free", reasoning_effort="max"),
        _model("np/upstage/solar-mini4:free"),
        {},
    )
    assert body["reasoning_effort"] == "max"


def test_no_account_uses_simulation():
    plugin = NousOAuthPlugin()
    response = asyncio.run(plugin.chat_completion(_request(), _model(), {}))
    assert isinstance(response, ChatCompletionResponse)
    assert "Nous Portal OAuth" in response.choices[0].message.content


def test_refresh_rotates_refresh_token_and_calls_inference(monkeypatch):
    token = _jwt()

    def handler(url, data, payload):
        if url.endswith("/api/oauth/token"):
            return FakeResponse(200, _refresh_payload(token))
        if url.endswith("/chat/completions"):
            return FakeResponse(200, _chat_payload())
        raise AssertionError(f"unexpected URL {url}")

    calls = _install(monkeypatch, handler)
    plugin = NousOAuthPlugin()
    account = _account()

    response = asyncio.run(
        plugin.chat_completion(_request(reasoning_effort="high"), _model(), {}, account=account)
    )

    assert [call["url"] for call in calls] == [
        "https://portal.nousresearch.com/api/oauth/token",
        "https://inference-api.nousresearch.com/v1/chat/completions",
    ]
    assert calls[0]["headers"]["x-nous-refresh-token"] == "rt_initial"
    assert calls[0]["data"] == {"grant_type": "refresh_token", "client_id": "hermes-cli"}
    assert calls[1]["headers"]["Authorization"] == f"Bearer {token}"
    assert calls[1]["json"]["model"] == "qwen/qwen4-max:free"
    assert calls[1]["json"]["reasoning_effort"] == "high"

    assert response.choices[0].message.content == "Hello from the plan."
    assert response.usage.total_tokens == 18
    assert response.usage.cache_read_input_tokens == 5
    assert account.extra_config["refresh_token"] == "rt_rotated"
    assert account.extra_config["access_token"] == token


def test_cached_inference_jwt_skips_refresh(monkeypatch):
    token = _jwt(ttl=600)

    def handler(url, data, payload):
        if url.endswith("/chat/completions"):
            return FakeResponse(200, _chat_payload())
        raise AssertionError(f"unexpected URL {url}")

    calls = _install(monkeypatch, handler)
    plugin = NousOAuthPlugin()
    account = _account(
        access_token=token,
        access_token_expires_at=time.time() + 600,
        inference_base_url="https://inference-api.nousresearch.com/v1",
    )

    asyncio.run(plugin.chat_completion(_request(), _model(), {}, account=account))

    assert [call["url"] for call in calls] == [
        "https://inference-api.nousresearch.com/v1/chat/completions"
    ]
    assert calls[0]["headers"]["Authorization"] == f"Bearer {token}"


def test_401_refreshes_once_and_retries(monkeypatch):
    state = {"refreshes": 0, "chats": 0}

    def handler(url, data, payload):
        if url.endswith("/api/oauth/token"):
            state["refreshes"] += 1
            return FakeResponse(200, _refresh_payload(_jwt(), refresh_token=f"rt_{state['refreshes']}"))
        if url.endswith("/chat/completions"):
            state["chats"] += 1
            if state["chats"] == 1:
                return FakeResponse(401, {"error": "expired"}, text='{"error":"expired"}')
            return FakeResponse(200, _chat_payload())
        raise AssertionError(f"unexpected URL {url}")

    calls = _install(monkeypatch, handler)
    plugin = NousOAuthPlugin()
    account = _account(
        access_token=_jwt(ttl=600),
        access_token_expires_at=time.time() + 600,
        inference_base_url="https://inference-api.nousresearch.com/v1",
    )

    response = asyncio.run(plugin.chat_completion(_request(), _model(), {}, account=account))

    assert response.usage.total_tokens == 18
    assert state == {"refreshes": 1, "chats": 2}
    assert account.extra_config["refresh_token"] == "rt_1"


def test_invalid_grant_requires_a_new_login(monkeypatch):
    def handler(url, data, payload):
        return FakeResponse(400, {"error": "invalid_grant", "error_description": "expired"})

    _install(monkeypatch, handler)
    plugin = NousOAuthPlugin()
    account = _account()

    with pytest.raises(NousOAuthLoginRequired, match="login helper"):
        asyncio.run(plugin.chat_completion(_request(), _model(), {}, account=account))
    assert account.extra_config["refresh_token"] is None
    assert account.extra_config["needs_relogin"] is True


def test_edge_challenge_keeps_refresh_token(monkeypatch):
    def handler(url, data, payload):
        return FakeResponse(
            403,
            {"error": "Forbidden"},
            text="<!DOCTYPE html><title>Vercel Security Checkpoint</title>",
            headers={"x-vercel-mitigated": "challenge"},
        )

    _install(monkeypatch, handler)
    plugin = NousOAuthPlugin()
    account = _account()

    with pytest.raises(RuntimeError, match="edge challenge"):
        asyncio.run(plugin.chat_completion(_request(), _model(), {}, account=account))
    assert account.extra_config["refresh_token"] == "rt_initial"
    assert account.extra_config["needs_relogin"] is not True


def test_missing_refresh_token_requires_login():
    plugin = NousOAuthPlugin()
    account = _account(refresh_token=None)

    with pytest.raises(NousOAuthLoginRequired, match="no refresh token"):
        asyncio.run(plugin.chat_completion(_request(), _model(), {}, account=account))


def test_upstream_rate_limit_propagates(monkeypatch):
    token = _jwt()

    def handler(url, data, payload):
        if url.endswith("/chat/completions"):
            return FakeResponse(429, {"error": "rate limit"}, text='{"error":"rate limit"}')
        return FakeResponse(200, _refresh_payload(token))

    _install(monkeypatch, handler)
    plugin = NousOAuthPlugin()
    account = _account()

    with pytest.raises(RuntimeError, match="429"):
        asyncio.run(plugin.chat_completion(_request(), _model(), {}, account=account))


def test_streaming_forwards_chunks_and_usage(monkeypatch):
    token = _jwt()
    payloads = [
        {
            "id": "gen-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "qwen/qwen4-max:free",
            "choices": [{"index": 0, "delta": {"content": "Hi", "role": "assistant"}, "finish_reason": None}],
        },
        {
            "id": "gen-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "qwen/qwen4-max:free",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 2,
                "total_tokens": 14,
                "prompt_tokens_details": {"cached_tokens": 3},
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
        lambda *args, **kwargs: StreamClient(lambda u, d, p: FakeResponse(200, _refresh_payload(token)), calls),
    )

    plugin = NousOAuthPlugin()
    account = _account(
        access_token=token,
        access_token_expires_at=time.time() + 600,
        inference_base_url="https://inference-api.nousresearch.com/v1",
    )

    async def collect():
        stream = await plugin.chat_completion(_request(stream=True), _model(), {}, account=account)
        return [chunk async for chunk in stream]

    chunks = asyncio.run(collect())
    assert all(isinstance(chunk, ChatCompletionChunk) for chunk in chunks)
    assert chunks[0].choices[0].delta["role"] == "assistant"
    assert chunks[-1].choices[0].finish_reason == "stop"
    assert chunks[-1].usage.total_tokens == 14
    assert chunks[-1].usage.cache_read_input_tokens == 3


def test_plugin_owned_device_login_flow(monkeypatch):
    """The device flow lives in the plugin and produces a complete account."""
    from templates.nous_oauth import NousDeviceLoginManager

    token = _jwt()
    calls = {"token": 0}

    class DeviceClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, headers=None, data=None, json=None):
            if url.endswith("/api/oauth/device/code"):
                return FakeResponse(
                    200,
                    {
                        "device_code": "dc_test",
                        "user_code": "ABCD-1234",
                        "verification_uri": "https://portal.nousresearch.com/manage-subscription",
                        "verification_uri_complete": "https://portal.nousresearch.com/manage-subscription?user_code=ABCD-1234",
                        "expires_in": 600,
                        "interval": 5,
                    },
                )
            if url.endswith("/api/oauth/token"):
                calls["token"] += 1
                if data.get("grant_type") == "refresh_token":
                    return FakeResponse(
                        200,
                        {
                            "access_token": token,
                            "refresh_token": "rt_rotated",
                            "expires_in": 900,
                            "scope": "inference:invoke",
                            "inference_base_url": "https://inference-api.nousresearch.com/v1",
                        },
                    )
                if calls["token"] == 1:
                    return FakeResponse(200, {"error": "authorization_pending"})
                return FakeResponse(
                    200,
                    {"access_token": "device-token", "refresh_token": "rt_device"},
                )
            raise AssertionError(url)

    monkeypatch.setattr(
        nous_module.httpx,
        "AsyncClient",
        lambda *args, **kwargs: DeviceClient(),
    )

    manager = NousDeviceLoginManager()
    start = asyncio.run(manager.start_flow())
    assert start["mode"] == "device"
    assert start["user_code"] == "ABCD-1234"

    pending = asyncio.run(manager.check_flow_status(start["flow_id"]))
    assert pending["status"] == "pending"

    manager.active_flows[start["flow_id"]]["next_poll_at"] = 0
    done = asyncio.run(manager.check_flow_status(start["flow_id"]))
    assert done["status"] == "complete"
    assert done["account"].provider_id == "nous_oauth"
    assert done["account"].extra_config["refresh_token"] == "rt_rotated"
    assert done["account"].extra_config["access_token"] == token
