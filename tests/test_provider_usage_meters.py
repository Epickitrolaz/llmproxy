"""Tests for generic provider usage meters (rate-limit headers + Usage hooks)."""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional, Union

import pytest

from llmproxy.core.interface import BasePlugin
from llmproxy.core.manager import plugin_manager
from llmproxy.core.usage import (
    capture_rate_limit_headers,
    rate_limit_groups_from_headers,
    stored_usage_limits,
)
from llmproxy.dashboard.router import get_provider_usage_status
from llmproxy.schemas import (
    Account,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    PluginMetadata,
    PluginModelConfig,
)

ROOT = Path(__file__).resolve().parent.parent

MUNITO_HEADERS = {
    "x-ratelimit-limit-requests": "20",
    "x-ratelimit-remaining-requests": "19",
    "x-ratelimit-reset-requests": "44s",
    "x-ratelimit-limit-input-tokens": "40000",
    "x-ratelimit-remaining-input-tokens": "39993",
    "x-ratelimit-reset-input-tokens": "44s",
    "x-ratelimit-limit-output-tokens": "6000",
    "x-ratelimit-remaining-output-tokens": "6000",
    "x-ratelimit-reset-output-tokens": "44s",
    "x-ratelimit-limit-tokens": "46000",
    "x-ratelimit-remaining-tokens": "45993",
    "x-ratelimit-reset-tokens": "44s",
}

ANTHROPIC_HEADERS = {
    "anthropic-ratelimit-input-tokens-limit": "5000000",
    "anthropic-ratelimit-input-tokens-remaining": "5000000",
    "anthropic-ratelimit-input-tokens-reset": "2026-11-17T09:21:46Z",
    "anthropic-ratelimit-output-tokens-limit": "1000000",
    "anthropic-ratelimit-output-tokens-remaining": "1000000",
    "anthropic-ratelimit-output-tokens-reset": "2026-11-17T09:21:46Z",
    "anthropic-ratelimit-requests-limit": "5000",
    "anthropic-ratelimit-requests-remaining": "4999",
    "anthropic-ratelimit-requests-reset": "2026-11-17T09:21:46Z",
    "anthropic-ratelimit-tokens-limit": "6000000",
    "anthropic-ratelimit-tokens-remaining": "6000000",
    "anthropic-ratelimit-tokens-reset": "2026-11-17T09:21:46Z",
}

MOONMATH_HEADERS = {
    "x-ratelimit-api_key-limit-max_parallel_requests": "2",
    "x-ratelimit-api_key-remaining-max_parallel_requests": "1",
}


def test_munito_rate_limits_are_normalized_and_combined_tokens_removed():
    groups = rate_limit_groups_from_headers(MUNITO_HEADERS)

    assert len(groups) == 1
    buckets = groups[0]["buckets"]
    assert [bucket["bucketId"] for bucket in buckets] == [
        "requests",
        "input_tokens",
        "output_tokens",
    ]
    assert buckets[0]["displayName"] == "Requests"
    assert buckets[0]["usedPercent"] == pytest.approx(5.0)
    assert buckets[0]["remainingFraction"] == pytest.approx(0.95)
    assert buckets[0]["resetTime"]


def test_anthropic_rate_limits_keep_absolute_reset_times():
    groups = rate_limit_groups_from_headers(ANTHROPIC_HEADERS)

    buckets = groups[0]["buckets"]
    assert [bucket["bucketId"] for bucket in buckets] == [
        "requests",
        "input_tokens",
        "output_tokens",
    ]
    assert buckets[0]["usedPercent"] == pytest.approx(0.02)
    assert buckets[0]["resetTime"] == "2026-11-17T09:21:46Z"


def test_moonmath_scope_is_preserved_and_parallel_limit_is_metered():
    groups = rate_limit_groups_from_headers(MOONMATH_HEADERS)

    bucket = groups[0]["buckets"][0]
    assert bucket["bucketId"] == "api_key_max_parallel_requests"
    assert bucket["displayName"] == "Parallel requests"
    assert bucket["usedPercent"] == pytest.approx(50.0)


def test_capture_and_stored_round_trip():
    account = Account(provider_id="munito_ai", name="Munito", api_key="key")

    assert capture_rate_limit_headers(account, MUNITO_HEADERS) is True
    snapshot = stored_usage_limits(account)
    assert snapshot is not None
    assert snapshot["source"] == "response-headers"
    assert snapshot["groups"][0]["buckets"][0]["bucketId"] == "requests"

    # A response without rate-limit headers must not erase the last snapshot.
    assert capture_rate_limit_headers(account, {"content-type": "application/json"}) is False
    assert stored_usage_limits(account) is not None


def test_provider_usage_status_returns_stored_meters(monkeypatch):
    account = Account(
        provider_id="munito_ai",
        name="Munito",
        api_key="key",
        extra_config={
            "usage_limits": {
                "checked_at": 1,
                "source": "response-headers",
                "groups": rate_limit_groups_from_headers(MUNITO_HEADERS),
            }
        },
    )
    monkeypatch.setattr(plugin_manager, "accounts", {account.id: account})
    monkeypatch.setattr(plugin_manager, "configs", {})

    result = asyncio.run(get_provider_usage_status())

    assert len(result["accounts"]) == 1
    entry = result["accounts"][0]
    assert entry["account_id"] == account.id
    assert entry["quota"]["groups"][0]["buckets"][0]["bucketId"] == "requests"


def test_provider_usage_status_omits_providers_without_meter(monkeypatch):
    account = Account(provider_id="cloudflare_workers_ai", name="Cloudflare", api_key="key")
    monkeypatch.setattr(plugin_manager, "accounts", {account.id: account})
    monkeypatch.setattr(plugin_manager, "configs", {})
    monkeypatch.setattr(plugin_manager, "get_plugin", lambda provider_id: None)

    result = asyncio.run(get_provider_usage_status())

    assert result["accounts"] == []


class _HookPlugin(BasePlugin):
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            id="hook_provider",
            name="Hook Provider",
            supports_usage_meter=True,
        )

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        variables: Dict[str, Any],
        account: Optional[Account] = None,
    ) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionChunk]]:
        raise NotImplementedError

    async def fetch_usage_quota(self, account, variables):
        return {
            "groups": [],
            "balance": {"value": 12.5, "label": "available credit"},
            "summary": [{"label": "Requests", "value": "7"}],
        }


def test_provider_usage_status_calls_plugin_hook(monkeypatch):
    account = Account(provider_id="hook_provider", name="Hook", api_key="key")
    monkeypatch.setattr(plugin_manager, "accounts", {account.id: account})
    monkeypatch.setattr(plugin_manager, "configs", {})
    monkeypatch.setattr(plugin_manager, "get_plugin", lambda provider_id: _HookPlugin())

    result = asyncio.run(get_provider_usage_status())

    entry = result["accounts"][0]
    assert entry["quota"]["balance"]["value"] == 12.5
    assert entry["quota"]["summary"][0]["value"] == "7"


def _load_template(name: str):
    path = ROOT / "templates" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_template_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_anthropic_template_declares_usage_meter():
    module = _load_template("anthropic_proxy")
    assert module.plugin.metadata.supports_usage_meter is True


def test_groq_headers_with_ms_and_compound_reset_are_normalized():
    groq_headers = {
        "x-ratelimit-limit-requests": "1000",
        "x-ratelimit-remaining-requests": "999",
        "x-ratelimit-reset-requests": "1m26.4s",
        "x-ratelimit-limit-tokens": "8000",
        "x-ratelimit-remaining-tokens": "7927",
        "x-ratelimit-reset-tokens": "547ms",
    }
    groups = rate_limit_groups_from_headers(groq_headers)
    buckets = {b["bucketId"]: b for b in groups[0]["buckets"]}
    assert buckets["requests"]["usedPercent"] == pytest.approx(0.1)
    assert buckets["requests"]["resetTime"]
    assert buckets["tokens"]["usedPercent"] == pytest.approx(0.91, abs=0.02)
    assert buckets["tokens"]["resetTime"]


def test_expired_rate_limit_windows_reset_to_full_remaining():
    from datetime import datetime, timedelta, timezone

    now = datetime(2026, 11, 25, 19, 0, 0, tzinfo=timezone.utc)
    past_reset = (now - timedelta(minutes=5)).isoformat().replace("+00:00", "Z")
    future_reset = (now + timedelta(minutes=5)).isoformat().replace("+00:00", "Z")
    account = Account(
        provider_id="munito_ai",
        name="Munito",
        api_key="key",
        extra_config={
            "usage_limits": {
                "checked_at": (now - timedelta(minutes=10)).timestamp(),
                "source": "response-headers",
                "groups": [
                    {
                        "displayName": "Rate limits",
                        "buckets": [
                            {
                                "bucketId": "requests",
                                "displayName": "Requests",
                                "usedPercent": 30.0,
                                "remainingFraction": 0.7,
                                "resetTime": past_reset,
                            },
                            {
                                "bucketId": "input_tokens",
                                "displayName": "Input tokens",
                                "usedPercent": 20.0,
                                "remainingFraction": 0.8,
                                "resetTime": future_reset,
                            },
                        ],
                    }
                ],
            }
        },
    )

    snapshot = stored_usage_limits(account, now=now)
    assert snapshot is not None
    buckets = {b["bucketId"]: b for b in snapshot["groups"][0]["buckets"]}
    assert buckets["requests"]["usedPercent"] == 0.0
    assert buckets["requests"]["remainingFraction"] == 1.0
    assert buckets["input_tokens"]["usedPercent"] == 20.0
    assert buckets["input_tokens"]["remainingFraction"] == 0.8


def test_inferred_minute_window_expires_after_one_minute():
    from datetime import datetime, timedelta, timezone

    checked = datetime(2026, 11, 25, 19, 0, 0, tzinfo=timezone.utc)
    groups = rate_limit_groups_from_headers(
        {
            "x-ratelimit-limit-req-minute": "50",
            "x-ratelimit-remaining-req-minute": "49",
        },
        now=checked,
    )
    assert groups[0]["buckets"][0]["resetTime"] == "2026-11-25T19:01:00Z"

    account = Account(
        provider_id="mistral_vibe",
        name="Mistral Vibe",
        api_key="key",
        extra_config={
            "usage_limits": {
                "checked_at": checked.timestamp(),
                "source": "response-headers",
                "groups": [
                    {
                        "displayName": "Rate limits",
                        "buckets": [
                            {
                                "bucketId": "req_minute",
                                "displayName": "Req Minute",
                                "usedPercent": 2.0,
                                "remainingFraction": 0.98,
                                "resetTime": None,
                            }
                        ],
                    }
                ],
            }
        },
    )

    later = checked + timedelta(seconds=90)
    snapshot = stored_usage_limits(account, now=later)
    assert snapshot is not None
    bucket = snapshot["groups"][0]["buckets"][0]
    assert bucket["usedPercent"] == 0.0
    assert bucket["remainingFraction"] == 1.0


def test_capture_skips_when_usage_tracking_disabled():
    account = Account(
        provider_id="groqcloud",
        name="Groq",
        api_key="key",
        disable_usage_tracking=True,
    )
    assert capture_rate_limit_headers(account, MUNITO_HEADERS) is False
    assert stored_usage_limits(account) is None


def test_provider_usage_status_ignores_stored_limits_when_plugin_not_metered(monkeypatch):
    class _UnmeteredPlugin(BasePlugin):
        @property
        def metadata(self) -> PluginMetadata:
            return PluginMetadata(
                id="moonmath_zro",
                name="MoonMath",
                supports_usage_meter=False,
            )

        async def chat_completion(self, request, model_config, variables, account=None):
            raise NotImplementedError

    account = Account(
        provider_id="moonmath_zro",
        name="MoonMath",
        api_key="key",
        extra_config={
            "usage_limits": {
                "checked_at": 1,
                "source": "response-headers",
                "groups": rate_limit_groups_from_headers(MOONMATH_HEADERS),
            }
        },
    )
    monkeypatch.setattr(plugin_manager, "accounts", {account.id: account})
    monkeypatch.setattr(plugin_manager, "configs", {})
    monkeypatch.setattr(plugin_manager, "get_plugin", lambda provider_id: _UnmeteredPlugin())

    result = asyncio.run(get_provider_usage_status())
    assert result["accounts"] == []


def test_provider_usage_status_omits_empty_quota_from_hook(monkeypatch):
    class _EmptyHookPlugin(_HookPlugin):
        async def fetch_usage_quota(self, account, variables):
            return {"groups": []}

    account = Account(provider_id="hook_provider", name="Empty", api_key="key")
    monkeypatch.setattr(plugin_manager, "accounts", {account.id: account})
    monkeypatch.setattr(plugin_manager, "configs", {})
    monkeypatch.setattr(plugin_manager, "get_plugin", lambda provider_id: _EmptyHookPlugin())

    result = asyncio.run(get_provider_usage_status())
    assert result["accounts"] == []


@pytest.mark.parametrize(
    ("template_name", "expected"),
    [
        ("anthropic_proxy", True),
        ("antigravity", True),
        ("codex", True),
        ("groqcloud", True),
        ("mistral_vibe", True),
        ("munito_ai", True),
        ("nanogpt", True),
        ("nous_oauth", True),
        ("opencode_go", True),
        ("openrouter", True),
        ("agentrouter", False),
        ("cloudflare_workers_ai", False),
        ("colab_ollama", False),
        ("gemini", False),
        ("mock_provider", False),
        ("modelscope", False),
        ("moonmath_zro", False),
        ("nvidia_nim", False),
        ("openai_compatible", False),
        ("opencode_free", False),
    ],
)
def test_all_builtin_templates_declare_accurate_usage_meter_flag(template_name, expected):
    module = _load_template(template_name)
    assert module.plugin.metadata.supports_usage_meter is expected


def test_nanogpt_fetch_usage_quota_reports_balance_and_active_subscription(monkeypatch):
    import httpx

    module = _load_template("nanogpt")
    plugin = module.plugin
    account = Account(
        provider_id="nanogpt",
        name="NanoGPT",
        api_key="ng-live-key",
        base_url="https://api.nano-gpt.com/api/v1",
    )

    class _Resp:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self._payload = payload

        def json(self):
            return self._payload

    class _Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, headers=None):
            assert url == "https://api.nano-gpt.com/api/check-balance"
            assert headers["x-api-key"] == "ng-live-key"
            return _Resp(200, {"usd_balance": "4.75", "usd_spendable_balance": 4.75})

        async def get(self, url, headers=None):
            assert url == "https://api.nano-gpt.com/api/subscription/v1/usage"
            return _Resp(
                200,
                {
                    "active": True,
                    "weeklyInputTokens": {"percentUsed": 25, "resetAt": 1796000000000},
                },
            )

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    quota = asyncio.run(plugin.fetch_usage_quota(account, {}))
    assert quota is not None
    assert quota["balance"] == {"value": 4.75, "label": "available balance"}
    assert quota["groups"][0]["buckets"][0]["usedPercent"] == pytest.approx(25.0)
    assert quota["groups"][0]["buckets"][0]["remainingFraction"] == pytest.approx(0.75)


@pytest.mark.parametrize("template_name", ["groqcloud", "munito_ai"])
def test_groq_and_munito_capture_rate_limits_on_non_streaming_and_streaming(
    monkeypatch, template_name
):
    import httpx

    module = _load_template(template_name)
    plugin = module.plugin
    model = plugin.metadata.models[0]
    account = Account(provider_id=plugin.metadata.id, name="Test", api_key="live-key")

    headers = (
        {
            "x-ratelimit-limit-requests": "1000",
            "x-ratelimit-remaining-requests": "999",
            "x-ratelimit-reset-requests": "1m26.4s",
            "x-ratelimit-limit-tokens": "8000",
            "x-ratelimit-remaining-tokens": "7927",
            "x-ratelimit-reset-tokens": "547ms",
        }
        if template_name == "groqcloud"
        else MUNITO_HEADERS
    )

    class _MockResp:
        def __init__(self):
            self.status_code = 200
            self.headers = headers
            self.text = ""

        def json(self):
            return {
                "id": "chatcmpl-1",
                "created": 1,
                "model": model.upstream_model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            }

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def aiter_lines(self):
            yield 'data: {"id":"chatcmpl-1","model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"hi"}}]}'
            yield 'data: {"id":"chatcmpl-1","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}'
            yield "data: [DONE]"

    class _MockClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, headers=None, json=None):
            return _MockResp()

        def stream(self, method, url, headers=None, json=None):
            return _MockResp()

    monkeypatch.setattr(httpx, "AsyncClient", _MockClient)

    req = ChatCompletionRequest(
        model=model.id,
        messages=[ChatMessage(role="user", content="hi")],
    )
    resp = asyncio.run(plugin.chat_completion(req, model, {}, account=account))
    assert isinstance(resp, ChatCompletionResponse)
    assert stored_usage_limits(account) is not None

    account_stream = Account(provider_id=plugin.metadata.id, name="Test2", api_key="live-key")
    stream_req = ChatCompletionRequest(
        model=model.id,
        messages=[ChatMessage(role="user", content="hi")],
        stream=True,
    )

    async def _consume():
        stream = await plugin.chat_completion(stream_req, model, {}, account=account_stream)
        return [chunk async for chunk in stream]

    chunks = asyncio.run(_consume())
    assert chunks
    assert stored_usage_limits(account_stream) is not None
