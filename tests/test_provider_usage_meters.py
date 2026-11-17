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
