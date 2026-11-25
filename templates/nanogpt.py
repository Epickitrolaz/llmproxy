"""NanoGPT OpenAI-compatible provider plugin for LLMProxy.

Speaks the OpenAI Chat Completions protocol against ``https://nano-gpt.com/api/v1``
(or ``https://api.nano-gpt.com/api/v1``) with live balance and subscription usage
reporting through ``fetch_usage_quota``.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import httpx

from llmproxy.core.interface import BasePlugin
from llmproxy.schemas import (
    Account,
    ChatChoice,
    ChatChoiceDelta,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    CompletionUsage,
    PluginMetadata,
    PluginModelConfig,
    PluginVariable,
)

DEFAULT_BASE_URL = "https://nano-gpt.com/api/v1"
TEST_KEY_PREFIXES = ("test", "mock", "sk-test", "dummy", "fake", "offline", "sample")
TEST_KEYS = ("test-key", "test_key", "secret123")


class NanoGPTPlugin(BasePlugin):
    """NanoGPT OpenAI-compatible provider with live balance and subscription metering."""

    def __init__(self) -> None:
        self._metadata = PluginMetadata(
            id="nanogpt",
            name="NanoGPT",
            description="NanoGPT OpenAI-compatible inference with live USD balance and subscription usage metering",
            version="2.1.0",
            author="Community",
            enabled=True,
            filename="nanogpt.py",
            supports_usage_meter=True,
            variables=[
                PluginVariable(
                    name="timeout",
                    label="Timeout (seconds)",
                    type="number",
                    default=60,
                    description="Request timeout in seconds",
                ),
                PluginVariable(
                    name="load_balancing_strategy",
                    label="Load Balancing Strategy",
                    type="select",
                    default="round_robin",
                    options=["round_robin", "least_used", "fallback", "random"],
                    description="Strategy for distributing requests across connected accounts",
                ),
                PluginVariable(
                    name="account_switch_cooldown",
                    label="Account Cooldown (s)",
                    type="number",
                    default=10,
                    description="Cooldown period when an account hits rate limits before retrying",
                ),
                PluginVariable(
                    name="account_switch_wait",
                    label="Switch Wait Delay (s)",
                    type="number",
                    default=0.1,
                    description="Wait delay before switching to another account on failover",
                ),
                PluginVariable(
                    name="max_retries",
                    label="Max Failover Retries",
                    type="number",
                    default=2,
                    description="Maximum number of account failovers per request",
                ),
                PluginVariable(
                    name="base_url",
                    label="Default Base URL",
                    type="string",
                    default=DEFAULT_BASE_URL,
                    description="NanoGPT OpenAI-compatible API base URL",
                ),
                PluginVariable(
                    name="custom_headers",
                    label="Custom Headers",
                    type="string",
                    default="{}",
                    description="JSON dictionary of default custom HTTP headers",
                ),
                PluginVariable(
                    name="reasoning_effort",
                    label="Default Reasoning Effort",
                    type="select",
                    default="default",
                    options=[
                        "default",
                        "none",
                        "minimal",
                        "low",
                        "medium",
                        "high",
                        "xhigh",
                        "max",
                    ],
                    description=(
                        "Forwarded to NanoGPT as reasoning_effort when supported. "
                        "A request-level value takes precedence."
                    ),
                ),
                PluginVariable(
                    name="proxy_enabled",
                    label="Enable External Proxy",
                    type="boolean",
                    default=False,
                    description="Route provider requests through external proxies (off by default)",
                ),
                PluginVariable(
                    name="proxy_strategy",
                    label="Proxy Strategy",
                    type="select",
                    default="sequential",
                    options=["sequential", "random", "specific", "switch_on_limit"],
                    description="Strategy for selecting proxies from pool",
                ),
                PluginVariable(
                    name="specific_proxy_id",
                    label="Specific Proxy ID / Tag",
                    type="string",
                    default="",
                    description="Target proxy ID, URL, or tag to route through when strategy is 'specific'",
                ),
                PluginVariable(
                    name="proxy",
                    label="Direct Proxy URL",
                    type="string",
                    default="",
                    description="Direct HTTP or SOCKS5 proxy URL to route through",
                ),
            ],
            models=[
                PluginModelConfig(
                    id="ng/qwen/qwen3.8-27b-uncensored",
                    name="Qwen3.8 27B Uncensored",
                    upstream_model="qwen/qwen3.8-27b-uncensored:thinking",
                    context_window=128000,
                    max_tokens=16384,
                    supports_vision=True,
                ),
                PluginModelConfig(
                    id="ng/z-ai/glm-5.3-flash-uncensored",
                    name="GLM 5.3 Flash Uncensored",
                    upstream_model="z-ai/glm-5.3-flash-uncensored",
                    context_window=128000,
                    max_tokens=16384,
                    supports_vision=True,
                ),
                PluginModelConfig(
                    id="ng/deepseek/deepseek-v4.1-flash-1017-uncensored",
                    name="Deepseek V4.1 Flash Uncensored",
                    upstream_model="deepseek/deepseek-v4.1-flash-1017-uncensored:thinking",
                    context_window=128000,
                    max_tokens=16384,
                    supports_vision=True,
                ),
                PluginModelConfig(
                    id="ng/z-ai/glm-5.3-flash-cybersecurity",
                    name="GLM 5.3 Flash Cybersecurity",
                    upstream_model="z-ai/glm-5.3-flash-cybersecurity",
                    context_window=128000,
                    max_tokens=16384,
                    supports_vision=True,
                ),
            ],
        )

    @property
    def metadata(self) -> PluginMetadata:
        return self._metadata

    @staticmethod
    def _is_test_mode(api_key: str, account: Optional[Account] = None) -> bool:
        key = (api_key or "").strip().lower()
        if any(key.startswith(prefix) for prefix in TEST_KEY_PREFIXES) or key in TEST_KEYS:
            return True
        if key:
            return False
        if account is None or str(account.id or "").startswith("default-"):
            return True
        raise ValueError(
            f"Account '{account.name}' ({account.id}) has no API key configured for NanoGPT."
        )

    @staticmethod
    def _apply_cache_fields(body: Dict[str, Any], request: ChatCompletionRequest) -> None:
        extra = request.extra_body if isinstance(request.extra_body, dict) else {}
        for key in (
            "prompt_cache_key",
            "prompt_cache_retention",
            "cache_control",
            "cached_content",
            "cachedContent",
        ):
            value = getattr(request, key, None)
            if value is None:
                value = extra.get(key)
            if value is not None:
                body[key] = value
        cache = extra.get("cache")
        if isinstance(cache, dict):
            body["cache"] = cache

    @staticmethod
    def _resolve_reasoning_effort(
        request: ChatCompletionRequest, variables: Dict[str, Any]
    ) -> Optional[str]:
        raw: Any = request.reasoning_effort
        extra = request.extra_body if isinstance(request.extra_body, dict) else {}
        if raw is None:
            raw = extra.get("reasoning_effort")
        if raw is None:
            reasoning = extra.get("reasoning")
            if isinstance(reasoning, dict):
                raw = reasoning.get("effort")
            elif reasoning:
                raw = reasoning
        if raw is None:
            raw = variables.get("reasoning_effort")
        level = str(raw or "").strip().lower()
        if not level or level in ("default", "auto"):
            return None
        if level in ("off", "disabled", "false", "0"):
            return "none"
        return level

    def _mock_response(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        account: Optional[Account] = None,
    ) -> ChatCompletionResponse:
        model_name = request.model or model_config.id
        return ChatCompletionResponse(
            id=f"chatcmpl-ng-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=model_name,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=f"Simulated response from NanoGPT ({model_name}).",
                    ),
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(prompt_tokens=10, completion_tokens=12, total_tokens=22),
        )

    async def _mock_stream(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        account: Optional[Account] = None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        model_name = request.model or model_config.id
        chunk_id = f"chatcmpl-ng-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        yield ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=model_name,
            choices=[
                ChatChoiceDelta(
                    index=0,
                    delta={"role": "assistant", "content": ""},
                    finish_reason=None,
                )
            ],
        )
        yield ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=model_name,
            choices=[
                ChatChoiceDelta(
                    index=0,
                    delta={"content": f"Simulated response from NanoGPT ({model_name})."},
                    finish_reason=None,
                )
            ],
        )
        yield ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=model_name,
            choices=[ChatChoiceDelta(index=0, delta={}, finish_reason="stop")],
            usage=CompletionUsage(prompt_tokens=10, completion_tokens=12, total_tokens=22),
        )

    async def _stream_request(
        self,
        url: str,
        headers: Dict[str, str],
        body: Dict[str, Any],
        proxy: Optional[str] = None,
        timeout: float = 60.0,
    ) -> AsyncIterator[ChatCompletionChunk]:
        async with httpx.AsyncClient(timeout=timeout, proxy=proxy) as client:
            async with client.stream("POST", url, headers=headers, json=body) as resp:
                if resp.status_code >= 400:
                    detail = (await resp.aread()).decode("utf-8", "replace")
                    raise httpx.HTTPStatusError(
                        f"NanoGPT upstream returned {resp.status_code}: {detail[:2000]}",
                        request=resp.request,
                        response=resp,
                    )
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line or line.startswith(":"):
                        continue
                    if line.startswith("data:"):
                        data = line[5:].strip()
                        if not data or data == "[DONE]":
                            break
                        try:
                            parsed = json.loads(data)
                            yield ChatCompletionChunk.model_validate(parsed)
                        except Exception:
                            continue

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        variables: Dict[str, Any],
        account: Optional[Account] = None,
    ) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionChunk]]:
        api_key = (account.api_key if account else "") or ""
        if self._is_test_mode(api_key, account):
            if request.stream:
                return self._mock_stream(request, model_config, account=account)
            return self._mock_response(request, model_config, account=account)

        base_url = (
            account.base_url
            if (account and account.base_url)
            else str(variables.get("base_url") or DEFAULT_BASE_URL)
        ).rstrip("/")
        raw_custom_headers = str(variables.get("custom_headers") or "{}")
        target_model = model_config.upstream_model or model_config.id

        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        if raw_custom_headers:
            try:
                parsed_headers = json.loads(raw_custom_headers)
                if isinstance(parsed_headers, dict):
                    headers.update({str(k): str(v) for k, v in parsed_headers.items()})
            except Exception:
                pass

        body: Dict[str, Any] = {
            "model": target_model,
            "messages": [m.model_dump(exclude_none=True) for m in request.messages],
            "stream": bool(request.stream),
        }
        if request.temperature is not None:
            body["temperature"] = request.temperature
        max_tokens = (
            request.max_tokens
            if request.max_tokens is not None
            else request.max_completion_tokens
        )
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if request.top_p is not None:
            body["top_p"] = request.top_p
        if request.tools is not None:
            body["tools"] = request.tools
        if request.tool_choice is not None:
            body["tool_choice"] = request.tool_choice
        if request.parallel_tool_calls is not None:
            body["parallel_tool_calls"] = request.parallel_tool_calls
        if request.response_format is not None:
            body["response_format"] = request.response_format
        if request.stop is not None:
            body["stop"] = request.stop

        reasoning_effort = self._resolve_reasoning_effort(request, variables)
        if reasoning_effort:
            body["reasoning_effort"] = reasoning_effort
            if isinstance(request.extra_body, dict) and isinstance(
                request.extra_body.get("reasoning"), dict
            ):
                body.setdefault("reasoning", request.extra_body["reasoning"])

        if request.stream:
            stream_options = dict(request.stream_options or {})
            stream_options["include_usage"] = True
            body["stream_options"] = stream_options

        self._apply_cache_fields(body, request)

        url = f"{base_url}/chat/completions"
        proxy = (
            str(variables.get("proxy") or (getattr(account, "proxy_url", None) or "")).strip()
            or None
        )
        timeout = float(variables.get("timeout") or 60.0)

        if request.stream:
            return self._stream_request(url, headers, body, proxy=proxy, timeout=timeout)

        async with httpx.AsyncClient(timeout=timeout, proxy=proxy) as client:
            resp = await client.post(url, headers=headers, json=body)
            if resp.status_code >= 400:
                raise httpx.HTTPStatusError(
                    f"NanoGPT upstream returned {resp.status_code}: {resp.text[:2000]}",
                    request=resp.request,
                    response=resp,
                )
            return ChatCompletionResponse.model_validate(resp.json())

    async def fetch_usage_quota(
        self,
        account: Optional[Account],
        variables: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Fetch live NanoGPT balance and active subscription usage."""
        api_key = ((account.api_key if account else "") or "").strip()
        if not api_key:
            return None
        raw_base = (
            (account.base_url if account and account.base_url else None)
            or str(variables.get("base_url") or DEFAULT_BASE_URL)
        ).strip().rstrip("/")
        if raw_base.endswith("/api/v1"):
            api_root = raw_base[:-7]
        elif raw_base.endswith("/v1"):
            api_root = raw_base[:-3]
        else:
            api_root = "https://nano-gpt.com"
        if not api_root.startswith("http"):
            api_root = "https://nano-gpt.com"

        proxy = (
            str(variables.get("proxy") or (getattr(account, "proxy_url", None) or "")).strip()
            or None
        )
        balance_val: Optional[float] = None
        groups: List[Dict[str, Any]] = []
        last_err: Optional[str] = None

        async with httpx.AsyncClient(timeout=15.0, proxy=proxy) as client:
            try:
                bal_resp = await client.post(
                    f"{api_root}/api/check-balance",
                    headers={"x-api-key": api_key},
                )
                if bal_resp.status_code == 200:
                    bal_data = bal_resp.json() or {}
                    raw_bal = bal_data.get("usd_spendable_balance")
                    if raw_bal is None:
                        raw_bal = bal_data.get("usd_balance")
                    if raw_bal is not None:
                        balance_val = float(raw_bal)
                else:
                    last_err = f"NanoGPT balance returned HTTP {bal_resp.status_code}"
            except Exception as exc:
                last_err = f"NanoGPT balance request failed: {exc}"

            try:
                sub_resp = await client.get(
                    f"{api_root}/api/subscription/v1/usage",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                if sub_resp.status_code == 200:
                    sub_data = sub_resp.json() or {}
                    if sub_data.get("active"):
                        buckets: List[Dict[str, Any]] = []
                        for key, label in (
                            ("dailyInputTokens", "Daily Input Tokens"),
                            ("weeklyInputTokens", "Weekly Input Tokens"),
                            ("dailyImages", "Daily Images"),
                        ):
                            win = sub_data.get(key)
                            if not isinstance(win, dict):
                                continue
                            pct = win.get("percentUsed")
                            if pct is None:
                                continue
                            used_pct = max(0.0, min(100.0, float(pct)))
                            reset_at = win.get("resetAt")
                            reset_iso = None
                            if isinstance(reset_at, (int, float)) and reset_at > 0:
                                ts = reset_at / 1000.0 if reset_at > 1e12 else float(reset_at)
                                reset_iso = (
                                    datetime.fromtimestamp(ts, tz=timezone.utc)
                                    .isoformat()
                                    .replace("+00:00", "Z")
                                )
                            buckets.append(
                                {
                                    "bucketId": f"nanogpt_{key}",
                                    "displayName": label,
                                    "usedPercent": round(used_pct, 2),
                                    "remainingFraction": round((100.0 - used_pct) / 100.0, 4),
                                    "resetTime": reset_iso,
                                }
                            )
                        if buckets:
                            groups.append(
                                {"displayName": "NanoGPT Subscription", "buckets": buckets}
                            )
            except Exception:
                pass

        if balance_val is None and not groups:
            return {"error": last_err} if last_err else None
        payload: Dict[str, Any] = {"groups": groups}
        if balance_val is not None:
            payload["balance"] = {
                "value": round(balance_val, 4),
                "label": "available balance",
            }
        return payload


plugin = NanoGPTPlugin()
