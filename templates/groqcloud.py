"""GroqCloud OpenAI-compatible provider plugin for LLMProxy.

Speaks the OpenAI Chat Completions protocol against ``https://api.groq.com/openai/v1``
with API-key accounts, proxy routing, reasoning effort controls, prompt-cache hints,
streaming usage telemetry, and live ``x-ratelimit-*`` response header capture.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, AsyncIterator, Dict, Optional, Tuple, Union

import httpx

from llmproxy.core.interface import BasePlugin
from llmproxy.core.usage import capture_rate_limit_headers
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

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
TEST_KEY_PREFIXES = ("test", "mock", "sk-test", "dummy", "fake", "offline", "sample")
TEST_KEYS = ("test-key", "test_key", "secret123")


class GroqCloudPlugin(BasePlugin):
    """GroqCloud provider."""

    def __init__(self) -> None:
        self._metadata = PluginMetadata(
            id="groqcloud",
            name="GroqCloud",
            description="GroqCloud OpenAI-compatible inference with live rate-limit metering",
            version="1.1.0",
            author="Groq",
            enabled=True,
            filename="groqcloud.py",
            supports_usage_meter=True,
            variables=[
                PluginVariable(
                    name="base_url",
                    label="Base URL",
                    type="string",
                    default=DEFAULT_BASE_URL,
                    description="Provider API base URL.",
                ),
                PluginVariable(
                    name="timeout",
                    label="Timeout (seconds)",
                    type="number",
                    default=300,
                    description="Request timeout in seconds.",
                ),
                PluginVariable(
                    name="load_balancing_strategy",
                    label="Load Balancing Strategy",
                    type="select",
                    default="fallback",
                    options=["fallback", "round_robin", "least_used", "random"],
                    description="How requests are spread across accounts.",
                ),
                PluginVariable(
                    name="account_switch_cooldown",
                    label="Account Cooldown (s)",
                    type="number",
                    default=60,
                    description="Cooldown after a rate-limit response.",
                ),
                PluginVariable(
                    name="account_switch_wait",
                    label="Switch Wait Delay (s)",
                    type="number",
                    default=0.1,
                    description="Delay before switching accounts.",
                ),
                PluginVariable(
                    name="max_retries",
                    label="Max Failover Retries",
                    type="number",
                    default=2,
                    description="Maximum account failovers per request.",
                ),
                PluginVariable(
                    name="reasoning_effort",
                    label="Default Reasoning Effort",
                    type="select",
                    default="default",
                    options=["default", "low", "medium", "high", "max", "xhigh"],
                    description="Used when the client does not send reasoning_effort.",
                ),
                PluginVariable(
                    name="custom_headers",
                    label="Custom Headers",
                    type="string",
                    default="{}",
                    description="JSON object of extra HTTP headers.",
                ),
                PluginVariable(
                    name="proxy_enabled",
                    label="Enable External Proxy",
                    type="boolean",
                    default=False,
                    description="Route requests through external proxies.",
                ),
                PluginVariable(
                    name="proxy_strategy",
                    label="Proxy Strategy",
                    type="select",
                    default="sequential",
                    options=["sequential", "random", "specific", "switch_on_limit"],
                    description="How the proxy pool is selected.",
                ),
                PluginVariable(
                    name="specific_proxy_id",
                    label="Specific Proxy ID / Tag",
                    type="string",
                    default="",
                    description="Target proxy ID, URL, or tag for the specific strategy.",
                ),
                PluginVariable(
                    name="proxy",
                    label="Direct Proxy URL",
                    type="string",
                    default="",
                    description="Direct HTTP or SOCKS5 proxy URL.",
                ),
            ],
            models=[
                PluginModelConfig(
                    id="groq/openai/gpt-oss-120b",
                    name="GPT-OSS 120B",
                    upstream_model="openai/gpt-oss-120b",
                    context_window=131072,
                    max_tokens=65536,
                    reasoning_levels=["low", "medium", "high"],
                ),
                PluginModelConfig(
                    id="groq/openai/gpt-oss-20b",
                    name="GPT-OSS 20B",
                    upstream_model="openai/gpt-oss-20b",
                    context_window=131072,
                    max_tokens=65536,
                    reasoning_levels=["low", "medium", "high"],
                ),
                PluginModelConfig(
                    id="groq/qwen/qwen3.8-27b",
                    name="Qwen3.8 27B",
                    upstream_model="qwen/qwen3.8-27b",
                    context_window=131072,
                    max_tokens=16384,
                    reasoning_levels=["low", "medium", "high"],
                    supports_vision=True,
                ),
                PluginModelConfig(
                    id="groq/openai/gpt-oss-safeguard-20b",
                    name="GPT-OSS Safeguard 20B",
                    upstream_model="openai/gpt-oss-safeguard-20b",
                    context_window=131072,
                    max_tokens=65536,
                    reasoning_levels=["low", "medium", "high"],
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
            f"Account '{account.name}' ({account.id}) has no API key configured for GroqCloud."
        )

    @staticmethod
    def _normalize_effort(value: Any) -> Optional[str]:
        if value is None:
            return None
        raw = str(value).strip().lower()
        if not raw or raw in ("default", "auto", "off", "none", "disabled"):
            return None
        return raw

    def _resolve_model(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        variables: Dict[str, Any],
    ) -> Tuple[str, Optional[str]]:
        target = model_config.upstream_model or model_config.id
        requested = (request.model or "").strip()

        levels = {str(level).lower() for level in model_config.reasoning_levels}
        if ":" in requested:
            base, suffix = requested.rsplit(":", 1)
            if suffix.lower() in levels:
                target = model_config.upstream_model or base

        raw_effort = request.reasoning_effort
        extra = request.extra_body if isinstance(request.extra_body, dict) else {}
        if raw_effort is None:
            raw_effort = extra.get("reasoning_effort")
        if raw_effort is None:
            raw_effort = (model_config.parameters or {}).get("reasoning_effort")
        if raw_effort is None:
            raw_effort = variables.get("reasoning_effort")
        effort = self._normalize_effort(raw_effort)
        return target, effort

    def _resolve_base_url(
        self, account: Optional[Account], variables: Dict[str, Any]
    ) -> str:
        raw = (
            account.base_url
            if account and account.base_url
            else (variables.get("base_url") or DEFAULT_BASE_URL)
        )
        return str(raw).strip().rstrip("/")

    @staticmethod
    def _apply_cache_fields(
        body: Dict[str, Any], request: ChatCompletionRequest
    ) -> None:
        extra = request.extra_body if isinstance(request.extra_body, dict) else {}
        for key in ("prompt_cache_key", "prompt_cache_retention"):
            value = getattr(request, key, None)
            if value is None:
                value = extra.get(key)
            if value is not None:
                body[key] = value

        cache_control = extra.get("cache_control")
        if not cache_control:
            return
        messages = body.get("messages") or []
        if not messages:
            return
        content = messages[-1].get("content")
        if isinstance(content, list) and content:
            block = content[-1]
            if isinstance(block, dict):
                block.setdefault("cache_control", cache_control)
        elif isinstance(content, str):
            messages[-1]["content"] = [
                {"type": "text", "text": content, "cache_control": cache_control}
            ]

    def _build_body(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        variables: Dict[str, Any],
    ) -> Dict[str, Any]:
        target_model, effort = self._resolve_model(request, model_config, variables)
        body: Dict[str, Any] = {
            "model": target_model,
            "messages": [
                message.model_dump(exclude_none=True) for message in request.messages
            ],
            "stream": bool(request.stream),
        }

        for field in (
            "temperature",
            "top_p",
            "n",
            "stop",
            "presence_penalty",
            "frequency_penalty",
            "user",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "response_format",
            "seed",
            "logprobs",
            "top_logprobs",
        ):
            value = getattr(request, field, None)
            if value is None:
                continue
            if field == "n" and value == 1:
                continue
            if field == "top_p" and value == 1:
                continue
            if field in ("presence_penalty", "frequency_penalty") and value == 0:
                continue
            body[field] = value

        max_tokens = (
            request.max_tokens
            if request.max_tokens is not None
            else request.max_completion_tokens
        )
        if max_tokens is not None:
            body["max_tokens"] = max_tokens

        extra = request.extra_body if isinstance(request.extra_body, dict) else {}
        for key, value in extra.items():
            if key in (
                "model",
                "messages",
                "stream",
                "max_tokens",
                "max_completion_tokens",
                "cache_control",
                "munito_affinity",
                "affinity",
            ):
                continue
            body[key] = value

        if effort:
            body["reasoning_effort"] = effort

        if request.stream:
            stream_options = dict(request.stream_options or {})
            stream_options["include_usage"] = True
            body["stream_options"] = stream_options

        self._apply_cache_fields(body, request)
        return body

    def _headers(
        self,
        api_key: str,
        variables: Dict[str, Any],
        request: ChatCompletionRequest,
    ) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        raw_headers = variables.get("custom_headers")
        if raw_headers:
            try:
                parsed = json.loads(str(raw_headers))
                if isinstance(parsed, dict):
                    headers.update({str(k): str(v) for k, v in parsed.items()})
            except Exception:
                logger.warning("Ignoring invalid custom_headers JSON for groqcloud")

        extra = request.extra_body if isinstance(request.extra_body, dict) else {}
        affinity = extra.get("prompt_cache_key")
        if affinity:
            headers["x-session-affinity"] = str(affinity)
        return headers

    def _proxy(
        self, account: Optional[Account], variables: Dict[str, Any]
    ) -> Optional[str]:
        raw = variables.get("proxy") or (getattr(account, "proxy_url", None) or "")
        return str(raw).strip() or None

    def _mock_response(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        account: Optional[Account] = None,
    ) -> ChatCompletionResponse:
        model_name = request.model or model_config.id
        return ChatCompletionResponse(
            id=f"chatcmpl-groq-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=model_name,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=f"Simulated response from GroqCloud ({model_name}).",
                    ),
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=10,
                completion_tokens=12,
                total_tokens=22,
            ),
        )

    async def _mock_stream(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        account: Optional[Account] = None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        model_name = request.model or model_config.id
        chunk_id = f"chatcmpl-groq-{uuid.uuid4().hex[:12]}"
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
                    delta={"content": f"Simulated response from GroqCloud ({model_name})."},
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
        variables: Dict[str, Any],
        account: Optional[Account],
    ) -> AsyncIterator[ChatCompletionChunk]:
        timeout = float(variables.get("timeout") or 300.0)
        async with httpx.AsyncClient(
            timeout=timeout,
            proxy=self._proxy(account, variables),
            follow_redirects=True,
        ) as client:
            async with client.stream("POST", url, headers=headers, json=body) as resp:
                if account is not None:
                    capture_rate_limit_headers(account, resp.headers)
                if resp.status_code >= 400:
                    detail = (await resp.aread()).decode("utf-8", "replace")
                    raise httpx.HTTPStatusError(
                        f"GroqCloud upstream returned {resp.status_code}: {detail[:2000]}",
                        request=resp.request,
                        response=resp,
                    )

                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        break
                    try:
                        parsed = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(parsed, dict):
                        continue
                    parsed.setdefault("id", "chatcmpl-groqcloud")
                    parsed.setdefault("model", body.get("model", ""))
                    parsed.setdefault("choices", [])
                    try:
                        yield ChatCompletionChunk.model_validate(parsed)
                    except Exception as exc:
                        logger.debug("Skipping malformed groqcloud stream chunk: %s", exc)

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

        base_url = self._resolve_base_url(account, variables)
        url = f"{base_url}/chat/completions"
        body = self._build_body(request, model_config, variables)
        headers = self._headers(api_key, variables, request)

        if request.stream:
            return self._stream_request(url, headers, body, variables, account)

        timeout = float(variables.get("timeout") or 300.0)
        async with httpx.AsyncClient(
            timeout=timeout,
            proxy=self._proxy(account, variables),
            follow_redirects=True,
        ) as client:
            resp = await client.post(url, headers=headers, json=body)
            if account is not None:
                capture_rate_limit_headers(account, resp.headers)
            if resp.status_code >= 400:
                detail = resp.text[:2000]
                raise httpx.HTTPStatusError(
                    f"GroqCloud upstream returned {resp.status_code}: {detail}",
                    request=resp.request,
                    response=resp,
                )
            return ChatCompletionResponse.model_validate(resp.json())


plugin = GroqCloudPlugin()
