"""OpenRouter provider plugin for LLMProxy.

OpenRouter exposes an OpenAI-compatible Chat Completions API, so this plugin
forwards requests directly and lets OpenRouter handle provider routing. A small
set of currently strong free models ships as defaults; users can add any other
OpenRouter model from the dashboard.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union

import httpx

from llmproxy.core.interface import BasePlugin
from llmproxy.schemas import (
    Account,
    ChatChoice,
    ChatChoiceDelta,
    ChatMessage,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionUsage,
    PluginMetadata,
    PluginModelConfig,
    PluginVariable,
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

REASONING_SUFFIXES: Tuple[Tuple[str, str], ...] = (
    (":none", "none"),
    ("-none", "none"),
    (":off", "none"),
    ("-off", "none"),
    (":low", "low"),
    ("-low", "low"),
    (":medium", "medium"),
    ("-medium", "medium"),
    (":high", "high"),
    ("-high", "high"),
)


class OpenRouterPlugin(BasePlugin):
    """OpenRouter provider with free-model defaults and reasoning controls."""

    def __init__(self) -> None:
        self._metadata = PluginMetadata(
            id="openrouter",
            name="OpenRouter",
            description=(
                "OpenRouter multi-provider gateway with a curated set of strong free "
                "models and support for any OpenRouter model ID"
            ),
            version="1.0.0",
            author="OpenRouter",
            enabled=True,
            filename="openrouter.py",
            variables=[
                PluginVariable(
                    name="api_key",
                    label="OpenRouter API Key",
                    type="secret",
                    default="",
                    description="OpenRouter API key (accounts can also supply this)",
                ),
                PluginVariable(
                    name="base_url",
                    label="Base URL",
                    type="string",
                    default=OPENROUTER_BASE_URL,
                    description="OpenRouter API base URL",
                ),
                PluginVariable(
                    name="timeout",
                    label="Timeout (seconds)",
                    type="number",
                    default=120,
                    description="Request timeout in seconds",
                ),
                PluginVariable(
                    name="load_balancing_strategy",
                    label="Load Balancing Strategy",
                    type="select",
                    default="fallback",
                    options=["fallback", "round_robin", "least_used", "random"],
                    description="Strategy for distributing requests across connected accounts",
                ),
                PluginVariable(
                    name="account_switch_cooldown",
                    label="Account Cooldown (s)",
                    type="number",
                    default=60,
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
                    name="site_url",
                    label="Site URL",
                    type="string",
                    default="",
                    description=(
                        "Optional HTTP-Referer value sent to OpenRouter for app "
                        "attribution and free-model eligibility"
                    ),
                ),
                PluginVariable(
                    name="app_name",
                    label="App Name",
                    type="string",
                    default="LLMProxy",
                    description="Optional X-Title value sent to OpenRouter",
                ),
                PluginVariable(
                    name="reasoning_effort",
                    label="Reasoning Effort",
                    type="select",
                    default="default",
                    options=["default", "none", "low", "medium", "high"],
                    description=(
                        "Default reasoning effort for models that support it. "
                        "Request-level reasoning_effort or a model suffix takes precedence."
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
                    description=(
                        "Strategy for selecting proxies from the pool: sequential, random, "
                        "specific, or switch_on_limit"
                    ),
                ),
                PluginVariable(
                    name="specific_proxy_id",
                    label="Specific Proxy ID / Tag",
                    type="string",
                    default="",
                    description="Target proxy ID, URL, or tag when the strategy is 'specific'",
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
                    id="nvidia/nemotron-3-ultra-550b-a55b:free",
                    name="Nemotron 3 Ultra (free)",
                    upstream_model="nvidia/nemotron-3-ultra-550b-a55b:free",
                    context_window=1000000,
                    max_tokens=65536,
                ),
                PluginModelConfig(
                    id="baidu/ernie-5.5-flash-preview:free",
                    name="ERNIE 5.5 Flash Preview (free)",
                    upstream_model="baidu/ernie-5.5-flash-preview:free",
                    context_window=1000000,
                    max_tokens=65536,
                ),
                PluginModelConfig(
                    id="thinkingmachines/inkling-2-small:free",
                    name="Inkling 2 Small (free)",
                    upstream_model="thinkingmachines/inkling-2-small:free",
                    context_window=1048576,
                    max_tokens=131072,
                ),
                PluginModelConfig(
                    id="nvidia/nemotron-3.5-super-vl:free",
                    name="Nemotron 3.5 Super VL (free)",
                    upstream_model="nvidia/nemotron-3.5-super-vl:free",
                    context_window=1000000,
                    max_tokens=131072,
                ),
                PluginModelConfig(
                    id="google/gemma-4-31b-it:free",
                    name="Gemma 4 31B (free)",
                    upstream_model="google/gemma-4-31b-it:free",
                    context_window=262144,
                    max_tokens=32768,
                ),
                PluginModelConfig(
                    id="cohere/north-mini-code:free",
                    name="North Mini Code (free)",
                    upstream_model="cohere/north-mini-code:free",
                    context_window=256000,
                    max_tokens=64000,
                ),
            ],
        )

    @property
    def metadata(self) -> PluginMetadata:
        return self._metadata

    @staticmethod
    def _is_test_mode(api_key: str) -> bool:
        if not api_key:
            return True
        key = api_key.lower().strip()
        prefixes = ("test", "mock", "sk-test", "dummy", "fake", "offline", "sample")
        return any(key.startswith(p) for p in prefixes) or key in (
            "test-key",
            "test_key",
            "secret123",
        )

    @staticmethod
    def _normalize_effort(value: Any) -> Optional[str]:
        if value is None:
            return None
        raw = str(value).strip().lower()
        if not raw or raw in ("default", "auto"):
            return None
        if raw in ("none", "off", "disabled", "false", "0"):
            return "none"
        if raw in ("low", "medium", "high"):
            return raw
        return None

    def _resolve_reasoning(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        variables: Dict[str, Any],
    ) -> Tuple[str, Optional[str]]:
        """Return ``(target_model, effort)`` without stripping OpenRouter's ``:free``."""
        target_model = model_config.upstream_model or model_config.id

        raw_effort: Optional[str] = None
        if request.reasoning_effort:
            raw_effort = request.reasoning_effort
        elif request.extra_body:
            raw_effort = request.extra_body.get("reasoning_effort") or request.extra_body.get(
                "reasoning"
            )

        requested_model = (request.model or "").strip().lower()
        for suffix, level in REASONING_SUFFIXES:
            if requested_model.endswith(suffix):
                if raw_effort is None:
                    raw_effort = level
                break

        effort = self._normalize_effort(raw_effort)
        if effort is None and raw_effort is None:
            effort = self._normalize_effort(variables.get("reasoning_effort"))

        # Strip only reasoning suffixes (never the required :free marker).
        for suffix, _ in REASONING_SUFFIXES:
            if target_model.lower().endswith(suffix):
                target_model = target_model[: -len(suffix)]
                break

        return target_model, effort

    def _build_headers(
        self, api_key: str, variables: Dict[str, Any]
    ) -> Dict[str, str]:
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        site_url = str(variables.get("site_url") or "").strip()
        app_name = str(variables.get("app_name") or "").strip()
        if site_url:
            headers["HTTP-Referer"] = site_url
        if app_name:
            headers["X-Title"] = app_name
        return headers

    @staticmethod
    def _apply_cache_fields(body: Dict[str, Any], request: ChatCompletionRequest) -> None:
        """Forward prompt-cache hints and an optional Anthropic-style breakpoint."""
        extra = request.extra_body if isinstance(request.extra_body, dict) else {}
        for key in ("prompt_cache_key", "prompt_cache_retention"):
            value = getattr(request, key, None)
            if value is None:
                value = extra.get(key)
            if value is not None:
                body[key] = value

        cache_control = getattr(request, "cache_control", None) or extra.get("cache_control")
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
        """Build the OpenRouter Chat Completions body (also used by tests)."""
        target_model, effort = self._resolve_reasoning(request, model_config, variables)

        body: Dict[str, Any] = {
            "model": target_model,
            "messages": [
                message.model_dump(exclude_none=True) for message in request.messages
            ],
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
        if request.stop is not None:
            body["stop"] = request.stop
        if request.tools is not None:
            body["tools"] = request.tools
        if request.tool_choice is not None:
            body["tool_choice"] = request.tool_choice
        if request.parallel_tool_calls is not None:
            body["parallel_tool_calls"] = request.parallel_tool_calls
        if request.response_format is not None:
            body["response_format"] = request.response_format

        if effort and effort != "none":
            body["reasoning"] = {"effort": effort}

        if request.stream:
            # Always request usage so prompt-cache counters reach the request
            # log. The router strips it again unless the client opted in.
            stream_options = dict(request.stream_options or {})
            stream_options["include_usage"] = True
            body["stream_options"] = stream_options

        self._apply_cache_fields(body, request)
        return body

    def _mock_response(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        account: Optional[Account] = None,
    ) -> ChatCompletionResponse:
        model_name = model_config.id
        account_label = f" via account '{account.name}'" if account else ""

        if request.tools and not any(message.role == "tool" for message in request.messages[-2:]):
            fn = (
                request.tools[0].get("function", {})
                if "function" in request.tools[0]
                else request.tools[0]
            )
            call_id = f"call_openrouter_{uuid.uuid4().hex[:8]}"
            tool_name = fn.get("name", "test_tool")
            mock_args = "{}"
            props = fn.get("parameters", {}).get("properties", {})
            if props:
                first_key = next(iter(props.keys()))
                mock_args = json.dumps({first_key: "test_value"})

            return ChatCompletionResponse(
                id=f"chatcmpl-openrouter-{uuid.uuid4().hex[:12]}",
                created=int(time.time()),
                model=model_name,
                choices=[
                    ChatChoice(
                        index=0,
                        message=ChatMessage(
                            role="assistant",
                            content=None,
                            tool_calls=[
                                {
                                    "id": call_id,
                                    "type": "function",
                                    "function": {"name": tool_name, "arguments": mock_args},
                                }
                            ],
                        ),
                        finish_reason="tool_calls",
                    )
                ],
                usage=CompletionUsage(
                    prompt_tokens=12,
                    completion_tokens=22,
                    total_tokens=34,
                ),
            )

        content = (
            f"Simulated response from OpenRouter ({model_name}){account_label}. "
            "Configure a real OpenRouter API key to reach the upstream gateway."
        )
        if any(message.role == "tool" for message in request.messages):
            content = (
                f"Simulated response from OpenRouter ({model_name}){account_label} "
                "with tool output processed."
            )

        return ChatCompletionResponse(
            id=f"chatcmpl-openrouter-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=model_name,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=content),
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=12,
                completion_tokens=22,
                total_tokens=34,
            ),
        )

    async def _mock_stream(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        account: Optional[Account] = None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        model_name = model_config.id
        chunk_id = f"chatcmpl-openrouter-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        account_label = f" via account '{account.name}'" if account else ""

        if request.tools and not any(message.role == "tool" for message in request.messages[-2:]):
            fn = (
                request.tools[0].get("function", {})
                if "function" in request.tools[0]
                else request.tools[0]
            )
            call_id = f"call_openrouter_{uuid.uuid4().hex[:8]}"
            tool_name = fn.get("name", "test_tool")
            mock_args = "{}"
            props = fn.get("parameters", {}).get("properties", {})
            if props:
                first_key = next(iter(props.keys()))
                mock_args = json.dumps({first_key: "test_value"})

            yield ChatCompletionChunk(
                id=chunk_id,
                created=created,
                model=model_name,
                choices=[
                    ChatChoiceDelta(
                        index=0,
                        delta={
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": call_id,
                                    "type": "function",
                                    "function": {"name": tool_name, "arguments": ""},
                                }
                            ],
                        },
                        finish_reason=None,
                    )
                ],
            )
            for piece in (mock_args[: len(mock_args) // 2], mock_args[len(mock_args) // 2 :]):
                if piece:
                    yield ChatCompletionChunk(
                        id=chunk_id,
                        created=created,
                        model=model_name,
                        choices=[
                            ChatChoiceDelta(
                                index=0,
                                delta={
                                    "tool_calls": [
                                        {"index": 0, "function": {"arguments": piece}}
                                    ]
                                },
                                finish_reason=None,
                            )
                        ],
                    )
            yield ChatCompletionChunk(
                id=chunk_id,
                created=created,
                model=model_name,
                choices=[ChatChoiceDelta(index=0, delta={}, finish_reason="tool_calls")],
            )
            return

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

        for token in [
            "Simulated ",
            "stream ",
            "response ",
            f"from OpenRouter for {model_name}{account_label}.",
        ]:
            yield ChatCompletionChunk(
                id=chunk_id,
                created=created,
                model=model_name,
                choices=[
                    ChatChoiceDelta(index=0, delta={"content": token}, finish_reason=None)
                ],
            )

        yield ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=model_name,
            choices=[ChatChoiceDelta(index=0, delta={}, finish_reason="stop")],
        )

    async def _stream_request(
        self,
        url: str,
        headers: Dict[str, str],
        body: Dict[str, Any],
        model_config: PluginModelConfig,
        account: Optional[Account] = None,
        proxy: Optional[str] = None,
        timeout: float = 120.0,
    ) -> AsyncIterator[ChatCompletionChunk]:
        async with httpx.AsyncClient(timeout=timeout, proxy=proxy) as client:
            async with client.stream("POST", url, headers=headers, json=body) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        parsed = json.loads(data)
                    except Exception:
                        continue
                    if isinstance(parsed, dict) and parsed.get("error"):
                        error = parsed["error"]
                        message = (
                            error.get("message")
                            if isinstance(error, dict)
                            else str(error)
                        )
                        raise RuntimeError(message or "OpenRouter streaming error")
                    try:
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
        api_key = (account.api_key if account else "") or str(variables.get("api_key") or "")
        base_url = (
            account.base_url
            if (account and account.base_url)
            else str(variables.get("base_url") or OPENROUTER_BASE_URL)
        ).rstrip("/")

        if self._is_test_mode(api_key):
            if request.stream:
                return self._mock_stream(request, model_config, account=account)
            return self._mock_response(request, model_config, account=account)

        headers = self._build_headers(api_key, variables)
        body = self._build_body(request, model_config, variables)
        url = f"{base_url}/chat/completions"
        proxy = (
            str(variables.get("proxy") or (getattr(account, "proxy_url", None) or "")).strip()
            or None
        )
        timeout = float(variables.get("timeout") or 120.0)

        if request.stream:
            return self._stream_request(
                url,
                headers,
                body,
                model_config,
                account=account,
                proxy=proxy,
                timeout=timeout,
            )

        async with httpx.AsyncClient(timeout=timeout, proxy=proxy) as client:
            response = await client.post(url, headers=headers, json=body)
            response.raise_for_status()
            data = response.json()
        return ChatCompletionResponse.model_validate(data)


plugin = OpenRouterPlugin()
