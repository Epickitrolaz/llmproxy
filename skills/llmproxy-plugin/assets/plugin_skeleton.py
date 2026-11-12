"""
LLMProxy provider plugin skeleton.

Copy this file to `templates/<provider>.py` for a reusable built-in template, or
to `plugins/<provider>.py` for a local runtime plugin. Then replace the metadata,
models, URL/payload construction, and response parsing with your upstream API.

The plugin is intentionally self-contained: it only imports the LLMProxy contract,
Pydantic schemas, httpx, and the standard library. It supports streaming,
non-streaming, tool calls, usage reporting, account credentials, proxy routing,
and a deterministic simulation mode for tests.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncIterator, Dict, Optional, Union

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


class MyProviderPlugin(BasePlugin):
    """Replace this class name and implementation with your provider."""

    def __init__(self):
        self._metadata = PluginMetadata(
            id="my_provider",
            name="My Provider",
            description="Short description shown in the provider dashboard.",
            version="1.0.0",
            author="your-name",
            enabled=True,
            filename="my_provider.py",
            variables=[
                PluginVariable(
                    name="timeout",
                    label="Timeout (seconds)",
                    type="number",
                    default=60,
                    description="Request timeout in seconds",
                ),
                PluginVariable(
                    name="base_url",
                    label="Default Base URL",
                    type="string",
                    default="https://api.example.com/v1",
                    description="Upstream OpenAI-compatible base URL",
                ),
                PluginVariable(
                    name="api_key",
                    label="Default API Key",
                    type="secret",
                    default="",
                    description="Used when no account is configured. Accounts override this.",
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
                    description="Cooldown after an account returns a rate limit",
                ),
                PluginVariable(
                    name="account_switch_wait",
                    label="Switch Wait Delay (s)",
                    type="number",
                    default=0,
                    description="Optional delay before switching accounts on non-rate-limit failures",
                ),
                PluginVariable(
                    name="max_retries",
                    label="Max Plugin Retries",
                    type="number",
                    default=2,
                    description="Retries performed inside the plugin itself",
                ),
                PluginVariable(
                    name="proxy_enabled",
                    label="Enable External Proxy",
                    type="boolean",
                    default=False,
                    description="Route provider requests through external proxies",
                ),
                PluginVariable(
                    name="proxy_strategy",
                    label="Proxy Strategy",
                    type="select",
                    default="sequential",
                    options=["sequential", "random", "specific", "switch_on_limit"],
                    description="How to choose a proxy from the proxy pool",
                ),
                PluginVariable(
                    name="specific_proxy_id",
                    label="Specific Proxy ID / Tag",
                    type="string",
                    default="",
                    description="Proxy ID, URL, or tag used by the 'specific' strategy",
                ),
                PluginVariable(
                    name="proxy",
                    label="Direct Proxy URL",
                    type="string",
                    default="",
                    description="Direct HTTP/SOCKS proxy URL. The router injects this per request.",
                ),
            ],
            models=[
                PluginModelConfig(
                    id="my-model",
                    name="My Model",
                    upstream_model="my-model",
                    context_window=128000,
                    max_tokens=4096,
                ),
            ],
        )

    @property
    def metadata(self) -> PluginMetadata:
        return self._metadata

    # ------------------------------------------------------------------
    # Simulation mode: keeps dashboard tests and offline development usable
    # ------------------------------------------------------------------

    def _is_test_mode(self, api_key: str) -> bool:
        key = (api_key or "").lower().strip()
        prefixes = ("test", "mock", "sk-test", "dummy", "fake", "offline", "sample")
        return not key or any(key.startswith(prefix) for prefix in prefixes) or key in (
            "test-key",
            "test_key",
            "secret123",
        )

    def _format_details(
        self,
        model_name: str,
        account: Optional[Account],
        prompt_tokens: int,
        completion_tokens: int,
    ) -> str:
        now_str = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        account_status = (
            "No account configured (Simulation Mode)"
            if (not account or account.id.startswith("default-") or not account.api_key)
            else f"Account '{account.name}' (Test mode)"
        )
        return (
            f"\n\n---\n"
            f"**Execution Details:**\n"
            f"• **Timestamp:** {now_str}\n"
            f"• **Model:** {model_name}\n"
            f"• **Tokens:** {prompt_tokens} prompt, {completion_tokens} completion "
            f"({prompt_tokens + completion_tokens} total)\n"
            f"• **Account status:** {account_status}"
        )

    def _mock_response(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        account: Optional[Account] = None,
    ) -> ChatCompletionResponse:
        model_name = request.model or model_config.id

        if request.tools and not any(m.role in ("tool", "function") for m in request.messages[-2:]):
            fn = request.tools[0].get("function", {}) if "function" in request.tools[0] else request.tools[0]
            properties = fn.get("parameters", {}).get("properties", {})
            arguments = json.dumps({next(iter(properties)): "test_value"}) if properties else "{}"
            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
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
                                    "id": f"call_{uuid.uuid4().hex[:8]}",
                                    "type": "function",
                                    "function": {
                                        "name": fn.get("name", "test_tool"),
                                        "arguments": arguments,
                                    },
                                }
                            ],
                        ),
                        finish_reason="tool_calls",
                    )
                ],
                usage=CompletionUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
            )

        prompt_tokens = 10
        completion_tokens = 20
        content = f"Simulated response from My Provider ({model_name})."
        content += self._format_details(model_name, account, prompt_tokens, completion_tokens)
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
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
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    async def _mock_stream(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        account: Optional[Account] = None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        model_name = request.model or model_config.id
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        if request.tools and not any(m.role in ("tool", "function") for m in request.messages[-2:]):
            fn = request.tools[0].get("function", {}) if "function" in request.tools[0] else request.tools[0]
            properties = fn.get("parameters", {}).get("properties", {})
            arguments = json.dumps({next(iter(properties)): "test_value"}) if properties else "{}"
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
                                    "id": f"call_{uuid.uuid4().hex[:8]}",
                                    "type": "function",
                                    "function": {
                                        "name": fn.get("name", "test_tool"),
                                        "arguments": "",
                                    },
                                }
                            ],
                        },
                    )
                ],
            )
            for piece in (arguments[: len(arguments) // 2], arguments[len(arguments) // 2 :]):
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
                            )
                        ],
                    )
            yield ChatCompletionChunk(
                id=chunk_id,
                created=created,
                model=model_name,
                choices=[ChatChoiceDelta(index=0, delta={}, finish_reason="tool_calls")],
                usage=CompletionUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
            )
            return

        yield ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=model_name,
            choices=[ChatChoiceDelta(index=0, delta={"role": "assistant", "content": ""})],
        )
        for token in ("Simulated ", "stream ", "response ", "from My Provider."):
            yield ChatCompletionChunk(
                id=chunk_id,
                created=created,
                model=model_name,
                choices=[ChatChoiceDelta(index=0, delta={"content": token})],
            )
        details = self._format_details(model_name, account, 10, 20)
        yield ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=model_name,
            choices=[ChatChoiceDelta(index=0, delta={"content": details})],
        )
        yield ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=model_name,
            choices=[ChatChoiceDelta(index=0, delta={}, finish_reason="stop")],
            usage=CompletionUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )

    # ------------------------------------------------------------------
    # Upstream request helpers
    # ------------------------------------------------------------------

    def _resolve_connection(
        self,
        variables: Dict[str, Any],
        account: Optional[Account],
    ) -> tuple[str, str, Optional[str]]:
        api_key = account.api_key if account else str(variables.get("api_key") or "")
        base_url = (
            account.base_url
            if (account and account.base_url)
            else str(variables.get("base_url") or "https://api.example.com/v1")
        ).rstrip("/")
        if base_url.endswith("/chat/completions"):
            url = base_url
        else:
            url = f"{base_url}/chat/completions"

        proxy = str(
            variables.get("proxy")
            or (getattr(account, "proxy_url", None) if account else "")
            or ""
        ).strip() or None
        return api_key, url, proxy

    def _build_payload(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": model_config.upstream_model or model_config.id,
            "messages": [message.model_dump(exclude_none=True) for message in request.messages],
            "stream": bool(request.stream),
        }
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.top_p is not None:
            body["top_p"] = request.top_p
        max_tokens = request.max_tokens if request.max_tokens is not None else request.max_completion_tokens
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
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
        if request.stream and request.stream_options is not None:
            body["stream_options"] = request.stream_options
        return body

    async def _complete(
        self,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        timeout: float,
        proxy: Optional[str],
    ) -> ChatCompletionResponse:
        async with httpx.AsyncClient(timeout=timeout, proxy=proxy) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return ChatCompletionResponse.model_validate(response.json())

    async def _stream(
        self,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        timeout: float,
        proxy: Optional[str],
    ) -> AsyncIterator[ChatCompletionChunk]:
        async with httpx.AsyncClient(timeout=timeout, proxy=proxy) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        yield ChatCompletionChunk.model_validate(json.loads(data))
                    except Exception:
                        continue

    # ------------------------------------------------------------------
    # BasePlugin entry point
    # ------------------------------------------------------------------

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        variables: Dict[str, Any],
        account: Optional[Account] = None,
    ) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionChunk]]:
        api_key, url, proxy = self._resolve_connection(variables, account)

        if self._is_test_mode(api_key):
            if request.stream:
                return self._mock_stream(request, model_config, account=account)
            return self._mock_response(request, model_config, account=account)

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = self._build_payload(request, model_config)
        timeout = float(variables.get("timeout") or 60.0)

        if request.stream:
            return self._stream(url, headers, payload, timeout, proxy)
        return await self._complete(url, headers, payload, timeout, proxy)


plugin = MyProviderPlugin()
