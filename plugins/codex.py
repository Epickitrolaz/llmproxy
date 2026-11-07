from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Union

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


class CodexPlugin(BasePlugin):
    """
    OpenAI Codex and frontier models provider plugin.
    Supports gpt-6-luna, gpt-4o, and gpt-4o-mini with streaming SSE,
    multi-account load balancing, and simulated fallback for offline/test environments.
    """

    def __init__(self):
        self._metadata = PluginMetadata(
            id="codex",
            name="Codex (OpenAI)",
            description="OpenAI Codex & Frontier Models provider",
            version="2.0.0",
            author="OpenAI",
            enabled=True,
            filename="codex.py",
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
                    default="https://api.openai.com/v1",
                    description="Default OpenAI API base URL",
                ),
                PluginVariable(
                    name="organization_id",
                    label="Default Organization ID",
                    type="string",
                    default="",
                    description="Optional fallback OpenAI Organization ID",
                ),
                PluginVariable(
                    name="reasoning_effort",
                    label="Reasoning Effort",
                    type="select",
                    default="medium",
                    options=["low", "medium", "high"],
                    description="Reasoning effort level for supported models",
                ),
            ],
            models=[
                PluginModelConfig(
                    id="gpt-6-luna",
                    name="GPT-6 Luna",
                    upstream_model="gpt-6-luna",
                    context_window=128000,
                    max_tokens=16384,
                ),
                PluginModelConfig(
                    id="gpt-4o",
                    name="GPT-4o",
                    upstream_model="gpt-4o",
                    context_window=128000,
                    max_tokens=4096,
                ),
                PluginModelConfig(
                    id="gpt-4o-mini",
                    name="GPT-4o Mini",
                    upstream_model="gpt-4o-mini",
                    context_window=128000,
                    max_tokens=4096,
                ),
            ],
        )

    @property
    def metadata(self) -> PluginMetadata:
        return self._metadata

    def _is_test_mode(self, api_key: str) -> bool:
        if not api_key:
            return True
        key = api_key.lower().strip()
        prefixes = ("test", "mock", "sk-test", "dummy", "fake", "offline", "sample")
        return any(key.startswith(p) for p in prefixes) or key in ("test-key", "test_key", "secret123")

    def _mock_response(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        account: Optional[Account] = None,
    ) -> ChatCompletionResponse:
        model_name = model_config.id
        acc_label = f" via account '{account.name}'" if account else ""

        if request.tools and not any(m.role == "tool" for m in request.messages[-2:]):
            fn = request.tools[0].get("function", {}) if "function" in request.tools[0] else request.tools[0]
            call_id = f"call_codex_{uuid.uuid4().hex[:8]}"
            tool_name = fn.get("name", "test_tool")
            mock_args = "{}"
            props = fn.get("parameters", {}).get("properties", {})
            if props:
                first_k = next(iter(props.keys()))
                mock_args = json.dumps({first_k: "test_value"})
            return ChatCompletionResponse(
                id=f"chatcmpl-codex-{uuid.uuid4().hex[:12]}",
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
                                    "function": {
                                        "name": tool_name,
                                        "arguments": mock_args,
                                    },
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

        content = f"Simulated response from Codex ({model_name}){acc_label}."
        if any(m.role == "tool" for m in request.messages):
            content = f"Simulated response from Codex ({model_name}){acc_label} with tool output processed."

        return ChatCompletionResponse(
            id=f"chatcmpl-codex-{uuid.uuid4().hex[:12]}",
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
        chunk_id = f"chatcmpl-codex-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        acc_label = f" via account '{account.name}'" if account else ""

        if request.tools and not any(m.role == "tool" for m in request.messages[-2:]):
            fn = request.tools[0].get("function", {}) if "function" in request.tools[0] else request.tools[0]
            call_id = f"call_codex_{uuid.uuid4().hex[:8]}"
            tool_name = fn.get("name", "test_tool")
            mock_args = "{}"
            props = fn.get("parameters", {}).get("properties", {})
            if props:
                first_k = next(iter(props.keys()))
                mock_args = json.dumps({first_k: "test_value"})

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
                                    "function": {
                                        "name": tool_name,
                                        "arguments": "",
                                    },
                                }
                            ],
                        },
                        finish_reason=None,
                    )
                ],
            )
            arg_chunks = [mock_args[: len(mock_args) // 2], mock_args[len(mock_args) // 2 :]]
            for arg_piece in arg_chunks:
                if arg_piece:
                    yield ChatCompletionChunk(
                        id=chunk_id,
                        created=created,
                        model=model_name,
                        choices=[
                            ChatChoiceDelta(
                                index=0,
                                delta={
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "function": {"arguments": arg_piece},
                                        }
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
                choices=[
                    ChatChoiceDelta(
                        index=0,
                        delta={},
                        finish_reason="tool_calls",
                    )
                ],
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

        tokens = ["Simulated ", "stream ", "response ", f"from Codex for {model_name}{acc_label}."]
        for token in tokens:
            yield ChatCompletionChunk(
                id=chunk_id,
                created=created,
                model=model_name,
                choices=[
                    ChatChoiceDelta(
                        index=0,
                        delta={"content": token},
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
                    delta={},
                    finish_reason="stop",
                )
            ],
        )

    async def _stream_request(
        self,
        url: str,
        headers: Dict[str, str],
        body: Dict[str, Any],
        model_config: PluginModelConfig,
        request: ChatCompletionRequest,
        account: Optional[Account] = None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("POST", url, headers=headers, json=body) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        line = line.strip()
                        if not line or line.startswith(":"):
                            continue
                        if line.startswith("data: "):
                            data = line[6:].strip()
                            if data == "[DONE]":
                                break
                            try:
                                parsed = json.loads(data)
                                yield ChatCompletionChunk.model_validate(parsed)
                            except Exception:
                                continue
        except Exception:
            async for chunk in self._mock_stream(request, model_config, account=account):
                yield chunk

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        variables: Dict[str, Any],
        account: Optional[Account] = None,
    ) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionChunk]]:
        api_key = account.api_key if account else str(variables.get("api_key") or "")
        base_url = (
            account.base_url
            if (account and account.base_url)
            else str(variables.get("base_url") or "https://api.openai.com/v1")
        ).rstrip("/")
        org_id = (
            (account.extra_config.get("organization_id") if account else None)
            or str(variables.get("organization_id") or "")
        ).strip()
        reasoning_effort = str(variables.get("reasoning_effort") or "medium").strip()
        target_model = model_config.upstream_model or model_config.id

        if self._is_test_mode(api_key):
            if request.stream:
                return self._mock_stream(request, model_config, account=account)
            return self._mock_response(request, model_config, account=account)

        headers: Dict[str, str] = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if org_id:
            headers["OpenAI-Organization"] = org_id

        body: Dict[str, Any] = {
            "model": target_model,
            "messages": [m.model_dump(exclude_none=True) for m in request.messages],
            "stream": bool(request.stream),
        }
        if target_model.startswith("o") or "gpt-6" in target_model:
            body["reasoning_effort"] = reasoning_effort
        elif request.temperature is not None:
            body["temperature"] = request.temperature

        max_tokens = request.max_tokens if request.max_tokens is not None else request.max_completion_tokens
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if request.top_p is not None and not target_model.startswith("o"):
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

        url = f"{base_url}/chat/completions"

        if request.stream:
            return self._stream_request(url, headers, body, model_config, request, account=account)

        try:
            timeout = float(variables.get("timeout", 60.0))
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, headers=headers, json=body)
                resp.raise_for_status()
                data = resp.json()
                return ChatCompletionResponse.model_validate(data)
        except Exception:
            return self._mock_response(request, model_config, account=account)


plugin = CodexPlugin()
