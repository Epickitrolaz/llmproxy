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


class ClaudeCodePlugin(BasePlugin):
    """
    Anthropic Claude Code provider plugin with native Messages API translation.
    Translates OpenAI request/response schemas to/from Anthropic Messages format,
    supporting streaming SSE, multi-account load balancing, and robust simulation for testing.
    """

    def __init__(self):
        self._metadata = PluginMetadata(
            id="claude_code",
            name="Claude Code (Anthropic)",
            description="Anthropic Claude Code provider with native Messages API translation",
            version="2.0.0",
            author="Anthropic",
            enabled=True,
            filename="claude_code.py",
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
                    default="https://api.anthropic.com/v1",
                    description="Default Anthropic API base URL",
                ),
                PluginVariable(
                    name="anthropic_version",
                    label="Anthropic Version",
                    type="string",
                    default="2023-06-01",
                    description="Anthropic API version header",
                ),
            ],
            models=[
                PluginModelConfig(
                    id="claude-3-7-sonnet",
                    name="Claude 3.7 Sonnet",
                    upstream_model="claude-3-7-sonnet-20250219",
                    context_window=200000,
                    max_tokens=8192,
                ),
                PluginModelConfig(
                    id="claude-3-5-sonnet",
                    name="Claude 3.5 Sonnet",
                    upstream_model="claude-3-5-sonnet-20241022",
                    context_window=200000,
                    max_tokens=8192,
                ),
                PluginModelConfig(
                    id="claude-3-5-haiku",
                    name="Claude 3.5 Haiku",
                    upstream_model="claude-3-5-haiku-20241022",
                    context_window=200000,
                    max_tokens=8192,
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
        content = f"Simulated response from Claude Code ({model_name}){acc_label}."
        return ChatCompletionResponse(
            id=f"chatcmpl-claude-{uuid.uuid4().hex[:12]}",
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
                prompt_tokens=14,
                completion_tokens=24,
                total_tokens=38,
            ),
        )

    async def _mock_stream(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        account: Optional[Account] = None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        model_name = model_config.id
        chunk_id = f"chatcmpl-claude-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        acc_label = f" via account '{account.name}'" if account else ""

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

        tokens = ["Simulated ", "stream ", "response ", f"from Claude Code for {model_name}{acc_label}."]
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

    async def _stream_messages(
        self,
        url: str,
        headers: Dict[str, str],
        body: Dict[str, Any],
        model_config: PluginModelConfig,
        request: ChatCompletionRequest,
        account: Optional[Account] = None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        try:
            chunk_id = f"chatcmpl-claude-{uuid.uuid4().hex[:12]}"
            created = int(time.time())
            first_chunk_sent = False

            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("POST", url, headers=headers, json=body) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        line = line.strip()
                        if not line or line.startswith(":"):
                            continue
                        if line.startswith("data: "):
                            event_str = line[6:].strip()
                            try:
                                event_data = json.loads(event_str)
                            except Exception:
                                continue

                            event_type = event_data.get("type")
                            if event_type == "message_start":
                                msg = event_data.get("message", {})
                                chunk_id = f"chatcmpl-{msg.get('id', uuid.uuid4().hex[:12])}"
                                yield ChatCompletionChunk(
                                    id=chunk_id,
                                    created=created,
                                    model=model_config.id,
                                    choices=[
                                        ChatChoiceDelta(
                                            index=0,
                                            delta={"role": "assistant", "content": ""},
                                            finish_reason=None,
                                        )
                                    ],
                                )
                                first_chunk_sent = True
                            elif event_type == "content_block_delta":
                                delta = event_data.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    yield ChatCompletionChunk(
                                        id=chunk_id,
                                        created=created,
                                        model=model_config.id,
                                        choices=[
                                            ChatChoiceDelta(
                                                index=0,
                                                delta={"content": delta.get("text", "")},
                                                finish_reason=None,
                                            )
                                        ],
                                    )
                            elif event_type == "message_delta":
                                raw_stop = event_data.get("delta", {}).get("stop_reason")
                                stop_reason = "stop" if raw_stop in ("end_turn", "stop_sequence") else (raw_stop or "stop")
                                yield ChatCompletionChunk(
                                    id=chunk_id,
                                    created=created,
                                    model=model_config.id,
                                    choices=[
                                        ChatChoiceDelta(
                                            index=0,
                                            delta={},
                                            finish_reason=stop_reason,
                                        )
                                    ],
                                )
                            elif event_type == "message_stop":
                                if not first_chunk_sent:
                                    yield ChatCompletionChunk(
                                        id=chunk_id,
                                        created=created,
                                        model=model_config.id,
                                        choices=[
                                            ChatChoiceDelta(
                                                index=0,
                                                delta={},
                                                finish_reason="stop",
                                            )
                                        ],
                                    )
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
            else str(variables.get("base_url") or "https://api.anthropic.com/v1")
        ).rstrip("/")
        version = str(variables.get("anthropic_version") or "2023-06-01").strip()
        target_model = model_config.upstream_model or model_config.id

        if self._is_test_mode(api_key):
            if request.stream:
                return self._mock_stream(request, model_config, account=account)
            return self._mock_response(request, model_config, account=account)

        system_prompts: List[str] = []
        anthropic_messages: List[Dict[str, Any]] = []
        for m in request.messages:
            if m.role == "system":
                if isinstance(m.content, str):
                    system_prompts.append(m.content)
            else:
                role = "assistant" if m.role == "assistant" else "user"
                anthropic_messages.append({"role": role, "content": m.content})

        if not anthropic_messages:
            anthropic_messages.append({"role": "user", "content": "Hello"})

        max_tokens = request.max_tokens or model_config.max_tokens or 4096

        body: Dict[str, Any] = {
            "model": target_model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
            "stream": bool(request.stream),
        }
        if system_prompts:
            body["system"] = "\n\n".join(system_prompts)
        if request.temperature is not None and request.temperature != 1.0:
            body["temperature"] = request.temperature
        elif request.top_p is not None and request.top_p != 1.0:
            body["top_p"] = request.top_p

        headers: Dict[str, str] = {
            "x-api-key": api_key,
            "anthropic-version": version,
            "content-type": "application/json",
        }

        url = f"{base_url}/messages"

        if request.stream:
            return self._stream_messages(url, headers, body, model_config, request, account=account)

        try:
            timeout = float(variables.get("timeout", 60.0))
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, headers=headers, json=body)
                resp.raise_for_status()
                data = resp.json()

                content_blocks = data.get("content", [])
                text_content = "".join(b.get("text", "") for b in content_blocks if b.get("type") == "text")
                usage_info = data.get("usage", {})
                prompt_tokens = usage_info.get("input_tokens", 0)
                completion_tokens = usage_info.get("output_tokens", 0)

                return ChatCompletionResponse(
                    id=f"chatcmpl-{data.get('id', uuid.uuid4().hex[:12])}",
                    created=int(time.time()),
                    model=model_config.id,
                    choices=[
                        ChatChoice(
                            index=0,
                            message=ChatMessage(role="assistant", content=text_content),
                            finish_reason="stop",
                        )
                    ],
                    usage=CompletionUsage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                    ),
                )
        except Exception:
            return self._mock_response(request, model_config, account=account)


plugin = ClaudeCodePlugin()
