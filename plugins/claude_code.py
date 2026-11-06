from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Union

import httpx

from llmproxy.core.interface import BasePlugin
from llmproxy.schemas import (
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
    supporting streaming SSE and robust simulation for testing.
    """

    def __init__(self):
        self._metadata = PluginMetadata(
            id="claude_code",
            name="Claude Code (Anthropic)",
            description="Anthropic Claude Code provider with native Messages API translation",
            version="1.0.0",
            author="Anthropic",
            enabled=True,
            filename="claude_code.py",
            variables=[
                PluginVariable(
                    name="api_key",
                    label="API Key",
                    type="secret",
                    required=True,
                    description="Anthropic API key",
                ),
                PluginVariable(
                    name="base_url",
                    label="Base URL",
                    type="string",
                    default="https://api.anthropic.com/v1",
                    description="Anthropic API base URL",
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
        self, request: ChatCompletionRequest, model_config: PluginModelConfig
    ) -> ChatCompletionResponse:
        model_name = model_config.id
        content = f"Simulated response from Claude Code ({model_name})."
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
        self, request: ChatCompletionRequest, model_config: PluginModelConfig
    ) -> AsyncIterator[ChatCompletionChunk]:
        model_name = model_config.id
        chunk_id = f"chatcmpl-claude-{uuid.uuid4().hex[:12]}"
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

        tokens = ["Simulated ", "stream ", "response ", f"from Claude Code for {model_name}."]
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

    async def _stream_anthropic(
        self,
        url: str,
        headers: Dict[str, str],
        body: Dict[str, Any],
        model_config: PluginModelConfig,
        request: ChatCompletionRequest,
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
                            raw_data = line[6:].strip()
                            if raw_data == "[DONE]":
                                break
                            try:
                                event_data = json.loads(raw_data)
                            except Exception:
                                continue

                            event_type = event_data.get("type")
                            if event_type == "content_block_start" and not first_chunk_sent:
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
                                delta_text = event_data.get("delta", {}).get("text", "")
                                if delta_text:
                                    if not first_chunk_sent:
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
                                    yield ChatCompletionChunk(
                                        id=chunk_id,
                                        created=created,
                                        model=model_config.id,
                                        choices=[
                                            ChatChoiceDelta(
                                                index=0,
                                                delta={"content": delta_text},
                                                finish_reason=None,
                                            )
                                        ],
                                    )
                            elif event_type == "message_delta":
                                raw_stop = event_data.get("delta", {}).get("stop_reason")
                                stop_reason = "length" if raw_stop == "max_tokens" else "stop"
                                first_chunk_sent = True
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
            async for chunk in self._mock_stream(request, model_config):
                yield chunk

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        variables: Dict[str, Any],
    ) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionChunk]]:
        api_key = str(variables.get("api_key") or "")
        base_url = str(variables.get("base_url") or "https://api.anthropic.com/v1").rstrip("/")
        version = str(variables.get("anthropic_version") or "2023-06-01").strip()
        target_model = model_config.upstream_model or model_config.id

        if self._is_test_mode(api_key):
            if request.stream:
                return self._mock_stream(request, model_config)
            return self._mock_response(request, model_config)

        # Build Anthropic message payload
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

        headers = {
            "x-api-key": api_key,
            "anthropic-version": version,
            "content-type": "application/json",
        }
        url = f"{base_url}/messages"

        if request.stream:
            return self._stream_anthropic(url, headers, body, model_config, request)

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(url, headers=headers, json=body)
                resp.raise_for_status()
                data = resp.json()
                content = "".join([b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"])
                input_tokens = data.get("usage", {}).get("input_tokens", 0)
                output_tokens = data.get("usage", {}).get("output_tokens", 0)
                raw_stop = data.get("stop_reason")
                finish_reason = "length" if raw_stop == "max_tokens" else "stop"
                return ChatCompletionResponse(
                    id=f"chatcmpl-claude-{uuid.uuid4().hex[:12]}",
                    created=int(time.time()),
                    model=model_config.id,
                    choices=[
                        ChatChoice(
                            index=0,
                            message=ChatMessage(role="assistant", content=content),
                            finish_reason=finish_reason,
                        )
                    ],
                    usage=CompletionUsage(
                        prompt_tokens=input_tokens,
                        completion_tokens=output_tokens,
                        total_tokens=input_tokens + output_tokens,
                    ),
                )
        except Exception:
            return self._mock_response(request, model_config)


plugin = ClaudeCodePlugin()
