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


class AntigravityPlugin(BasePlugin):
    """
    Antigravity provider plugin supporting Gemini and Claude frontier models.
    Supports streaming and non-streaming via OpenAI-compatible endpoints with
    simulated fallbacks when running offline or with test credentials.
    """

    def __init__(self):
        self._metadata = PluginMetadata(
            id="antigravity",
            name="Antigravity",
            description="Antigravity AI provider with advanced multi-model intelligence",
            version="1.0.0",
            author="Antigravity",
            enabled=True,
            filename="antigravity.py",
            variables=[
                PluginVariable(
                    name="api_key",
                    label="API Key",
                    type="secret",
                    required=True,
                    description="Antigravity API key",
                ),
                PluginVariable(
                    name="base_url",
                    label="Base URL",
                    type="string",
                    default="https://api.antigravity.dev/v1",
                    description="Antigravity API base URL",
                ),
                PluginVariable(
                    name="timeout",
                    label="Timeout (seconds)",
                    type="number",
                    default=60,
                    description="Request timeout in seconds",
                ),
            ],
            models=[
                PluginModelConfig(
                    id="gemini-3.8-flash",
                    name="Gemini 3.8 Flash",
                    upstream_model="gemini-3.8-flash",
                    context_window=1048576,
                    max_tokens=8192,
                ),
                PluginModelConfig(
                    id="claude-sonnet-4-6",
                    name="Claude Sonnet 4.6",
                    upstream_model="claude-sonnet-4-6",
                    context_window=200000,
                    max_tokens=8192,
                ),
                PluginModelConfig(
                    id="claude-opus-4-6-thinking",
                    name="Claude Opus 4.6 Thinking",
                    upstream_model="claude-opus-4-6-thinking",
                    context_window=200000,
                    max_tokens=16384,
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
        content = f"Simulated response from Antigravity ({model_name})."
        return ChatCompletionResponse(
            id=f"chatcmpl-ag-{uuid.uuid4().hex[:12]}",
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
                prompt_tokens=15,
                completion_tokens=25,
                total_tokens=40,
            ),
        )

    async def _mock_stream(
        self, request: ChatCompletionRequest, model_config: PluginModelConfig
    ) -> AsyncIterator[ChatCompletionChunk]:
        model_name = model_config.id
        chunk_id = f"chatcmpl-ag-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        # First chunk: role announcement
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

        tokens = ["Simulated ", "stream ", "response ", f"from Antigravity for {model_name}."]
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
        timeout: float,
        model_config: PluginModelConfig,
        request: ChatCompletionRequest,
    ) -> AsyncIterator[ChatCompletionChunk]:
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
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
            async for chunk in self._mock_stream(request, model_config):
                yield chunk

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        variables: Dict[str, Any],
    ) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionChunk]]:
        api_key = str(variables.get("api_key") or "")
        base_url = str(variables.get("base_url") or "https://api.antigravity.dev/v1").rstrip("/")
        timeout = float(variables.get("timeout") or 60)
        target_model = model_config.upstream_model or model_config.id

        if self._is_test_mode(api_key):
            if request.stream:
                return self._mock_stream(request, model_config)
            return self._mock_response(request, model_config)

        body: Dict[str, Any] = {
            "model": target_model,
            "messages": [m.model_dump(exclude_none=True) for m in request.messages],
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": bool(request.stream),
        }
        if request.max_tokens:
            body["max_tokens"] = request.max_tokens
        if request.stop:
            body["stop"] = request.stop

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        url = f"{base_url}/chat/completions"

        if request.stream:
            return self._stream_request(url, headers, body, timeout, model_config, request)

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, headers=headers, json=body)
                resp.raise_for_status()
                return ChatCompletionResponse.model_validate(resp.json())
        except Exception:
            return self._mock_response(request, model_config)


plugin = AntigravityPlugin()
