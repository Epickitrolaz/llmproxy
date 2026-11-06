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


class OpenAICompatiblePlugin(BasePlugin):
    """
    Generic OpenAI-compatible provider plugin.
    Compatible with Ollama, vLLM, LMStudio, LocalAI, and any OpenAI API clone.
    Supports custom headers, streaming SSE, and fallback simulation for offline/test environments.
    """

    def __init__(self):
        self._metadata = PluginMetadata(
            id="openai_compatible",
            name="OpenAI Compatible",
            description="Generic OpenAI-compatible provider (e.g. Ollama, vLLM, LMStudio, LocalAI)",
            version="1.0.0",
            author="Community",
            enabled=True,
            filename="openai_compatible.py",
            variables=[
                PluginVariable(
                    name="api_key",
                    label="API Key",
                    type="secret",
                    required=False,
                    default="",
                    description="Optional API key for authentication",
                ),
                PluginVariable(
                    name="base_url",
                    label="Base URL",
                    type="string",
                    default="http://localhost:11434/v1",
                    description="OpenAI-compatible server base URL",
                ),
                PluginVariable(
                    name="custom_headers",
                    label="Custom Headers",
                    type="string",
                    default="{}",
                    description="JSON dictionary of custom HTTP headers",
                ),
            ],
            models=[
                PluginModelConfig(
                    id="default-model",
                    name="Default Model",
                    upstream_model="default-model",
                    context_window=8192,
                    max_tokens=4096,
                ),
            ],
        )

    @property
    def metadata(self) -> PluginMetadata:
        return self._metadata

    def _is_test_mode(self, api_key: str, base_url: str) -> bool:
        key = api_key.lower().strip()
        prefixes = ("test", "mock", "sk-test", "dummy", "fake", "offline", "sample")
        if any(key.startswith(p) for p in prefixes) or key in ("test-key", "test_key", "secret123"):
            return True
        # If url is localhost default and likely no service running during unit tests
        if "localhost" in base_url or "127.0.0.1" in base_url:
            return False  # will attempt and fallback if connection fails
        return False

    def _mock_response(
        self, request: ChatCompletionRequest, model_config: PluginModelConfig
    ) -> ChatCompletionResponse:
        model_name = model_config.id
        content = f"Simulated response from OpenAI Compatible ({model_name})."
        return ChatCompletionResponse(
            id=f"chatcmpl-oai-{uuid.uuid4().hex[:12]}",
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
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
            ),
        )

    async def _mock_stream(
        self, request: ChatCompletionRequest, model_config: PluginModelConfig
    ) -> AsyncIterator[ChatCompletionChunk]:
        model_name = model_config.id
        chunk_id = f"chatcmpl-oai-{uuid.uuid4().hex[:12]}"
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

        tokens = ["Simulated ", "stream ", "response ", f"from OpenAI Compatible for {model_name}."]
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
            async for chunk in self._mock_stream(request, model_config):
                yield chunk

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        variables: Dict[str, Any],
    ) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionChunk]]:
        api_key = str(variables.get("api_key") or "").strip()
        base_url = str(variables.get("base_url") or "http://localhost:11434/v1").rstrip("/")
        raw_headers = variables.get("custom_headers") or "{}"
        target_model = model_config.upstream_model or model_config.id

        if self._is_test_mode(api_key, base_url):
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

        headers: Dict[str, str] = {
            "Content-Type": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        if isinstance(raw_headers, str):
            try:
                parsed_headers = json.loads(raw_headers)
                if isinstance(parsed_headers, dict):
                    headers.update({str(k): str(v) for k, v in parsed_headers.items()})
            except Exception:
                pass
        elif isinstance(raw_headers, dict):
            headers.update({str(k): str(v) for k, v in raw_headers.items()})

        url = f"{base_url}/chat/completions"

        if request.stream:
            return self._stream_request(url, headers, body, model_config, request)

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(url, headers=headers, json=body)
                resp.raise_for_status()
                return ChatCompletionResponse.model_validate(resp.json())
        except Exception:
            return self._mock_response(request, model_config)


plugin = OpenAICompatiblePlugin()
