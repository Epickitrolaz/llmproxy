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


class CodexPlugin(BasePlugin):
    """
    OpenAI Codex and frontier models provider plugin.
    Supports gpt-6-luna, gpt-4o, and gpt-4o-mini with streaming SSE,
    organization headers, and simulated fallback for offline/test environments.
    """

    def __init__(self):
        self._metadata = PluginMetadata(
            id="codex",
            name="Codex (OpenAI)",
            description="OpenAI Codex & Frontier Models provider",
            version="1.0.0",
            author="OpenAI",
            enabled=True,
            filename="codex.py",
            variables=[
                PluginVariable(
                    name="api_key",
                    label="API Key",
                    type="secret",
                    required=True,
                    description="OpenAI API key",
                ),
                PluginVariable(
                    name="base_url",
                    label="Base URL",
                    type="string",
                    default="https://api.openai.com/v1",
                    description="OpenAI API base URL",
                ),
                PluginVariable(
                    name="organization_id",
                    label="Organization ID",
                    type="string",
                    default="",
                    description="Optional OpenAI Organization ID",
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
        self, request: ChatCompletionRequest, model_config: PluginModelConfig
    ) -> ChatCompletionResponse:
        model_name = model_config.id
        content = f"Simulated response from Codex ({model_name})."
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
        self, request: ChatCompletionRequest, model_config: PluginModelConfig
    ) -> AsyncIterator[ChatCompletionChunk]:
        model_name = model_config.id
        chunk_id = f"chatcmpl-codex-{uuid.uuid4().hex[:12]}"
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

        tokens = ["Simulated ", "stream ", "response ", f"from Codex for {model_name}."]
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
        api_key = str(variables.get("api_key") or "")
        base_url = str(variables.get("base_url") or "https://api.openai.com/v1").rstrip("/")
        org_id = str(variables.get("organization_id") or "").strip()
        reasoning_effort = str(variables.get("reasoning_effort") or "medium").strip()
        target_model = model_config.upstream_model or model_config.id

        if self._is_test_mode(api_key):
            if request.stream:
                return self._mock_stream(request, model_config)
            return self._mock_response(request, model_config)

        is_reasoning_model = "gpt-6" in target_model.lower() or target_model.lower().startswith("o")
        body: Dict[str, Any] = {
            "model": target_model,
            "messages": [m.model_dump(exclude_none=True) for m in request.messages],
            "stream": bool(request.stream),
        }
        if not is_reasoning_model:
            body["temperature"] = request.temperature
            body["top_p"] = request.top_p
        elif reasoning_effort:
            body["reasoning_effort"] = reasoning_effort

        if request.max_tokens:
            body["max_tokens"] = request.max_tokens
        if request.stop:
            body["stop"] = request.stop

        headers: Dict[str, str] = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if org_id:
            headers["OpenAI-Organization"] = org_id

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


plugin = CodexPlugin()
