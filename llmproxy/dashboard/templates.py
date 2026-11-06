from __future__ import annotations

from typing import Any, Dict, List

STARTER_TEMPLATES: List[Dict[str, Any]] = [
    {
        "id": "mock_provider",
        "name": "Offline Mock Provider",
        "filename": "mock_provider.py",
        "description": "Ultra-fast offline test provider generating immediate responses with token & latency metrics.",
        "code": '''"""
Offline Mock Provider Plugin for LLMProxy.
Simulates instant chat completion without external network calls.
"""
import time
import uuid
from llmproxy.core.interface import BasePlugin
from llmproxy.schemas import (
    ChatChoice,
    ChatMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionUsage,
    PluginMetadata,
    PluginModelConfig,
    PluginVariable,
)


class MockProviderPlugin(BasePlugin):
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            id="mock_provider",
            name="Offline Mock Provider",
            description="Ultra-fast offline test provider for local verification and playground testing.",
            version="1.0.0",
            author="llmproxy",
            enabled=True,
            variables=[
                PluginVariable(
                    name="response_prefix",
                    label="Response Prefix",
                    type="string",
                    default="[Mock AI]",
                    value="[Mock AI]",
                    required=False,
                    description="Prefix prepended to every generated response",
                ),
            ],
            models=[
                PluginModelConfig(
                    id="mock-gpt-4o",
                    name="Mock GPT-4o",
                    upstream_model="mock-gpt-4o",
                    context_window=128000,
                    max_tokens=4096,
                ),
                PluginModelConfig(
                    id="mock-claude-3-5",
                    name="Mock Claude 3.5 Sonnet",
                    upstream_model="mock-claude-3-5-sonnet",
                    context_window=200000,
                    max_tokens=8192,
                ),
            ],
        )

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        variables: dict,
    ) -> ChatCompletionResponse:
        prefix = variables.get("response_prefix") or "[Mock AI]"
        last_user_msg = "Hello!"
        for msg in reversed(request.messages):
            if msg.role == "user":
                last_user_msg = str(msg.content)
                break

        reply = f"{prefix} Received prompt for model '{request.model}':\\n\\n> {last_user_msg}\\n\\nEverything is functioning properly in LLMProxy router."

        prompt_tokens = max(1, len(last_user_msg.split()))
        completion_tokens = len(reply.split())

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            model=request.model,
            created=int(time.time()),
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=reply),
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )


plugin = MockProviderPlugin()
''',
    },
    {
        "id": "openai_compatible",
        "name": "OpenAI Compatible Proxy",
        "filename": "openai_compatible.py",
        "description": "Standard proxy for OpenAI, Groq, Together, DeepSeek, Ollama, and compatible APIs.",
        "code": '''"""
OpenAI-Compatible Provider Plugin for LLMProxy.
Forwards chat completion requests to any OpenAI-compatible upstream endpoint.
"""
import httpx
from llmproxy.core.interface import BasePlugin
from llmproxy.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    PluginMetadata,
    PluginModelConfig,
    PluginVariable,
)


class OpenAICompatiblePlugin(BasePlugin):
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            id="openai_compatible",
            name="OpenAI Compatible Provider",
            description="Upstream proxy for OpenAI, Groq, Together, Ollama, DeepSeek.",
            version="1.0.0",
            author="custom",
            enabled=True,
            variables=[
                PluginVariable(
                    name="api_key",
                    label="API Key",
                    type="secret",
                    default="",
                    value="",
                    required=True,
                    description="Authentication token for upstream API",
                ),
                PluginVariable(
                    name="base_url",
                    label="Base URL",
                    type="string",
                    default="https://api.openai.com/v1",
                    value="https://api.openai.com/v1",
                    required=True,
                    description="Base URL of upstream provider",
                ),
            ],
            models=[
                PluginModelConfig(
                    id="gpt-4o",
                    name="GPT-4o Omnimodel",
                    upstream_model="gpt-4o",
                    context_window=128000,
                    max_tokens=4096,
                ),
                PluginModelConfig(
                    id="gpt-4o-mini",
                    name="GPT-4o Mini Fast",
                    upstream_model="gpt-4o-mini",
                    context_window=128000,
                    max_tokens=4096,
                ),
            ],
        )

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        variables: dict,
    ) -> ChatCompletionResponse:
        base_url = (variables.get("base_url") or "https://api.openai.com/v1").rstrip("/")
        api_key = variables.get("api_key") or ""

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = request.model_dump(exclude_unset=True)
        if model_config.upstream_model:
            payload["model"] = model_config.upstream_model

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            return ChatCompletionResponse.model_validate(resp.json())


plugin = OpenAICompatiblePlugin()
''',
    },
    {
        "id": "anthropic_proxy",
        "name": "Anthropic Claude Proxy",
        "filename": "anthropic_proxy.py",
        "description": "Proxy translating OpenAI Chat format into Anthropic Messages API format.",
        "code": '''"""
Anthropic Messages API Provider Plugin for LLMProxy.
Translates OpenAI chat format into Anthropic Claude Messages API.
"""
import time
import uuid
import httpx
from llmproxy.core.interface import BasePlugin
from llmproxy.schemas import (
    ChatChoice,
    ChatMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionUsage,
    PluginMetadata,
    PluginModelConfig,
    PluginVariable,
)


class AnthropicProxyPlugin(BasePlugin):
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            id="anthropic_proxy",
            name="Anthropic Claude Provider",
            description="Upstream proxy converting requests to Claude 3.5 / Claude 3 Opus.",
            version="1.0.0",
            author="custom",
            enabled=True,
            variables=[
                PluginVariable(
                    name="anthropic_api_key",
                    label="Anthropic API Key",
                    type="secret",
                    default="",
                    value="",
                    required=True,
                    description="Anthropic console API Key",
                ),
                PluginVariable(
                    name="anthropic_version",
                    label="API Version",
                    type="string",
                    default="2023-06-01",
                    value="2023-06-01",
                    required=False,
                    description="Anthropic API version header",
                ),
            ],
            models=[
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

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        variables: dict,
    ) -> ChatCompletionResponse:
        api_key = variables.get("anthropic_api_key") or ""
        version = variables.get("anthropic_version") or "2023-06-01"

        headers = {
            "x-api-key": api_key,
            "anthropic-version": version,
            "content-type": "application/json",
        }

        system_prompt = None
        messages = []
        for msg in request.messages:
            if msg.role == "system":
                system_prompt = str(msg.content)
            else:
                role = "assistant" if msg.role == "assistant" else "user"
                messages.append({"role": role, "content": str(msg.content)})

        payload = {
            "model": model_config.upstream_model or "claude-3-5-sonnet-20241022",
            "messages": messages,
            "max_tokens": request.max_tokens or model_config.max_tokens or 4096,
            "temperature": request.temperature or 1.0,
        }
        if system_prompt:
            payload["system"] = system_prompt

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

            text_content = ""
            for block in data.get("content", []):
                if block.get("type") == "text":
                    text_content += block.get("text", "")

            usage = data.get("usage", {})
            return ChatCompletionResponse(
                id=f"chatcmpl-{data.get('id', uuid.uuid4().hex[:12])}",
                model=request.model,
                created=int(time.time()),
                choices=[
                    ChatChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content=text_content),
                        finish_reason="stop",
                    )
                ],
                usage=CompletionUsage(
                    prompt_tokens=usage.get("input_tokens", 0),
                    completion_tokens=usage.get("output_tokens", 0),
                    total_tokens=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                ),
            )


plugin = AnthropicProxyPlugin()
''',
    },
]
