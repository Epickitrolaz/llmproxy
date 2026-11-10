"""
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
                    description="Strategy for selecting proxies from pool: sequential (round-robin), random, specific proxy, or switch on limit",
                ),
                PluginVariable(
                    name="specific_proxy_id",
                    label="Specific Proxy ID / Tag",
                    type="string",
                    default="",
                    description="Target proxy ID, URL, or tag to route through when strategy is 'specific'",
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
        account: Optional[Account] = None,
    ) -> ChatCompletionResponse:
        api_key = (account.api_key if account else None) or variables.get("anthropic_api_key") or ""
        version = variables.get("anthropic_version") or "2023-06-01"
        proxy = str(variables.get("proxy") or (getattr(account, "proxy_url", None) or "")).strip() or None

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

        async with httpx.AsyncClient(timeout=60.0, proxy=proxy) as client:
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
