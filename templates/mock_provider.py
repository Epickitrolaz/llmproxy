"""
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
        start_t = time.perf_counter()
        prefix = variables.get("response_prefix") or "[Mock AI]"
        last_user_msg = "Hello!"
        for msg in reversed(request.messages):
            if msg.role == "user":
                last_user_msg = str(msg.content)
                break

        base_reply = f"{prefix} Received prompt for model '{request.model}':\n\n> {last_user_msg}\n\nEverything is functioning properly in LLMProxy router."
        time_taken_ms = max(0.5, round((time.perf_counter() - start_t) * 1000.0, 2))
        now_str = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        prompt_tokens = max(1, len(last_user_msg.split()))
        completion_tokens = len(base_reply.split())
        total_tokens = prompt_tokens + completion_tokens

        details = (
            f"\n\n---\n"
            f"**Execution Details:**\n"
            f"• **Time taken:** {time_taken_ms:.2f} ms\n"
            f"• **Timestamp:** {now_str}\n"
            f"• **Model:** {request.model}\n"
            f"• **Tokens:** {prompt_tokens} prompt, {completion_tokens} completion ({total_tokens} total)\n"
            f"• **Account status:** No account configured (Simulation Mode)"
        )
        reply = base_reply + details

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
