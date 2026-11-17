"""
Offline Mock Provider Plugin for LLMProxy.
Simulates instant chat completion without external network calls.
"""
from __future__ import annotations

import time
import uuid
from typing import Any, AsyncIterator, Dict, Optional, Union

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


class MockProviderPlugin(BasePlugin):
    REASONING_LEVELS = ("none", "low", "medium", "high", "xhigh", "max")

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            id="mock_provider",
            name="Offline Mock Provider",
            description="Ultra-fast offline test provider for local verification and playground testing.",
            version="1.1.0",
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
                PluginVariable(
                    name="reasoning_effort",
                    label="Default Reasoning Effort",
                    type="select",
                    default="default",
                    options=["default", *self.REASONING_LEVELS],
                    description=(
                        "Simulated reasoning level. Request-level reasoning_effort "
                        "takes precedence."
                    ),
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
                    id="mock/mock-gpt-4o",
                    name="Mock GPT-4o",
                    upstream_model="mock-gpt-4o",
                    context_window=128000,
                    max_tokens=4096,
                    supports_vision=True,
                    reasoning_levels=list(self.REASONING_LEVELS),
                ),
                PluginModelConfig(
                    id="mock/mock-claude-3-5",
                    name="Mock Claude 3.5 Sonnet",
                    upstream_model="mock-claude-3-5-sonnet",
                    context_window=200000,
                    max_tokens=8192,
                    supports_vision=True,
                    reasoning_levels=list(self.REASONING_LEVELS),
                ),
            ],
        )

    @staticmethod
    def _context_cache_requested(request: ChatCompletionRequest) -> bool:
        """Detect cache hints so the offline mock reports a simulated cache hit."""
        extra = request.extra_body if isinstance(request.extra_body, dict) else {}
        if any(
            key in extra
            for key in (
                "prompt_cache_key",
                "prompt_cache_retention",
                "cache_control",
                "cached_content",
                "cachedContent",
            )
        ):
            return True
        if getattr(request, "prompt_cache_key", None) or getattr(
            request, "cache_control", None
        ):
            return True
        for message in request.messages:
            if isinstance(message.content, list):
                for part in message.content:
                    if isinstance(part, dict) and part.get("cache_control"):
                        return True
        return False

    @classmethod
    def _normalize_reasoning_effort(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        raw = str(value).strip().lower()
        if not raw or raw in ("default", "auto"):
            return None
        if raw in ("off", "disabled", "false", "0"):
            return "none"
        return raw if raw in cls.REASONING_LEVELS else None

    def _resolve_reasoning_effort(
        self, request: ChatCompletionRequest, variables: Dict[str, Any]
    ) -> Optional[str]:
        raw: Any = request.reasoning_effort
        extra = request.extra_body if isinstance(request.extra_body, dict) else {}
        if raw is None:
            raw = extra.get("reasoning_effort")
        if raw is None:
            reasoning = extra.get("reasoning")
            if isinstance(reasoning, dict):
                raw = reasoning.get("effort")
            elif reasoning:
                raw = reasoning
        effort = self._normalize_reasoning_effort(raw)
        if effort is None and raw is None:
            effort = self._normalize_reasoning_effort(variables.get("reasoning_effort"))
        return effort

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        variables: dict,
        account: Optional[Account] = None,
    ) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionChunk]]:
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
        effort = self._resolve_reasoning_effort(request, variables or {})
        reasoning_text = None
        if effort and effort != "none":
            reasoning_text = (
                f"Simulated reasoning for model '{request.model}' at "
                f"'{effort}' effort."
            )
        reasoning_tokens = len(reasoning_text.split()) if reasoning_text else 0
        completion_tokens = len(base_reply.split()) + reasoning_tokens
        total_tokens = prompt_tokens + completion_tokens
        cache_requested = self._context_cache_requested(request)
        cached_tokens = max(0, prompt_tokens // 2) if cache_requested else 0

        cache_line = (
            f"• **Context cache:** simulated ({cached_tokens} cached prompt tokens)\n"
            if cache_requested
            else ""
        )
        reasoning_line = (
            f"• **Reasoning:** simulated ({effort})\n" if reasoning_text else ""
        )
        details = (
            f"\n\n---\n"
            f"**Execution Details:**\n"
            f"• **Time taken:** {time_taken_ms:.2f} ms\n"
            f"• **Timestamp:** {now_str}\n"
            f"• **Model:** {request.model}\n"
            f"• **Tokens:** {prompt_tokens} prompt, {completion_tokens} completion ({total_tokens} total)\n"
            f"{cache_line}"
            f"{reasoning_line}"
            f"• **Account status:** No account configured (Simulation Mode)"
        )
        reply = base_reply + details
        usage_kwargs: Dict[str, Any] = {}
        if cache_requested:
            usage_kwargs["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
            usage_kwargs["cache_read_input_tokens"] = cached_tokens
        if reasoning_tokens:
            usage_kwargs["completion_tokens_details"] = {
                "reasoning_tokens": reasoning_tokens
            }
        usage = CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            **usage_kwargs,
        )

        if request.stream:
            chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            created = int(time.time())

            async def stream_response() -> AsyncIterator[ChatCompletionChunk]:
                yield ChatCompletionChunk(
                    id=chunk_id,
                    created=created,
                    model=request.model,
                    choices=[
                        ChatChoiceDelta(
                            index=0,
                            delta={"role": "assistant", "content": ""},
                            finish_reason=None,
                        )
                    ],
                )
                if reasoning_text:
                    yield ChatCompletionChunk(
                        id=chunk_id,
                        created=created,
                        model=request.model,
                        choices=[
                            ChatChoiceDelta(
                                index=0,
                                delta={"reasoning_content": reasoning_text},
                                finish_reason=None,
                            )
                        ],
                    )
                for index in range(0, len(reply), 24):
                    yield ChatCompletionChunk(
                        id=chunk_id,
                        created=created,
                        model=request.model,
                        choices=[
                            ChatChoiceDelta(
                                index=0,
                                delta={"content": reply[index : index + 24]},
                                finish_reason=None,
                            )
                        ],
                    )
                yield ChatCompletionChunk(
                    id=chunk_id,
                    created=created,
                    model=request.model,
                    choices=[ChatChoiceDelta(index=0, delta={}, finish_reason="stop")],
                    usage=usage,
                )

            return stream_response()

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            model=request.model,
            created=int(time.time()),
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=reply,
                        **(
                            {"reasoning_content": reasoning_text}
                            if reasoning_text
                            else {}
                        ),
                    ),
                    finish_reason="stop",
                )
            ],
            usage=usage,
        )


plugin = MockProviderPlugin()
