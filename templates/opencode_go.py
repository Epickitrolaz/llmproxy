"""OpenCode Go provider plugin for LLMProxy.

OpenCode Go is the subscription tier of the OpenCode Console gateway. It serves
coding models over an OpenAI-compatible Chat Completions API at
``https://opencode.ai/inference/go/openai/v1`` and authenticates with a Console
API key (``Authorization: Bearer oc_sk_...``).

Go asks clients to identify themselves with a real user agent and to send a
stable ``x-opencode-session`` header per conversation so requests can be routed
consistently and the prompt cache reused. This plugin derives that session from
the conversation itself (the first system/user turn) unless the caller supplies
``session_id`` / ``opencode_session`` in ``extra_body``.

Every text model available on Go also works through the Chat Completions API,
which Console translates to the model's native protocol, so a single endpoint
covers GPT, Claude, Gemini, and open-weight models.

Docs: https://opencode.ai/docs/console/go
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union

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

# Template gallery identity (read by the dashboard template loader).
id = "opencode_go"
name = "OpenCode Go"
description = (
    "OpenCode Go subscription gateway with OpenAI-compatible Chat Completions, "
    "API-key accounts, stable session routing, and reasoning controls"
)

DEFAULT_BASE_URL = "https://opencode.ai/inference/go/openai/v1"
DEFAULT_USER_AGENT = "llmproxy/1.0"
DEFAULT_TIMEOUT = 120.0

REASONING_SUFFIXES: Tuple[Tuple[str, str], ...] = (
    (":none", "none"),
    ("-none", "none"),
    (":off", "none"),
    ("-off", "none"),
    (":low", "low"),
    ("-low", "low"),
    (":medium", "medium"),
    ("-medium", "medium"),
    (":high", "high"),
    ("-high", "high"),
    (":xhigh", "xhigh"),
    ("-xhigh", "xhigh"),
    (":max", "max"),
    ("-max", "max"),
)

EFFORT_LEVELS = ("none", "low", "medium", "high", "xhigh", "max")

# Models included with OpenCode Go at the time this template was written.
# The list changes as OpenCode tests new models; users can add or remove
# models from the dashboard's Models tab.
DEFAULT_MODELS: List[PluginModelConfig] = [
    PluginModelConfig(
        id="deepseek-v4.1-flash",
        name="DeepSeek V4.1 Flash",
        upstream_model="deepseek-v4.1-flash",
        context_window=1000000,
        max_tokens=384000,
    ),
    PluginModelConfig(
        id="deepseek-v4-pro",
        name="DeepSeek V4 Pro",
        upstream_model="deepseek-v4-pro",
        context_window=1000000,
        max_tokens=384000,
    ),
    PluginModelConfig(
        id="glm-5.3",
        name="GLM-5.3",
        upstream_model="glm-5.3",
        context_window=1000000,
        max_tokens=131072,
    ),
    PluginModelConfig(
        id="glm-5.3-flash",
        name="GLM-5.3-Flash",
        upstream_model="glm-5.3-flash",
        context_window=1000000,
        max_tokens=131072,
    ),
    PluginModelConfig(
        id="glm-5.2",
        name="GLM-5.2",
        upstream_model="glm-5.2",
        context_window=1000000,
        max_tokens=131072,
    ),
    PluginModelConfig(
        id="kimi-k3",
        name="Kimi K3",
        upstream_model="kimi-k3",
        context_window=1048576,
        max_tokens=131072,
    ),
    PluginModelConfig(
        id="kimi-k3.1-pro",
        name="Kimi K3.1 Pro",
        upstream_model="kimi-k3.1-pro",
        context_window=1048576,
        max_tokens=131072,
    ),
    PluginModelConfig(
        id="kimi-k2.7-code",
        name="Kimi K2.7 Code",
        upstream_model="kimi-k2.7-code",
        context_window=262144,
        max_tokens=262144,
    ),
    PluginModelConfig(
        id="gpt-6-luna",
        name="GPT 6 Luna",
        upstream_model="gpt-6-luna",
        context_window=1050000,
        max_tokens=128000,
    ),
    PluginModelConfig(
        id="gpt-5.6-luna",
        name="GPT 5.6 Luna",
        upstream_model="gpt-5.6-luna",
        context_window=1050000,
        max_tokens=128000,
    ),
    PluginModelConfig(
        id="claude-haiku-5-5",
        name="Claude Haiku 5.5",
        upstream_model="claude-haiku-5-5",
        context_window=1000000,
        max_tokens=128000,
    ),
    PluginModelConfig(
        id="gemini-4-argon",
        name="Gemini 4 Argon",
        upstream_model="gemini-4-argon",
        context_window=1048576,
        max_tokens=65536,
    ),
    PluginModelConfig(
        id="grok-4.7",
        name="Grok 4.7",
        upstream_model="grok-4.7",
        context_window=500000,
        max_tokens=500000,
    ),
    PluginModelConfig(
        id="minimax-m3",
        name="MiniMax M3",
        upstream_model="minimax-m3",
        context_window=512000,
        max_tokens=128000,
    ),
    PluginModelConfig(
        id="minimax-m3.1-flash",
        name="MiniMax M3.1 Flash",
        upstream_model="minimax-m3.1-flash",
        context_window=512000,
        max_tokens=128000,
    ),
    PluginModelConfig(
        id="qwen3.8-max",
        name="Qwen3.8 Max",
        upstream_model="qwen3.8-max",
        context_window=262144,
        max_tokens=131072,
    ),
    PluginModelConfig(
        id="qwen3.8-flash",
        name="Qwen3.8 Flash",
        upstream_model="qwen3.8-flash",
        context_window=1000000,
        max_tokens=131072,
    ),
    PluginModelConfig(
        id="qwen3.7-plus",
        name="Qwen3.7 Plus",
        upstream_model="qwen3.7-plus",
        context_window=262144,
        max_tokens=131072,
    ),
    PluginModelConfig(
        id="longcat-2.0",
        name="LongCat-2.0",
        upstream_model="longcat-2.0",
        context_window=1000000,
        max_tokens=262144,
    ),
    PluginModelConfig(
        id="longcat-2.5-free",
        name="LongCat 2.5 Free",
        upstream_model="longcat-2.5-free",
        context_window=1000000,
        max_tokens=262144,
    ),
    PluginModelConfig(
        id="step-5-preview",
        name="Step 5 Preview",
        upstream_model="step-5-preview",
        context_window=262144,
        max_tokens=131072,
    ),
    PluginModelConfig(
        id="mimo-v2.6-flash",
        name="MiMo-V2.6-Flash",
        upstream_model="mimo-v2.6-flash",
        context_window=1000000,
        max_tokens=131072,
    ),
    PluginModelConfig(
        id="mimo-v2.6-pro",
        name="MiMo-V2.6-Pro",
        upstream_model="mimo-v2.6-pro",
        context_window=1000000,
        max_tokens=131072,
    ),
    PluginModelConfig(
        id="mimo-v2.5",
        name="MiMo-V2.5",
        upstream_model="mimo-v2.5",
        context_window=1000000,
        max_tokens=131072,
    ),
    PluginModelConfig(
        id="mimo-v2.5-pro",
        name="MiMo-V2.5-Pro",
        upstream_model="mimo-v2.5-pro",
        context_window=1000000,
        max_tokens=131072,
    ),
    PluginModelConfig(
        id="hy4",
        name="Hy4",
        upstream_model="hy4",
        context_window=262144,
        max_tokens=131072,
    ),
    PluginModelConfig(
        id="hy4-preview",
        name="Hy4 preview",
        upstream_model="hy4-preview",
        context_window=262144,
        max_tokens=131072,
    ),
    PluginModelConfig(
        id="hy3",
        name="Hy3",
        upstream_model="hy3",
        context_window=262144,
        max_tokens=131072,
    ),
    PluginModelConfig(
        id="muse-spark-1.3-contributor",
        name="Muse Spark 1.3 Contributor",
        upstream_model="muse-spark-1.3-contributor",
        context_window=262144,
        max_tokens=131072,
    ),
    PluginModelConfig(
        id="muse-spark-1.2-contributor",
        name="Muse Spark 1.2 Contributor",
        upstream_model="muse-spark-1.2-contributor",
        context_window=262144,
        max_tokens=131072,
    ),
    PluginModelConfig(
        id="beam-free",
        name="Beam Free",
        upstream_model="beam-free",
        context_window=1048576,
        max_tokens=131072,
    ),
    PluginModelConfig(
        id="iceman-contributor-free",
        name="Iceman Contributor Free",
        upstream_model="iceman-contributor-free",
        context_window=200000,
        max_tokens=32000,
    ),
]

TEST_KEY_PREFIXES = ("test", "mock", "sk-test", "dummy", "fake", "offline", "sample")
TEST_KEYS = ("test-key", "test_key", "secret123")


def _content_text(content: Any) -> str:
    """Extract plain text from a chat message content value."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text") or ""))
        return "\n".join(parts)
    return ""


def _number(value: Any, default: float) -> float:
    try:
        if value not in (None, ""):
            return float(value)
    except (TypeError, ValueError):
        pass
    return default


class OpenCodeGoPlugin(BasePlugin):
    """OpenCode Go provider with API-key accounts and stable sessions."""

    def __init__(self) -> None:
        self._metadata = PluginMetadata(
            id="opencode_go",
            name="OpenCode Go",
            description=(
                "OpenCode Go subscription gateway with OpenAI-compatible Chat "
                "Completions, API-key accounts, stable session routing, and "
                "reasoning controls"
            ),
            version="1.0.0",
            author="OpenCode",
            enabled=True,
            filename="opencode_go.py",
            variables=[
                PluginVariable(
                    name="api_key",
                    label="OpenCode API Key",
                    type="secret",
                    default="",
                    description=(
                        "Console API key (oc_sk_...). Accounts can also supply this."
                    ),
                ),
                PluginVariable(
                    name="base_url",
                    label="Base URL",
                    type="string",
                    default=DEFAULT_BASE_URL,
                    description="OpenCode Go OpenAI-compatible base URL",
                ),
                PluginVariable(
                    name="user_agent",
                    label="User Agent",
                    type="string",
                    default=DEFAULT_USER_AGENT,
                    description=(
                        "User agent sent to OpenCode Go. Go asks clients to identify "
                        "themselves instead of using a generic SDK name."
                    ),
                ),
                PluginVariable(
                    name="timeout",
                    label="Timeout (seconds)",
                    type="number",
                    default=DEFAULT_TIMEOUT,
                    description="Request timeout in seconds",
                ),
                PluginVariable(
                    name="load_balancing_strategy",
                    label="Load Balancing Strategy",
                    type="select",
                    default="fallback",
                    options=["fallback", "round_robin", "least_used", "random"],
                    description="Strategy for distributing requests across connected accounts",
                ),
                PluginVariable(
                    name="account_switch_cooldown",
                    label="Account Cooldown (s)",
                    type="number",
                    default=60,
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
                    name="reasoning_effort",
                    label="Reasoning Effort",
                    type="select",
                    default="default",
                    options=["default", *EFFORT_LEVELS],
                    description=(
                        "Default reasoning effort for models that support it. A "
                        "request-level reasoning_effort or a model suffix takes precedence."
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
                    description=(
                        "Strategy for selecting proxies from the pool: sequential, "
                        "random, specific, or switch_on_limit"
                    ),
                ),
                PluginVariable(
                    name="specific_proxy_id",
                    label="Specific Proxy ID / Tag",
                    type="string",
                    default="",
                    description="Target proxy ID, URL, or tag when the strategy is 'specific'",
                ),
                PluginVariable(
                    name="proxy",
                    label="Direct Proxy URL",
                    type="string",
                    default="",
                    description="Direct HTTP or SOCKS5 proxy URL to route through",
                ),
            ],
            models=list(DEFAULT_MODELS),
        )

    @property
    def metadata(self) -> PluginMetadata:
        return self._metadata

    def validate_variables(self, variables: Dict[str, Any]) -> List[str]:
        """Validate the operational variables shared by all requests."""
        errors: List[str] = []
        timeout = variables.get("timeout")
        if timeout not in (None, ""):
            try:
                if float(timeout) <= 0:
                    errors.append("Timeout must be a positive number.")
            except (TypeError, ValueError):
                errors.append("Timeout must be a valid number.")

        effort = str(variables.get("reasoning_effort") or "default").strip().lower()
        if effort not in ("default", *EFFORT_LEVELS):
            errors.append(
                "Reasoning effort must be one of: default, "
                + ", ".join(EFFORT_LEVELS)
                + "."
            )
        return errors

    # ==========================================================================
    # Session identity
    # ==========================================================================

    @staticmethod
    def _request_extra(request: ChatCompletionRequest) -> Dict[str, Any]:
        return request.extra_body if isinstance(request.extra_body, dict) else {}

    def _explicit_session_id(self, request: ChatCompletionRequest) -> Optional[str]:
        """Return a caller-provided session ID, if any."""
        extra = self._request_extra(request)
        for key in ("session_id", "opencode_session", "x-opencode-session"):
            value = extra.get(key)
            if value is None:
                value = getattr(request, key, None)
            if value:
                raw = str(value).strip()
                if raw:
                    return raw[:128]
        return None

    def _conversation_session_id(self, request: ChatCompletionRequest) -> str:
        """Derive a stable ``x-opencode-session`` value for the conversation.

        The first system and user turns identify a conversation and stay stable
        while the conversation grows, which lets Go reuse its prompt cache
        without requiring the client to send a session header.
        """
        explicit = self._explicit_session_id(request)
        if explicit:
            return explicit

        seed_parts: List[str] = []
        if request.user:
            seed_parts.append(str(request.user))
        for role in ("system", "user"):
            for message in request.messages:
                if message.role == role:
                    text = _content_text(message.content)
                    if text:
                        seed_parts.append(text)
                        break

        if not seed_parts:
            seed_parts.append(str(request.model or "unknown"))

        digest = hashlib.sha256("\x00".join(seed_parts).encode("utf-8")).hexdigest()
        return f"ses_{digest[:32]}"

    # ==========================================================================
    # Reasoning
    # ==========================================================================

    @staticmethod
    def _normalize_effort(value: Any) -> Optional[str]:
        if value is None:
            return None
        raw = str(value).strip().lower()
        if not raw or raw in ("default", "auto"):
            return None
        if raw in ("off", "disabled", "false", "0"):
            return "none"
        if raw in EFFORT_LEVELS:
            return raw
        return None

    def _resolve_reasoning(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        variables: Dict[str, Any],
    ) -> Tuple[str, Optional[str]]:
        """Return the upstream model ID and the reasoning effort to send."""
        target_model = model_config.upstream_model or model_config.id

        raw_effort: Any = None
        if request.reasoning_effort:
            raw_effort = request.reasoning_effort
        else:
            extra = self._request_extra(request)
            raw_effort = extra.get("reasoning_effort")
            if raw_effort is None:
                reasoning = extra.get("reasoning")
                if isinstance(reasoning, dict):
                    raw_effort = reasoning.get("effort")
                elif reasoning:
                    raw_effort = reasoning

        requested_model = (request.model or "").strip().lower()
        for suffix, level in REASONING_SUFFIXES:
            if requested_model.endswith(suffix):
                if raw_effort is None:
                    raw_effort = level
                break

        effort = self._normalize_effort(raw_effort)
        if effort is None and raw_effort is None:
            effort = self._normalize_effort(variables.get("reasoning_effort"))

        for suffix, _ in REASONING_SUFFIXES:
            if target_model.lower().endswith(suffix):
                target_model = target_model[: -len(suffix)]
                break

        return target_model, effort

    # ==========================================================================
    # Request building
    # ==========================================================================

    def _build_headers(
        self,
        api_key: str,
        request: ChatCompletionRequest,
        variables: Dict[str, Any],
    ) -> Dict[str, str]:
        user_agent = str(variables.get("user_agent") or DEFAULT_USER_AGENT).strip()
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": user_agent or DEFAULT_USER_AGENT,
            "x-opencode-session": self._conversation_session_id(request),
        }

    @staticmethod
    def _apply_cache_fields(body: Dict[str, Any], request: ChatCompletionRequest) -> None:
        """Forward prompt-cache routing hints to the OpenCode gateway."""
        extra = request.extra_body if isinstance(request.extra_body, dict) else {}
        for key in ("prompt_cache_key", "prompt_cache_retention"):
            value = getattr(request, key, None)
            if value is None:
                value = extra.get(key)
            if value is not None:
                body[key] = value

    def _build_body(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        variables: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build the OpenAI Chat Completions body sent to OpenCode Go."""
        target_model, effort = self._resolve_reasoning(request, model_config, variables)

        body: Dict[str, Any] = {
            "model": target_model,
            "messages": [
                message.model_dump(exclude_none=True) for message in request.messages
            ],
            "stream": bool(request.stream),
        }

        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.top_p is not None:
            body["top_p"] = request.top_p

        max_tokens = (
            request.max_tokens
            if request.max_tokens is not None
            else request.max_completion_tokens
        )
        if max_tokens is not None:
            body["max_tokens"] = max_tokens

        if request.stop is not None:
            body["stop"] = request.stop
        if request.tools is not None:
            body["tools"] = request.tools
        if request.tool_choice is not None:
            body["tool_choice"] = request.tool_choice
        if request.parallel_tool_calls is not None:
            body["parallel_tool_calls"] = request.parallel_tool_calls
        if request.response_format is not None:
            body["response_format"] = request.response_format

        if effort:
            body["reasoning_effort"] = effort

        if request.stream and request.stream_options:
            body["stream_options"] = request.stream_options

        self._apply_cache_fields(body, request)
        return body

    @staticmethod
    def _resolve_proxy(variables: Dict[str, Any], account: Optional[Account]) -> Optional[str]:
        proxy = str(
            variables.get("proxy")
            or (getattr(account, "proxy_url", None) if account else "")
            or ""
        ).strip()
        return proxy or None

    @staticmethod
    def _is_test_mode(api_key: str) -> bool:
        if not api_key:
            return True
        key = api_key.lower().strip()
        return key.startswith(TEST_KEY_PREFIXES) or key in TEST_KEYS

    # ==========================================================================
    # Simulation mode
    # ==========================================================================

    def _mock_response(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        account: Optional[Account] = None,
    ) -> ChatCompletionResponse:
        model_name = model_config.upstream_model or model_config.id
        account_label = f" via account '{account.name}'" if account else ""
        created = int(time.time())
        now_str = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

        if request.tools and not any(
            message.role == "tool" for message in request.messages[-2:]
        ):
            fn = (
                request.tools[0].get("function", {})
                if "function" in request.tools[0]
                else request.tools[0]
            )
            call_id = f"call_opencode_{uuid.uuid4().hex[:8]}"
            tool_name = fn.get("name", "test_tool")
            properties = fn.get("parameters", {}).get("properties", {})
            mock_args = "{}"
            if properties:
                mock_args = json.dumps({next(iter(properties.keys())): "test_value"})

            return ChatCompletionResponse(
                id=f"chatcmpl-opencode-go-{uuid.uuid4().hex[:12]}",
                created=created,
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

        content = (
            f"Simulated response from OpenCode Go ({model_name}){account_label}.\n\n"
            f"---\n"
            f"**Execution Details:**\n"
            f"- **Timestamp:** {now_str}\n"
            f"- **Model:** {model_name}\n"
            f"- **Status:** Simulation (configure an OpenCode API key to reach Go)"
        )
        return ChatCompletionResponse(
            id=f"chatcmpl-opencode-go-{uuid.uuid4().hex[:12]}",
            created=created,
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
                completion_tokens=30,
                total_tokens=42,
            ),
        )

    async def _mock_stream(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        account: Optional[Account] = None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        model_name = model_config.upstream_model or model_config.id
        chunk_id = f"chatcmpl-opencode-go-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        account_label = f" via account '{account.name}'" if account else ""

        if request.tools and not any(
            message.role == "tool" for message in request.messages[-2:]
        ):
            fn = (
                request.tools[0].get("function", {})
                if "function" in request.tools[0]
                else request.tools[0]
            )
            call_id = f"call_opencode_{uuid.uuid4().hex[:8]}"
            tool_name = fn.get("name", "test_tool")
            properties = fn.get("parameters", {}).get("properties", {})
            mock_args = "{}"
            if properties:
                mock_args = json.dumps({next(iter(properties.keys())): "test_value"})

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
                                    "function": {"name": tool_name, "arguments": ""},
                                }
                            ],
                        },
                    )
                ],
            )
            midpoint = len(mock_args) // 2
            for piece in (mock_args[:midpoint], mock_args[midpoint:]):
                if not piece:
                    continue
                yield ChatCompletionChunk(
                    id=chunk_id,
                    created=created,
                    model=model_name,
                    choices=[
                        ChatChoiceDelta(
                            index=0,
                            delta={"tool_calls": [{"index": 0, "function": {"arguments": piece}}]},
                        )
                    ],
                )
            yield ChatCompletionChunk(
                id=chunk_id,
                created=created,
                model=model_name,
                choices=[ChatChoiceDelta(index=0, delta={}, finish_reason="tool_calls")],
            )
            return

        yield ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=model_name,
            choices=[
                ChatChoiceDelta(index=0, delta={"role": "assistant", "content": ""})
            ],
        )
        for token in [
            "Simulated ",
            "stream ",
            "response ",
            f"from OpenCode Go for {model_name}{account_label}.",
        ]:
            yield ChatCompletionChunk(
                id=chunk_id,
                created=created,
                model=model_name,
                choices=[ChatChoiceDelta(index=0, delta={"content": token})],
            )
        yield ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=model_name,
            choices=[ChatChoiceDelta(index=0, delta={}, finish_reason="stop")],
        )

    # ==========================================================================
    # Upstream execution
    # ==========================================================================

    @staticmethod
    async def _raise_for_status(response: httpx.Response, *, streaming: bool) -> None:
        """Raise an HTTPStatusError that keeps the upstream body excerpt."""
        if response.status_code < 400:
            return
        if streaming:
            raw = await response.aread()
        else:
            raw = response.content
        detail = raw.decode("utf-8", errors="replace")[:500]
        message = f"OpenCode Go returned HTTP {response.status_code}: {detail}"
        raise httpx.HTTPStatusError(
            message,
            request=response.request,
            response=response,
        )

    async def _stream_request(
        self,
        url: str,
        headers: Dict[str, str],
        body: Dict[str, Any],
        target_model: str,
        timeout: float,
        proxy: Optional[str],
    ) -> AsyncIterator[ChatCompletionChunk]:
        async with httpx.AsyncClient(timeout=timeout, proxy=proxy) as client:
            async with client.stream("POST", url, headers=headers, json=body) as response:
                await self._raise_for_status(response, streaming=True)
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        parsed = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(parsed, dict) and parsed.get("error"):
                        error = parsed["error"]
                        message = (
                            error.get("message") if isinstance(error, dict) else str(error)
                        )
                        raise RuntimeError(message or "OpenCode Go streaming error")
                    try:
                        parsed.setdefault("model", target_model)
                        yield ChatCompletionChunk.model_validate(parsed)
                    except Exception:
                        continue

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        variables: Dict[str, Any],
        account: Optional[Account] = None,
    ) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionChunk]]:
        api_key = (account.api_key if account else "") or str(
            variables.get("api_key") or ""
        )
        base_url = (
            account.base_url
            if (account and account.base_url)
            else str(variables.get("base_url") or DEFAULT_BASE_URL)
        ).rstrip("/")
        timeout = _number(variables.get("timeout"), DEFAULT_TIMEOUT)
        proxy = self._resolve_proxy(variables, account)

        if self._is_test_mode(api_key):
            if request.stream:
                return self._mock_stream(request, model_config, account=account)
            return self._mock_response(request, model_config, account=account)

        headers = self._build_headers(api_key, request, variables)
        body = self._build_body(request, model_config, variables)
        target_model = body["model"]
        url = f"{base_url}/chat/completions"

        if request.stream:
            return self._stream_request(
                url=url,
                headers=headers,
                body=body,
                target_model=target_model,
                timeout=timeout,
                proxy=proxy,
            )

        async with httpx.AsyncClient(timeout=timeout, proxy=proxy) as client:
            response = await client.post(url, headers=headers, json=body)
            await self._raise_for_status(response, streaming=False)
            data = response.json()

        data.setdefault("model", target_model)
        data.setdefault("id", f"chatcmpl-opencode-go-{uuid.uuid4().hex[:12]}")
        return ChatCompletionResponse.model_validate(data)


# Exported plugin singleton
plugin = OpenCodeGoPlugin()
