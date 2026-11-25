from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import uuid
from collections import OrderedDict
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union

import httpx

from llmproxy.core.interface import BasePlugin
from llmproxy.core.multimodal import (
    content_text,
    extract_image_url,
    image_detail,
    is_image_part,
    messages_have_images,
    to_anthropic_image_block,
    to_openai_image_part,
)
from llmproxy.core.usage import capture_rate_limit_headers
from llmproxy.schemas import (
    Account,
    ChatChoice,
    ChatChoiceDelta,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    CompletionUsage,
    PluginMetadata,
    PluginModelConfig,
    PluginVariable,
)

logger = logging.getLogger("llmproxy.plugin.agentrouter")

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

DEFAULT_BASE_URL = "https://agentrouter.org"
DEFAULT_USER_AGENT = "opencode/1.0.180"
DEFAULT_ANTHROPIC_USER_AGENT = "claude-cli/1.0.108 (external, cli)"
DEFAULT_ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_ANTHROPIC_BETA = "claude-code-20250219,oauth-2025-04-20"

PROTOCOL_AUTO = "auto"
PROTOCOL_RESPONSES = "responses"
PROTOCOL_ANTHROPIC = "anthropic"
PROTOCOL_OPENAI = "openai"

RESPONSES_REASONING_LEVELS: Tuple[str, ...] = (
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)

# AgentRouter rejects Responses requests whose max_output_tokens is below 16:
# "Invalid 'max_output_tokens': integer below minimum value. Expected a value
# >= 16". Reasoning models also spend output tokens on hidden reasoning, so a
# literal 16-token cap frequently returns an empty message. Use a practical
# floor of 256: the cap only limits the answer, it does not force generation, so
# short replies still stop early and cost the same.
RESPONSES_MIN_OUTPUT_TOKENS = 256
# Vision requests and high-effort reasoning spend more hidden tokens before the
# first visible word, so they get a larger floor to avoid empty truncated
# answers. Caps do not force generation; short answers still stop early.
RESPONSES_VISION_OUTPUT_FLOOR = 1024
RESPONSES_HIGH_EFFORT_FLOOR = 2048
ANTHROPIC_REASONING_LEVELS: Tuple[str, ...] = (
    "none",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)

ANTHROPIC_THINKING_BUDGETS: Dict[str, int] = {
    "low": 1024,
    "medium": 4096,
    "high": 16384,
    "xhigh": 32768,
    "max": 60000,
}

# Keys accepted by ``extra_body`` for the Responses API. Unrelated keys are
# dropped so a client cannot inject a second ``messages``/``input`` payload.
RESPONSES_EXTRA_KEYS = {
    "background",
    "conversation",
    "include",
    "max_output_tokens",
    "max_tool_calls",
    "metadata",
    "previous_response_id",
    "prompt_cache_key",
    "prompt_cache_retention",
    "safety_identifier",
    "service_tier",
    "store",
    "text",
    "truncation",
    "user",
}

# Responses reasoning items are replayed before their function calls. They are
# short-lived (a tool loop) and keyed by AgentRouter's call id.
_REASONING_ITEMS: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
_REASONING_ITEMS_LIMIT = 512

# Anthropic requires signed thinking blocks to be replayed with a tool result.
_THINKING_BLOCKS: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
_THINKING_BLOCKS_LIMIT = 512


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on", "enabled")


def _as_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _json_object(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _split_level(model_name: str, levels: Tuple[str, ...]) -> Tuple[str, Optional[str]]:
    """Strip a trailing ``:<level>`` from a custom upstream model name."""
    name = str(model_name or "")
    for level in sorted(levels, key=len, reverse=True):
        for separator in (":", "-"):
            suffix = f"{separator}{level}"
            if name.lower().endswith(suffix):
                return name[: -len(suffix)], level
    return name, None


def _remember_reasoning(call_id: Optional[str], items: List[Dict[str, Any]]) -> None:
    if not call_id or not items:
        return
    replayable = [dict(item) for item in items if isinstance(item, dict) and item.get("type") == "reasoning"]
    if not replayable:
        return
    _REASONING_ITEMS[call_id] = replayable
    _REASONING_ITEMS.move_to_end(call_id)
    while len(_REASONING_ITEMS) > _REASONING_ITEMS_LIMIT:
        _REASONING_ITEMS.popitem(last=False)


def _cached_reasoning(call_id: Optional[str]) -> List[Dict[str, Any]]:
    if not call_id:
        return []
    items = _REASONING_ITEMS.get(call_id)
    if not items:
        return []
    _REASONING_ITEMS.move_to_end(call_id)
    return [dict(item) for item in items]


def _remember_thinking(tool_use_id: Optional[str], blocks: List[Dict[str, Any]]) -> None:
    if not tool_use_id or not blocks:
        return
    replayable = [dict(block) for block in blocks if _is_replayable_thinking(block)]
    if not replayable:
        return
    _THINKING_BLOCKS[tool_use_id] = replayable
    _THINKING_BLOCKS.move_to_end(tool_use_id)
    while len(_THINKING_BLOCKS) > _THINKING_BLOCKS_LIMIT:
        _THINKING_BLOCKS.popitem(last=False)


def _is_replayable_thinking(block: Dict[str, Any]) -> bool:
    if not isinstance(block, dict):
        return False
    if block.get("type") == "redacted_thinking":
        return bool(block.get("data"))
    return block.get("type") == "thinking" and bool(block.get("signature"))


def _cached_thinking(tool_use_id: Optional[str]) -> List[Dict[str, Any]]:
    if not tool_use_id:
        return []
    blocks = _THINKING_BLOCKS.get(tool_use_id)
    if not blocks:
        return []
    _THINKING_BLOCKS.move_to_end(tool_use_id)
    return [dict(block) for block in blocks]


def _reasoning_text(block: Any) -> str:
    """Extract readable reasoning text from a thinking/summary block."""
    if isinstance(block, str):
        return block
    if not isinstance(block, dict):
        return ""
    for key in ("thinking", "text", "reasoning", "summary_text", "delta"):
        value = block.get(key)
        if isinstance(value, str) and value:
            return value
    content = block.get("content")
    if isinstance(content, list):
        return "".join(_reasoning_text(part) for part in content)
    nested = block.get("part")
    if isinstance(nested, dict):
        return _reasoning_text(nested)
    return ""


class AgentRouterPlugin(BasePlugin):
    """AgentRouter gateway provider (Responses, Anthropic, and Chat protocols)."""

    def __init__(self) -> None:
        self._metadata = PluginMetadata(
            id="agentrouter",
            name="AgentRouter",
            description=(
                "AgentRouter gateway with native Responses, Anthropic Messages, and "
                "OpenAI Chat protocols, vision, reasoning, tools, and prompt caching"
            ),
            version="1.0.0",
            author="LLMProxy",
            enabled=True,
            filename="agentrouter.py",
            supports_usage_meter=False,
            variables=[
                PluginVariable(
                    name="base_url",
                    label="Base URL",
                    type="string",
                    default=DEFAULT_BASE_URL,
                    description="AgentRouter gateway root (without /v1)",
                ),
                PluginVariable(
                    name="protocol",
                    label="Protocol",
                    type="select",
                    default="auto",
                    options=["auto", "responses", "openai", "anthropic"],
                    description=(
                        "auto: gpt-* uses Responses, claude-* uses Anthropic Messages, "
                        "other models use Chat Completions"
                    ),
                ),
                PluginVariable(
                    name="user_agent",
                    label="User-Agent",
                    type="string",
                    default=DEFAULT_USER_AGENT,
                    description=(
                        "AgentRouter rejects clients without an allow-listed User-Agent; "
                        "keep the opencode/<version> form"
                    ),
                ),
                PluginVariable(
                    name="anthropic_user_agent",
                    label="Anthropic User-Agent",
                    type="string",
                    default=DEFAULT_ANTHROPIC_USER_AGENT,
                    description="User-Agent used for the Anthropic Messages endpoint",
                ),
                PluginVariable(
                    name="anthropic_version",
                    label="Anthropic Version",
                    type="string",
                    default=DEFAULT_ANTHROPIC_VERSION,
                    description="anthropic-version header for the Messages endpoint",
                ),
                PluginVariable(
                    name="anthropic_beta",
                    label="Anthropic Beta Headers",
                    type="string",
                    default=DEFAULT_ANTHROPIC_BETA,
                    description="Comma-separated anthropic-beta header value",
                ),
                PluginVariable(
                    name="timeout",
                    label="Timeout (seconds)",
                    type="number",
                    default=180,
                    description="Upstream request timeout",
                ),
                PluginVariable(
                    name="reasoning_effort",
                    label="Default Reasoning Effort",
                    type="string",
                    default="default",
                    description=(
                        "Optional default level (none, minimal, low, medium, high, "
                        "xhigh, max). 'default' lets each client choose."
                    ),
                ),
                PluginVariable(
                    name="auto_prompt_cache",
                    label="Automatic Prompt Caching",
                    type="boolean",
                    default=True,
                    description=(
                        "Derive a stable prompt_cache_key for long Responses requests and "
                        "add Anthropic cache breakpoints when the client sends none"
                    ),
                ),
                PluginVariable(
                    name="prompt_cache_retention",
                    label="Prompt Cache Retention",
                    type="string",
                    default="",
                    description="Optional Responses prompt_cache_retention (for example 24h)",
                ),
                PluginVariable(
                    name="replay_reasoning",
                    label="Replay Reasoning Items",
                    type="boolean",
                    default=False,
                    description=(
                        "Replay cached Responses reasoning items with tool results. "
                        "Off by default because AgentRouter may route the replay to a "
                        "different backend and reject the item"
                    ),
                ),
                PluginVariable(
                    name="custom_headers",
                    label="Custom Headers",
                    type="string",
                    default="{}",
                    description="JSON object of extra HTTP headers sent upstream",
                ),
                PluginVariable(
                    name="load_balancing_strategy",
                    label="Load Balancing Strategy",
                    type="select",
                    default="round_robin",
                    options=["round_robin", "least_used", "fallback", "random"],
                    description="Strategy for distributing requests across accounts",
                ),
                PluginVariable(
                    name="account_switch_cooldown",
                    label="Account Cooldown (s)",
                    type="number",
                    default=10,
                    description="Cooldown after an account returns a rate limit",
                ),
                PluginVariable(
                    name="account_switch_wait",
                    label="Switch Wait Delay (s)",
                    type="number",
                    default=0,
                    description="Delay before failing over to another account",
                ),
                PluginVariable(
                    name="max_retries",
                    label="Max Plugin Retries",
                    type="number",
                    default=2,
                    description="Maximum account failovers per request",
                ),
                PluginVariable(
                    name="proxy_enabled",
                    label="Enable External Proxy",
                    type="boolean",
                    default=False,
                    description="Route requests through the dashboard proxy pool",
                ),
                PluginVariable(
                    name="proxy_strategy",
                    label="Proxy Strategy",
                    type="select",
                    default="sequential",
                    options=["sequential", "random", "specific", "switch_on_limit"],
                    description="Proxy pool selection strategy",
                ),
                PluginVariable(
                    name="specific_proxy_id",
                    label="Specific Proxy ID / Tag",
                    type="string",
                    default="",
                    description="Proxy ID, URL, or tag for the 'specific' strategy",
                ),
                PluginVariable(
                    name="proxy",
                    label="Direct Proxy URL",
                    type="string",
                    default="",
                    description="Direct HTTP or SOCKS5 proxy URL",
                ),
            ],
            models=[
                PluginModelConfig(
                    id="ar/gpt-5.6-sol",
                    name="GPT-5.6 Sol",
                    upstream_model="gpt-5.6-sol",
                    context_window=272000,
                    max_tokens=65536,
                    supports_vision=True,
                    reasoning_levels=list(RESPONSES_REASONING_LEVELS),
                    parameters={"protocol": "responses"},
                ),
                PluginModelConfig(
                    id="ar/gpt-6.1-sol",
                    name="GPT-6.1 Sol",
                    upstream_model="gpt-6.1-sol",
                    context_window=272000,
                    max_tokens=65536,
                    supports_vision=True,
                    reasoning_levels=list(RESPONSES_REASONING_LEVELS),
                    parameters={"protocol": "responses"},
                ),
                PluginModelConfig(
                    id="ar/gpt-6-astra",
                    name="GPT-6 Astra",
                    upstream_model="gpt-6-astra",
                    context_window=272000,
                    max_tokens=65536,
                    supports_vision=True,
                    reasoning_levels=list(RESPONSES_REASONING_LEVELS),
                    parameters={"protocol": "responses"},
                ),
                PluginModelConfig(
                    id="ar/claude-opus-5",
                    name="Claude Opus 5",
                    upstream_model="claude-opus-5",
                    context_window=200000,
                    max_tokens=64000,
                    supports_vision=True,
                    reasoning_levels=list(ANTHROPIC_REASONING_LEVELS),
                    parameters={"protocol": "anthropic"},
                ),
                PluginModelConfig(
                    id="ar/claude-opus-4-8",
                    name="Claude Opus 4.8",
                    upstream_model="claude-opus-4-8",
                    context_window=200000,
                    max_tokens=64000,
                    supports_vision=True,
                    reasoning_levels=list(ANTHROPIC_REASONING_LEVELS),
                    parameters={"protocol": "anthropic"},
                ),
                PluginModelConfig(
                    id="ar/glm-5.3",
                    name="GLM 5.3",
                    upstream_model="glm-5.3",
                    context_window=128000,
                    max_tokens=32768,
                    supports_vision=False,
                    parameters={"protocol": "openai"},
                ),
                PluginModelConfig(
                    id="ar/deepseek-v4-flash",
                    name="DeepSeek V4 Flash",
                    upstream_model="deepseek-v4-flash",
                    context_window=128000,
                    max_tokens=32768,
                    supports_vision=False,
                    parameters={"protocol": "openai"},
                ),
            ],
        )

    @property
    def metadata(self) -> PluginMetadata:
        return self._metadata

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_test_mode(api_key: str, account: Optional[Account] = None) -> bool:
        key = (api_key or "").strip()
        if not key:
            if account is not None and not str(account.id).startswith("default-"):
                raise ValueError(
                    f"Account '{account.name}' has no API key configured. "
                    "Add an AgentRouter API key in the Accounts tab."
                )
            return True
        lowered = key.lower()
        prefixes = ("test", "mock", "sk-test", "dummy", "fake", "offline", "sample")
        return any(lowered.startswith(prefix) for prefix in prefixes)

    @staticmethod
    def _api_key(account: Optional[Account]) -> str:
        return str(account.api_key or "").strip() if account else ""

    @staticmethod
    def _root_url(base_url: str) -> str:
        root = str(base_url or DEFAULT_BASE_URL).strip().rstrip("/")
        if root.endswith("/v1"):
            root = root[:-3].rstrip("/")
        return root or DEFAULT_BASE_URL

    @staticmethod
    def _user_agent(value: Any, default: str) -> str:
        agent = str(value or "").strip() or default
        # AgentRouter rejects bare client names; it expects name/version.
        if "/" not in agent:
            agent = f"{agent}/1.0.0"
        return agent

    @staticmethod
    def _resolve_protocol(
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        variables: Dict[str, Any],
        account: Optional[Account] = None,
    ) -> str:
        parameters = model_config.parameters or {}
        extra = request.extra_body if isinstance(request.extra_body, dict) else {}
        candidates = [
            account.extra_config.get("protocol") if account else None,
            parameters.get("protocol"),
            extra.get("protocol"),
            variables.get("protocol"),
        ]
        for candidate in candidates:
            value = str(candidate or "").strip().lower()
            if value in (PROTOCOL_RESPONSES, PROTOCOL_ANTHROPIC, PROTOCOL_OPENAI):
                return value

        name = str(model_config.upstream_model or model_config.id or "").lower()
        if "claude" in name:
            return PROTOCOL_ANTHROPIC
        if name.startswith(("gpt-", "o1", "o3", "o4", "o5")) or "codex" in name:
            return PROTOCOL_RESPONSES
        return PROTOCOL_OPENAI

    @staticmethod
    def _resolve_reasoning(
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        variables: Dict[str, Any],
    ) -> Tuple[str, Optional[str]]:
        """Return ``(upstream_model, reasoning_level)``."""
        levels = RESPONSES_REASONING_LEVELS + ANTHROPIC_REASONING_LEVELS
        raw_model = str(model_config.upstream_model or model_config.id or "")
        upstream_model, suffix_level = _split_level(raw_model, levels)

        extra = request.extra_body if isinstance(request.extra_body, dict) else {}
        level = ""
        if request.reasoning_effort:
            level = str(request.reasoning_effort)
        elif extra.get("reasoning_effort"):
            level = str(extra.get("reasoning_effort"))
        else:
            reasoning = extra.get("reasoning")
            if isinstance(reasoning, dict):
                level = str(reasoning.get("effort") or "")
            elif isinstance(reasoning, str):
                level = reasoning
        if not level and variables.get("reasoning_effort"):
            level = str(variables.get("reasoning_effort"))
        if not level and suffix_level:
            level = suffix_level
        if not level:
            level = str(model_config.parameters.get("reasoning_effort") or "")

        normalized = level.strip().lower()
        if normalized in ("auto", "default", "off", "disabled", ""):
            normalized = ""
        if normalized not in RESPONSES_REASONING_LEVELS and normalized != "none":
            normalized = ""
        return upstream_model, (normalized or None)

    @staticmethod
    def _headers(
        protocol: str,
        api_key: str,
        variables: Dict[str, Any],
        account: Optional[Account] = None,
    ) -> Dict[str, str]:
        default_ua = (
            DEFAULT_ANTHROPIC_USER_AGENT
            if protocol == PROTOCOL_ANTHROPIC
            else DEFAULT_USER_AGENT
        )
        ua_setting = (
            variables.get("anthropic_user_agent")
            if protocol == PROTOCOL_ANTHROPIC
            else variables.get("user_agent")
        )
        if account and account.extra_config.get("user_agent"):
            ua_setting = account.extra_config.get("user_agent")
        user_agent = AgentRouterPlugin._user_agent(ua_setting, default_ua)

        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "User-Agent": user_agent,
        }
        if protocol == PROTOCOL_ANTHROPIC:
            headers.update(
                {
                    "x-api-key": api_key,
                    "Authorization": f"Bearer {api_key}",
                    "anthropic-version": str(
                        variables.get("anthropic_version") or DEFAULT_ANTHROPIC_VERSION
                    ).strip(),
                    "x-app": "cli",
                    "anthropic-dangerous-direct-browser-access": "true",
                    "x-stainless-lang": "js",
                    "x-stainless-package-version": "0.55.1",
                    "x-stainless-os": "Linux",
                    "x-stainless-arch": "x64",
                    "x-stainless-runtime": "node",
                    "x-stainless-runtime-version": "v22.0.0",
                }
            )
            beta = str(variables.get("anthropic_beta") or "").strip()
            if beta:
                headers["anthropic-beta"] = beta
        else:
            headers["Authorization"] = f"Bearer {api_key}"

        headers.update(_json_object(variables.get("custom_headers")))
        return headers

    @staticmethod
    def _proxy(variables: Dict[str, Any], account: Optional[Account]) -> Optional[str]:
        value = str(
            variables.get("proxy")
            or (getattr(account, "proxy_url", None) if account else "")
            or ""
        ).strip()
        return value or None

    @staticmethod
    def _timeout(variables: Dict[str, Any]) -> float:
        try:
            return max(5.0, float(variables.get("timeout") or 180))
        except (TypeError, ValueError):
            return 180.0

    # ------------------------------------------------------------------
    # Usage helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _responses_usage(usage: Optional[Dict[str, Any]]) -> CompletionUsage:
        usage = usage or {}
        prompt_tokens = _as_int(usage.get("input_tokens")) or 0
        completion_tokens = _as_int(usage.get("output_tokens")) or 0
        total_tokens = _as_int(usage.get("total_tokens")) or (
            prompt_tokens + completion_tokens
        )

        input_details = usage.get("input_tokens_details") or {}
        output_details = usage.get("output_tokens_details") or {}
        cached_tokens = _as_int(input_details.get("cached_tokens"))
        cache_write_tokens = _as_int(input_details.get("cache_write_tokens"))
        reasoning_tokens = _as_int(output_details.get("reasoning_tokens"))

        result = CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
        if cached_tokens is not None:
            result.prompt_tokens_details = {"cached_tokens": cached_tokens}
            result.cache_read_input_tokens = cached_tokens
        if cache_write_tokens is not None:
            result.cache_creation_input_tokens = cache_write_tokens
        if reasoning_tokens is not None:
            result.completion_tokens_details = {"reasoning_tokens": reasoning_tokens}
        return result

    @staticmethod
    def _anthropic_usage(
        usage: Optional[Dict[str, Any]],
        previous: Optional[CompletionUsage] = None,
    ) -> CompletionUsage:
        usage = usage or {}
        cache_read = _as_int(usage.get("cache_read_input_tokens"))
        cache_created = _as_int(usage.get("cache_creation_input_tokens"))
        if cache_read is None and previous is not None:
            cache_read = previous.cache_read_input_tokens
        if cache_created is None and previous is not None:
            cache_created = previous.cache_creation_input_tokens

        reported_input = _as_int(usage.get("input_tokens"))
        if reported_input is not None:
            prompt_tokens = reported_input + (cache_read or 0) + (cache_created or 0)
        elif previous is not None:
            prompt_tokens = previous.prompt_tokens
        else:
            prompt_tokens = 0
        completion_tokens = _as_int(usage.get("output_tokens"))
        if completion_tokens is None:
            completion_tokens = previous.completion_tokens if previous else 0
        total = _as_int(usage.get("total_tokens")) or (prompt_tokens + completion_tokens)

        result = CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total,
        )
        if cache_read is not None:
            result.cache_read_input_tokens = cache_read
            result.prompt_tokens_details = {"cached_tokens": cache_read}
        if cache_created is not None:
            result.cache_creation_input_tokens = cache_created
        reasoning_tokens = _as_int(
            (usage.get("output_tokens_details") or {}).get("reasoning_tokens")
        )
        if reasoning_tokens is not None:
            result.completion_tokens_details = {"reasoning_tokens": reasoning_tokens}
        return result

    # ------------------------------------------------------------------
    # Upstream error helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def _raise_for_status(response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        try:
            raw = await response.aread()
            body = raw.decode("utf-8", errors="replace")
        except Exception:
            body = ""
        excerpt = re.sub(r"\s+", " ", body).strip()[:500]
        message = f"AgentRouter upstream error {response.status_code}"
        if excerpt:
            message = f"{message}: {excerpt}"
        raise httpx.HTTPStatusError(
            message, request=response.request, response=response
        )

    # ------------------------------------------------------------------
    # OpenAI Chat Completions protocol
    # ------------------------------------------------------------------

    @staticmethod
    def _chat_message_dict(message: ChatMessage) -> Dict[str, Any]:
        payload = message.model_dump(exclude_none=True)
        content = payload.get("content")
        if isinstance(content, list):
            normalized: List[Any] = []
            for part in content:
                if is_image_part(part):
                    converted = to_openai_image_part(part)
                    if converted is not None:
                        normalized.append(converted)
                    continue
                if isinstance(part, dict):
                    text_part = dict(part)
                    if text_part.get("type") in ("input_text", "output_text"):
                        text_part["type"] = "text"
                    normalized.append(text_part)
                else:
                    normalized.append(part)
            payload["content"] = normalized
        return payload

    def _build_chat_payload(
        self,
        request: ChatCompletionRequest,
        upstream_model: str,
        reasoning: Optional[str],
        variables: Dict[str, Any],
    ) -> Dict[str, Any]:
        extra = request.extra_body if isinstance(request.extra_body, dict) else {}
        payload: Dict[str, Any] = {}
        for key, value in extra.items():
            if key in ("messages", "model", "stream", "api_key", "headers"):
                continue
            payload[key] = value

        payload["model"] = upstream_model
        payload["messages"] = [self._chat_message_dict(m) for m in request.messages]
        payload["stream"] = bool(request.stream)

        if request.temperature is not None and request.temperature != 1.0:
            payload["temperature"] = request.temperature
        if request.top_p is not None and request.top_p != 1.0:
            payload["top_p"] = request.top_p
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.max_completion_tokens is not None:
            payload["max_completion_tokens"] = request.max_completion_tokens
        if request.stop is not None:
            payload["stop"] = request.stop
        if request.presence_penalty:
            payload["presence_penalty"] = request.presence_penalty
        if request.frequency_penalty:
            payload["frequency_penalty"] = request.frequency_penalty
        if request.user:
            payload["user"] = request.user
        if request.n and request.n != 1:
            payload["n"] = request.n
        if request.response_format:
            payload["response_format"] = request.response_format
        if request.tools:
            payload["tools"] = request.tools
        if request.tool_choice is not None:
            payload["tool_choice"] = request.tool_choice
        if request.parallel_tool_calls is not None:
            payload["parallel_tool_calls"] = request.parallel_tool_calls
        if request.functions:
            payload["functions"] = request.functions
        if request.function_call is not None:
            payload["function_call"] = request.function_call
        if reasoning:
            payload["reasoning_effort"] = reasoning

        stream_options = dict(request.stream_options or {})
        stream_options["include_usage"] = True
        if request.stream:
            payload["stream_options"] = stream_options
        elif "stream_options" in payload:
            payload.pop("stream_options", None)
        return payload

    async def _stream_chat(
        self,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        model_config: PluginModelConfig,
        account: Optional[Account],
        proxy: Optional[str],
        timeout: float,
    ) -> AsyncIterator[ChatCompletionChunk]:
        chunk_id = f"chatcmpl-ar-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        finish_sent = False
        last_usage: Optional[CompletionUsage] = None

        async with httpx.AsyncClient(timeout=timeout, proxy=proxy) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                if account is not None:
                    capture_rate_limit_headers(account, response.headers)
                await self._raise_for_status(response)
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
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]" or data.lower() == "null":
                        continue
                    try:
                        event = json.loads(data)
                    except Exception:
                        continue
                    if not isinstance(event, dict):
                        continue
                    if event.get("error"):
                        error = event.get("error")
                        message = (
                            error.get("message") if isinstance(error, dict) else str(error)
                        )
                        raise RuntimeError(f"AgentRouter streaming error: {message}")
                    try:
                        chunk = ChatCompletionChunk.model_validate(event)
                    except Exception:
                        continue
                    if chunk.usage is not None:
                        last_usage = chunk.usage
                    if any(choice.finish_reason for choice in chunk.choices):
                        finish_sent = True
                    yield chunk

        if not finish_sent:
            yield ChatCompletionChunk(
                id=chunk_id,
                created=created,
                model=model_config.id,
                choices=[ChatChoiceDelta(index=0, delta={}, finish_reason="stop")],
                usage=last_usage,
            )

    # ------------------------------------------------------------------
    # OpenAI Responses protocol
    # ------------------------------------------------------------------

    @staticmethod
    def _responses_cache_key(request: ChatCompletionRequest) -> Optional[str]:
        texts: List[str] = []
        for message in request.messages:
            if message.role == "system":
                texts.append(content_text(message.content))
        for message in request.messages:
            if message.role == "user":
                texts.append(content_text(message.content))
                break
        fingerprint = "\n".join(part for part in texts if part)
        if len(fingerprint) < 4096:
            return None
        digest = hashlib.sha256(fingerprint.encode("utf-8", errors="replace")).hexdigest()
        return f"llmproxy-ar-{digest[:24]}"

    def _responses_input(
        self, request: ChatCompletionRequest, replay_reasoning: bool = False
    ) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        instructions: List[str] = []
        items: List[Dict[str, Any]] = []

        for message in request.messages:
            role = message.role
            if role in ("system", "developer"):
                text = content_text(message.content)
                if text:
                    instructions.append(text)
                continue

            if role in ("tool", "function"):
                call_id = (
                    message.tool_call_id
                    or (message.model_extra or {}).get("call_id")
                    or "call_unknown"
                )
                if isinstance(message.content, list):
                    output = content_text(message.content)
                else:
                    output = "" if message.content is None else str(message.content)
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": output,
                    }
                )
                continue

            if role == "assistant":
                text = content_text(message.content)
                if text:
                    items.append(
                        {
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": text}],
                        }
                    )
                for tool_call in message.tool_calls or []:
                    if not isinstance(tool_call, dict):
                        continue
                    function = tool_call.get("function") or {}
                    call_id = tool_call.get("id") or function.get("call_id") or "call_unknown"
                    if replay_reasoning:
                        for reasoning_item in _cached_reasoning(call_id):
                            items.append(reasoning_item)
                    arguments = function.get("arguments", "{}")
                    if not isinstance(arguments, str):
                        arguments = json.dumps(arguments, ensure_ascii=False)
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": call_id,
                            "name": function.get("name", ""),
                            "arguments": arguments,
                        }
                    )
                continue

            # user / developer-like input
            parts: List[Dict[str, Any]] = []
            content = message.content
            if isinstance(content, str):
                if content:
                    parts.append({"type": "input_text", "text": content})
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, str):
                        parts.append({"type": "input_text", "text": part})
                        continue
                    if not isinstance(part, dict):
                        continue
                    if is_image_part(part):
                        url = extract_image_url(part)
                        if url:
                            image_part: Dict[str, Any] = {
                                "type": "input_image",
                                "image_url": url,
                            }
                            detail = image_detail(part)
                            if detail:
                                image_part["detail"] = detail
                            parts.append(image_part)
                        continue
                    text = part.get("text")
                    if isinstance(text, str):
                        parts.append({"type": "input_text", "text": text})
                    else:
                        parts.append(dict(part))
            if not parts:
                parts.append({"type": "input_text", "text": ""})
            items.append({"role": "user", "content": parts})

        if not items:
            items.append(
                {"role": "user", "content": [{"type": "input_text", "text": "Hello"}]}
            )
        joined = "\n\n".join(text for text in instructions if text)
        return (joined or None), items

    @staticmethod
    def _responses_tools(tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        if not tools:
            return None
        converted: List[Dict[str, Any]] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            if tool.get("type") == "function" and "function" in tool:
                function = tool.get("function") or {}
                item: Dict[str, Any] = {
                    "type": "function",
                    "name": function.get("name", ""),
                    "description": function.get("description", ""),
                    "parameters": function.get("parameters")
                    or {"type": "object", "properties": {}},
                }
                if function.get("strict") is not None:
                    item["strict"] = function["strict"]
                converted.append(item)
            elif tool.get("type") == "function" and "name" in tool:
                converted.append(dict(tool))
            else:
                converted.append(dict(tool))
        return converted or None

    @staticmethod
    def _responses_tool_choice(tool_choice: Any) -> Any:
        if not tool_choice:
            return None
        if isinstance(tool_choice, str):
            return tool_choice if tool_choice in ("auto", "required", "none") else None
        if isinstance(tool_choice, dict):
            if tool_choice.get("type") == "function":
                name = (
                    (tool_choice.get("function") or {}).get("name")
                    or tool_choice.get("name")
                )
                if name:
                    return {"type": "function", "name": name}
            if tool_choice.get("type") in ("auto", "required", "none"):
                return tool_choice["type"]
        return None

    @staticmethod
    def _responses_text_format(response_format: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(response_format, dict):
            return None
        format_type = response_format.get("type")
        if format_type == "json_schema":
            schema = response_format.get("json_schema") or {}
            body: Dict[str, Any] = {"type": "json_schema"}
            for key in ("name", "description", "schema", "strict"):
                if schema.get(key) is not None:
                    body[key] = schema[key]
            return {"format": body}
        if format_type == "json_object":
            return {"format": {"type": "json_object"}}
        if format_type == "text":
            return {"format": {"type": "text"}}
        return None

    def _build_responses_payload(
        self,
        request: ChatCompletionRequest,
        upstream_model: str,
        reasoning: Optional[str],
        variables: Dict[str, Any],
        replay_reasoning: bool = False,
    ) -> Dict[str, Any]:
        extra = request.extra_body if isinstance(request.extra_body, dict) else {}
        instructions, items = self._responses_input(
            request, replay_reasoning=replay_reasoning
        )

        payload: Dict[str, Any] = {
            "model": upstream_model,
            "input": items,
            "stream": bool(request.stream),
        }
        if instructions:
            payload["instructions"] = instructions
        if extra.get("instructions"):
            payload["instructions"] = str(extra["instructions"])

        for key in RESPONSES_EXTRA_KEYS:
            if key in extra and extra[key] is not None:
                payload[key] = extra[key]

        max_tokens = request.max_completion_tokens or request.max_tokens
        if max_tokens:
            payload["max_output_tokens"] = int(max_tokens)
        if payload.get("max_output_tokens") is not None:
            floor = RESPONSES_MIN_OUTPUT_TOKENS
            if messages_have_images(request.messages):
                floor = max(floor, RESPONSES_VISION_OUTPUT_FLOOR)
            if reasoning in ("high", "xhigh", "max"):
                floor = max(floor, RESPONSES_HIGH_EFFORT_FLOOR)
            try:
                payload["max_output_tokens"] = max(floor, int(payload["max_output_tokens"]))
            except (TypeError, ValueError):
                payload.pop("max_output_tokens", None)

        if request.temperature is not None and request.temperature != 1.0:
            payload["temperature"] = request.temperature
        if request.top_p is not None and request.top_p != 1.0:
            payload["top_p"] = request.top_p
        if request.parallel_tool_calls is not None:
            payload["parallel_tool_calls"] = request.parallel_tool_calls
        if request.user:
            payload.setdefault("user", request.user)

        tools = self._responses_tools(request.tools)
        if tools:
            payload["tools"] = tools
            tool_choice = self._responses_tool_choice(request.tool_choice)
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice

        text_format = self._responses_text_format(request.response_format)
        if text_format:
            text = dict(payload.get("text") or {})
            text.update(text_format)
            payload["text"] = text

        if reasoning and reasoning != "none":
            reasoning_body: Dict[str, Any] = {"effort": reasoning, "summary": "auto"}
            requested = extra.get("reasoning")
            if isinstance(requested, dict):
                reasoning_body.update(requested)
            payload["reasoning"] = reasoning_body

        include = payload.get("include")
        include_list = list(include) if isinstance(include, list) else []
        if "reasoning.encrypted_content" not in include_list:
            include_list.append("reasoning.encrypted_content")
        payload["include"] = include_list

        cache_key = payload.get("prompt_cache_key")
        if not cache_key and _as_bool(variables.get("auto_prompt_cache"), True):
            cache_key = self._responses_cache_key(request)
        if cache_key:
            payload["prompt_cache_key"] = cache_key

        retention = payload.get("prompt_cache_retention") or variables.get(
            "prompt_cache_retention"
        )
        if retention:
            payload["prompt_cache_retention"] = str(retention)
        return payload

    @staticmethod
    def _responses_output_text(item: Dict[str, Any]) -> str:
        parts: List[str] = []
        for content in item.get("content") or []:
            if not isinstance(content, dict):
                continue
            content_type = content.get("type")
            if content_type in ("output_text", "text"):
                parts.append(str(content.get("text") or ""))
            elif content_type == "refusal":
                parts.append(str(content.get("refusal") or ""))
        return "".join(parts)

    @staticmethod
    def _responses_item_reasoning(item: Dict[str, Any]) -> str:
        parts: List[str] = []
        for summary in item.get("summary") or []:
            text = _reasoning_text(summary)
            if text:
                parts.append(text)
        for content in item.get("content") or []:
            text = _reasoning_text(content)
            if text:
                parts.append(text)
        return "".join(parts)

    def _responses_to_response(
        self, data: Dict[str, Any], model_config: PluginModelConfig
    ) -> ChatCompletionResponse:
        status = str(data.get("status") or "")
        if status == "failed":
            error = data.get("error") or {}
            message = (
                error.get("message") if isinstance(error, dict) else str(error)
            )
            raise RuntimeError(f"AgentRouter response failed: {message or 'unknown error'}")

        text_parts: List[str] = []
        reasoning_parts: List[str] = []
        reasoning_items: List[Dict[str, Any]] = []
        tool_calls: List[Dict[str, Any]] = []
        call_ids: List[str] = []

        for item in data.get("output") or []:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "message":
                text_parts.append(self._responses_output_text(item))
            elif item_type == "reasoning":
                reasoning_items.append(dict(item))
                text = self._responses_item_reasoning(item)
                if text:
                    reasoning_parts.append(text)
            elif item_type == "function_call":
                call_id = str(
                    item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex[:12]}"
                )
                call_ids.append(call_id)
                tool_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": item.get("name", ""),
                            "arguments": item.get("arguments") or "{}",
                        },
                    }
                )

        for call_id in call_ids:
            _remember_reasoning(call_id, reasoning_items)

        if tool_calls:
            finish_reason = "tool_calls"
        elif status == "incomplete":
            details = data.get("incomplete_details") or {}
            finish_reason = (
                "length"
                if details.get("reason") in ("max_output_tokens", "max_tokens")
                else "stop"
            )
        elif self._responses_blocked(data):
            finish_reason = "content_filter"
        else:
            finish_reason = "stop"

        content = "".join(text_parts) if text_parts else (None if tool_calls else "")
        message_kwargs: Dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            message_kwargs["tool_calls"] = tool_calls
        reasoning_content = "".join(reasoning_parts)
        if reasoning_content:
            message_kwargs["reasoning_content"] = reasoning_content

        return ChatCompletionResponse(
            id=f"chatcmpl-{data.get('id') or uuid.uuid4().hex[:12]}",
            created=int(data.get("created_at") or time.time()),
            model=model_config.id,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(**message_kwargs),
                    finish_reason=finish_reason,
                )
            ],
            usage=self._responses_usage(data.get("usage")),
        )

    @staticmethod
    def _responses_blocked(data: Dict[str, Any]) -> bool:
        for entry in data.get("content_filters") or []:
            if isinstance(entry, dict) and entry.get("blocked"):
                return True
        return False

    async def _stream_responses(
        self,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        model_config: PluginModelConfig,
        account: Optional[Account],
        proxy: Optional[str],
        timeout: float,
    ) -> AsyncIterator[ChatCompletionChunk]:
        chunk_id = f"chatcmpl-ar-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        reasoning_items: List[Dict[str, Any]] = []
        tool_calls: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        sent_items: set = set()
        finish_sent = False
        role_sent = False
        content_sent = False
        reasoning_sent = False
        last_usage: Optional[CompletionUsage] = None

        def make_chunk(
            delta: Dict[str, Any],
            finish_reason: Optional[str] = None,
            usage: Optional[CompletionUsage] = None,
        ) -> ChatCompletionChunk:
            return ChatCompletionChunk(
                id=chunk_id,
                created=created,
                model=model_config.id,
                choices=[
                    ChatChoiceDelta(index=0, delta=delta, finish_reason=finish_reason)
                ],
                usage=usage,
            )

        async with httpx.AsyncClient(timeout=timeout, proxy=proxy) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                if account is not None:
                    capture_rate_limit_headers(account, response.headers)
                await self._raise_for_status(response)

                role_sent = True
                yield make_chunk({"role": "assistant", "content": ""})

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if not data_str or data_str == "[DONE]" or data_str.lower() == "null":
                        continue
                    try:
                        event = json.loads(data_str)
                    except Exception:
                        continue
                    if not isinstance(event, dict):
                        continue

                    event_type = str(event.get("type") or "")
                    if event_type in ("error", "response.error"):
                        error = event.get("error") or event
                        message = (
                            error.get("message") if isinstance(error, dict) else str(error)
                        )
                        raise RuntimeError(f"AgentRouter streaming error: {message}")
                    if event_type == "response.created":
                        response_data = event.get("response") or {}
                        chunk_id = f"chatcmpl-{response_data.get('id') or uuid.uuid4().hex[:12]}"
                        created = int(response_data.get("created_at") or created)

                    if event_type == "response.output_text.delta":
                        delta_text = str(event.get("delta") or "")
                        if delta_text:
                            content_sent = True
                            yield make_chunk({"content": delta_text})
                        continue

                    if event_type in (
                        "response.reasoning_summary_text.delta",
                        "response.reasoning_text.delta",
                        "response.reasoning_summary_part.added",
                    ):
                        reasoning_text = _reasoning_text(event)
                        if reasoning_text:
                            reasoning_sent = True
                            yield make_chunk({"reasoning_content": reasoning_text})
                        continue

                    if event_type == "response.output_item.added":
                        item = event.get("item") or {}
                        if item.get("type") == "function_call":
                            item_id = str(item.get("id") or "")
                            call_id = str(
                                item.get("call_id")
                                or item_id
                                or f"call_{uuid.uuid4().hex[:12]}"
                            )
                            index = len(tool_calls)
                            tool_calls[item_id] = {
                                "index": index,
                                "id": call_id,
                                "name": item.get("name", ""),
                                "arguments": "",
                                "delta_sent": False,
                            }
                            sent_items.add(item_id)
                            yield make_chunk(
                                {
                                    "tool_calls": [
                                        {
                                            "index": index,
                                            "id": call_id,
                                            "type": "function",
                                            "function": {
                                                "name": item.get("name", ""),
                                                "arguments": "",
                                            },
                                        }
                                    ]
                                }
                            )
                        continue

                    if event_type == "response.function_call_arguments.delta":
                        item_id = str(event.get("item_id") or "")
                        state = tool_calls.get(item_id)
                        delta_text = str(event.get("delta") or "")
                        if state is not None and delta_text:
                            state["arguments"] += delta_text
                            state["delta_sent"] = True
                            yield make_chunk(
                                {
                                    "tool_calls": [
                                        {
                                            "index": state["index"],
                                            "function": {"arguments": delta_text},
                                        }
                                    ]
                                }
                            )
                        continue

                    if event_type == "response.function_call_arguments.done":
                        item_id = str(event.get("item_id") or "")
                        state = tool_calls.get(item_id)
                        if state is not None and not state["delta_sent"]:
                            arguments = str(event.get("arguments") or "")
                            if arguments:
                                state["arguments"] = arguments
                                yield make_chunk(
                                    {
                                        "tool_calls": [
                                            {
                                                "index": state["index"],
                                                "function": {"arguments": arguments},
                                            }
                                        ]
                                    }
                                )
                        continue

                    if event_type == "response.output_item.done":
                        item = event.get("item") or {}
                        if item.get("type") == "reasoning":
                            reasoning_items.append(dict(item))
                        elif item.get("type") == "function_call":
                            item_id = str(item.get("id") or "")
                            call_id = str(item.get("call_id") or item_id or "")
                            state = tool_calls.get(item_id)
                            arguments = str(item.get("arguments") or "")
                            if state is None:
                                index = len(tool_calls)
                                state = {
                                    "index": index,
                                    "id": call_id or f"call_{uuid.uuid4().hex[:12]}",
                                    "name": item.get("name", ""),
                                    "arguments": "",
                                    "delta_sent": False,
                                }
                                tool_calls[item_id] = state
                                yield make_chunk(
                                    {
                                        "tool_calls": [
                                            {
                                                "index": index,
                                                "id": state["id"],
                                                "type": "function",
                                                "function": {
                                                    "name": state["name"],
                                                    "arguments": "",
                                                },
                                            }
                                        ]
                                    }
                                )
                            if arguments and not state["delta_sent"]:
                                state["arguments"] = arguments
                                yield make_chunk(
                                    {
                                        "tool_calls": [
                                            {
                                                "index": state["index"],
                                                "function": {"arguments": arguments},
                                            }
                                        ]
                                    }
                                )
                            if state["id"]:
                                _remember_reasoning(state["id"], reasoning_items)
                        continue

                    if event_type in ("response.completed", "response.incomplete"):
                        response_data = event.get("response") or {}
                        output = response_data.get("output") or []
                        for item in output:
                            if not isinstance(item, dict):
                                continue
                            if item.get("type") == "reasoning":
                                reasoning_items.append(dict(item))
                                if not reasoning_sent:
                                    text = self._responses_item_reasoning(item)
                                    if text:
                                        reasoning_sent = True
                                        yield make_chunk({"reasoning_content": text})
                        for item in output:
                            if not isinstance(item, dict):
                                continue
                            if item.get("type") == "function_call":
                                call_id = str(item.get("call_id") or item.get("id") or "")
                                if call_id:
                                    _remember_reasoning(call_id, reasoning_items)

                        if not content_sent and not tool_calls:
                            for item in output:
                                if not isinstance(item, dict):
                                    continue
                                if item.get("type") == "message":
                                    text = self._responses_output_text(item)
                                    if text:
                                        content_sent = True
                                        yield make_chunk({"content": text})
                                elif item.get("type") == "function_call":
                                    call_id = str(
                                        item.get("call_id")
                                        or item.get("id")
                                        or f"call_{uuid.uuid4().hex[:12]}"
                                    )
                                    index = len(tool_calls)
                                    arguments = str(item.get("arguments") or "{}")
                                    tool_calls[str(item.get("id") or call_id)] = {
                                        "index": index,
                                        "id": call_id,
                                        "name": item.get("name", ""),
                                        "arguments": arguments,
                                        "delta_sent": True,
                                    }
                                    yield make_chunk(
                                        {
                                            "tool_calls": [
                                                {
                                                    "index": index,
                                                    "id": call_id,
                                                    "type": "function",
                                                    "function": {
                                                        "name": item.get("name", ""),
                                                        "arguments": arguments,
                                                    },
                                                }
                                            ]
                                        }
                                    )

                        last_usage = self._responses_usage(response_data.get("usage"))
                        if tool_calls:
                            finish_reason = "tool_calls"
                        elif event_type == "response.incomplete":
                            finish_reason = "length"
                        elif self._responses_blocked(response_data):
                            finish_reason = "content_filter"
                        else:
                            finish_reason = "stop"
                        yield make_chunk({}, finish_reason=finish_reason, usage=last_usage)
                        finish_sent = True
                        break

                    if event_type == "response.failed":
                        response_data = event.get("response") or {}
                        error = response_data.get("error") or {}
                        message = (
                            error.get("message")
                            if isinstance(error, dict)
                            else str(error)
                        )
                        raise RuntimeError(
                            f"AgentRouter response failed: {message or 'unknown error'}"
                        )

        if not finish_sent:
            yield make_chunk(
                {},
                finish_reason="tool_calls" if tool_calls else "stop",
                usage=last_usage,
            )

    # ------------------------------------------------------------------
    # Anthropic Messages protocol
    # ------------------------------------------------------------------

    @staticmethod
    def _anthropic_tools(
        tools: Optional[List[Dict[str, Any]]]
    ) -> Optional[List[Dict[str, Any]]]:
        if not tools:
            return None
        converted: List[Dict[str, Any]] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            if "function" in tool:
                function = tool.get("function") or {}
                name = function.get("name", "")
                description = function.get("description", "")
                schema = function.get("parameters") or {"type": "object", "properties": {}}
            else:
                name = tool.get("name", "")
                description = tool.get("description", "")
                schema = tool.get("input_schema") or tool.get("parameters") or {
                    "type": "object",
                    "properties": {},
                }
            if not isinstance(schema, dict) or schema.get("type") != "object":
                schema = {"type": "object", "properties": {}}
            item: Dict[str, Any] = {
                "name": name,
                "description": description,
                "input_schema": schema,
            }
            cache_control = tool.get("cache_control")
            if cache_control:
                item["cache_control"] = cache_control
            converted.append(item)
        return converted or None

    @staticmethod
    def _anthropic_tool_choice(tool_choice: Any) -> Optional[Dict[str, Any]]:
        if not tool_choice or tool_choice == "auto":
            return {"type": "auto"}
        if tool_choice == "required":
            return {"type": "any"}
        if tool_choice == "none":
            return None
        if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            name = (tool_choice.get("function") or {}).get("name") or tool_choice.get(
                "name"
            )
            if name:
                return {"type": "tool", "name": name}
        return None

    @staticmethod
    def _anthropic_blocks(content: Any, allow_images: bool = True) -> List[Dict[str, Any]]:
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        blocks: List[Dict[str, Any]] = []
        if not isinstance(content, list):
            return blocks
        for part in content:
            if isinstance(part, str):
                blocks.append({"type": "text", "text": part})
                continue
            if not isinstance(part, dict):
                continue
            if is_image_part(part):
                if not allow_images:
                    blocks.append(
                        {
                            "type": "text",
                            "text": "[Image omitted: images are not allowed in this message role]",
                        }
                    )
                    continue
                image_block = to_anthropic_image_block(part)
                if image_block is not None:
                    if part.get("cache_control"):
                        image_block.setdefault("cache_control", part["cache_control"])
                    blocks.append(image_block)
                else:
                    blocks.append({"type": "text", "text": "[Image could not be loaded]"})
                continue
            part_type = str(part.get("type") or "").lower()
            if part_type in ("text", "input_text", "output_text"):
                text_block = dict(part)
                text_block["type"] = "text"
                text_block["text"] = str(part.get("text") or "")
                blocks.append(text_block)
                continue
            blocks.append(dict(part))
        return blocks

    def _anthropic_messages(
        self, messages: List[ChatMessage], thinking_active: bool
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        system_blocks: List[Dict[str, Any]] = []
        converted: List[Dict[str, Any]] = []

        for message in messages:
            if message.role == "system":
                system_blocks.extend(self._anthropic_blocks(message.content, allow_images=False))
                continue

            if message.role in ("tool", "function"):
                tool_use_id = message.tool_call_id or "call_unknown"
                if isinstance(message.content, list):
                    tool_content: Any = self._anthropic_blocks(message.content)
                else:
                    tool_content = "" if message.content is None else str(message.content)
                tool_block = {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": tool_content,
                }
                if converted and converted[-1]["role"] == "user":
                    previous = converted[-1]["content"]
                    if isinstance(previous, list):
                        previous.append(tool_block)
                    else:
                        converted[-1]["content"] = [
                            {"type": "text", "text": str(previous)},
                            tool_block,
                        ]
                else:
                    converted.append({"role": "user", "content": [tool_block]})
                continue

            if message.role == "assistant":
                text_blocks = (
                    self._anthropic_blocks(message.content, allow_images=False)
                    if message.content
                    else []
                )
                blocks: List[Dict[str, Any]] = []
                text_placed = False
                explicit_thinking = (message.model_extra or {}).get("thinking_blocks")

                for index, tool_call in enumerate(message.tool_calls or []):
                    if not isinstance(tool_call, dict):
                        continue
                    thinking: List[Dict[str, Any]] = []
                    if thinking_active:
                        if index == 0 and isinstance(explicit_thinking, list):
                            thinking = [
                                dict(block)
                                for block in explicit_thinking
                                if _is_replayable_thinking(block)
                            ]
                        else:
                            thinking = _cached_thinking(tool_call.get("id"))
                    if thinking:
                        blocks.extend(thinking)
                        if text_blocks and not text_placed:
                            blocks.extend(text_blocks)
                            text_placed = True

                    function = tool_call.get("function") or {}
                    raw_args = function.get("arguments", "{}")
                    if isinstance(raw_args, str):
                        try:
                            parsed_args = json.loads(raw_args)
                        except Exception:
                            parsed_args = {}
                    else:
                        parsed_args = raw_args or {}
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tool_call.get("id") or f"toolu_{uuid.uuid4().hex[:12]}",
                            "name": function.get("name", ""),
                            "input": parsed_args,
                        }
                    )

                if text_blocks and not text_placed:
                    blocks = text_blocks + blocks
                if not blocks:
                    blocks.append({"type": "text", "text": ""})

                if converted and converted[-1]["role"] == "assistant":
                    previous = converted[-1]["content"]
                    if isinstance(previous, list):
                        previous.extend(blocks)
                    else:
                        converted[-1]["content"] = [
                            {"type": "text", "text": str(previous)}
                        ] + blocks
                else:
                    converted.append({"role": "assistant", "content": blocks})
                continue

            blocks = self._anthropic_blocks(message.content)
            if converted and converted[-1]["role"] == "user":
                previous = converted[-1]["content"]
                if isinstance(previous, list):
                    previous.extend(blocks)
                else:
                    converted[-1]["content"] = [
                        {"type": "text", "text": str(previous)}
                    ] + blocks
            else:
                converted.append({"role": "user", "content": blocks})

        if not converted:
            converted.append(
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
            )
        return system_blocks, converted

    @staticmethod
    def _uses_adaptive_thinking(model_name: str) -> bool:
        name = (model_name or "").lower()
        family_first = re.search(
            r"(?:claude-)?(opus|sonnet|haiku|fable|mythos)-(\d{1,2})(?:[.-](\d{1,2}))?(?!\d)",
            name,
        )
        version_first = re.search(
            r"claude-(\d{1,2})(?:[.-](\d{1,2}))?-(opus|sonnet|haiku|fable|mythos)",
            name,
        )
        if family_first:
            family = family_first.group(1)
            major = int(family_first.group(2))
            minor = int(family_first.group(3) or 0)
        elif version_first:
            family = version_first.group(3)
            major = int(version_first.group(1))
            minor = int(version_first.group(2) or 0)
        else:
            return False
        if family in ("fable", "mythos"):
            return True
        return major > 4 or (major == 4 and minor >= 6)

    @staticmethod
    def _has_cache_control(payload: Dict[str, Any]) -> bool:
        def scan(value: Any) -> bool:
            if isinstance(value, dict):
                if value.get("cache_control"):
                    return True
                return any(scan(item) for item in value.values())
            if isinstance(value, list):
                return any(scan(item) for item in value)
            return False

        return any(
            scan(payload.get(section)) for section in ("system", "messages", "tools")
        )

    @staticmethod
    def _estimate_payload_tokens(payload: Dict[str, Any]) -> int:
        counted = {
            "system": payload.get("system") or [],
            "messages": payload.get("messages") or [],
            "tools": payload.get("tools") or [],
        }
        serialized = json.dumps(counted, default=str)
        return max(1, len(serialized) // 4)

    def _apply_auto_cache(
        self, payload: Dict[str, Any], variables: Dict[str, Any], model: str
    ) -> None:
        if not _as_bool(variables.get("auto_prompt_cache"), True):
            return
        if self._has_cache_control(payload):
            return
        minimum = 2048 if "haiku" in model.lower() else 1024
        if self._estimate_payload_tokens(payload) < minimum:
            return

        cache_control = {"type": "ephemeral"}
        system = payload.get("system")
        if isinstance(system, list) and system:
            last_block = system[-1]
            if isinstance(last_block, dict) and not last_block.get("cache_control"):
                last_block["cache_control"] = dict(cache_control)

        messages = payload.get("messages") or []
        if messages:
            content = messages[-1].get("content")
            if isinstance(content, str) and content:
                messages[-1]["content"] = [
                    {"type": "text", "text": content, "cache_control": dict(cache_control)}
                ]
            elif isinstance(content, list):
                for block in reversed(content):
                    if (
                        isinstance(block, dict)
                        and block.get("type")
                        in ("text", "tool_result", "tool_use", "image", "document")
                        and not block.get("cache_control")
                    ):
                        block["cache_control"] = dict(cache_control)
                        break

    def _build_anthropic_payload(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        upstream_model: str,
        reasoning: Optional[str],
        variables: Dict[str, Any],
    ) -> Dict[str, Any]:
        extra = request.extra_body if isinstance(request.extra_body, dict) else {}
        max_tokens = (
            request.max_tokens
            or request.max_completion_tokens
            or model_config.max_tokens
            or 4096
        )
        max_tokens = int(max_tokens)

        effort = reasoning if reasoning and reasoning != "none" else ""
        adaptive = self._uses_adaptive_thinking(upstream_model)
        budget: Optional[int] = None
        if effort and not adaptive:
            budget = min(
                ANTHROPIC_THINKING_BUDGETS.get(effort, 4096), max(1024, max_tokens - 1)
            )
            # Anthropic rejects thinking when max_tokens <= budget_tokens.
            if max_tokens <= budget:
                max_tokens = budget + 1024
        thinking_active = bool(effort and (adaptive or budget))

        system_blocks, messages = self._anthropic_messages(
            request.messages, thinking_active
        )
        payload: Dict[str, Any] = {
            "model": upstream_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": bool(request.stream),
        }
        if system_blocks:
            payload["system"] = system_blocks

        if thinking_active:
            if adaptive:
                payload["thinking"] = {
                    "type": "adaptive",
                    "display": "summarized",
                }
                payload["output_config"] = {"effort": effort}
            elif budget is not None:
                payload["thinking"] = {"type": "enabled", "budget_tokens": budget}

        if not thinking_active:
            if request.temperature is not None and request.temperature != 1.0:
                payload["temperature"] = request.temperature
            elif request.top_p is not None and request.top_p != 1.0:
                payload["top_p"] = request.top_p

        tools = self._anthropic_tools(request.tools)
        if tools:
            payload["tools"] = tools
            tool_choice = self._anthropic_tool_choice(request.tool_choice)
            if tool_choice:
                if thinking_active and tool_choice.get("type") in ("any", "tool"):
                    tool_choice = {"type": "auto"}
                payload["tool_choice"] = tool_choice
        if request.stop is not None:
            payload["stop_sequences"] = (
                request.stop if isinstance(request.stop, list) else [request.stop]
            )

        cache_control = extra.get("cache_control")
        if cache_control:
            messages = payload.get("messages") or []
            if messages:
                content = messages[-1].get("content")
                if isinstance(content, list) and content:
                    content[-1].setdefault("cache_control", cache_control)
                elif isinstance(content, str):
                    messages[-1]["content"] = [
                        {
                            "type": "text",
                            "text": content,
                            "cache_control": cache_control,
                        }
                    ]
        else:
            self._apply_auto_cache(payload, variables, upstream_model)
        return payload

    def _anthropic_to_response(
        self, data: Dict[str, Any], model_config: PluginModelConfig
    ) -> ChatCompletionResponse:
        text_parts: List[str] = []
        reasoning_parts: List[str] = []
        pending_thinking: List[Dict[str, Any]] = []
        tool_calls: List[Dict[str, Any]] = []

        for block in data.get("content") or []:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                text_parts.append(str(block.get("text") or ""))
            elif block_type == "thinking":
                text = _reasoning_text(block)
                if text:
                    reasoning_parts.append(text)
                thinking_block = {"type": "thinking", "thinking": text}
                if block.get("signature"):
                    thinking_block["signature"] = block["signature"]
                pending_thinking.append(thinking_block)
            elif block_type == "redacted_thinking":
                if block.get("data"):
                    pending_thinking.append(
                        {"type": "redacted_thinking", "data": block["data"]}
                    )
            elif block_type == "tool_use":
                tool_id = block.get("id") or f"toolu_{uuid.uuid4().hex[:12]}"
                _remember_thinking(tool_id, pending_thinking)
                pending_thinking = []
                tool_calls.append(
                    {
                        "id": tool_id,
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(
                                block.get("input") or {}, ensure_ascii=False
                            ),
                        },
                    }
                )

        raw_stop = data.get("stop_reason")
        if tool_calls or raw_stop == "tool_use":
            finish_reason = "tool_calls"
        elif raw_stop in ("max_tokens", "model_context_window_exceeded"):
            finish_reason = "length"
        elif raw_stop == "refusal":
            finish_reason = "content_filter"
        else:
            finish_reason = "stop"

        message_kwargs: Dict[str, Any] = {
            "role": "assistant",
            "content": "".join(text_parts) if text_parts else (None if tool_calls else ""),
        }
        if tool_calls:
            message_kwargs["tool_calls"] = tool_calls
        if reasoning_parts:
            message_kwargs["reasoning_content"] = "".join(reasoning_parts)

        return ChatCompletionResponse(
            id=f"chatcmpl-{data.get('id') or uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=model_config.id,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(**message_kwargs),
                    finish_reason=finish_reason,
                )
            ],
            usage=self._anthropic_usage(data.get("usage")),
        )

    async def _stream_anthropic(
        self,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        model_config: PluginModelConfig,
        account: Optional[Account],
        proxy: Optional[str],
        timeout: float,
    ) -> AsyncIterator[ChatCompletionChunk]:
        chunk_id = f"chatcmpl-ar-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        current_tool: Optional[Dict[str, Any]] = None
        current_thinking: Optional[Dict[str, Any]] = None
        pending_thinking: List[Dict[str, Any]] = []
        tool_index = 0
        finish_sent = False
        role_sent = False
        last_usage = CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

        def make_chunk(
            delta: Dict[str, Any],
            finish_reason: Optional[str] = None,
            usage: Optional[CompletionUsage] = None,
        ) -> ChatCompletionChunk:
            return ChatCompletionChunk(
                id=chunk_id,
                created=created,
                model=model_config.id,
                choices=[
                    ChatChoiceDelta(index=0, delta=delta, finish_reason=finish_reason)
                ],
                usage=usage,
            )

        async with httpx.AsyncClient(timeout=timeout, proxy=proxy) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                if account is not None:
                    capture_rate_limit_headers(account, response.headers)
                await self._raise_for_status(response)
                role_sent = True
                yield make_chunk({"role": "assistant", "content": ""})

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if not data_str or data_str == "[DONE]" or data_str.lower() == "null":
                        continue
                    try:
                        event = json.loads(data_str)
                    except Exception:
                        continue
                    if not isinstance(event, dict):
                        continue

                    event_type = event.get("type")
                    if event_type == "error":
                        error = event.get("error") or {}
                        message = (
                            error.get("message") if isinstance(error, dict) else str(error)
                        )
                        raise RuntimeError(f"AgentRouter streaming error: {message}")

                    if event_type == "message_start":
                        message = event.get("message") or {}
                        chunk_id = f"chatcmpl-{message.get('id') or uuid.uuid4().hex[:12]}"
                        last_usage = self._anthropic_usage(message.get("usage"))
                    elif event_type == "content_block_start":
                        block = event.get("content_block") or {}
                        block_type = block.get("type")
                        if block_type == "thinking":
                            current_thinking = {
                                "type": "thinking",
                                "thinking": _reasoning_text(block),
                            }
                            if block.get("signature"):
                                current_thinking["signature"] = block["signature"]
                        elif block_type == "redacted_thinking":
                            if block.get("data"):
                                pending_thinking.append(
                                    {"type": "redacted_thinking", "data": block["data"]}
                                )
                        elif block_type == "tool_use":
                            current_tool = {
                                "index": tool_index,
                                "id": block.get("id") or f"toolu_{uuid.uuid4().hex[:8]}",
                                "name": block.get("name", ""),
                            }
                            _remember_thinking(current_tool["id"], pending_thinking)
                            pending_thinking = []
                            yield make_chunk(
                                {
                                    "tool_calls": [
                                        {
                                            "index": tool_index,
                                            "id": current_tool["id"],
                                            "type": "function",
                                            "function": {
                                                "name": current_tool["name"],
                                                "arguments": "",
                                            },
                                        }
                                    ]
                                }
                            )
                            tool_index += 1
                    elif event_type == "content_block_delta":
                        delta = event.get("delta") or {}
                        delta_type = delta.get("type")
                        if delta_type == "text_delta":
                            text = str(delta.get("text") or "")
                            if text:
                                yield make_chunk({"content": text})
                        elif delta_type == "thinking_delta":
                            text = _reasoning_text(delta)
                            if current_thinking is not None and text:
                                current_thinking["thinking"] = (
                                    current_thinking.get("thinking", "") + text
                                )
                            if text:
                                yield make_chunk({"reasoning_content": text})
                        elif delta_type == "signature_delta" and current_thinking is not None:
                            current_thinking["signature"] = delta.get("signature", "")
                        elif delta_type == "input_json_delta" and current_tool is not None:
                            partial = str(delta.get("partial_json") or "")
                            if partial:
                                yield make_chunk(
                                    {
                                        "tool_calls": [
                                            {
                                                "index": current_tool["index"],
                                                "function": {"arguments": partial},
                                            }
                                        ]
                                    }
                                )
                    elif event_type == "content_block_stop":
                        if current_thinking is not None:
                            pending_thinking.append(current_thinking)
                            current_thinking = None
                        current_tool = None
                    elif event_type == "message_delta":
                        raw_stop = (event.get("delta") or {}).get("stop_reason")
                        if raw_stop == "tool_use":
                            finish_reason = "tool_calls"
                        elif raw_stop in ("max_tokens", "model_context_window_exceeded"):
                            finish_reason = "length"
                        elif raw_stop == "refusal":
                            finish_reason = "content_filter"
                        elif raw_stop in ("end_turn", "stop_sequence"):
                            finish_reason = "stop"
                        else:
                            finish_reason = raw_stop or "stop"
                        if event.get("usage"):
                            last_usage = self._anthropic_usage(
                                event.get("usage"), previous=last_usage
                            )
                        yield make_chunk(
                            {}, finish_reason=finish_reason, usage=last_usage
                        )
                        finish_sent = True
                    elif event_type == "message_stop" and not finish_sent:
                        yield make_chunk({}, finish_reason="stop", usage=last_usage)
                        finish_sent = True

        if not finish_sent:
            yield make_chunk({}, finish_reason="stop", usage=last_usage)

    # ------------------------------------------------------------------
    # Simulation (offline / synthetic default account only)
    # ------------------------------------------------------------------

    def _mock_reasoning(
        self, reasoning: Optional[str], model_config: PluginModelConfig
    ) -> Optional[str]:
        if not reasoning or reasoning == "none":
            return None
        return (
            f"Simulated reasoning for {model_config.id} at {reasoning} effort: "
            "no further tool call is required, so the answer can be produced now."
        )

    def _mock_tool(
        self, request: ChatCompletionRequest
    ) -> Optional[Tuple[str, str, str]]:
        if not request.tools:
            return None
        if any(message.role == "tool" for message in request.messages[-2:]):
            return None
        tool = request.tools[0]
        function = tool.get("function", {}) if isinstance(tool, dict) else {}
        name = function.get("name") or (tool.get("name", "") if isinstance(tool, dict) else "")
        if not name:
            return None
        arguments = "{}"
        properties = (function.get("parameters") or {}).get("properties") or {}
        if properties:
            first_key = next(iter(properties))
            arguments = json.dumps({first_key: "test_value"})
        return f"call_ar_{uuid.uuid4().hex[:8]}", name, arguments

    def _mock_response(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        variables: Dict[str, Any],
        account: Optional[Account],
    ) -> ChatCompletionResponse:
        _, reasoning = self._resolve_reasoning(request, model_config, variables)
        account_label = f" via account '{account.name}'" if account else ""
        mock_tool = self._mock_tool(request)
        if mock_tool:
            call_id, name, arguments = mock_tool
            message_kwargs: Dict[str, Any] = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }
                ],
            }
            reasoning_text = self._mock_reasoning(reasoning, model_config)
            if reasoning_text:
                message_kwargs["reasoning_content"] = reasoning_text
            finish_reason = "tool_calls"
        else:
            content = (
                f"Simulated response from AgentRouter ({model_config.id}){account_label}. "
                "Configure a real API key to reach agentrouter.org."
            )
            message_kwargs = {"role": "assistant", "content": content}
            reasoning_text = self._mock_reasoning(reasoning, model_config)
            if reasoning_text:
                message_kwargs["reasoning_content"] = reasoning_text
            finish_reason = "stop"

        return ChatCompletionResponse(
            id=f"chatcmpl-ar-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=model_config.id,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(**message_kwargs),
                    finish_reason=finish_reason,
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=18,
                completion_tokens=26,
                total_tokens=44,
                prompt_tokens_details={"cached_tokens": 0},
            ),
        )

    async def _mock_stream(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        variables: Dict[str, Any],
        account: Optional[Account],
    ) -> AsyncIterator[ChatCompletionChunk]:
        chunk_id = f"chatcmpl-ar-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        _, reasoning = self._resolve_reasoning(request, model_config, variables)
        account_label = f" via account '{account.name}'" if account else ""

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

        reasoning_text = self._mock_reasoning(reasoning, model_config)
        if reasoning_text:
            yield ChatCompletionChunk(
                id=chunk_id,
                created=created,
                model=model_config.id,
                choices=[
                    ChatChoiceDelta(
                        index=0,
                        delta={"reasoning_content": reasoning_text},
                        finish_reason=None,
                    )
                ],
            )

        mock_tool = self._mock_tool(request)
        if mock_tool:
            call_id, name, arguments = mock_tool
            yield ChatCompletionChunk(
                id=chunk_id,
                created=created,
                model=model_config.id,
                choices=[
                    ChatChoiceDelta(
                        index=0,
                        delta={
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": call_id,
                                    "type": "function",
                                    "function": {"name": name, "arguments": ""},
                                }
                            ]
                        },
                        finish_reason=None,
                    )
                ],
            )
            midpoint = max(1, len(arguments) // 2)
            for piece in (arguments[:midpoint], arguments[midpoint:]):
                if not piece:
                    continue
                yield ChatCompletionChunk(
                    id=chunk_id,
                    created=created,
                    model=model_config.id,
                    choices=[
                        ChatChoiceDelta(
                            index=0,
                            delta={
                                "tool_calls": [
                                    {"index": 0, "function": {"arguments": piece}}
                                ]
                            },
                            finish_reason=None,
                        )
                    ],
                )
            yield ChatCompletionChunk(
                id=chunk_id,
                created=created,
                model=model_config.id,
                choices=[
                    ChatChoiceDelta(index=0, delta={}, finish_reason="tool_calls")
                ],
                usage=CompletionUsage(prompt_tokens=18, completion_tokens=26, total_tokens=44),
            )
            return

        now = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        tokens = [
            "Simulated ",
            "stream ",
            "response ",
            f"from AgentRouter for {model_config.id}{account_label}.",
            "\n\n---\n**Execution Details:**\n",
            f"• **Timestamp:** {now}\n",
            f"• **Model:** {model_config.id}\n",
            "• **Account status:** Simulation Mode",
        ]
        for token in tokens:
            yield ChatCompletionChunk(
                id=chunk_id,
                created=created,
                model=model_config.id,
                choices=[
                    ChatChoiceDelta(index=0, delta={"content": token}, finish_reason=None)
                ],
            )

        yield ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=model_config.id,
            choices=[ChatChoiceDelta(index=0, delta={}, finish_reason="stop")],
            usage=CompletionUsage(prompt_tokens=18, completion_tokens=26, total_tokens=44),
        )

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        variables: Dict[str, Any],
        account: Optional[Account] = None,
    ) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionChunk]]:
        api_key = self._api_key(account)
        if self._is_test_mode(api_key, account):
            if request.stream:
                return self._mock_stream(request, model_config, variables, account)
            return self._mock_response(request, model_config, variables, account)

        protocol = self._resolve_protocol(request, model_config, variables, account)
        upstream_model, reasoning = self._resolve_reasoning(
            request, model_config, variables
        )
        base_url = self._root_url(
            (account.base_url if account and account.base_url else "")
            or str(variables.get("base_url") or DEFAULT_BASE_URL)
        )
        headers = self._headers(protocol, api_key, variables, account)
        proxy = self._proxy(variables, account)
        timeout = self._timeout(variables)

        if protocol == PROTOCOL_RESPONSES:
            url = f"{base_url}/v1/responses"
            payload = self._build_responses_payload(
                request,
                upstream_model,
                reasoning,
                variables,
                replay_reasoning=_as_bool(variables.get("replay_reasoning"), False),
            )
            if request.stream:
                return self._stream_responses(
                    url, headers, payload, model_config, account, proxy, timeout
                )
            async with httpx.AsyncClient(timeout=timeout, proxy=proxy) as client:
                response = await client.post(url, headers=headers, json=payload)
                if account is not None:
                    capture_rate_limit_headers(account, response.headers)
                await self._raise_for_status(response)
                data = response.json()
            if not isinstance(data, dict):
                raise RuntimeError("AgentRouter returned an unexpected Responses payload")
            return self._responses_to_response(data, model_config)

        if protocol == PROTOCOL_ANTHROPIC:
            url = f"{base_url}/v1/messages"
            payload = self._build_anthropic_payload(
                request, model_config, upstream_model, reasoning, variables
            )
            if request.stream:
                return self._stream_anthropic(
                    url, headers, payload, model_config, account, proxy, timeout
                )
            async with httpx.AsyncClient(timeout=timeout, proxy=proxy) as client:
                response = await client.post(url, headers=headers, json=payload)
                if account is not None:
                    capture_rate_limit_headers(account, response.headers)
                await self._raise_for_status(response)
                data = response.json()
            if not isinstance(data, dict):
                raise RuntimeError("AgentRouter returned an unexpected Messages payload")
            return self._anthropic_to_response(data, model_config)

        url = f"{base_url}/v1/chat/completions"
        payload = self._build_chat_payload(request, upstream_model, reasoning, variables)
        if request.stream:
            return self._stream_chat(
                url, headers, payload, model_config, account, proxy, timeout
            )
        async with httpx.AsyncClient(timeout=timeout, proxy=proxy) as client:
            response = await client.post(url, headers=headers, json=payload)
            if account is not None:
                capture_rate_limit_headers(account, response.headers)
            await self._raise_for_status(response)
            data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError("AgentRouter returned an unexpected Chat payload")
        if not data.get("id"):
            data["id"] = f"chatcmpl-ar-{uuid.uuid4().hex[:12]}"
        data.setdefault("model", model_config.id)
        return ChatCompletionResponse.model_validate(data)


def model_config_max_tokens(request: ChatCompletionRequest) -> Optional[int]:
    """Kept as a helper so the Anthropic payload builder stays readable."""
    return None


plugin = AgentRouterPlugin()
