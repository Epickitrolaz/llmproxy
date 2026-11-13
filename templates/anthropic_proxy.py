"""Anthropic Messages API provider plugin for LLMProxy.

This plugin exposes Anthropic's native Messages API through the OpenAI-compatible
``/v1/chat/completions`` endpoint. It supports streaming, tool calls, usage
reporting, and extended thinking controlled by a reasoning effort level.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from collections import OrderedDict
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

# Claude 4.6 and newer (including the 5.x models below) use adaptive thinking
# with an ``output_config.effort`` level. Older models require an explicit
# thinking budget. The values below are deliberately conservative defaults.
REASONING_BUDGETS: Dict[str, int] = {
    "low": 1024,
    "medium": 4096,
    "high": 16384,
    "xhigh": 32768,
    "max": 60000,
}

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

# Levels accepted through ``reasoning_effort``. Models are listed once under
# their base ID; the proxy maps the level to the upstream thinking controls.
REASONING_LEVELS: Tuple[str, ...] = ("none", "low", "medium", "high", "xhigh", "max")

# Anthropic requires the signed ``thinking`` / ``redacted_thinking`` blocks that
# preceded a ``tool_use`` block to be replayed byte-for-byte when the tool
# result is sent back. OpenAI-compatible clients only see ``reasoning_content``,
# so the raw blocks are kept here, keyed by tool-use ID, for the duration of
# the tool loop. The cache is intentionally small and in-memory: tool loops are
# short-lived and signatures are only valid for the current conversation.
_THINKING_BLOCKS: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
_THINKING_BLOCKS_LIMIT = 512

# Anthropic reports a safety-classifier fallback in two places: the top-level
# ``model`` field names the model that actually answered, and ``usage.iterations``
# records the hop as a ``fallback_message`` entry. Sticky routing sends later
# turns straight to the fallback model, so the ``fallback`` content block is not
# always present and must not be relied on. Detection keys on the model field,
# using ``usage.iterations`` only to reject unrelated alias/date differences.
FALLBACK_NOTICE_TEMPLATE = "The model has been downgraded to {model}"
FALLBACK_BETA_HEADER = "server-side-fallback-2026-07-01"


def _normalize_model_id(value: Any) -> str:
    """Normalize a model ID for comparison (case and dated snapshots)."""
    name = str(value or "").strip().lower()
    if not name:
        return ""
    return re.sub(r"-\d{8}$", "", name)


def remember_thinking_blocks(
    tool_use_id: Optional[str], blocks: Optional[List[Dict[str, Any]]]
) -> None:
    """Cache signed thinking blocks so a later tool result can replay them."""
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
    """Return True when a thinking block carries the data Anthropic requires."""
    if not isinstance(block, dict):
        return False
    if block.get("type") == "redacted_thinking":
        return bool(block.get("data"))
    return block.get("type") == "thinking" and bool(block.get("signature"))


def cached_thinking_blocks(tool_use_id: Optional[str]) -> List[Dict[str, Any]]:
    """Return a copy of the signed thinking blocks stored for a tool call."""
    if not tool_use_id:
        return []
    blocks = _THINKING_BLOCKS.get(tool_use_id)
    if not blocks:
        return []
    _THINKING_BLOCKS.move_to_end(tool_use_id)
    return [dict(block) for block in blocks]


class AnthropicProxyPlugin(BasePlugin):
    """Anthropic Messages API provider with an OpenAI-compatible surface."""

    def __init__(self) -> None:
        self._metadata = PluginMetadata(
            id="anthropic_proxy",
            name="Anthropic (Messages API)",
            description=(
                "Anthropic Claude Messages API provider with streaming, tool calls, "
                "and extended thinking support"
            ),
            version="2.2.0",
            author="Anthropic",
            enabled=True,
            filename="anthropic_proxy.py",
            variables=[
                PluginVariable(
                    name="api_key",
                    label="Anthropic API Key",
                    type="secret",
                    default="",
                    description="Anthropic console API key (accounts can also supply this)",
                ),
                PluginVariable(
                    name="base_url",
                    label="Base URL",
                    type="string",
                    default="https://api.anthropic.com/v1",
                    description="Anthropic Messages API base URL",
                ),
                PluginVariable(
                    name="anthropic_version",
                    label="Anthropic Version",
                    type="string",
                    default="2023-06-01",
                    description="Value sent in the anthropic-version header",
                ),
                PluginVariable(
                    name="timeout",
                    label="Timeout (seconds)",
                    type="number",
                    default=60,
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
                    default=1,
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
                    default="medium",
                    options=["none", "low", "medium", "high", "xhigh", "max"],
                    description=(
                        "Default extended-thinking level. Request-level reasoning_effort "
                        "or a model suffix such as ':high' takes precedence."
                    ),
                ),
                PluginVariable(
                    name="thinking_display",
                    label="Thinking Display",
                    type="select",
                    default="summarized",
                    options=["summarized", "omitted"],
                    description=(
                        "Anthropic hides reasoning text by default on current models. "
                        "'summarized' asks for readable reasoning summaries in "
                        "reasoning_content; 'omitted' keeps the signature only."
                    ),
                ),
                PluginVariable(
                    name="fallbacks",
                    label="Safety Fallbacks",
                    type="select",
                    default="off",
                    options=["off", "default"],
                    description=(
                        "Opt into Anthropic server-side safety fallbacks. 'default' "
                        "retries a declined request on Anthropic's recommended model; "
                        "downgrades are reported to the client."
                    ),
                ),
                PluginVariable(
                    name="auto_prompt_cache",
                    label="Automatic Prompt Caching",
                    type="boolean",
                    default=True,
                    description=(
                        "Add Anthropic cache breakpoints to the system prompt and the last "
                        "message when the client sends none. Skipped below the model's "
                        "minimum cacheable prefix (1024 tokens, 2048 for Haiku)."
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
                        "Strategy for selecting proxies from the pool: sequential, random, "
                        "specific, or switch_on_limit"
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
            models=self._build_models(),
        )

    @staticmethod
    def _build_models() -> List[PluginModelConfig]:
        reasoning_levels = list(REASONING_LEVELS)
        return [
            PluginModelConfig(
                id="claude-haiku-5-5",
                name="Claude Haiku 5.5",
                upstream_model="claude-haiku-5-5",
                context_window=200000,
                max_tokens=64000,
                reasoning_levels=reasoning_levels,
            ),
            PluginModelConfig(
                id="claude-sonnet-5-5",
                name="Claude Sonnet 5.5",
                upstream_model="claude-sonnet-5-5",
                context_window=200000,
                max_tokens=64000,
                reasoning_levels=reasoning_levels,
            ),
            PluginModelConfig(
                id="claude-opus-5-5",
                name="Claude Opus 5.5",
                upstream_model="claude-opus-5-5",
                context_window=200000,
                max_tokens=64000,
                reasoning_levels=reasoning_levels,
            ),
            PluginModelConfig(
                id="claude-fable-5-2",
                name="Claude Fable 5.2",
                upstream_model="claude-fable-5-2",
                context_window=1000000,
                max_tokens=64000,
                reasoning_levels=reasoning_levels,
            ),
        ]

    @property
    def metadata(self) -> PluginMetadata:
        return self._metadata

    # ------------------------------------------------------------------
    # Reasoning helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_effort(value: Any) -> Optional[str]:
        if value is None:
            return None
        raw = str(value).strip().lower()
        if not raw or raw in ("default", "auto"):
            return None
        if raw in ("none", "off", "disabled", "false", "0"):
            return "none"
        if raw in ("low", "medium", "high", "xhigh", "max"):
            return raw
        return None

    @staticmethod
    def _normalize_thinking_display(value: Any) -> Optional[str]:
        raw = str(value or "").strip().lower()
        if raw in ("summarized", "summary", "summaries"):
            return "summarized"
        if raw in ("omitted", "omit", "hidden", "off"):
            return "omitted"
        return None

    def _resolve_thinking_display(
        self,
        request: ChatCompletionRequest,
        variables: Dict[str, Any],
    ) -> str:
        """Return the ``thinking.display`` value to send upstream.

        Current Anthropic models default to ``"omitted"``, which returns empty
        thinking text. The proxy asks for summaries so reasoning reaches clients
        as ``reasoning_content`` unless the client or provider opts out.
        """
        extra = request.extra_body if isinstance(request.extra_body, dict) else {}
        requested: Any = None
        thinking = extra.get("thinking")
        if isinstance(thinking, dict):
            requested = thinking.get("display")
        if requested is None:
            requested = extra.get("thinking_display")
        display = self._normalize_thinking_display(requested)
        if display is None:
            display = self._normalize_thinking_display(variables.get("thinking_display"))
        return display or "summarized"

    @staticmethod
    def _thinking_text(block: Dict[str, Any]) -> str:
        """Extract readable reasoning from a thinking block or delta.

        Summarized thinking normally arrives in ``thinking``. Some gateways move
        the summary to ``summaries`` (a list of ``summary_text`` items) or a
        plain ``summary`` string, so accept those shapes too instead of dropping
        the reasoning entirely.
        """
        if not isinstance(block, dict):
            return ""
        text = block.get("thinking")
        if isinstance(text, str) and text:
            return text
        for key in ("summaries", "summary"):
            value = block.get(key)
            parts: List[str] = []
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict):
                        part = (
                            item.get("text")
                            or item.get("summary_text")
                            or item.get("thinking")
                        )
                        if isinstance(part, str):
                            parts.append(part)
            joined = "".join(parts)
            if joined:
                return joined
        return text if isinstance(text, str) else ""

    @staticmethod
    def _served_model(payload: Dict[str, Any]) -> Optional[str]:
        """Return the model that actually produced a Messages API response."""
        if not isinstance(payload, dict):
            return None
        model = payload.get("model")
        return model.strip() if isinstance(model, str) and model.strip() else None

    @staticmethod
    def _fallback_evidence(payload: Dict[str, Any]) -> bool:
        """Return True when the response records a server-side fallback hop."""
        content = payload.get("content")
        if isinstance(content, list):
            if any(
                isinstance(block, dict) and block.get("type") == "fallback"
                for block in content
            ):
                return True
        usage = payload.get("usage")
        if isinstance(usage, dict):
            iterations = usage.get("iterations")
            if isinstance(iterations, list):
                return any(
                    isinstance(attempt, dict)
                    and attempt.get("type") == "fallback_message"
                    for attempt in iterations
                )
        return False

    def _downgrade_notice(
        self,
        requested_model: str,
        payload: Dict[str, Any],
    ) -> Optional[str]:
        """Return a user-facing notice when Anthropic served a fallback model."""
        served = self._served_model(payload)
        if not served:
            return None
        if _normalize_model_id(served) == _normalize_model_id(requested_model):
            return None

        # When fallback metadata is present, require it to agree. This rejects
        # gateways that merely normalize a dated snapshot; sticky fallback turns
        # still record a ``fallback_message`` iteration, so they are caught even
        # though they omit the ``fallback`` content block.
        usage = payload.get("usage")
        has_metadata = bool(
            isinstance(usage, dict) and usage.get("iterations")
        ) or any(
            isinstance(block, dict) and block.get("type") == "fallback"
            for block in (payload.get("content") or [])
        )
        if has_metadata and not self._fallback_evidence(payload):
            return None
        return FALLBACK_NOTICE_TEMPLATE.format(model=served)

    @staticmethod
    def _resolve_fallbacks(
        request: ChatCompletionRequest,
        variables: Dict[str, Any],
    ) -> Optional[Any]:
        """Return the ``fallbacks`` request value, if any."""
        extra = request.extra_body if isinstance(request.extra_body, dict) else {}
        if "fallbacks" in extra:
            return extra.get("fallbacks")
        value = variables.get("fallbacks")
        if isinstance(value, str) and value.strip().lower() == "default":
            return "default"
        if isinstance(value, list) and value:
            return value
        return None

    @staticmethod
    def _beta_header(
        request: ChatCompletionRequest,
        fallbacks: Optional[Any],
    ) -> Optional[str]:
        """Return the ``anthropic-beta`` header value for the request."""
        extra = request.extra_body if isinstance(request.extra_body, dict) else {}
        raw = extra.get("anthropic_beta")
        values: List[str] = []
        if isinstance(raw, str):
            values = [item.strip() for item in raw.split(",") if item.strip()]
        elif isinstance(raw, list):
            values = [str(item).strip() for item in raw if str(item).strip()]
        if fallbacks and FALLBACK_BETA_HEADER not in values:
            values.append(FALLBACK_BETA_HEADER)
        return ",".join(values) or None


    def _resolve_reasoning(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        variables: Dict[str, Any],
    ) -> Tuple[str, str]:
        """Return ``(target_model, effort)`` with a fully normalized effort."""
        target_model = model_config.upstream_model or model_config.id

        raw_effort: Optional[str] = None
        if request.reasoning_effort:
            raw_effort = request.reasoning_effort
        elif request.extra_body:
            raw_effort = request.extra_body.get("reasoning_effort")
            if raw_effort is None:
                reasoning = request.extra_body.get("reasoning")
                if isinstance(reasoning, dict):
                    raw_effort = reasoning.get("effort") or reasoning.get(
                        "reasoning_effort"
                    )
                elif reasoning:
                    raw_effort = reasoning

        # A suffix on the requested model wins over the provider default.
        requested_model = (request.model or "").strip().lower()
        for suffix, level in REASONING_SUFFIXES:
            if requested_model.endswith(suffix):
                if raw_effort is None:
                    raw_effort = level
                break

        effort = self._normalize_effort(raw_effort)
        if effort is None and model_config.parameters:
            effort = self._normalize_effort(
                model_config.parameters.get("reasoning_effort")
            )
        if effort is None:
            effort = self._normalize_effort(variables.get("reasoning_effort")) or "none"

        # Never forward a reasoning routing suffix to the upstream API.
        for suffix, _ in REASONING_SUFFIXES:
            if target_model.lower().endswith(suffix):
                target_model = target_model[: -len(suffix)]
                break

        return target_model, effort

    @staticmethod
    def _thinking_budget(effort: str, max_tokens: int) -> Optional[int]:
        if effort in ("none", "default", ""):
            return None
        desired = REASONING_BUDGETS.get(effort)
        if not desired:
            return None
        # Anthropic requires budget_tokens >= 1024 and < max_tokens.
        upper_bound = int(max_tokens) - 1024
        if upper_bound < 1024:
            return None
        return min(desired, upper_bound)

    @staticmethod
    def _uses_adaptive_thinking(model_name: str) -> bool:
        """Return True when the model expects adaptive thinking instead of a budget."""
        name = (model_name or "").lower()
        family_first = re.search(
            r"(?:claude-)?(opus|sonnet|haiku|fable|mythos)-(\d{1,2})(?:[.-](\d{1,2}))?(?!\d)",
            name,
        )
        version_first = re.search(
            r"claude-(\d{1,2})(?:[.-](\d{1,2}))?-(opus|sonnet|haiku|fable|mythos)", name
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
        # Claude 4.6+ and the 5.x family use adaptive thinking with an effort.
        return major > 4 or (major == 4 and minor >= 6)

    @staticmethod
    def _is_test_mode(api_key: str) -> bool:
        if not api_key:
            return True
        key = api_key.lower().strip()
        prefixes = ("test", "mock", "sk-test", "dummy", "fake", "offline", "sample")
        return any(key.startswith(p) for p in prefixes) or key in (
            "test-key",
            "test_key",
            "secret123",
        )

    # ------------------------------------------------------------------
    # Anthropic request/response translation
    # ------------------------------------------------------------------

    def _format_anthropic_tools(
        self, tools: Optional[List[Dict[str, Any]]]
    ) -> Optional[List[Dict[str, Any]]]:
        if not tools:
            return None

        anthropic_tools: List[Dict[str, Any]] = []
        for tool in tools:
            cache_control = None
            if "function" in tool:
                fn = tool["function"]
                name = fn.get("name", "")
                description = fn.get("description", "")
                parameters = fn.get("parameters", {"type": "object", "properties": {}})
                cache_control = tool.get("cache_control") or fn.get("cache_control")
            else:
                name = tool.get("name", "")
                description = tool.get("description", "")
                parameters = tool.get("input_schema") or tool.get(
                    "parameters", {"type": "object", "properties": {}}
                )
                cache_control = tool.get("cache_control")

            if not isinstance(parameters, dict) or parameters.get("type") != "object":
                parameters = {"type": "object", "properties": {}}

            anthropic_tool: Dict[str, Any] = {
                "name": name,
                "description": description,
                "input_schema": parameters,
            }
            if cache_control:
                anthropic_tool["cache_control"] = cache_control
            anthropic_tools.append(anthropic_tool)

        return anthropic_tools or None

    def _format_anthropic_tool_choice(
        self, tool_choice: Optional[Union[str, Dict[str, Any]]]
    ) -> Optional[Dict[str, Any]]:
        if not tool_choice or tool_choice == "auto":
            return {"type": "auto"}
        if tool_choice == "required":
            return {"type": "any"}
        if tool_choice == "none":
            return None
        if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            fn_name = tool_choice.get("function", {}).get("name")
            if fn_name:
                return {"type": "tool", "name": fn_name}
        return None

    def _build_anthropic_messages(
        self,
        messages: List[ChatMessage],
        thinking_active: bool = False,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        system_blocks: List[Dict[str, Any]] = []
        anthropic_messages: List[Dict[str, Any]] = []

        def _text_blocks(content: Any) -> List[Dict[str, Any]]:
            """Normalize content into Anthropic blocks while preserving cache_control."""
            if isinstance(content, str):
                return [{"type": "text", "text": content}]
            blocks: List[Dict[str, Any]] = []
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, str):
                        blocks.append({"type": "text", "text": part})
                    elif isinstance(part, dict):
                        blocks.append(dict(part))
            return blocks

        for message in messages:
            if message.role == "system":
                system_blocks.extend(_text_blocks(message.content))
                continue

            if message.role in ("tool", "function"):
                tool_use_id = message.tool_call_id or getattr(message, "id", None) or "call_unknown"
                if isinstance(message.content, list):
                    tool_content: Any = _text_blocks(message.content)
                else:
                    tool_content = str(message.content) if message.content is not None else ""
                tool_block = {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": tool_content,
                }
                if anthropic_messages and anthropic_messages[-1]["role"] == "user":
                    previous = anthropic_messages[-1]["content"]
                    if isinstance(previous, list):
                        previous.append(tool_block)
                    else:
                        anthropic_messages[-1]["content"] = [
                            {"type": "text", "text": str(previous)},
                            tool_block,
                        ]
                else:
                    anthropic_messages.append({"role": "user", "content": [tool_block]})
                continue

            if message.role == "assistant":
                text_blocks: List[Dict[str, Any]] = (
                    _text_blocks(message.content) if message.content else []
                )
                explicit_thinking = getattr(message, "thinking_blocks", None)

                blocks: List[Dict[str, Any]] = []
                text_placed = False
                for index, tool_call in enumerate(message.tool_calls or []):
                    thinking: List[Dict[str, Any]] = []
                    if thinking_active:
                        if index == 0 and isinstance(explicit_thinking, list):
                            # A client that round-trips the raw blocks wins over
                            # the proxy-side cache.
                            thinking = [
                                dict(block)
                                for block in explicit_thinking
                                if _is_replayable_thinking(block)
                            ]
                        else:
                            thinking = cached_thinking_blocks(tool_call.get("id"))

                    if thinking:
                        if text_blocks and not text_placed:
                            blocks.extend(thinking)
                            blocks.extend(text_blocks)
                            text_placed = True
                        else:
                            blocks.extend(thinking)

                    raw_args = tool_call.get("function", {}).get("arguments", "{}")
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
                            "name": tool_call.get("function", {}).get("name", ""),
                            "input": parsed_args,
                        }
                    )

                if text_blocks and not text_placed:
                    blocks = text_blocks + blocks

                if not blocks:
                    blocks.append({"type": "text", "text": ""})

                if anthropic_messages and anthropic_messages[-1]["role"] == "assistant":
                    previous = anthropic_messages[-1]["content"]
                    if isinstance(previous, list):
                        previous.extend(blocks)
                    else:
                        anthropic_messages[-1]["content"] = [
                            {"type": "text", "text": str(previous)}
                        ] + blocks
                else:
                    anthropic_messages.append({"role": "assistant", "content": blocks})
                continue

            # role == "user"
            blocks = _text_blocks(message.content)
            if anthropic_messages and anthropic_messages[-1]["role"] == "user":
                previous = anthropic_messages[-1]["content"]
                if isinstance(previous, list):
                    previous.extend(blocks)
                else:
                    anthropic_messages[-1]["content"] = [
                        {"type": "text", "text": str(previous)}
                    ] + blocks
            else:
                anthropic_messages.append({"role": "user", "content": blocks})

        if not anthropic_messages:
            anthropic_messages.append({"role": "user", "content": [{"type": "text", "text": "Hello"}]})

        return system_blocks, anthropic_messages

    @staticmethod
    def _apply_cache_control(payload: Dict[str, Any], request: ChatCompletionRequest) -> bool:
        """Apply a top-level cache_control hint to the final message block.

        Anthropic places cache breakpoints on content blocks. Most clients send
        an explicit block-level ``cache_control``, which is preserved as-is. This
        helper additionally accepts the common convenience form
        ``{"cache_control": {"type": "ephemeral"}}`` in ``extra_body``.

        Returns True when an explicit hint was applied.
        """
        extra = request.extra_body if isinstance(request.extra_body, dict) else {}
        cache_control = getattr(request, "cache_control", None) or extra.get("cache_control")
        if not cache_control:
            return False

        messages = payload.get("messages") or []
        if not messages:
            return False
        content = messages[-1].get("content")
        if isinstance(content, list) and content:
            block = content[-1]
            if isinstance(block, dict):
                block.setdefault("cache_control", cache_control)
        elif isinstance(content, str):
            messages[-1]["content"] = [
                {"type": "text", "text": content, "cache_control": cache_control}
            ]
        return True

    @staticmethod
    def _as_bool(value: Any, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "on", "enabled")

    @staticmethod
    def _has_cache_control(payload: Dict[str, Any]) -> bool:
        """Return True when the payload already carries any cache breakpoint."""

        def scan(value: Any) -> bool:
            if isinstance(value, dict):
                if value.get("cache_control"):
                    return True
                return any(scan(item) for item in value.values())
            if isinstance(value, list):
                return any(scan(item) for item in value)
            return False

        return any(
            scan(payload.get(section))
            for section in ("system", "messages", "tools")
        )

    @staticmethod
    def _estimate_payload_tokens(payload: Dict[str, Any]) -> int:
        """Cheap approximation used only to skip prompts Anthropic cannot cache."""
        counted = {
            "system": payload.get("system") or [],
            "messages": payload.get("messages") or [],
            "tools": payload.get("tools") or [],
        }
        try:
            serialized = json.dumps(counted, default=str)
        except Exception:
            serialized = str(counted)
        return max(1, len(serialized) // 4)

    @staticmethod
    def _auto_cache_min_tokens(target_model: str) -> int:
        # Anthropic's minimum cacheable prefix is 1024 tokens for Opus/Sonnet
        # and 2048 tokens for Haiku.
        return 2048 if "haiku" in (target_model or "").lower() else 1024

    def _apply_auto_cache_control(
        self,
        payload: Dict[str, Any],
        variables: Dict[str, Any],
        target_model: str,
    ) -> None:
        """Add cache breakpoints for clients that send no cache_control hints.

        Places at most two ephemeral breakpoints: one at the end of the system
        prompt and one at the end of the final message. Requests below
        Anthropic's minimum cacheable size are left untouched because a cache
        write would cost 1.25x without ever being read.
        """
        if not self._as_bool(variables.get("auto_prompt_cache"), True):
            return
        if self._has_cache_control(payload):
            return
        if self._estimate_payload_tokens(payload) < self._auto_cache_min_tokens(target_model):
            return

        cache_control = {"type": "ephemeral"}
        added = 0

        system = payload.get("system")
        if isinstance(system, list) and system:
            last_block = system[-1]
            if isinstance(last_block, dict) and not last_block.get("cache_control"):
                last_block["cache_control"] = dict(cache_control)
                added += 1

        messages = payload.get("messages") or []
        if messages and added < 4:
            last_message = messages[-1]
            content = last_message.get("content")
            if isinstance(content, str) and content:
                last_message["content"] = [
                    {
                        "type": "text",
                        "text": content,
                        "cache_control": dict(cache_control),
                    }
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

    def _build_payload(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        variables: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build the native Anthropic Messages payload (also used by tests)."""
        target_model, effort = self._resolve_reasoning(request, model_config, variables)
        max_tokens = (
            request.max_tokens
            or request.max_completion_tokens
            or model_config.max_tokens
            or 4096
        )
        max_tokens = int(max_tokens)
        effort_active = bool(effort not in ("none", "default", ""))
        adaptive = self._uses_adaptive_thinking(target_model)
        budget = None if adaptive else self._thinking_budget(effort, max_tokens)
        thinking_active = bool(effort_active and (adaptive or budget is not None))

        system_blocks, anthropic_messages = self._build_anthropic_messages(
            request.messages, thinking_active=thinking_active
        )
        payload: Dict[str, Any] = {
            "model": target_model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
            "stream": bool(request.stream),
        }

        if system_blocks:
            # An array of content blocks is required for system-level cache_control.
            payload["system"] = system_blocks

        if effort_active:
            if adaptive:
                payload["thinking"] = {
                    "type": "adaptive",
                    "display": self._resolve_thinking_display(request, variables),
                }
                payload["output_config"] = {"effort": effort}
            elif budget is not None:
                payload["thinking"] = {"type": "enabled", "budget_tokens": budget}

        fallbacks = self._resolve_fallbacks(request, variables)
        if fallbacks:
            payload["fallbacks"] = fallbacks

        # Anthropic rejects temperature/top_p when extended thinking is active.
        if not thinking_active:
            if request.temperature is not None and request.temperature != 1.0:
                payload["temperature"] = request.temperature
            elif request.top_p is not None and request.top_p != 1.0:
                payload["top_p"] = request.top_p

        anthropic_tools = self._format_anthropic_tools(request.tools)
        if anthropic_tools:
            payload["tools"] = anthropic_tools
            tool_choice = self._format_anthropic_tool_choice(request.tool_choice)
            if tool_choice:
                # Anthropic only supports auto/none tool choice while thinking.
                if thinking_active and tool_choice.get("type") in ("any", "tool"):
                    tool_choice = {"type": "auto"}
                payload["tool_choice"] = tool_choice

        explicit_cache = self._apply_cache_control(payload, request)
        if not explicit_cache:
            self._apply_auto_cache_control(payload, variables, target_model)
        return payload

    # ------------------------------------------------------------------
    # Simulation mode
    # ------------------------------------------------------------------

    def _mock_reasoning(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        variables: Optional[Dict[str, Any]],
        tool_name: Optional[str] = None,
    ) -> Optional[str]:
        """Return simulated reasoning text when thinking is configured on."""
        _, effort = self._resolve_reasoning(request, model_config, variables or {})
        if effort in ("none", "default", ""):
            return None
        model_name = model_config.id
        if tool_name:
            return (
                f"Simulated reasoning for {model_name} at {effort} effort: "
                f"the request needs the {tool_name} tool before answering."
            )
        return (
            f"Simulated reasoning for {model_name} at {effort} effort: "
            "no further tool call is required, so the answer can be produced now."
        )

    def _mock_response(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        account: Optional[Account] = None,
        variables: Optional[Dict[str, Any]] = None,
    ) -> ChatCompletionResponse:
        model_name = model_config.id
        account_label = f" via account '{account.name}'" if account else ""

        if request.tools and not any(message.role == "tool" for message in request.messages[-2:]):
            fn = (
                request.tools[0].get("function", {})
                if "function" in request.tools[0]
                else request.tools[0]
            )
            call_id = f"toolu_anthropic_{uuid.uuid4().hex[:8]}"
            tool_name = fn.get("name", "test_tool")
            mock_args = "{}"
            props = fn.get("parameters", {}).get("properties", {}) or fn.get(
                "input_schema", {}
            ).get("properties", {})
            if props:
                first_key = next(iter(props.keys()))
                mock_args = json.dumps({first_key: "test_value"})

            message_kwargs: Dict[str, Any] = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": mock_args,
                        },
                    }
                ],
            }
            reasoning = self._mock_reasoning(request, model_config, variables, tool_name)
            if reasoning:
                message_kwargs["reasoning_content"] = reasoning

            return ChatCompletionResponse(
                id=f"chatcmpl-anthropic-{uuid.uuid4().hex[:12]}",
                created=int(time.time()),
                model=model_name,
                choices=[
                    ChatChoice(
                        index=0,
                        message=ChatMessage(**message_kwargs),
                        finish_reason="tool_calls",
                    )
                ],
                usage=CompletionUsage(
                    prompt_tokens=14,
                    completion_tokens=24,
                    total_tokens=38,
                ),
            )

        content = (
            f"Simulated response from Anthropic ({model_name}){account_label}. "
            "Configure a real API key to reach the Anthropic Messages API."
        )
        if any(message.role == "tool" for message in request.messages):
            content = (
                f"Simulated response from Anthropic ({model_name}){account_label} "
                "with tool output processed."
            )

        message_kwargs = {"role": "assistant", "content": content}
        reasoning = self._mock_reasoning(request, model_config, variables)
        if reasoning:
            message_kwargs["reasoning_content"] = reasoning

        return ChatCompletionResponse(
            id=f"chatcmpl-anthropic-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=model_name,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(**message_kwargs),
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
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        account: Optional[Account] = None,
        variables: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        model_name = model_config.id
        chunk_id = f"chatcmpl-anthropic-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        account_label = f" via account '{account.name}'" if account else ""

        if request.tools and not any(message.role == "tool" for message in request.messages[-2:]):
            fn = (
                request.tools[0].get("function", {})
                if "function" in request.tools[0]
                else request.tools[0]
            )
            call_id = f"toolu_anthropic_{uuid.uuid4().hex[:8]}"
            tool_name = fn.get("name", "test_tool")
            mock_args = "{}"
            props = fn.get("parameters", {}).get("properties", {}) or fn.get(
                "input_schema", {}
            ).get("properties", {})
            if props:
                first_key = next(iter(props.keys()))
                mock_args = json.dumps({first_key: "test_value"})

            reasoning = self._mock_reasoning(
                request, model_config, variables, tool_name
            )
            if reasoning:
                yield ChatCompletionChunk(
                    id=chunk_id,
                    created=created,
                    model=model_name,
                    choices=[
                        ChatChoiceDelta(
                            index=0,
                            delta={
                                "role": "assistant",
                                "reasoning_content": reasoning,
                            },
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
                        finish_reason=None,
                    )
                ],
            )
            for piece in (mock_args[: len(mock_args) // 2], mock_args[len(mock_args) // 2 :]):
                if piece:
                    yield ChatCompletionChunk(
                        id=chunk_id,
                        created=created,
                        model=model_name,
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
                model=model_name,
                choices=[ChatChoiceDelta(index=0, delta={}, finish_reason="tool_calls")],
            )
            return

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

        reasoning = self._mock_reasoning(request, model_config, variables)
        if reasoning:
            yield ChatCompletionChunk(
                id=chunk_id,
                created=created,
                model=model_name,
                choices=[
                    ChatChoiceDelta(
                        index=0,
                        delta={"reasoning_content": reasoning},
                        finish_reason=None,
                    )
                ],
            )

        for token in [
            "Simulated ",
            "stream ",
            "response ",
            f"from Anthropic for {model_name}{account_label}.",
        ]:
            yield ChatCompletionChunk(
                id=chunk_id,
                created=created,
                model=model_name,
                choices=[
                    ChatChoiceDelta(index=0, delta={"content": token}, finish_reason=None)
                ],
            )

        yield ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=model_name,
            choices=[ChatChoiceDelta(index=0, delta={}, finish_reason="stop")],
        )

    # ------------------------------------------------------------------
    # Upstream streaming
    # ------------------------------------------------------------------

    @staticmethod
    def _build_usage(
        usage_obj: Optional[Dict[str, Any]],
        default_prompt: int = 0,
        default_completion: int = 0,
        previous: Optional[CompletionUsage] = None,
    ) -> CompletionUsage:
        """Map Anthropic usage, including prompt-cache counters, to OpenAI shape.

        ``prompt_tokens`` follows OpenAI semantics and includes cached tokens.
        Anthropic's ``input_tokens`` excludes cache reads/writes, so both cache
        counters are added to the reported input when present. ``default_prompt``
        is expected to already be normalized (used for streaming continuations).
        """
        usage_obj = usage_obj or {}

        cache_read = usage_obj.get("cache_read_input_tokens")
        cache_created = usage_obj.get("cache_creation_input_tokens")
        if cache_read is None and previous is not None:
            cache_read = getattr(previous, "cache_read_input_tokens", None)
        if cache_created is None and previous is not None:
            cache_created = getattr(previous, "cache_creation_input_tokens", None)
        try:
            cache_read = int(cache_read) if cache_read is not None else None
        except (TypeError, ValueError):
            cache_read = None
        try:
            cache_created = (
                int(cache_created) if cache_created is not None else None
            )
        except (TypeError, ValueError):
            cache_created = None

        reported_input = usage_obj.get("input_tokens")
        if reported_input is not None:
            prompt_tokens = (
                int(reported_input)
                + (cache_read or 0)
                + (cache_created or 0)
            )
        else:
            prompt_tokens = int(default_prompt or 0)
        completion_tokens = int(usage_obj.get("output_tokens") or default_completion or 0)
        total_tokens = int(
            usage_obj.get("total_tokens") or (prompt_tokens + completion_tokens)
        )

        usage = CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
        if cache_read is not None:
            usage.cache_read_input_tokens = cache_read
            usage.prompt_tokens_details = {"cached_tokens": cache_read}
        if cache_created is not None:
            usage.cache_creation_input_tokens = cache_created
        return usage

    async def _stream_messages(
        self,
        url: str,
        headers: Dict[str, str],
        body: Dict[str, Any],
        model_config: PluginModelConfig,
        account: Optional[Account] = None,
        proxy: Optional[str] = None,
        timeout: float = 60.0,
    ) -> AsyncIterator[ChatCompletionChunk]:
        chunk_id = f"chatcmpl-anthropic-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        current_tool_call: Optional[Dict[str, Any]] = None
        current_thinking: Optional[Dict[str, Any]] = None
        pending_thinking: List[Dict[str, Any]] = []
        tool_call_index = 0
        finish_sent = False
        last_usage = CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

        async with httpx.AsyncClient(timeout=timeout, proxy=proxy) as client:
            async with client.stream("POST", url, headers=headers, json=body) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        continue

                    event_str = line[5:].strip()
                    if not event_str or event_str == "[DONE]":
                        continue

                    try:
                        event = json.loads(event_str)
                    except Exception:
                        continue

                    event_type = event.get("type")
                    if event_type == "error":
                        error = event.get("error") or {}
                        raise RuntimeError(
                            error.get("message") or "Anthropic streaming error"
                        )

                    if event_type == "message_start":
                        message = event.get("message") or {}
                        chunk_id = f"chatcmpl-{message.get('id', uuid.uuid4().hex[:12])}"
                        last_usage = self._build_usage(message.get("usage"))
                        # The routing decision is made before the stream opens, so
                        # ``message.model`` already names the served model. Surface
                        # a downgrade in the first chunk, ahead of any answer text.
                        notice = self._downgrade_notice(
                            str(body.get("model") or ""), message
                        )
                        yield ChatCompletionChunk(
                            id=chunk_id,
                            created=created,
                            model=model_config.id,
                            choices=[
                                ChatChoiceDelta(
                                    index=0,
                                    delta={
                                        "role": "assistant",
                                        "content": f"{notice}\n\n" if notice else "",
                                    },
                                    finish_reason=None,
                                )
                            ],
                            usage=last_usage,
                        )
                    elif event_type == "content_block_start":
                        block = event.get("content_block") or {}
                        block_type = block.get("type")
                        if block_type == "thinking":
                            current_thinking = {
                                "type": "thinking",
                                "thinking": self._thinking_text(block),
                            }
                            if block.get("signature"):
                                current_thinking["signature"] = block["signature"]
                        elif block_type == "redacted_thinking":
                            if block.get("data"):
                                pending_thinking.append(
                                    {"type": "redacted_thinking", "data": block["data"]}
                                )
                        elif block_type == "tool_use":
                            current_tool_call = {
                                "index": tool_call_index,
                                "id": block.get("id", f"toolu_{uuid.uuid4().hex[:8]}"),
                                "name": block.get("name", ""),
                            }
                            remember_thinking_blocks(
                                current_tool_call["id"], pending_thinking
                            )
                            pending_thinking = []
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
                                                    "index": tool_call_index,
                                                    "id": current_tool_call["id"],
                                                    "type": "function",
                                                    "function": {
                                                        "name": current_tool_call["name"],
                                                        "arguments": "",
                                                    },
                                                }
                                            ]
                                        },
                                        finish_reason=None,
                                    )
                                ],
                            )
                            tool_call_index += 1
                    elif event_type == "content_block_delta":
                        delta = event.get("delta") or {}
                        delta_type = delta.get("type")
                        if delta_type == "text_delta":
                            yield ChatCompletionChunk(
                                id=chunk_id,
                                created=created,
                                model=model_config.id,
                                choices=[
                                    ChatChoiceDelta(
                                        index=0,
                                        delta={"content": delta.get("text", "")},
                                        finish_reason=None,
                                    )
                                ],
                            )
                        elif delta_type == "thinking_delta":
                            reasoning_text = self._thinking_text(delta)
                            if current_thinking is not None and reasoning_text:
                                current_thinking["thinking"] = (
                                    current_thinking.get("thinking", "") + reasoning_text
                                )
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
                        elif delta_type == "signature_delta" and current_thinking is not None:
                            current_thinking["signature"] = delta.get("signature", "")
                        elif delta_type == "input_json_delta" and current_tool_call is not None:
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
                                                    "index": current_tool_call["index"],
                                                    "function": {
                                                        "arguments": delta.get("partial_json", "")
                                                    },
                                                }
                                            ]
                                        },
                                        finish_reason=None,
                                    )
                                ],
                            )
                    elif event_type == "content_block_stop":
                        if current_thinking is not None:
                            pending_thinking.append(current_thinking)
                            current_thinking = None
                        current_tool_call = None
                    elif event_type == "message_delta":
                        raw_stop = (event.get("delta") or {}).get("stop_reason")
                        finish_reason = (
                            "tool_calls"
                            if raw_stop == "tool_use"
                            else "stop"
                            if raw_stop in ("end_turn", "stop_sequence")
                            else (raw_stop or "stop")
                        )
                        if event.get("usage"):
                            last_usage = self._build_usage(
                                event.get("usage"),
                                default_prompt=last_usage.prompt_tokens,
                                default_completion=last_usage.completion_tokens,
                                previous=last_usage,
                            )
                        yield ChatCompletionChunk(
                            id=chunk_id,
                            created=created,
                            model=model_config.id,
                            choices=[
                                ChatChoiceDelta(
                                    index=0, delta={}, finish_reason=finish_reason
                                )
                            ],
                            usage=last_usage,
                        )
                        finish_sent = True
                    elif event_type == "message_stop" and not finish_sent:
                        yield ChatCompletionChunk(
                            id=chunk_id,
                            created=created,
                            model=model_config.id,
                            choices=[
                                ChatChoiceDelta(index=0, delta={}, finish_reason="stop")
                            ],
                            usage=last_usage,
                        )
                        finish_sent = True

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        variables: Dict[str, Any],
        account: Optional[Account] = None,
    ) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionChunk]]:
        api_key = (
            (account.api_key if account else "")
            or str(variables.get("api_key") or "")
            or str(variables.get("anthropic_api_key") or "")
        )
        base_url = (
            account.base_url
            if (account and account.base_url)
            else str(variables.get("base_url") or "https://api.anthropic.com/v1")
        ).rstrip("/")
        version = str(variables.get("anthropic_version") or "2023-06-01").strip()

        if self._is_test_mode(api_key):
            if request.stream:
                return self._mock_stream(
                    request, model_config, account=account, variables=variables
                )
            return self._mock_response(
                request, model_config, account=account, variables=variables
            )

        payload = self._build_payload(request, model_config, variables)
        headers: Dict[str, str] = {
            "x-api-key": api_key,
            "anthropic-version": version,
            "content-type": "application/json",
        }
        beta = self._beta_header(request, self._resolve_fallbacks(request, variables))
        if beta:
            headers["anthropic-beta"] = beta
        url = f"{base_url}/messages"
        proxy = (
            str(variables.get("proxy") or (getattr(account, "proxy_url", None) or "")).strip()
            or None
        )
        timeout = float(variables.get("timeout") or 60.0)

        if request.stream:
            return self._stream_messages(
                url,
                headers,
                payload,
                model_config,
                account=account,
                proxy=proxy,
                timeout=timeout,
            )

        async with httpx.AsyncClient(timeout=timeout, proxy=proxy) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        text_parts: List[str] = []
        thinking_parts: List[str] = []
        pending_thinking: List[Dict[str, Any]] = []
        tool_calls: List[Dict[str, Any]] = []
        for block in data.get("content", []):
            block_type = block.get("type")
            if block_type == "text":
                text_parts.append(block.get("text", ""))
            elif block_type == "thinking":
                thinking_text = self._thinking_text(block)
                if thinking_text:
                    thinking_parts.append(thinking_text)
                thinking_block: Dict[str, Any] = {
                    "type": "thinking",
                    "thinking": thinking_text,
                }
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
                remember_thinking_blocks(tool_id, pending_thinking)
                pending_thinking = []
                tool_calls.append(
                    {
                        "id": tool_id,
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input", {})),
                        },
                    }
                )

        raw_stop = data.get("stop_reason")
        finish_reason = (
            "tool_calls"
            if (raw_stop == "tool_use" or tool_calls)
            else "stop"
            if raw_stop in ("end_turn", "stop_sequence")
            else (raw_stop or "stop")
        )
        text_content = "".join(text_parts) if text_parts else (None if tool_calls else "")
        notice = self._downgrade_notice(str(payload.get("model") or ""), data)
        if notice:
            text_content = f"{notice}\n\n{text_content}" if text_content else notice
        reasoning_content = "".join(thinking_parts) or None

        message_kwargs: Dict[str, Any] = {
            "role": "assistant",
            "content": text_content,
        }
        if tool_calls:
            message_kwargs["tool_calls"] = tool_calls
        if reasoning_content:
            message_kwargs["reasoning_content"] = reasoning_content

        usage = self._build_usage(data.get("usage"))

        return ChatCompletionResponse(
            id=f"chatcmpl-{data.get('id', uuid.uuid4().hex[:12])}",
            created=int(time.time()),
            model=model_config.id,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(**message_kwargs),
                    finish_reason=finish_reason,
                )
            ],
            usage=usage,
        )


plugin = AnthropicProxyPlugin()
