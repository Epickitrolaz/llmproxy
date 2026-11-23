"""Mistral Vibe provider plugin for LLMProxy.

Mistral Vibe is Mistral's coding agent. This provider connects exactly the way
the Vibe CLI does: the dashboard starts the same browser sign-in flow at
``console.mistral.ai/api/vibe/sign-in`` and exchanges the completed flow for a
Vibe API key. That key works with the Free plan's included monthly usage, and
with paid or pay-as-you-go plans.

Wire protocol (``POST https://api.mistral.ai/v1/chat/completions``):

* Bearer API key, OpenAI-compatible messages, tools, tool results and images.
* ``reasoning_effort`` accepts only ``none`` and ``high`` for the Vibe models;
  reasoning output arrives as ``thinking`` content chunks and is returned to
  clients as ``reasoning_content``. Assistant reasoning is replayed as thinking
  chunks, because Mistral rejects a bare ``reasoning_content`` field.
* Prompt caching uses ``prompt_cache_key``; cached prompt tokens are reported in
  ``usage.prompt_tokens_details.cached_tokens``.
* Streaming responses include usage on the final chunk without
  ``stream_options``, so cache and reasoning counters are always reported.
* Images use OpenAI's ``image_url`` part shape (data URLs are passed through).

The two shipped models are the models Vibe itself uses: ``mistral-vibe-cli-latest``
(Mistral Medium 3.5) and ``mistral-vibe-cli-fast`` (Mistral Small 4).

Local import reads ``MISTRAL_API_KEY`` from the environment or
``$VIBE_HOME/.env`` (``~/.vibe/.env`` by default). A key stored only in the OS
keyring can still be imported by exporting it or by using browser sign-in.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import secrets
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import httpx

from llmproxy.core.interface import BasePlugin
from llmproxy.core.multimodal import (
    is_image_part,
    materialize_message_images,
    messages_have_images,
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

logger = logging.getLogger(__name__)

DEFAULT_API_BASE = "https://api.mistral.ai/v1"
DEFAULT_CONSOLE_URL = "https://console.mistral.ai"
DEFAULT_BROWSER_API_URL = f"{DEFAULT_CONSOLE_URL}/api"
MODEL_MEDIUM = "mv/mistral-medium-3.5"
MODEL_SMALL = "mv/mistral-small"
MAX_CONTEXT = 262_144
MAX_OUTPUT = 262_144
MAX_IMAGES_PER_REQUEST = 8
DROPPED_IMAGE_TEXT = "[Image removed: only the 8 most recent images are sent]"
FLOW_TTL_SECONDS = 900

# Mistral exposes only these adjustable-reasoning levels on the Vibe models.
_REASONING_LEVELS = ["none", "high"]

# Fields Mistral's chat endpoint understands and that callers may put in
# ``extra_body``. Anything else stays in the proxy instead of risking a 400.
_EXTRA_CHAT_FIELDS = {
    "top_k",
    "random_seed",
    "metadata",
    "safe_prompt",
    "service_tier",
    "min_tokens",
    "repetition_penalty",
    "prediction",
    "logprobs",
    "top_logprobs",
    "prompt_logprobs",
    "top_prompt_logprobs",
    "guardrails",
    "prompt_mode",
}

_FINISH_REASONS = {
    "stop": "stop",
    "length": "length",
    "model_length": "length",
    "tool_calls": "tool_calls",
    "content_filter": "content_filter",
}

_THINKING_PART_TYPES = {"thinking", "reasoning"}
_TEXT_PART_TYPES = {"text", "input_text", "output_text"}


def _origin(value: str) -> Optional[Tuple[str, str, int]]:
    """Return ``(scheme, host, effective_port)`` for an absolute HTTP URL."""
    try:
        parsed = urlparse(value)
        port = parsed.port
    except ValueError:
        return None
    scheme = (parsed.scheme or "").lower()
    host = (parsed.hostname or "").lower()
    if scheme not in ("http", "https") or not host or parsed.username or parsed.password:
        return None
    effective = port if port is not None else (443 if scheme == "https" else 80)
    return scheme, host, effective


def _safe_origin(value: str, fallback: str) -> str:
    """Keep a configured HTTP(S) base only when it is an origin + path."""
    candidate = str(value or "").strip().rstrip("/")
    parsed = urlparse(candidate)
    if (
        candidate
        and _origin(candidate) is not None
        and not parsed.query
        and not parsed.fragment
        and (parsed.scheme.lower() == "https" or parsed.hostname in {"127.0.0.1", "::1", "localhost"})
    ):
        return candidate
    return fallback


def _same_origin(candidate: str, base: str) -> bool:
    return _origin(candidate) is not None and _origin(candidate) == _origin(base)


def _parse_expiry(value: Any, default_seconds: float = 600.0) -> float:
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.timestamp()
        except ValueError:
            pass
    return time.time() + default_seconds


def _short_excerpt(value: Any, limit: int = 300) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit]


def _workspace_denied(response: Any) -> bool:
    try:
        payload = response.json()
    except (ValueError, AttributeError):
        return False
    if not isinstance(payload, dict):
        return False
    return payload.get("error_code") == 9004 or payload.get("code") == 9004


def _upstream_message(response: Any, body: Optional[bytes] = None) -> str:
    raw = body if body is not None else getattr(response, "text", "")
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        payload = None
    if isinstance(payload, dict):
        detail = payload.get("message") or payload.get("detail") or payload.get("error")
        if detail is not None:
            return _short_excerpt(detail)
    return _short_excerpt(raw)


class MistralVibePlugin(BasePlugin):
    """Mistral Vibe (CLI browser sign-in) provider."""

    def __init__(self) -> None:
        self._metadata = PluginMetadata(
            id="mistral_vibe",
            name="Mistral Vibe",
            description=(
                "Mistral Vibe through the CLI browser sign-in, serving the "
                "Vibe models on the Free plan's included usage"
            ),
            version="1.0.0",
            author="Mistral AI",
            enabled=True,
            filename="mistral_vibe.py",
            supports_usage_meter=True,
            oauth_login=True,
            oauth_local_import=True,
            oauth_hint="Browser sign-in · Mistral Vibe Free or paid plan",
            variables=self._build_variables(),
            models=self._build_models(),
        )
        self._flows: Dict[str, Dict[str, Any]] = {}
        self._flows_lock = asyncio.Lock()

    @property
    def metadata(self) -> PluginMetadata:
        return self._metadata

    # ------------------------------------------------------------- metadata
    @staticmethod
    def _build_variables() -> List[PluginVariable]:
        return [
            PluginVariable(
                name="base_url",
                label="API Base URL",
                type="string",
                default=DEFAULT_API_BASE,
                description="Mistral chat-completions API base URL",
            ),
            PluginVariable(
                name="console_url",
                label="Console URL",
                type="string",
                default=DEFAULT_CONSOLE_URL,
                description="Mistral console used by the browser sign-in flow",
            ),
            PluginVariable(
                name="browser_api_url",
                label="Console API URL",
                type="string",
                default=DEFAULT_BROWSER_API_URL,
                description="Console API that creates and exchanges Vibe sign-in flows",
            ),
            PluginVariable(
                name="timeout",
                label="Timeout (seconds)",
                type="number",
                default=300,
                description="Request timeout for inference calls",
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
                description="Cooldown after an account returns a rate limit",
            ),
            PluginVariable(
                name="account_switch_wait",
                label="Switch Wait Delay (s)",
                type="number",
                default=0.1,
                description="Optional delay before switching accounts on other failures",
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
                label="Default Reasoning Effort",
                type="select",
                default="default",
                options=["default", "none", "high"],
                description=(
                    "Forwarded as reasoning_effort when the selected Vibe model "
                    "supports it. Mistral accepts only none or high; a request-level "
                    "value takes precedence."
                ),
            ),
            PluginVariable(
                name="proxy_enabled",
                label="Enable External Proxy",
                type="boolean",
                default=False,
                description="Route provider requests through external proxies",
            ),
            PluginVariable(
                name="proxy_strategy",
                label="Proxy Strategy",
                type="select",
                default="sequential",
                options=["sequential", "random", "specific", "switch_on_limit"],
                description="How to choose a proxy from the proxy pool",
            ),
            PluginVariable(
                name="specific_proxy_id",
                label="Specific Proxy ID / Tag",
                type="string",
                default="",
                description="Proxy ID, URL, or tag used by the 'specific' strategy",
            ),
            PluginVariable(
                name="proxy",
                label="Direct Proxy URL",
                type="string",
                default="",
                description="Direct HTTP/SOCKS proxy URL. The router injects this per request.",
            ),
        ]

    @staticmethod
    def _build_models() -> List[PluginModelConfig]:
        return [
            PluginModelConfig(
                id=MODEL_MEDIUM,
                name="Mistral Medium 3.5 (Vibe)",
                upstream_model="mistral-vibe-cli-latest",
                context_window=MAX_CONTEXT,
                max_tokens=MAX_OUTPUT,
                reasoning_levels=list(_REASONING_LEVELS),
                supports_vision=True,
            ),
            PluginModelConfig(
                id=MODEL_SMALL,
                name="Mistral Small 4 (Vibe Fast)",
                upstream_model="mistral-vibe-cli-fast",
                context_window=MAX_CONTEXT,
                max_tokens=MAX_OUTPUT,
                reasoning_levels=list(_REASONING_LEVELS),
                supports_vision=True,
            ),
        ]

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _is_test_mode(
        api_key: str, account: Optional[Account] = None
    ) -> bool:
        key = (api_key or "").strip()
        if not key:
            if account is not None and not str(account.id).startswith("default-"):
                raise ValueError(
                    f"Account '{account.name}' has no API key configured for "
                    "Mistral Vibe. Sign in from the Accounts tab or add an API key."
                )
            return True
        lowered = key.lower()
        prefixes = ("test", "mock", "sk-test", "dummy", "fake", "offline", "sample")
        return any(lowered.startswith(prefix) for prefix in prefixes) or lowered in (
            "test-key",
            "test_key",
            "secret123",
        )

    @staticmethod
    def _resolve_proxy(
        variables: Dict[str, Any], account: Optional[Account]
    ) -> Optional[str]:
        value = str(
            variables.get("proxy")
            or (getattr(account, "proxy_url", None) if account else "")
            or ""
        ).strip()
        return value or None

    @staticmethod
    def _api_base(variables: Dict[str, Any], account: Optional[Account]) -> str:
        configured = (
            account.base_url
            if account is not None and account.base_url
            else variables.get("base_url")
        )
        return str(configured or DEFAULT_API_BASE).rstrip("/")

    @staticmethod
    def _chat_url(base_url: str) -> str:
        return base_url if base_url.endswith("/chat/completions") else f"{base_url}/chat/completions"

    @staticmethod
    def _api_origin(base_url: str) -> str:
        parsed = urlparse(base_url)
        if parsed.scheme and parsed.netloc:
            return f"{parsed.scheme}://{parsed.netloc}"
        return "https://api.mistral.ai"

    @staticmethod
    def _thinking_text(part: Dict[str, Any]) -> str:
        value = part.get("thinking")
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            pieces: List[str] = []
            for inner in value:
                if isinstance(inner, str):
                    pieces.append(inner)
                elif isinstance(inner, dict) and isinstance(inner.get("text"), str):
                    pieces.append(inner["text"])
            if pieces:
                return "".join(pieces)
        for key in ("text", "reasoning", "reasoning_content"):
            if isinstance(part.get(key), str):
                return part[key]
        return ""

    @classmethod
    def _split_content(cls, content: Any) -> Tuple[str, str]:
        """Split Mistral content into ``(text, reasoning)``.

        Mistral reasoning models answer with a list of content chunks:
        ``{"type": "thinking", "thinking": [...]}`` mixed with
        ``{"type": "text", "text": ...}``. Plain string content is text.
        """
        if isinstance(content, str):
            return content, ""
        if not isinstance(content, list):
            return "", ""
        text_parts: List[str] = []
        reasoning_parts: List[str] = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
                continue
            if not isinstance(part, dict):
                continue
            ptype = str(part.get("type") or "").lower()
            if ptype in _THINKING_PART_TYPES or "thinking" in part:
                reasoning = cls._thinking_text(part)
                if reasoning:
                    reasoning_parts.append(reasoning)
            elif ptype in _TEXT_PART_TYPES or isinstance(part.get("text"), str):
                text = part.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
        return "".join(text_parts), "".join(reasoning_parts)

    @staticmethod
    def _mistral_image_part(part: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        item = to_openai_image_part(part)
        if not item:
            return None
        url = item.get("image_url")
        if isinstance(url, dict):
            url = url.get("url")
        if not isinstance(url, str) or not url:
            return None
        return {"type": "image_url", "image_url": {"url": url}}

    @classmethod
    def _content_parts(
        cls, content: Any, *, allow_images: bool
    ) -> Tuple[str, List[Dict[str, Any]], str]:
        """Return ``(text, image_parts, reasoning)`` for a message's content."""
        if isinstance(content, str):
            return content, [], ""
        if not isinstance(content, list):
            return "", [], ""
        text_parts: List[str] = []
        image_parts: List[Dict[str, Any]] = []
        reasoning_parts: List[str] = []
        for part in content:
            if not isinstance(part, dict):
                if isinstance(part, str):
                    text_parts.append(part)
                continue
            if is_image_part(part):
                if allow_images:
                    translated = cls._mistral_image_part(part)
                    if translated is not None:
                        image_parts.append(translated)
                continue
            ptype = str(part.get("type") or "").lower()
            if ptype in _THINKING_PART_TYPES or "thinking" in part:
                reasoning = cls._thinking_text(part)
                if reasoning:
                    reasoning_parts.append(reasoning)
            elif ptype in _TEXT_PART_TYPES or isinstance(part.get("text"), str):
                text = part.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
        return "\n".join(text_parts), image_parts, "".join(reasoning_parts)

    @classmethod
    def _convert_message(cls, message: ChatMessage) -> Dict[str, Any]:
        role = "tool" if message.role == "function" else message.role
        allow_images = role in ("user", "tool")
        text, images, reasoning = cls._content_parts(
            message.content, allow_images=allow_images
        )
        reasoning_content = getattr(message, "reasoning_content", None)
        if isinstance(reasoning_content, str) and reasoning_content:
            reasoning = f"{reasoning}{reasoning_content}"

        payload: Dict[str, Any] = {"role": role}
        if role == "assistant":
            pieces: List[Dict[str, Any]] = []
            if reasoning:
                pieces.append({
                    "type": "thinking",
                    "thinking": [{"type": "text", "text": reasoning}],
                })
            if text:
                pieces.append({"type": "text", "text": text})
            if pieces:
                payload["content"] = pieces
            else:
                payload["content"] = None
            if message.tool_calls:
                payload["tool_calls"] = message.tool_calls
            if message.name:
                payload["name"] = message.name
            return payload

        if images:
            parts: List[Dict[str, Any]] = []
            if text:
                parts.append({"type": "text", "text": text})
            parts.extend(images)
            payload["content"] = parts
        else:
            payload["content"] = text
        if role == "tool":
            if message.tool_call_id:
                payload["tool_call_id"] = message.tool_call_id
            if message.name:
                payload["name"] = message.name
        return payload

    @staticmethod
    def _limit_images(payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Keep only the most recent images, as the Vibe CLI does."""
        remaining = MAX_IMAGES_PER_REQUEST
        for payload in reversed(payloads):
            content = payload.get("content")
            if not isinstance(content, list):
                continue
            for index in range(len(content) - 1, -1, -1):
                part = content[index]
                if not isinstance(part, dict) or part.get("type") != "image_url":
                    continue
                if remaining > 0:
                    remaining -= 1
                else:
                    content[index] = {"type": "text", "text": DROPPED_IMAGE_TEXT}
        return payloads

    @classmethod
    def _convert_messages(
        cls, messages: List[ChatMessage]
    ) -> List[Dict[str, Any]]:
        return cls._limit_images(
            [cls._convert_message(message) for message in messages]
        )

    @staticmethod
    def _resolve_reasoning(
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        variables: Dict[str, Any],
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
        if raw is None:
            raw = variables.get("reasoning_effort")

        level = str(raw or "").strip().lower()
        if not level or level in ("default", "auto"):
            return None
        accepted = {str(entry).lower() for entry in model_config.reasoning_levels}
        if not accepted:
            return None
        if level in ("none", "off", "disabled", "false", "0"):
            return "none" if "none" in accepted else None
        # Mistral accepts only none/high for the Vibe models. Clamp any other
        # positive effort to high instead of letting the upstream return 400.
        return "high" if "high" in accepted else None

    @staticmethod
    def _apply_cache_fields(
        body: Dict[str, Any], request: ChatCompletionRequest
    ) -> None:
        extra = request.extra_body if isinstance(request.extra_body, dict) else {}
        value = getattr(request, "prompt_cache_key", None)
        if value is None:
            value = extra.get("prompt_cache_key")
        if value is not None and str(value).strip():
            body["prompt_cache_key"] = str(value).strip()

    @staticmethod
    def _apply_extra_fields(
        body: Dict[str, Any], request: ChatCompletionRequest
    ) -> None:
        extra = request.extra_body if isinstance(request.extra_body, dict) else {}
        for key in _EXTRA_CHAT_FIELDS:
            value = extra.get(key)
            if value is not None:
                body[key] = value

    def _build_body(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        variables: Dict[str, Any],
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": model_config.upstream_model or model_config.id,
            "messages": self._convert_messages(request.messages),
            "stream": bool(request.stream),
        }
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.top_p is not None:
            body["top_p"] = request.top_p
        if request.presence_penalty is not None:
            body["presence_penalty"] = request.presence_penalty
        if request.frequency_penalty is not None:
            body["frequency_penalty"] = request.frequency_penalty
        max_tokens = (
            request.max_tokens
            if request.max_tokens is not None
            else request.max_completion_tokens
        )
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if request.stop is not None:
            body["stop"] = request.stop
        if request.n is not None and request.n != 1:
            body["n"] = request.n
        if request.tools is not None:
            body["tools"] = request.tools
        if request.tool_choice is not None:
            body["tool_choice"] = request.tool_choice
        if request.parallel_tool_calls is not None:
            body["parallel_tool_calls"] = request.parallel_tool_calls
        if request.response_format is not None:
            body["response_format"] = request.response_format

        reasoning = self._resolve_reasoning(request, model_config, variables)
        if reasoning:
            body["reasoning_effort"] = reasoning

        # Mistral includes usage in the last streamed chunk on its own; it does
        # not understand OpenAI's stream_options. Never forward it.
        self._apply_cache_fields(body, request)
        self._apply_extra_fields(body, request)
        return body

    # ---------------------------------------------------------------- usage
    @staticmethod
    def _to_int(value: Any) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _usage(self, raw: Any) -> Optional[CompletionUsage]:
        if not isinstance(raw, dict):
            return None
        prompt = self._to_int(raw.get("prompt_tokens"))
        completion = self._to_int(raw.get("completion_tokens"))
        if prompt is None and completion is None:
            return None
        prompt = prompt or 0
        completion = completion or 0
        total = self._to_int(raw.get("total_tokens"))
        if total is None:
            total = prompt + completion

        payload: Dict[str, Any] = {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
        }
        details = raw.get("prompt_tokens_details")
        cached = None
        if isinstance(details, dict):
            cached = details.get("cached_tokens")
        if cached is None:
            cached = raw.get("cached_tokens")
        if cached is not None:
            payload["prompt_tokens_details"] = {"cached_tokens": cached}
        completion_details = raw.get("completion_tokens_details")
        if isinstance(completion_details, dict) and "reasoning_tokens" in completion_details:
            payload["completion_tokens_details"] = {
                "reasoning_tokens": completion_details["reasoning_tokens"]
            }
        return CompletionUsage.model_validate(payload)

    # ------------------------------------------------------------ responses
    @staticmethod
    def _finish_reason(raw: Any) -> Optional[str]:
        if raw is None:
            return None
        reason = str(raw)
        if reason == "error":
            raise RuntimeError("Mistral Vibe ended the stream with an error finish reason")
        return _FINISH_REASONS.get(reason, reason)

    def _normalize_message(self, message: Any) -> ChatMessage:
        if not isinstance(message, dict):
            message = {}
        text, reasoning = self._split_content(message.get("content"))
        if not reasoning:
            reasoning_content = message.get("reasoning_content")
            if isinstance(reasoning_content, str):
                reasoning = reasoning_content
        payload: Dict[str, Any] = {
            "role": "assistant",
            "content": text if text else None,
        }
        if reasoning:
            payload["reasoning_content"] = reasoning
        if message.get("tool_calls") is not None:
            payload["tool_calls"] = message.get("tool_calls")
        if message.get("name"):
            payload["name"] = message.get("name")
        return ChatMessage.model_validate(payload)

    def _normalize_response(
        self,
        data: Dict[str, Any],
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
    ) -> ChatCompletionResponse:
        choices: List[ChatChoice] = []
        for index, choice in enumerate(data.get("choices") or []):
            if not isinstance(choice, dict):
                continue
            choices.append(
                ChatChoice(
                    index=choice.get("index", index),
                    message=self._normalize_message(choice.get("message")),
                    finish_reason=self._finish_reason(choice.get("finish_reason")),
                    logprobs=choice.get("logprobs"),
                )
            )
        created = self._to_int(data.get("created")) or int(time.time())
        return ChatCompletionResponse(
            id=str(data.get("id") or f"chatcmpl-vibe-{uuid.uuid4().hex[:12]}"),
            object=str(data.get("object") or "chat.completion"),
            created=created,
            model=request.model or model_config.id,
            choices=choices,
            usage=self._usage(data.get("usage")),
            system_fingerprint=data.get("system_fingerprint"),
        )

    # --------------------------------------------------------------- stream
    async def _stream_request(
        self,
        url: str,
        headers: Dict[str, str],
        body: Dict[str, Any],
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        account: Optional[Account],
        timeout: float,
        proxy: Optional[str],
    ) -> AsyncIterator[ChatCompletionChunk]:
        async with httpx.AsyncClient(timeout=timeout, proxy=proxy) as client:
            async with client.stream("POST", url, headers=headers, json=body) as response:
                if account is not None:
                    capture_rate_limit_headers(account, response.headers)
                if response.status_code >= 400:
                    raw = await response.aread()
                    raise RuntimeError(
                        "Mistral Vibe upstream error "
                        f"(HTTP {response.status_code}): "
                        f"{_upstream_message(response, raw)}"
                    )

                chunk_id = f"chatcmpl-vibe-{uuid.uuid4().hex[:12]}"
                created = int(time.time())
                first = True
                finish_reason: Optional[str] = None
                saw_tool_calls = False
                usage: Optional[CompletionUsage] = None
                usage_yielded = False
                finish_yielded = False

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
                    except ValueError:
                        continue
                    if not isinstance(parsed, dict):
                        continue
                    if parsed.get("error"):
                        error = parsed["error"]
                        detail = (
                            error.get("message")
                            if isinstance(error, dict)
                            else error
                        )
                        raise RuntimeError(
                            f"Mistral Vibe streaming error: {_short_excerpt(detail)}"
                        )

                    chunk_usage = self._usage(parsed.get("usage"))
                    if chunk_usage is not None:
                        usage = chunk_usage
                    if parsed.get("id"):
                        chunk_id = str(parsed["id"])
                    if parsed.get("created") is not None:
                        created = self._to_int(parsed.get("created")) or created
                    model_name = str(parsed.get("model") or request.model or model_config.id)

                    choices = parsed.get("choices") or []
                    if not choices:
                        if chunk_usage is not None:
                            yield ChatCompletionChunk(
                                id=chunk_id,
                                created=created,
                                model=model_name,
                                choices=[],
                                usage=chunk_usage,
                            )
                            usage_yielded = True
                        continue

                    choice = choices[0] if isinstance(choices[0], dict) else {}
                    delta_raw = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
                    text, reasoning = self._split_content(delta_raw.get("content"))
                    reasoning_content = delta_raw.get("reasoning_content")
                    if isinstance(reasoning_content, str) and reasoning_content:
                        reasoning = f"{reasoning}{reasoning_content}"
                    if not reasoning:
                        _, extra_reasoning = self._split_content(
                            delta_raw.get("reasoning")
                        )
                        reasoning = extra_reasoning

                    delta: Dict[str, Any] = {}
                    if first:
                        delta["role"] = delta_raw.get("role") or "assistant"
                    if reasoning:
                        delta["reasoning_content"] = reasoning
                    if text:
                        delta["content"] = text
                    tool_calls = delta_raw.get("tool_calls")
                    if tool_calls:
                        delta["tool_calls"] = tool_calls
                        saw_tool_calls = True

                    mapped_finish = self._finish_reason(choice.get("finish_reason"))
                    if mapped_finish is not None:
                        finish_reason = mapped_finish

                    has_usage = chunk_usage is not None
                    if delta or mapped_finish is not None or has_usage:
                        yield ChatCompletionChunk(
                            id=chunk_id,
                            created=created,
                            model=model_name,
                            choices=[
                                ChatChoiceDelta(
                                    index=choice.get("index", 0),
                                    delta=delta,
                                    finish_reason=mapped_finish,
                                )
                            ],
                            usage=chunk_usage,
                        )
                        first = False
                        if mapped_finish is not None:
                            finish_yielded = True
                    if has_usage:
                        usage = chunk_usage
                        usage_yielded = True

                if finish_reason is None:
                    finish_reason = "tool_calls" if saw_tool_calls else "stop"
                    yield ChatCompletionChunk(
                        id=chunk_id,
                        created=created,
                        model=request.model or model_config.id,
                        choices=[
                            ChatChoiceDelta(
                                index=0,
                                delta={},
                                finish_reason=finish_reason,
                            )
                        ],
                        usage=None if usage_yielded else usage,
                    )
                    finish_yielded = True
                    usage_yielded = usage_yielded or usage is not None
                elif not finish_yielded:
                    yield ChatCompletionChunk(
                        id=chunk_id,
                        created=created,
                        model=request.model or model_config.id,
                        choices=[
                            ChatChoiceDelta(
                                index=0,
                                delta={},
                                finish_reason=finish_reason,
                            )
                        ],
                    )
                    finish_yielded = True

                if usage is not None and not usage_yielded:
                    # Mistral normally reports usage on its finish chunk. Keep
                    # the counter visible even if it arrived before the finish.
                    yield ChatCompletionChunk(
                        id=chunk_id,
                        created=created,
                        model=request.model or model_config.id,
                        choices=[],
                        usage=usage,
                    )

    # -------------------------------------------------------------- upstream
    async def _complete_request(
        self,
        url: str,
        headers: Dict[str, str],
        body: Dict[str, Any],
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        account: Optional[Account],
        timeout: float,
        proxy: Optional[str],
    ) -> ChatCompletionResponse:
        async with httpx.AsyncClient(timeout=timeout, proxy=proxy) as client:
            response = await client.post(url, headers=headers, json=body)
        if account is not None:
            capture_rate_limit_headers(account, response.headers)
        if response.status_code >= 400:
            raise RuntimeError(
                "Mistral Vibe upstream error "
                f"(HTTP {response.status_code}): {_upstream_message(response)}"
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError("Mistral Vibe returned a non-JSON response") from exc
        if not isinstance(data, dict):
            raise RuntimeError("Mistral Vibe returned an unexpected response shape")
        return self._normalize_response(data, request, model_config)

    # -------------------------------------------------------------- simulate
    @staticmethod
    def _mock_tokens() -> CompletionUsage:
        return CompletionUsage(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        )

    def _mock_response(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        account: Optional[Account] = None,
    ) -> ChatCompletionResponse:
        model_name = request.model or model_config.id
        if request.tools and not any(
            message.role in ("tool", "function") for message in request.messages[-2:]
        ):
            fn = (
                request.tools[0].get("function", {})
                if isinstance(request.tools[0], dict) and "function" in request.tools[0]
                else request.tools[0]
            )
            properties = fn.get("parameters", {}).get("properties", {}) if isinstance(fn, dict) else {}
            arguments = (
                json.dumps({next(iter(properties)): "test_value"})
                if properties
                else "{}"
            )
            return ChatCompletionResponse(
                id=f"chatcmpl-vibe-{uuid.uuid4().hex[:12]}",
                created=int(time.time()),
                model=model_name,
                choices=[
                    ChatChoice(
                        index=0,
                        message=ChatMessage(
                            role="assistant",
                            content=None,
                            tool_calls=[
                                {
                                    "id": f"call_vibe_{uuid.uuid4().hex[:8]}",
                                    "type": "function",
                                    "function": {
                                        "name": fn.get("name", "test_tool") if isinstance(fn, dict) else "test_tool",
                                        "arguments": arguments,
                                    },
                                }
                            ],
                        ),
                        finish_reason="tool_calls",
                    )
                ],
                usage=self._mock_tokens(),
            )

        label = f" via account '{account.name}'" if account else ""
        content = (
            f"Simulated response from Mistral Vibe ({model_name}){label}. "
            "Sign in from the Accounts tab to use your Mistral plan."
        )
        return ChatCompletionResponse(
            id=f"chatcmpl-vibe-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=model_name,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=content),
                    finish_reason="stop",
                )
            ],
            usage=self._mock_tokens(),
        )

    async def _mock_stream(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        account: Optional[Account] = None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        model_name = request.model or model_config.id
        chunk_id = f"chatcmpl-vibe-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        if request.tools and not any(
            message.role in ("tool", "function") for message in request.messages[-2:]
        ):
            fn = (
                request.tools[0].get("function", {})
                if isinstance(request.tools[0], dict) and "function" in request.tools[0]
                else request.tools[0]
            )
            properties = fn.get("parameters", {}).get("properties", {}) if isinstance(fn, dict) else {}
            arguments = (
                json.dumps({next(iter(properties)): "test_value"})
                if properties
                else "{}"
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
                                    "id": f"call_vibe_{uuid.uuid4().hex[:8]}",
                                    "type": "function",
                                    "function": {
                                        "name": fn.get("name", "test_tool") if isinstance(fn, dict) else "test_tool",
                                        "arguments": "",
                                    },
                                }
                            ],
                        },
                    )
                ],
            )
            for piece in (arguments[: len(arguments) // 2], arguments[len(arguments) // 2 :]):
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
                            )
                        ],
                    )
            yield ChatCompletionChunk(
                id=chunk_id,
                created=created,
                model=model_name,
                choices=[
                    ChatChoiceDelta(index=0, delta={}, finish_reason="tool_calls")
                ],
                usage=self._mock_tokens(),
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
        label = f" via account '{account.name}'" if account else ""
        for token in (
            "Simulated ",
            "stream ",
            "response ",
            "from Mistral Vibe",
            f" ({model_name}){label}.",
        ):
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
            usage=self._mock_tokens(),
        )

    # ------------------------------------------------------------ BasePlugin
    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        variables: Dict[str, Any],
        account: Optional[Account] = None,
    ) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionChunk]]:
        api_key = account.api_key if account else ""
        if self._is_test_mode(api_key, account):
            if request.stream:
                return self._mock_stream(request, model_config, account=account)
            return self._mock_response(request, model_config, account=account)

        base_url = self._api_base(variables, account)
        url = self._chat_url(base_url)
        proxy = self._resolve_proxy(variables, account)
        timeout = float(variables.get("timeout") or 300.0)
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if request.stream else "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "mistral-vibe",
        }

        messages = request.messages
        if messages_have_images(messages):
            messages = await materialize_message_images(
                messages, proxy=proxy, timeout=min(timeout, 60.0)
            )
        # Build a shallow request copy so the caller's messages stay untouched.
        effective = request.model_copy(update={"messages": messages})
        body = self._build_body(effective, model_config, variables)

        if request.stream:
            return self._stream_request(
                url,
                headers,
                body,
                effective,
                model_config,
                account,
                timeout,
                proxy,
            )
        return await self._complete_request(
            url,
            headers,
            body,
            effective,
            model_config,
            account,
            timeout,
            proxy,
        )

    # ---------------------------------------------------------------- OAuth
    def _oauth_bases(self) -> Tuple[str, str]:
        variables: Dict[str, Any] = {}
        try:
            from llmproxy.core.manager import plugin_manager

            config = plugin_manager.configs.get(self.metadata.id) or {}
            configured = config.get("variables")
            if isinstance(configured, dict):
                variables = configured
        except Exception:
            variables = {}

        console = _safe_origin(
            str(variables.get("console_url") or DEFAULT_CONSOLE_URL),
            DEFAULT_CONSOLE_URL,
        )
        browser_api_default = (
            f"{console}/api" if console != DEFAULT_CONSOLE_URL else DEFAULT_BROWSER_API_URL
        )
        browser_api = _safe_origin(
            str(variables.get("browser_api_url") or browser_api_default),
            browser_api_default,
        )
        return console, browser_api

    async def _prune_flows(self) -> None:
        now = time.time()
        async with self._flows_lock:
            for flow_id, flow in list(self._flows.items()):
                if now - float(flow.get("created_at") or 0) > FLOW_TTL_SECONDS:
                    self._flows.pop(flow_id, None)

    async def _get_flow(self, flow_id: str) -> Optional[Dict[str, Any]]:
        async with self._flows_lock:
            flow = self._flows.get(flow_id)
            if flow is None:
                return None
            if time.time() - float(flow.get("created_at") or 0) > FLOW_TTL_SECONDS:
                self._flows.pop(flow_id, None)
                return None
            return flow

    @staticmethod
    def _flow_error(flow: Dict[str, Any], message: str) -> Dict[str, Any]:
        flow["status"] = "error"
        flow["error"] = message
        flow.pop("account", None)
        return {"status": "error", "error": message}

    async def oauth_start(self) -> Dict[str, Any]:
        await self._prune_flows()
        console, browser_api = self._oauth_bases()
        verifier = secrets.token_urlsafe(64)
        challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest())
            .decode("ascii")
            .rstrip("=")
        )
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(30.0, connect=15.0),
                follow_redirects=False,
            ) as client:
                response = await client.post(
                    f"{browser_api}/vibe/sign-in",
                    json={
                        "code_challenge": challenge,
                        "code_challenge_method": "S256",
                    },
                    headers={
                        "Accept": "application/json",
                        "Content-Type": "application/json",
                        "User-Agent": "mistral-vibe",
                    },
                )
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Could not start Mistral Vibe sign-in: {exc}") from exc

        if response.status_code >= 400:
            raise RuntimeError(
                "Could not start Mistral Vibe sign-in "
                f"(HTTP {response.status_code}): {_upstream_message(response)}"
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError("Mistral Vibe sign-in returned a non-JSON response") from exc
        if not isinstance(data, dict):
            raise RuntimeError("Mistral Vibe sign-in returned an unexpected response")

        process_id = str(data.get("process_id") or "").strip()
        sign_in_url = str(data.get("sign_in_url") or "").strip()
        poll_url = str(data.get("poll_url") or "").strip()
        if not process_id or not sign_in_url or not poll_url:
            raise RuntimeError("Mistral Vibe sign-in response is missing its flow URLs")
        if not _same_origin(sign_in_url, console):
            raise RuntimeError("Mistral Vibe sign-in returned an unexpected browser host")
        if not _same_origin(poll_url, browser_api):
            raise RuntimeError("Mistral Vibe sign-in returned an unexpected poll host")

        expires_at = _parse_expiry(data.get("expires_at"))
        flow_id = secrets.token_urlsafe(24)
        flow = {
            "process_id": process_id,
            "poll_url": poll_url,
            "code_verifier": verifier,
            "browser_api_url": browser_api,
            "console_url": console,
            "created_at": time.time(),
            "expires_at": expires_at,
            "status": "pending",
            "lock": asyncio.Lock(),
        }
        async with self._flows_lock:
            self._flows[flow_id] = flow

        return {
            "flow_id": flow_id,
            "mode": "device",
            "verification_uri_complete": sign_in_url,
            "expires_in": max(0, int(expires_at - time.time())),
            "interval": 3,
            "instructions": (
                "Approve the Mistral Vibe sign-in in your browser. This creates a "
                "Vibe API key for your plan; the Free plan's included usage works."
            ),
        }

    async def oauth_status(self, flow_id: str) -> Dict[str, Any]:
        flow = await self._get_flow(flow_id)
        if flow is None:
            return {
                "status": "error",
                "error": "This Mistral Vibe sign-in flow is unknown or expired. Start again.",
            }
        if flow.get("status") == "complete" and flow.get("account") is not None:
            return {"status": "complete", "account": flow["account"]}
        if flow.get("status") == "error":
            return {"status": "error", "error": flow.get("error") or "Sign-in failed."}
        if time.time() > float(flow.get("expires_at") or 0) + 30:
            return self._flow_error(flow, "Mistral Vibe sign-in expired. Start again.")

        async with flow["lock"]:
            if flow.get("status") == "complete" and flow.get("account") is not None:
                return {"status": "complete", "account": flow["account"]}
            if flow.get("status") == "error":
                return {"status": "error", "error": flow.get("error") or "Sign-in failed."}

            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(30.0, connect=15.0),
                    follow_redirects=False,
                ) as client:
                    response = await client.get(
                        flow["poll_url"],
                        headers={
                            "Accept": "application/json",
                            "User-Agent": "mistral-vibe",
                        },
                    )
                    if response.status_code == 410:
                        return self._flow_error(
                            flow, "Mistral Vibe sign-in expired. Start again."
                        )
                    if response.status_code >= 500:
                        # A transient console failure is not a verdict; the
                        # dashboard polls again in a few seconds.
                        return {"status": "pending"}
                    if response.status_code >= 400:
                        return self._flow_error(
                            flow,
                            "Could not read the Mistral Vibe sign-in status "
                            f"(HTTP {response.status_code}).",
                        )
                    try:
                        payload = response.json()
                    except ValueError:
                        return self._flow_error(
                            flow, "Mistral Vibe sign-in returned a non-JSON status."
                        )
                    if not isinstance(payload, dict):
                        return self._flow_error(
                            flow, "Mistral Vibe sign-in returned an unexpected status."
                        )
                    state = str(payload.get("status") or "").strip().lower()
                    if state == "pending":
                        return {"status": "pending"}
                    if state != "completed":
                        message = str(
                            payload.get("message")
                            or (
                                "Mistral Vibe sign-in was denied."
                                if state == "denied"
                                else "Mistral Vibe sign-in expired."
                                if state == "expired"
                                else "Mistral Vibe sign-in failed."
                            )
                        )
                        return self._flow_error(flow, _short_excerpt(message, 200))

                    exchange_token = str(
                        payload.get("exchange_token") or ""
                    ).strip()
                    if not exchange_token:
                        return self._flow_error(
                            flow, "Mistral Vibe sign-in completed without an exchange token."
                        )

                    flow["status"] = "exchanging"
                    exchange = await client.post(
                        f"{flow['browser_api_url']}/vibe/sign-in/"
                        f"{flow['process_id']}/exchange",
                        json={
                            "exchange_token": exchange_token,
                            "code_verifier": flow["code_verifier"],
                        },
                        headers={
                            "Accept": "application/json",
                            "Content-Type": "application/json",
                            "User-Agent": "mistral-vibe",
                        },
                    )
            except httpx.HTTPError as exc:
                flow["status"] = "pending"
                logger.warning("Mistral Vibe sign-in poll failed: %s", type(exc).__name__)
                return {"status": "pending"}

            if exchange.status_code == 403 and _workspace_denied(exchange):
                return self._flow_error(
                    flow,
                    "This Mistral workspace cannot create a Vibe API key. "
                    "Sign in again and choose an eligible workspace.",
                )
            if exchange.status_code >= 400:
                return self._flow_error(
                    flow,
                    "Could not finish Mistral Vibe sign-in "
                    f"(HTTP {exchange.status_code}): "
                    f"{_upstream_message(exchange)}",
                )
            try:
                exchanged = exchange.json()
            except ValueError:
                return self._flow_error(
                    flow, "Mistral Vibe sign-in exchange returned a non-JSON response."
                )
            api_key = (
                str(exchanged.get("api_key") or "").strip()
                if isinstance(exchanged, dict)
                else ""
            )
            if not api_key:
                return self._flow_error(
                    flow, "Mistral Vibe sign-in exchange did not return an API key."
                )

            account = await self._account_from_api_key(
                api_key,
                console_url=flow["console_url"],
                browser_api_url=flow["browser_api_url"],
            )
            flow["status"] = "complete"
            flow["account"] = account
            flow["code_verifier"] = ""
            flow["exchange_token"] = ""
            return {"status": "complete", "account": account}

    async def oauth_callback(
        self, flow_id: str, callback_url: str
    ) -> Dict[str, Any]:
        return {
            "status": "error",
            "error": (
                "Mistral Vibe uses a browser poll, not a redirect callback. "
                "Open the sign-in link and approve it in the browser."
            ),
        }

    async def oauth_import_local(self) -> Optional[Account]:
        api_key = str(os.environ.get("MISTRAL_API_KEY") or "").strip()
        source = "MISTRAL_API_KEY environment variable"
        if not api_key:
            vibe_home = Path(
                os.environ.get("VIBE_HOME") or str(Path.home() / ".vibe")
            ).expanduser()
            env_path = vibe_home / ".env"
            api_key = self._read_vibe_env(env_path)
            if api_key:
                source = str(env_path)
        if not api_key:
            return None

        console, browser_api = self._oauth_bases()
        account = await self._account_from_api_key(
            api_key,
            console_url=console,
            browser_api_url=browser_api,
        )
        account.extra_config["imported_from"] = source
        return account

    @staticmethod
    def _read_vibe_env(path: Path) -> str:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return ""
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if line.startswith("export "):
                line = line[7:].strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() != "MISTRAL_API_KEY":
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            elif " #" in value:
                value = value.split(" #", 1)[0]
            return value.strip()
        return ""

    async def refresh_account(self, account: Account) -> bool:
        # Vibe API keys do not expire. A refresh may still adopt a tenant API
        # host advertised by /whoami, so the dashboard Refresh button works.
        if not account.api_key:
            return False
        info = await self._fetch_whoami(account.api_key, account.base_url)
        if not info:
            return False
        changed = False
        base_url = self._base_from_whoami(info)
        if base_url and account.base_url != base_url:
            account.base_url = base_url
            changed = True
        extra = account.extra_config if isinstance(account.extra_config, dict) else {}
        plan = self._plan_info(info)
        for key, value in plan.items():
            if value and extra.get(key) != value:
                extra[key] = value
                changed = True
        account.extra_config = extra
        return changed

    async def _fetch_whoami(
        self, api_key: str, base_url: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        if not api_key:
            return None
        variables: Dict[str, Any] = {}
        try:
            from llmproxy.core.manager import plugin_manager

            config = plugin_manager.configs.get(self.metadata.id) or {}
            configured = config.get("variables")
            if isinstance(configured, dict):
                variables = configured
        except Exception:
            variables = {}
        configured_base = base_url or str(variables.get("base_url") or DEFAULT_API_BASE)
        origin = self._api_origin(configured_base)
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{origin}/api/vibe/whoami",
                    headers={
                        "Accept": "application/json",
                        "Authorization": f"Bearer {api_key}",
                        "User-Agent": "mistral-vibe",
                    },
                )
        except httpx.HTTPError:
            return None
        if response.status_code >= 400:
            return None
        try:
            payload = response.json()
        except ValueError:
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _plan_info(info: Dict[str, Any]) -> Dict[str, Any]:
        plan_type = str(info.get("plan_type") or "").strip().lower()
        plan_name = str(info.get("plan_name") or "").strip()
        upper = plan_name.upper()
        if plan_type == "chat":
            label = {
                "FREE": "Free",
                "INDIVIDUAL": "Pro",
                "EDU": "Student",
                "TEAM": "Team",
            }.get(upper, plan_name.title())
        elif plan_type == "mistral_code":
            label = {"F": "Free Codestral", "E": "Code Enterprise"}.get(
                upper, plan_name.title()
            )
        else:
            label = "Free API" if "FREE" in upper else ("PAYG API" if upper else "")
        return {
            "plan_type": plan_type or None,
            "plan_name": plan_name or None,
            "user_plan": label or None,
            "customer_id": info.get("customer_id"),
            "organization_kind": info.get("organization_kind"),
            "api_base": info.get("api_base"),
            "vibe_base": info.get("vibe_base"),
        }

    @staticmethod
    def _base_from_whoami(info: Dict[str, Any]) -> Optional[str]:
        candidate = str(info.get("api_base") or "").strip().rstrip("/")
        if not candidate:
            return None
        parsed = urlparse(candidate)
        if (
            parsed.scheme.lower() != "https"
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or ".." in parsed.path
        ):
            return None
        if candidate.endswith("/v1"):
            return candidate
        return f"{candidate}/v1"

    async def _account_from_api_key(
        self,
        api_key: str,
        *,
        console_url: str,
        browser_api_url: str,
    ) -> Account:
        info = await self._fetch_whoami(api_key) or {}
        plan = self._plan_info(info)
        plan_label = plan.get("user_plan") or ""
        customer_id = str(plan.get("customer_id") or "").strip()
        identity = customer_id[-6:] if customer_id else ""
        name = "Mistral Vibe"
        if plan_label:
            name += f" ({plan_label})"
        if identity:
            name += f" · {identity}"
        extra = {
            "auth_type": "vibe-browser",
            "console_url": console_url,
            "browser_api_url": browser_api_url,
            "signed_in_at": time.time(),
        }
        extra.update({key: value for key, value in plan.items() if value})
        return Account(
            id=f"acc_mistral_vibe_{uuid.uuid4().hex[:8]}",
            provider_id=self.metadata.id,
            name=name,
            api_key=api_key,
            base_url=self._base_from_whoami(info),
            priority=1,
            weight=1,
            enabled=True,
            extra_config=extra,
        )


plugin = MistralVibePlugin()
