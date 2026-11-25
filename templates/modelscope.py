from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union

import httpx

from llmproxy.core.interface import BasePlugin
from llmproxy.core.multimodal import infer_vision_support
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

MODELSCOPE_BASE_URL = "https://api-inference.modelscope.ai/v1"
PREFIX = "ms"

THINK_OPEN = " thinking"
THINK_CLOSE = ""

_LEVEL_ALIASES = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
}
_NONE_LEVELS = {"none", "off", "disabled", "false", "0"}

_HYBRID_LEVELS = ["none", "low", "medium", "high"]
_ALWAYS_ON_LEVELS = ["low", "medium", "high"]

_REASONING_SUFFIXES = (
    ":none", "-none", ":off", "-off",
    ":low", "-low", ":medium", "-medium", ":high", "-high",
)


class ModelScopeAPIError(RuntimeError):
    """Upstream failure carrying the HTTP status and a short body excerpt."""

    def __init__(self, status_code: int, message: str, request_id: Optional[str] = None) -> None:
        self.status_code = status_code
        self.upstream_message = message
        self.request_id = request_id
        label = "rate limit" if status_code == 429 else "error"
        detail = message.strip() or "no response body"
        suffix = f" (request_id {request_id})" if request_id else ""
        super().__init__(f"ModelScope API {label} {status_code}: {detail}{suffix}")


class _ThinkTagSplitter:
    """Split inline ``think`` markup out of streamed content.

    Tags can straddle chunk boundaries, so the tail of a chunk is held back while
    it could still become a tag. ``feed`` returns ``(reasoning, content)`` and
    ``flush`` drains whatever is left once the stream ends.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.in_think = False
        self._buffer = ""

    @staticmethod
    def _pending_prefix(buffer: str, tag: str) -> int:
        """Length of the longest buffer suffix that is a prefix of ``tag``."""
        limit = min(len(buffer), len(tag) - 1)
        for size in range(limit, 0, -1):
            if buffer.endswith(tag[:size]):
                return size
        return 0

    def feed(self, text: str) -> Tuple[str, str]:
        if not text:
            return "", ""
        if not self.enabled:
            return "", text

        self._buffer += text
        reasoning: List[str] = []
        content: List[str] = []

        while True:
            tag = THINK_CLOSE if self.in_think else THINK_OPEN
            index = self._buffer.find(tag)
            if index == -1:
                break
            head = self._buffer[:index]
            (reasoning if self.in_think else content).append(head)
            self._buffer = self._buffer[index + len(tag):]
            self.in_think = not self.in_think

        tag = THINK_CLOSE if self.in_think else THINK_OPEN
        hold = self._pending_prefix(self._buffer, tag)
        if hold:
            emit, self._buffer = self._buffer[:-hold], self._buffer[-hold:]
        else:
            emit, self._buffer = self._buffer, ""
        if emit:
            (reasoning if self.in_think else content).append(emit)
        return "".join(reasoning), "".join(content)

    def flush(self) -> Tuple[str, str]:
        remainder, self._buffer = self._buffer, ""
        if not remainder:
            return "", ""
        return (remainder, "") if self.in_think else ("", remainder)


def _split_think_text(text: str) -> Tuple[str, str]:
    """Non-streaming variant: ``(content_without_tags, reasoning)``."""
    if THINK_OPEN not in text and THINK_CLOSE not in text:
        return text, ""
    reasoning_parts: List[str] = []
    content_parts: List[str] = []
    cursor = 0
    pattern = re.compile(
        re.escape(THINK_OPEN) + r"(.*?)" + re.escape(THINK_CLOSE), re.S
    )
    for match in pattern.finditer(text):
        content_parts.append(text[cursor:match.start()])
        reasoning_parts.append(match.group(1))
        cursor = match.end()
    content_parts.append(text[cursor:])
    cleaned = "".join(content_parts)
    # An unterminated opening tag means everything after it is still reasoning.
    if THINK_OPEN in cleaned:
        head, _, tail = cleaned.partition(THINK_OPEN)
        reasoning_parts.append(tail)
        cleaned = head
    cleaned = cleaned.replace(THINK_CLOSE, "")
    return cleaned.strip(), "".join(reasoning_parts).strip()


class ModelScopePlugin(BasePlugin):
    """ModelScope API-Inference provider with normalized reasoning output."""

    def __init__(self) -> None:
        models = [
            PluginModelConfig(
                id=f"{PREFIX}/Qwen/Qwen3.5-27B",
                name="Qwen3.5 27B",
                upstream_model="Qwen/Qwen3.5-27B",
                context_window=262144,
                max_tokens=32768,
                reasoning_levels=list(_HYBRID_LEVELS),
                supports_vision=True,
            ),
            PluginModelConfig(
                id=f"{PREFIX}/Qwen/Qwen3.5-35B-A3B",
                name="Qwen3.5 35B A3B",
                upstream_model="Qwen/Qwen3.5-35B-A3B",
                context_window=262144,
                max_tokens=32768,
                reasoning_levels=list(_HYBRID_LEVELS),
                supports_vision=True,
            ),
            PluginModelConfig(
                id=f"{PREFIX}/Qwen/Qwen3.5-397B-A17B",
                name="Qwen3.5 397B A17B",
                upstream_model="Qwen/Qwen3.5-397B-A17B",
                context_window=262144,
                max_tokens=32768,
                reasoning_levels=list(_HYBRID_LEVELS),
                supports_vision=True,
            ),
            PluginModelConfig(
                id=f"{PREFIX}/Qwen-Ambassador/Qwen3.8-Max",
                name="Qwen3.8 Max",
                upstream_model="Qwen-Ambassador/Qwen3.8-Max",
                context_window=262144,
                max_tokens=32768,
                reasoning_levels=list(_HYBRID_LEVELS),
                supports_vision=True,
            ),
            PluginModelConfig(
                id=f"{PREFIX}/MiniMax/MiniMax-M3",
                name="MiniMax M3",
                upstream_model="MiniMax/MiniMax-M3",
                context_window=204800,
                max_tokens=32768,
                reasoning_levels=list(_ALWAYS_ON_LEVELS),
            ),
            PluginModelConfig(
                id=f"{PREFIX}/moonshotai/Kimi-K3.1-Pro",
                name="Kimi K3.1 Pro",
                upstream_model="moonshotai/Kimi-K3.1-Pro",
                context_window=262144,
                max_tokens=32768,
                reasoning_levels=list(_ALWAYS_ON_LEVELS),
            ),
            PluginModelConfig(
                id=f"{PREFIX}/OpenGVLab/InternVL3_5-241B-A28B",
                name="InternVL3.5 241B",
                upstream_model="OpenGVLab/InternVL3_5-241B-A28B",
                context_window=131072,
                max_tokens=16384,
                supports_vision=True,
            ),
        ]
        for model in models:
            model.supports_vision = model.supports_vision or infer_vision_support(
                model.id, model.name
            )

        self._metadata = PluginMetadata(
            id="modelscope",
            name="ModelScope",
            description=(
                "ModelScope API-Inference (Qwen, MiniMax, Kimi, InternVL) with "
                "normalized reasoning traces and thinking controls"
            ),
            version="1.0.0",
            author="LLMProxy",
            enabled=True,
            filename="modelscope.py",
            variables=[
                PluginVariable(
                    name="base_url",
                    label="Base URL",
                    type="string",
                    default=MODELSCOPE_BASE_URL,
                    description="ModelScope API-Inference base URL (international or cn domain)",
                ),
                PluginVariable(
                    name="timeout",
                    label="Timeout (seconds)",
                    type="number",
                    default=300,
                    description="Request timeout; reasoning models can think for minutes",
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
                    description="Cooldown after a 429 before the account is retried",
                ),
                PluginVariable(
                    name="account_switch_wait",
                    label="Switch Wait Delay (s)",
                    type="number",
                    default=0.1,
                    description="Delay before switching accounts on failover",
                ),
                PluginVariable(
                    name="max_retries",
                    label="Max Failover Retries",
                    type="number",
                    default=2,
                    description="Maximum account failovers per request",
                ),
                PluginVariable(
                    name="reasoning_effort",
                    label="Default Reasoning Effort",
                    type="select",
                    default="default",
                    options=["default", "none", "low", "medium", "high"],
                    description=(
                        "Used when a client sends no reasoning_effort. 'none' sends "
                        "enable_thinking=false instead of reasoning_effort."
                    ),
                ),
                PluginVariable(
                    name="split_think_tags",
                    label="Split Inline think Tags",
                    type="boolean",
                    default=True,
                    description=(
                        "Move think markup embedded in content into reasoning_content "
                        "(useful when a model's chat template is not parsed upstream)"
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
                    description="Strategy for selecting proxies from the pool",
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
            models=models,
        )

    @property
    def metadata(self) -> PluginMetadata:
        return self._metadata

    # ------------------------------------------------------------------ config
    @staticmethod
    def _is_test_mode(api_key: str, account: Optional[Account] = None) -> bool:
        key = (api_key or "").strip()
        if not key:
            if account is not None and not str(account.id).startswith("default-"):
                raise ValueError(
                    f"Account '{account.name}' has no API key configured for ModelScope."
                )
            return True
        return key.lower().startswith(
            ("test", "mock", "dummy", "fake", "offline", "sample")
        )

    @staticmethod
    def _strip_reasoning_suffix(model_id: str) -> Tuple[str, Optional[str]]:
        lowered = model_id.lower()
        for suffix in _REASONING_SUFFIXES:
            if lowered.endswith(suffix):
                return model_id[: -len(suffix)], suffix[1:]
        return model_id, None

    def _resolve_level(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        variables: Dict[str, Any],
    ) -> Optional[str]:
        """Return ``none``/``low``/``medium``/``high`` for the upstream control."""
        extra = request.extra_body if isinstance(request.extra_body, dict) else {}
        raw: Any = request.reasoning_effort
        if raw is None:
            raw = extra.get("reasoning_effort")
        if raw is None:
            reasoning = extra.get("reasoning")
            if isinstance(reasoning, dict):
                raw = reasoning.get("effort")
            elif reasoning:
                raw = reasoning
        if raw is None:
            _, suffix_level = self._strip_reasoning_suffix(request.model or "")
            raw = suffix_level
        if raw is None:
            raw = variables.get("reasoning_effort")

        level = str(raw or "").strip().lower()
        if not level or level in ("default", "auto"):
            return None
        if level in _NONE_LEVELS:
            return "none"
        return _LEVEL_ALIASES.get(level)

    @staticmethod
    def _apply_cache_fields(body: Dict[str, Any], request: ChatCompletionRequest) -> None:
        """Forward prompt-cache routing hints the gateway may honour."""
        extra = request.extra_body if isinstance(request.extra_body, dict) else {}
        for key in ("prompt_cache_key", "prompt_cache_retention"):
            value = getattr(request, key, None)
            if value is None:
                value = extra.get(key)
            if value is not None:
                body[key] = value
        template_kwargs = extra.get("chat_template_kwargs")
        if isinstance(template_kwargs, dict):
            body["chat_template_kwargs"] = template_kwargs

    @staticmethod
    def _outbound_messages(request: ChatCompletionRequest) -> List[Dict[str, Any]]:
        """Serialize history, dropping reasoning traces the upstream never needs."""
        messages: List[Dict[str, Any]] = []
        for message in request.messages:
            payload = message.model_dump(exclude_none=True)
            payload.pop("reasoning_content", None)
            payload.pop("reasoning", None)
            messages.append(payload)
        return messages

    def _build_body(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        variables: Dict[str, Any],
        include_thinking_controls: bool = True,
    ) -> Dict[str, Any]:
        upstream = model_config.upstream_model or model_config.id
        target_model, _ = self._strip_reasoning_suffix(upstream)

        body: Dict[str, Any] = {
            "model": target_model,
            "messages": self._outbound_messages(request),
            "stream": bool(request.stream),
        }

        if request.temperature is not None:
            body["temperature"] = request.temperature
        max_tokens = (
            request.max_tokens
            if request.max_tokens is not None
            else request.max_completion_tokens
        )
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if request.top_p is not None:
            body["top_p"] = request.top_p
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

        extra = request.extra_body if isinstance(request.extra_body, dict) else {}
        if isinstance(extra.get("enable_thinking"), bool):
            body["enable_thinking"] = extra["enable_thinking"]

        if include_thinking_controls:
            level = self._resolve_level(request, model_config, variables)
            if level == "none":
                # ModelScope rejects reasoning_effort="none"; the hybrid toggle is
                # the documented way to turn thinking off.
                body["enable_thinking"] = False
            elif level:
                body["reasoning_effort"] = level

        if request.stream:
            stream_options = dict(request.stream_options or {})
            stream_options["include_usage"] = True
            body["stream_options"] = stream_options

        self._apply_cache_fields(body, request)

        # Forward unrelated vendor extras the client knowingly supplied.
        for key, value in extra.items():
            if key in body or key in (
                "enable_thinking",
                "reasoning",
                "reasoning_effort",
                "chat_template_kwargs",
                "prompt_cache_key",
                "prompt_cache_retention",
            ):
                continue
            body.setdefault(key, value)
        return body

    @staticmethod
    def _has_thinking_controls(body: Dict[str, Any]) -> bool:
        return "reasoning_effort" in body or "enable_thinking" in body

    @staticmethod
    def _without_thinking_controls(body: Dict[str, Any]) -> Dict[str, Any]:
        stripped = dict(body)
        stripped.pop("reasoning_effort", None)
        stripped.pop("enable_thinking", None)
        return stripped

    @staticmethod
    def _thinking_retry_needed(status_code: int, raw: str) -> bool:
        """Always-thinking models reject enable_thinking/reasoning_effort with 400."""
        if status_code != 400:
            return False
        lowered = raw.lower()
        return any(
            marker in lowered
            for marker in ("enable_thinking", "reasoning_effort", "reasoning", "thinking")
        )

    # ------------------------------------------------------------------ usage
    @staticmethod
    def _usage_from(raw: Any) -> Optional[CompletionUsage]:
        if not isinstance(raw, dict) or not raw:
            return None
        try:
            prompt = int(raw.get("prompt_tokens") or 0)
            completion = int(raw.get("completion_tokens") or 0)
            total = int(raw.get("total_tokens") or (prompt + completion))
        except (TypeError, ValueError):
            return None

        completion_details = dict(raw.get("completion_tokens_details") or {})
        prompt_details = dict(raw.get("prompt_tokens_details") or {})

        reasoning = completion_details.get("reasoning_tokens")
        if reasoning is None:
            text_tokens = completion_details.get("text_tokens")
            if text_tokens is not None:
                try:
                    reasoning = max(0, completion - int(text_tokens))
                except (TypeError, ValueError):
                    reasoning = None
        if reasoning is not None:
            try:
                completion_details["reasoning_tokens"] = max(0, int(reasoning))
            except (TypeError, ValueError):
                completion_details.pop("reasoning_tokens", None)

        return CompletionUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            prompt_tokens_details=prompt_details or None,
            completion_tokens_details=completion_details or None,
        )

    # ------------------------------------------------------------------ shapes
    @staticmethod
    def _normalize_tool_calls(raw: Any) -> Optional[List[Dict[str, Any]]]:
        if not isinstance(raw, list) or not raw:
            return None
        calls: List[Dict[str, Any]] = []
        for call in raw:
            if not isinstance(call, dict):
                continue
            function = call.get("function") or {}
            arguments = function.get("arguments")
            if arguments is None:
                arguments = "{}"
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False)
            calls.append(
                {
                    "id": call.get("id") or f"call_ms_{uuid.uuid4().hex[:10]}",
                    "type": call.get("type") or "function",
                    "function": {
                        "name": function.get("name") or "unknown_tool",
                        "arguments": arguments,
                    },
                }
            )
        return calls or None

    def _normalize_response(
        self,
        data: Dict[str, Any],
        model_config: PluginModelConfig,
        split_tags: bool,
    ) -> ChatCompletionResponse:
        choices: List[ChatChoice] = []
        for position, choice in enumerate(data.get("choices") or []):
            message = choice.get("message") or {}
            content = message.get("content")
            reasoning = message.get("reasoning_content") or message.get("reasoning") or ""

            if isinstance(content, list):
                pass  # multimodal assistant content is passed through untouched
            elif content is None:
                content = ""
            elif split_tags and isinstance(content, str):
                content, inline_reasoning = _split_think_text(content)
                if inline_reasoning and not reasoning:
                    reasoning = inline_reasoning

            tool_calls = self._normalize_tool_calls(
                message.get("tool_calls") or message.get("function_calls")
            )

            payload: Dict[str, Any] = {
                "role": message.get("role") or "assistant",
                "content": content,
            }
            if tool_calls:
                payload["tool_calls"] = tool_calls
                if payload["content"] == "":
                    payload["content"] = None
            if reasoning:
                payload["reasoning_content"] = reasoning

            finish_reason = choice.get("finish_reason") or (
                "tool_calls" if tool_calls else "stop"
            )
            choices.append(
                ChatChoice(
                    index=choice.get("index", position),
                    message=ChatMessage.model_validate(payload),
                    finish_reason=finish_reason,
                )
            )

        if not choices:
            raise ModelScopeAPIError(502, "upstream returned no choices")

        return ChatCompletionResponse(
            id=data.get("id") or f"chatcmpl-ms-{uuid.uuid4().hex[:12]}",
            created=data.get("created") or int(time.time()),
            model=model_config.id,
            choices=choices,
            usage=self._usage_from(data.get("usage")),
            system_fingerprint=data.get("system_fingerprint") or None,
        )

    # ------------------------------------------------------------------ errors
    @staticmethod
    def _upstream_error(status_code: int, raw: str) -> ModelScopeAPIError:
        message = raw.strip()
        request_id = None
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            error = parsed.get("error")
            if isinstance(error, dict):
                message = str(error.get("message") or message)
                request_id = error.get("request_id") or parsed.get("request_id")
            elif error:
                message = str(error)
            elif parsed.get("message"):
                message = str(parsed["message"])
        return ModelScopeAPIError(status_code, message[:600], request_id)

    # --------------------------------------------------------------- simulate
    def _mock_response(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        account: Optional[Account] = None,
    ) -> ChatCompletionResponse:
        model_name = model_config.id
        label = f" via account '{account.name}'" if account else ""

        if request.tools and not any(m.role == "tool" for m in request.messages[-2:]):
            tool = request.tools[0]
            function = tool.get("function", tool)
            properties = (function.get("parameters") or {}).get("properties") or {}
            arguments = (
                json.dumps({next(iter(properties)): "test_value"})
                if properties
                else "{}"
            )
            return ChatCompletionResponse(
                id=f"chatcmpl-ms-{uuid.uuid4().hex[:12]}",
                model=model_name,
                choices=[
                    ChatChoice(
                        index=0,
                        message=ChatMessage.model_validate(
                            {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": f"call_ms_{uuid.uuid4().hex[:8]}",
                                        "type": "function",
                                        "function": {
                                            "name": function.get("name", "test_tool"),
                                            "arguments": arguments,
                                        },
                                    }
                                ],
                            }
                        ),
                        finish_reason="tool_calls",
                    )
                ],
                usage=CompletionUsage(
                    prompt_tokens=18, completion_tokens=24, total_tokens=42
                ),
            )

        content = (
            f"Simulated ModelScope response ({model_name}){label}. "
            "Add a ModelScope access token in the Accounts tab to reach the upstream API."
        )
        return ChatCompletionResponse(
            id=f"chatcmpl-ms-{uuid.uuid4().hex[:12]}",
            model=model_name,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage.model_validate(
                        {
                            "role": "assistant",
                            "content": content,
                            "reasoning_content": (
                                "Simulated reasoning trace: the request was routed "
                                "through ModelScope simulation mode."
                            ),
                        }
                    ),
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=18,
                completion_tokens=24,
                total_tokens=42,
                completion_tokens_details={"reasoning_tokens": 14, "text_tokens": 10},
            ),
        )

    async def _mock_stream(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        account: Optional[Account] = None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        model_name = model_config.id
        chunk_id = f"chatcmpl-ms-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        label = f" via account '{account.name}'" if account else ""

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
        for piece in ("Simulated ", "reasoning ", "trace."):
            yield ChatCompletionChunk(
                id=chunk_id,
                created=created,
                model=model_name,
                choices=[
                    ChatChoiceDelta(
                        index=0, delta={"reasoning_content": piece}, finish_reason=None
                    )
                ],
            )
        for piece in ("Simulated ", "ModelScope ", "stream ", f"for {model_name}{label}."):
            yield ChatCompletionChunk(
                id=chunk_id,
                created=created,
                model=model_name,
                choices=[
                    ChatChoiceDelta(index=0, delta={"content": piece}, finish_reason=None)
                ],
            )
        yield ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=model_name,
            choices=[ChatChoiceDelta(index=0, delta={}, finish_reason="stop")],
            usage=CompletionUsage(
                prompt_tokens=18,
                completion_tokens=24,
                total_tokens=42,
                completion_tokens_details={"reasoning_tokens": 14, "text_tokens": 10},
            ),
        )

    # ----------------------------------------------------------------- stream
    async def _stream_request(
        self,
        url: str,
        headers: Dict[str, str],
        body: Dict[str, Any],
        model_config: PluginModelConfig,
        split_tags: bool,
        proxy: Optional[str],
        timeout: float,
    ) -> AsyncIterator[ChatCompletionChunk]:
        chunk_id = f"chatcmpl-ms-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        splitter = _ThinkTagSplitter(split_tags)
        usage: Optional[CompletionUsage] = None
        finish_reason: Optional[str] = None
        role_sent = False

        async with httpx.AsyncClient(timeout=timeout, proxy=proxy) as client:
            request_cm = client.stream("POST", url, headers=headers, json=body)
            response = await request_cm.__aenter__()
            try:
                if response.status_code >= 400:
                    raw = (await response.aread()).decode("utf-8", "replace")
                    if (
                        self._thinking_retry_needed(response.status_code, raw)
                        and self._has_thinking_controls(body)
                    ):
                        await request_cm.__aexit__(None, None, None)
                        request_cm = client.stream(
                            "POST",
                            url,
                            headers=headers,
                            json=self._without_thinking_controls(body),
                        )
                        response = await request_cm.__aenter__()
                        if response.status_code >= 400:
                            raw = (await response.aread()).decode("utf-8", "replace")
                            raise self._upstream_error(response.status_code, raw)
                    else:
                        raise self._upstream_error(response.status_code, raw)

                def chunk(
                    delta: Dict[str, Any],
                    finish: Optional[str] = None,
                    chunk_usage: Optional[CompletionUsage] = None,
                ) -> ChatCompletionChunk:
                    return ChatCompletionChunk(
                        id=chunk_id,
                        created=created,
                        model=model_config.id,
                        choices=[
                            ChatChoiceDelta(index=0, delta=delta, finish_reason=finish)
                        ]
                        if (delta or finish)
                        else [],
                        usage=chunk_usage,
                    )

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or line.startswith(":") or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        parsed = json.loads(data)
                    except Exception:
                        continue
                    if not isinstance(parsed, dict):
                        continue
                    if parsed.get("error"):
                        raise self._upstream_error(502, json.dumps(parsed))

                    maybe_usage = self._usage_from(parsed.get("usage"))
                    if maybe_usage is not None:
                        usage = maybe_usage

                    for choice in parsed.get("choices") or []:
                        if not isinstance(choice, dict):
                            continue
                        # ModelScope mirrors an empty ``message`` onto every chunk;
                        # it is not part of the OpenAI chunk schema, so ignore it.
                        delta = choice.get("delta") or {}
                        if not isinstance(delta, dict):
                            delta = {}

                        if not role_sent:
                            role_sent = True
                            yield chunk({"role": "assistant", "content": ""})

                        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                        if isinstance(reasoning, str) and reasoning:
                            yield chunk({"reasoning_content": reasoning})

                        piece = delta.get("content")
                        if isinstance(piece, str) and piece:
                            tag_reasoning, tag_content = splitter.feed(piece)
                            if tag_reasoning:
                                yield chunk({"reasoning_content": tag_reasoning})
                            if tag_content:
                                yield chunk({"content": tag_content})

                        calls = delta.get("tool_calls") or delta.get("function_calls")
                        if isinstance(calls, list) and calls:
                            normalized_calls = []
                            for position, call in enumerate(calls):
                                if not isinstance(call, dict):
                                    continue
                                function = call.get("function") or {}
                                entry: Dict[str, Any] = {
                                    "index": call.get("index", position),
                                    "type": call.get("type") or "function",
                                    "function": {},
                                }
                                if call.get("id"):
                                    entry["id"] = call["id"]
                                if function.get("name"):
                                    entry["function"]["name"] = function["name"]
                                arguments = function.get("arguments")
                                if isinstance(arguments, str):
                                    entry["function"]["arguments"] = arguments
                                elif isinstance(arguments, dict):
                                    entry["function"]["arguments"] = json.dumps(
                                        arguments, ensure_ascii=False
                                    )
                                normalized_calls.append(entry)
                            if normalized_calls:
                                yield chunk({"tool_calls": normalized_calls})

                        candidate = choice.get("finish_reason")
                        if candidate:
                            finish_reason = candidate

                if not role_sent:
                    yield chunk({"role": "assistant", "content": ""})

                leftover_reasoning, leftover_content = splitter.flush()
                if leftover_reasoning:
                    yield chunk({"reasoning_content": leftover_reasoning})
                if leftover_content:
                    yield chunk({"content": leftover_content})

                yield chunk({}, finish=finish_reason or "stop", chunk_usage=usage)
            finally:
                await request_cm.__aexit__(None, None, None)

    # ------------------------------------------------------------------ entry
    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        variables: Dict[str, Any],
        account: Optional[Account] = None,
    ) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionChunk]]:
        api_key = (account.api_key if account else "") or ""
        base_url = (
            account.base_url
            if (account and account.base_url)
            else str(variables.get("base_url") or MODELSCOPE_BASE_URL)
        ).rstrip("/")

        if self._is_test_mode(api_key, account):
            if request.stream:
                return self._mock_stream(request, model_config, account=account)
            return self._mock_response(request, model_config, account=account)

        split_tags = bool(variables.get("split_think_tags", True))
        proxy = str(
            variables.get("proxy") or (getattr(account, "proxy_url", None) or "")
        ).strip() or None
        timeout = float(variables.get("timeout") or 300.0)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        body = self._build_body(request, model_config, variables)
        url = f"{base_url}/chat/completions"

        if request.stream:
            return self._stream_request(
                url,
                headers,
                body,
                model_config,
                split_tags=split_tags,
                proxy=proxy,
                timeout=timeout,
            )

        async with httpx.AsyncClient(timeout=timeout, proxy=proxy) as client:
            response = await client.post(url, headers=headers, json=body)
            if response.status_code >= 400:
                raw = response.text
                if (
                    self._thinking_retry_needed(response.status_code, raw)
                    and self._has_thinking_controls(body)
                ):
                    retry = await client.post(
                        url,
                        headers=headers,
                        json=self._without_thinking_controls(body),
                    )
                    if retry.status_code >= 400:
                        raise self._upstream_error(retry.status_code, retry.text)
                    data = retry.json()
                else:
                    raise self._upstream_error(response.status_code, raw)
            else:
                data = response.json()

        return self._normalize_response(data, model_config, split_tags)


plugin = ModelScopePlugin()
