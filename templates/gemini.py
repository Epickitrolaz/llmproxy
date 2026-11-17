"""
Google Gemini Provider Plugin
=============================

Native integration with the public Google Gemini API
(``https://generativelanguage.googleapis.com/v1beta/models``).

The request/response translation mirrors the Gemini provider in 9Router and the
Google Cloud Code plugin in ``templates/antigravity.py``, while targeting the
public API instead of Antigravity's internal Cloud Code envelope.

Highlights
----------
- API-key auth (``x-goog-api-key``), OAuth bearer tokens, or query-string keys.
- Streaming (``:streamGenerateContent?alt=sse``) and non-streaming requests.
- OpenAI tool calling -> Gemini ``functionDeclarations`` + ``functionResponse``.
- Gemini thinking controls for Gemini 4/3 (``thinkingLevel``) and 2.5
  (``thinkingBudget``), exposed through ``reasoning_effort``.
- Text, image, and inline audio/file parts.
- Explicit cache references (``cachedContent``) and cache-aware usage mapping.
- Usage tracking from ``usageMetadata`` (including cached and reasoning tokens).
- Configurable safety thresholds, custom headers, proxy routing, and simulation
  mode for offline tests.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union
from urllib.parse import quote

import httpx

from llmproxy.core.interface import BasePlugin
from llmproxy.core.multimodal import (
    content_text,
    is_image_part,
    materialize_message_images,
    messages_have_images,
    parse_data_url,
    to_gemini_part,
)
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

GEMINI_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"
GEMINI_OAUTH_TOKEN_URI = "https://oauth2.googleapis.com/token"

SAFETY_CATEGORIES = (
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
    "HARM_CATEGORY_CIVIC_INTEGRITY",
)

GEMINI4_REASONING_LEVELS = ("low", "medium", "high")
GEMINI3_REASONING_LEVELS = ("minimal", "low", "medium", "high")
GEMINI25_REASONING_LEVELS = ("none", "low", "medium", "high")

_LEVEL_TO_BUDGET: Dict[str, int] = {
    "minimal": 512,
    "low": 1024,
    "medium": 8192,
    "high": 24576,
    "xhigh": 32768,
    "max": 128000,
}

_FINISH_REASON_MAP: Dict[str, str] = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "BLOCKLIST": "content_filter",
    "PROHIBITED_CONTENT": "content_filter",
    "SPII": "content_filter",
    "IMAGE_SAFETY": "content_filter",
    "MALFORMED_FUNCTION_CALL": "tool_calls",
    "UNEXPECTED_TOOL_CALL": "tool_calls",
    "OTHER": "stop",
}

# Thought signatures are required when a Gemini 3 conversation continues after a
# tool call. OpenAI clients do not know about them, so the plugin caches them by
# tool-call id for later turns in the same process.
_THOUGHT_SIGNATURES: Dict[str, str] = {}
_THOUGHT_SIGNATURE_LIMIT = 2000


def _remember_thought_signature(call_id: Optional[str], signature: Optional[str]) -> None:
    if not call_id or not signature:
        return
    _THOUGHT_SIGNATURES[call_id] = signature
    if len(_THOUGHT_SIGNATURES) > _THOUGHT_SIGNATURE_LIMIT:
        for key in list(_THOUGHT_SIGNATURES.keys())[: _THOUGHT_SIGNATURE_LIMIT // 2]:
            _THOUGHT_SIGNATURES.pop(key, None)


def _recall_thought_signature(call_id: Optional[str]) -> Optional[str]:
    if not call_id:
        return None
    return _THOUGHT_SIGNATURES.get(call_id)


def _sanitize_function_name(name: Optional[str]) -> str:
    """Gemini requires ``[a-zA-Z_][a-zA-Z0-9_.:-]{0,63}`` for function names."""
    sanitized = re.sub(r"[^a-zA-Z0-9_.:\-]", "_", str(name or "_unknown"))
    if not re.match(r"^[a-zA-Z_]", sanitized):
        sanitized = "_" + sanitized
    return sanitized[:64] or "_unknown"


def _json_object(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return {}


def _clean_schema(schema: Any, depth: int = 0) -> Dict[str, Any]:
    """Normalise a JSON Schema for Gemini's subset of OpenAPI schema fields."""
    if not isinstance(schema, dict) or depth > 20:
        return {"type": "STRING"}

    if "const" in schema and "enum" not in schema:
        schema = dict(schema)
        schema["enum"] = [schema.pop("const")]

    variants = schema.get("anyOf") or schema.get("oneOf")
    if isinstance(variants, list) and variants:
        chosen = next(
            (
                variant
                for variant in variants
                if isinstance(variant, dict)
                and str(variant.get("type") or "").lower() != "null"
            ),
            None,
        )
        if chosen is not None:
            merged = {
                key: value
                for key, value in schema.items()
                if key not in ("anyOf", "oneOf")
            }
            merged.update(chosen)
            return _clean_schema(merged, depth + 1)

    type_map = {
        "string": "STRING",
        "number": "NUMBER",
        "integer": "INTEGER",
        "boolean": "BOOLEAN",
        "array": "ARRAY",
        "object": "OBJECT",
        "null": "STRING",
    }

    out: Dict[str, Any] = {}
    raw_type = schema.get("type")
    if isinstance(raw_type, list):
        types = [item for item in raw_type if item != "null"]
        if "null" in raw_type:
            out["nullable"] = True
        raw_type = types[0] if types else "string"

    if raw_type is not None:
        out["type"] = type_map.get(str(raw_type).lower(), str(raw_type).upper())

    for key in (
        "format",
        "description",
        "nullable",
        "pattern",
        "minItems",
        "maxItems",
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
        "minProperties",
        "maxProperties",
        "default",
    ):
        if key in schema:
            out[key] = schema[key]

    if "exclusiveMinimum" in schema and "minimum" not in out:
        value = schema["exclusiveMinimum"]
        if isinstance(value, (int, float)):
            out["minimum"] = value + 1 if isinstance(value, int) else value
    if "exclusiveMaximum" in schema and "maximum" not in out:
        value = schema["exclusiveMaximum"]
        if isinstance(value, (int, float)):
            out["maximum"] = value - 1 if isinstance(value, int) else value

    if isinstance(schema.get("enum"), list):
        out["enum"] = [str(item) for item in schema["enum"]]

    if isinstance(schema.get("properties"), dict):
        out["properties"] = {
            name: _clean_schema(child, depth + 1)
            for name, child in schema["properties"].items()
        }
        out.setdefault("type", "OBJECT")

    if isinstance(schema.get("items"), dict):
        out["items"] = _clean_schema(schema["items"], depth + 1)

    if out.get("type") == "OBJECT" and not isinstance(out.get("properties"), dict):
        out["properties"] = {}

    if isinstance(schema.get("required"), list) and isinstance(out.get("properties"), dict):
        required = [name for name in schema["required"] if name in out["properties"]]
        if required:
            out["required"] = required

    if not out:
        return {"type": "OBJECT", "properties": {}}
    return out


class _ToolIdMapper:
    """Keep Gemini function-call ids unique across the whole request history."""

    def __init__(self) -> None:
        self.used: set[str] = set()
        self.emitted: Dict[str, List[str]] = {}
        self.cursor: Dict[str, int] = {}

    def emit(self, original: Optional[str]) -> str:
        original = original or f"call_{uuid.uuid4().hex[:12]}"
        candidate = original
        counter = 2
        while candidate in self.used:
            candidate = f"{original}-{counter}"
            counter += 1
        self.used.add(candidate)
        self.emitted.setdefault(original, []).append(candidate)
        return candidate

    def resolve(self, original: Optional[str]) -> Optional[str]:
        if not original:
            return None
        ids = self.emitted.get(original) or []
        index = self.cursor.get(original, 0)
        if index < len(ids):
            self.cursor[original] = index + 1
            return ids[index]
        return original


class GeminiPlugin(BasePlugin):
    """Google Gemini provider plugin."""

    def __init__(self) -> None:
        self._metadata = PluginMetadata(
            id="gemini",
            name="Google Gemini",
            description=(
                "Google Gemini API with streaming, tool calling, multimodal input, "
                "thinking controls, prompt caching, and usage tracking"
            ),
            version="1.0.0",
            author="LLMProxy",
            enabled=True,
            filename="gemini.py",
            variables=[
                PluginVariable(
                    name="timeout",
                    label="Timeout (seconds)",
                    type="number",
                    default=60,
                    description="Request timeout in seconds",
                ),
                PluginVariable(
                    name="base_url",
                    label="API Base URL",
                    type="string",
                    default=GEMINI_DEFAULT_BASE_URL,
                    description="Gemini API host. Do not include /v1beta; use API Version below",
                ),
                PluginVariable(
                    name="api_version",
                    label="API Version",
                    type="select",
                    default="v1beta",
                    options=["v1beta", "v1"],
                    description="Gemini API version path segment",
                ),
                PluginVariable(
                    name="auth_style",
                    label="Authentication Style",
                    type="select",
                    default="auto",
                    options=["auto", "api_key", "bearer", "query"],
                    description=(
                        "auto uses x-goog-api-key unless the credential looks like an "
                        "OAuth access token; query appends ?key= (less secure)"
                    ),
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
                    default=0,
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
                    label="Default Reasoning Effort",
                    type="select",
                    default="auto",
                    options=["auto", "none", "minimal", "low", "medium", "high"],
                    description=(
                        "Default thinking level. Gemini 4 uses low/medium/high, "
                        "Gemini 3 adds minimal, Gemini 2.5 maps levels to a token budget"
                    ),
                ),
                PluginVariable(
                    name="thinking_budget",
                    label="Thinking Budget (2.5 only)",
                    type="number",
                    default=-1,
                    description=(
                        "Override Gemini 2.5 thinkingBudget. -1 lets the model decide, "
                        "0 disables thinking, positive values are token budgets"
                    ),
                ),
                PluginVariable(
                    name="include_thoughts",
                    label="Return Thinking Text",
                    type="boolean",
                    default=True,
                    description="Expose Gemini thought summaries as reasoning_content",
                ),
                PluginVariable(
                    name="safety_threshold",
                    label="Safety Threshold",
                    type="select",
                    default="OFF",
                    options=[
                        "OFF",
                        "BLOCK_NONE",
                        "BLOCK_ONLY_HIGH",
                        "BLOCK_MEDIUM_AND_ABOVE",
                        "BLOCK_LOW_AND_ABOVE",
                    ],
                    description="Default threshold applied to every Gemini safety category",
                ),
                PluginVariable(
                    name="safety_settings",
                    label="Safety Overrides (JSON)",
                    type="string",
                    default="",
                    description=(
                        "Optional JSON object mapping a safety category to a threshold, "
                        "for example {\"HARM_CATEGORY_HARASSMENT\": \"BLOCK_ONLY_HIGH\"}"
                    ),
                ),
                PluginVariable(
                    name="cached_content",
                    label="Default Cached Content",
                    type="string",
                    default="",
                    description=(
                        "Optional cachedContents/... resource used when the request does "
                        "not provide extra_body.cached_content"
                    ),
                ),
                PluginVariable(
                    name="media_resolution",
                    label="Media Resolution",
                    type="select",
                    default="auto",
                    options=["auto", "low", "medium", "high"],
                    description="Default resolution for image/video input parts",
                ),
                PluginVariable(
                    name="custom_headers",
                    label="Custom Headers (JSON)",
                    type="string",
                    default="{}",
                    description="JSON object of extra HTTP headers sent to Gemini",
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
                    description="Strategy for selecting proxies from the proxy pool",
                ),
                PluginVariable(
                    name="specific_proxy_id",
                    label="Specific Proxy ID / Tag",
                    type="string",
                    default="",
                    description="Target proxy ID, URL, or tag when strategy is 'specific'",
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
                    id="gm/gemini-4-argon",
                    name="Gemini 4 Argon",
                    upstream_model="gemini-4-argon",
                    context_window=1048576,
                    max_tokens=65536,
                ),
                PluginModelConfig(
                    id="gm/gemini-3.8-flash",
                    name="Gemini 3.8 Flash",
                    upstream_model="gemini-3.8-flash",
                    context_window=1048576,
                    max_tokens=65536,
                ),
                PluginModelConfig(
                    id="gm/gemini-3.7-flash",
                    name="Gemini 3.7 Flash",
                    upstream_model="gemini-3.7-flash",
                    context_window=1048576,
                    max_tokens=65536,
                ),
                PluginModelConfig(
                    id="gm/gemini-3.6-flash",
                    name="Gemini 3.6 Flash",
                    upstream_model="gemini-3.6-flash",
                    context_window=1048576,
                    max_tokens=65536,
                ),
                PluginModelConfig(
                    id="gm/gemini-3.5-flash-lite",
                    name="Gemini 3.5 Flash Lite",
                    upstream_model="gemini-3.5-flash-lite",
                    context_window=1048576,
                    max_tokens=65536,
                ),
                PluginModelConfig(
                    id="gm/gemini-3.1-pro-preview",
                    name="Gemini 3.1 Pro Preview",
                    upstream_model="gemini-3.1-pro-preview",
                    context_window=1048576,
                    max_tokens=65536,
                ),
                PluginModelConfig(
                    id="gm/gemini-3.1-flash-lite-preview",
                    name="Gemini 3.1 Flash Lite Preview",
                    upstream_model="gemini-3.1-flash-lite-preview",
                    context_window=1048576,
                    max_tokens=65536,
                ),
                PluginModelConfig(
                    id="gm/gemini-3-flash-preview",
                    name="Gemini 3 Flash Preview",
                    upstream_model="gemini-3-flash-preview",
                    context_window=1048576,
                    max_tokens=65536,
                ),
                PluginModelConfig(
                    id="gm/gemini-2.5-pro",
                    name="Gemini 2.5 Pro",
                    upstream_model="gemini-2.5-pro",
                    context_window=1048576,
                    max_tokens=65536,
                ),
                PluginModelConfig(
                    id="gm/gemini-2.5-flash",
                    name="Gemini 2.5 Flash",
                    upstream_model="gemini-2.5-flash",
                    context_window=1048576,
                    max_tokens=65536,
                ),
                PluginModelConfig(
                    id="gm/gemini-2.5-flash-lite",
                    name="Gemini 2.5 Flash Lite",
                    upstream_model="gemini-2.5-flash-lite",
                    context_window=1048576,
                    max_tokens=65536,
                ),
                PluginModelConfig(
                    id="gm/gemma-4-31b-it",
                    name="Gemma 4 31B IT",
                    upstream_model="gemma-4-31b-it",
                    context_window=128000,
                    max_tokens=8192,
                ),
            ],
        )
        for model in self._metadata.models:
            model.supports_vision = True
            upstream = (model.upstream_model or model.id).lower()
            if "gemini-4" in upstream:
                model.reasoning_levels = list(GEMINI4_REASONING_LEVELS)
            elif "gemini-3" in upstream:
                model.reasoning_levels = list(GEMINI3_REASONING_LEVELS)
            elif "gemini-2.5" in upstream:
                model.reasoning_levels = list(GEMINI25_REASONING_LEVELS)

    @property
    def metadata(self) -> PluginMetadata:
        return self._metadata

    # ------------------------------------------------------------------
    # Simulation mode
    # ------------------------------------------------------------------

    def _is_test_mode(self, api_key: str) -> bool:
        if not api_key:
            return True
        key = api_key.lower().strip()
        prefixes = ("test", "mock", "sk-test", "dummy", "fake", "offline", "sample")
        return any(key.startswith(prefix) for prefix in prefixes) or key in (
            "test-key",
            "test_key",
            "secret123",
        )

    def _details_footer(
        self,
        model_name: str,
        account: Optional[Account],
        prompt_tokens: int,
        completion_tokens: int,
    ) -> str:
        now_str = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        status = (
            "No account configured (Simulation Mode)"
            if (not account or account.id.startswith("default-") or not account.api_key)
            else f"Account '{account.name}' (Test mode)"
        )
        return (
            f"\n\n---\n"
            f"**Execution Details:**\n"
            f"• **Timestamp:** {now_str}\n"
            f"• **Model:** {model_name}\n"
            f"• **Tokens:** {prompt_tokens} prompt, {completion_tokens} completion "
            f"({prompt_tokens + completion_tokens} total)\n"
            f"• **Account status:** {status}"
        )

    def _mock_response(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        account: Optional[Account] = None,
    ) -> ChatCompletionResponse:
        model_name = request.model or model_config.id
        prompt_tokens = 15
        completion_tokens = 25

        if request.tools and not any(
            message.role in ("tool", "function") for message in request.messages[-2:]
        ):
            tool = request.tools[0]
            fn = tool.get("function", tool) if isinstance(tool, dict) else {}
            properties = (fn.get("parameters") or {}).get("properties") or {}
            arguments = (
                json.dumps({next(iter(properties)): "test_value"})
                if properties
                else "{}"
            )
            return ChatCompletionResponse(
                id=f"chatcmpl-gm-{uuid.uuid4().hex[:12]}",
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
                                    "id": f"call_gm_{uuid.uuid4().hex[:8]}",
                                    "type": "function",
                                    "function": {
                                        "name": fn.get("name", "test_tool"),
                                        "arguments": arguments,
                                    },
                                }
                            ],
                        ),
                        finish_reason="tool_calls",
                    )
                ],
                usage=CompletionUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
            )

        message = ChatMessage(
            role="assistant",
            content=(
                f"Simulated response from Google Gemini ({model_name})."
                + self._details_footer(
                    model_name, account, prompt_tokens, completion_tokens
                )
            ),
        )
        if self._mock_thoughts_enabled(request):
            message = message.model_copy(
                update={"reasoning_content": "Simulated Gemini reasoning."}
            )
        return ChatCompletionResponse(
            id=f"chatcmpl-gm-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=model_name,
            choices=[ChatChoice(index=0, message=message, finish_reason="stop")],
            usage=CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    @staticmethod
    def _mock_thoughts_enabled(request: ChatCompletionRequest) -> bool:
        extra = request.extra_body if isinstance(request.extra_body, dict) else {}
        return bool(extra.get("mock_thought") or getattr(request, "mock_thought", False))

    async def _mock_stream(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        account: Optional[Account] = None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        model_name = request.model or model_config.id
        chunk_id = f"chatcmpl-gm-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        prompt_tokens = 15
        completion_tokens = 25

        if request.tools and not any(
            message.role in ("tool", "function") for message in request.messages[-2:]
        ):
            tool = request.tools[0]
            fn = tool.get("function", tool) if isinstance(tool, dict) else {}
            properties = (fn.get("parameters") or {}).get("properties") or {}
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
                        delta={"role": "assistant", "content": ""},
                    )
                ],
            )
            if self._mock_thoughts_enabled(request):
                yield ChatCompletionChunk(
                    id=chunk_id,
                    created=created,
                    model=model_name,
                    choices=[
                        ChatChoiceDelta(
                            index=0,
                            delta={"reasoning_content": "Simulated Gemini reasoning."},
                        )
                    ],
                )
            call_id = f"call_gm_{uuid.uuid4().hex[:8]}"
            yield ChatCompletionChunk(
                id=chunk_id,
                created=created,
                model=model_name,
                choices=[
                    ChatChoiceDelta(
                        index=0,
                        delta={
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": call_id,
                                    "type": "function",
                                    "function": {
                                        "name": fn.get("name", "test_tool"),
                                        "arguments": "",
                                    },
                                }
                            ]
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
                choices=[ChatChoiceDelta(index=0, delta={}, finish_reason="tool_calls")],
                usage=CompletionUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
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
        if self._mock_thoughts_enabled(request):
            yield ChatCompletionChunk(
                id=chunk_id,
                created=created,
                model=model_name,
                choices=[
                    ChatChoiceDelta(
                        index=0,
                        delta={"reasoning_content": "Simulated Gemini reasoning."},
                    )
                ],
            )
        for token in ("Simulated ", "stream ", "response ", "from Google Gemini."):
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
            choices=[
                ChatChoiceDelta(
                    index=0,
                    delta={
                        "content": self._details_footer(
                            model_name, account, prompt_tokens, completion_tokens
                        )
                    },
                )
            ],
        )
        yield ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=model_name,
            choices=[ChatChoiceDelta(index=0, delta={}, finish_reason="stop")],
            usage=CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    # ------------------------------------------------------------------
    # Connection, auth, and URL
    # ------------------------------------------------------------------

    async def _ensure_token_freshness(
        self, account: Optional[Account], variables: Dict[str, Any]
    ) -> None:
        """Refresh an OAuth access token when the account provides refresh fields."""
        if not account:
            return
        extra = account.extra_config or {}
        refresh_token = extra.get("refresh_token")
        if not refresh_token:
            return

        expires_at = float(extra.get("expires_at") or 0)
        now = time.time()
        if account.api_key and (expires_at <= 0 or now < expires_at - 120):
            return

        client_id = extra.get("client_id")
        client_secret = extra.get("client_secret")
        if not client_id or not client_secret:
            return

        token_uri = str(extra.get("token_uri") or GEMINI_OAUTH_TOKEN_URI)
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    token_uri,
                    data={
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "refresh_token": refresh_token,
                        "grant_type": "refresh_token",
                    },
                )
                response.raise_for_status()
                payload = response.json()
        except Exception:
            return

        access_token = payload.get("access_token")
        if not access_token:
            return
        account.api_key = access_token
        account.extra_config["expires_at"] = now + int(payload.get("expires_in") or 3600)
        if payload.get("refresh_token"):
            account.extra_config["refresh_token"] = payload["refresh_token"]
        try:
            from llmproxy.core.manager import plugin_manager

            plugin_manager._save_accounts()
        except Exception:
            pass

    def _resolve_auth_style(
        self, api_key: str, variables: Dict[str, Any], account: Optional[Account]
    ) -> str:
        style = ""
        if account and isinstance(account.extra_config, dict):
            style = str(account.extra_config.get("auth_style") or "")
        style = style or str(variables.get("auth_style") or "auto")
        style = style.lower().strip()
        if style == "auto":
            token = (api_key or "").strip()
            style = "bearer" if token.startswith(("ya29.", "Bearer ")) else "api_key"
        return style

    def _build_headers(
        self,
        api_key: str,
        variables: Dict[str, Any],
        account: Optional[Account],
        stream: bool,
    ) -> Dict[str, str]:
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if stream:
            headers["Accept"] = "text/event-stream"

        custom = _json_object(variables.get("custom_headers"))
        if account and isinstance(account.extra_config, dict):
            account_headers = account.extra_config.get("headers")
            if isinstance(account_headers, dict):
                custom.update(account_headers)
        for name, value in custom.items():
            headers[str(name)] = str(value)

        style = self._resolve_auth_style(api_key, variables, account)
        key = (api_key or "").strip()
        if key.lower().startswith("bearer "):
            key = key[7:].strip()

        if style == "bearer" and key:
            headers["Authorization"] = f"Bearer {key}"
        elif style == "api_key" and key:
            headers["x-goog-api-key"] = key
        return headers

    def _build_url(
        self,
        variables: Dict[str, Any],
        account: Optional[Account],
        model: str,
        stream: bool,
        api_key: str,
    ) -> str:
        base = str(
            (account.base_url if account and account.base_url else None)
            or variables.get("base_url")
            or GEMINI_DEFAULT_BASE_URL
        ).rstrip("/")
        api_version = str(variables.get("api_version") or "v1beta").strip().strip("/")
        endpoint = "streamGenerateContent?alt=sse" if stream else "generateContent"

        if "{model}" in base:
            url = base.replace("{model}", model)
            if stream:
                url = re.sub(
                    r":generateContent(?=\?|$)", ":streamGenerateContent", url
                )
                if (
                    ":streamGenerateContent" not in url
                    and ":generateContent" not in url
                ):
                    url = f"{url}:streamGenerateContent"
                if "alt=sse" not in url:
                    url = f"{url}{'&' if '?' in url else '?'}alt=sse"
            else:
                url = re.sub(
                    r":streamGenerateContent(?=\?|$)", ":generateContent", url
                )
                if (
                    ":generateContent" not in url
                    and ":streamGenerateContent" not in url
                ):
                    url = f"{url}:generateContent"
                url = re.sub(r"[?&]alt=sse", "", url)
        elif base.endswith(":generateContent") or base.endswith(":streamGenerateContent"):
            url = base
            if stream:
                url = re.sub(r":generateContent(?=\?|$)", ":streamGenerateContent", url)
                if "alt=sse" not in url:
                    url = f"{url}{'&' if '?' in url else '?'}alt=sse"
            else:
                url = re.sub(r":streamGenerateContent(?=\?|$)", ":generateContent", url)
                url = re.sub(r"[?&]alt=sse", "", url)
        else:
            if not re.search(r"/v1(?:beta)?$", base):
                base = f"{base}/{api_version}"
            if base.endswith("/models"):
                url = f"{base}/{model}:{endpoint}"
            else:
                url = f"{base}/models/{model}:{endpoint}"

        style = self._resolve_auth_style(api_key, variables, account)
        if style == "query" and api_key:
            url = f"{url}{'&' if '?' in url else '?'}key={quote(api_key, safe='')}"
        return url

    # ------------------------------------------------------------------
    # Request translation
    # ------------------------------------------------------------------

    def _reasoning_family(self, model: str) -> Optional[str]:
        name = model.lower()
        if "gemma" in name:
            return None
        if "gemini-2.5" in name:
            return "budget"
        if "gemini-4" in name or "gemini-3" in name:
            return "level"
        return None

    @staticmethod
    def _resolve_reasoning_effort(
        request: ChatCompletionRequest, variables: Dict[str, Any]
    ) -> str:
        raw = ""
        if request.reasoning_effort:
            raw = str(request.reasoning_effort)
        extra = request.extra_body if isinstance(request.extra_body, dict) else {}
        if not raw:
            raw = str(extra.get("reasoning_effort") or "")
        if not raw and isinstance(extra.get("reasoning"), dict):
            raw = str(extra["reasoning"].get("effort") or "")
        if not raw and isinstance(extra.get("reasoning"), str):
            raw = extra["reasoning"]
        if not raw and request.model:
            match = re.search(
                r"[:\-](none|minimal|low|medium|high|xhigh|max)$",
                request.model.lower(),
            )
            if match:
                raw = match.group(1)
        if not raw:
            raw = str(variables.get("reasoning_effort") or "")
        raw = raw.strip().lower()
        return "" if raw in ("", "auto", "default") else raw

    def _build_thinking_config(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        variables: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        model = model_config.upstream_model or model_config.id
        family = self._reasoning_family(model)
        if not family:
            return None

        extra = request.extra_body if isinstance(request.extra_body, dict) else {}
        native = extra.get("thinkingConfig") or extra.get("thinking_config")
        if isinstance(native, dict):
            return native

        effort = self._resolve_reasoning_effort(request, variables)
        include_thoughts = bool(variables.get("include_thoughts", True))
        try:
            budget_override = int(variables.get("thinking_budget", -1))
        except (TypeError, ValueError):
            budget_override = -1

        if family == "level":
            if effort in ("none", "off", "disabled", "false", "0"):
                level = "low" if "gemini-4" in model.lower() else "minimal"
            elif effort in ("", "auto"):
                level = "high"
            elif effort in ("xhigh", "max"):
                level = "high"
            else:
                level = effort
            if "gemini-4" in model.lower() and level == "minimal":
                level = "low"
            if level not in {"minimal", "low", "medium", "high"}:
                level = "high"
            return {
                "thinkingLevel": level,
                "includeThoughts": include_thoughts and level != "minimal",
            }

        if effort in ("none", "off", "disabled", "false", "0"):
            return {"thinkingBudget": 0, "includeThoughts": False}
        if budget_override >= 0:
            budget = budget_override
        elif effort in ("", "auto"):
            return {"includeThoughts": include_thoughts}
        else:
            budget = _LEVEL_TO_BUDGET.get(effort, -1)
        if budget == 0:
            return {"thinkingBudget": 0, "includeThoughts": False}
        if budget > 24576:
            budget = 24576
        return {"thinkingBudget": budget, "includeThoughts": include_thoughts}

    def _build_generation_config(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        variables: Dict[str, Any],
    ) -> Dict[str, Any]:
        extra = request.extra_body if isinstance(request.extra_body, dict) else {}
        target_model = (model_config.upstream_model or model_config.id).lower()
        # Gemini 4 removed sampling controls; forwarding them fails the request.
        allow_sampling = "gemini-4" not in target_model
        max_tokens = (
            request.max_tokens
            if request.max_tokens is not None
            else request.max_completion_tokens
        )
        if max_tokens is None:
            max_tokens = model_config.max_tokens
        config: Dict[str, Any] = {}
        if max_tokens is not None:
            config["maxOutputTokens"] = int(max_tokens)
        if allow_sampling and request.temperature is not None and request.temperature != 1.0:
            config["temperature"] = request.temperature
        if allow_sampling and request.top_p is not None and request.top_p != 1.0:
            config["topP"] = request.top_p
        if request.n is not None and request.n > 1:
            config["candidateCount"] = int(request.n)
        if allow_sampling and request.stop is not None:
            config["stopSequences"] = (
                [request.stop] if isinstance(request.stop, str) else list(request.stop)
            )
        if allow_sampling and request.presence_penalty:
            config["presencePenalty"] = request.presence_penalty
        if allow_sampling and request.frequency_penalty:
            config["frequencyPenalty"] = request.frequency_penalty

        top_k = extra.get("top_k") or extra.get("topK")
        if allow_sampling and top_k is not None:
            config["topK"] = int(top_k)
        seed = extra.get("seed")
        if seed is not None:
            config["seed"] = int(seed)

        response_format = request.response_format or {}
        response_type = response_format.get("type") if isinstance(response_format, dict) else None
        if response_type == "json_object":
            config["responseMimeType"] = "application/json"
        elif response_type == "json_schema":
            json_schema = response_format.get("json_schema") or {}
            schema = json_schema.get("schema") if isinstance(json_schema, dict) else None
            config["responseMimeType"] = "application/json"
            if isinstance(schema, dict):
                config["responseSchema"] = _clean_schema(schema)

        if "response_mime_type" in extra:
            config["responseMimeType"] = extra["response_mime_type"]
        if isinstance(extra.get("response_schema"), dict):
            config["responseSchema"] = _clean_schema(extra["response_schema"])

        thinking = self._build_thinking_config(request, model_config, variables)
        if thinking:
            config["thinkingConfig"] = thinking

        media_resolution = str(
            extra.get("media_resolution")
            or extra.get("mediaResolution")
            or variables.get("media_resolution")
            or "auto"
        ).lower()
        if media_resolution in ("low", "medium", "high"):
            config["mediaResolution"] = f"MEDIA_RESOLUTION_{media_resolution.upper()}"
        return config

    def _build_safety_settings(self, variables: Dict[str, Any]) -> List[Dict[str, str]]:
        threshold = str(variables.get("safety_threshold") or "OFF").upper()
        overrides = _json_object(variables.get("safety_settings"))
        settings: List[Dict[str, str]] = []
        for category in SAFETY_CATEGORIES:
            category_threshold = str(overrides.get(category, threshold)).upper()
            settings.append(
                {"category": category, "threshold": category_threshold or "OFF"}
            )
        return settings

    @staticmethod
    def _convert_tools(
        tools: Optional[List[Dict[str, Any]]],
        functions: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Optional[List[Dict[str, Any]]], Dict[str, str]]:
        declarations: List[Dict[str, Any]] = []
        name_map: Dict[str, str] = {}
        candidates = list(tools or [])
        for function in functions or []:
            candidates.append({"type": "function", "function": function})

        for tool in candidates:
            if not isinstance(tool, dict):
                continue
            fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
            name = fn.get("name")
            if not name:
                continue
            sanitized = _sanitize_function_name(name)
            name_map[sanitized] = str(name)
            declaration: Dict[str, Any] = {
                "name": sanitized,
                "description": str(fn.get("description") or ""),
                "parameters": _clean_schema(
                    fn.get("parameters")
                    or fn.get("input_schema")
                    or {"type": "object", "properties": {}}
                ),
            }
            declarations.append(declaration)

        if not declarations:
            return None, name_map
        return [{"functionDeclarations": declarations}], name_map

    @staticmethod
    def _convert_tool_choice(
        tool_choice: Union[str, Dict[str, Any], None],
        name_map: Dict[str, str],
    ) -> Optional[Dict[str, Any]]:
        if not tool_choice or tool_choice == "auto":
            return {"functionCallingConfig": {"mode": "AUTO"}}
        if tool_choice == "none":
            return {"functionCallingConfig": {"mode": "NONE"}}
        if tool_choice == "required":
            return {"functionCallingConfig": {"mode": "ANY"}}
        if isinstance(tool_choice, dict):
            if tool_choice.get("type") == "function":
                name = (tool_choice.get("function") or {}).get("name")
            else:
                name = tool_choice.get("name")
            if name:
                sanitized = _sanitize_function_name(name)
                if sanitized not in name_map:
                    name_map[sanitized] = str(name)
                return {
                    "functionCallingConfig": {
                        "mode": "ANY",
                        "allowedFunctionNames": [sanitized],
                    }
                }
        return None

    @staticmethod
    def _content_parts(content: Any) -> List[Dict[str, Any]]:
        parts: List[Dict[str, Any]] = []
        if isinstance(content, str):
            return [{"text": content}]
        if not isinstance(content, list):
            return parts

        for part in content:
            if isinstance(part, str):
                parts.append({"text": part})
                continue
            if not isinstance(part, dict):
                continue
            if is_image_part(part):
                image_part = to_gemini_part(part)
                if image_part:
                    image_part.pop("cache_control", None)
                    parts.append(image_part)
                continue

            part_type = str(part.get("type") or "").lower()
            if part_type in ("text", "input_text", "output_text"):
                parts.append({"text": str(part.get("text") or "")})
            elif part_type == "input_audio":
                audio = part.get("input_audio") or {}
                data = audio.get("data")
                fmt = str(audio.get("format") or "wav").lower()
                if data:
                    mime = "audio/mpeg" if fmt == "mp3" else f"audio/{fmt}"
                    parts.append({"inlineData": {"mimeType": mime, "data": data}})
            elif part_type == "file":
                file_obj = part.get("file") or {}
                file_data = file_obj.get("file_data") or file_obj.get("fileData")
                parsed = parse_data_url(file_data)
                if parsed:
                    mime_type, data = parsed
                    parts.append(
                        {"inlineData": {"mimeType": mime_type, "data": data}}
                    )
                elif file_obj.get("file_id") or file_obj.get("fileId"):
                    parts.append(
                        {
                            "fileData": {
                                "fileUri": str(
                                    file_obj.get("file_id") or file_obj.get("fileId")
                                )
                            }
                        }
                    )
            elif "inlineData" in part or "inline_data" in part:
                inline = part.get("inlineData") or part.get("inline_data") or {}
                mime = inline.get("mimeType") or inline.get("mime_type") or "application/octet-stream"
                data = inline.get("data")
                if data:
                    parts.append({"inlineData": {"mimeType": mime, "data": data}})
            elif "fileData" in part or "file_data" in part:
                file_data = part.get("fileData") or part.get("file_data") or {}
                mime = file_data.get("mimeType") or file_data.get("mime_type")
                uri = file_data.get("fileUri") or file_data.get("file_uri")
                if uri:
                    entry: Dict[str, Any] = {"fileUri": uri}
                    if mime:
                        entry["mimeType"] = mime
                    parts.append({"fileData": entry})
            elif part.get("text") is not None:
                parts.append({"text": str(part.get("text") or "")})
        return parts

    def _build_contents(
        self, request: ChatCompletionRequest
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        system_parts: List[Dict[str, Any]] = []
        contents: List[Dict[str, Any]] = []
        mapper = _ToolIdMapper()

        call_id_to_name: Dict[str, str] = {}
        for message in request.messages:
            for tool_call in message.tool_calls or []:
                call_id = tool_call.get("id")
                name = (tool_call.get("function") or {}).get("name")
                if call_id and name:
                    call_id_to_name[call_id] = name

        for message in request.messages:
            role = message.role
            if role == "system":
                if isinstance(message.content, list):
                    for part in message.content:
                        if isinstance(part, str):
                            system_parts.append({"text": part})
                        elif isinstance(part, dict) and not is_image_part(part):
                            text = part.get("text")
                            if text is not None:
                                system_parts.append({"text": str(text)})
                elif message.content:
                    system_parts.append({"text": str(message.content)})
                continue

            if role in ("tool", "function"):
                call_id = message.tool_call_id or getattr(message, "id", None)
                function_name = (
                    message.name
                    or call_id_to_name.get(call_id)
                    or "tool"
                )
                emitted_id = mapper.resolve(call_id) or call_id
                content = message.content
                try:
                    parsed_content = (
                        json.loads(content) if isinstance(content, str) else content
                    )
                except Exception:
                    parsed_content = content
                if not isinstance(parsed_content, dict):
                    parsed_content = {"result": parsed_content}
                response_part: Dict[str, Any] = {
                    "name": _sanitize_function_name(function_name),
                    "response": {"result": parsed_content},
                }
                if emitted_id:
                    response_part["id"] = emitted_id
                contents.append(
                    {"role": "user", "parts": [{"functionResponse": response_part}]}
                )
                continue

            if role == "assistant":
                parts: List[Dict[str, Any]] = []
                text = content_text(message.content).strip()
                if text:
                    parts.append({"text": text})
                for tool_call in message.tool_calls or []:
                    function = tool_call.get("function") or {}
                    original_id = tool_call.get("id")
                    emitted_id = mapper.emit(original_id)
                    raw_arguments = function.get("arguments") or {}
                    if isinstance(raw_arguments, str):
                        try:
                            arguments = json.loads(raw_arguments)
                        except Exception:
                            arguments = {}
                    elif isinstance(raw_arguments, dict):
                        arguments = raw_arguments
                    else:
                        arguments = {}
                    call_part: Dict[str, Any] = {
                        "functionCall": {
                            "name": _sanitize_function_name(
                                function.get("name") or "tool"
                            ),
                            "args": arguments,
                            "id": emitted_id,
                        }
                    }
                    signature = (
                        tool_call.get("thoughtSignature")
                        or tool_call.get("thought_signature")
                        or _recall_thought_signature(emitted_id)
                        or _recall_thought_signature(original_id)
                    )
                    if signature:
                        call_part["thoughtSignature"] = signature
                    parts.append(call_part)
                if not parts:
                    parts.append({"text": ""})
                contents.append({"role": "model", "parts": parts})
                continue

            parts = self._content_parts(message.content)
            contents.append({"role": "user", "parts": parts or [{"text": ""}]})

        merged: List[Dict[str, Any]] = []
        for item in contents:
            if merged and merged[-1]["role"] == item["role"]:
                merged[-1]["parts"].extend(item["parts"])
            else:
                merged.append({"role": item["role"], "parts": list(item["parts"])})

        if not merged:
            merged.append({"role": "user", "parts": [{"text": "Hello"}]})
        elif merged[0]["role"] != "user":
            merged.insert(0, {"role": "user", "parts": [{"text": "..."}]})
        return merged, system_parts

    def _build_payload(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        variables: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, str]]:
        contents, system_parts = self._build_contents(request)
        target_model = model_config.upstream_model or model_config.id

        tools, name_map = self._convert_tools(request.tools, request.functions)
        tool_choice = request.tool_choice or request.function_call

        inner: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": self._build_generation_config(
                request, model_config, variables
            ),
            "safetySettings": self._build_safety_settings(variables),
        }
        if system_parts:
            inner["systemInstruction"] = {"role": "user", "parts": system_parts}
        if tools:
            inner["tools"] = tools
            tool_config = self._convert_tool_choice(tool_choice, name_map)
            if tool_config:
                inner["toolConfig"] = tool_config

        extra = request.extra_body if isinstance(request.extra_body, dict) else {}
        cached_content = (
            getattr(request, "cached_content", None)
            or getattr(request, "cachedContent", None)
            or extra.get("cached_content")
            or extra.get("cachedContent")
            or variables.get("cached_content")
        )
        if cached_content:
            inner["cachedContent"] = str(cached_content)

        return inner, name_map

    # ------------------------------------------------------------------
    # Response translation
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_usage(
        previous: Optional[CompletionUsage], current: Optional[CompletionUsage]
    ) -> Optional[CompletionUsage]:
        if current is None:
            return previous
        if previous is None:
            return current
        merged = CompletionUsage(
            prompt_tokens=max(previous.prompt_tokens, current.prompt_tokens),
            completion_tokens=max(
                previous.completion_tokens, current.completion_tokens
            ),
            total_tokens=max(previous.total_tokens, current.total_tokens),
        )
        details: Dict[str, Any] = {}
        for usage in (previous, current):
            if isinstance(usage.prompt_tokens_details, dict):
                details.update(usage.prompt_tokens_details)
        if details:
            merged.prompt_tokens_details = details
        if previous.cache_read_input_tokens is not None or current.cache_read_input_tokens is not None:
            merged.cache_read_input_tokens = max(
                previous.cache_read_input_tokens or 0,
                current.cache_read_input_tokens or 0,
            )
        completion_details: Dict[str, Any] = {}
        for usage in (previous, current):
            value = getattr(usage, "completion_tokens_details", None)
            if isinstance(value, dict):
                completion_details.update(value)
        if completion_details:
            setattr(merged, "completion_tokens_details", completion_details)
        return merged

    def _build_usage(
        self,
        usage_obj: Optional[Dict[str, Any]],
        default_prompt: int = 0,
        default_completion: int = 0,
        previous: Optional[CompletionUsage] = None,
    ) -> CompletionUsage:
        """Map Gemini ``usageMetadata`` to OpenAI usage semantics."""
        usage_obj = usage_obj or {}
        prompt_tokens = int(usage_obj.get("promptTokenCount") or default_prompt or 0)
        candidates = int(
            usage_obj.get("candidatesTokenCount") or default_completion or 0
        )
        thoughts = int(usage_obj.get("thoughtsTokenCount") or 0)
        completion_tokens = candidates + thoughts
        total_tokens = int(
            usage_obj.get("totalTokenCount") or (prompt_tokens + completion_tokens)
        )
        usage = CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

        cached_tokens = usage_obj.get("cachedContentTokenCount")
        if cached_tokens is None and previous is not None:
            cached_tokens = previous.cache_read_input_tokens
        if cached_tokens is not None:
            cached_value = max(0, int(cached_tokens))
            usage.prompt_tokens_details = {"cached_tokens": cached_value}
            usage.cache_read_input_tokens = cached_value
        if thoughts:
            setattr(
                usage,
                "completion_tokens_details",
                {"reasoning_tokens": thoughts},
            )
        return usage

    @staticmethod
    def _unwrap_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        if "candidates" not in payload and isinstance(payload.get("response"), dict):
            return payload["response"]
        return payload

    def _parse_payload(
        self,
        payload: Dict[str, Any],
        name_map: Dict[str, str],
        pending_signature: Optional[str] = None,
    ) -> Dict[str, Any]:
        response = self._unwrap_payload(payload)
        usage_data = response.get("usageMetadata") or payload.get("usageMetadata")
        usage = (
            self._build_usage(usage_data)
            if isinstance(usage_data, dict) and usage_data
            else None
        )

        texts: List[str] = []
        thoughts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        finish_reason: Optional[str] = None

        for candidate in response.get("candidates") or []:
            if not isinstance(candidate, dict):
                continue
            for part in (candidate.get("content") or {}).get("parts") or []:
                if not isinstance(part, dict):
                    continue
                signature = part.get("thoughtSignature") or part.get(
                    "thought_signature"
                )
                if signature:
                    pending_signature = str(signature)

                function_call = part.get("functionCall")
                if isinstance(function_call, dict):
                    raw_name = str(function_call.get("name") or "")
                    call_id = function_call.get("id") or f"call_{uuid.uuid4().hex[:12]}"
                    _remember_thought_signature(call_id, pending_signature)
                    pending_signature = None
                    tool_calls.append(
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name_map.get(raw_name, raw_name),
                                "arguments": json.dumps(
                                    function_call.get("args") or {}
                                ),
                            },
                        }
                    )
                    continue

                text = part.get("text")
                if text is None or text == "":
                    continue
                if part.get("thought") is True:
                    thoughts.append(str(text))
                else:
                    texts.append(str(text))

            candidate_finish = candidate.get("finishReason")
            if candidate_finish:
                finish_reason = _FINISH_REASON_MAP.get(
                    str(candidate_finish).upper(), "stop"
                )

        block_reason = (response.get("promptFeedback") or {}).get("blockReason")
        if block_reason:
            finish_reason = "content_filter"

        return {
            "texts": texts,
            "thoughts": thoughts,
            "tool_calls": tool_calls,
            "finish_reason": finish_reason,
            "usage": usage,
            "pending_signature": pending_signature,
            "block_reason": block_reason,
        }

    # ------------------------------------------------------------------
    # Upstream calls
    # ------------------------------------------------------------------

    @staticmethod
    def _raise_upstream_error(response: httpx.Response, body: str) -> None:
        if response.status_code >= 400:
            excerpt = (body or "").strip()
            if len(excerpt) > 800:
                excerpt = excerpt[:800] + "..."
            raise httpx.HTTPStatusError(
                f"Gemini upstream error ({response.status_code}): {excerpt}",
                request=response.request,
                response=response,
            )

    async def _stream(
        self,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        model_name: str,
        timeout: float,
        proxy: Optional[str],
        name_map: Dict[str, str],
    ) -> AsyncIterator[ChatCompletionChunk]:
        chunk_id = f"chatcmpl-gm-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        last_usage: Optional[CompletionUsage] = None
        pending_signature: Optional[str] = None
        finish_reason: Optional[str] = None
        has_tool_call = False
        tool_call_index = 0

        yield ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=model_name,
            choices=[
                ChatChoiceDelta(index=0, delta={"role": "assistant", "content": ""})
            ],
        )

        async with httpx.AsyncClient(timeout=timeout, proxy=proxy) as client:
            async with client.stream(
                "POST", url, headers=headers, json=payload
            ) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", errors="replace")
                    self._raise_upstream_error(response, body)

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
                        payload_chunk = json.loads(data)
                    except Exception:
                        continue
                    result = self._parse_payload(
                        payload_chunk, name_map, pending_signature
                    )
                    pending_signature = result["pending_signature"]
                    last_usage = self._merge_usage(last_usage, result["usage"])

                    for thought in result["thoughts"]:
                        yield ChatCompletionChunk(
                            id=chunk_id,
                            created=created,
                            model=model_name,
                            choices=[
                                ChatChoiceDelta(
                                    index=0,
                                    delta={"reasoning_content": thought},
                                )
                            ],
                        )
                    for text in result["texts"]:
                        yield ChatCompletionChunk(
                            id=chunk_id,
                            created=created,
                            model=model_name,
                            choices=[
                                ChatChoiceDelta(index=0, delta={"content": text})
                            ],
                        )
                    for tool_call in result["tool_calls"]:
                        has_tool_call = True
                        yield ChatCompletionChunk(
                            id=chunk_id,
                            created=created,
                            model=model_name,
                            choices=[
                                ChatChoiceDelta(
                                    index=0,
                                    delta={
                                        "tool_calls": [
                                            {
                                                "index": tool_call_index,
                                                "id": tool_call["id"],
                                                "type": "function",
                                                "function": tool_call["function"],
                                            }
                                        ]
                                    },
                                )
                            ],
                        )
                        tool_call_index += 1
                    if result["finish_reason"]:
                        finish_reason = result["finish_reason"]

        final_reason = "tool_calls" if has_tool_call else (finish_reason or "stop")
        yield ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=model_name,
            choices=[ChatChoiceDelta(index=0, delta={}, finish_reason=final_reason)],
            usage=last_usage,
        )

    async def _complete(
        self,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        model_name: str,
        timeout: float,
        proxy: Optional[str],
        name_map: Dict[str, str],
    ) -> ChatCompletionResponse:
        async with httpx.AsyncClient(timeout=timeout, proxy=proxy) as client:
            response = await client.post(url, headers=headers, json=payload)
            if response.status_code >= 400:
                self._raise_upstream_error(response, response.text)
            payload_data = response.json()

        result = self._parse_payload(payload_data, name_map)
        text = "".join(result["texts"])
        tool_calls = result["tool_calls"]
        finish_reason = (
            "tool_calls" if tool_calls else (result["finish_reason"] or "stop")
        )
        message_payload: Dict[str, Any] = {
            "role": "assistant",
            "content": text if (text or not tool_calls) else None,
            "tool_calls": tool_calls or None,
        }
        if result["thoughts"]:
            message_payload["reasoning_content"] = "".join(result["thoughts"])

        return ChatCompletionResponse(
            id=f"chatcmpl-gm-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=model_name,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage.model_validate(message_payload),
                    finish_reason=finish_reason,
                )
            ],
            usage=result["usage"],
        )

    # ------------------------------------------------------------------
    # BasePlugin entry point
    # ------------------------------------------------------------------

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        variables: Dict[str, Any],
        account: Optional[Account] = None,
    ) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionChunk]]:
        api_key = account.api_key if account else ""
        if self._is_test_mode(api_key):
            if request.stream:
                return self._mock_stream(request, model_config, account=account)
            return self._mock_response(request, model_config, account=account)

        await self._ensure_token_freshness(account, variables)
        api_key = account.api_key if account else ""

        timeout = float(variables.get("timeout") or 60.0)
        proxy = str(
            variables.get("proxy")
            or (getattr(account, "proxy_url", None) if account else "")
            or ""
        ).strip() or None

        if messages_have_images(request.messages):
            materialized = await materialize_message_images(
                request.messages,
                timeout=min(timeout, 30.0),
                proxy=proxy,
            )
            if materialized is not request.messages:
                request = request.model_copy(update={"messages": materialized})

        model_name = model_config.upstream_model or model_config.id
        url = self._build_url(
            variables, account, model_name, bool(request.stream), api_key
        )
        headers = self._build_headers(
            api_key, variables, account, bool(request.stream)
        )
        payload, name_map = self._build_payload(request, model_config, variables)

        if request.stream:
            return self._stream(
                url,
                headers,
                payload,
                request.model or model_config.id,
                timeout,
                proxy,
                name_map,
            )
        return await self._complete(
            url,
            headers,
            payload,
            request.model or model_config.id,
            timeout,
            proxy,
            name_map,
        )


plugin = GeminiPlugin()
