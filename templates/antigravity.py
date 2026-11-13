"""
Google Antigravity Provider Plugin
==================================

Implementation based on 9Router IDE protocols and authentic Google Antigravity
Cloud Code PA internal endpoints (/v1internal:streamGenerateContent?alt=sse).

Key Features:
1. Authentic Google Account Connection:
   Connects to Google accounts using Antigravity client credentials with OAuth 2.0,
   automatic token refresh using refresh_token, and Code Assist project mapping.
2. 9Router IDE Trajectory Algorithm:
   Generates structured hierarchical trajectory paths:
     agent/<conversation_id>/<timestamp>/<trajectory_id>/<step>
   where conversation_id and trajectory_id are generated using deterministic SHA-256
   derived UUIDs seeded from session/model/client metadata.
3. Official Antigravity CLI Telemetry Headers:
   User-Agent: antigravity/cli/1.5.3 (aidev_client; os_type=linux; arch=amd64; cl=1016078699; auth_method=consumer)
   Authorization: Bearer <access_token>
   Content-Type: application/json
4. Endpoint Protocol:
   POST /v1internal:streamGenerateContent?alt=sse
   Payload matching official agy CLI with contents, generationConfig, project, requestId,
   model, userAgent, and requestType.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Set, Tuple, Union

import httpx

from llmproxy.core.antigravity_oauth import (
    ANTIGRAVITY_CLIENT_ID,
    ANTIGRAVITY_CLIENT_SECRET,
    ANTIGRAVITY_DEFAULT_PROJECT,
    ANTIGRAVITY_TOKEN_URI,
    ANTIGRAVITY_USER_AGENT,
    CLOUDCODE_PA_ENDPOINTS,
    antigravity_oauth,
    sanitize_ssl_env,
)
from llmproxy.core.interface import BasePlugin
from llmproxy.core.multimodal import (
    content_text,
    is_image_part,
    materialize_message_images,
    messages_have_images,
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

logger = logging.getLogger(__name__)

_THOUGHT_SIGNATURES: Dict[str, str] = {}
_THOUGHT_SIGNATURES_FILE: Optional[str] = None


def _get_thought_signatures_file() -> str:
    global _THOUGHT_SIGNATURES_FILE
    if _THOUGHT_SIGNATURES_FILE:
        return _THOUGHT_SIGNATURES_FILE
    for candidate in ["/app/data", "./data", "/tmp/antigravity", "/tmp"]:
        p = Path(candidate)
        if p.exists() and p.is_dir():
            _THOUGHT_SIGNATURES_FILE = str(p / "thought_signatures.json")
            return _THOUGHT_SIGNATURES_FILE
    _THOUGHT_SIGNATURES_FILE = "/tmp/thought_signatures.json"
    return _THOUGHT_SIGNATURES_FILE


def _load_thought_signatures() -> None:
    global _THOUGHT_SIGNATURES
    fpath = _get_thought_signatures_file()
    if os.path.exists(fpath):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                _THOUGHT_SIGNATURES = json.load(f)
        except Exception:
            _THOUGHT_SIGNATURES = {}


def _save_thought_signature(
    call_id: Optional[str], func_name: Optional[str], args: Any, signature: str
) -> None:
    global _THOUGHT_SIGNATURES
    if not signature:
        return
    if call_id:
        _THOUGHT_SIGNATURES[call_id] = signature
    if func_name:
        _THOUGHT_SIGNATURES[f"name:{func_name}"] = signature
        if args is not None:
            args_key = json.dumps(args, sort_keys=True) if isinstance(args, dict) else str(args)
            _THOUGHT_SIGNATURES[f"{func_name}:{args_key}"] = signature
    _THOUGHT_SIGNATURES["latest"] = signature
    try:
        fpath = _get_thought_signatures_file()
        if len(_THOUGHT_SIGNATURES) > 1000:
            items = list(_THOUGHT_SIGNATURES.items())[-500:]
            _THOUGHT_SIGNATURES = dict(items)
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(_THOUGHT_SIGNATURES, f)
    except Exception:
        pass


def _get_thought_signature(
    call_id: Optional[str], func_name: Optional[str] = None, args: Any = None
) -> Optional[str]:
    if not _THOUGHT_SIGNATURES:
        _load_thought_signatures()
    if call_id and call_id in _THOUGHT_SIGNATURES:
        return _THOUGHT_SIGNATURES[call_id]
    if func_name and args is not None:
        args_key = json.dumps(args, sort_keys=True) if isinstance(args, dict) else str(args)
        sig = _THOUGHT_SIGNATURES.get(f"{func_name}:{args_key}")
        if sig:
            return sig
    if func_name and f"name:{func_name}" in _THOUGHT_SIGNATURES:
        return _THOUGHT_SIGNATURES[f"name:{func_name}"]
    return _THOUGHT_SIGNATURES.get("latest")


def _to_gemini_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(schema, dict):
        return {"type": "OBJECT", "properties": {}}

    type_map = {
        "string": "STRING",
        "number": "NUMBER",
        "integer": "INTEGER",
        "boolean": "BOOLEAN",
        "array": "ARRAY",
        "object": "OBJECT",
    }

    out: Dict[str, Any] = {}

    if "exclusiveMinimum" in schema and "minimum" not in schema:
        val = schema["exclusiveMinimum"]
        if isinstance(val, (int, float)):
            out["minimum"] = val + 1 if isinstance(val, int) else val
    if "exclusiveMaximum" in schema and "maximum" not in schema:
        val = schema["exclusiveMaximum"]
        if isinstance(val, (int, float)):
            out["maximum"] = val - 1 if isinstance(val, int) else val

    ALLOWED_FIELDS = {
        "type",
        "format",
        "description",
        "nullable",
        "enum",
        "maxItems",
        "minItems",
        "properties",
        "required",
        "items",
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
        "default",
    }

    for k, v in schema.items():
        if k not in ALLOWED_FIELDS:
            continue
        if k == "type":
            if isinstance(v, str):
                out["type"] = type_map.get(v.lower(), v.upper())
            elif isinstance(v, list):
                types = [t for t in v if t != "null"]
                t0 = types[0] if types else "string"
                out["type"] = type_map.get(t0.lower(), t0.upper())
                if "null" in v:
                    out["nullable"] = True
            else:
                out["type"] = "STRING"
        elif k == "properties" and isinstance(v, dict):
            out["properties"] = {pk: _to_gemini_schema(pv) for pk, pv in v.items()}
        elif k == "items" and isinstance(v, dict):
            out["items"] = _to_gemini_schema(v)
        elif k == "enum" and isinstance(v, list):
            out["enum"] = [str(x) for x in v]
        elif k in ALLOWED_FIELDS:
            out[k] = v

    if "type" not in out:
        out["type"] = "OBJECT"
    return out


def _convert_openai_tools_to_gemini(
    tools: Optional[List[Dict[str, Any]]],
) -> Optional[List[Dict[str, Any]]]:
    if not tools:
        return None
    gemini_tools: List[Dict[str, Any]] = []
    for t in tools:
        if "function" in t:
            fn = t["function"]
            decl = {
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "parameters": _to_gemini_schema(fn.get("parameters", {})),
            }
            gemini_tools.append({"functionDeclarations": [decl]})
        elif "functionDeclarations" in t:
            gemini_tools.append(t)
        elif "name" in t:
            decl = {
                "name": t.get("name", ""),
                "description": t.get("description", ""),
                "parameters": _to_gemini_schema(t.get("parameters", {})),
            }
            gemini_tools.append({"functionDeclarations": [decl]})
    return gemini_tools or None


def _convert_tool_choice_to_gemini(
    tool_choice: Optional[Union[str, Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    if not tool_choice or tool_choice == "auto":
        return {"functionCallingConfig": {"mode": "AUTO"}}
    if tool_choice == "none":
        return {"functionCallingConfig": {"mode": "NONE"}}
    if tool_choice == "required":
        return {"functionCallingConfig": {"mode": "ANY"}}
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") == "function":
            fn_name = tool_choice.get("function", {}).get("name")
            if fn_name:
                return {
                    "functionCallingConfig": {
                        "mode": "ANY",
                        "allowedFunctionNames": [fn_name],
                    }
                }
    return None


class AntigravityPlugin(BasePlugin):
    """
    Google Antigravity Provider Plugin with 9Router reference algorithms
    and direct Cloud Code PA integration matching official `agy` CLI 1:1.
    """

    REJECTED_GOOGLE_FIELDS: Set[str] = {
        "output_config",
        "thinking",
        "reasoning_effort",
        "reasoning",
        "enable_thinking",
        "thinking_budget",
        "thinkingConfig",
        "reasoning_tags",
        "reasoning_tag",
        "reasoning_tag_name",
        "frequency_penalty",
        "presence_penalty",
        "logprobs",
        "top_logprobs",
        "logit_bias",
        "user",
        "seed",
        "service_tier",
        "disallowed_field",
        "unsupported_field",
    }
    DISALLOWED_FIELDS: Set[str] = REJECTED_GOOGLE_FIELDS

    def __init__(self):
        self._metadata = PluginMetadata(
            id="antigravity",
            name="Antigravity",
            description="Google Antigravity provider with 9Router IDE trajectory/session ID algorithms, multi-account load balancing, and authentic Cloud Code PA integration",
            version="2.2.0",
            author="9Router",
            enabled=True,
            filename="antigravity.py",
            variables=[
                PluginVariable(
                    name="timeout",
                    label="Timeout (seconds)",
                    type="number",
                    default=60,
                    value=60,
                    required=True,
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
                    default=60,
                    description="Wait delay before switching to another account on failover (seconds)",
                ),
                PluginVariable(
                    name="max_retries",
                    label="Max Failover Retries",
                    type="number",
                    default=2,
                    description="Maximum number of account failovers per request",
                ),
                PluginVariable(
                    name="base_url",
                    label="Default Base URL",
                    type="string",
                    default=CLOUDCODE_PA_ENDPOINTS[0],
                    description="Google Antigravity CloudCode PA base URL",
                ),
                PluginVariable(
                    name="reasoning_effort",
                    label="Reasoning Effort",
                    type="select",
                    default="high",
                    options=["high", "medium", "low", "none"],
                    description="Default reasoning effort mapped to model suffix (-high, -medium, -low, or none)",
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
                    id="gemini-3.8-flash",
                    name="Gemini 3.8 Flash",
                    upstream_model="gemini-3.8-flash",
                    context_window=1048576,
                    max_tokens=65536,
                ),
                PluginModelConfig(
                    id="gemini-3.1-pro",
                    name="Gemini 3.1 Pro",
                    upstream_model="gemini-3.1-pro",
                    context_window=2097152,
                    max_tokens=65536,
                ),
                PluginModelConfig(
                    id="gemini-2.5-pro",
                    name="Gemini 2.5 Pro",
                    upstream_model="gemini-2.5-pro",
                    context_window=1048576,
                    max_tokens=8192,
                ),
                PluginModelConfig(
                    id="gemini-2.5-flash",
                    name="Gemini 2.5 Flash",
                    upstream_model="gemini-2.5-flash",
                    context_window=1048576,
                    max_tokens=8192,
                ),
                PluginModelConfig(
                    id="claude-sonnet-4-6",
                    name="Claude Sonnet 4.6",
                    upstream_model="claude-sonnet-4-6",
                    context_window=200000,
                    max_tokens=8192,
                ),
                PluginModelConfig(
                    id="claude-opus-4-6-thinking",
                    name="Claude Opus 4.6 Thinking",
                    upstream_model="claude-opus-4-6-thinking",
                    context_window=200000,
                    max_tokens=16384,
                ),
            ],
        )
        for model in self._metadata.models:
            model.supports_vision = True

    @property
    def metadata(self) -> PluginMetadata:
        return self._metadata

    @classmethod
    def generate_sha256_uuid(cls, seed: str) -> str:
        """Derives an RFC 4122 compliant UUID string deterministically from a SHA-256 digest."""
        digest = hashlib.sha256(seed.encode("utf-8")).digest()
        return str(uuid.UUID(bytes=digest[:16]))

    @classmethod
    def generate_trajectory_id(
        cls,
        conversation_seed: Optional[str] = None,
        trajectory_seed: Optional[str] = None,
        step: int = 1,
        timestamp_ms: Optional[int] = None,
    ) -> str:
        """
        Exact 9Router IDE trajectory ID algorithm:
          agent/<conversation_id>/<timestamp>/<trajectory_id>/<step>
        """
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)

        conv_seed = conversation_seed or f"session_{uuid.uuid4().hex}_{timestamp_ms}"
        traj_seed = trajectory_seed or f"traj_{uuid.uuid4().hex}_{step}"

        conversation_id = cls.generate_sha256_uuid(conv_seed)
        trajectory_id = cls.generate_sha256_uuid(traj_seed)

        return f"agent/{conversation_id}/{timestamp_ms}/{trajectory_id}/{step}"

    @classmethod
    def generate_ide_request_id(cls, seed: Optional[str] = None) -> str:
        """Generates the IDE request ID header required by Antigravity endpoints."""
        req_seed = seed or f"req_{uuid.uuid4().hex}_{time.time()}"
        return f"ide_req_{cls.generate_sha256_uuid(req_seed).replace('-', '')}"

    def build_headers(
        self, account: Optional[Account] = None, api_key: str = ""
    ) -> Dict[str, str]:
        """Constructs authentic HTTP headers matching agy CLI 1:1."""
        key = account.api_key if account else api_key
        user_agent = (
            (account.extra_config.get("user_agent") if account else None)
            or ANTIGRAVITY_USER_AGENT
        )
        headers = {
            "Content-Type": "application/json",
            "User-Agent": user_agent,
        }
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def sanitize_payload(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Strips Google-rejected parameters from the payload."""
        return {k: v for k, v in body.items() if k not in self.REJECTED_GOOGLE_FIELDS}

    @staticmethod
    def _apply_cache_fields(
        inner_request: Dict[str, Any],
        request: ChatCompletionRequest,
        supports_block_cache: bool = False,
    ) -> None:
        """Forward Gemini explicit cache references and Anthropic-style breakpoints."""
        extra = request.extra_body if isinstance(request.extra_body, dict) else {}
        cached_content = (
            getattr(request, "cached_content", None)
            or getattr(request, "cachedContent", None)
            or extra.get("cached_content")
            or extra.get("cachedContent")
        )
        if cached_content:
            inner_request["cachedContent"] = cached_content

        cache_control = getattr(request, "cache_control", None) or extra.get("cache_control")
        if not cache_control or not supports_block_cache:
            return
        contents = inner_request.get("contents") or []
        if not contents:
            return
        parts = contents[-1].get("parts")
        if isinstance(parts, list) and parts and isinstance(parts[-1], dict):
            parts[-1].setdefault("cache_control", cache_control)

    @staticmethod
    def _build_usage(
        usage_obj: Optional[Dict[str, Any]],
        default_prompt: int = 0,
        default_completion: int = 0,
        previous: Optional[CompletionUsage] = None,
    ) -> CompletionUsage:
        """Map Google usageMetadata, including cached content tokens, to OpenAI shape."""
        usage_obj = usage_obj or {}
        prompt_tokens = int(usage_obj.get("promptTokenCount") or default_prompt or 0)
        completion_tokens = int(
            usage_obj.get("candidatesTokenCount") or default_completion or 0
        )
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
            cached_tokens = getattr(previous, "cache_read_input_tokens", None)
        if cached_tokens is not None:
            usage.prompt_tokens_details = {"cached_tokens": int(cached_tokens)}
            usage.cache_read_input_tokens = int(cached_tokens)
        return usage

    def _is_test_mode(self, api_key: str) -> bool:
        if not api_key:
            return True
        key = api_key.lower().strip()
        prefixes = ("test", "mock", "sk-test", "dummy", "fake", "offline", "sample")
        return any(key.startswith(p) for p in prefixes) or key in ("test-key", "test_key", "secret123")

    def _parse_reasoning_level(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        variables: Dict[str, Any],
    ) -> Tuple[str, str, bool]:
        """
        Parse and resolve reasoning level for Antigravity provider.
        Streamlines Google's model naming (gemini-3.8-flash-high/-medium/-low)
        so clients only see and send gemini-3.8-flash, etc. without suffixes.

        Returns:
            (actual_model_name, reasoning_level, enable_reasoning)
        """
        raw = None
        if hasattr(request, "reasoning_effort") and request.reasoning_effort:
            raw = str(request.reasoning_effort).strip().lower()
        elif request.extra_body:
            if "reasoning_effort" in request.extra_body:
                raw = str(request.extra_body.get("reasoning_effort") or "").strip().lower()
            elif "reasoning" in request.extra_body:
                r_val = request.extra_body.get("reasoning")
                if isinstance(r_val, dict):
                    raw = str(r_val.get("effort") or "").strip().lower()
                else:
                    raw = str(r_val or "").strip().lower()

        # Check model name suffix on the requested model ID (e.g. gemini-3.8-flash:high, gemini-3.8-flash-high)
        if not raw and request.model:
            m_lower = request.model.lower()
            for suffix, level in [
                (":none", "none"), ("-none", "none"), (":off", "none"), ("-off", "none"),
                (":low", "low"), ("-low", "low"),
                (":medium", "medium"), ("-medium", "medium"),
                (":high", "high"), ("-high", "high"),
            ]:
                if m_lower.endswith(suffix):
                    raw = level
                    break

        if not raw:
            var_val = str(variables.get("reasoning_effort") or "").strip().lower()
            if var_val:
                raw = var_val

        if not raw or raw == "default":
            raw = "high"

        base_upstream = model_config.upstream_model or model_config.id

        # Normalize base_upstream by stripping any existing suffix
        cleaned_base = base_upstream
        for sfx in ["-high", "-medium", "-low", "-thinking"]:
            if cleaned_base.endswith(sfx):
                cleaned_base = cleaned_base[:-len(sfx)]
                break

        if raw in ("none", "off", "disabled", "false", "0"):
            return cleaned_base, "none", False

        if raw in ("low", "medium", "high"):
            if "claude" in cleaned_base:
                actual_model = f"{cleaned_base}-thinking"
            elif any(k in cleaned_base for k in ["gemini-3", "gemini-2.5", "flash", "pro"]):
                actual_model = f"{cleaned_base}-{raw}"
            else:
                actual_model = f"{cleaned_base}-{raw}"
            return actual_model, raw, True

        return f"{cleaned_base}-high", "high", True

    def _is_thought_part(self, part: Dict[str, Any]) -> bool:
        """Safely determine if an upstream content part is an internal thought/reasoning part."""
        th = part.get("thought")
        if th is not None:
            if isinstance(th, bool):
                return th
            if isinstance(th, str):
                return th.strip().lower() in ("true", "1", "yes")
            return bool(th)
        if "isThought" in part:
            it = part["isThought"]
            if isinstance(it, bool):
                return it
            if isinstance(it, str):
                return it.strip().lower() in ("true", "1", "yes")
            return bool(it)
        if part.get("type") == "thought":
            return True
        return False

    def _mock_response(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        account: Optional[Account] = None,
        enable_reasoning_tags: bool = False,
        tag_name: str = "think",
        actual_model: Optional[str] = None,
        enable_reasoning: bool = True,
        time_taken_ms: float = 0.0,
    ) -> ChatCompletionResponse:
        model_name = request.model or model_config.id
        acc_label = f" via account '{account.name}'" if account else ""
        if time_taken_ms <= 0.0:
            time_taken_ms = 1.25

        mock_thought = bool(
            enable_reasoning
            and (
                (request.extra_body and request.extra_body.get("mock_thought"))
                or getattr(request, "mock_thought", False)
            )
        )
        reasoning_content = None
        if mock_thought:
            reasoning_content = f"Simulated reasoning for {model_name}."

        if request.tools and not any(m.role in ("tool", "function") for m in request.messages[-2:]):
            fn = request.tools[0].get("function", {}) if "function" in request.tools[0] else request.tools[0]
            call_id = f"call_ag_{uuid.uuid4().hex[:8]}"
            tool_name = fn.get("name", "test_tool")
            mock_args = "{}"
            props = fn.get("parameters", {}).get("properties", {})
            if props:
                first_k = next(iter(props.keys()))
                mock_args = json.dumps({first_k: "test_value"})

            msg_dict: Dict[str, Any] = {
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
            if reasoning_content:
                msg_dict["reasoning_content"] = reasoning_content

            return ChatCompletionResponse(
                id=f"chatcmpl-ag-{uuid.uuid4().hex[:12]}",
                created=int(time.time()),
                model=model_name,
                choices=[
                    ChatChoice(
                        index=0,
                        message=ChatMessage.model_validate(msg_dict),
                        finish_reason="tool_calls",
                    )
                ],
                usage=CompletionUsage(
                    prompt_tokens=15,
                    completion_tokens=25,
                    total_tokens=40,
                ),
            )

        base_content = f"Simulated response from 9Router Antigravity ({model_name}){acc_label}."
        if any(m.role in ("tool", "function") for m in request.messages):
            base_content = f"Simulated response from 9Router Antigravity ({model_name}){acc_label} with tool output processed."

        now_str = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        status_desc = (
            "No account configured (Simulation Mode)"
            if (not account or account.id.startswith("default-") or not account.api_key)
            else f"Account '{account.name}' (Test mode)"
        )
        prompt_tokens = 15
        completion_tokens = 25
        total_tokens = prompt_tokens + completion_tokens

        details = (
            f"\n\n---\n"
            f"**Execution Details:**\n"
            f"• **Time taken:** {time_taken_ms:.2f} ms\n"
            f"• **Timestamp:** {now_str}\n"
            f"• **Model:** {model_name}\n"
            f"• **Tokens:** {prompt_tokens} prompt, {completion_tokens} completion ({total_tokens} total)\n"
            f"• **Account status:** {status_desc}"
        )
        content = base_content + details

        resp_msg_dict: Dict[str, Any] = {
            "role": "assistant",
            "content": content,
        }
        if reasoning_content:
            resp_msg_dict["reasoning_content"] = reasoning_content

        return ChatCompletionResponse(
            id=f"chatcmpl-ag-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=model_name,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage.model_validate(resp_msg_dict),
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=15,
                completion_tokens=25,
                total_tokens=40,
            ),
        )

    async def _mock_stream(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        account: Optional[Account] = None,
        enable_reasoning_tags: bool = False,
        tag_name: str = "think",
        actual_model: Optional[str] = None,
        enable_reasoning: bool = True,
        time_taken_ms: float = 0.0,
    ) -> AsyncIterator[ChatCompletionChunk]:
        model_name = model_config.id
        chunk_id = f"chatcmpl-ag-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        acc_label = f" via account '{account.name}'" if account else ""
        if time_taken_ms <= 0.0:
            time_taken_ms = 1.25

        mock_thought = bool(
            enable_reasoning
            and (
                (request.extra_body and request.extra_body.get("mock_thought"))
                or getattr(request, "mock_thought", False)
            )
        )

        if request.tools and not any(m.role in ("tool", "function") for m in request.messages[-2:]):
            fn = request.tools[0].get("function", {}) if "function" in request.tools[0] else request.tools[0]
            call_id = f"call_ag_{uuid.uuid4().hex[:8]}"
            tool_name = fn.get("name", "test_tool")
            mock_args = "{}"
            props = fn.get("parameters", {}).get("properties", {})
            if props:
                first_k = next(iter(props.keys()))
                mock_args = json.dumps({first_k: "test_value"})

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
            if mock_thought:
                yield ChatCompletionChunk(
                    id=chunk_id,
                    created=created,
                    model=model_name,
                    choices=[ChatChoiceDelta(index=0, delta={"reasoning_content": "Thinking about tools..."})],
                )

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
                                        "name": tool_name,
                                        "arguments": "",
                                    },
                                }
                            ],
                        },
                        finish_reason=None,
                    )
                ],
            )
            arg_chunks = [mock_args[: len(mock_args) // 2], mock_args[len(mock_args) // 2 :]]
            for arg_piece in arg_chunks:
                if arg_piece:
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
                                            "function": {"arguments": arg_piece},
                                        }
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
                choices=[
                    ChatChoiceDelta(
                        index=0,
                        delta={},
                        finish_reason="tool_calls",
                    )
                ],
            )
            return

        base_text = f"Simulated 9Router Antigravity stream response for {model_name}{acc_label}."
        words = base_text.split(" ")

        yield ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=model_name,
            choices=[ChatChoiceDelta(index=0, delta={"role": "assistant", "content": ""})],
        )

        if mock_thought:
            yield ChatCompletionChunk(
                id=chunk_id,
                created=created,
                model=model_name,
                choices=[ChatChoiceDelta(index=0, delta={"reasoning_content": "Simulated reasoning."})],
            )

        for i, word in enumerate(words):
            yield ChatCompletionChunk(
                id=chunk_id,
                created=created,
                model=model_name,
                choices=[
                    ChatChoiceDelta(
                        index=0,
                        delta={"content": word + (" " if i < len(words) - 1 else "")},
                    )
                ],
            )

        now_str = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        status_desc = (
            "No account configured (Simulation Mode)"
            if (not account or account.id.startswith("default-") or not account.api_key)
            else f"Account '{account.name}' (Test mode)"
        )
        details = (
            f"\n\n---\n"
            f"**Execution Details:**\n"
            f"• **Time taken:** {time_taken_ms:.2f} ms\n"
            f"• **Timestamp:** {now_str}\n"
            f"• **Model:** {model_name}\n"
            f"• **Account status:** {status_desc}"
        )
        yield ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=model_name,
            choices=[ChatChoiceDelta(index=0, delta={"content": details}, finish_reason=None)],
        )

        yield ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=model_name,
            choices=[ChatChoiceDelta(index=0, delta={}, finish_reason="stop")],
        )

    async def _ensure_token_freshness(self, account: Optional[Account]) -> None:
        """Transparently refreshes Google OAuth access token if expired or near expiration."""
        if not account:
            return
        refresh_token = account.extra_config.get("refresh_token")
        if not refresh_token:
            return

        expires_at = float(account.extra_config.get("expires_at", 0))
        now = time.time()
        if not account.api_key or (expires_at > 0 and now >= expires_at - 120):
            refreshed = await antigravity_oauth.refresh_account_token(account)
            if refreshed:
                from llmproxy.core.manager import plugin_manager
                plugin_manager._save_accounts()

    def _build_upstream_request(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        account: Optional[Account],
        actual_model: Optional[str] = None,
        enable_reasoning: bool = True,
        reasoning_level: str = "default",
    ) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """Constructs headers and request body matching agy CLI 1:1."""
        headers = self.build_headers(account=account)
        project_id = (account.extra_config.get("project_id") if account else None) or ANTIGRAVITY_DEFAULT_PROJECT
        target_model = actual_model or model_config.upstream_model or model_config.id

        # Generate trajectory and conversation seeds
        acc_id = account.id if account else "default"
        conv_seed = (account.extra_config.get("session_id") if account else None) or f"conv_{acc_id}_{int(time.time() // 3600)}"
        traj_seed = (account.extra_config.get("trajectory_seed") if account else None) or f"traj_{acc_id}_{uuid.uuid4().hex[:8]}"
        step = int(account.extra_config.get("step", 1)) if account else 1

        conv_id = self.generate_sha256_uuid(conv_seed)
        traj_id = self.generate_sha256_uuid(traj_seed)
        timestamp_ms = int(time.time() * 1000)
        request_id = f"agent/{conv_id}/{timestamp_ms}/{traj_id}/{step}"

        # 64-bit integer session ID matching agy
        session_hash = int(hashlib.sha256(conv_id.encode("utf-8")).hexdigest()[:16], 16)
        session_id_str = str(session_hash - (1 << 63))

        # Build contents from messages
        contents: List[Dict[str, Any]] = []
        system_instruction_parts: List[Dict[str, Any]] = []

        # Map call_id -> function_name for tool response mapping
        call_id_to_name: Dict[str, str] = {}
        for m in request.messages:
            if getattr(m, "tool_calls", None):
                for tc in m.tool_calls:
                    cid = tc.get("id")
                    cname = tc.get("function", {}).get("name")
                    if cid and cname:
                        call_id_to_name[cid] = cname

        for m in request.messages:
            if m.role == "system":
                if isinstance(m.content, list):
                    for part in m.content:
                        if isinstance(part, str):
                            system_instruction_parts.append({"text": part})
                        elif isinstance(part, dict):
                            if is_image_part(part):
                                continue
                            text_part: Dict[str, Any] = {
                                "text": part.get("text", str(part))
                            }
                            if "cache_control" in part:
                                text_part["cache_control"] = part["cache_control"]
                            system_instruction_parts.append(text_part)
                elif m.content:
                    system_instruction_parts.append({"text": str(m.content)})
            elif m.role in ("tool", "function"):
                call_id = m.tool_call_id or getattr(m, "id", None)
                func_name = m.name or (call_id_to_name.get(call_id) if call_id else None) or "tool"
                resp_content = m.content
                if isinstance(resp_content, str):
                    try:
                        parsed_resp = json.loads(resp_content)
                        if not isinstance(parsed_resp, dict):
                            parsed_resp = {"output": resp_content}
                    except Exception:
                        parsed_resp = {"output": resp_content}
                elif isinstance(resp_content, dict):
                    parsed_resp = resp_content
                else:
                    parsed_resp = {"output": str(resp_content)}

                fr_part: Dict[str, Any] = {
                    "name": func_name,
                    "response": parsed_resp,
                }
                if call_id:
                    fr_part["id"] = call_id
                contents.append({"role": "user", "parts": [{"functionResponse": fr_part}]})
            elif m.role == "user":
                parts: List[Dict[str, Any]] = []
                if isinstance(m.content, list):
                    for p in m.content:
                        if isinstance(p, str):
                            parts.append({"text": p})
                        elif isinstance(p, dict):
                            if is_image_part(p):
                                image_part = to_gemini_part(p)
                                if image_part is not None:
                                    parts.append(image_part)
                                else:
                                    parts.append({"text": "[Image could not be loaded]"})
                                continue
                            ptype = p.get("type")
                            if ptype in ("text", "input_text"):
                                part: Dict[str, Any] = {"text": p.get("text", "")}
                            else:
                                part = dict(p)
                            if "cache_control" in p:
                                part["cache_control"] = p["cache_control"]
                            parts.append(part)
                else:
                    parts.append({"text": str(m.content)})
                contents.append({
                    "role": "user",
                    "parts": parts or [{"text": ""}]
                })
            elif m.role in ("assistant", "model"):
                parts = []
                if m.content:
                    text_val = content_text(m.content)
                    # Strip reasoning tags (e.g. <think>...</think>, <thought>...</thought>, <thinking>...</thinking>)
                    # including unclosed trailing tags to prevent upstream model prompt poisoning.
                    cleaned_text = re.sub(
                        r"<(think|thought|thinking)>[\s\S]*?(?:</\1>|$)",
                        "",
                        text_val,
                        flags=re.IGNORECASE,
                    ).strip()
                    if cleaned_text:
                        parts.append({"text": cleaned_text})
                if getattr(m, "tool_calls", None):
                    for tc in m.tool_calls:
                        call_id = tc.get("id")
                        func = tc.get("function", {})
                        name = func.get("name", "")
                        raw_args = func.get("arguments", "{}")
                        if isinstance(raw_args, str):
                            try:
                                parsed_args = json.loads(raw_args)
                            except Exception:
                                parsed_args = {}
                        else:
                            parsed_args = raw_args or {}

                        fc_part: Dict[str, Any] = {
                            "name": name,
                            "args": parsed_args,
                        }
                        if call_id:
                            fc_part["id"] = call_id

                        part_entry: Dict[str, Any] = {"functionCall": fc_part}
                        thought_sig = _get_thought_signature(call_id, name, parsed_args)
                        if thought_sig:
                            part_entry["thoughtSignature"] = thought_sig
                        parts.append(part_entry)
                if not parts:
                    parts.append({"text": ""})
                contents.append({
                    "role": "model",
                    "parts": parts,
                })

        if not contents:
            contents.append({
                "role": "user",
                "parts": [{"text": "Hello"}]
            })

        supports_block_cache = "claude" in target_model.lower()
        if not supports_block_cache:
            # Gemini uses cachedContent resources instead of block-level breakpoints.
            for item in contents:
                for part in item.get("parts", []):
                    if isinstance(part, dict):
                        part.pop("cache_control", None)
            for part in system_instruction_parts:
                part.pop("cache_control", None)

        max_tokens = request.max_tokens or model_config.max_tokens or 65536
        generation_config: Dict[str, Any] = {
            "maxOutputTokens": max_tokens,
        }
        if enable_reasoning:
            thinking_level = "HIGH" if reasoning_level in ("high", "default") else ("LOW" if reasoning_level == "low" else "MEDIUM")
            generation_config["thinkingConfig"] = {
                "includeThoughts": True,
                "thinkingLevel": thinking_level,
            }
        else:
            generation_config["thinkingConfig"] = {
                "includeThoughts": False,
            }
        if request.temperature is not None and request.temperature != 1.0:
            generation_config["temperature"] = request.temperature
        if request.top_p is not None and request.top_p != 1.0:
            generation_config["topP"] = request.top_p

        inner_request: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": generation_config,
            "sessionId": session_id_str,
        }
        if system_instruction_parts:
            inner_request["systemInstruction"] = {
                "role": "user",
                "parts": system_instruction_parts,
            }

        self._apply_cache_fields(
            inner_request, request, supports_block_cache=supports_block_cache
        )

        gemini_tools = _convert_openai_tools_to_gemini(request.tools)
        if gemini_tools:
            inner_request["tools"] = gemini_tools
            gemini_tool_config = _convert_tool_choice_to_gemini(request.tool_choice)
            if gemini_tool_config:
                inner_request["toolConfig"] = gemini_tool_config

        body: Dict[str, Any] = {
            "project": project_id,
            "requestId": request_id,
            "request": inner_request,
            "model": target_model,
            "userAgent": "antigravity",
        }

        return headers, body

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        variables: Dict[str, Any],
        account: Optional[Account] = None,
    ) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionChunk]]:
        start_t = time.perf_counter()
        sanitize_ssl_env()
        api_key = account.api_key if account else str(variables.get("api_key", ""))
        base_url = (
            account.base_url
            if (account and account.base_url)
            else str(variables.get("base_url") or CLOUDCODE_PA_ENDPOINTS[0])
        ).rstrip("/")
        timeout = float(variables.get("timeout") or 60.0)

        # Resolve reasoning level and actual model name
        actual_model, reasoning_level, enable_reasoning = self._parse_reasoning_level(
            request, model_config, variables
        )

        if self._is_test_mode(api_key):
            time_taken_ms = max(0.5, round((time.perf_counter() - start_t) * 1000.0, 2))
            if request.stream:
                return self._mock_stream(
                    request,
                    model_config,
                    account=account,
                    actual_model=actual_model,
                    enable_reasoning=enable_reasoning,
                    time_taken_ms=time_taken_ms,
                )
            return self._mock_response(
                request,
                model_config,
                account=account,
                actual_model=actual_model,
                enable_reasoning=enable_reasoning,
                time_taken_ms=time_taken_ms,
            )

        await self._ensure_token_freshness(account)

        proxy = str(variables.get("proxy") or (getattr(account, "proxy_url", None) or "")).strip() or None

        # Gemini accepts inline image data; download remote URLs before building
        # the request so vision works with ordinary OpenAI ``image_url`` parts.
        if messages_have_images(request.messages):
            materialized = await materialize_message_images(
                request.messages,
                timeout=min(timeout, 30.0),
                proxy=proxy,
            )
            if materialized is not request.messages:
                request = request.model_copy(update={"messages": materialized})

        headers, body = self._build_upstream_request(
            request,
            model_config,
            account,
            actual_model=actual_model,
            enable_reasoning=enable_reasoning,
            reasoning_level=reasoning_level,
        )
        target_model = model_config.id

        url = f"{base_url}/v1internal:streamGenerateContent?alt=sse"

        if request.stream:
            async def sse_stream_generator() -> AsyncIterator[ChatCompletionChunk]:
                chunk_id = f"chatcmpl-ag-{uuid.uuid4().hex[:12]}"
                created = int(time.time())
                current_thought_sig = None
                tool_call_idx = 0
                has_tool_call = False
                in_thought = False
                thought_ended_with_newline = False
                last_usage = CompletionUsage(
                    prompt_tokens=0, completion_tokens=0, total_tokens=0
                )

                yield ChatCompletionChunk(
                    id=chunk_id,
                    created=created,
                    model=target_model,
                    choices=[ChatChoiceDelta(index=0, delta={"role": "assistant", "content": ""})],
                )

                async with httpx.AsyncClient(timeout=timeout, proxy=proxy) as client:
                    async def get_stream():
                        async with client.stream("POST", url, headers=headers, json=body) as stream_resp:
                            if stream_resp.status_code == 401 and account and account.extra_config.get("refresh_token"):
                                refreshed = await antigravity_oauth.refresh_account_token(account)
                                if refreshed:
                                    new_headers = self.build_headers(account=account)
                                    async with client.stream("POST", url, headers=new_headers, json=body) as retry_resp:
                                        if retry_resp.status_code >= 400:
                                            err_bytes = await retry_resp.aread()
                                            raise httpx.HTTPStatusError(
                                                f"Google Antigravity upstream error ({retry_resp.status_code}): {err_bytes.decode('utf-8', errors='replace')}",
                                                request=retry_resp.request,
                                                response=retry_resp,
                                            )
                                        async for line in retry_resp.aiter_lines():
                                            yield line
                                    return
                            if stream_resp.status_code >= 400:
                                err_bytes = await stream_resp.aread()
                                raise httpx.HTTPStatusError(
                                    f"Google Antigravity upstream error ({stream_resp.status_code}): {err_bytes.decode('utf-8', errors='replace')}",
                                    request=stream_resp.request,
                                    response=stream_resp,
                                )
                            async for line in stream_resp.aiter_lines():
                                yield line

                    async for line in get_stream():
                        if not line or not line.startswith("data: "):
                            continue
                        raw_data = line[6:].strip()
                        if raw_data == "[DONE]":
                            break
                        try:
                            payload = json.loads(raw_data)
                            response_obj = payload.get("response", {})
                            usage_meta = response_obj.get("usageMetadata", {})
                            if usage_meta:
                                last_usage = self._build_usage(
                                    usage_meta,
                                    default_prompt=last_usage.prompt_tokens,
                                    default_completion=last_usage.completion_tokens,
                                    previous=last_usage,
                                )
                            candidates = response_obj.get("candidates", [])
                            for cand in candidates:
                                parts = cand.get("content", {}).get("parts", [])
                                for part in parts:
                                    if "thoughtSignature" in part and part["thoughtSignature"]:
                                        current_thought_sig = part["thoughtSignature"]

                                    if "functionCall" in part:
                                        fc = part["functionCall"]
                                        name = fc.get("name", "")
                                        args = fc.get("args", {})
                                        call_id = fc.get("id") or f"call_{uuid.uuid4().hex[:12]}"
                                        has_tool_call = True
                                        if current_thought_sig:
                                            _save_thought_signature(call_id, name, args, current_thought_sig)

                                        yield ChatCompletionChunk(
                                            id=chunk_id,
                                            created=created,
                                            model=target_model,
                                            choices=[
                                                ChatChoiceDelta(
                                                    index=0,
                                                    delta={
                                                        "tool_calls": [
                                                            {
                                                                "index": tool_call_idx,
                                                                "id": call_id,
                                                                "type": "function",
                                                                "function": {
                                                                    "name": name,
                                                                    "arguments": json.dumps(args),
                                                                },
                                                            }
                                                        ]
                                                    },
                                                    finish_reason=None,
                                                )
                                            ],
                                        )
                                        tool_call_idx += 1
                                    elif "text" in part and part["text"]:
                                        is_thought = self._is_thought_part(part)
                                        if is_thought:
                                            raw_text = part["text"]
                                            yield ChatCompletionChunk(
                                                id=chunk_id,
                                                created=created,
                                                model=target_model,
                                                choices=[
                                                    ChatChoiceDelta(
                                                        index=0,
                                                        delta={"reasoning_content": raw_text},
                                                    )
                                                ],
                                            )
                                        else:
                                            yield ChatCompletionChunk(
                                                id=chunk_id,
                                                created=created,
                                                model=target_model,
                                                choices=[
                                                    ChatChoiceDelta(
                                                        index=0,
                                                        delta={"content": part["text"]},
                                                    )
                                                ],
                                            )
                        except Exception:
                            continue

                    finish_reason = "tool_calls" if has_tool_call else "stop"
                    yield ChatCompletionChunk(
                        id=chunk_id,
                        created=created,
                        model=target_model,
                        choices=[ChatChoiceDelta(index=0, delta={}, finish_reason=finish_reason)],
                        usage=last_usage,
                    )

            return sse_stream_generator()

        # Non-streaming: accumulate response parts
        thought_parts: List[str] = []
        regular_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        current_thought_sig = None
        last_usage = CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

        async with httpx.AsyncClient(timeout=timeout, proxy=proxy) as client:
            async def get_stream():
                async with client.stream("POST", url, headers=headers, json=body) as stream_resp:
                    if stream_resp.status_code == 401 and account and account.extra_config.get("refresh_token"):
                        refreshed = await antigravity_oauth.refresh_account_token(account)
                        if refreshed:
                            new_headers = self.build_headers(account=account)
                            async with client.stream("POST", url, headers=new_headers, json=body) as retry_resp:
                                if retry_resp.status_code >= 400:
                                    err_bytes = await retry_resp.aread()
                                    raise httpx.HTTPStatusError(
                                        f"Google Antigravity upstream error ({retry_resp.status_code}): {err_bytes.decode('utf-8', errors='replace')}",
                                        request=retry_resp.request,
                                        response=retry_resp,
                                    )
                                async for line in retry_resp.aiter_lines():
                                    yield line
                            return
                    if stream_resp.status_code >= 400:
                        err_bytes = await stream_resp.aread()
                        raise httpx.HTTPStatusError(
                            f"Google Antigravity upstream error ({stream_resp.status_code}): {err_bytes.decode('utf-8', errors='replace')}",
                            request=stream_resp.request,
                            response=stream_resp,
                        )
                    async for line in stream_resp.aiter_lines():
                        yield line

            async for line in get_stream():
                if not line or not line.startswith("data: "):
                    continue
                raw_data = line[6:].strip()
                if raw_data == "[DONE]":
                    break
                try:
                    payload = json.loads(raw_data)
                    resp_obj = payload.get("response", {})
                    usage = resp_obj.get("usageMetadata", {})
                    if usage:
                        last_usage = self._build_usage(
                            usage,
                            default_prompt=last_usage.prompt_tokens,
                            default_completion=last_usage.completion_tokens,
                            previous=last_usage,
                        )

                    candidates = resp_obj.get("candidates", [])
                    for cand in candidates:
                        parts = cand.get("content", {}).get("parts", [])
                        for part in parts:
                            if "thoughtSignature" in part and part["thoughtSignature"]:
                                current_thought_sig = part["thoughtSignature"]
                            if "functionCall" in part:
                                fc = part["functionCall"]
                                name = fc.get("name", "")
                                args = fc.get("args", {})
                                call_id = fc.get("id") or f"call_{uuid.uuid4().hex[:12]}"
                                if current_thought_sig:
                                    _save_thought_signature(call_id, name, args, current_thought_sig)
                                tool_calls.append({
                                    "id": call_id,
                                    "type": "function",
                                    "function": {
                                        "name": name,
                                        "arguments": json.dumps(args),
                                    },
                                })
                            elif "text" in part and part["text"]:
                                is_thought = self._is_thought_part(part)
                                if is_thought:
                                    thought_parts.append(part["text"])
                                else:
                                    regular_parts.append(part["text"])
                except Exception:
                    continue

        thought_text = "".join(thought_parts)
        regular_text = "".join(regular_parts)

        finish_reason = "tool_calls" if tool_calls else "stop"
        msg_content = regular_text if (regular_text or not tool_calls) else None

        msg_payload: Dict[str, Any] = {
            "role": "assistant",
            "content": msg_content,
            "tool_calls": tool_calls if tool_calls else None,
        }
        if thought_text:
            msg_payload["reasoning_content"] = thought_text

        return ChatCompletionResponse(
            id=f"chatcmpl-ag-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=request.model or target_model,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage.model_validate(msg_payload),
                    finish_reason=finish_reason,
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=last_usage.prompt_tokens or 15,
                completion_tokens=last_usage.completion_tokens or max(1, len(regular_text) // 4),
                total_tokens=(
                    last_usage.total_tokens
                    or (last_usage.prompt_tokens + last_usage.completion_tokens)
                ),
                **(
                    {"prompt_tokens_details": last_usage.prompt_tokens_details}
                    if getattr(last_usage, "prompt_tokens_details", None)
                    else {}
                ),
                **(
                    {"cache_read_input_tokens": last_usage.cache_read_input_tokens}
                    if getattr(last_usage, "cache_read_input_tokens", None) is not None
                    else {}
                ),
            ),
        )


plugin = AntigravityPlugin()
