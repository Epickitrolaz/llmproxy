"""
OpenAI Codex Provider Plugin
============================

Native OpenAI Codex provider with official ChatGPT OAuth 2.0 PKCE authentication,
direct wire protocol communication matching official Codex CLI and 9Router,
multi-tier reasoning controls (none to max), and multi-account fallback/load balancing.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import http.server
import json
import logging
import os
from pathlib import Path
import re
import secrets
import socket
import threading
import time
import urllib.parse
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Set, Tuple, Union

import httpx

from llmproxy.core.interface import BasePlugin

CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_SCOPES = [
    "openid",
    "profile",
    "email",
    "offline_access",
    "api.connectors.read",
    "api.connectors.invoke",
]
CODEX_AUTH_URI = "https://auth.openai.com/oauth/authorize"
CODEX_TOKEN_URI = "https://auth.openai.com/oauth/token"
CODEX_BACKEND_API = "https://chatgpt.com/backend-api"
CODEX_USER_AGENT = "codex/cli/0.167.0 (linux; x86_64)"
from llmproxy.core.multimodal import (
    content_text,
    is_image_part,
    to_openai_image_part,
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

# Public model IDs are namespaced by provider. Reasoning levels are advertised
# only on models that accept the Codex reasoning control.
CODEX_REASONING_LEVELS = ("none", "low", "medium", "high", "xhigh", "max", "ultra")
CODEX_REASONING_MODELS = {
    "gpt-6-luna",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gpt-reserve",
    "codex-auto-review",
    "gpt-6-astra",
    "gpt-6.1-sol",
}


class CodexPlugin(BasePlugin):
    """
    OpenAI Codex Provider Plugin.
    Communicates directly with upstream OpenAI / ChatGPT backend API over the network
    matching the official Codex CLI and 9Router wire protocols identically.
    Supports reasoning models (gpt-6-luna, gpt-5.6-terra, gpt-5.6-luna, etc.),
    multi-account load balancing, and authentic execution telemetry.
    """

    DISALLOWED_FIELDS: Set[str] = {
        "service_tier",
        "disallowed_field",
        "unsupported_field",
    }

    def __init__(self):
        self._metadata = PluginMetadata(
            id="codex",
            name="Codex",
            description="OpenAI Codex frontier models provider with direct ChatGPT OAuth 2.0 network wire protocol, reasoning controls, and multi-account load balancing",
            version="3.1.0",
            author="OpenAI / Codex",
            enabled=True,
            filename="codex.py",
            oauth_login=True,
            oauth_local_import=True,
            oauth_hint="ChatGPT OAuth 2.0 PKCE loopback redirect with auto-refreshing tokens",
            variables=[
                PluginVariable(
                    name="timeout",
                    label="Timeout (seconds)",
                    type="number",
                    default=60,
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
                    default=CODEX_BACKEND_API,
                    description="OpenAI Codex backend API base URL",
                ),
                PluginVariable(
                    name="reasoning_effort",
                    label="Reasoning Effort",
                    type="select",
                    default="medium",
                    options=["none", "low", "medium", "high", "xhigh", "max"],
                    description="Default reasoning effort for supported Codex models",
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
                    id="cx/gpt-6-luna",
                    name="GPT-6 Luna",
                    upstream_model="gpt-6-luna",
                    context_window=272000,
                    max_tokens=65536,
                ),
                PluginModelConfig(
                    id="cx/gpt-5.6-terra",
                    name="GPT-5.6 Terra",
                    upstream_model="gpt-5.6-terra",
                    context_window=272000,
                    max_tokens=65536,
                ),
                PluginModelConfig(
                    id="cx/gpt-5.6-luna",
                    name="GPT-5.6 Luna",
                    upstream_model="gpt-5.6-luna",
                    context_window=272000,
                    max_tokens=65536,
                ),
                PluginModelConfig(
                    id="cx/gpt-reserve",
                    name="GPT Reserve",
                    upstream_model="gpt-reserve",
                    context_window=272000,
                    max_tokens=65536,
                ),
                PluginModelConfig(
                    id="cx/codex-auto-review",
                    name="Codex Auto Review",
                    upstream_model="codex-auto-review",
                    context_window=272000,
                    max_tokens=65536,
                ),
                PluginModelConfig(
                    id="cx/gpt-6-astra",
                    name="GPT-6 Astra",
                    upstream_model="gpt-6-astra",
                    context_window=272000,
                    max_tokens=65536,
                ),
                PluginModelConfig(
                    id="cx/gpt-6.1-sol",
                    name="GPT-6.1 Sol",
                    upstream_model="gpt-6.1-sol",
                    context_window=272000,
                    max_tokens=65536,
                ),
                PluginModelConfig(
                    id="cx/gpt-4o",
                    name="GPT-4o",
                    upstream_model="gpt-4o",
                    context_window=128000,
                    max_tokens=4096,
                ),
                PluginModelConfig(
                    id="cx/gpt-4o-mini",
                    name="GPT-4o Mini",
                    upstream_model="gpt-4o-mini",
                    context_window=128000,
                    max_tokens=4096,
                ),
            ],
        )
        for model in self._metadata.models:
            model.supports_vision = True
            if (model.upstream_model or model.id) in CODEX_REASONING_MODELS:
                model.reasoning_levels = list(CODEX_REASONING_LEVELS)

    @property
    def metadata(self) -> PluginMetadata:
        return self._metadata

    @classmethod
    def generate_sha256_uuid(cls, seed: str) -> str:
        """Derives an RFC 4122 compliant UUID string deterministically from a SHA-256 digest."""
        digest = hashlib.sha256(seed.encode("utf-8")).digest()
        return str(uuid.UUID(bytes=digest[:16]))

    @classmethod
    def generate_session_id(cls, seed: Optional[str] = None) -> str:
        """Generates a standard UUID session identifier for Codex."""
        if seed:
            return cls.generate_sha256_uuid(seed)
        return str(uuid.uuid4())

    def build_headers(
        self, account: Optional[Account] = None, api_key: str = ""
    ) -> Dict[str, str]:
        """Constructs authentic HTTP headers matching official Codex CLI and 9Router wire protocols."""
        key = account.api_key if account else api_key
        headers = {
            "Content-Type": "application/json",
            "User-Agent": CODEX_USER_AGENT,
            "Accept": "text/event-stream",
        }
        if key:
            headers["Authorization"] = f"Bearer {key}"
        if account and account.extra_config.get("account_id"):
            headers["ChatGPT-Account-Id"] = account.extra_config["account_id"]
        return headers

    def sanitize_payload(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Strips disallowed parameters from the payload."""
        return {k: v for k, v in body.items() if k not in self.DISALLOWED_FIELDS}

    def _is_test_mode(self, api_key: str, account: Optional[Account] = None) -> bool:
        key = (api_key or "").lower().strip()
        if not key:
            if account is not None and not str(account.id).startswith("default-"):
                raise ValueError(
                    f"Account '{account.name}' has no API key configured."
                )
            return True
        prefixes = ("test", "mock", "sk-test", "dummy", "fake", "offline", "sample")
        return any(key.startswith(p) for p in prefixes) or key in ("test-key", "test_key", "secret123")

    def _parse_reasoning_effort(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        variables: Dict[str, Any],
    ) -> Optional[str]:
        """
        Parse and resolve reasoning effort for Codex.
        Supported levels: 'low', 'medium', 'high', 'xhigh', 'max', 'ultra', 'default', 'none' / 'off'.
        Checks request.reasoning_effort, request.extra_body, model name suffix, or variables.
        """
        raw = None
        if hasattr(request, "reasoning_effort") and request.reasoning_effort:
            raw = str(request.reasoning_effort).strip().lower()
        elif request.extra_body and ("reasoning_effort" in request.extra_body or "reasoning" in request.extra_body):
            raw = str(request.extra_body.get("reasoning_effort") or request.extra_body.get("reasoning") or "").strip().lower()

        # Check model name suffix (e.g. gpt-6-luna:high, gpt-6-luna-high, gpt-6-luna:none)
        if not raw and request.model:
            m_lower = request.model.lower()
            for suffix, level in [
                (":none", "none"), ("-none", "none"), (":off", "none"), ("-off", "none"),
                (":low", "low"), ("-low", "low"),
                (":medium", "medium"), ("-medium", "medium"),
                (":high", "high"), ("-high", "high"),
                (":xhigh", "xhigh"), ("-xhigh", "xhigh"),
                (":max", "max"), ("-max", "max"),
                (":ultra", "ultra"), ("-ultra", "ultra"),
            ]:
                if m_lower.endswith(suffix):
                    raw = level
                    break

        if not raw:
            var_val = str(variables.get("reasoning_effort") or "").strip().lower()
            if var_val and var_val != "default":
                raw = var_val

        if not raw or raw in ("default", ""):
            return "default"

        if raw in ("none", "off", "disabled", "false", "0"):
            return "none"

        if raw in ("low", "medium", "high", "xhigh", "max", "ultra"):
            return raw

        return "default"

    @staticmethod
    def _apply_cache_fields(payload: Dict[str, Any], request: ChatCompletionRequest) -> None:
        """Forward OpenAI prompt-cache routing hints to the Responses API."""
        extra = request.extra_body if isinstance(request.extra_body, dict) else {}
        for key in ("prompt_cache_key", "prompt_cache_retention"):
            value = getattr(request, key, None)
            if value is None:
                value = extra.get(key)
            if value is not None:
                payload[key] = value

    @staticmethod
    def _build_usage(
        usage_obj: Optional[Dict[str, Any]],
        default_prompt: int = 0,
        default_completion: int = 0,
    ) -> CompletionUsage:
        """Map Responses API usage, including cached input tokens, to OpenAI shape."""
        usage_obj = usage_obj or {}
        prompt_tokens = int(
            usage_obj.get("input_tokens")
            or usage_obj.get("prompt_tokens")
            or default_prompt
            or 0
        )
        completion_tokens = int(
            usage_obj.get("output_tokens")
            or usage_obj.get("completion_tokens")
            or default_completion
            or 0
        )
        total_tokens = int(
            usage_obj.get("total_tokens") or (prompt_tokens + completion_tokens)
        )

        usage = CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
        details = (
            usage_obj.get("input_tokens_details")
            or usage_obj.get("prompt_tokens_details")
            or {}
        )
        if details.get("cached_tokens") is not None:
            cached_tokens = int(details.get("cached_tokens") or 0)
            usage.prompt_tokens_details = {"cached_tokens": cached_tokens}
            usage.cache_read_input_tokens = cached_tokens
        output_details = usage_obj.get("output_tokens_details") or {}
        if output_details.get("reasoning_tokens") is not None:
            usage.completion_tokens_details = {
                "reasoning_tokens": int(output_details.get("reasoning_tokens") or 0)
            }
        return usage

    def _format_useful_details(
        self,
        time_taken_ms: float,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        account: Optional[Account] = None,
    ) -> str:
        """Constructs rich useful execution details to append to example responses."""
        now_str = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        status_desc = (
            "No account configured (Simulation Mode)"
            if (not account or account.id.startswith("default-") or not account.api_key)
            else f"Account '{account.name}' (Test mode)"
        )
        total_tokens = prompt_tokens + completion_tokens
        return (
            f"\n\n---\n"
            f"**Execution Details:**\n"
            f"• **Time taken:** {time_taken_ms:.2f} ms\n"
            f"• **Timestamp:** {now_str}\n"
            f"• **Model:** {model_name}\n"
            f"• **Tokens:** {prompt_tokens} prompt, {completion_tokens} completion ({total_tokens} total)\n"
            f"• **Account status:** {status_desc}"
        )

    def _mock_response(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        account: Optional[Account] = None,
        time_taken_ms: float = 0.0,
    ) -> ChatCompletionResponse:
        model_name = request.model or model_config.id
        acc_label = f" via account '{account.name}'" if account else ""
        if time_taken_ms <= 0.0:
            time_taken_ms = 1.25

        # Handle tool calling mock response
        if request.tools and not any(m.role in ("tool", "function") for m in request.messages[-2:]):
            fn = request.tools[0].get("function", {}) if "function" in request.tools[0] else request.tools[0]
            call_id = f"call_codex_{uuid.uuid4().hex[:8]}"
            tool_name = fn.get("name", "test_tool")
            mock_args = "{}"
            props = fn.get("parameters", {}).get("properties", {})
            if props:
                first_k = next(iter(props.keys()))
                mock_args = json.dumps({first_k: "test_value"})

            return ChatCompletionResponse(
                id=f"chatcmpl-codex-{uuid.uuid4().hex[:12]}",
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

        base_content = f"Simulated response from Codex ({model_name}){acc_label}."
        if any(m.role in ("tool", "function") for m in request.messages):
            base_content = f"Simulated response from Codex ({model_name}){acc_label} with tool output processed."

        prompt_tokens = 12
        completion_tokens = 22
        details = self._format_useful_details(
            time_taken_ms=time_taken_ms,
            model_name=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            account=account,
        )
        full_content = base_content + details

        return ChatCompletionResponse(
            id=f"chatcmpl-codex-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=model_name,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=full_content),
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    async def _mock_stream(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        account: Optional[Account] = None,
        time_taken_ms: float = 0.0,
    ) -> AsyncIterator[ChatCompletionChunk]:
        model_name = request.model or model_config.id
        chunk_id = f"chatcmpl-codex-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        acc_label = f" via account '{account.name}'" if account else ""
        if time_taken_ms <= 0.0:
            time_taken_ms = 1.25

        if request.tools and not any(m.role in ("tool", "function") for m in request.messages[-2:]):
            fn = request.tools[0].get("function", {}) if "function" in request.tools[0] else request.tools[0]
            call_id = f"call_codex_{uuid.uuid4().hex[:8]}"
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
                        delta={
                            "role": "assistant",
                            "content": None,
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

        base_tokens = ["Simulated ", "stream ", "response ", f"from Codex for {model_name}{acc_label}."]
        for token in base_tokens:
            yield ChatCompletionChunk(
                id=chunk_id,
                created=created,
                model=model_name,
                choices=[ChatChoiceDelta(index=0, delta={"content": token}, finish_reason=None)],
            )

        details = self._format_useful_details(
            time_taken_ms=time_taken_ms,
            model_name=model_name,
            prompt_tokens=12,
            completion_tokens=22,
            account=account,
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
        """Refreshes OpenAI OAuth access token if expiring or expired."""
        if not account:
            return
        refresh_token = account.extra_config.get("refresh_token")
        if not refresh_token:
            return

        expires_at = float(account.extra_config.get("expires_at", 0))
        now = time.time()
        if not account.api_key or (expires_at > 0 and now >= expires_at - 120):
            refreshed = await codex_oauth.refresh_account_token(account)
            if refreshed:
                from llmproxy.core.manager import plugin_manager
                plugin_manager._save_accounts()

    def _build_upstream_request(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        account: Optional[Account],
        variables: Dict[str, Any],
    ) -> Tuple[str, Dict[str, str], Dict[str, Any], str]:
        """
        Constructs upstream URL, authentic HTTP headers, and Responses payload
        matching official Codex CLI and 9Router wire protocols.
        """
        # Resolve target model and strip route/reasoning suffixes
        raw_model = request.model or model_config.upstream_model or model_config.id
        target_model = raw_model
        if target_model.startswith("cx/"):
            target_model = target_model[3:]
        for sfx in (
            ":none", "-none", ":off", "-off",
            ":low", "-low", ":medium", "-medium",
            ":high", "-high", ":xhigh", "-xhigh",
            ":max", "-max", ":ultra", "-ultra",
        ):
            if target_model.lower().endswith(sfx):
                target_model = target_model[:-len(sfx)]
                break

        parsed_effort = self._parse_reasoning_effort(request, model_config, variables)
        api_key = account.api_key if account else str(variables.get("api_key") or "")
        headers = self.build_headers(account=account, api_key=api_key)

        base = (
            account.base_url
            if (account and account.base_url)
            else str(variables.get("base_url") or CODEX_BACKEND_API)
        ).rstrip("/")
        if not base.endswith("/codex") and not base.endswith("/codex/responses"):
            url = f"{base}/codex/responses"
        elif base.endswith("/codex"):
            url = f"{base}/responses"
        else:
            url = base

        payload: Dict[str, Any] = {
            "model": target_model,
            "store": False,
            "stream": True,
        }

        if parsed_effort and parsed_effort not in ("default", "none"):
            payload["reasoning"] = {"effort": parsed_effort}
        elif parsed_effort == "none":
            payload["reasoning"] = {"effort": "none"}

        self._apply_cache_fields(payload, request)

        # Tools formatting (Responses API expects flat tool schemas)
        if request.tools:
            tools_list: List[Dict[str, Any]] = []
            for t in request.tools:
                if isinstance(t, dict):
                    if t.get("type") == "function" and "function" in t:
                        fn = t["function"]
                        tools_list.append({
                            "type": "function",
                            "name": fn.get("name"),
                            "description": fn.get("description", ""),
                            "parameters": fn.get("parameters", {}),
                        })
                    else:
                        tools_list.append(t)
                else:
                    tools_list.append(t)
            payload["tools"] = tools_list

        # Transform messages into Responses API input items
        input_items: List[Dict[str, Any]] = []
        for msg in request.messages:
            role = msg.role
            if role in ("system", "developer"):
                text = content_text(msg.content)
                input_items.append({"role": "developer", "content": [{"type": "input_text", "text": text}]})
            elif role == "user":
                if isinstance(msg.content, list):
                    parts: List[Dict[str, Any]] = []
                    for p in msg.content:
                        if isinstance(p, dict):
                            if is_image_part(p):
                                image_part = to_openai_image_part(p)
                                if image_part is not None:
                                    parts.append(image_part)
                                else:
                                    parts.append({"type": "input_text", "text": "[Image could not be loaded]"})
                                continue
                            ptype = p.get("type")
                            if ptype in ("text", "input_text"):
                                parts.append({"type": "input_text", "text": p.get("text", "")})
                            else:
                                text = p.get("text")
                                if isinstance(text, str):
                                    parts.append({"type": "input_text", "text": text})
                        elif isinstance(p, str):
                            parts.append({"type": "input_text", "text": p})
                    input_items.append({"role": "user", "content": parts})
                else:
                    text = content_text(msg.content)
                    input_items.append({"role": "user", "content": [{"type": "input_text", "text": text}]})
            elif role == "assistant":
                if getattr(msg, "tool_calls", None):
                    for tc in msg.tool_calls:
                        fn = tc.get("function", {}) if isinstance(tc, dict) else getattr(tc, "function", {})
                        cid = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                        fname = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", "")
                        fargs = fn.get("arguments") if isinstance(fn, dict) else getattr(fn, "arguments", "{}")
                        input_items.append({
                            "type": "function_call",
                            "call_id": cid or f"call_{uuid.uuid4().hex[:8]}",
                            "name": fname,
                            "arguments": fargs if isinstance(fargs, str) else json.dumps(fargs),
                        })
                if msg.content:
                    assistant_text = content_text(msg.content)
                    if assistant_text:
                        input_items.append({"role": "assistant", "content": [{"type": "output_text", "text": assistant_text}]})
            elif role in ("tool", "function"):
                cid = getattr(msg, "tool_call_id", None) or getattr(msg, "id", None) or ""
                out = content_text(msg.content) if msg.content is not None else ""
                input_items.append({
                    "type": "function_call_output",
                    "call_id": cid,
                    "output": out,
                })

        payload["input"] = input_items
        return url, headers, payload, target_model

    async def _execute_upstream_streaming(
        self,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        target_model: str,
        timeout: float,
        account: Optional[Account],
        proxy: Optional[str] = None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Streams real upstream response chunks over SSE directly matching Codex CLI wire protocol."""
        chunk_id = f"chatcmpl-codex-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        tool_call_indices: Dict[str, int] = {}
        has_tool_calls = False

        yield ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=target_model,
            choices=[ChatChoiceDelta(index=0, delta={"role": "assistant", "content": ""})],
        )

        async with httpx.AsyncClient(timeout=timeout, proxy=proxy) as client:
            async def get_stream():
                async with client.stream("POST", url, headers=headers, json=payload) as stream_resp:
                    # Token refresh retry on 401
                    if stream_resp.status_code == 401 and account and account.extra_config.get("refresh_token"):
                        refreshed = await codex_oauth.refresh_account_token(account)
                        if refreshed:
                            from llmproxy.core.manager import plugin_manager
                            plugin_manager._save_accounts()
                            new_headers = self.build_headers(account=account)
                            async with client.stream("POST", url, headers=new_headers, json=payload) as retry_resp:
                                if retry_resp.status_code >= 400:
                                    err_bytes = await retry_resp.aread()
                                    raise httpx.HTTPStatusError(
                                        f"OpenAI Codex upstream error ({retry_resp.status_code}): {err_bytes.decode('utf-8', errors='replace')}",
                                        request=retry_resp.request,
                                        response=retry_resp,
                                    )
                                async for line in retry_resp.aiter_lines():
                                    yield line
                            return
                    if stream_resp.status_code >= 400:
                        err_bytes = await stream_resp.aread()
                        raise httpx.HTTPStatusError(
                            f"OpenAI Codex upstream error ({stream_resp.status_code}): {err_bytes.decode('utf-8', errors='replace')}",
                            request=stream_resp.request,
                            response=stream_resp,
                        )
                    async for line in stream_resp.aiter_lines():
                        yield line

            async for line in get_stream():
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    event = json.loads(data_str)
                except Exception:
                    continue

                ev_type = event.get("type")
                if ev_type == "response.output_text.delta":
                    delta_text = event.get("delta", "")
                    if delta_text:
                        yield ChatCompletionChunk(
                            id=chunk_id,
                            created=created,
                            model=target_model,
                            choices=[ChatChoiceDelta(index=0, delta={"content": delta_text})],
                        )
                elif ev_type == "response.output_item.added":
                    item = event.get("item", {})
                    if item.get("type") == "function_call":
                        has_tool_calls = True
                        call_id = item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex[:8]}"
                        idx = len(tool_call_indices)
                        tool_call_indices[call_id] = idx
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
                                                "index": idx,
                                                "id": call_id,
                                                "type": "function",
                                                "function": {
                                                    "name": item.get("name", ""),
                                                    "arguments": "",
                                                },
                                            }
                                        ]
                                    },
                                )
                            ],
                        )
                elif ev_type == "response.function_call_arguments.delta":
                    arg_delta = event.get("delta", "")
                    item_id = event.get("item_id", "")
                    idx = tool_call_indices.get(item_id, 0)
                    if arg_delta:
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
                                                "index": idx,
                                                "function": {"arguments": arg_delta},
                                            }
                                        ]
                                    },
                                )
                            ],
                        )
                elif ev_type == "response.completed":
                    resp_obj = event.get("response", {})
                    usage_obj = resp_obj.get("usage", {})
                    usage = None
                    if usage_obj:
                        usage = self._build_usage(usage_obj)
                    finish_reason = "tool_calls" if has_tool_calls else "stop"
                    yield ChatCompletionChunk(
                        id=chunk_id,
                        created=created,
                        model=target_model,
                        choices=[ChatChoiceDelta(index=0, delta={}, finish_reason=finish_reason)],
                        usage=usage,
                    )

    async def _execute_upstream_non_streaming(
        self,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        target_model: str,
        timeout: float,
        account: Optional[Account],
        proxy: Optional[str] = None,
    ) -> ChatCompletionResponse:
        """Executes non-streaming request against upstream Responses API and returns assembled ChatCompletionResponse."""
        chunk_id = f"chatcmpl-codex-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        text_pieces: List[str] = []
        tool_calls_map: Dict[str, Dict[str, Any]] = {}
        usage = CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

        async with httpx.AsyncClient(timeout=timeout, proxy=proxy) as client:
            async def get_stream():
                async with client.stream("POST", url, headers=headers, json=payload) as stream_resp:
                    # Token refresh retry on 401
                    if stream_resp.status_code == 401 and account and account.extra_config.get("refresh_token"):
                        refreshed = await codex_oauth.refresh_account_token(account)
                        if refreshed:
                            from llmproxy.core.manager import plugin_manager
                            plugin_manager._save_accounts()
                            new_headers = self.build_headers(account=account)
                            async with client.stream("POST", url, headers=new_headers, json=payload) as retry_resp:
                                if retry_resp.status_code >= 400:
                                    err_bytes = await retry_resp.aread()
                                    raise httpx.HTTPStatusError(
                                        f"OpenAI Codex upstream error ({retry_resp.status_code}): {err_bytes.decode('utf-8', errors='replace')}",
                                        request=retry_resp.request,
                                        response=retry_resp,
                                    )
                                async for line in retry_resp.aiter_lines():
                                    yield line
                            return
                    if stream_resp.status_code >= 400:
                        err_bytes = await stream_resp.aread()
                        raise httpx.HTTPStatusError(
                            f"OpenAI Codex upstream error ({stream_resp.status_code}): {err_bytes.decode('utf-8', errors='replace')}",
                            request=stream_resp.request,
                            response=stream_resp,
                        )
                    async for line in stream_resp.aiter_lines():
                        yield line

            async for line in get_stream():
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    event = json.loads(data_str)
                except Exception:
                    continue

                ev_type = event.get("type")
                if ev_type == "response.output_text.delta":
                    delta_text = event.get("delta", "")
                    if delta_text:
                        text_pieces.append(delta_text)
                elif ev_type == "response.output_item.added":
                    item = event.get("item", {})
                    if item.get("type") == "function_call":
                        call_id = item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex[:8]}"
                        item_id = item.get("id") or call_id
                        tool_calls_map[item_id] = {
                            "id": call_id,
                            "type": "function",
                            "name": item.get("name", ""),
                            "arguments": item.get("arguments", ""),
                        }
                elif ev_type == "response.function_call_arguments.delta":
                    arg_delta = event.get("delta", "")
                    item_id = event.get("item_id", "")
                    if item_id in tool_calls_map:
                        tool_calls_map[item_id]["arguments"] += arg_delta
                elif ev_type == "response.output_item.done":
                    item = event.get("item", {})
                    if item.get("type") == "function_call":
                        item_id = item.get("id")
                        if item_id in tool_calls_map and item.get("arguments"):
                            tool_calls_map[item_id]["arguments"] = item["arguments"]
                elif ev_type == "response.completed":
                    resp_obj = event.get("response", {})
                    resp_id = resp_obj.get("id")
                    if resp_id:
                        chunk_id = resp_id
                    u_obj = resp_obj.get("usage", {})
                    if u_obj:
                        usage = self._build_usage(u_obj)

        content = "".join(text_pieces) if text_pieces else None
        if tool_calls_map:
            formatted_tools = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                    },
                }
                for tc in tool_calls_map.values()
            ]
            message = ChatMessage(role="assistant", content=content, tool_calls=formatted_tools)
            finish_reason = "tool_calls"
        else:
            message = ChatMessage(role="assistant", content=content or "")
            finish_reason = "stop"

        return ChatCompletionResponse(
            id=chunk_id,
            created=created,
            model=target_model,
            choices=[
                ChatChoice(
                    index=0,
                    message=message,
                    finish_reason=finish_reason,
                )
            ],
            usage=usage,
        )

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        variables: Dict[str, Any],
        account: Optional[Account] = None,
    ) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionChunk]]:
        start_t = time.perf_counter()
        sanitize_ssl_env()
        api_key = account.api_key if account else str(variables.get("api_key") or "")
        timeout = float(variables.get("timeout") or 60.0)

        # Simulation mode when no account / test key
        if self._is_test_mode(api_key, account):
            time_taken_ms = max(0.5, round((time.perf_counter() - start_t) * 1000.0, 2))
            if request.stream:
                return self._mock_stream(
                    request,
                    model_config,
                    account=account,
                    time_taken_ms=time_taken_ms,
                )
            return self._mock_response(
                request,
                model_config,
                account=account,
                time_taken_ms=time_taken_ms,
            )

        # Authenticated account execution (Direct upstream wire protocol)
        await self._ensure_token_freshness(account)
        url, headers, payload, target_model = self._build_upstream_request(
            request, model_config, account, variables
        )

        proxy = str(variables.get("proxy") or (getattr(account, "proxy_url", None) or "")).strip() or None

        if request.stream:
            return self._execute_upstream_streaming(
                url=url,
                headers=headers,
                payload=payload,
                target_model=target_model,
                timeout=timeout,
                account=account,
                proxy=proxy,
            )

        return await self._execute_upstream_non_streaming(
            url=url,
            headers=headers,
            payload=payload,
            target_model=target_model,
            timeout=timeout,
            account=account,
            proxy=proxy,
        )


    # ---------------------------------------------------------------- OAuth
    async def oauth_start(self) -> Dict[str, Any]:
        return codex_oauth.start_flow()

    async def oauth_status(self, flow_id: str) -> Dict[str, Any]:
        return await codex_oauth.check_flow_status(flow_id)

    async def oauth_callback(self, flow_id: str, callback_url: str) -> Dict[str, Any]:
        return await codex_oauth.complete_callback(flow_id, callback_url)

    async def oauth_import_local(self) -> Optional[Account]:
        return await codex_oauth.import_local_session()

    async def refresh_account(self, account: Account) -> bool:
        return await codex_oauth.refresh_account_token(account)


def sanitize_ssl_env() -> None:
    """Clear SSL certificate environment variables if they point to non-existent files to prevent FileNotFoundError."""
    for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        val = os.environ.get(var)
        if val and not os.path.exists(val):
            logger.warning(
                "Clearing invalid %s='%s' (file not found) to prevent SSL failures",
                var,
                val,
            )
            del os.environ[var]
    for var in ("SSL_CERT_DIR",):
        val = os.environ.get(var)
        if val and not os.path.isdir(val):
            logger.warning(
                "Clearing invalid %s='%s' (directory not found) to prevent SSL failures",
                var,
                val,
            )
            del os.environ[var]


sanitize_ssl_env()

# Official OpenAI Codex OAuth Client Configuration
CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_SCOPES = [
    "openid",
    "profile",
    "email",
    "offline_access",
    "api.connectors.read",
    "api.connectors.invoke",
]
CODEX_ISSUER = "https://auth.openai.com"
CODEX_AUTH_URI = "https://auth.openai.com/oauth/authorize"
CODEX_TOKEN_URI = "https://auth.openai.com/oauth/token"
CODEX_REVOKE_URI = "https://auth.openai.com/oauth/revoke"
CODEX_DEVICE_USERCODE_URI = "https://auth.openai.com/deviceauth/usercode"
CODEX_DEVICE_TOKEN_URI = "https://auth.openai.com/deviceauth/token"
CODEX_BACKEND_API = "https://chatgpt.com/backend-api"
CODEX_USER_AGENT = "codex/cli/0.167.0 (linux; x86_64)"


def decode_jwt_payload(token: str) -> Dict[str, Any]:
    """Decodes JWT payload without cryptographic signature verification."""
    if not token or "." not in token:
        return {}
    try:
        parts = token.split(".")
        if len(parts) >= 2:
            padded = parts[1] + "=" * (4 - len(parts[1]) % 4)
            return json.loads(base64.urlsafe_b64decode(padded).decode("utf-8", errors="replace"))
    except Exception as e:
        logger.debug("Failed to decode JWT payload: %s", e)
    return {}


class _CodexOAuthCallbackHandler(http.server.BaseHTTPRequestHandler):
    """Loopback callback HTTP server handler for Codex / OpenAI OAuth redirect."""

    def log_message(self, format: str, *args: Any) -> None:
        pass  # Suppress default server access logging

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)

        code = params.get("code", [None])[0]
        state = params.get("state", [None])[0]
        error = params.get("error", [None])[0]

        self.server.received_code = code
        self.server.received_state = state
        self.server.received_error = error

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()

        success_html = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Codex Sign-In Complete • LLMProxy</title>
  <style>
    body { background: #09090b; color: #f4f4f5; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; }
    .box { background: #121214; border: 1px solid #27272a; border-radius: 8px; padding: 32px 40px; text-align: center; max-width: 420px; box-shadow: 0 4px 20px rgba(0,0,0,0.5); }
    h2 { font-size: 18px; font-weight: 600; margin: 0 0 8px 0; color: #ffffff; }
    p { font-size: 13px; color: #a1a1aa; line-height: 1.5; margin: 0 0 20px 0; }
    .badge { display: inline-block; background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); color: #34d399; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 500; }
  </style>
</head>
<body>
  <div class="box">
    <div style="font-size: 32px; margin-bottom: 12px;">✓</div>
    <h2>ChatGPT / Codex Account Connected</h2>
    <p>Your OpenAI Codex account has been authenticated. You can safely close this window and return to the LLMProxy dashboard.</p>
    <span class="badge">Connection Active</span>
  </div>
  <script>setTimeout(function() { window.close(); }, 2500);</script>
</body>
</html>"""
        self.wfile.write(success_html.encode("utf-8"))


class CodexOAuthManager:
    """Manages OpenAI Codex OAuth flows, PKCE token exchange, token refresh, and local import."""

    sanitize_ssl_env = staticmethod(sanitize_ssl_env)

    def __init__(self):
        self.active_flows: Dict[str, Dict[str, Any]] = {}

    def _find_free_port(self) -> int:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        return port

    @staticmethod
    def generate_pkce() -> Tuple[str, str]:
        """Generates RFC 7636 PKCE code_verifier and code_challenge (S256)."""
        verifier = secrets.token_urlsafe(64)
        digest = hashlib.sha256(verifier.encode("utf-8")).digest()
        challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("utf-8")
        return verifier, challenge

    def start_flow(self) -> Dict[str, Any]:
        """Start a new OAuth flow with PKCE, loopback listener, and authorization URL."""
        flow_id = f"flow_codex_{uuid.uuid4().hex[:12]}"
        state = uuid.uuid4().hex
        verifier, challenge = self.generate_pkce()
        port = self._find_free_port()
        redirect_uri = f"http://127.0.0.1:{port}/auth/callback"

        # Build OpenAI Codex OAuth URL with S256 PKCE matching official Codex CLI
        query = urllib.parse.urlencode({
            "response_type": "code",
            "client_id": CODEX_CLIENT_ID,
            "redirect_uri": redirect_uri,
            "scope": " ".join(CODEX_SCOPES),
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "id_token_add_organizations": "true",
            "codex_cli_simplified_flow": "true",
            "originator": "codex_cli_rs",
        })
        auth_url = f"{CODEX_AUTH_URI}?{query}"

        # Start loopback server
        server = http.server.HTTPServer(("127.0.0.1", port), _CodexOAuthCallbackHandler)
        server.received_code = None
        server.received_state = None
        server.received_error = None

        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()

        flow_info = {
            "flow_id": flow_id,
            "state": state,
            "code_verifier": verifier,
            "code_challenge": challenge,
            "port": port,
            "redirect_uri": redirect_uri,
            "auth_url": auth_url,
            "server": server,
            "thread": thread,
            "created_at": time.time(),
            "status": "waiting",
            "account": None,
            "error": None,
        }
        self.active_flows[flow_id] = flow_info
        return {
            "flow_id": flow_id,
            "mode": "loopback",
            "auth_url": auth_url,
            "redirect_uri": redirect_uri,
            "state": state,
            "supports_callback_paste": True,
        }

    async def check_flow_status(self, flow_id: str) -> Dict[str, Any]:
        """Check if loopback callback received code or if flow finished."""
        flow = self.active_flows.get(flow_id)
        if not flow:
            return {"status": "error", "error": "Codex OAuth flow not found or expired"}

        if flow["status"] == "complete":
            return {"status": "complete", "account": flow["account"]}
        if flow["status"] == "error":
            return {"status": "error", "error": flow["error"]}

        server = flow.get("server")
        if server and server.received_code:
            code = server.received_code
            redirect_uri = flow["redirect_uri"]
            code_verifier = flow["code_verifier"]
            try:
                account = await self.exchange_code_and_create_account(
                    code=code,
                    redirect_uri=redirect_uri,
                    code_verifier=code_verifier,
                )
                flow["status"] = "complete"
                flow["account"] = account
                return {"status": "complete", "account": account}
            except Exception as e:
                flow["status"] = "error"
                flow["error"] = str(e)
                return {"status": "error", "error": str(e)}

        if server and server.received_error:
            flow["status"] = "error"
            flow["error"] = f"OpenAI OAuth error: {server.received_error}"
            return {"status": "error", "error": flow["error"]}

        # 10 minute expiration
        if time.time() - flow["created_at"] > 600:
            flow["status"] = "error"
            flow["error"] = "OAuth flow timed out"
            return {"status": "error", "error": "OAuth flow timed out"}

        return {"status": "waiting"}

    async def complete_callback(self, flow_id: str, callback_url: str) -> Dict[str, Any]:
        """Complete a loopback flow from a pasted redirect URL (headless installs)."""
        flow = self.active_flows.get(flow_id)
        if not flow:
            return {"status": "error", "error": "Codex OAuth flow not found or expired"}
        parsed = urllib.parse.urlparse(callback_url or "")
        params = urllib.parse.parse_qs(parsed.query)
        code = params.get("code", [None])[0]
        state = params.get("state", [None])[0]
        if not code:
            return {"status": "error", "error": "Could not extract Codex authorization code"}
        verifier = flow.get("code_verifier")
        if state:
            for candidate in self.active_flows.values():
                if candidate.get("state") == state:
                    flow = candidate
                    verifier = candidate.get("code_verifier")
                    break
            else:
                return {"status": "error", "error": "OAuth state mismatch"}
        redirect_uri = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        try:
            account = await self.exchange_code_and_create_account(
                code=code, redirect_uri=redirect_uri, code_verifier=verifier
            )
            flow["status"] = "complete"
            flow["account"] = account
            return {"status": "complete", "account": account}
        except Exception as exc:
            flow["status"] = "error"
            flow["error"] = str(exc)
            return {"status": "error", "error": str(exc)}

    async def exchange_code_and_create_account(
        self,
        code: str,
        redirect_uri: str,
        code_verifier: Optional[str] = None,
    ) -> Account:
        """Exchange authorization code for tokens, parse JWT profile, and return Account."""
        sanitize_ssl_env()
        data = {
            "client_id": CODEX_CLIENT_ID,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": redirect_uri,
        }
        if code_verifier:
            data["code_verifier"] = code_verifier

        async with httpx.AsyncClient(timeout=15.0) as client:
            token_resp = await client.post(CODEX_TOKEN_URI, data=data)
            if token_resp.status_code >= 400:
                raise ValueError(f"Token exchange failed ({token_resp.status_code}): {token_resp.text}")

            token_data = token_resp.json()
            access_token = token_data.get("access_token", "")
            refresh_token = token_data.get("refresh_token", "")
            id_token = token_data.get("id_token", "")
            expires_in = int(token_data.get("expires_in", 864000))

            if not access_token:
                raise ValueError("No access_token returned by OpenAI OAuth")

            # Parse user information and chatgpt_account_id from JWT tokens
            id_claims = decode_jwt_payload(id_token)
            access_claims = decode_jwt_payload(access_token)

            auth_meta = access_claims.get("https://api.openai.com/auth", {}) or id_claims.get("https://api.openai.com/auth", {})
            profile_meta = access_claims.get("https://api.openai.com/profile", {})

            email = id_claims.get("email") or profile_meta.get("email") or "ChatGPT Account"
            name = id_claims.get("name") or profile_meta.get("name") or "Codex User"
            account_id = auth_meta.get("chatgpt_account_id") or ""
            user_id = auth_meta.get("chatgpt_user_id") or ""
            plan_type = auth_meta.get("chatgpt_plan_type") or "standard"

            account = Account(
                id=f"acc_codex_{uuid.uuid4().hex[:8]}",
                provider_id="codex",
                name=f"Codex ({email})",
                api_key=access_token,
                base_url=CODEX_BACKEND_API,
                priority=1,
                weight=1,
                enabled=True,
                extra_config={
                    "auth_type": "chatgpt-oauth",
                    "client_id": CODEX_CLIENT_ID,
                    "refresh_token": refresh_token,
                    "id_token": id_token,
                    "account_id": account_id,
                    "user_id": user_id,
                    "plan_type": plan_type,
                    "email": email,
                    "name": name,
                    "expires_at": time.time() + expires_in,
                },
            )
            return account

    async def refresh_account_token(self, account: Account) -> bool:
        """Refresh OpenAI access token using the account's refresh_token."""
        sanitize_ssl_env()
        refresh_token = account.extra_config.get("refresh_token")
        if not refresh_token:
            return False

        client_id = account.extra_config.get("client_id") or CODEX_CLIENT_ID

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    CODEX_TOKEN_URI,
                    data={
                        "client_id": client_id,
                        "grant_type": "refresh_token",
                        "refresh_token": refresh_token,
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    access_token = data.get("access_token")
                    expires_in = int(data.get("expires_in", 864000))
                    new_refresh = data.get("refresh_token")
                    new_id = data.get("id_token")

                    if access_token:
                        account.api_key = access_token
                        account.extra_config["expires_at"] = time.time() + expires_in
                        if new_refresh:
                            account.extra_config["refresh_token"] = new_refresh
                        if new_id:
                            account.extra_config["id_token"] = new_id
                            id_claims = decode_jwt_payload(new_id)
                            auth_meta = id_claims.get("https://api.openai.com/auth", {})
                            if auth_meta.get("chatgpt_account_id"):
                                account.extra_config["account_id"] = auth_meta["chatgpt_account_id"]
                        return True
        except Exception as e:
            logger.error("Failed to refresh Codex token for account %s: %s", account.id, e)
        return False

    async def import_local_session(self) -> Optional[Account]:
        """Auto-discover and import local Codex credentials from ~/.codex/auth.json."""
        sanitize_ssl_env()
        search_paths = [
            Path("/root/.codex/auth.json"),
            Path.home() / ".codex/auth.json",
            Path("data/auth.json"),
            Path("auth.json"),
        ]

        found_path: Optional[Path] = None
        for p in search_paths:
            if p.exists() and p.is_file():
                found_path = p
                break

        if not found_path:
            return None

        try:
            content = json.loads(found_path.read_text(encoding="utf-8"))
            tokens = content.get("tokens") or {}
            access_token = tokens.get("access_token", "")
            refresh_token = tokens.get("refresh_token", "")
            id_token = tokens.get("id_token", "")
            account_id = tokens.get("account_id") or ""
            client_id = (
                tokens.get("oauth_provenance", {}).get("client_id")
                or CODEX_CLIENT_ID
            )

            # Check if token needs refresh
            expires_in = 864000
            if refresh_token:
                try:
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        t_resp = await client.post(
                            CODEX_TOKEN_URI,
                            data={
                                "client_id": client_id,
                                "grant_type": "refresh_token",
                                "refresh_token": refresh_token,
                            },
                        )
                        if t_resp.status_code == 200:
                            t_json = t_resp.json()
                            access_token = t_json.get("access_token", access_token)
                            refresh_token = t_json.get("refresh_token", refresh_token)
                            id_token = t_json.get("id_token", id_token)
                            expires_in = int(t_json.get("expires_in", 864000))
                except Exception as e:
                    logger.warning("Could not refresh local Codex token: %s", e)

            id_claims = decode_jwt_payload(id_token)
            access_claims = decode_jwt_payload(access_token)

            auth_meta = access_claims.get("https://api.openai.com/auth", {}) or id_claims.get("https://api.openai.com/auth", {})
            profile_meta = access_claims.get("https://api.openai.com/profile", {})

            email = id_claims.get("email") or profile_meta.get("email") or "ChatGPT User"
            name = id_claims.get("name") or profile_meta.get("name") or "Codex User"
            if not account_id:
                account_id = auth_meta.get("chatgpt_account_id") or ""
            user_id = auth_meta.get("chatgpt_user_id") or ""
            plan_type = auth_meta.get("chatgpt_plan_type") or "standard"

            account = Account(
                id=f"acc_codex_{uuid.uuid4().hex[:8]}",
                provider_id="codex",
                name=f"Codex ({email})",
                api_key=access_token,
                base_url=CODEX_BACKEND_API,
                priority=1,
                weight=1,
                enabled=True,
                extra_config={
                    "auth_type": "chatgpt-oauth",
                    "client_id": client_id,
                    "refresh_token": refresh_token,
                    "id_token": id_token,
                    "account_id": account_id,
                    "user_id": user_id,
                    "plan_type": plan_type,
                    "email": email,
                    "name": name,
                    "expires_at": time.time() + expires_in,
                    "source": str(found_path),
                },
            )
            return account
        except Exception as e:
            logger.error("Error importing local Codex session from %s: %s", found_path, e)
            return None





codex_oauth = CodexOAuthManager()



plugin = CodexPlugin()
