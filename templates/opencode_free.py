"""
OpenCode Free Models Provider Plugin
====================================

Implementation for the anonymous, zero-authentication OpenCode Zen free tier
(https://opencode.ai/inference/openai/v1 or https://opencode.ai/zen/v1).

Key Features:
1. Zero-Authentication / Anonymous Access:
   OpenCode Zen provides a free tier that does not require any account, registration,
   or personal API key. Requests use standard anonymous bearer credentials
   (`Authorization: Bearer public`) along with official client identification headers:
     - User-Agent: opencode/latest/2.2.6/cli
     - x-opencode-client: cli
     - x-opencode-project: <project_id>
     - x-opencode-session: ses_<hex><entropy>
     - x-opencode-session-id: ses_<hex><entropy>
     - x-session-affinity: ses_<hex><entropy>
     - x-session-id: ses_<hex><entropy>
     - x-opencode-request: msg_<hex><entropy>

2. Model Management via Models Tab:
   The provider's exposed models are managed directly via LLMProxy's Models tab,
   consistent with other built-in providers (Codex, Antigravity, OpenAI Compatible).
   No model exposure modes or manual model lists are stored in the configuration variables.

3. How OpenCode Limits Work & Reset Requirements:
   - Mechanism: OpenCode Zen limits free tier usage based on:
     * Client IP address / network (enforced at the Cloudflare / gateway boundary).
     * The 'x-opencode-session' conversation token and 'x-opencode-project'.
   - Error Response: When the free usage quota or rate limit is reached, upstream returns
     HTTP 429 with `{"type": "FreeUsageLimitError", "message": "Free usage limit reached..."}`.
   - Reset Requirements:
     * Time-window cooldown: Standard reset requires waiting for the rolling quota window
       (typically hourly or daily rolling quota, or the Retry-After interval).
     * Network/IP rotation: When an IP address has exhausted its free quota, requests from
       that IP will receive HTTP 429 even with new session IDs. An outbound proxy is required
       to establish a fresh IP identity.

4. Limit Bypassing (Controlled by Constant):
   - The constant `ENABLE_LIMIT_BYPASS: bool = False` controls whether automatic limit
     bypassing via session rotation and retry is enabled. It is DISABLED by default.
   - When enabled (ENABLE_LIMIT_BYPASS=True or via variable 'bypass_rate_limits'=True):
     Upon receiving HTTP 429 or FreeUsageLimitError, the plugin automatically rotates the
     session ID ('ses_...'), project ID ('proj_...'), and request ID ('msg_...'), and retries
     the request with a clean session identity.
   - When disabled (default):
     Standard HTTP 429 rate limit errors with reset details are returned to the caller.

5. Reasoning Levels:
   - `reasoning_effort` can be set per request (or in `extra_body`), appended to the
     model ID as a suffix (`deepseek-v4.1-flash-free:high`), or configured as the
     plugin default variable. Request-level values take precedence over the model
     suffix, which takes precedence over the plugin default.
   - Supported levels: none, low, medium, high, xhigh, max.
   - The resolved level is sent upstream as `reasoning_effort` and any suffix is
     stripped from the model ID before the request leaves the proxy.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Set, Tuple, Union

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

logger = logging.getLogger(__name__)

# ==============================================================================
# Global Constants & Configuration
# ==============================================================================

# Plugin Template Identity
id="opencode_free"
name="OpenCode Free"
description="OpenCode Zen free models provider with models managed via Models tab, anonymous zero-auth access, proxy support, reasoning controls, and rate limit control"

# Constant controlling whether rate limit bypassing (via session and project rotation) is enabled.
# Disabled by default as required.
ENABLE_LIMIT_BYPASS: bool = False

DEFAULT_BASE_URL: str = "https://opencode.ai/inference/openai/v1"
DEFAULT_USER_AGENT: str = "opencode/latest/2.2.6/cli"
DEFAULT_CLIENT: str = "cli"
DEFAULT_PROJECT: str = "global"

# Reasoning levels a caller can request for models that support them. The suffix
# forms are accepted on the request model ID and stripped before the upstream
# call, e.g. ``deepseek-v4.1-flash-free:high``.
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

# Default OpenCode free models exposed in the Models tab
DEFAULT_FREE_MODELS: List[PluginModelConfig] = [
    PluginModelConfig(
        id="mimo-v2.6-flash-free",
        name="MiMo-V2.6-Flash Free",
        upstream_model="mimo-v2.6-flash-free",
        context_window=1048576,
        max_tokens=131072,
    ),
    PluginModelConfig(
        id="deepseek-v4.1-flash-free",
        name="DeepSeek V4.1 Flash Free",
        upstream_model="deepseek-v4.1-flash-free",
        context_window=330000,
        max_tokens=128000,
    ),
    PluginModelConfig(
        id="glm-5.3-flash-free",
        name="GLM-5.3-Flash Free",
        upstream_model="glm-5.3-flash-free",
        context_window=1000000,
        max_tokens=128000,
    ),
    PluginModelConfig(
        id="crouton-alpha-free",
        name="Crouton Alpha Free",
        upstream_model="crouton-alpha-free",
        context_window=1048576,
        max_tokens=270000,
    ),
    PluginModelConfig(
        id="longcat-2.5-free",
        name="LongCat 2.5 Free",
        upstream_model="longcat-2.5-free",
        context_window=1000000,
        max_tokens=262144,
    ),
    PluginModelConfig(
        id="iceman-free",
        name="Iceman Free",
        upstream_model="iceman-free",
        context_window=262144,
        max_tokens=131072,
    ),
    PluginModelConfig(
        id="inkling-2-small-free",
        name="Inkling 2 Small Free",
        upstream_model="inkling-2-small-free",
        context_window=131072,
        max_tokens=8192,
    ),
    PluginModelConfig(
        id="beam-free",
        name="Beam Free",
        upstream_model="beam-free",
        context_window=1048576,
        max_tokens=131072,
    ),
    PluginModelConfig(
        id="flint-40b-free",
        name="Flint 40B Free",
        upstream_model="flint-40b-free",
        context_window=131072,
        max_tokens=8192,
    ),
    PluginModelConfig(
        id="nemotron-3.5-super-vl-free",
        name="Nemotron 3.5 Super VL Free",
        upstream_model="nemotron-3.5-super-vl-free",
        context_window=128000,
        max_tokens=64000,
    ),
    PluginModelConfig(
        id="nemotron-3.5-lightning-free",
        name="Nemotron 3.5 Lightning Free",
        upstream_model="nemotron-3.5-lightning-free",
        context_window=128000,
        max_tokens=64000,
    ),
    PluginModelConfig(
        id="big-pickle",
        name="Big Pickle",
        upstream_model="big-pickle",
        context_window=200000,
        max_tokens=32000,
    ),
]


# ==============================================================================
# Helper Functions: Identifiers
# ==============================================================================

def generate_session_id() -> str:
    """Generate authentic OpenCode session ID: ses_<hex_timestamp><random_entropy>."""
    ts = hex(int(time.time() * 1000))[2:]
    rnd = uuid.uuid4().hex[:14]
    return f"ses_{ts}{rnd}"


def generate_request_id() -> str:
    """Generate authentic OpenCode request ID: msg_<hex_timestamp><random_entropy>."""
    ts = hex(int(time.time() * 1000))[2:]
    rnd = uuid.uuid4().hex[:14]
    return f"msg_{ts}{rnd}"


def generate_project_id() -> str:
    """Generate authentic OpenCode project ID."""
    return f"proj_{uuid.uuid4().hex[:12]}"


# ==============================================================================
# OpenCode Free Plugin Implementation
# ==============================================================================

class OpenCodeFreePlugin(BasePlugin):
    """
    OpenCode Free Models Provider Plugin.
    Enables direct connection to OpenCode Zen free tier without authentication.
    Models are configured and managed via the standard Models tab in LLMProxy.
    Supports outbound proxy routing and rate limit bypass controls.
    """

    def __init__(self):
        self._current_session_id: str = generate_session_id()
        self._current_project_id: str = DEFAULT_PROJECT
        self._session_created_at: float = time.time()

        self._metadata = PluginMetadata(
            id="opencode_free",
            name="OpenCode Free",
            description="OpenCode Zen free models provider with models managed via Models tab, anonymous zero-auth access, proxy support, reasoning controls, and rate limit control",
            version="1.0.0",
            author="OpenCode",
            enabled=True,
            filename="opencode_free.py",
            variables=[
                PluginVariable(
                    name="base_url",
                    label="Default Base URL",
                    type="string",
                    default=DEFAULT_BASE_URL,
                    description="OpenCode Zen inference base URL",
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
                PluginVariable(
                    name="timeout",
                    label="Timeout (seconds)",
                    type="number",
                    default=60,
                    description="Request timeout in seconds",
                ),
                PluginVariable(
                    name="max_retries",
                    label="Max Retries",
                    type="number",
                    default=2,
                    description="Maximum retry attempts on transient network or rate limit errors",
                ),
                PluginVariable(
                    name="reasoning_effort",
                    label="Reasoning Effort",
                    type="select",
                    default="default",
                    options=["default", *EFFORT_LEVELS],
                    description=(
                        "Default reasoning effort for free models that support it. A "
                        "request-level reasoning_effort or a model suffix takes precedence."
                    ),
                ),
                PluginVariable(
                    name="bypass_rate_limits",
                    label="Bypass Rate Limits",
                    type="boolean",
                    default=False,
                    description="Whether to rotate session ID and retry on HTTP 429 / FreeUsageLimitError (defaults to ENABLE_LIMIT_BYPASS constant)",
                ),
            ],
            models=list(DEFAULT_FREE_MODELS),
        )

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata. Models are exposed for management via the Models tab."""
        return self._metadata

    def validate_variables(self, variables: Dict[str, Any]) -> List[str]:
        """Validate operational variables. OpenCode free tier requires zero credentials."""
        errors: List[str] = []
        timeout = variables.get("timeout")
        if timeout not in (None, ""):
            try:
                if float(timeout) <= 0:
                    errors.append("Timeout must be a positive number.")
            except (ValueError, TypeError):
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
    # Session Management & Limit Bypassing Logic
    # ==========================================================================

    def _get_or_rotate_session(self, rotate: bool = False) -> Tuple[str, str]:
        """
        Get or rotate session ID and project ID.
        Under normal conditions, sessions rotate periodically (every 30 mins).
        When rotate=True (e.g. rate limit bypass), a fresh identity is generated immediately.
        """
        now = time.time()
        if rotate or (now - self._session_created_at > 1800):  # 30 min rotation
            self._current_session_id = generate_session_id()
            self._current_project_id = generate_project_id() if rotate else DEFAULT_PROJECT
            self._session_created_at = now
            if rotate:
                logger.info(
                    "OpenCode session rotated for limit bypass: session=%s, project=%s",
                    self._current_session_id,
                    self._current_project_id,
                )
        return self._current_session_id, self._current_project_id

    def _is_limit_bypass_active(self, variables: Dict[str, Any]) -> bool:
        """Check whether limit bypassing is active via const variable or setting."""
        return bool(ENABLE_LIMIT_BYPASS or variables.get("bypass_rate_limits", False))

    def _get_client_kwargs(self, timeout: float, variables: Dict[str, Any]) -> Dict[str, Any]:
        """Build httpx client kwargs with optional proxy configuration."""
        kwargs: Dict[str, Any] = {"timeout": timeout}
        proxy = str(variables.get("proxy") or "").strip()
        if proxy:
            kwargs["proxy"] = proxy
        return kwargs

    # ==========================================================================
    # Upstream Wire Protocol
    # ==========================================================================

    def _build_upstream_headers(
        self,
        session_id: str,
        project_id: str,
        custom_headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """Build authentic OpenCode client headers for zero-auth upstream communication."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer public",
            "User-Agent": DEFAULT_USER_AGENT,
            "x-opencode-client": DEFAULT_CLIENT,
            "x-opencode-project": project_id,
            "x-opencode-session": session_id,
            "x-opencode-session-id": session_id,
            "x-session-affinity": session_id,
            "x-session-id": session_id,
            "x-opencode-request": generate_request_id(),
        }
        if custom_headers:
            headers.update(custom_headers)
        return headers

    @staticmethod
    def _normalize_effort(value: Any) -> Optional[str]:
        """Normalize a caller- or config-provided reasoning level."""
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
        variables: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Optional[str]]:
        """Return the upstream model ID and the reasoning effort to send.

        Precedence: request ``reasoning_effort`` / ``extra_body`` > model ID
        suffix > plugin ``reasoning_effort`` variable.
        """
        variables = variables or {}
        target_model = model_config.upstream_model or model_config.id

        raw_effort: Any = None
        if getattr(request, "reasoning_effort", None):
            raw_effort = request.reasoning_effort
        else:
            extra = request.extra_body if isinstance(request.extra_body, dict) else {}
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

    @staticmethod
    def _apply_cache_fields(payload: Dict[str, Any], request: ChatCompletionRequest) -> None:
        """Forward OpenAI prompt-cache routing hints to OpenCode Zen."""
        extra = request.extra_body if isinstance(request.extra_body, dict) else {}
        for key in ("prompt_cache_key", "prompt_cache_retention"):
            value = getattr(request, key, None)
            if value is None:
                value = extra.get(key)
            if value is not None:
                payload[key] = value

    def _build_upstream_payload(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        variables: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build standard OpenAI-compatible payload accepted by OpenCode Zen."""
        actual_model, effort = self._resolve_reasoning(request, model_config, variables)

        messages_payload: List[Dict[str, Any]] = []
        for msg in request.messages:
            m_dict: Dict[str, Any] = {"role": msg.role, "content": msg.content or ""}
            if msg.name:
                m_dict["name"] = msg.name
            if msg.tool_calls:
                m_dict["tool_calls"] = msg.tool_calls
            if msg.tool_call_id:
                m_dict["tool_call_id"] = msg.tool_call_id
            messages_payload.append(m_dict)

        payload: Dict[str, Any] = {
            "model": actual_model,
            "messages": messages_payload,
            "stream": bool(request.stream),
        }

        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.tools:
            payload["tools"] = request.tools
        if request.tool_choice:
            payload["tool_choice"] = request.tool_choice
        if request.stop:
            payload["stop"] = request.stop

        if effort:
            payload["reasoning_effort"] = effort

        self._apply_cache_fields(payload, request)
        return payload

    def _is_test_mode(self, api_key: str) -> bool:
        """Check if request is in simulation/test mode."""
        if not api_key:
            return False
        key = api_key.lower().strip()
        prefixes = ("test", "mock", "sk-test", "dummy", "fake", "offline", "sample")
        return any(key.startswith(p) for p in prefixes) or key in ("test-key", "test_key", "secret123")

    def _mock_response(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        latency_ms: float = 1.0,
    ) -> ChatCompletionResponse:
        """Simulation response for testing environments."""
        actual_model = model_config.upstream_model or model_config.id
        now_str = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        reply_content = (
            f"Simulated response from OpenCode Free Models ({actual_model}).\n\n---\n"
            f"**Execution Details:**\n"
            f"• **Time taken:** {latency_ms:.2f} ms\n"
            f"• **Timestamp:** {now_str}\n"
            f"• **Model:** {actual_model}\n"
            f"• **Status:** Anonymous Zero-Auth Execution (Simulation)"
        )
        return ChatCompletionResponse(
            id=f"chatcmpl-oc-{uuid.uuid4().hex[:12]}",
            model=actual_model,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=reply_content),
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(prompt_tokens=15, completion_tokens=30, total_tokens=45),
        )

    def _mock_stream(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        latency_ms: float = 1.0,
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Simulation stream for testing environments."""
        actual_model = model_config.upstream_model or model_config.id
        chunk_id = f"chatcmpl-oc-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        async def generator() -> AsyncIterator[ChatCompletionChunk]:
            yield ChatCompletionChunk(
                id=chunk_id,
                created=created,
                model=actual_model,
                choices=[ChatChoiceDelta(index=0, delta={"role": "assistant", "content": ""})],
            )
            words = [f"Simulated ", f"OpenCode ", f"Free ", f"response ", f"({actual_model})."]
            for w in words:
                yield ChatCompletionChunk(
                    id=chunk_id,
                    created=created,
                    model=actual_model,
                    choices=[ChatChoiceDelta(index=0, delta={"content": w})],
                )
            yield ChatCompletionChunk(
                id=chunk_id,
                created=created,
                model=actual_model,
                choices=[ChatChoiceDelta(index=0, delta={}, finish_reason="stop")],
            )

        return generator()

    # ==========================================================================
    # Upstream Request Execution
    # ==========================================================================

    async def _execute_upstream_non_streaming(
        self,
        base_url: str,
        payload: Dict[str, Any],
        timeout: float,
        variables: Dict[str, Any],
        target_model: str,
    ) -> ChatCompletionResponse:
        """Execute non-streaming POST /chat/completions with retry, proxy/relay, and limit bypass support."""
        proxy_raw = str(variables.get("proxy") or "").strip()
        proxy = proxy_raw.split("#")[0].rstrip("/") if ("workers.dev" in proxy_raw or "relay" in proxy_raw) else proxy_raw
        is_relay = bool(proxy and ("workers.dev" in proxy or "relay" in proxy))
        client_kwargs: Dict[str, Any] = {"timeout": timeout}
        if proxy and not is_relay:
            client_kwargs["proxy"] = proxy

        url = proxy.rstrip("/") if is_relay else f"{base_url.rstrip('/')}/chat/completions"
        retries_val = variables.get("max_retries")
        try:
            max_retries = int(retries_val) if retries_val not in (None, "") else 2
        except (ValueError, TypeError):
            max_retries = 2
        bypass_enabled = self._is_limit_bypass_active(variables)

        for attempt in range(max_retries + 1):
            session_id, project_id = self._get_or_rotate_session(rotate=False)
            headers = self._build_upstream_headers(session_id, project_id)
            if is_relay:
                headers["x-relay-target"] = f"{base_url.rstrip('/')}/chat/completions"

            try:
                async with httpx.AsyncClient(**client_kwargs) as client:
                    resp = await client.post(url, json=payload, headers=headers)

                    # Handle 429 / Rate Limit
                    if resp.status_code == 429:
                        error_text = resp.text
                        if bypass_enabled and attempt < max_retries:
                            logger.warning(
                                "Hit OpenCode rate limit (429). Bypass enabled; rotating session and retrying (attempt %d/%d)...",
                                attempt + 1,
                                max_retries,
                            )
                            self._get_or_rotate_session(rotate=True)
                            await asyncio.sleep(0.2)
                            continue

                        # Limit reached without bypass
                        raise RuntimeError(
                            f"OpenCode Free Tier Limit Exceeded (HTTP 429): {error_text}. "
                            "Standard reset requires waiting for quota window cooldown, or configuring an outbound proxy."
                        )

                    resp.raise_for_status()
                    data = resp.json()
                    data.setdefault("model", target_model)
                    data.setdefault("id", f"chatcmpl-oc-{uuid.uuid4().hex[:12]}")
                    return ChatCompletionResponse.model_validate(data)

            except (httpx.ProxyError, httpx.ConnectError, httpx.ConnectTimeout) as conn_err:
                logger.warning(
                    "Proxy/connection failed (%s). Attempt %d/%d...",
                    conn_err,
                    attempt + 1,
                    max_retries,
                )
                if attempt < max_retries:
                    if "proxy" in client_kwargs:
                        client_kwargs.pop("proxy", None)
                    if is_relay:
                        url = f"{base_url.rstrip('/')}/chat/completions"
                        is_relay = False
                    await asyncio.sleep(0.2)
                    continue
                raise

            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 429 and bypass_enabled and attempt < max_retries:
                    self._get_or_rotate_session(rotate=True)
                    await asyncio.sleep(0.2)
                    continue
                raise

        raise RuntimeError("OpenCode Free upstream request failed after maximum retries.")

    def _execute_upstream_streaming(
        self,
        base_url: str,
        payload: Dict[str, Any],
        timeout: float,
        variables: Dict[str, Any],
        target_model: str,
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Execute streaming POST /chat/completions with retry, proxy/relay, and limit bypass support."""
        proxy_raw = str(variables.get("proxy") or "").strip()
        proxy = proxy_raw.split("#")[0].rstrip("/") if ("workers.dev" in proxy_raw or "relay" in proxy_raw) else proxy_raw
        is_relay = bool(proxy and ("workers.dev" in proxy or "relay" in proxy))
        client_kwargs: Dict[str, Any] = {"timeout": timeout}
        if proxy and not is_relay:
            client_kwargs["proxy"] = proxy

        url = proxy.rstrip("/") if is_relay else f"{base_url.rstrip('/')}/chat/completions"
        bypass_enabled = self._is_limit_bypass_active(variables)
        retries_val = variables.get("max_retries")
        try:
            max_retries = int(retries_val) if retries_val not in (None, "") else 2
        except (ValueError, TypeError):
            max_retries = 2

        async def sse_generator() -> AsyncIterator[ChatCompletionChunk]:
            nonlocal is_relay, url
            for attempt in range(max_retries + 1):
                session_id, project_id = self._get_or_rotate_session(rotate=False)
                headers = self._build_upstream_headers(session_id, project_id)
                if is_relay:
                    headers["x-relay-target"] = f"{base_url.rstrip('/')}/chat/completions"

                try:
                    async with httpx.AsyncClient(**client_kwargs) as client:
                        async with client.stream("POST", url, json=payload, headers=headers) as stream_resp:
                            if stream_resp.status_code == 429:
                                err_body = await stream_resp.aread()
                                if bypass_enabled and attempt < max_retries:
                                    logger.warning(
                                        "Hit OpenCode streaming rate limit (429). Bypass enabled; rotating session and retrying...",
                                    )
                                    self._get_or_rotate_session(rotate=True)
                                    await asyncio.sleep(0.2)
                                    continue

                                raise RuntimeError(
                                    f"OpenCode Free Tier Limit Exceeded (HTTP 429): {err_body.decode('utf-8', errors='ignore')}. "
                                    "Standard reset requires waiting for quota window cooldown, or configuring an outbound proxy."
                                )

                            stream_resp.raise_for_status()

                            async for line in stream_resp.aiter_lines():
                                line_clean = line.strip()
                                if not line_clean or line_clean.startswith(":"):
                                    continue
                                if line_clean.startswith("data:"):
                                    data_str = line_clean[5:].strip()
                                    if data_str == "[DONE]":
                                        return

                                    # Check for in-stream rate limit error JSON
                                    if "FreeUsageLimitError" in data_str:
                                        if bypass_enabled and attempt < max_retries:
                                            logger.warning("In-stream FreeUsageLimitError detected. Rotating session...")
                                            self._get_or_rotate_session(rotate=True)
                                            break  # retry outer loop

                                        raise RuntimeError(
                                            f"OpenCode Free Tier Limit Exceeded: {data_str}. Standard reset requires waiting for quota cooldown, or configuring an outbound proxy."
                                        )

                                    try:
                                        chunk_dict = json.loads(data_str)
                                        chunk_dict.setdefault("model", target_model)
                                        chunk_dict.setdefault("id", f"chatcmpl-oc-{uuid.uuid4().hex[:12]}")
                                        yield ChatCompletionChunk.model_validate(chunk_dict)
                                    except Exception:
                                        continue
                            return

                except (httpx.ProxyError, httpx.ConnectError, httpx.ConnectTimeout) as conn_err:
                    logger.warning(
                        "Streaming proxy/connection failed (%s). Attempt %d/%d...",
                        conn_err,
                        attempt + 1,
                        max_retries,
                    )
                    if attempt < max_retries:
                        if "proxy" in client_kwargs:
                            client_kwargs.pop("proxy", None)
                        if is_relay:
                            url = f"{base_url.rstrip('/')}/chat/completions"
                            is_relay = False
                        await asyncio.sleep(0.2)
                        continue
                    raise

                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code == 429 and bypass_enabled and attempt < max_retries:
                        self._get_or_rotate_session(rotate=True)
                        await asyncio.sleep(0.2)
                        continue
                    raise

        return sse_generator()

    # ==========================================================================
    # Main chat_completion Entrypoint
    # ==========================================================================

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        variables: Dict[str, Any],
        account: Optional[Account] = None,
    ) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionChunk]]:
        """
        Execute chat completion on OpenCode free models.
        Directly executes upstream without requiring an account or API key.
        """
        start_t = time.perf_counter()
        base_url_val = variables.get("base_url")
        base_url = (
            account.base_url
            if (account and account.base_url)
            else (str(base_url_val).strip() if base_url_val else DEFAULT_BASE_URL)
        ).rstrip("/")
        timeout_val = variables.get("timeout")
        try:
            timeout = float(timeout_val) if timeout_val not in (None, "") else 60.0
        except (ValueError, TypeError):
            timeout = 60.0

        # Simulation mode when test key is provided
        api_key = account.api_key if account else str(variables.get("api_key") or "")
        if self._is_test_mode(api_key):
            time_taken_ms = max(0.5, round((time.perf_counter() - start_t) * 1000.0, 2))
            if request.stream:
                return self._mock_stream(request, model_config, latency_ms=time_taken_ms)
            return self._mock_response(request, model_config, latency_ms=time_taken_ms)

        payload = self._build_upstream_payload(request, model_config, variables)
        target_model = payload["model"]

        if request.stream:
            return self._execute_upstream_streaming(
                base_url=base_url,
                payload=payload,
                timeout=timeout,
                variables=variables,
                target_model=target_model,
            )

        return await self._execute_upstream_non_streaming(
            base_url=base_url,
            payload=payload,
            timeout=timeout,
            variables=variables,
            target_model=target_model,
        )


# Exported plugin singleton instance
plugin = OpenCodeFreePlugin()
