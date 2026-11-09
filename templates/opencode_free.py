"""
OpenCode Free Models Provider Plugin
====================================

Implementation for the anonymous, zero-authentication OpenCode Zen free tier
(https://opencode.ai/zen/v1).

Key Features:
1. Zero-Authentication / Anonymous Access:
   OpenCode Zen provides a free tier that does not require any account, registration,
   or personal API key. Requests use standard anonymous bearer credentials
   (`Authorization: Bearer public`) along with official client identification headers:
     - User-Agent: opencode/1.18.31 ai-sdk/provider-utils/4.0.23 runtime/bun/1.3.13
     - x-opencode-client: cli (or desktop)
     - x-opencode-project: global (or project hash)
     - x-opencode-session: ses_<hex><entropy>
     - x-opencode-request: msg_<hex><entropy>

2. Automatic vs. Manual Model Exposure:
   - Automatic (default): Dynamically scans upstream models (`GET /zen/v1/models`) and
     strictly filters for models whose identifier ends with '-free' (e.g.
     deepseek-v4-flash-free, minimax-m2.5-free, qwen3.6-plus-free, nemotron-3-super-free).
     Models without the '-free' suffix are filtered out to guarantee only free models are exposed.
   - Manual: Allows explicit configuration of exposed models via the 'manual_models'
     setting or programmatic API, letting administrators expose specific curated models.

3. How OpenCode Limits Work & Reset Requirements:
   - Mechanism: OpenCode Zen limits free tier usage anonymously based on the
     'x-opencode-session' conversation header, 'x-opencode-project', client IP, and
     request frequency.
   - Error Response: When the free usage quota or rate limit is exceeded, upstream returns
     HTTP 429 with `{"type": "FreeUsageLimitError", "message": "Rate limit exceeded"}`.
   - Reset Requirements:
     * Standard Reset: Requires waiting for the rolling quota cooldown window (typically
       hourly or daily rolling quota or the Retry-After interval) to expire for that session/IP.
     * Session-Level Reset: Because the free tier is account-less and keyed to the session
       token, creating a fresh session context ('x-opencode-session' and 'x-opencode-project')
       establishes a new anonymous usage track.

4. Limit Bypassing (Controlled by Constant):
   - The constant `ENABLE_LIMIT_BYPASS: bool = False` controls whether automatic limit
     bypassing via session rotation is enabled. It is DISABLED by default.
   - When enabled (ENABLE_LIMIT_BYPASS=True or via variable 'bypass_rate_limits'=True):
     Upon receiving HTTP 429 or FreeUsageLimitError, the plugin automatically rotates the
     session ID ('ses_...'), project ID ('proj_...'), and request ID ('msg_...'), and retries
     the request with a clean session identity.
   - When disabled (default):
     The session ID is preserved stably and HTTP 429 rate limit errors are returned to the caller,
     allowing standard cooldown behavior.
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
description="OpenCode Zen free models provider with automatic (-free filter) and manual model exposure, anonymous zero-auth access, and rate limit control"

# Constant controlling whether rate limit bypassing (via session and project rotation) is enabled.
# Disabled by default as required.
ENABLE_LIMIT_BYPASS: bool = False

DEFAULT_BASE_URL: str = "https://opencode.ai/zen/v1"
DEFAULT_USER_AGENT: str = "opencode/1.18.31 ai-sdk/provider-utils/4.0.23 runtime/bun/1.3.13"
OPENCODE_CLIENT_VERSION: str = "1.18.31"
DEFAULT_CLIENT: str = "cli"
DEFAULT_PROJECT: str = "global"

# Curated fallback list of known OpenCode free models (all ending in -free)
DEFAULT_FREE_MODELS: List[PluginModelConfig] = [
    PluginModelConfig(
        id="deepseek-v4-flash-free",
        name="DeepSeek V4 Flash Free",
        upstream_model="deepseek-v4-flash-free",
        context_window=131072,
        max_tokens=8192,
    ),
    PluginModelConfig(
        id="minimax-m2.5-free",
        name="MiniMax M2.5 Free",
        upstream_model="minimax-m2.5-free",
        context_window=131072,
        max_tokens=8192,
    ),
    PluginModelConfig(
        id="qwen3.6-plus-free",
        name="Qwen 3.6 Plus Free",
        upstream_model="qwen3.6-plus-free",
        context_window=131072,
        max_tokens=8192,
    ),
    PluginModelConfig(
        id="nemotron-3-super-free",
        name="Nemotron 3 Super Free",
        upstream_model="nemotron-3-super-free",
        context_window=131072,
        max_tokens=8192,
    ),
    PluginModelConfig(
        id="mimo-v2.6-flash-free",
        name="Mimo v2.6 Flash Free",
        upstream_model="mimo-v2.6-flash-free",
        context_window=131072,
        max_tokens=8192,
    ),
]


# ==============================================================================
# Helper Functions: Identifiers & Upstream Model Discovery
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


def scan_upstream_models(
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 10.0,
) -> List[PluginModelConfig]:
    """
    Scan models from OpenCode Zen API and strictly filter for those ending with '-free'.
    Falls back gracefully to DEFAULT_FREE_MODELS if upstream is unreachable or offline.
    """
    clean_base = base_url.rstrip("/")
    url = f"{clean_base}/models"
    headers = {
        "Authorization": "Bearer public",
        "User-Agent": DEFAULT_USER_AGENT,
        "x-opencode-client": DEFAULT_CLIENT,
        "x-opencode-project": DEFAULT_PROJECT,
        "x-opencode-session": generate_session_id(),
        "x-opencode-request": generate_request_id(),
    }

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(url, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                raw_models = data.get("data", []) if isinstance(data, dict) else []
                free_configs: List[PluginModelConfig] = []
                seen_ids: Set[str] = set()

                for m in raw_models:
                    mid = m.get("id", "") if isinstance(m, dict) else str(m)
                    # Filter condition: Only expose models that end with '-free'
                    if mid.endswith("-free") and mid not in seen_ids:
                        name_pretty = mid.replace("-free", "").replace("-", " ").title() + " Free"
                        free_configs.append(
                            PluginModelConfig(
                                id=mid,
                                name=name_pretty,
                                upstream_model=mid,
                                context_window=131072,
                                max_tokens=8192,
                            )
                        )
                        seen_ids.add(mid)

                if free_configs:
                    return free_configs
    except Exception as exc:
        logger.debug("Upstream model scan encountered: %s. Using default free models.", exc)

    return list(DEFAULT_FREE_MODELS)


# ==============================================================================
# OpenCode Free Plugin Implementation
# ==============================================================================

class OpenCodeFreePlugin(BasePlugin):
    """
    OpenCode Free Models Provider Plugin.
    Enables direct connection to OpenCode Zen free tier without authentication.
    Supports automatic (-free filter) and manual model exposure, streaming SSE,
    and configurable rate limit bypass controls.
    """

    def __init__(self):
        self._plugin_id = "opencode_free"
        self._model_exposure: str = "automatic"
        self._manual_models: List[PluginModelConfig] = []
        self._cached_scanned_models: Optional[List[PluginModelConfig]] = None
        self._last_scan_time: float = 0.0
        self._scan_cache_ttl: float = 600.0  # 10 minutes cache

        # Session tracking state
        self._current_session_id: str = generate_session_id()
        self._current_project_id: str = DEFAULT_PROJECT
        self._session_created_at: float = time.time()

        self._metadata_base = PluginMetadata(
            id="opencode_free",
            name="OpenCode Free",
            description="OpenCode Zen free models provider with automatic (-free filter) and manual model exposure, anonymous zero-auth access, and rate limit control",
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
                    description="OpenCode Zen API base URL",
                ),
                PluginVariable(
                    name="model_exposure",
                    label="Model Exposure Mode",
                    type="select",
                    default="automatic",
                    options=["automatic", "manual"],
                    description="Strategy for exposing models: 'automatic' scans upstream and exposes models ending in -free, 'manual' exposes manually set models",
                ),
                PluginVariable(
                    name="manual_models",
                    label="Manual Models List",
                    type="string",
                    default="deepseek-v4-flash-free, minimax-m2.5-free, qwen3.6-plus-free",
                    description="Comma or newline separated list of models to expose when Model Exposure is set to manual",
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
                    name="bypass_rate_limits",
                    label="Bypass Rate Limits",
                    type="boolean",
                    default=False,
                    description="Whether to rotate session ID and bypass rate limits on HTTP 429 / FreeUsageLimitError (defaults to ENABLE_LIMIT_BYPASS constant)",
                ),
            ],
            models=list(DEFAULT_FREE_MODELS),
        )

    def _get_active_variables(self) -> Dict[str, Any]:
        """Fetch active variables from plugin_manager configs if available."""
        try:
            from llmproxy.core.manager import plugin_manager
            return plugin_manager.get_plugin_variables(self._plugin_id)
        except Exception:
            return {}

    def set_exposure_mode(self, mode: str) -> None:
        """Programmatically switch between 'automatic' and 'manual' model exposure."""
        clean_mode = mode.lower().strip()
        if clean_mode in ("automatic", "manual"):
            self._model_exposure = clean_mode

    def set_manual_models(self, models: List[Union[str, PluginModelConfig]]) -> None:
        """Manually configure exposed models."""
        configs: List[PluginModelConfig] = []
        for m in models:
            if isinstance(m, PluginModelConfig):
                configs.append(m)
            elif isinstance(m, str) and m.strip():
                mid = m.strip()
                configs.append(
                    PluginModelConfig(
                        id=mid,
                        name=mid.replace("-", " ").title(),
                        upstream_model=mid,
                        context_window=131072,
                        max_tokens=8192,
                    )
                )
        self._manual_models = configs

    def scan_models(self, base_url: Optional[str] = None) -> List[PluginModelConfig]:
        """Trigger an upstream scan, filter for models ending in -free, and update the cache."""
        target_url = base_url or DEFAULT_BASE_URL
        scanned = scan_upstream_models(base_url=target_url)
        self._cached_scanned_models = scanned
        self._last_scan_time = time.time()
        return scanned

    def get_exposed_models(
        self,
        exposure_mode: Optional[str] = None,
        variables: Optional[Dict[str, Any]] = None,
    ) -> List[PluginModelConfig]:
        """
        Return the list of exposed models according to the exposure strategy:
        - 'automatic': scans/returns only models ending with '-free'.
        - 'manual': returns manually configured models.
        """
        vars_dict = variables if variables is not None else self._get_active_variables()
        if exposure_mode:
            mode = exposure_mode.lower().strip()
        elif variables is not None and "model_exposure" in variables:
            mode = str(variables["model_exposure"]).lower().strip()
        elif self._model_exposure:
            mode = self._model_exposure
        else:
            mode = str(vars_dict.get("model_exposure") or "automatic").lower().strip()

        if mode == "manual":
            # 1. Check programmatic manual models
            if self._manual_models:
                return list(self._manual_models)

            # 2. Check variable manual_models string
            manual_str = str(vars_dict.get("manual_models") or "").strip()
            if manual_str:
                items = [p.strip() for p in re.split(r"[,\n]+", manual_str) if p.strip()]
                return [
                    PluginModelConfig(
                        id=item,
                        name=item.replace("-", " ").title(),
                        upstream_model=item,
                        context_window=131072,
                        max_tokens=8192,
                    )
                    for item in items
                ]
            return list(DEFAULT_FREE_MODELS)

        # Automatic mode: scan and expose ONLY models ending with -free
        now = time.time()
        if self._cached_scanned_models and (now - self._last_scan_time < self._scan_cache_ttl):
            return [m for m in self._cached_scanned_models if m.id.endswith("-free")]

        base_url = str(vars_dict.get("base_url") or DEFAULT_BASE_URL)
        scanned = self.scan_models(base_url=base_url)
        return [m for m in scanned if m.id.endswith("-free")]

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata with dynamic model list matching the active exposure mode."""
        vars_dict = self._get_active_variables()
        exposed_models = self.get_exposed_models(exposure_mode=self._model_exposure, variables=vars_dict)
        return PluginMetadata(
            id=self._metadata_base.id,
            name=self._metadata_base.name,
            description=self._metadata_base.description,
            version=self._metadata_base.version,
            author=self._metadata_base.author,
            enabled=self._metadata_base.enabled,
            filename=self._metadata_base.filename,
            variables=self._metadata_base.variables,
            models=exposed_models,
        )

    def validate_variables(self, variables: Dict[str, Any]) -> List[str]:
        """Validate operational variables. OpenCode free tier requires zero credentials."""
        errors: List[str] = []
        timeout = variables.get("timeout")
        if timeout is not None:
            try:
                if float(timeout) <= 0:
                    errors.append("Timeout must be a positive number.")
            except (ValueError, TypeError):
                errors.append("Timeout must be a valid number.")
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
            "x-opencode-request": generate_request_id(),
        }
        if custom_headers:
            headers.update(custom_headers)
        return headers

    def _build_upstream_payload(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
    ) -> Dict[str, Any]:
        """Build standard OpenAI-compatible payload accepted by OpenCode Zen."""
        actual_model = model_config.upstream_model or model_config.id

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
        """Execute non-streaming POST /chat/completions with retry and limit bypass support."""
        url = f"{base_url.rstrip('/')}/chat/completions"
        max_retries = int(variables.get("max_retries") or 2)
        bypass_enabled = self._is_limit_bypass_active(variables)

        for attempt in range(max_retries + 1):
            session_id, project_id = self._get_or_rotate_session(rotate=False)
            headers = self._build_upstream_headers(session_id, project_id)

            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
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
                            "Standard reset requires waiting for quota window cooldown."
                        )

                    resp.raise_for_status()
                    data = resp.json()
                    data.setdefault("model", target_model)
                    data.setdefault("id", f"chatcmpl-oc-{uuid.uuid4().hex[:12]}")
                    return ChatCompletionResponse.model_validate(data)

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
        """Execute streaming POST /chat/completions with retry and limit bypass support."""
        url = f"{base_url.rstrip('/')}/chat/completions"
        bypass_enabled = self._is_limit_bypass_active(variables)
        max_retries = int(variables.get("max_retries") or 2)

        async def sse_generator() -> AsyncIterator[ChatCompletionChunk]:
            for attempt in range(max_retries + 1):
                session_id, project_id = self._get_or_rotate_session(rotate=False)
                headers = self._build_upstream_headers(session_id, project_id)

                try:
                    async with httpx.AsyncClient(timeout=timeout) as client:
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
                                    "Standard reset requires waiting for quota window cooldown."
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
                                            f"OpenCode Free Tier Limit Exceeded: {data_str}. Standard reset requires waiting for quota cooldown."
                                        )

                                    try:
                                        chunk_dict = json.loads(data_str)
                                        chunk_dict.setdefault("model", target_model)
                                        chunk_dict.setdefault("id", f"chatcmpl-oc-{uuid.uuid4().hex[:12]}")
                                        yield ChatCompletionChunk.model_validate(chunk_dict)
                                    except Exception:
                                        continue
                            return

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
        base_url = (
            account.base_url
            if (account and account.base_url)
            else str(variables.get("base_url") or DEFAULT_BASE_URL)
        ).rstrip("/")
        timeout = float(variables.get("timeout") or 60.0)

        # Simulation mode when test key is provided
        api_key = account.api_key if account else str(variables.get("api_key") or "")
        if self._is_test_mode(api_key):
            time_taken_ms = max(0.5, round((time.perf_counter() - start_t) * 1000.0, 2))
            if request.stream:
                return self._mock_stream(request, model_config, latency_ms=time_taken_ms)
            return self._mock_response(request, model_config, latency_ms=time_taken_ms)

        payload = self._build_upstream_payload(request, model_config)

        target_model = model_config.upstream_model or model_config.id

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
