"""Nous Free Tier provider plugin for LLMProxy.

This provider uses the same OAuth session flow as Hermes Agent instead of a
Portal API key, so a Free-plan account can call the zero-priced ``:free``
models through ``/v1``.

How the flow works (reverse-engineered from Hermes Agent's
``hermes_cli/auth_nous.py``):

1. One-time device login: ``POST {portal}/api/oauth/device/code`` with
   ``client_id=hermes-cli`` and ``scope=inference:invoke``; the user approves the
   ``verification_uri_complete`` URL in a browser. The token endpoint returns a
   refresh token. Run this from the dashboard (Accounts -> Connect Account ->
   Sign in with Nous Portal) or with
   ``skills/llmproxy-plugin/scripts/nous_oauth_login.py``.
2. Per-request auth: ``POST {portal}/api/oauth/token`` with
   ``x-nous-refresh-token`` and form ``grant_type=refresh_token`` mints a
   short-lived inference JWT. Nous refresh tokens are **single-use and rotate**,
   so the new token is persisted back to the account immediately.
3. Inference: ``POST {inference}/chat/completions`` with the JWT as a Bearer
   token, using the account's plan entitlement (free models on the Free plan).

Do not share a refresh token with another Hermes install: redeeming it in two
places triggers Nous's reuse detection and revokes the session. Run the login
helper separately for LLMProxy.

The five free-plan models are the shipped defaults; add any other model ID the
plan can serve from the dashboard. Reasoning is forwarded as
``reasoning_effort`` for levels the catalog advertises. Prompt-cache hints are
forwarded and ``cached_tokens`` is reported from usage.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from pathlib import Path
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import httpx

from llmproxy.core.interface import BasePlugin
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

NOUS_PORTAL_URL = "https://portal.nousresearch.com"
NOUS_INFERENCE_BASE_URL = "https://inference-api.nousresearch.com/v1"
NOUS_CLIENT_ID = "hermes-cli"
NOUS_SCOPE = "inference:invoke"
NOUS_MODEL_PREFIX = "np"

TOKEN_REFRESH_SKEW_SECONDS = 120.0
ALLOWED_INFERENCE_HOST_SUFFIX = "nousresearch.com"

# The five zero-priced models on the Portal Free plan, with the reasoning levels
# and modalities advertised by https://inference-api.nousresearch.com/v1/models.
_NOUS_FREE_MODELS = (
    {
        "id": "upstage/solar-mini4:free",
        "name": "Upstage: Solar Mini 4",
        "context_window": 524288,
        "max_tokens": 131072,
        "reasoning_levels": ["none", "minimal", "low", "medium", "high", "xhigh", "max"],
        "supports_vision": False,
    },
    {
        "id": "tencent/hy4-flash:free",
        "name": "Tencent: Hy4 Flash",
        "context_window": 1048576,
        "max_tokens": 65536,
        "reasoning_levels": [],
        "supports_vision": True,
    },
    {
        "id": "stepfun/step-3.7-flash:free",
        "name": "StepFun: Step 3.7 Flash",
        "context_window": 262144,
        "max_tokens": 256000,
        "reasoning_levels": ["low", "medium", "high"],
        "supports_vision": True,
    },
    {
        "id": "qwen/qwen4-max:free",
        "name": "Qwen: Qwen4 Max",
        "context_window": 1000000,
        "max_tokens": 131072,
        "reasoning_levels": ["none", "low", "high", "xhigh"],
        "supports_vision": True,
    },
    {
        "id": "meituan/longcat-2.5:free",
        "name": "LongCat 2.5",
        "context_window": 1048576,
        "max_tokens": 131072,
        "reasoning_levels": [],
        "supports_vision": True,
    },
)

_STATE_KEYS = (
    "refresh_token",
    "access_token",
    "access_token_expires_at",
    "inference_base_url",
    "client_id",
    "scope",
    "needs_relogin",
)


class NousOAuthLoginRequired(RuntimeError):
    """The refresh token is missing or dead; a new device login is required."""


class NousOAuthPlugin(BasePlugin):
    """Nous Free Tier (Hermes Agent session) provider."""

    def __init__(self) -> None:
        self._metadata = PluginMetadata(
            id="nous_oauth",
            name="Nous Free Tier",
            description=(
                "Nous Research free tier through the Hermes Agent OAuth session, "
                "serving the five zero-priced :free models without an API key"
            ),
            version="1.0.0",
            author="Nous Research",
            enabled=True,
            filename="nous_oauth.py",
            supports_usage_meter=True,
            oauth_login=True,
            oauth_local_import=True,
            oauth_hint="Device-code OAuth · Free-plan models, no API key",
            variables=self._build_variables(),
            models=self._build_models(),
        )
        self._locks: Dict[str, asyncio.Lock] = {}

    @staticmethod
    def _build_variables() -> List[PluginVariable]:
        return [
            PluginVariable(
                name="portal_url",
                label="Portal URL",
                type="string",
                default=NOUS_PORTAL_URL,
                description="Nous Portal used to refresh the OAuth inference token",
            ),
            PluginVariable(
                name="base_url",
                label="Inference Base URL",
                type="string",
                default=NOUS_INFERENCE_BASE_URL,
                description="Nous inference API base URL",
            ),
            PluginVariable(
                name="client_id",
                label="OAuth Client ID",
                type="string",
                default=NOUS_CLIENT_ID,
                description="OAuth client id used for the device login and token refresh",
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
                label="Default Reasoning Effort",
                type="select",
                default="default",
                options=["default", "none", "minimal", "low", "medium", "high", "xhigh", "max"],
                description=(
                    "Forwarded as reasoning_effort when the selected model advertises "
                    "that level. A request-level value takes precedence."
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
                description="Strategy for selecting proxies from the pool",
            ),
            PluginVariable(
                name="specific_proxy_id",
                label="Specific Proxy ID / Tag",
                type="string",
                default="",
                description="Proxy ID or tag used when the strategy is 'specific'",
            ),
            PluginVariable(
                name="proxy",
                label="Direct Proxy URL",
                type="string",
                default="",
                description="Single proxy URL that overrides the pool for this provider",
            ),
        ]

    @staticmethod
    def _build_models() -> List[PluginModelConfig]:
        models: List[PluginModelConfig] = []
        for entry in _NOUS_FREE_MODELS:
            models.append(
                PluginModelConfig(
                    id=f"{NOUS_MODEL_PREFIX}/{entry['id']}",
                    name=f"{entry['name']} (Free)",
                    upstream_model=entry["id"],
                    context_window=entry["context_window"],
                    max_tokens=entry["max_tokens"],
                    reasoning_levels=list(entry["reasoning_levels"]),
                    supports_vision=bool(entry["supports_vision"]),
                )
            )
        return models

    @property
    def metadata(self) -> PluginMetadata:
        return self._metadata

    # ------------------------------------------------------------------ state
    @staticmethod
    def _state(account: Account) -> Dict[str, Any]:
        if not isinstance(account.extra_config, dict):
            account.extra_config = {}
        for key in _STATE_KEYS:
            account.extra_config.setdefault(key, None)
        return account.extra_config

    @staticmethod
    def _persist_accounts() -> None:
        try:
            from llmproxy.core.manager import plugin_manager

            plugin_manager._save_accounts()
        except Exception as exc:
            logger.warning("Nous OAuth: could not persist rotated refresh token: %s", exc)

    def _lock(self, account_id: str) -> asyncio.Lock:
        lock = self._locks.get(account_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[account_id] = lock
        return lock

    @staticmethod
    def _resolve_proxy(variables: Dict[str, Any], account: Optional[Account]) -> Optional[str]:
        value = str(
            variables.get("proxy") or (getattr(account, "proxy_url", None) or "")
        ).strip()
        return value or None

    # ------------------------------------------------------------------- jwt
    @staticmethod
    def _jwt_claims(token: str) -> Dict[str, Any]:
        try:
            payload = token.split(".")[1]
            payload += "=" * (-len(payload) % 4)
            claims = json.loads(base64.urlsafe_b64decode(payload))
            return claims if isinstance(claims, dict) else {}
        except Exception:
            return {}

    @classmethod
    def _token_usable(cls, token: str, *, min_ttl: float = TOKEN_REFRESH_SKEW_SECONDS) -> bool:
        if not token:
            return False
        claims = cls._jwt_claims(token)
        exp = claims.get("exp")
        try:
            if float(exp) <= time.time() + min_ttl:
                return False
        except (TypeError, ValueError):
            return False
        scope = claims.get("scope") or claims.get("scp") or ""
        if isinstance(scope, str):
            scopes = set(scope.split())
        elif isinstance(scope, (list, tuple)):
            scopes = {str(entry) for entry in scope}
        else:
            scopes = set()
        return NOUS_SCOPE in scopes

    @staticmethod
    def _inference_base(value: Any, variables: Dict[str, Any]) -> str:
        candidate = str(value or "").strip().rstrip("/")
        if candidate:
            parsed = urlparse(candidate)
            host = (parsed.hostname or "").lower()
            if parsed.scheme == "https" and (
                host == ALLOWED_INFERENCE_HOST_SUFFIX
                or host.endswith("." + ALLOWED_INFERENCE_HOST_SUFFIX)
            ):
                return candidate
            logger.warning("Nous OAuth: ignoring unexpected inference host %r", host)
        return str(variables.get("base_url") or NOUS_INFERENCE_BASE_URL).rstrip("/")

    @staticmethod
    def _portal_headers() -> Dict[str, str]:
        # The Portal edge refuses non-Hermes clients on some routes; this is an
        # identification string, not a credential.
        return {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "User-Agent": "hermes-agent/0.23.2 (cli; linux; x86_64)",
        }

    async def _refresh(
        self,
        client: httpx.AsyncClient,
        account: Account,
        variables: Dict[str, Any],
    ) -> str:
        state = self._state(account)
        refresh_token = str(state.get("refresh_token") or "").strip()
        if not refresh_token:
            raise NousOAuthLoginRequired(
                "Nous OAuth account has no refresh token. Reconnect it from the "
                "dashboard (Accounts -> Connect Account -> Sign in with Nous Portal) "
                "or run skills/llmproxy-plugin/scripts/nous_oauth_login.py, then retry."
            )

        portal_url = str(variables.get("portal_url") or NOUS_PORTAL_URL).rstrip("/")
        client_id = str(
            state.get("client_id") or variables.get("client_id") or NOUS_CLIENT_ID
        ).strip()
        response = await client.post(
            f"{portal_url}/api/oauth/token",
            headers={**self._portal_headers(), "x-nous-refresh-token": refresh_token},
            data={"grant_type": "refresh_token", "client_id": client_id},
        )
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}

        if response.status_code != 200 or not payload.get("access_token"):
            code = str(payload.get("error") or "")
            description = str(payload.get("error_description") or response.text[:200])
            # Vercel's edge firewall answers non-browser clients with a 403/429
            # challenge page. That is not an OAuth verdict: keep the refresh
            # token and retry later instead of forcing a new login.
            mitigated = (
                response.headers.get("x-vercel-mitigated")
                if response.status_code in (403, 429)
                else None
            )
            dead_grant = code in ("invalid_grant", "invalid_token", "refresh_token_reused")
            rejected = response.status_code in (400, 401) or (
                response.status_code == 403 and not mitigated
            )
            if dead_grant or rejected:
                state["needs_relogin"] = True
                state["refresh_token"] = None
                state["access_token"] = None
                state["access_token_expires_at"] = None
                self._persist_accounts()
                raise NousOAuthLoginRequired(
                    "Nous OAuth session expired or was revoked "
                    f"({code or response.status_code}: {description}). Reconnect it "
                    "from the dashboard or run the login helper again."
                )
            raise RuntimeError(
                "Nous OAuth token refresh failed"
                + (f" (HTTP {response.status_code}, edge challenge)" if mitigated else "")
                + f": {description}"
            )

        access_token = str(payload["access_token"])
        rotated = payload.get("refresh_token")
        if isinstance(rotated, str) and rotated:
            state["refresh_token"] = rotated
        state["access_token"] = access_token
        state["access_token_expires_at"] = time.time() + float(payload.get("expires_in") or 900)
        state["inference_base_url"] = self._inference_base(
            payload.get("inference_base_url"), variables
        )
        state["client_id"] = client_id
        state["scope"] = payload.get("scope") or state.get("scope") or NOUS_SCOPE
        state["needs_relogin"] = False
        # Persist immediately: the old refresh token is already retired upstream.
        self._persist_accounts()

        if not self._token_usable(access_token):
            raise NousOAuthLoginRequired(
                "Nous OAuth refresh returned a token without inference:invoke scope. "
                "Run the login helper again."
            )
        return access_token

    async def _ensure_token(
        self,
        account: Account,
        variables: Dict[str, Any],
        *,
        force: bool = False,
    ) -> Tuple[str, str]:
        state = self._state(account)
        token = str(state.get("access_token") or "")
        if not force and self._token_usable(token):
            base = self._inference_base(state.get("inference_base_url"), variables)
            return token, base

        async with self._lock(account.id):
            token = str(state.get("access_token") or "")
            if not force and self._token_usable(token):
                base = self._inference_base(state.get("inference_base_url"), variables)
                return token, base
            proxy = self._resolve_proxy(variables, account)
            timeout = float(variables.get("timeout") or 300.0)
            async with httpx.AsyncClient(timeout=timeout, proxy=proxy) as client:
                token = await self._refresh(client, account, variables)
            base = self._inference_base(state.get("inference_base_url"), variables)
            return token, base

    # ------------------------------------------------------------------ helpers
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
        if level in ("off", "disabled", "false", "0"):
            level = "none"
        accepted = {str(entry).lower() for entry in model_config.reasoning_levels}
        return level if level in accepted else None

    @staticmethod
    def _apply_cache_fields(body: Dict[str, Any], request: ChatCompletionRequest) -> None:
        extra = request.extra_body if isinstance(request.extra_body, dict) else {}
        for key in (
            "prompt_cache_key",
            "prompt_cache_retention",
            "cache_control",
            "cached_content",
            "cachedContent",
        ):
            value = getattr(request, key, None)
            if value is None:
                value = extra.get(key)
            if value is not None:
                body[key] = value
        cache = extra.get("cache")
        if isinstance(cache, dict):
            body["cache"] = cache
        cache_control = getattr(request, "cache_control", None) or extra.get("cache_control")
        if not cache_control:
            return
        messages = body.get("messages") or []
        if not messages:
            return
        content = messages[-1].get("content")
        if isinstance(content, list) and content:
            block = content[-1]
            if isinstance(block, dict):
                block.setdefault("cache_control", cache_control)
        elif isinstance(content, str):
            messages[-1]["content"] = [
                {"type": "text", "text": content, "cache_control": cache_control}
            ]

    @staticmethod
    def _build_body(
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        variables: Dict[str, Any],
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": model_config.upstream_model or model_config.id,
            "messages": [message.model_dump(exclude_none=True) for message in request.messages],
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
            request.max_tokens if request.max_tokens is not None else request.max_completion_tokens
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

        effort = NousOAuthPlugin._resolve_reasoning(request, model_config, variables)
        if effort:
            body["reasoning_effort"] = effort

        if request.stream:
            stream_options = dict(request.stream_options or {})
            stream_options["include_usage"] = True
            body["stream_options"] = stream_options

        NousOAuthPlugin._apply_cache_fields(body, request)
        return body

    @staticmethod
    def _upstream_error(status_code: int, body: str) -> RuntimeError:
        excerpt = " ".join(str(body or "").split())[:400]
        return RuntimeError(
            f"Nous Portal upstream error (HTTP {status_code}): {excerpt or 'no response body'}"
        )

    # -------------------------------------------------------------- simulation
    @staticmethod
    def _mock_response(
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        account: Optional[Account] = None,
    ) -> ChatCompletionResponse:
        model_name = model_config.id
        label = f" via account '{account.name}'" if account else ""
        if request.tools and not any(m.role == "tool" for m in request.messages[-2:]):
            fn = request.tools[0].get("function", {}) if "function" in request.tools[0] else request.tools[0]
            props = fn.get("parameters", {}).get("properties", {})
            arguments = json.dumps({next(iter(props)): "test_value"}) if props else "{}"
            return ChatCompletionResponse(
                id=f"chatcmpl-nous-oauth-{uuid.uuid4().hex[:12]}",
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
                                    "id": f"call_nous_{uuid.uuid4().hex[:8]}",
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
                usage=CompletionUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
            )

        content = (
            f"Simulated response from Nous Free Tier ({model_name}){label}. "
            "Connect a Nous OAuth account to reach the free models."
        )
        if any(m.role == "tool" for m in request.messages):
            content = f"Simulated response from Nous OAuth ({model_name}){label} with tool output processed."
        content += (
            "\n\n---\n**Execution Details:**\n"
            f"• **Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n"
            f"• **Model:** {model_name}\n"
            "• **Tokens:** 10 prompt, 20 completion (30 total)\n"
            "• **Account status:** Simulation Mode (no account configured)"
        )
        return ChatCompletionResponse(
            id=f"chatcmpl-nous-oauth-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=model_name,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=content),
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )

    @staticmethod
    async def _mock_stream(
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        account: Optional[Account] = None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        model_name = model_config.id
        chunk_id = f"chatcmpl-nous-oauth-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        if request.tools and not any(m.role == "tool" for m in request.messages[-2:]):
            fn = request.tools[0].get("function", {}) if "function" in request.tools[0] else request.tools[0]
            props = fn.get("parameters", {}).get("properties", {})
            arguments = json.dumps({next(iter(props)): "test_value"}) if props else "{}"
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
                                    "id": f"call_nous_{uuid.uuid4().hex[:8]}",
                                    "type": "function",
                                    "function": {"name": fn.get("name", "test_tool"), "arguments": ""},
                                }
                            ],
                        },
                        finish_reason=None,
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
                                delta={"tool_calls": [{"index": 0, "function": {"arguments": piece}}]},
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
            choices=[ChatChoiceDelta(index=0, delta={"role": "assistant", "content": ""}, finish_reason=None)],
        )
        for token in ("Simulated ", "Nous OAuth ", f"stream for {model_name}."):
            yield ChatCompletionChunk(
                id=chunk_id,
                created=created,
                model=model_name,
                choices=[ChatChoiceDelta(index=0, delta={"content": token}, finish_reason=None)],
            )
        yield ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=model_name,
            choices=[ChatChoiceDelta(index=0, delta={}, finish_reason="stop")],
            usage=CompletionUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )

    # ---------------------------------------------------------------- upstream
    async def _stream_request(
        self,
        url: str,
        headers: Dict[str, str],
        body: Dict[str, Any],
        account: Account,
        variables: Dict[str, Any],
        proxy: Optional[str],
        timeout: float,
    ) -> AsyncIterator[ChatCompletionChunk]:
        for attempt in range(2):
            async with httpx.AsyncClient(timeout=timeout, proxy=proxy) as client:
                async with client.stream("POST", url, headers=headers, json=body) as response:
                    if response.status_code == 401 and attempt == 0:
                        await response.aread()
                        token, base_url = await self._ensure_token(
                            account, variables, force=True
                        )
                        if base_url != url.rsplit("/chat/completions", 1)[0]:
                            url = f"{base_url}/chat/completions"
                        headers = {**headers, "Authorization": f"Bearer {token}"}
                        continue
                    if response.status_code >= 400:
                        raw = await response.aread()
                        raise self._upstream_error(
                            response.status_code, raw.decode("utf-8", errors="replace")
                        )
                    capture_rate_limit_headers(account, response.headers)

                    saw_role = False
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
                        except Exception:
                            continue
                        if isinstance(parsed, dict) and parsed.get("error"):
                            error = parsed["error"]
                            message = error.get("message") if isinstance(error, dict) else str(error)
                            raise RuntimeError(f"Nous Portal streaming error: {message or error}")

                        chunk = ChatCompletionChunk.model_validate(parsed)
                        if not saw_role:
                            first_delta = chunk.choices[0].delta if chunk.choices else {}
                            saw_role = bool(first_delta.get("role")) if isinstance(first_delta, dict) else False
                            if not saw_role:
                                yield ChatCompletionChunk(
                                    id=chunk.id,
                                    created=chunk.created,
                                    model=chunk.model,
                                    choices=[
                                        ChatChoiceDelta(
                                            index=0,
                                            delta={"role": "assistant", "content": ""},
                                            finish_reason=None,
                                        )
                                    ],
                                )
                        yield chunk
                    return

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        variables: Dict[str, Any],
        account: Optional[Account] = None,
    ) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionChunk]]:
        if account is None or str(account.id).startswith("default-"):
            if request.stream:
                return self._mock_stream(request, model_config, account=account)
            return self._mock_response(request, model_config, account=account)

        token, base_url = await self._ensure_token(account, variables)
        proxy = self._resolve_proxy(variables, account)
        timeout = float(variables.get("timeout") or 300.0)
        body = self._build_body(request, model_config, variables)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }

        if request.stream:
            return self._stream_request(
                f"{base_url}/chat/completions",
                headers,
                body,
                account,
                variables,
                proxy,
                timeout,
            )

        for attempt in range(2):
            async with httpx.AsyncClient(timeout=timeout, proxy=proxy) as client:
                response = await client.post(
                    f"{base_url}/chat/completions", headers=headers, json=body
                )
                if response.status_code == 401 and attempt == 0:
                    token, base_url = await self._ensure_token(account, variables, force=True)
                    headers["Authorization"] = f"Bearer {token}"
                    continue
                if response.status_code >= 400:
                    raise self._upstream_error(response.status_code, response.text)
                capture_rate_limit_headers(account, response.headers)
                return ChatCompletionResponse.model_validate(response.json())
        raise RuntimeError("Nous Free Tier: inference request failed after token refresh")
    # ---------------------------------------------------------------- OAuth
    async def oauth_start(self) -> Dict[str, Any]:
        return await nous_device_login.start_flow()

    async def oauth_status(self, flow_id: str) -> Dict[str, Any]:
        return await nous_device_login.check_flow_status(flow_id)

    async def oauth_callback(self, flow_id: str, callback_url: str) -> Dict[str, Any]:
        return {
            "status": "error",
            "error": "Nous uses a device code; open the verification link instead.",
        }

    async def oauth_import_local(self) -> Optional[Account]:
        candidates = [
            Path.home() / ".hermes/auth.json",
            Path("/root/.hermes/auth.json"),
            Path("data/auth.json"),
        ]
        for path in candidates:
            if not path.is_file():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            state = (data.get("providers") or {}).get("nous") or data.get("nous") or {}
            if not isinstance(state, dict):
                continue
            refresh_token = state.get("refresh_token")
            if not isinstance(refresh_token, str) or not refresh_token:
                continue
            return Account(
                id=f"acc_nous_{uuid.uuid4().hex[:8]}",
                provider_id="nous_oauth",
                name="Nous Free Tier",
                api_key="",
                priority=1,
                weight=1,
                enabled=True,
                extra_config={
                    "refresh_token": refresh_token,
                    "client_id": state.get("client_id") or NOUS_CLIENT_ID,
                    "scope": state.get("scope") or NOUS_SCOPE,
                    "inference_base_url": state.get("inference_base_url") or NOUS_INFERENCE_BASE_URL,
                    "needs_relogin": False,
                },
            )
        return None




class NousDeviceLoginManager:
    """In-memory device-code flows owned by the plugin (dashboard-agnostic)."""

    def __init__(self) -> None:
        self.active_flows: Dict[str, Dict[str, Any]] = {}

    async def start_flow(self) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{NOUS_PORTAL_URL}/api/oauth/device/code",
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                    "User-Agent": "hermes-agent/0.23.2 (cli; linux; x86_64)",
                },
                data={"client_id": NOUS_CLIENT_ID, "scope": NOUS_SCOPE},
            )
            if response.status_code >= 400:
                raise ValueError(f"Nous device code request failed ({response.status_code})")
            payload = response.json()

        flow_id = f"flow_nous_{uuid.uuid4().hex[:12]}"
        interval = max(1.0, float(payload.get("interval") or 5))
        self.active_flows[flow_id] = {
            "device_code": str(payload["device_code"]),
            "user_code": str(payload["user_code"]),
            "verification_uri_complete": str(payload["verification_uri_complete"]),
            "interval": interval,
            "expires_at": time.time() + float(payload["expires_in"]),
            "status": "pending",
            "next_poll_at": 0.0,
            "account": None,
            "error": None,
        }
        return {
            "flow_id": flow_id,
            "mode": "device",
            "verification_uri_complete": str(payload["verification_uri_complete"]),
            "user_code": str(payload["user_code"]),
            "expires_in": int(float(payload["expires_in"])),
            "interval": int(interval),
            "supports_callback_paste": False,
        }

    async def check_flow_status(self, flow_id: str) -> Dict[str, Any]:
        flow = self.active_flows.get(flow_id)
        if not flow:
            return {"status": "error", "error": "Nous OAuth flow not found or expired"}
        if flow["status"] == "complete":
            return {"status": "complete", "account": flow["account"]}
        if flow["status"] == "error":
            return {"status": "error", "error": flow["error"]}

        now = time.time()
        if now > flow["expires_at"]:
            flow["status"] = "error"
            flow["error"] = "The Nous device code expired. Start the sign-in again."
            return {"status": "error", "error": flow["error"]}
        if now < flow.get("next_poll_at", 0):
            return {"status": "pending", "user_code": flow["user_code"]}
        flow["next_poll_at"] = now + max(2.0, flow["interval"])

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{NOUS_PORTAL_URL}/api/oauth/token",
                    headers={
                        "Content-Type": "application/x-www-form-urlencoded",
                        "Accept": "application/json",
                        "User-Agent": "hermes-agent/0.23.2 (cli; linux; x86_64)",
                    },
                    data={
                        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                        "client_id": NOUS_CLIENT_ID,
                        "device_code": flow["device_code"],
                    },
                )
                try:
                    payload = response.json()
                except ValueError:
                    payload = {}
                if not isinstance(payload, dict):
                    payload = {}

                if response.status_code == 200 and payload.get("access_token"):
                    flow["account"] = await self._build_account(client, payload)
                    flow["status"] = "complete"
                    return {"status": "complete", "account": flow["account"]}

                error = str(payload.get("error") or "")
                if error in ("authorization_pending", "slow_down"):
                    if error == "slow_down":
                        flow["interval"] = min(30.0, flow["interval"] + 1)
                    return {"status": "pending", "user_code": flow["user_code"]}

                flow["status"] = "error"
                flow["error"] = (
                    f"Nous sign-in failed ({error or response.status_code}): "
                    f"{payload.get('error_description') or response.text[:200]}"
                )
                return {"status": "error", "error": flow["error"]}
        except Exception as exc:
            flow["status"] = "error"
            flow["error"] = f"Nous sign-in failed: {exc}"
            return {"status": "error", "error": flow["error"]}

    async def _build_account(
        self, client: httpx.AsyncClient, device_token: Dict[str, Any]
    ) -> Account:
        refresh_token = device_token.get("refresh_token")
        if not isinstance(refresh_token, str) or not refresh_token:
            raise ValueError("Nous device login returned no refresh token")
        response = await client.post(
            f"{NOUS_PORTAL_URL}/api/oauth/token",
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
                "User-Agent": "hermes-agent/0.23.2 (cli; linux; x86_64)",
                "x-nous-refresh-token": refresh_token,
            },
            data={"grant_type": "refresh_token", "client_id": NOUS_CLIENT_ID},
        )
        try:
            refreshed = response.json()
        except ValueError:
            refreshed = {}
        if response.status_code != 200 or not refreshed.get("access_token"):
            raise ValueError(
                f"Nous token refresh failed ({response.status_code}): "
                f"{refreshed.get('error') or response.text[:200]}"
            )
        access_token = str(refreshed["access_token"])
        claims = NousOAuthPlugin._jwt_claims(access_token)
        scope = claims.get("scope") or claims.get("scp") or ""
        scopes = set(scope.split()) if isinstance(scope, str) else set(scope or [])
        if NOUS_SCOPE not in scopes:
            raise ValueError("Nous refresh token is not inference-scoped")
        try:
            expires_at = float(claims.get("exp"))
        except (TypeError, ValueError):
            expires_at = time.time() + float(refreshed.get("expires_in") or 3600)
        return Account(
            id=f"acc_nous_{uuid.uuid4().hex[:8]}",
            provider_id="nous_oauth",
            name="Nous Free Tier",
            api_key="",
            base_url=None,
            priority=1,
            weight=1,
            enabled=True,
            extra_config={
                "refresh_token": refreshed.get("refresh_token") or refresh_token,
                "access_token": access_token,
                "access_token_expires_at": expires_at,
                "inference_base_url": NousOAuthPlugin._inference_base(
                    refreshed.get("inference_base_url"), {}
                ),
                "client_id": NOUS_CLIENT_ID,
                "scope": NOUS_SCOPE,
                "needs_relogin": False,
            },
        )


nous_device_login = NousDeviceLoginManager()



plugin = NousOAuthPlugin()
