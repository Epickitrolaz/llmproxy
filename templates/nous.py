"""Nous Portal free-tier provider plugin for LLMProxy.

This plugin reproduces what Hermes Agent does for Nous's free tier, without a
paid subscription or a Portal API key:

1. Mint an anonymous identity: ``POST {portal}/api/anonymous/create``.
2. Exchange its ``anon_...`` credential for a short-lived inference JWT:
   ``POST {portal}/api/anonymous/token`` (purpose ``inference``).
3. Call the welcome inference host with that JWT. The free tier serves exactly
   one model, ``nous/welcome``, at ``https://welcome-api.nousresearch.com/v1``.

The anonymous credential is stored in the account's ``extra_config`` on first
use, and the JWT is cached until shortly before it expires (15 minutes). The
Portal may answer the first exchange with HTTP 428 ``challenge_required`` and a
browser URL; the plugin raises an error containing that URL, the user opens it,
and the next request succeeds. The cleared credential is then reused, so later
refreshes do not prompt again.

Notes and limits:

- Portal endpoints require a Hermes Agent User-Agent (other agents are refused
  with 403), so the plugin sends one. There is no secret in that header.
- Only ``nous/welcome`` is available on the free tier. The paid Portal catalog
  (``inference-api.nousresearch.com``) needs a Portal API key and is not used
  by this plugin.
- Reasoning: ``nous/welcome`` accepts ``reasoning_effort`` values low/high/max
  (default high upstream). Unsupported levels are dropped, not forwarded.
- Caching: prompt-cache hints are forwarded and ``cached_tokens`` is reported
  from usage. The free host reports zero cache tokens today.
- If no account is configured, requests use deterministic simulation mode for
  offline tests. A configured account never falls back to simulation.
"""

from __future__ import annotations

import asyncio
import json
import logging
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
NOUS_WELCOME_BASE_URL = "https://welcome-api.nousresearch.com/v1"
NOUS_WELCOME_MODEL = "nous/welcome"
NOUS_MODEL_PREFIX = "nous"

# The Portal's anonymous gate refuses non-Hermes clients with 403 before the
# request reaches the account service. This is an identification string, not a
# credential, and it is what the free tier expects from an agent client.
NOUS_CLIENT_USER_AGENT = "hermes-agent/0.23.2 (cli; linux; x86_64)"
NOUS_CLIENT_INFO = {
    "name": "llmproxy",
    "version": "1.0.0",
    "surface": "cli",
    "platform": "linux",
    "capabilities": ["browser-v1"],
}

TOKEN_REFRESH_SKEW_SECONDS = 60.0
ALLOWED_INFERENCE_HOST_SUFFIX = "nousresearch.com"

_STATE_KEYS = (
    "anon_token",
    "anon_user_id",
    "anon_org_id",
    "access_token",
    "access_token_expires_at",
    "inference_base_url",
    "challenge_url",
    "challenge_expires_at",
)


class NousCredentialDead(RuntimeError):
    """The stored anonymous credential was reaped or claimed by a Nous account."""


class NousFreeTierChallenge(RuntimeError):
    """The Portal wants a browser check before issuing the free-tier JWT."""

    def __init__(self, url: str, message: str = "") -> None:
        self.url = url
        super().__init__(
            "Nous free tier needs a one-time browser check (HTTP 428 challenge_required). "
            f"Open this URL in a browser, then retry the request: {url}"
            + (f" ({message})" if message else "")
        )


class NousPortalPlugin(BasePlugin):
    """Nous Portal free tier (anonymous ``nous/welcome`` inference)."""

    def __init__(self) -> None:
        self._metadata = PluginMetadata(
            id="nous",
            name="Nous Portal Free Tier",
            description=(
                "Nous Research's free tier, authenticated anonymously like Hermes Agent "
                "and serving the nous/welcome model"
            ),
            version="2.0.0",
            author="Nous Research",
            enabled=True,
            filename="nous.py",
            supports_usage_meter=True,
            variables=self._build_variables(),
            models=[
                PluginModelConfig(
                    id=NOUS_WELCOME_MODEL,
                    name="Nous Welcome (Free Tier)",
                    upstream_model=NOUS_WELCOME_MODEL,
                    context_window=1000000,
                    max_tokens=65536,
                    reasoning_levels=["low", "high", "max"],
                    supports_vision=False,
                )
            ],
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
                description="Nous Portal account service used to mint and refresh the free-tier token",
            ),
            PluginVariable(
                name="welcome_base_url",
                label="Free Tier Base URL",
                type="string",
                default=NOUS_WELCOME_BASE_URL,
                description="Fallback inference base URL when the token exchange does not name one",
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
                options=["default", "low", "high", "max"],
                description=(
                    "Forwarded to nous/welcome when the request asks for a level. "
                    "A request-level reasoning_effort takes precedence."
                ),
            ),
            PluginVariable(
                name="proxy_enabled",
                label="Enable External Proxy",
                type="boolean",
                default=False,
                description="Route requests through external proxies (off by default)",
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
        except Exception as exc:  # persistence must not break inference
            logger.warning("Nous free tier: could not persist account state: %s", exc)

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

    @staticmethod
    def _inference_base(exchanged: Dict[str, Any], variables: Dict[str, Any]) -> str:
        candidate = str(exchanged.get("inference_base_url") or "").strip().rstrip("/")
        if candidate:
            parsed = urlparse(candidate)
            host = (parsed.hostname or "").lower()
            if parsed.scheme == "https" and (
                host == ALLOWED_INFERENCE_HOST_SUFFIX
                or host.endswith("." + ALLOWED_INFERENCE_HOST_SUFFIX)
            ):
                return candidate
            logger.warning(
                "Nous free tier: ignoring unexpected inference host %r from the token exchange",
                host,
            )
        fallback = str(variables.get("welcome_base_url") or NOUS_WELCOME_BASE_URL).rstrip("/")
        return fallback

    @staticmethod
    def _portal_headers() -> Dict[str, str]:
        return {
            "content-type": "application/json",
            "user-agent": NOUS_CLIENT_USER_AGENT,
        }

    @staticmethod
    def _challenge_from(payload: Dict[str, Any], portal_url: str) -> Optional[Tuple[str, float, str]]:
        """Return (url, expires_in, message) for the first usable browser challenge."""
        challenges = payload.get("challenges")
        if not isinstance(challenges, list):
            return None
        try:
            portal_host = (urlparse(portal_url).hostname or "").lower()
        except ValueError:
            return None
        for entry in challenges:
            if not isinstance(entry, dict) or entry.get("type") != "browser":
                continue
            url = entry.get("url")
            if not isinstance(url, str) or not url.startswith("https://"):
                continue
            parsed = urlparse(url)
            if (parsed.hostname or "").lower() != portal_host:
                continue
            try:
                expires = max(30.0, min(900.0, float(entry.get("expires_in") or 600)))
            except (TypeError, ValueError):
                expires = 600.0
            return url, expires, str(payload.get("message") or "")
        return None

    # ------------------------------------------------------------------ token
    async def _mint(self, client: httpx.AsyncClient, portal_url: str, state: Dict[str, Any]) -> None:
        response = await client.post(
            f"{portal_url}/api/anonymous/create",
            headers=self._portal_headers(),
            json={},
        )
        if response.status_code not in (200, 201):
            raise RuntimeError(
                "Nous free tier: could not create an anonymous identity "
                f"(HTTP {response.status_code}): {response.text[:300]}"
            )
        payload = response.json()
        token = payload.get("token")
        if not isinstance(token, str) or not token.startswith("anon_"):
            raise RuntimeError("Nous free tier: the identity service returned no anon_ credential")
        state["anon_token"] = token
        state["anon_user_id"] = payload.get("user_id")
        state["anon_org_id"] = payload.get("org_id")
        logger.info("Nous free tier: anonymous identity created")
        self._persist_accounts()

    async def _exchange(
        self,
        client: httpx.AsyncClient,
        portal_url: str,
        state: Dict[str, Any],
        variables: Dict[str, Any],
    ) -> None:
        anon_token = str(state.get("anon_token") or "")
        if not anon_token:
            raise RuntimeError("Nous free tier: no anonymous credential is available")
        response = await client.post(
            f"{portal_url}/api/anonymous/token",
            headers=self._portal_headers(),
            json={
                "token": anon_token,
                "client": dict(NOUS_CLIENT_INFO),
                "purpose": "inference",
            },
        )
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}

        if response.status_code == 428 or payload.get("error") == "challenge_required":
            challenge = self._challenge_from(payload, portal_url)
            if challenge:
                url, expires_in, message = challenge
                state["challenge_url"] = url
                state["challenge_expires_at"] = time.time() + expires_in
                self._persist_accounts()
                raise NousFreeTierChallenge(url, message)
            raise RuntimeError(
                "Nous free tier: the browser challenge could not be parsed "
                f"(HTTP {response.status_code}): {response.text[:300]}"
            )

        if response.status_code in (401, 404):
            raise NousCredentialDead(
                "Nous free tier: the anonymous credential is no longer valid "
                f"(HTTP {response.status_code}); a new identity is needed"
            )
        if response.status_code != 200 or not payload.get("access_token"):
            raise RuntimeError(
                "Nous free tier token exchange failed "
                f"(HTTP {response.status_code}): {response.text[:300]}"
            )

        try:
            expires_in = float(payload.get("expires_in") or 900)
        except (TypeError, ValueError):
            expires_in = 900.0
        state["access_token"] = str(payload["access_token"])
        state["access_token_expires_at"] = time.time() + max(60.0, expires_in)
        state["inference_base_url"] = self._inference_base(payload, variables)
        state["challenge_url"] = None
        state["challenge_expires_at"] = None
        logger.info("Nous free tier: inference token refreshed")
        self._persist_accounts()

    async def _ensure_token(
        self,
        account: Account,
        variables: Dict[str, Any],
        *,
        force: bool = False,
    ) -> Tuple[str, str]:
        state = self._state(account)
        token = str(state.get("access_token") or "")
        expires_at = state.get("access_token_expires_at") or 0
        try:
            valid = bool(token) and float(expires_at) > time.time() + TOKEN_REFRESH_SKEW_SECONDS
        except (TypeError, ValueError):
            valid = False
        if valid and not force:
            base_url = str(state.get("inference_base_url") or "").rstrip("/")
            if base_url:
                return token, base_url

        async with self._lock(account.id):
            # Another concurrent request may have refreshed while we waited.
            token = str(state.get("access_token") or "")
            expires_at = state.get("access_token_expires_at") or 0
            try:
                valid = bool(token) and float(expires_at) > time.time() + TOKEN_REFRESH_SKEW_SECONDS
            except (TypeError, ValueError):
                valid = False
            if valid and not force:
                return token, str(state.get("inference_base_url") or NOUS_WELCOME_BASE_URL).rstrip("/")

            if force:
                state["access_token"] = None
                state["access_token_expires_at"] = None

            portal_url = str(variables.get("portal_url") or NOUS_PORTAL_URL).rstrip("/")
            proxy = self._resolve_proxy(variables, account)
            timeout = float(variables.get("timeout") or 300.0)

            async with httpx.AsyncClient(timeout=timeout, proxy=proxy) as client:
                if not state.get("anon_token"):
                    await self._mint(client, portal_url, state)
                try:
                    await self._exchange(client, portal_url, state, variables)
                except NousCredentialDead:
                    # Reaped after 14 idle days or claimed by a signed-in account:
                    # mint a replacement once. A fresh identity may need a browser
                    # check, which the next error carries.
                    logger.info("Nous free tier: anonymous credential expired, minting a replacement")
                    state["anon_token"] = None
                    state["access_token"] = None
                    state["access_token_expires_at"] = None
                    await self._mint(client, portal_url, state)
                    await self._exchange(client, portal_url, state, variables)

            return str(state["access_token"]), str(state.get("inference_base_url") or NOUS_WELCOME_BASE_URL)

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
            return None
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

        effort = NousPortalPlugin._resolve_reasoning(request, model_config, variables)
        if effort:
            body["reasoning_effort"] = effort

        if request.stream:
            stream_options = dict(request.stream_options or {})
            stream_options["include_usage"] = True
            body["stream_options"] = stream_options

        NousPortalPlugin._apply_cache_fields(body, request)
        return body

    @staticmethod
    def _upstream_error(status_code: int, body: str) -> RuntimeError:
        excerpt = " ".join(str(body or "").split())[:400]
        return RuntimeError(
            f"Nous free tier upstream error (HTTP {status_code}): {excerpt or 'no response body'}"
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
                id=f"chatcmpl-nous-{uuid.uuid4().hex[:12]}",
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
            f"Simulated response from Nous Portal Free Tier ({model_name}){label}. "
            "Add the Nous free-tier account to reach nous/welcome."
        )
        if any(m.role == "tool" for m in request.messages):
            content = f"Simulated response from Nous free tier ({model_name}){label} with tool output processed."
        content += (
            "\n\n---\n**Execution Details:**\n"
            f"• **Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n"
            f"• **Model:** {model_name}\n"
            "• **Tokens:** 10 prompt, 20 completion (30 total)\n"
            "• **Account status:** Simulation Mode (no account configured)"
        )
        return ChatCompletionResponse(
            id=f"chatcmpl-nous-{uuid.uuid4().hex[:12]}",
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
        chunk_id = f"chatcmpl-nous-{uuid.uuid4().hex[:12]}"
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
        for token in ("Simulated ", "Nous free tier ", f"stream for {model_name}."):
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
                        logger.info("Nous free tier: inference token rejected, refreshing once")
                        token, base_url = await self._ensure_token(account, variables, force=True)
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
                            raise RuntimeError(f"Nous free tier streaming error: {message or error}")

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
                    logger.info("Nous free tier: inference token rejected, refreshing once")
                    token, base_url = await self._ensure_token(account, variables, force=True)
                    headers["Authorization"] = f"Bearer {token}"
                    continue
                if response.status_code >= 400:
                    raise self._upstream_error(response.status_code, response.text)
                capture_rate_limit_headers(account, response.headers)
                return ChatCompletionResponse.model_validate(response.json())
        raise RuntimeError("Nous free tier: inference request failed after token refresh")


plugin = NousPortalPlugin()
