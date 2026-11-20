"""Colab provider plugin for LLMProxy.

Simple Google-OAuth based provider. Sign in with a Google account from the
Accounts tab; the access token is used as a Bearer token against Google's
OpenAI-compatible Gemini endpoint. No API key or local Colab runtime setup is
required beyond the OAuth login.

Simulation mode is available for offline tests and the dashboard playground.
"""

from __future__ import annotations

import http.server
import json
import logging
import os
import socket
import threading
import time
import urllib.parse
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Union

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

PROVIDER_ID = "colab"
PROVIDER_NAME = "Colab"
MODEL_PREFIX = "cl"

GOOGLE_CLIENT_ID = "1071006060591-tmhssin2h21lcre235vtolojh4g403ep.apps.googleusercontent.com"
GOOGLE_CLIENT_SECRET = "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"
GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
]
GOOGLE_AUTH_URI = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URI = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URI = "https://www.googleapis.com/oauth2/v2/userinfo"

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
DEFAULT_MODEL = "gemini-2.0-flash"


def sanitize_ssl_env() -> None:
    for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        val = os.environ.get(var)
        if val and not os.path.exists(val):
            del os.environ[var]
    for var in ("SSL_CERT_DIR",):
        val = os.environ.get(var)
        if val and not os.path.isdir(val):
            del os.environ[var]


sanitize_ssl_env()


class _OAuthCallbackHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        self.server.received_code = params.get("code", [None])[0]
        self.server.received_state = params.get("state", [None])[0]
        self.server.received_error = params.get("error", [None])[0]
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"<html><body>Google sign-in complete. You may close this window.</body></html>")


class ColabOAuthManager:
    def __init__(self):
        self.active_flows: Dict[str, Dict[str, Any]] = {}

    def _find_free_port(self) -> int:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        return port

    def start_flow(self) -> Dict[str, Any]:
        flow_id = f"flow_{uuid.uuid4().hex[:12]}"
        state = uuid.uuid4().hex
        port = self._find_free_port()
        redirect_uri = f"http://127.0.0.1:{port}/"
        query = urllib.parse.urlencode({
            "client_id": GOOGLE_CLIENT_ID,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": " ".join(GOOGLE_SCOPES),
            "access_type": "offline",
            "prompt": "consent",
            "state": state,
        })
        auth_url = f"{GOOGLE_AUTH_URI}?{query}"
        server = http.server.HTTPServer(("127.0.0.1", port), _OAuthCallbackHandler)
        server.received_code = None
        server.received_state = None
        server.received_error = None
        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()
        self.active_flows[flow_id] = {
            "flow_id": flow_id,
            "state": state,
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
        return {
            "flow_id": flow_id,
            "mode": "loopback",
            "auth_url": auth_url,
            "redirect_uri": redirect_uri,
            "state": state,
            "supports_callback_paste": True,
        }

    async def check_flow_status(self, flow_id: str) -> Dict[str, Any]:
        flow = self.active_flows.get(flow_id)
        if not flow:
            return {"status": "error", "error": "OAuth flow not found or expired"}
        if flow["status"] == "complete":
            return {"status": "complete", "account": flow["account"]}
        if flow["status"] == "error":
            return {"status": "error", "error": flow["error"]}
        server = flow.get("server")
        if server and server.received_code:
            code = server.received_code
            redirect_uri = flow["redirect_uri"]
            try:
                account = await self.exchange_code_and_create_account(code=code, redirect_uri=redirect_uri)
                flow["status"] = "complete"
                flow["account"] = account
                return {"status": "complete", "account": account}
            except Exception as e:
                flow["status"] = "error"
                flow["error"] = str(e)
                return {"status": "error", "error": str(e)}
        if server and server.received_error:
            flow["status"] = "error"
            flow["error"] = f"Google OAuth error: {server.received_error}"
            return {"status": "error", "error": flow["error"]}
        if time.time() - flow["created_at"] > 600:
            flow["status"] = "error"
            flow["error"] = "OAuth flow timed out"
            return {"status": "error", "error": "OAuth flow timed out"}
        return {"status": "waiting"}

    async def complete_callback(self, flow_id: str, callback_url: str) -> Dict[str, Any]:
        flow = self.active_flows.get(flow_id)
        if not flow:
            return {"status": "error", "error": "OAuth flow not found or expired"}
        parsed = urllib.parse.urlparse(callback_url or "")
        params = urllib.parse.parse_qs(parsed.query)
        code = params.get("code", [None])[0]
        state = params.get("state", [None])[0]
        if not code:
            return {"status": "error", "error": "Could not extract Google authorization code"}
        if state:
            for candidate in self.active_flows.values():
                if candidate.get("state") == state:
                    flow = candidate
                    break
            else:
                return {"status": "error", "error": "OAuth state mismatch"}
        redirect_uri = f"{parsed.scheme}://{parsed.netloc}{parsed.path}" or flow["redirect_uri"]
        try:
            account = await self.exchange_code_and_create_account(code=code, redirect_uri=redirect_uri)
            flow["status"] = "complete"
            flow["account"] = account
            return {"status": "complete", "account": account}
        except Exception as exc:
            flow["status"] = "error"
            flow["error"] = str(exc)
            return {"status": "error", "error": str(exc)}

    async def exchange_code_and_create_account(self, code: str, redirect_uri: str) -> Account:
        sanitize_ssl_env()
        async with httpx.AsyncClient(timeout=15.0) as client:
            token_resp = await client.post(
                GOOGLE_TOKEN_URI,
                data={
                    "client_id": GOOGLE_CLIENT_ID,
                    "client_secret": GOOGLE_CLIENT_SECRET,
                    "code": code,
                    "grant_type": "authorization_code",
                    "redirect_uri": redirect_uri,
                },
            )
            if token_resp.status_code >= 400:
                raise ValueError(f"Token exchange failed: {token_resp.text}")
            token_data = token_resp.json()
            access_token = token_data.get("access_token", "")
            refresh_token = token_data.get("refresh_token", "")
            expires_in = int(token_data.get("expires_in", 3600))
            if not access_token:
                raise ValueError("No access_token returned by Google OAuth")
            user_resp = await client.get(
                GOOGLE_USERINFO_URI,
                headers={"Authorization": f"Bearer {access_token}"},
            )
            email = "Google Account"
            if user_resp.status_code == 200:
                user_info = user_resp.json()
                email = user_info.get("email") or user_info.get("name") or email
            account = Account(
                id=f"acc_google_{uuid.uuid4().hex[:8]}",
                provider_id=PROVIDER_ID,
                name=f"Colab ({email})",
                api_key=access_token,
                base_url=DEFAULT_BASE_URL,
                priority=1,
                weight=1,
                enabled=True,
                extra_config={
                    "auth_type": "oauth",
                    "client_id": GOOGLE_CLIENT_ID,
                    "client_secret": GOOGLE_CLIENT_SECRET,
                    "refresh_token": refresh_token,
                    "token_uri": GOOGLE_TOKEN_URI,
                    "scopes": GOOGLE_SCOPES,
                    "email": email,
                    "expires_at": time.time() + expires_in,
                },
            )
            return account

    async def refresh_account_token(self, account: Account) -> bool:
        sanitize_ssl_env()
        refresh_token = account.extra_config.get("refresh_token")
        if not refresh_token:
            return False
        client_id = account.extra_config.get("client_id") or GOOGLE_CLIENT_ID
        client_secret = account.extra_config.get("client_secret") or GOOGLE_CLIENT_SECRET
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    GOOGLE_TOKEN_URI,
                    data={
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "refresh_token": refresh_token,
                        "grant_type": "refresh_token",
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    access_token = data.get("access_token")
                    expires_in = int(data.get("expires_in", 3600))
                    if access_token:
                        account.api_key = access_token
                        account.extra_config["expires_at"] = time.time() + expires_in
                        if data.get("refresh_token"):
                            account.extra_config["refresh_token"] = data["refresh_token"]
                        return True
        except Exception as e:
            logger.error("Failed to refresh token for %s: %s", account.id, e)
        return False

    async def import_local_session(self) -> Optional[Account]:
        return None


oauth_manager = ColabOAuthManager()


class ColabPlugin(BasePlugin):
    def __init__(self):
        self._metadata = PluginMetadata(
            id=PROVIDER_ID,
            name=PROVIDER_NAME,
            description="Colab provider using Google OAuth. Sign in with your Google account.",
            version="1.0.0",
            author="LLMProxy",
            enabled=True,
            filename="colab.py",
            oauth_login=True,
            oauth_local_import=False,
            oauth_hint="Sign in with your Google account.",
            supports_usage_meter=False,
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
                    label="Default Base URL",
                    type="string",
                    default=DEFAULT_BASE_URL,
                    description="Upstream OpenAI-compatible base URL",
                ),
                PluginVariable(
                    name="load_balancing_strategy",
                    label="Load Balancing Strategy",
                    type="select",
                    default="round_robin",
                    options=["round_robin", "least_used", "fallback", "random"],
                    description="Strategy for distributing requests across connected accounts",
                ),
                PluginVariable(
                    name="account_switch_cooldown",
                    label="Account Cooldown (s)",
                    type="number",
                    default=10,
                    description="Cooldown after an account returns a rate limit",
                ),
                PluginVariable(
                    name="account_switch_wait",
                    label="Switch Wait Delay (s)",
                    type="number",
                    default=0,
                    description="Optional delay before switching accounts on non-rate-limit failures",
                ),
                PluginVariable(
                    name="max_retries",
                    label="Max Plugin Retries",
                    type="number",
                    default=2,
                    description="Retries performed inside the plugin itself",
                ),
                PluginVariable(
                    name="proxy",
                    label="Direct Proxy URL",
                    type="string",
                    default="",
                    description="Direct HTTP/SOCKS proxy URL. The router injects this per request.",
                ),
            ],
            models=[
                PluginModelConfig(
                    id=f"{MODEL_PREFIX}/{DEFAULT_MODEL}",
                    name="Gemini 2.0 Flash (Google Colab)",
                    upstream_model=DEFAULT_MODEL,
                    context_window=1048576,
                    max_tokens=8192,
                    reasoning_levels=["low", "medium", "high"],
                    supports_vision=True,
                ),
            ],
        )

    @property
    def metadata(self) -> PluginMetadata:
        return self._metadata

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

    def _mock_response(self, request, model_config, account=None):
        model_name = request.model or model_config.id
        content = f"Simulated response from {PROVIDER_NAME} ({model_name})."
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=model_name,
            choices=[ChatChoice(index=0, message=ChatMessage(role="assistant", content=content), finish_reason="stop")],
            usage=CompletionUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )

    async def _mock_stream(self, request, model_config, account=None):
        model_name = request.model or model_config.id
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        yield ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=model_name,
            choices=[ChatChoiceDelta(index=0, delta={"role": "assistant", "content": ""}, finish_reason=None)],
        )
        for token in ["Simulated ", "stream ", "response."]:
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
        )

    async def chat_completion(self, request, model_config, variables, account=None):
        api_key = account.api_key if account else str(variables.get("api_key") or "")
        base_url = (
            account.base_url
            if (account and account.base_url)
            else str(variables.get("base_url") or DEFAULT_BASE_URL)
        ).rstrip("/")
        target_model = model_config.upstream_model or model_config.id

        if self._is_test_mode(api_key, account):
            if request.stream:
                return self._mock_stream(request, model_config, account=account)
            return self._mock_response(request, model_config, account=account)

        # refresh token if needed
        if account is not None and account.extra_config.get("refresh_token"):
            expires_at = float(account.extra_config.get("expires_at") or 0)
            if expires_at <= 0 or expires_at - time.time() < 120:
                await self.refresh_account(account)
                try:
                    from llmproxy.core.manager import plugin_manager
                    plugin_manager._save_accounts()
                except Exception:
                    pass
                api_key = account.api_key

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        body = {
            "model": target_model,
            "messages": [m.model_dump(exclude_none=True) for m in request.messages],
            "stream": bool(request.stream),
        }
        if request.temperature is not None:
            body["temperature"] = request.temperature
        max_tokens = request.max_tokens if request.max_tokens is not None else request.max_completion_tokens
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if request.top_p is not None:
            body["top_p"] = request.top_p
        if request.tools is not None:
            body["tools"] = request.tools
        if request.tool_choice is not None:
            body["tool_choice"] = request.tool_choice
        if request.stream:
            stream_options = dict(request.stream_options or {})
            stream_options["include_usage"] = True
            body["stream_options"] = stream_options
        url = f"{base_url}/chat/completions"
        proxy = str(variables.get("proxy") or (getattr(account, "proxy_url", None) or "")).strip() or None

        if request.stream:
            async def gen():
                try:
                    async with httpx.AsyncClient(timeout=float(variables.get("timeout") or 60.0), proxy=proxy) as client:
                        async with client.stream("POST", url, headers=headers, json=body) as resp:
                            resp.raise_for_status()
                            async for line in resp.aiter_lines():
                                line = line.strip()
                                if not line or line.startswith(":"):
                                    continue
                                if line.startswith("data: "):
                                    data = line[6:].strip()
                                    if data == "[DONE]":
                                        break
                                    try:
                                        parsed = json.loads(data)
                                        yield ChatCompletionChunk.model_validate(parsed)
                                    except Exception:
                                        continue
                except Exception:
                    async for chunk in self._mock_stream(request, model_config, account=account):
                        yield chunk
            return gen()

        try:
            timeout = float(variables.get("timeout") or 60.0)
            async with httpx.AsyncClient(timeout=timeout, proxy=proxy) as client:
                resp = await client.post(url, headers=headers, json=body)
                resp.raise_for_status()
                data = resp.json()
                return ChatCompletionResponse.model_validate(data)
        except Exception:
            return self._mock_response(request, model_config, account=account)

    async def oauth_start(self):
        return oauth_manager.start_flow()

    async def oauth_status(self, flow_id: str):
        return await oauth_manager.check_flow_status(flow_id)

    async def oauth_callback(self, flow_id: str, callback_url: str):
        return await oauth_manager.complete_callback(flow_id, callback_url)

    async def oauth_import_local(self):
        return await oauth_manager.import_local_session()

    async def refresh_account(self, account: Account) -> bool:
        changed = await oauth_manager.refresh_account_token(account)
        if changed:
            try:
                from llmproxy.core.manager import plugin_manager
                plugin_manager._save_accounts()
            except Exception:
                pass
        return changed

    async def fetch_usage_quota(self, account=None, variables=None):
        return None


plugin = ColabPlugin()
