"""
OpenAI Codex OAuth and Authentication Utilities
===============================================
Implements authentic OpenAI / ChatGPT OAuth 2.0 connection, PKCE token exchange,
loopback callback listener, JWT token parsing, token refresh, and local session import
matching official Codex CLI authentication.
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
import secrets
import socket
import threading
import time
import urllib.parse
import uuid
from typing import Any, Dict, List, Optional, Tuple

import httpx

from llmproxy.schemas import Account

logger = logging.getLogger(__name__)


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
            "auth_url": auth_url,
            "redirect_uri": redirect_uri,
            "state": state,
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
