"""
Google Antigravity OAuth and Upstream Client Utilities
======================================================
Implements authentic Google Antigravity OAuth 2.0 connection, token exchange,
loopback callback listener, userinfo retrieval, and Code Assist onboarding.
"""

from __future__ import annotations

import asyncio
import http.server
import json
import logging
import socket
import threading
import time
import urllib.parse
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from llmproxy.schemas import Account

logger = logging.getLogger(__name__)

# Exact Google Antigravity OAuth parameters from 9Router / Antigravity ACP
ANTIGRAVITY_CLIENT_ID = "1071006060591-tmhssin2h21lcre235vtolojh4g403ep.apps.googleusercontent.com"
ANTIGRAVITY_CLIENT_SECRET = "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"
ANTIGRAVITY_SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/aicode",
]
ANTIGRAVITY_AUTH_URI = "https://accounts.google.com/o/oauth2/v2/auth"
ANTIGRAVITY_TOKEN_URI = "https://oauth2.googleapis.com/token"
ANTIGRAVITY_USERINFO_URI = "https://www.googleapis.com/oauth2/v2/userinfo"
ANTIGRAVITY_DEFAULT_PROJECT = "aicode-consumers"
CLOUDCODE_PA_ENDPOINTS = [
    "https://daily-cloudcode-pa.googleapis.com",
    "https://cloudcode-pa.googleapis.com",
]
ANTIGRAVITY_USER_AGENT = (
    "antigravity/cli/1.5.3 (aidev_client; os_type=linux; arch=amd64; cl=1016078699; auth_method=consumer)"
)


class _OAuthCallbackHandler(http.server.BaseHTTPRequestHandler):
    """Loopback callback HTTP server handler for Google OAuth redirect."""

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
  <title>Google Sign-In Complete • LLMProxy</title>
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
    <h2>Google Account Connected</h2>
    <p>Your Google Antigravity account has been authenticated. You can safely close this window and return to the LLMProxy dashboard.</p>
    <span class="badge">Connection Active</span>
  </div>
  <script>setTimeout(function() { window.close(); }, 2500);</script>
</body>
</html>"""
        self.wfile.write(success_html.encode("utf-8"))


class AntigravityOAuthManager:
    """Manages Google Antigravity OAuth flows, token exchange, and local import."""

    def __init__(self):
        self.active_flows: Dict[str, Dict[str, Any]] = {}

    def _find_free_port(self) -> int:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        return port

    def start_flow(self) -> Dict[str, Any]:
        """Start a new OAuth flow with loopback listener and consent URL."""
        flow_id = f"flow_{uuid.uuid4().hex[:12]}"
        state = uuid.uuid4().hex
        port = self._find_free_port()
        redirect_uri = f"http://127.0.0.1:{port}/"

        # Construct authentic Antigravity Google OAuth URL
        query = urllib.parse.urlencode({
            "client_id": ANTIGRAVITY_CLIENT_ID,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": " ".join(ANTIGRAVITY_SCOPES),
            "access_type": "offline",
            "prompt": "consent",
            "state": state,
        })
        auth_url = f"{ANTIGRAVITY_AUTH_URI}?{query}"

        # Start loopback server
        server = http.server.HTTPServer(("127.0.0.1", port), _OAuthCallbackHandler)
        server.received_code = None
        server.received_state = None
        server.received_error = None

        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()

        flow_info = {
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
                account = await self.exchange_code_and_create_account(
                    code=code, redirect_uri=redirect_uri
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
            flow["error"] = f"Google OAuth error: {server.received_error}"
            return {"status": "error", "error": flow["error"]}

        # Check expiration (10 minutes)
        if time.time() - flow["created_at"] > 600:
            flow["status"] = "error"
            flow["error"] = "OAuth flow timed out"
            return {"status": "error", "error": "OAuth flow timed out"}

        return {"status": "waiting"}

    async def exchange_code_and_create_account(
        self,
        code: str,
        redirect_uri: str,
    ) -> Account:
        """Exchange authorization code for tokens, fetch user info, and return Account."""
        async with httpx.AsyncClient(timeout=15.0) as client:
            # 1. Exchange code for access & refresh tokens
            token_resp = await client.post(
                ANTIGRAVITY_TOKEN_URI,
                data={
                    "client_id": ANTIGRAVITY_CLIENT_ID,
                    "client_secret": ANTIGRAVITY_CLIENT_SECRET,
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

            # 2. Fetch User Profile
            user_resp = await client.get(
                ANTIGRAVITY_USERINFO_URI,
                headers={"Authorization": f"Bearer {access_token}"},
            )
            email = "Google Account"
            if user_resp.status_code == 200:
                user_info = user_resp.json()
                email = user_info.get("email") or user_info.get("name") or email

            # 3. Call loadCodeAssist for Companion Project & Tier
            headers = {
                "Authorization": f"Bearer {access_token}",
                "User-Agent": ANTIGRAVITY_USER_AGENT,
                "x-goog-api-client": "ide/antigravity",
                "Content-Type": "application/json",
            }
            project_id = ANTIGRAVITY_DEFAULT_PROJECT
            tier_id = "free-tier"
            try:
                ca_resp = await client.post(
                    f"{CLOUDCODE_PA_ENDPOINTS[0]}/v1internal:loadCodeAssist",
                    headers=headers,
                    json={"metadata": {"ideType": "ANTIGRAVITY"}},
                )
                if ca_resp.status_code == 200:
                    ca_data = ca_resp.json()
                    project_id = ca_data.get("cloudaicompanionProject") or project_id
                    tier_id = ca_data.get("currentTier", {}).get("id") or tier_id
            except Exception as e:
                logger.warning("loadCodeAssist lookup warning: %s", e)

            account = Account(
                id=f"acc_google_{uuid.uuid4().hex[:8]}",
                provider_id="antigravity",
                name=f"Google ({email})",
                api_key=access_token,
                base_url=CLOUDCODE_PA_ENDPOINTS[0],
                priority=1,
                weight=1,
                enabled=True,
                extra_config={
                    "auth_type": "oauth-personal",
                    "client_id": ANTIGRAVITY_CLIENT_ID,
                    "client_secret": ANTIGRAVITY_CLIENT_SECRET,
                    "refresh_token": refresh_token,
                    "token_uri": ANTIGRAVITY_TOKEN_URI,
                    "scopes": ANTIGRAVITY_SCOPES,
                    "project_id": project_id,
                    "email": email,
                    "expires_at": time.time() + expires_in,
                    "tier": tier_id,
                },
            )
            return account

    async def refresh_account_token(self, account: Account) -> bool:
        """Refresh Google access token using the account's refresh_token."""
        refresh_token = account.extra_config.get("refresh_token")
        if not refresh_token:
            return False

        client_id = account.extra_config.get("client_id") or ANTIGRAVITY_CLIENT_ID
        client_secret = account.extra_config.get("client_secret") or ANTIGRAVITY_CLIENT_SECRET

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    ANTIGRAVITY_TOKEN_URI,
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
                        return True
        except Exception as e:
            logger.error("Failed to refresh Antigravity token for account %s: %s", account.id, e)
        return False

    async def import_local_session(self) -> Optional[Account]:
        """Auto-discover and import local Antigravity acp_token.json credentials."""
        search_paths = [
            Path("/root/.t3/userdata/providers/antigravity"),
            Path.home() / ".t3/userdata/providers/antigravity",
            Path("data"),
            Path("."),
        ]

        token_files: List[Path] = []
        for base in search_paths:
            if base.exists():
                token_files.extend(base.glob("**/acp_token.json"))

        if not token_files:
            return None

        # Pick the most recently modified token file
        token_path = sorted(token_files, key=lambda p: p.stat().st_mtime, reverse=True)[0]
        try:
            content = json.loads(token_path.read_text(encoding="utf-8"))
            client_id = content.get("client_id") or ANTIGRAVITY_CLIENT_ID
            client_secret = content.get("client_secret") or ANTIGRAVITY_CLIENT_SECRET
            refresh_token = content.get("refresh_token")
            project_id = content.get("project_id") or ANTIGRAVITY_DEFAULT_PROJECT

            if not refresh_token:
                return None

            async with httpx.AsyncClient(timeout=15.0) as client:
                # Refresh access token
                t_resp = await client.post(
                    ANTIGRAVITY_TOKEN_URI,
                    data={
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "refresh_token": refresh_token,
                        "grant_type": "refresh_token",
                    },
                )
                if t_resp.status_code != 200:
                    logger.warning("Failed refreshing local acp_token: %s", t_resp.text)
                    return None

                t_json = t_resp.json()
                access_token = t_json["access_token"]
                expires_in = int(t_json.get("expires_in", 3600))

                # Fetch user info
                u_resp = await client.get(
                    ANTIGRAVITY_USERINFO_URI,
                    headers={"Authorization": f"Bearer {access_token}"},
                )
                email = "Google Account"
                if u_resp.status_code == 200:
                    u_data = u_resp.json()
                    email = u_data.get("email") or u_data.get("name") or email

                # Load Code Assist companion project
                headers = {
                    "Authorization": f"Bearer {access_token}",
                    "User-Agent": ANTIGRAVITY_USER_AGENT,
                    "x-goog-api-client": "ide/antigravity",
                    "Content-Type": "application/json",
                }
                tier_id = "free-tier"
                try:
                    ca_resp = await client.post(
                        f"{CLOUDCODE_PA_ENDPOINTS[0]}/v1internal:loadCodeAssist",
                        headers=headers,
                        json={"metadata": {"ideType": "ANTIGRAVITY"}},
                    )
                    if ca_resp.status_code == 200:
                        ca_data = ca_resp.json()
                        project_id = ca_data.get("cloudaicompanionProject") or project_id
                        tier_id = ca_data.get("currentTier", {}).get("id") or tier_id
                except Exception:
                    pass

                account = Account(
                    id=f"acc_google_{uuid.uuid4().hex[:8]}",
                    provider_id="antigravity",
                    name=f"Google ({email})",
                    api_key=access_token,
                    base_url=CLOUDCODE_PA_ENDPOINTS[0],
                    priority=1,
                    weight=1,
                    enabled=True,
                    extra_config={
                        "auth_type": "oauth-personal",
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "refresh_token": refresh_token,
                        "token_uri": ANTIGRAVITY_TOKEN_URI,
                        "scopes": ANTIGRAVITY_SCOPES,
                        "project_id": project_id,
                        "email": email,
                        "expires_at": time.time() + expires_in,
                        "tier": tier_id,
                        "source": str(token_path),
                    },
                )
                return account
        except Exception as e:
            logger.error("Error importing local session from %s: %s", token_path, e)
            return None


antigravity_oauth = AntigravityOAuthManager()
