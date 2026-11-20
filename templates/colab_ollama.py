"""
Google Colab Ollama Provider Plugin for LLMProxy.

Spins up a Google Colab GPU runtime (T4 by default) on demand, installs and
starts Ollama on it with the lightweight official installer (no pip on the
VM), pulls the requested model from the Ollama registry, and serves
OpenAI-compatible chat completions through the hosted Colab MCP endpoint
(``https://colaboratory.googleapis.com/mcp``).

Authentication — same as the official Colab CLI
-----------------------------------------------
The plugin owns the Google OAuth login, exactly like ``google-colab-cli``:

* It ships the same **public installed-app OAuth client** the CLI bundles
  (``764086051850-...``) and the same scope set
  (``openid``, ``userinfo.profile``, ``userinfo.email``, ``cloud-platform``,
  ``colaboratory``, ``drive.file``).
* It uses the CLI's **remote copy-paste flow** (the one
  ``gcloud auth application-default login`` uses): the dashboard shows a Google
  consent URL, Google displays an authorization code on
  ``sdk.cloud.google.com/applicationdefaultauthcode.html``, and the code is
  pasted back into the generic plugin callback route. No local server, no
  browser on the proxy host required.
* The refresh token is stored on the account and every request refreshes the
  short-lived access token itself, so a login lasts until it is revoked.
* ``Import local login`` reuses an existing CLI sign-in
  (``~/.config/colab-cli/token.json`` or the ``google-colab-cli`` keyring entry),
  so users who already ran ``colab new`` do not sign in twice.
* A Desktop-app client can be supplied through
  ``~/.colab-cli-oauth-config.json`` (the CLI's ``-c`` file) or the
  ``oauth_client_id`` / ``oauth_client_secret`` variables.

Runtime lifecycle
-----------------
The plugin lists existing runtimes and reuses one with the ``llmproxy-``
prefix, otherwise creates ``runtimes/llmproxy-*``. The spec comes from
``list_runtime_specs`` so free and Pro accounts both land on the best eligible
accelerator (T4 preferred, then L4/A100/G4/H100, optional CPU fallback).

Ollama runs on the VM at 127.0.0.1:11434. Colab's public proxy only forwards
the Jupyter port (8080), so the plugin drives the VM through ``execute_code``:
a worker cell ensures Ollama is installed/serving, pulls the model when
missing, and runs the completion. Streaming chunks are appended to
``/content/llmproxy_stream.jsonl`` and tailed with a second kernel session, so
deltas reach the client as SSE while the VM is still generating.

Latency notes
-------------
Cold path: VM allocation (~30-60s) + Ollama install (~45s) + model pull
(~5-25s for 0.5B-3B on Colab's network). Warm path: ~2-5s plus generation.
Nothing is installed with ``pip`` on the VM; ``apt`` is used once for ``zstd``
(the single dependency of Ollama's official installer).

Reasoning / cache / vision
--------------------------
Default models are text-only with no native reasoning controls, so
``reasoning_levels`` stays empty and image input raises a clear error. Ollama
hints (``keep_alive``, ``think``, ``cache_prompt``, ``num_ctx``,
``num_predict``) are forwarded from ``extra_body`` when supplied.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import secrets
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional, Union
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from llmproxy.core.interface import BasePlugin
from llmproxy.core.multimodal import messages_have_images
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MCP_URL = "https://colaboratory.googleapis.com/mcp"
RUNTIME_PREFIX = "llmproxy"
WORKER_OUT_FILE = "/content/llmproxy_stream.jsonl"
RESULT_MARK = "__COLABLLM_RESULT__"
ERROR_MARK = "__COLABLLM_ERROR__"
PROBE_MARK = "__COLABLLM_PROBE__"

# Public installed-app client shipped with google-colab-cli (colab_cli/oauth_config.json).
CLI_CLIENT_ID = "764086051850-6qr4p6gpi6hn506pt8ejuq83di341hur.apps.googleusercontent.com"
CLI_CLIENT_SECRET = "d-FL95Q19q7MQmFpd7hHD0Ty"
CLI_AUTH_URI = "https://accounts.google.com/o/oauth2/auth"
CLI_TOKEN_URI = "https://oauth2.googleapis.com/token"
# Registered landing page for the cloud-SDK client: Google shows the auth code here.
REMOTE_REDIRECT_URI = "https://sdk.cloud.google.com/applicationdefaultauthcode.html"
USERINFO_URI = "https://openidconnect.googleapis.com/v1/userinfo"

PUBLIC_SCOPES = (
    "openid",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/colaboratory",
    "https://www.googleapis.com/auth/drive.file",
)

CLI_CLIENT_CONFIG_PATH = "~/.colab-cli-oauth-config.json"
CLI_TOKEN_PATH = "~/.config/colab-cli/token.json"
CLI_KEYRING_SERVICE = "google-colab-cli"
CLI_KEYRING_USER = "oauth_token"

OLLAMA_EXTRA_FIELDS = (
    "keep_alive",
    "think",
    "cache_prompt",
    "num_ctx",
    "num_predict",
    "options",
    "prompt_cache_key",
    "prompt_cache_retention",
    "seed",
)

_SIM_PREFIXES = ("test", "mock", "sk-test", "dummy", "fake", "offline", "sample")


class ColabError(Exception):
    """Raised for Colab API failures; the router uses the message for failover."""


WORKER_CELL = r"""
import base64, json, os, shlex, subprocess, sys, time, urllib.request, urllib.error

OUT_FILE = @@OUT_FILE_JSON@@
REQ = json.loads(base64.b64decode(@@PAYLOAD_B64_JSON@@).decode('utf-8'))
MODEL = str(REQ.get('model', ''))

def _run(cmd, timeout=1800):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)

def _emit(obj):
    try:
        with open(OUT_FILE, 'a') as f:
            f.write(json.dumps(obj) + '\n')
            f.flush()
    except Exception:
        pass

def _fail(message):
    print('@@ERROR_MARK@@ ' + json.dumps({'message': str(message)[-2000:]}), flush=True)
    sys.exit(0)

def _ollama_up():
    try:
        with urllib.request.urlopen('http://127.0.0.1:11434/api/version', timeout=3) as r:
            return r.status == 200
    except Exception:
        return False

def _ensure_ollama():
    if not (os.path.exists('/usr/local/bin/ollama') or os.path.exists('/usr/bin/ollama')):
        r = _run('apt-get update -qq && apt-get install -y -qq zstd', timeout=300)
        if r.returncode != 0:
            _fail('apt zstd install failed: ' + (r.stderr or r.stdout)[-500:])
        r = _run('curl -fsSL https://ollama.com/install.sh | sh', timeout=560)
        if r.returncode != 0:
            _fail('ollama install failed: ' + (r.stderr or r.stdout)[-500:])
    if not _ollama_up():
        try:
            with open('/content/ollama.err.log', 'a') as errf:
                subprocess.Popen(['nohup', 'ollama', 'serve'],
                                 stdout=errf, stderr=errf,
                                 start_new_session=True)
        except Exception as exc:
            _fail('could not start ollama serve: ' + str(exc))
        for _ in range(60):
            if _ollama_up():
                break
            time.sleep(1)
    if not _ollama_up():
        _fail('ollama serve did not come up')

def _ensure_model(model):
    if not model:
        _fail('no model specified')
    names = []
    r = _run('ollama list --json', timeout=60)
    if r.returncode == 0:
        try:
            names = [m.get('name') for m in json.loads(r.stdout).get('models', [])]
        except Exception:
            names = []
    if model not in names:
        r = _run('ollama pull ' + shlex.quote(model), timeout=1800)
        if r.returncode != 0:
            _fail('ollama pull failed for ' + model + ': ' + (r.stderr or r.stdout)[-600:])

def _chat(req):
    url = 'http://127.0.0.1:11434/v1/chat/completions'
    body = json.dumps(req).encode('utf-8')
    r = urllib.request.Request(url, data=body, headers={'Content-Type': 'application/json'})
    return urllib.request.urlopen(r, timeout=1800)

def _strip(obj):
    if isinstance(obj, dict):
        obj.pop('timings', None)
    return obj

try:
    _ensure_ollama()
    _ensure_model(MODEL)
    if OUT_FILE:
        try:
            open(OUT_FILE, 'w').close()
        except Exception:
            pass
    if REQ.get('stream'):
        resp = _chat({**REQ, 'stream': True, 'stream_options': {'include_usage': True}})
        for raw in resp:
            line = raw.decode('utf-8', 'replace').strip()
            if not line.startswith('data:'):
                continue
            data = line[5:].strip()
            if not data or data == '[DONE]':
                continue
            try:
                _emit(_strip(json.loads(data)))
            except Exception:
                pass
        _emit({'__coLabDone__': True})
        print('@@RESULT_MARK@@ ' + json.dumps({'done': True, 'mode': 'stream'}), flush=True)
    else:
        resp = _chat({**REQ, 'stream': False})
        data = json.loads(resp.read().decode('utf-8', 'replace'))
        print('@@RESULT_MARK@@ ' + json.dumps(_strip(data)), flush=True)
except SystemExit:
    raise
except Exception as exc:
    _fail(str(exc))
"""

PROBE_CELL = r"""
import os, sys
p = @@OUT_FILE_JSON@@
try:
    if not os.path.exists(p):
        print('@@PROBE_MARK@@ <empty>', flush=True)
    else:
        lines = open(p, 'r', encoding='utf-8', errors='replace').read().splitlines()
        print('@@PROBE_MARK@@ ' + json.dumps(lines), flush=True)
except Exception as exc:
    print('@@PROBE_MARK@@ ' + json.dumps({'__probeError__': str(exc)}), flush=True)
"""


class ColabOllamaPlugin(BasePlugin):
    """LLMProxy provider that runs Ollama on a Google Colab GPU runtime."""

    def __init__(self):
        self._metadata = PluginMetadata(
            id="colab_ollama",
            name="Colab Ollama",
            description=(
                "Runs Ollama on a Google Colab GPU runtime (T4 default); signs in "
                "with Google like the Colab CLI and serves an OpenAI-compatible API."
            ),
            version="1.1.0",
            author="llmproxy",
            enabled=True,
            filename="colab_ollama.py",
            oauth_login=True,
            oauth_local_import=True,
            oauth_hint=(
                "Sign in with Google, or import an existing Colab CLI login "
                "(colab new / ~/.config/colab-cli/token.json)."
            ),
            variables=[
                PluginVariable(name="timeout", label="Request Timeout (s)", type="number", default=620, description="Timeout for the Colab MCP calls."),
                PluginVariable(name="mcp_base_url", label="Colab MCP Endpoint", type="string", default=DEFAULT_MCP_URL, description="Hosted Colab MCP JSON-RPC endpoint."),
                PluginVariable(name="oauth_client_config", label="OAuth Client Config Path", type="string", default=CLI_CLIENT_CONFIG_PATH, description="Installed-app client JSON (same file as colab -c). Defaults to the CLI's public client."),
                PluginVariable(name="oauth_client_id", label="OAuth Client ID Override", type="string", default="", description="Optional Desktop-app client id; overrides the bundled CLI client."),
                PluginVariable(name="oauth_client_secret", label="OAuth Client Secret Override", type="secret", default="", description="Optional secret for the client id above."),
                PluginVariable(name="quota_project", label="Default GCP Quota Project", type="string", default="", description="Optional x-goog-user-project. The CLI client usually needs none."),
                PluginVariable(name="colab_headers", label="Extra HTTP Headers (JSON)", type="string", default="{}", description="Extra JSON headers for Colab API calls."),
                PluginVariable(name="accelerator", label="Runtime Accelerator", type="string", default="T4", description="Preferred accelerator (T4, L4, A100, G4)."),
                PluginVariable(name="runtime_variant", label="Runtime Variant", type="select", default="VARIANT_GPU", options=["VARIANT_GPU", "VARIANT_CPU", "VARIANT_TPU"], description="Colab runtime variant."),
                PluginVariable(name="runtime_shape", label="Runtime Shape", type="select", default="SHAPE_STANDARD", options=["SHAPE_STANDARD", "SHAPE_HIGHMEM"], description="Colab runtime shape."),
                PluginVariable(name="allow_cpu_fallback", label="Allow CPU Fallback", type="boolean", default=False, description="When no GPU spec is eligible, fall back to a CPU runtime instead of failing."),
                PluginVariable(name="reuse_runtime", label="Reuse Runtime", type="boolean", default=True, description="Reuse an existing llmproxy-* runtime instead of creating a new one per request."),
                PluginVariable(name="runtime_prefix", label="Runtime Name Prefix", type="string", default=RUNTIME_PREFIX, description="Prefix for runtimes created by this provider."),
                PluginVariable(name="max_startup_wait", label="Max Startup Wait (s)", type="number", default=420, description="How long to wait for runtime creation/setup."),
                PluginVariable(name="poll_interval", label="Stream Poll Interval (s)", type="number", default=2, description="Seconds between VM file-tail polls while streaming."),
                PluginVariable(name="num_ctx", label="Context Window (num_ctx)", type="number", default=0, description="Optional Ollama num_ctx override. 0 = model default."),
                PluginVariable(name="load_balancing_strategy", label="Load Balancing Strategy", type="select", default="round_robin", options=["round_robin", "least_used", "fallback", "random"], description="Strategy for distributing requests across connected Colab accounts."),
                PluginVariable(name="account_switch_cooldown", label="Account Cooldown (s)", type="number", default=10, description="Cooldown after an account hits a rate limit."),
                PluginVariable(name="account_switch_wait", label="Switch Wait Delay (s)", type="number", default=0, description="Delay before switching accounts on non-rate-limit failures."),
                PluginVariable(name="max_retries", label="Max Failover Retries", type="number", default=2, description="Additional account failover attempts."),
                PluginVariable(name="proxy_enabled", label="Enable External Proxy", type="boolean", default=False, description="Route Colab API calls through external proxies (off by default)."),
                PluginVariable(name="proxy_strategy", label="Proxy Strategy", type="select", default="sequential", options=["sequential", "random", "specific", "switch_on_limit"], description="How to choose a proxy from the proxy pool."),
                PluginVariable(name="specific_proxy_id", label="Specific Proxy ID / Tag", type="string", default="", description="Proxy ID, URL, or tag used by the 'specific' strategy."),
                PluginVariable(name="proxy", label="Direct Proxy URL", type="string", default="", description="Direct HTTP/SOCKS proxy URL. Injected per request by the router."),
            ],
            models=[
                PluginModelConfig(id="cl/llama-3.2-3b", name="Llama 3.2 3B", upstream_model="llama3.2:3b", context_window=128000, max_tokens=4096),
                PluginModelConfig(id="cl/qwen-2.5-7b", name="Qwen 2.5 7B", upstream_model="qwen2.5:7b", context_window=32768, max_tokens=8192),
                PluginModelConfig(id="cl/llama-3.2-1b", name="Llama 3.2 1B", upstream_model="llama3.2:1b", context_window=128000, max_tokens=4096),
                PluginModelConfig(id="cl/qwen-2.5-0.5b", name="Qwen 2.5 0.5B (fast)", upstream_model="qwen2.5:0.5b", context_window=32768, max_tokens=4096),
            ],
        )
        self._runtimes: Dict[str, str] = {}
        self._oauth_flows: Dict[str, Dict[str, Any]] = {}

    @property
    def metadata(self) -> PluginMetadata:
        return self._metadata

    # ------------------------------------------------------------------
    # Google OAuth (same client + flow as the official Colab CLI)
    # ------------------------------------------------------------------

    def _plugin_variables(self) -> Dict[str, Any]:
        try:
            from llmproxy.core.manager import plugin_manager
            return dict(plugin_manager.get_plugin_variables(self.metadata.id) or {})
        except Exception:
            return {}

    def _oauth_client(self, variables: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """Resolve the installed-app OAuth client the CLI would use."""
        variables = variables or {}
        client = {
            "client_id": CLI_CLIENT_ID,
            "client_secret": CLI_CLIENT_SECRET,
            "auth_uri": CLI_AUTH_URI,
            "token_uri": CLI_TOKEN_URI,
        }
        raw_path = str(variables.get("oauth_client_config") or CLI_CLIENT_CONFIG_PATH)
        path = Path(os.path.expanduser(raw_path))
        try:
            if path.is_file():
                data = json.loads(path.read_text(encoding="utf-8"))
                installed = data.get("installed") or data.get("web") or data
                for key in ("client_id", "client_secret", "auth_uri", "token_uri"):
                    if installed.get(key):
                        client[key] = str(installed[key])
        except Exception:
            pass
        if str(variables.get("oauth_client_id") or "").strip():
            client["client_id"] = str(variables["oauth_client_id"]).strip()
        if str(variables.get("oauth_client_secret") or "").strip():
            client["client_secret"] = str(variables["oauth_client_secret"]).strip()
        return client

    @staticmethod
    def _prune_flows(flows: Dict[str, Dict[str, Any]]) -> None:
        cutoff = time.time() - 900
        for flow_id in [k for k, v in flows.items() if float(v.get("created") or 0) < cutoff]:
            flows.pop(flow_id, None)

    def _account_from_tokens(self, tokens: Dict[str, Any], client: Dict[str, str], email: Optional[str] = None) -> Account:
        access_token = str(tokens.get("access_token") or "").strip()
        if not access_token:
            raise ColabError("Google OAuth response contained no access token")
        scopes = tokens.get("scope") or " ".join(PUBLIC_SCOPES)
        if isinstance(scopes, list):
            scopes = " ".join(str(s) for s in scopes)
        cfg: Dict[str, Any] = {
            "client_id": client.get("client_id", CLI_CLIENT_ID),
            "client_secret": client.get("client_secret", CLI_CLIENT_SECRET),
            "token_uri": client.get("token_uri", CLI_TOKEN_URI),
            "scopes": scopes,
            "refresh_token": str(tokens.get("refresh_token") or ""),
            "expires_at": time.time() + int(tokens.get("expires_in") or 3600) - 60,
        }
        quota = os.environ.get("COLAB_QUOTA_PROJECT", "").strip()
        if quota:
            cfg["quota_project"] = quota
        if email:
            cfg["email"] = email
        name = f"Google ({email})" if email else "Google Colab"
        return Account(provider_id=self.metadata.id, name=name, api_key=access_token, extra_config=cfg)

    async def _fetch_email(self, access_token: str, proxy: Optional[str]) -> Optional[str]:
        try:
            async with httpx.AsyncClient(timeout=15, proxy=proxy) as client:
                resp = await client.get(USERINFO_URI, headers={"Authorization": f"Bearer {access_token}"})
                if resp.status_code < 400:
                    return str(resp.json().get("email") or "").strip() or None
        except Exception:
            pass
        return None

    async def _token_request(self, token_uri: str, data: Dict[str, str], proxy: Optional[str] = None) -> Dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=30, proxy=proxy) as client:
                resp = await client.post(token_uri or CLI_TOKEN_URI, data=data)
        except httpx.TransportError as exc:
            raise ColabError(f"Google OAuth transport error: {exc}") from exc
        if resp.status_code >= 400:
            raise ColabError(f"Google OAuth error ({resp.status_code}): {resp.text[:300]}")
        return resp.json()

    async def oauth_start(self) -> Dict[str, Any]:
        variables = self._plugin_variables()
        client = self._oauth_client(variables)
        self._prune_flows(self._oauth_flows)
        flow_id = secrets.token_urlsafe(16)
        params = {
            "client_id": client["client_id"],
            "redirect_uri": REMOTE_REDIRECT_URI,
            "response_type": "code",
            "scope": " ".join(PUBLIC_SCOPES),
            "access_type": "offline",
            "prompt": "consent",
            "token_usage": "remote",
            "state": flow_id,
        }
        auth_url = f"{client.get('auth_uri') or CLI_AUTH_URI}?{urlencode(params)}"
        self._oauth_flows[flow_id] = {
            "created": time.time(),
            "client": client,
            "account": None,
            "error": None,
        }
        return {
            "flow_id": flow_id,
            "mode": "loopback",
            "auth_url": auth_url,
            "redirect_uri": REMOTE_REDIRECT_URI,
            "state": flow_id,
            "supports_callback_paste": True,
            "instructions": (
                "Open the URL, approve access, then paste the authorization code "
                "Google shows on the page."
            ),
        }

    async def oauth_status(self, flow_id: str) -> Dict[str, Any]:
        flow = self._oauth_flows.get(flow_id)
        if not flow:
            return {"status": "error", "message": "Unknown or expired OAuth flow"}
        if flow.get("error"):
            return {"status": "error", "message": flow["error"]}
        account = flow.get("account")
        if account is not None:
            return {"status": "complete", "account": account}
        return {"status": "pending"}

    async def oauth_callback(self, flow_id: str, callback_url: str) -> Dict[str, Any]:
        flow = self._oauth_flows.get(flow_id)
        if not flow:
            return {"status": "error", "message": "Unknown or expired OAuth flow"}
        client = flow["client"]
        try:
            code = self._extract_code(callback_url)
            if not code:
                raise ColabError("No authorization code found in the pasted value")
            proxy = self._proxy(self._plugin_variables(), None)
            tokens = await self._token_request(
                client.get("token_uri", CLI_TOKEN_URI),
                {
                    "code": code,
                    "client_id": client["client_id"],
                    "client_secret": client["client_secret"],
                    "redirect_uri": REMOTE_REDIRECT_URI,
                    "grant_type": "authorization_code",
                },
                proxy,
            )
            email = await self._fetch_email(str(tokens.get("access_token") or ""), proxy)
            account = self._account_from_tokens(tokens, client, email)
            flow["account"] = account
            return {"status": "complete", "account": account}
        except Exception as exc:
            flow["error"] = str(exc)
            return {"status": "error", "message": str(exc)}

    @staticmethod
    def _extract_code(value: str) -> str:
        raw = (value or "").strip().strip('"').strip("'")
        if not raw:
            return ""
        if "code=" in raw or raw.startswith("http"):
            try:
                parsed = urlparse(raw)
                code = parse_qs(parsed.query).get("code")
                if code:
                    return code[0]
            except Exception:
                pass
        return raw

    def _read_cli_credential(self) -> Optional[Dict[str, Any]]:
        """Read the credential the Colab CLI stored (token file or OS keyring)."""
        candidates = []
        env_path = os.environ.get("COLAB_CLI_TOKEN_FILE", "").strip()
        if env_path:
            candidates.append(Path(os.path.expanduser(env_path)))
        candidates.append(Path(os.path.expanduser(CLI_TOKEN_PATH)))
        for path in candidates:
            try:
                if path.is_file():
                    data = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(data, dict):
                        return data
            except Exception:
                continue
        try:  # optional: only if the keyring package and backend exist
            import keyring  # type: ignore
            raw = keyring.get_password(CLI_KEYRING_SERVICE, CLI_KEYRING_USER)
            if raw:
                data = json.loads(raw)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
        return None

    async def oauth_import_local(self) -> Optional[Account]:
        data = self._read_cli_credential()
        if not data:
            raise ValueError(
                "No Colab CLI login found. Run `pip install google-colab-cli && colab new` "
                "(or `colab --auth oauth2 sessions`) once, then import again."
            )
        client = self._oauth_client(self._plugin_variables())
        client_id = str(data.get("client_id") or client["client_id"])
        client_secret = str(data.get("client_secret") or client["client_secret"])
        token_uri = str(data.get("token_uri") or client["token_uri"])
        refresh_token = str(data.get("refresh_token") or "")
        access_token = str(data.get("token") or data.get("access_token") or "")
        scopes = data.get("scopes") or list(PUBLIC_SCOPES)
        if isinstance(scopes, list) and not any("colaboratory" in str(s) for s in scopes):
            raise ValueError(
                "The stored CLI credential lacks the colaboratory scope. "
                "Re-run `colab --auth oauth2 sessions`."
            )
        if not refresh_token and not access_token:
            raise ValueError("The stored CLI credential has no usable token. Sign in again with colab.")
        client = {"client_id": client_id, "client_secret": client_secret, "token_uri": token_uri, "auth_uri": client["auth_uri"]}
        tokens: Dict[str, Any] = {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "scope": " ".join(scopes) if isinstance(scopes, list) else str(scopes),
            "expires_in": 0,
        }
        account = self._account_from_tokens(tokens, client)
        try:
            await self._refresh_access_token(account, self._plugin_variables())
        except Exception:
            if not access_token:
                raise
        email = await self._fetch_email(account.api_key, None)
        if email:
            account.name = f"Google ({email})"
            account.extra_config["email"] = email
        return account

    async def _refresh_access_token(self, account: Account, variables: Dict[str, Any]) -> str:
        cfg = dict(account.extra_config or {})
        refresh_token = str(cfg.get("refresh_token") or "").strip()
        client = self._oauth_client(variables)
        client_id = str(cfg.get("client_id") or client["client_id"])
        client_secret = str(cfg.get("client_secret") or client["client_secret"])
        token_uri = str(cfg.get("token_uri") or client["token_uri"])
        if not refresh_token:
            raise ColabError(
                f"Account '{account.name}' has no Google refresh token. "
                "Reconnect it with Sign in with Google or Import local login."
            )
        proxy = self._proxy(variables, account)
        tokens = await self._token_request(
            token_uri,
            {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": client_id,
                "client_secret": client_secret,
            },
            proxy,
        )
        access_token = str(tokens.get("access_token") or "").strip()
        if not access_token:
            raise ColabError("Google OAuth refresh returned no access token")
        account.api_key = access_token
        cfg["expires_at"] = time.time() + int(tokens.get("expires_in") or 3600) - 60
        if tokens.get("refresh_token"):
            cfg["refresh_token"] = tokens["refresh_token"]
        if tokens.get("scope"):
            scope = tokens["scope"]
            cfg["scopes"] = " ".join(scope) if isinstance(scope, list) else str(scope)
        account.extra_config = cfg
        try:
            from llmproxy.core.manager import plugin_manager
            plugin_manager._save_accounts()
        except Exception:
            pass
        return access_token

    async def refresh_account(self, account: Account) -> bool:
        try:
            before = account.api_key
            await self._refresh_access_token(account, self._plugin_variables())
            return account.api_key != before or bool(account.api_key)
        except Exception:
            return False

    async def _ensure_access_token(self, account: Optional[Account], variables: Dict[str, Any]) -> str:
        if account is None:
            return ""
        token = (account.api_key or "").strip()
        cfg = account.extra_config or {}
        refresh = str(cfg.get("refresh_token") or "").strip()
        expires_at = float(cfg.get("expires_at") or 0)
        if refresh and (not token or (expires_at and expires_at - 120 <= time.time())):
            token = await self._refresh_access_token(account, variables)
        return token

    # ------------------------------------------------------------------
    # Simulation mode (offline tests / no account configured)
    # ------------------------------------------------------------------

    def _is_test_mode(self, api_key: str, account: Optional[Account] = None) -> bool:
        key = (api_key or "").lower().strip()
        if not key:
            if account is not None and not str(account.id).startswith("default-"):
                raise ValueError(
                    f"Account '{account.name}' has no Google credential. Use "
                    "'Sign in with Google' or 'Import local login' in the Accounts tab."
                )
            return True
        return any(key.startswith(prefix) for prefix in _SIM_PREFIXES)

    def _format_details(self, model_name: str, account: Optional[Account], prompt_tokens: int, completion_tokens: int) -> str:
        now_str = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        if not account or account.id.startswith("default-") or not account.api_key:
            status = "No account configured (Simulation Mode)"
        else:
            status = f"Account '{account.name}' (Test mode)"
        return (
            f"\n\n---\n"
            f"**Execution Details:**\n"
            f"• **Timestamp:** {now_str}\n"
            f"• **Model:** {model_name}\n"
            f"• **Tokens:** {prompt_tokens} prompt, {completion_tokens} completion "
            f"({prompt_tokens + completion_tokens} total)\n"
            f"• **Account status:** {status}"
        )

    def _simulation_response(self, request: ChatCompletionRequest, model_config: PluginModelConfig) -> ChatCompletionResponse:
        model_name = request.model or model_config.id
        base = (
            f"Simulated Colab/Ollama response for '{model_name}' "
            f"({model_config.upstream_model}). No Google Colab account is "
            f"configured, so no VM was started."
        )
        prompt_tokens = max(1, len(str(request.messages[-1].content).split()) if request.messages else 1)
        completion_tokens = len(base.split()) + 10
        content = base + self._format_details(model_name, None, prompt_tokens, completion_tokens)
        usage = CompletionUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=prompt_tokens + completion_tokens)
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}", created=int(time.time()), model=model_name,
            choices=[ChatChoice(index=0, message=ChatMessage(role="assistant", content=content), finish_reason="stop")],
            usage=usage,
        )

    async def _simulation_stream(self, request: ChatCompletionRequest, model_config: PluginModelConfig) -> AsyncIterator[ChatCompletionChunk]:
        resp = self._simulation_response(request, model_config)
        content = resp.choices[0].message.content or ""
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        yield ChatCompletionChunk(id=chunk_id, created=created, model=request.model,
                                  choices=[ChatChoiceDelta(index=0, delta={"role": "assistant", "content": ""})])
        for i in range(0, len(content), 24):
            yield ChatCompletionChunk(id=chunk_id, created=created, model=request.model,
                                      choices=[ChatChoiceDelta(index=0, delta={"content": content[i:i + 24]})])
        yield ChatCompletionChunk(id=chunk_id, created=created, model=request.model,
                                  choices=[ChatChoiceDelta(index=0, delta={}, finish_reason="stop")], usage=resp.usage)

    # ------------------------------------------------------------------
    # Colab MCP JSON-RPC client
    # ------------------------------------------------------------------

    def _request_headers(self, account: Optional[Account], variables: Dict[str, Any]) -> Dict[str, str]:
        headers: Dict[str, str] = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        token = (account.api_key if account and account.api_key else "").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        quota = ""
        if account and account.extra_config.get("quota_project"):
            quota = str(account.extra_config["quota_project"]).strip()
        if not quota:
            quota = str(variables.get("quota_project") or "").strip()
        if quota:
            headers["x-goog-user-project"] = quota
        raw_extra = str(variables.get("colab_headers") or "{}")
        try:
            extra = json.loads(raw_extra)
            if isinstance(extra, dict):
                headers.update({str(k): str(v) for k, v in extra.items()})
        except Exception:
            pass
        return headers

    def _proxy(self, variables: Dict[str, Any], account: Optional[Account]) -> Optional[str]:
        proxy = str(variables.get("proxy") or (getattr(account, "proxy_url", None) if account else "") or "").strip()
        return proxy or None

    def _endpoint(self, variables: Dict[str, Any]) -> str:
        return str(variables.get("mcp_base_url") or DEFAULT_MCP_URL).rstrip("/")

    @staticmethod
    def _parse_mcp_response(resp: httpx.Response) -> Dict[str, Any]:
        text = resp.text
        content_type = resp.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            envelope: Optional[Dict[str, Any]] = None
            for line in text.splitlines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                try:
                    ev = json.loads(line[5:].strip())
                except Exception:
                    continue
                if isinstance(ev, dict) and ("result" in ev or "error" in ev):
                    envelope = ev
            if envelope is None:
                raise ColabError(f"Colab MCP SSE had no terminal result: {text[:300]}")
            return envelope
        try:
            return json.loads(text)
        except Exception as exc:
            raise ColabError(f"Colab MCP returned non-JSON response: {text[:300]}") from exc

    @staticmethod
    def _extract_tool_result(envelope: Dict[str, Any]) -> Dict[str, Any]:
        if envelope.get("error"):
            err = envelope["error"]
            raise ColabError(f"Colab MCP error ({err.get('code')}): {err.get('message')}")
        result = envelope.get("result")
        if not isinstance(result, dict):
            raise ColabError(f"Colab MCP result missing: {envelope}")
        structured = result.get("structuredContent")
        if isinstance(structured, dict):
            return structured
        content = result.get("content") or []
        texts = [c.get("text", "") for c in content if isinstance(c, dict) and isinstance(c.get("text"), str)]
        text = "\n".join(texts).strip()
        if result.get("isError"):
            raise ColabError(text or "Colab tool returned an error")
        if not text:
            return {}
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {"value": text}
        except Exception:
            return {"stdout": text}

    async def _call_tool(self, name: str, arguments: Dict[str, Any], account: Optional[Account], variables: Dict[str, Any], timeout: Optional[float] = None, request_id: Optional[int] = None) -> Dict[str, Any]:
        endpoint = self._endpoint(variables)
        headers = self._request_headers(account, variables)
        timeout = timeout or float(variables.get("timeout") or 620)
        payload = {"jsonrpc": "2.0", "id": request_id if request_id is not None else int(time.time_ns() % 1_000_000), "method": "tools/call", "params": {"name": name, "arguments": arguments}}
        try:
            proxy = self._proxy(variables, account)
            async with httpx.AsyncClient(timeout=timeout, proxy=proxy) as client:
                resp = await client.post(endpoint, headers=headers, json=payload)
                resp.raise_for_status()
                envelope = self._parse_mcp_response(resp)
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:400]
            if exc.response.status_code in (401, 403):
                raise ColabError(f"Colab API auth error ({exc.response.status_code}): {detail}") from exc
            raise ColabError(f"Colab API HTTP {exc.response.status_code}: {detail}") from exc
        except httpx.TimeoutException as exc:
            raise ColabError("DEADLINE_EXCEEDED: Colab MCP call timed out (re-attachable)") from exc
        except httpx.TransportError as exc:
            raise ColabError("UNAVAILABLE: Colab MCP transport error (re-attachable)") from exc
        return self._extract_tool_result(envelope)

    async def _call_tool_reattach(self, name: str, arguments: Dict[str, Any], account: Optional[Account], variables: Dict[str, Any], timeout: Optional[float] = None, attempts: int = 30) -> Dict[str, Any]:
        last: Optional[Exception] = None
        for _ in range(attempts):
            try:
                return await self._call_tool(name, arguments, account, variables, timeout=timeout)
            except ColabError as exc:
                last = exc
                message = str(exc)
                if "DEADLINE_EXCEEDED" in message or "UNAVAILABLE" in message:
                    await asyncio.sleep(2.0)
                    continue
                raise
        raise ColabError(f"Colab call never completed after {attempts} attempts: {last}")

    @staticmethod
    def _sanitize_runtime_id(prefix: str, account_id: str) -> str:
        raw = re.sub(r"[^a-z0-9-]", "", (account_id or "a").lower())[:24]
        if not raw:
            raw = "a"
        if not raw[0].isalpha():
            raw = "a" + raw
        return f"{prefix}-{raw}"[:63].rstrip("-")

    async def _list_runtimes(self, account: Optional[Account], variables: Dict[str, Any]) -> list:
        result = await self._call_tool("list_runtimes", {}, account, variables)
        runtimes = result.get("runtimes") or []
        if not isinstance(runtimes, list):
            runtimes = []
        return runtimes

    async def _list_runtime_specs(self, account: Optional[Account], variables: Dict[str, Any]) -> list:
        result = await self._call_tool("list_runtime_specs", {}, account, variables)
        specs = result.get("runtimeSpecs") or []
        if not isinstance(specs, list) and isinstance(result.get("value"), list):
            specs = result["value"]
        return specs

    async def _pick_spec(self, account: Optional[Account], variables: Dict[str, Any]) -> Dict[str, str]:
        accelerator = str(variables.get("accelerator") or "T4").upper()
        shape = str(variables.get("runtime_shape") or "SHAPE_STANDARD").upper()
        variant = str(variables.get("runtime_variant") or "VARIANT_GPU").upper()
        requested = {"variant": variant, "accelerator": accelerator, "shape": shape}
        try:
            specs = await self._list_runtime_specs(account, variables)
            eligible = [s.get("key") for s in specs if isinstance(s, dict) and s.get("eligible") and isinstance(s.get("key"), dict)]
            if not eligible:
                raise ColabError("No eligible Colab runtime specs for this account. Check the account login and Colab quota.")
            exact = next((k for k in eligible if k.get("variant") == variant and k.get("accelerator") == accelerator and k.get("shape") == shape), None)
            if exact:
                return exact
            preferred = [a for a in ("T4", "L4", "A100", "G4", "H100") if a != accelerator]
            for acc in preferred:
                alt = next((k for k in eligible if k.get("variant") == "VARIANT_GPU" and k.get("accelerator") == acc and k.get("shape") == shape), None)
                if alt:
                    return alt
            gpu = next((k for k in eligible if k.get("variant") == "VARIANT_GPU"), None)
            if gpu:
                return gpu
            if bool(variables.get("allow_cpu_fallback", False)):
                cpu = next((k for k in eligible if k.get("variant") == "VARIANT_CPU"), None)
                if cpu:
                    return cpu
            raise ColabError("No eligible GPU runtime spec found; enable 'Allow CPU Fallback' or use an account with GPU quota.")
        except ColabError:
            raise
        except Exception:
            return requested

    async def _get_or_create_runtime(self, account: Account, variables: Dict[str, Any]) -> str:
        account_key = account.id
        cached = self._runtimes.get(account_key)
        if cached and bool(variables.get("reuse_runtime", True)):
            return cached
        prefix = str(variables.get("runtime_prefix") or RUNTIME_PREFIX).strip("- _")
        if not prefix:
            prefix = RUNTIME_PREFIX
        reuse = bool(variables.get("reuse_runtime", True))
        if reuse:
            try:
                runtimes = await self._list_runtimes(account, variables)
                for rt in reversed(runtimes):
                    if not isinstance(rt, dict):
                        continue
                    name = rt.get("name") or ""
                    rid = name.split("/")[-1]
                    if rid.startswith(prefix):
                        self._runtimes[account_key] = name
                        return name
            except ColabError:
                pass
        runtime_id = self._sanitize_runtime_id(prefix, account.id)
        arguments = {"requestId": str(uuid.uuid4()), "runtimeId": runtime_id, "runtime": {"runtimeSpec": await self._pick_spec(account, variables)}}
        wait = float(variables.get("max_startup_wait") or 420)
        try:
            result = await self._call_tool_reattach("create_runtime", arguments, account, variables, timeout=wait)
        except ColabError as exc:
            if "ALREADY_EXISTS" in str(exc) or "already exists" in str(exc).lower():
                try:
                    result = await self._call_tool("get_runtime", {"name": f"runtimes/{runtime_id}"}, account, variables)
                except Exception:
                    raise exc
            else:
                raise
        name = result.get("name") or f"runtimes/{runtime_id}"
        self._runtimes[account_key] = name
        return name

    def _worker_code(self, payload: Dict[str, Any], out_file: str = WORKER_OUT_FILE) -> str:
        payload_b64 = base64.b64encode(json.dumps(payload, ensure_ascii=False).encode("utf-8")).decode("ascii")
        code = WORKER_CELL
        code = code.replace("@@PAYLOAD_B64_JSON@@", json.dumps(payload_b64))
        code = code.replace("@@OUT_FILE_JSON@@", json.dumps(out_file))
        code = code.replace("@@RESULT_MARK@@", RESULT_MARK)
        code = code.replace("@@ERROR_MARK@@", ERROR_MARK)
        return code

    def _probe_code(self, out_file: str = WORKER_OUT_FILE) -> str:
        code = PROBE_CELL
        code = code.replace("@@OUT_FILE_JSON@@", json.dumps(out_file))
        code = code.replace("@@PROBE_MARK@@", PROBE_MARK)
        return code

    @staticmethod
    def _parse_markers(stdout: str) -> Optional[Dict[str, Any]]:
        if not stdout:
            return None
        for line in stdout.splitlines():
            if line.startswith(RESULT_MARK):
                try:
                    return json.loads(line[len(RESULT_MARK):].strip())
                except Exception:
                    return None
            if line.startswith(ERROR_MARK):
                try:
                    err = json.loads(line[len(ERROR_MARK):].strip())
                except Exception:
                    err = {"message": line[len(ERROR_MARK):].strip()}
                raise ColabError(f"Colab VM worker failed: {err.get('message')}")
        return None

    def _encode_execute_args(self, runtime: str, session: str, code: str, execution_id: str) -> Dict[str, str]:
        return {"session": f"{runtime}/sessions/{session}", "code": code, "executionId": execution_id}

    async def _run_worker(self, runtime: str, session: str, code: str, account: Account, variables: Dict[str, Any], timeout: Optional[float] = None) -> Dict[str, Any]:
        execution_id = str(uuid.uuid4())
        result = await self._call_tool_reattach("execute_code", self._encode_execute_args(runtime, session, code, execution_id), account, variables, timeout=timeout)
        execution_error = result.get("executionError")
        if execution_error:
            name = execution_error.get("name", "Error")
            value = execution_error.get("value", "")
            raise ColabError(f"Colab kernel error: {name}: {value}")
        stdout = result.get("stdout") or ""
        marker = self._parse_markers(stdout)
        return {"stdout": stdout, "marker": marker}

    async def _probe_tail(self, runtime: str, session: str, out_file: str, account: Account, variables: Dict[str, Any]) -> list:
        code = self._probe_code(out_file)
        try:
            result = await self._call_tool("execute_code", self._encode_execute_args(runtime, session, code, str(uuid.uuid4())), account, variables, timeout=min(30.0, float(variables.get("timeout") or 620)))
        except ColabError:
            return []
        stdout = result.get("stdout") or ""
        for line in stdout.splitlines():
            if line.startswith(PROBE_MARK):
                payload = line[len(PROBE_MARK):].strip()
                try:
                    parsed = json.loads(payload)
                    if isinstance(parsed, list):
                        return parsed
                except Exception:
                    return []
        return []

    async def _delete_session(self, runtime: str, session: str, account: Account, variables: Dict[str, Any]) -> None:
        try:
            await self._call_tool("delete_session", {"name": f"{runtime}/sessions/{session}"}, account, variables, timeout=30)
        except Exception:
            pass

    def _build_payload(self, request: ChatCompletionRequest, model_config: PluginModelConfig, stream: bool, variables: Dict[str, Any]) -> Dict[str, Any]:
        body: Dict[str, Any] = {"model": model_config.upstream_model or model_config.id, "messages": [m.model_dump(exclude_none=True) for m in request.messages], "stream": stream}
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.top_p is not None:
            body["top_p"] = request.top_p
        if request.presence_penalty is not None:
            body["presence_penalty"] = request.presence_penalty
        if request.frequency_penalty is not None:
            body["frequency_penalty"] = request.frequency_penalty
        max_tokens = request.max_tokens if request.max_tokens is not None else request.max_completion_tokens
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if request.tools is not None:
            body["tools"] = request.tools
        if request.tool_choice is not None:
            body["tool_choice"] = request.tool_choice
        if request.parallel_tool_calls is not None:
            body["parallel_tool_calls"] = request.parallel_tool_calls
        if request.response_format is not None:
            body["response_format"] = request.response_format
        if request.stop is not None:
            body["stop"] = request.stop
        if stream:
            stream_options = dict(request.stream_options or {})
            stream_options["include_usage"] = True
            body["stream_options"] = stream_options
        extra = request.extra_body if isinstance(request.extra_body, dict) else {}
        for key in OLLAMA_EXTRA_FIELDS:
            if extra.get(key) is not None:
                body[key] = extra[key]
        num_ctx = variables.get("num_ctx")
        try:
            if num_ctx and int(num_ctx) > 0:
                body["num_ctx"] = int(num_ctx)
        except (TypeError, ValueError):
            pass
        return body

    async def chat_completion(self, request: ChatCompletionRequest, model_config: PluginModelConfig, variables: Dict[str, Any], account: Optional[Account] = None) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionChunk]]:
        api_key = await self._ensure_access_token(account, variables)
        if self._is_test_mode(api_key, account):
            if request.stream:
                return self._simulation_stream(request, model_config)
            return self._simulation_response(request, model_config)
        if messages_have_images(request.messages):
            raise ValueError(f"Colab Ollama model '{model_config.id}' does not support image input.")
        payload = self._build_payload(request, model_config, bool(request.stream), variables)
        try:
            runtime = await self._get_or_create_runtime(account, variables)
        except ColabError:
            if account:
                self._runtimes.pop(account.id, None)
            raise
        if request.stream:
            return self._stream_colab(request, model_config, variables, account, runtime, payload)
        return await self._complete_colab(request, model_config, variables, account, runtime, payload)

    async def _complete_colab(self, request: ChatCompletionRequest, model_config: PluginModelConfig, variables: Dict[str, Any], account: Account, runtime: str, payload: Dict[str, Any]) -> ChatCompletionResponse:
        req_id = uuid.uuid4().hex[:8]
        session = f"llmproxy-w-{req_id}"
        code = self._worker_code(payload, out_file="")
        timeout = min(float(variables.get("timeout") or 620), float(variables.get("max_startup_wait") or 420) + 200)
        try:
            result = await self._run_worker(runtime, session, code, account, variables, timeout=timeout)
            marker = result["marker"]
            if marker is None:
                raise ColabError("Colab worker returned no result marker. Inspect llmproxy logs or run the setup cell manually.")
            data = dict(marker)
            data.pop("mode", None)
            data.pop("done", None)
            if not data.get("choices"):
                raise ColabError(f"Colab Ollama completion returned no choices: {json.dumps(data)[:400]}")
            data["model"] = request.model
            data.pop("timings", None)
            response = ChatCompletionResponse.model_validate(data)
        finally:
            await self._delete_session(runtime, session, account, variables)
        return response

    async def _stream_colab(self, request: ChatCompletionRequest, model_config: PluginModelConfig, variables: Dict[str, Any], account: Account, runtime: str, payload: Dict[str, Any]) -> AsyncIterator[ChatCompletionChunk]:
        out_file = WORKER_OUT_FILE
        req_id = uuid.uuid4().hex[:8]
        worker_session = f"llmproxy-w-{req_id}"
        probe_session = f"llmproxy-p-{req_id}"
        code = self._worker_code(payload, out_file=out_file)
        call_timeout = min(float(variables.get("timeout") or 620), float(variables.get("max_startup_wait") or 420) + 200)
        poll = max(0.5, float(variables.get("poll_interval") or 2))
        worker_task = asyncio.create_task(self._run_worker(runtime, worker_session, code, account, variables, timeout=call_timeout))

        async def _stream_impl() -> AsyncIterator[ChatCompletionChunk]:
            chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            created = int(time.time())
            seen = 0
            yielded_finish = False
            worker_error: Optional[Exception] = None
            sent_role = False
            final_usage: Optional[CompletionUsage] = None
            worker_done = False

            def make_chunk(obj: Dict[str, Any]) -> Optional[ChatCompletionChunk]:
                nonlocal yielded_finish, final_usage, sent_role
                try:
                    obj.pop("timings", None)
                    chunk = ChatCompletionChunk.model_validate(obj)
                except Exception:
                    return None
                chunk = chunk.model_copy(update={"model": request.model})
                if chunk.usage is not None:
                    final_usage = chunk.usage
                for choice in chunk.choices:
                    delta = choice.delta or {}
                    if delta.get("role"):
                        sent_role = True
                    if choice.finish_reason is not None:
                        yielded_finish = True
                return chunk

            try:
                while True:
                    lines: list = []
                    if worker_task.done():
                        try:
                            worker_res = await worker_task
                        except Exception as exc:
                            worker_error = exc
                        if worker_error is None and worker_res is not None:
                            marker = worker_res.get("marker")
                            if marker is not None:
                                if marker.get("mode") == "stream" or marker.get("done"):
                                    worker_done = True
                    else:
                        lines = await self._probe_tail(runtime, probe_session, out_file, account, variables)
                        if not lines:
                            await asyncio.sleep(poll)
                            continue
                    if lines:
                        for line in lines:
                            seen += 1
                            if isinstance(line, dict) and line.get("__coLabDone__"):
                                worker_done = True
                                continue
                            if isinstance(line, dict):
                                chunk = make_chunk(line)
                                if chunk is not None:
                                    yield chunk
                    if worker_done:
                        if worker_error is None:
                            tail = await self._probe_tail(runtime, probe_session, out_file, account, variables)
                            for line in tail[seen:]:
                                seen += 1
                                if isinstance(line, dict) and line.get("__coLabDone__"):
                                    continue
                                if isinstance(line, dict):
                                    chunk = make_chunk(line)
                                    if chunk is not None:
                                        yield chunk
                        break
                    if worker_error is not None:
                        raise worker_error
                    await asyncio.sleep(poll)
                if worker_error is not None:
                    raise worker_error
                if not sent_role and not yielded_finish:
                    raise ColabError("Colab worker produced no streaming chunks; check the model name and Colab runtime logs.")
                if not yielded_finish:
                    yield ChatCompletionChunk(id=chunk_id, created=created, model=request.model, choices=[ChatChoiceDelta(index=0, delta={}, finish_reason="stop")], usage=final_usage)
            except Exception as exc:
                if not worker_task.done():
                    worker_task.cancel()
                raise
            finally:
                if not worker_task.done():
                    worker_task.cancel()
                await self._delete_session(runtime, worker_session, account, variables)
                await self._delete_session(runtime, probe_session, account, variables)

        return _stream_impl()


plugin = ColabOllamaPlugin()
