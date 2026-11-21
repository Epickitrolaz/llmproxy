"""Ollama on Google Colab, using the official google-colab-cli wire protocol.

This is a compute provider, not a Gemini API adapter. Google OAuth allocates an
owned Colab VM; an authenticated Jupyter connection installs/starts Ollama and
relays its native /api/chat output. No CLI subprocess, public tunnel, local
Google SDK, or network activity at import time is needed.

VMs, kernels, model downloads and loaded models are reused. Account selection
and failover remain router-owned. A VM executes one request at a time; enabling
split_across_accounts distributes independent requests over independent VMs,
not model shards. Only this plugin's assignments are ever released. Idle and
maximum-lifetime limits do not interrupt an in-flight request. There are no
artificial keep-alive pings; Colab may reclaim an idle runtime sooner.

Ollama reports prompt_eval_count, eval_count and, on recent releases,
prompt_eval_cached_count. Prefix/KV caching is implicit, not a durable explicit
cache API. Cache extension fields are forwarded, but are not claimed to be
implemented by Ollama. Thinking is model-specific; /api/show supplies the native
controls. Thinking text is exposed as reasoning_content. Ollama does not report
a separate reasoning-token count, so this plugin does not fabricate one.

Protocol reference: googlecolab/google-colab-cli at
0abf82dadc2b95c02b6ab2c08b1b4d664e616fbb (Apache-2.0), client.py/auth.py/runtime.py.
Implementation is independent, using only stdlib, httpx and LLMProxy contracts.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import math
import os
import re
import secrets
import struct
import time
import uuid
from contextlib import aclosing, asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Union
from urllib.parse import parse_qs, quote, urlencode, urlparse

import httpx

from llmproxy.core.interface import BasePlugin
from llmproxy.core.multimodal import (
    content_text,
    is_image_part,
    materialize_message_images,
    parse_data_url,
    to_openai_image_part,
)
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

_COLAB_URL = "https://colab.research.google.com"
_TOKEN_URL = "https://oauth2.googleapis.com/token"
_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"
_REDIRECT_URI = "https://sdk.cloud.google.com/applicationdefaultauthcode.html"
_SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/colaboratory",
]
# Public installed-application configuration shipped by Google's CLI. This is
# not an account credential or a confidential OAuth client secret. A local
# ~/.colab-cli-oauth-config.json can supply a different registered client.
_PUBLIC_CLIENT = {
    "client_id": "764086051850-6qr4p6gpi6hn506pt8ejuq83di341hur.apps.googleusercontent.com",
    "client_secret": "d-FL95Q19q7MQmFpd7hHD0Ty",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": _TOKEN_URL,
}
_RESULT_PREFIX = "LLMPROXY_COLAB:"
_STATE_KEY = "colab_runtime"
_GPU_CHOICES = ["CPU", "T4", "L4", "G4", "A100", "H100"]
_TEST_PREFIXES = ("test", "mock", "dummy", "fake", "offline", "sample")
_MAX_MESSAGE_BYTES = 32 * 1024 * 1024


class ColabError(RuntimeError):
    """A short, credential-safe error whose status remains visible to failover."""

    def __init__(self, message: str, status_code: int = 500):
        self.status_code = status_code
        super().__init__(f"Colab HTTP {status_code}: {message}")


class _KernelUnavailable(ColabError):
    """The saved kernel disappeared; reconnect without reallocating its VM."""


def _excerpt(value: Any, sensitive: tuple[str, ...] = ()) -> str:
    if isinstance(value, dict):
        error = value.get("error", value.get("message", "Request rejected"))
        if isinstance(error, dict):
            error = error.get("message", error.get("status", "Request rejected"))
        value = value.get("error_description", error)
    text = str(value or "Request rejected")
    for secret in sensitive:
        if secret:
            text = text.replace(secret, "[redacted]")
    text = re.sub(
        r"(?i)((?:bearer\s+)|(?:[\w-]*(?:token|secret|code_verifier)[\"']?\s*[:=]\s*[\"']?))"
        r"[^\s,\"'&}]+",
        r"\1[redacted]",
        text,
    )
    return " ".join(text.split())[:240]


def _json_response(response: httpx.Response) -> Dict[str, Any]:
    text = response.text
    if text.startswith(")]}'"):
        text = text.split("\n", 1)[-1]
    try:
        data = json.loads(text) if text.strip() else {}
    except ValueError:
        raise ColabError("Backend returned invalid JSON", 502) from None
    if not isinstance(data, dict):
        raise ColabError("Backend returned a non-object response", 502)
    return data


def _check_http(response: httpx.Response, sensitive: tuple[str, ...] = ()) -> None:
    if response.status_code < 400:
        return
    try:
        body: Any = _json_response(response)
    except ColabError:
        body = response.text
    status = response.status_code
    message = _excerpt(body, sensitive)
    if status in (401, 403):
        message += "; sign in again with Colab and email permissions"
    if status == 412:
        # The CLI describes this as an assignment/capacity limit. Let the
        # router put the account in cooldown instead of spinning new VMs.
        message = "Runtime allocation quota/capacity limit (HTTP 412); " + message
    raise ColabError(message, status)


def _number(values: Dict[str, Any], name: str, default: float, minimum: float,
            maximum: float, *, integer: bool = False) -> Union[int, float]:
    raw = values.get(name, default)
    try:
        if isinstance(raw, bool):
            raise ValueError
        value = float(raw)
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError
        if integer and not value.is_integer():
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be {'an integer' if integer else 'a number'} "
                         f"between {minimum:g} and {maximum:g}") from None
    return int(value) if integer else value


def _boolean(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    raise ValueError(f"{name} must be true or false")


def _colab_url(value: str) -> str:
    parsed = urlparse(value)
    if (parsed.scheme != "https" or parsed.hostname not in (
        "colab.research.google.com", "colab.sandbox.google.com"
    ) or parsed.username or parsed.password or parsed.port not in (None, 443)
            or parsed.path not in ("", "/") or parsed.query or parsed.fragment):
        raise ValueError("base_url must be a Google Colab HTTPS host, not a Gemini/Ollama API URL")
    return value.rstrip("/")


def _runtime_url(value: str, colab_url: str) -> str:
    parsed = urlparse(value)
    host = parsed.hostname or ""
    if (parsed.scheme != "https" or parsed.username or parsed.password
            or parsed.port not in (None, 443) or parsed.query or parsed.fragment
            or not (host == urlparse(colab_url).hostname or host.endswith(".googleusercontent.com"))):
        raise ColabError("Backend returned an untrusted runtime URL", 502)
    return value.rstrip("/")


@dataclass(frozen=True)
class _Settings:
    base_url: str
    gpu: str
    high_mem: bool
    timeout: float
    request_timeout: float
    startup_timeout: float
    assign_retries: int
    retry_backoff: float
    vm_idle_seconds: float
    vm_max_lifetime_seconds: float
    max_active_vms: int
    health_check_seconds: float
    model_keep_alive_seconds: int
    preload_model: bool
    release_vms_on_shutdown: bool
    ollama_version: str
    models_directory: str
    proxy: Optional[str] = field(repr=False)

    @classmethod
    def read(cls, variables: Dict[str, Any], account: Optional[Account] = None) -> _Settings:
        values = dict(variables)
        # Only settings, never credentials, may override provider defaults.
        if account:
            overrides = account.extra_config.get("colab_settings", {})
            if not isinstance(overrides, dict):
                raise ValueError("Account colab_settings must be a JSON object")
            values.update(overrides)
        gpu = str(values.get("gpu", "T4")).upper()
        if gpu not in _GPU_CHOICES:
            raise ValueError("gpu must be one of " + ", ".join(_GPU_CHOICES) + "; Ollama does not support TPU")
        version = str(values.get("ollama_version", "latest")).strip()
        if version != "latest" and not re.fullmatch(r"v?\d+\.\d+\.\d+(?:[-.][\w.-]+)?", version):
            raise ValueError("ollama_version must be latest or an Ollama release version")
        directory = str(values.get("models_directory", "/content/.llmproxy-colab/models")).strip()
        if not directory.startswith("/") or "\x00" in directory or ".." in directory.split("/"):
            raise ValueError("models_directory must be an absolute VM directory without '..'")
        return cls(
            base_url=_colab_url((account.base_url if account and account.base_url else
                                str(values.get("base_url", _COLAB_URL)))),
            gpu=gpu,
            high_mem=_boolean(values.get("high_mem", False), "high_mem") and gpu != "L4",
            timeout=float(_number(values, "timeout", 300, 1, 86400)),
            request_timeout=float(_number(values, "request_timeout", 30, 1, 600)),
            startup_timeout=float(_number(values, "startup_timeout", 900, 1, 3600)),
            assign_retries=int(_number(values, "assign_retries", 2, 0, 10, integer=True)),
            retry_backoff=float(_number(values, "assign_retry_backoff", 0.5, 0, 60)),
            vm_idle_seconds=float(_number(values, "vm_idle_seconds", 900, -1, 86400)),
            vm_max_lifetime_seconds=float(_number(values, "vm_max_lifetime_seconds", 3600, 0, 86400)),
            max_active_vms=int(_number(values, "max_active_vms", 0, 0, 1000, integer=True)),
            health_check_seconds=float(_number(values, "health_check_seconds", 30, 0, 3600)),
            model_keep_alive_seconds=int(_number(values, "model_keep_alive_seconds", -1, -1, 86400, integer=True)),
            preload_model=_boolean(values.get("preload_model", True), "preload_model"),
            release_vms_on_shutdown=_boolean(values.get("release_vms_on_shutdown", True), "release_vms_on_shutdown"),
            ollama_version=version.removeprefix("v"),
            models_directory=directory,
            # A resolved router proxy always wins over an account setting.
            proxy=str(variables.get("proxy") or (account.proxy_url if account else "") or "").strip() or None,
        )


class _KernelSocket:
    """Small RFC6455 transport over httpx's documented network_stream extension.

    This keeps the plugin standalone and honors httpx's HTTP/SOCKS proxy path.
    No private httpcore classes or optional WebSocket/Google SDK imports are used.
    The Jupyter default subprotocol uses JSON text or offset-framed binary JSON.
    """

    def __init__(self, response: httpx.Response, timeout: float):
        self.response = response
        self.stream = response.extensions["network_stream"]
        self.timeout = timeout
        self.closed = False
        self.buffer = bytearray()
        self.write_lock = asyncio.Lock()

    @classmethod
    async def connect(cls, client: httpx.AsyncClient, url: str,
                      headers: Dict[str, str], timeout: float) -> _KernelSocket:
        key = base64.b64encode(secrets.token_bytes(16)).decode("ascii")
        request = client.build_request("GET", url, headers={
            **headers, "Connection": "Upgrade", "Upgrade": "websocket",
            "Sec-WebSocket-Key": key, "Sec-WebSocket-Version": "13",
        }, timeout=timeout)
        response = await client.send(request, stream=True)
        expected = base64.b64encode(hashlib.sha1(
            (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")
        ).digest()).decode("ascii")
        if response.status_code != 101:
            try:
                await response.aread()
                _check_http(response, tuple(headers.values()))
                raise ColabError("Jupyter did not accept the WebSocket upgrade", 502)
            finally:
                await response.aclose()
        if (response.headers.get("sec-websocket-accept") != expected
                or response.headers.get("upgrade", "").lower() != "websocket"
                or "upgrade" not in response.headers.get("connection", "").lower()
                or response.headers.get("sec-websocket-extensions")
                or "network_stream" not in response.extensions):
            await response.aclose()
            raise ColabError("Invalid Jupyter WebSocket handshake", 502)
        return cls(response, timeout)

    async def _read_exactly(self, size: int) -> bytes:
        while len(self.buffer) < size:
            data = await self.stream.read(max_bytes=min(65536, size - len(self.buffer)),
                                          timeout=self.timeout)
            if not data:
                self.closed = True
                raise ColabError("Jupyter connection closed before execution finished", 502)
            self.buffer.extend(data)
        result = bytes(self.buffer[:size])
        del self.buffer[:size]
        return result

    async def _send_frame(self, opcode: int, payload: bytes) -> None:
        if self.closed:
            raise ColabError("Jupyter connection is closed", 502)
        size = len(payload)
        if size > _MAX_MESSAGE_BYTES:
            raise ValueError("Colab message exceeds 32 MiB")
        header = bytes([0x80 | opcode])
        if size < 126:
            header += bytes([0x80 | size])
        elif size < 65536:
            header += bytes([0x80 | 126]) + struct.pack("!H", size)
        else:
            header += bytes([0x80 | 127]) + struct.pack("!Q", size)
        mask = secrets.token_bytes(4)
        masked = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
        async with self.write_lock:
            await self.stream.write(header + mask + masked, timeout=self.timeout)

    async def send_json(self, message: Dict[str, Any]) -> None:
        await self._send_frame(1, json.dumps(message, ensure_ascii=False).encode("utf-8"))

    async def receive_json(self) -> Dict[str, Any]:
        payload = bytearray()
        message_opcode = 0
        while True:
            first, second = await self._read_exactly(2)
            final, opcode = bool(first & 0x80), first & 15
            if first & 0x70 or second & 0x80:
                raise ColabError("Invalid server WebSocket frame", 502)
            size = second & 127
            if size == 126:
                size = struct.unpack("!H", await self._read_exactly(2))[0]
            elif size == 127:
                size = struct.unpack("!Q", await self._read_exactly(8))[0]
            if size > _MAX_MESSAGE_BYTES or len(payload) + size > _MAX_MESSAGE_BYTES:
                raise ColabError("Jupyter message exceeds 32 MiB", 502)
            if opcode >= 8 and (not final or size > 125):
                raise ColabError("Invalid WebSocket control frame", 502)
            part = await self._read_exactly(size)
            if opcode == 8:
                self.closed = True
                raise ColabError("Jupyter disconnected before execution finished", 502)
            if opcode == 9:
                await self._send_frame(10, part)
                continue
            if opcode == 10:
                continue
            if opcode in (1, 2) and not message_opcode:
                message_opcode = opcode
            elif opcode != 0 or not message_opcode:
                raise ColabError("Invalid fragmented Jupyter message", 502)
            payload.extend(part)
            if not final:
                continue
            raw = bytes(payload)
            if message_opcode == 2:
                # Jupyter default binary framing: count, offsets, JSON, buffers.
                if len(raw) < 8:
                    raise ColabError("Invalid binary Jupyter message", 502)
                count = struct.unpack("!I", raw[:4])[0]
                if count < 1 or 4 * (count + 1) > len(raw):
                    raise ColabError("Invalid binary Jupyter offsets", 502)
                offsets = struct.unpack(f"!{count}I", raw[4:4 * (count + 1)])
                if (offsets[0] != 4 * (count + 1) or list(offsets) != sorted(offsets)
                        or offsets[-1] > len(raw)):
                    raise ColabError("Invalid binary Jupyter offsets", 502)
                raw = raw[offsets[0]:offsets[1] if count > 1 else len(raw)]
            try:
                result = json.loads(raw)
            except (ValueError, UnicodeError):
                raise ColabError("Jupyter returned invalid message JSON", 502) from None
            if not isinstance(result, dict):
                raise ColabError("Jupyter returned a non-object message", 502)
            return result

    async def aclose(self) -> None:
        try:
            if not self.closed:
                self.timeout = min(self.timeout, 3)
                async with asyncio.timeout(3):
                    await self._send_frame(8, struct.pack("!H", 1000))
        except (Exception, asyncio.CancelledError):
            pass
        finally:
            self.closed = True
            await self.response.aclose()


# Executed only on a plugin-owned VM. Request/model/settings values are passed
# as JSON literals, never interpolated into a shell command. Ollama is loopback
# only and cloud access is disabled; Google/account/proxy secrets never enter it.
_REMOTE_SOURCE = r'''
import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

PREFIX = "LLMPROXY_COLAB:"
BASE = "http://127.0.0.1:11434"
process = globals().get("process")
server_config = globals().get("server_config")

def emit(request_id, kind, data):
    print(PREFIX + json.dumps({"id": request_id, "type": kind, "data": data}), flush=True)

def api(path, payload=None, timeout=30):
    body = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(BASE + path, data=body, headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)

def prepare_model(config, request_id, deadline):
    def remaining():
        left = deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError("Ollama model preparation timed out")
        return left
    model = config["model"]
    try:
        with api("/api/show", {"model": model}, remaining()) as response:
            info = json.load(response)
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
        with api("/api/pull", {"model": model, "stream": True}, remaining()) as response:
            for line in response:
                remaining()
                progress = json.loads(line)
                if progress.get("error"):
                    raise RuntimeError(str(progress["error"])[:200])
        with api("/api/show", {"model": model}, remaining()) as response:
            info = json.load(response)
    if info.get("remote_host") or info.get("remote_model"):
        raise ValueError("Cloud Ollama models are not supported by the Colab compute provider")
    if config["preload_model"]:
        with api("/api/generate", {"model": model, "prompt": "", "stream": False,
                 "keep_alive": config["model_keep_alive_seconds"],
                 "options": {"num_ctx": config["num_ctx"]}}, remaining()) as response:
            ready = json.load(response)
            if ready.get("error"):
                raise RuntimeError(str(ready["error"])[:200])
    emit(request_id, "ready", {"capabilities": info.get("capabilities", []),
                              "thinking": info.get("thinking")})

def bootstrap(config, request_id):
    global process, server_config
    deadline = time.monotonic() + config["startup_timeout"]
    def remaining():
        left = deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError("Ollama startup timed out")
        return left
    def run(argv):
        result = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, timeout=remaining())
        if result.returncode:
            raise RuntimeError("Ollama setup command failed: " + argv[0] + " (exit " + str(result.returncode) + ")")
    signature = (config["ollama_version"], config["models_directory"])
    if process is not None and process.poll() is None and server_config != signature:
        process.terminate()
        try:
            process.wait(timeout=min(5, remaining()))
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=min(5, remaining()))
    running = False
    try:
        with api("/api/version", timeout=min(1, remaining())) as response:
            json.load(response)
        running = True
    except (OSError, urllib.error.URLError):
        pass
    if not running:
        root = Path("/content/.llmproxy-colab/ollama")
        binary = root / "bin/ollama"
        marker = root / "llmproxy-version"
        desired = config["ollama_version"]
        installed = binary.exists() and marker.exists() and marker.read_text() == desired
        system_binary = shutil.which("ollama") if desired == "latest" else None
        if not installed and not system_binary:
            arch = {"x86_64": "amd64", "aarch64": "arm64"}.get(platform.machine())
            if not arch:
                raise RuntimeError("Unsupported Colab VM architecture")
            if not shutil.which("zstd"):
                # Colab already has NVIDIA drivers. Install only the tiny archive
                # decoder, not the CLI, a Python inference stack or GPU drivers.
                run(["apt-get", "install", "-y", "-qq", "--no-install-recommends", "zstd"])
            root.mkdir(parents=True, exist_ok=True)
            suffix = "" if desired == "latest" else "?" + urllib.parse.urlencode({"version": desired})
            url = "https://ollama.com/download/ollama-linux-" + arch + ".tar.zst" + suffix
            with tempfile.TemporaryDirectory(prefix="llmproxy-ollama-") as temporary:
                archive = str(Path(temporary) / "ollama.tar.zst")
                run(["curl", "--fail", "--silent", "--show-error", "--location",
                     "--connect-timeout", "15", "--max-time", str(max(1, int(remaining()))),
                     "--output", archive, url])
                run(["tar", "--zstd", "-xf", archive, "-C", str(root)])
            if not binary.is_file():
                raise RuntimeError("Ollama archive did not contain bin/ollama")
            marker.write_text(desired)
        env = dict(os.environ, OLLAMA_HOST="127.0.0.1:11434", OLLAMA_NO_CLOUD="1",
                   OLLAMA_MODELS=config["models_directory"], OLLAMA_NUM_PARALLEL="1",
                   OLLAMA_MAX_LOADED_MODELS="1", OLLAMA_KEEP_ALIVE="-1")
        Path(config["models_directory"]).mkdir(parents=True, exist_ok=True)
        process = subprocess.Popen([system_binary or str(binary), "serve"], env=env,
                                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, start_new_session=True)
        server_config = signature
        while True:
            if process.poll() is not None:
                raise RuntimeError("Ollama server exited during startup")
            try:
                with api("/api/version", timeout=min(1, remaining())) as response:
                    json.load(response)
                break
            except (OSError, urllib.error.URLError):
                emit(request_id, "progress", {"phase": "starting Ollama"})
                time.sleep(min(0.1, remaining()))
    prepare_model(config, request_id, deadline)

def chat(body_json, request_id, timeout):
    body = json.loads(body_json)
    try:
        with api("/api/chat", body, timeout) as response:
            if body["stream"]:
                for line in response:
                    if line.strip():
                        emit(request_id, "response", json.loads(line))
            else:
                emit(request_id, "response", json.load(response))
    except urllib.error.HTTPError as error:
        raw = error.read(512).decode("utf-8", errors="replace")
        try:
            raw = json.loads(raw).get("error", "Ollama rejected the request")
        except (ValueError, AttributeError):
            raw = "Ollama rejected the request"
        emit(request_id, "error", {"status": error.code, "message": str(raw)[:240]})
'''


@dataclass
class _Runtime:
    account: Account = field(repr=False)
    settings: _Settings
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    active: int = 0  # includes requests waiting for this VM's execution lock
    client: Optional[httpx.AsyncClient] = field(default=None, repr=False)
    socket: Optional[_KernelSocket] = field(default=None, repr=False)
    timer: Optional[asyncio.TimerHandle] = field(default=None, repr=False)
    endpoint: str = ""
    url: str = ""
    token: str = field(default="", repr=False)
    token_expires_at: float = 0
    notebook_hash: str = ""
    kernel_id: str = ""
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = 0
    last_used_at: float = 0
    last_checked_at: float = 0
    hardware: tuple[str, bool] = ("", False)
    prepared: Dict[tuple[str, int, str, str], Dict[str, Any]] = field(default_factory=dict)
    bootstrapped: bool = False
    remote_loaded: bool = False
    bootstrap_signature: tuple[str, str] = ("", "")

    def saved(self) -> Dict[str, Any]:
        return {"owner": "llmproxy-colab", "endpoint": self.endpoint, "url": self.url,
                "token": self.token, "token_expires_at": self.token_expires_at,
                "notebook_hash": self.notebook_hash, "kernel_id": self.kernel_id,
                "session_id": self.session_id, "created_at": self.created_at,
                "last_used_at": self.last_used_at, "gpu": self.hardware[0],
                "high_mem": self.hardware[1]}


class ColabPlugin(BasePlugin):
    def __init__(self):
        self._runtimes: Dict[str, _Runtime] = {}
        self._pool_lock = asyncio.Lock()
        self._refresh_locks: Dict[str, asyncio.Lock] = {}
        self._flows: Dict[str, Dict[str, Any]] = {}
        self._cleanup_tasks: set[asyncio.Task] = set()
        self._closed = False
        self._metadata = PluginMetadata(
            id="colab", name="Colab · Ollama",
            description="Run Ollama on reusable Google Colab VMs with Google sign-in and configurable account splitting.",
            version="2.0.0", author="LLMProxy", filename="colab.py",
            oauth_login=True, oauth_local_import=True, supports_usage_meter=True,
            oauth_hint="Sign in like the official Colab CLI: approve Colab/email access, then paste Google's authorization code. Local import accepts user ADC or legacy CLI JSON credentials.",
            variables=[
                PluginVariable(name="split_across_accounts", label="Split requests across accounts", type="boolean", default=False,
                               description="Off: keep the priority account warm and use others only on failure. On: distribute independent requests using the account strategy; each account has its own VM."),
                PluginVariable(name="load_balancing_strategy", label="Account strategy", type="select", default="round_robin",
                               options=["round_robin", "least_used", "fallback", "random"],
                               description="Used when account splitting is enabled. Fallback minimizes cold VM starts."),
                PluginVariable(name="gpu", label="VM hardware", type="select", default="T4", options=_GPU_CHOICES,
                               description="No silent fallback to a more expensive accelerator. Availability and billing depend on your Colab plan. Ollama does not support TPU."),
                PluginVariable(name="high_mem", label="High-RAM VM", type="boolean", default=False,
                               description="Request shape=hm when supported; requires the corresponding Colab entitlement. L4 has only one shape."),
                PluginVariable(name="vm_idle_seconds", label="Release VM after idle (s)", type="number", default=900,
                               description="0: release after each request; -1: no plugin idle limit. Actual Colab idle limits still apply."),
                PluginVariable(name="vm_max_lifetime_seconds", label="Maximum VM lifetime (s)", type="number", default=3600,
                               description="Release/rotate after this age once work completes. 0 disables the plugin lifetime limit; it cannot extend Colab's limit."),
                PluginVariable(name="max_active_vms", label="Maximum active VMs", type="number", default=0,
                               description="0: one per used account, no extra cap. A positive limit prevents additional allocations; it never stops a busy VM."),
                PluginVariable(name="model_keep_alive_seconds", label="Keep model loaded (s)", type="number", default=-1,
                               description="-1: retain in VRAM for the VM lifetime; 0: unload after each request. This is not a VM keep-alive ping."),
                PluginVariable(name="preload_model", label="Preload model before serving", type="boolean", default=True,
                               description="Download only missing weights and load the requested model once, avoiding a second cold start at first inference."),
                PluginVariable(name="startup_timeout", label="VM/model startup budget (s)", type="number", default=900,
                               description="Total allocation, kernel connection, Ollama installation, model download and preload deadline."),
                PluginVariable(name="timeout", label="Inference/queue timeout (s)", type="number", default=300,
                               description="Per-request inference deadline; queueing uses the same bound. Startup has a separate budget."),
                PluginVariable(name="request_timeout", label="Control API timeout (s)", type="number", default=30),
                PluginVariable(name="assign_retries", label="Allocation recovery retries", type="number", default=2,
                               description="Recover uncertain 502/503/504/timeouts using the same notebook hash before resubmitting. 0 disables retries."),
                PluginVariable(name="assign_retry_backoff", label="Allocation retry backoff (s)", type="number", default=0.5,
                               description="Exponential backoff only after transient allocation failures; no fixed first-start delay."),
                PluginVariable(name="health_check_seconds", label="Runtime health-check interval (s)", type="number", default=30,
                               description="Reuse hot connections without a control round trip until this interval passes. 0 checks each request; no idle pings."),
                PluginVariable(name="ollama_version", label="Ollama release", type="string", default="latest",
                               description="latest or a release version. Existing installations are reused, not upgraded on each request."),
                PluginVariable(name="models_directory", label="Model cache directory on VM", type="string", default="/content/.llmproxy-colab/models",
                               description="May point to storage you have already mounted. The plugin does not auto-mount Drive. Colab ephemeral disk is lost on VM release."),
                PluginVariable(name="release_vms_on_shutdown", label="Release VMs on shutdown", type="boolean", default=True,
                               description="Graceful shutdown/deletion releases owned VMs. Disable to reconnect after a proxy restart. Hot reload always preserves VMs."),
                PluginVariable(name="base_url", label="Colab backend", type="string", default=_COLAB_URL),
                PluginVariable(name="reasoning_effort", label="Default reasoning effort", type="string", default="default",
                               description="default uses the model default. Qwen3 uses none/high as off/on; other models use their /api/show controls."),
                PluginVariable(name="account_switch_cooldown", label="Account quota cooldown (s)", type="number", default=60),
                PluginVariable(name="account_switch_wait", label="Account switch delay (s)", type="number", default=0),
                PluginVariable(name="proxy_enabled", label="Enable external proxy", type="boolean", default=False),
                PluginVariable(name="proxy_strategy", label="Proxy strategy", type="select", default="sequential",
                               options=["sequential", "random", "specific", "switch_on_limit"]),
                PluginVariable(name="specific_proxy_id", label="Specific proxy ID / tag", type="string", default=""),
                PluginVariable(name="proxy", label="Direct proxy URL", type="string", default=""),
            ],
            models=[PluginModelConfig(
                id="cl/qwen3:0.6b", name="Qwen3 0.6B · Colab (fast startup)", upstream_model="qwen3:0.6b",
                context_window=8192, max_tokens=4096, reasoning_levels=["none", "high"], supports_vision=False,
            )],
        )

    @property
    def metadata(self) -> PluginMetadata:
        return self._metadata

    async def astart(self, variables: Dict[str, Any], accounts: List[Account]) -> None:
        # No network/auth/allocation here. Re-arm saved ownership timers even
        # when no completion arrives after a proxy restart or hot reload.
        async with self._pool_lock:
            for account in accounts:
                saved = account.extra_config.get(_STATE_KEY, {})
                if not isinstance(saved, dict) or saved.get("owner") != "llmproxy-colab":
                    continue
                if account.id in self._runtimes:
                    continue
                try:
                    runtime = self._restore_runtime(account, _Settings.read(variables, account))
                except (TypeError, ValueError, ColabError):
                    logger.warning("Invalid saved Colab ownership/settings for account %s; manual recovery required", account.id)
                    continue
                self._runtimes[account.id] = runtime
                self._schedule_expiry(runtime)

    def validate_variables(self, variables: Dict[str, Any]) -> List[str]:
        errors = super().validate_variables(variables)
        try:
            _Settings.read(variables)
            _boolean(variables.get("split_across_accounts", False), "split_across_accounts")
            _number(variables, "account_switch_cooldown", 60, 0, 86400)
            _number(variables, "account_switch_wait", 0, 0, 600)
        except ValueError as exc:
            errors.append(str(exc))
        return errors

    @staticmethod
    def _persist(account: Account) -> None:
        # OAuth and runtime-proxy credentials are durable account data. Import
        # the manager only when handling a request/token rotation, never on load.
        from llmproxy.core.manager import plugin_manager
        if plugin_manager.accounts.get(account.id) is account:
            plugin_manager._save_accounts()

    def _save_runtime(self, runtime: _Runtime) -> None:
        runtime.account.extra_config[_STATE_KEY] = runtime.saved()
        self._persist(runtime.account)

    @staticmethod
    def _check_scopes(account: Account) -> None:
        scopes = account.extra_config.get("scopes")
        if scopes is None:
            return  # Manually supplied bearer token: backend validates its scopes.
        if isinstance(scopes, str):
            scopes = scopes.split()
        required = {_SCOPES[1], _SCOPES[3]}
        granted = set(scopes)
        if "email" in granted:
            granted.add(_SCOPES[1])
        if not required <= granted:
            raise ColabError("Google credential lacks Colab/email scopes; sign in again (old Gemini credentials cannot allocate VMs)", 401)

    @staticmethod
    def _client(runtime: _Runtime) -> httpx.AsyncClient:
        if runtime.client is None or runtime.client.is_closed:
            runtime.client = httpx.AsyncClient(timeout=runtime.settings.request_timeout,
                                               proxy=runtime.settings.proxy, follow_redirects=False)
        return runtime.client

    async def _control(self, runtime: _Runtime, method: str, path: str, **kwargs: Any) -> Dict[str, Any]:
        await self._refresh_token(runtime.account, runtime.settings.proxy)
        params = {"authuser": "0", **kwargs.pop("params", {})}
        headers = {"Authorization": f"Bearer {runtime.account.api_key}", "Accept": "application/json",
                   "X-Colab-Client-Agent": "colab-cli", **kwargs.pop("headers", {})}
        try:
            response = await self._client(runtime).request(method, runtime.settings.base_url + path,
                                                            params=params, headers=headers, **kwargs)
        except httpx.TimeoutException:
            raise  # allocation recovery needs to distinguish uncertain timeouts
        except httpx.HTTPError:
            raise ColabError("Colab control connection failed", 502) from None
        capture_rate_limit_headers(runtime.account, response.headers)
        _check_http(response, (runtime.account.api_key, runtime.token,
                               str(runtime.account.extra_config.get("refresh_token", "")),
                               headers.get("X-Goog-Colab-Token", "")))
        return _json_response(response)

    @staticmethod
    def _runtime_headers(runtime: _Runtime) -> Dict[str, str]:
        # Header auth avoids leaking the proxy token through URL/access logs.
        return {"X-Colab-Runtime-Proxy-Token": runtime.token, "X-Colab-Client-Agent": "colab-cli"}

    async def _runtime_request(self, runtime: _Runtime, method: str, path: str,
                               **kwargs: Any) -> httpx.Response:
        try:
            response = await self._client(runtime).request(method, runtime.url + path,
                                                            headers=self._runtime_headers(runtime), **kwargs)
        except httpx.TimeoutException:
            raise ColabError("Runtime HTTP connection timed out", 504) from None
        except httpx.HTTPError:
            raise ColabError("Runtime HTTP connection failed", 502) from None
        _check_http(response, (runtime.token, runtime.account.api_key))
        return response

    async def _reset_kernel_connection(self, runtime: _Runtime) -> None:
        if runtime.socket:
            await runtime.socket.aclose()
        runtime.socket = None
        runtime.kernel_id = ""
        runtime.remote_loaded = runtime.bootstrapped = False
        runtime.prepared.clear()
        self._save_runtime(runtime)

    def _apply_assignment(self, runtime: _Runtime, assignment: Dict[str, Any]) -> None:
        info = assignment.get("runtimeProxyInfo", {})
        endpoint = str(assignment.get("endpoint", ""))
        if not re.fullmatch(r"[\w.-]+", endpoint) or not info.get("token") or not info.get("url"):
            raise ColabError("Colab assignment omitted its runtime connection", 502)
        runtime.endpoint = endpoint
        runtime.url = _runtime_url(str(info["url"]), runtime.settings.base_url)
        runtime.token = str(info["token"])
        runtime.token_expires_at = time.time() + float(info.get("tokenExpiresInSeconds", 3600))
        runtime.last_checked_at = time.time()
        self._save_runtime(runtime)

    async def _allocate(self, runtime: _Runtime) -> None:
        if not runtime.notebook_hash:
            # Exactly the official CLI transformation: UUID hyphens become
            # underscores, padded with dots to 44 characters (not base64(uuid)).
            runtime.notebook_hash = str(uuid.uuid4()).replace("-", "_") + "." * 8
            runtime.created_at = runtime.last_used_at = time.time()
            runtime.hardware = (runtime.settings.gpu, runtime.settings.high_mem)
            # Persist before an uncertain POST so a process restart can recover
            # this very assignment rather than leaking/allocating another one.
            self._save_runtime(runtime)
        params = {"nbh": runtime.notebook_hash,
                  "variant": "DEFAULT" if runtime.settings.gpu == "CPU" else "GPU",
                  "accelerator": "NONE" if runtime.settings.gpu == "CPU" else runtime.settings.gpu}
        if runtime.settings.high_mem:
            params["shape"] = "hm"
        for attempt in range(runtime.settings.assign_retries + 1):
            assignment = await self._control(runtime, "GET", "/tun/m/assign", params=params)
            if assignment.get("endpoint"):
                self._apply_assignment(runtime, assignment)
                return
            xsrf = assignment.get("token")
            if not xsrf:
                raise ColabError("Assignment preflight omitted its XSRF token", 502)
            try:
                assigned = await self._control(runtime, "POST", "/tun/m/assign", params=params,
                                               headers={"X-Goog-Colab-Token": str(xsrf)})
            except (httpx.TimeoutException, ColabError) as exc:
                retryable = isinstance(exc, httpx.TimeoutException) or exc.status_code in (502, 503, 504)
                if not retryable:
                    # A definite allocation rejection did not create a VM. Do
                    # not occupy a local VM-cap slot while the router fails over.
                    await self._forget_assignment(runtime)
                if not retryable or attempt >= runtime.settings.assign_retries:
                    raise
                delay = runtime.settings.retry_backoff * (2 ** attempt)
                if delay:
                    await asyncio.sleep(delay)
                # The next iteration GET reconciles using the same notebook
                # hash and a fresh XSRF token before another allocation POST.
                continue
            self._apply_assignment(runtime, assigned)
            return

    async def _sync_assignment(self, runtime: _Runtime) -> bool:
        result = await self._control(runtime, "GET", "/tun/m/assignments")
        for assignment in result.get("assignments", []):
            if assignment.get("endpoint") == runtime.endpoint:
                old_token, old_url = runtime.token, runtime.url
                self._apply_assignment(runtime, assignment)
                if (old_token != runtime.token or old_url != runtime.url) and runtime.socket:
                    await runtime.socket.aclose()
                    runtime.socket = None
                return True
        await self._forget_assignment(runtime)
        return False

    async def _forget_assignment(self, runtime: _Runtime) -> None:
        if runtime.socket:
            await runtime.socket.aclose()
            runtime.socket = None
        runtime.endpoint = runtime.url = runtime.token = runtime.kernel_id = runtime.notebook_hash = ""
        runtime.token_expires_at = runtime.created_at = 0
        runtime.bootstrapped = False
        runtime.remote_loaded = False
        runtime.prepared.clear()
        runtime.account.extra_config.pop(_STATE_KEY, None)
        self._persist(runtime.account)

    async def _connect_kernel(self, runtime: _Runtime) -> None:
        if runtime.socket and not runtime.socket.closed:
            return
        if not runtime.kernel_id:
            response = await self._runtime_request(runtime, "POST", "/api/kernels", json={"name": "python3"})
            runtime.kernel_id = str(response.json().get("id", ""))
            if not re.fullmatch(r"[\w-]+", runtime.kernel_id):
                raise ColabError("Jupyter did not return a kernel ID", 502)
            self._save_runtime(runtime)
        elif not re.fullmatch(r"[\w-]+", runtime.kernel_id):
            raise ValueError("Invalid saved Colab kernel ID")
        url = runtime.url + "/api/kernels/" + quote(runtime.kernel_id, safe="") + "/channels?" + urlencode({"session_id": runtime.session_id})
        try:
            runtime.socket = await _KernelSocket.connect(self._client(runtime), url,
                                                         self._runtime_headers(runtime), runtime.settings.request_timeout)
        except ColabError as exc:
            if exc.status_code == 404:
                await self._reset_kernel_connection(runtime)
                raise _KernelUnavailable("Saved kernel/tunnel no longer exists", 404) from None
            raise

    async def _interrupt(self, runtime: _Runtime) -> None:
        if not runtime.kernel_id or not runtime.endpoint:
            return
        try:
            await self._runtime_request(runtime, "POST", "/api/kernels/" + quote(runtime.kernel_id, safe="") + "/interrupt",
                                        timeout=min(3, runtime.settings.request_timeout))
        except Exception:
            # Connection may already be gone. Never let cleanup hide the error
            # or log an exception containing a credential-bearing response.
            pass

    async def _execute(self, runtime: _Runtime, code: str, request_id: str,
                       timeout: float) -> AsyncIterator[Dict[str, Any]]:
        await self._connect_kernel(runtime)
        socket = runtime.socket
        assert socket is not None
        # Installing binaries/pulling weights or a long thinking phase can be
        # silent much longer than the *control API* timeout. The execution's
        # outer deadline bounds this read; don't abort a valid cold start at 30s.
        socket.timeout = timeout
        message = {"header": {"msg_id": request_id, "username": "llmproxy", "session": runtime.session_id,
                              "date": datetime.now(timezone.utc).isoformat(), "msg_type": "execute_request", "version": "5.3"},
                   "parent_header": {}, "metadata": {}, "channel": "shell", "buffers": [],
                   "content": {"code": code, "silent": False, "store_history": False,
                               "user_expressions": {}, "allow_stdin": False, "stop_on_error": True}}
        pending = ""
        idle = replied = False
        deadline = time.monotonic() + timeout
        async def wait(operation):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                operation.close()
                raise TimeoutError
            return await asyncio.wait_for(operation, remaining)
        try:
            # A generator primed by the router resumes in a different SSE task.
            # Don't keep an asyncio.timeout context open across yields: its
            # cancellation would target the original task, not the consumer.
            await wait(socket.send_json(message))
            while not (idle and replied):
                incoming = await wait(socket.receive_json())
                kind = incoming.get("header", {}).get("msg_type", incoming.get("msg_type"))
                data = incoming.get("content", {})
                if kind == "status" and data.get("execution_state") in ("dead", "restarting"):
                    raise _KernelUnavailable("Kernel stopped before execution finished", 502)
                if incoming.get("parent_header", {}).get("msg_id") != request_id:
                    continue
                if kind in ("error", "execute_reply") and data.get("status", "error" if kind == "error" else "ok") != "ok":
                    raise ColabError("Kernel " + _excerpt(data.get("ename", "Error")) + ": " +
                                     _excerpt(data.get("evalue", "Execution failed"), (runtime.token, runtime.account.api_key)), 502)
                if kind == "execute_reply":
                    replied = True
                elif kind == "status" and data.get("execution_state") == "idle":
                    idle = True
                elif kind == "stream" and data.get("name") == "stdout":
                    pending += str(data.get("text", ""))
                    if len(pending.encode("utf-8")) > _MAX_MESSAGE_BYTES:
                        raise ColabError("Kernel output exceeds 32 MiB", 502)
                    lines = pending.split("\n")
                    pending = lines.pop()
                    for line in lines:
                        if not line.startswith(_RESULT_PREFIX):
                            continue
                        try:
                            event = json.loads(line[len(_RESULT_PREFIX):])
                        except ValueError:
                            raise ColabError("Malformed Ollama bridge output", 502) from None
                        if event.get("id") != request_id:
                            continue
                        if event.get("type") == "error":
                            error = event.get("data", {})
                            raise ColabError(_excerpt(error.get("message")), int(error.get("status", 500)))
                        yield event
            if pending.startswith(_RESULT_PREFIX):
                raise ColabError("Truncated Ollama bridge output", 502)
        except BaseException as exc:
            await socket.aclose()
            runtime.socket = None
            await self._interrupt(runtime)
            runtime.prepared.clear()
            runtime.bootstrapped = False
            if isinstance(exc, _KernelUnavailable):
                await self._reset_kernel_connection(runtime)
            if isinstance(exc, TimeoutError):
                raise ColabError("Kernel execution exceeded its configured timeout", 504) from None
            raise

    @staticmethod
    def _model_name(model: PluginModelConfig) -> str:
        name = model.upstream_model or model.id
        if not re.fullmatch(r"[\w./:@-]+", name) or name.endswith(":cloud") or ".." in name.split("/"):
            raise ValueError("upstream_model must be a local Ollama model tag, not a cloud model or command")
        return name

    async def _load_bridge(self, runtime: _Runtime) -> None:
        if runtime.remote_loaded:
            return
        request_id = uuid.uuid4().hex
        code = ("_llmproxy_colab = globals().setdefault('_llmproxy_colab', {})\n"
                f"exec({_REMOTE_SOURCE!r}, _llmproxy_colab)")
        async with aclosing(self._execute(runtime, code, request_id, runtime.settings.request_timeout)) as events:
            async for _event in events:
                pass
        runtime.remote_loaded = True

    async def _prepare_model(self, runtime: _Runtime, model: PluginModelConfig) -> Dict[str, Any]:
        name = self._model_name(model)
        context = int(model.parameters.get("options", {}).get("num_ctx", model.context_window or 8192))
        signature = (runtime.settings.ollama_version, runtime.settings.models_directory)
        if runtime.bootstrap_signature != signature:
            runtime.bootstrapped = False
            runtime.prepared.clear()
        key = (name, context, runtime.settings.ollama_version, runtime.settings.models_directory)
        if key in runtime.prepared:
            return runtime.prepared[key]
        request_id = uuid.uuid4().hex
        config = {"model": name, "num_ctx": context, "startup_timeout": runtime.settings.startup_timeout,
                  "ollama_version": runtime.settings.ollama_version, "models_directory": runtime.settings.models_directory,
                  "model_keep_alive_seconds": runtime.settings.model_keep_alive_seconds, "preload_model": runtime.settings.preload_model}
        config_json = json.dumps(config)
        if not runtime.bootstrapped:
            code = f"_llmproxy_colab['bootstrap'](__import__('json').loads({config_json!r}), {request_id!r})"
        else:
            code = (f"_llmproxy_colab['prepare_model'](__import__('json').loads({config_json!r}), "
                    f"{request_id!r}, __import__('time').monotonic() + {runtime.settings.startup_timeout!r})")
        info = None
        async with aclosing(self._execute(runtime, code, request_id, runtime.settings.startup_timeout)) as events:
            async for event in events:
                if event.get("type") == "ready":
                    info = event.get("data", {})
        if not isinstance(info, dict):
            raise ColabError("Ollama setup completed without a readiness response", 502)
        runtime.bootstrapped = True
        runtime.bootstrap_signature = signature
        runtime.prepared[key] = info
        self._save_runtime(runtime)
        return info

    @staticmethod
    def _expired(runtime: _Runtime, settings: _Settings) -> bool:
        now = time.time()
        return bool(runtime.created_at and (
            (settings.vm_max_lifetime_seconds > 0 and now - runtime.created_at >= settings.vm_max_lifetime_seconds)
            or (settings.vm_idle_seconds >= 0 and now - runtime.last_used_at >= settings.vm_idle_seconds)
            or runtime.hardware != (settings.gpu, settings.high_mem)
        ))

    async def _ensure_runtime(self, runtime: _Runtime, model: PluginModelConfig) -> Dict[str, Any]:
        settings = runtime.settings
        if runtime.endpoint and self._expired(runtime, settings):
            await self._stop_runtime(runtime)
        reconnecting = runtime.socket is None or runtime.socket.closed
        if runtime.endpoint and (not runtime.last_checked_at or reconnecting):
            await self._sync_assignment(runtime)
        elif runtime.endpoint and (runtime.token_expires_at - time.time() <= 300):
            await self._sync_assignment(runtime)
        if runtime.endpoint and (reconnecting or not runtime.last_checked_at or
                                time.time() - runtime.last_checked_at >= settings.health_check_seconds):
            try:
                if runtime.kernel_id:
                    await self._runtime_request(runtime, "GET", "/api/kernels/" + quote(runtime.kernel_id, safe=""))
                else:
                    await self._runtime_request(runtime, "GET", "/api/kernels")
            except ColabError as exc:
                if exc.status_code not in (401, 403, 404):
                    raise
                if await self._sync_assignment(runtime):
                    # A reclaimed kernel, not a reclaimed VM: start just a new
                    # kernel, keeping downloaded weights and the Ollama process.
                    try:
                        if runtime.kernel_id:
                            await self._runtime_request(runtime, "GET", "/api/kernels/" + quote(runtime.kernel_id, safe=""))
                    except ColabError as refreshed:
                        if refreshed.status_code != 404:
                            raise
                        await self._reset_kernel_connection(runtime)
            runtime.last_checked_at = time.time()
        if not runtime.endpoint:
            await self._allocate(runtime)
        # VMs can be assigned before their Jupyter HTTP server is listening.
        # Retry only setup, never an inference that might have produced output.
        for attempt in range(4):
            try:
                await self._load_bridge(runtime)
                return await self._prepare_model(runtime, model)
            except _KernelUnavailable:
                if attempt == 3:
                    raise
                if runtime.endpoint:
                    await self._sync_assignment(runtime)
                await self._reset_kernel_connection(runtime)
                if not runtime.endpoint:
                    await self._allocate(runtime)
                continue
            except (httpx.TransportError, ColabError) as exc:
                retryable = isinstance(exc, httpx.TransportError) or exc.status_code in (502, 503, 504)
                # After loading the bridge, failure can be a real remote setup
                # error. Never repeat downloads/commands blindly.
                if runtime.remote_loaded or not retryable or attempt == 3:
                    raise
                if runtime.socket:
                    await runtime.socket.aclose()
                    runtime.socket = None
                await asyncio.sleep(0.1 * (2 ** attempt))
        raise ColabError("Ollama startup failed", 502)

    def _restore_runtime(self, account: Account, settings: _Settings) -> _Runtime:
        runtime = _Runtime(account=account, settings=settings)
        saved = account.extra_config.get(_STATE_KEY, {})
        if isinstance(saved, dict) and saved.get("owner") == "llmproxy-colab":
            for name in ("endpoint", "url", "token", "notebook_hash", "kernel_id", "session_id"):
                if saved.get(name):
                    setattr(runtime, name, str(saved[name]))
            for name in ("token_expires_at", "created_at", "last_used_at"):
                setattr(runtime, name, float(saved.get(name, 0)))
            runtime.hardware = (str(saved.get("gpu", settings.gpu)), bool(saved.get("high_mem", settings.high_mem)))
            if runtime.endpoint:
                if not re.fullmatch(r"[\w.-]+", runtime.endpoint):
                    raise ValueError("Invalid saved Colab endpoint")
                runtime.url = _runtime_url(runtime.url, settings.base_url)
            if runtime.notebook_hash and not re.fullmatch(r"[\w_]{8}_[\w_]{4}_[\w_]{4}_[\w_]{4}_[\w_]{12}\.{8}", runtime.notebook_hash):
                raise ValueError("Invalid saved Colab notebook hash")
        return runtime

    @asynccontextmanager
    async def _lease(self, account: Account, settings: _Settings) -> AsyncIterator[_Runtime]:
        async with self._pool_lock:
            if self._closed:
                raise ColabError("Plugin is closing; reload it before sending requests", 503)
            runtime = self._runtimes.get(account.id)
            if runtime is None:
                if settings.max_active_vms and len(self._runtimes) >= settings.max_active_vms:
                    raise ColabError("Configured max_active_vms reached; wait for an idle VM to release or increase the limit", 503)
                runtime = self._restore_runtime(account, settings)
                self._runtimes[account.id] = runtime
            runtime.active += 1
            if runtime.timer:
                runtime.timer.cancel()
                runtime.timer = None
        acquired = False
        try:
            async with asyncio.timeout(settings.timeout):
                await runtime.lock.acquire()
            acquired = True
            if self._closed:
                raise ColabError("Plugin was reloaded while this request was queued; retry on the replacement", 503)
            if runtime.settings.proxy != settings.proxy and runtime.client:
                if runtime.socket:
                    await runtime.socket.aclose()
                    runtime.socket = None
                await runtime.client.aclose()
                runtime.client = None
            runtime.account = account
            runtime.settings = settings
            yield runtime
        finally:
            if acquired:
                runtime.last_used_at = time.time()
                saved = account.extra_config.get(_STATE_KEY)
                if isinstance(saved, dict):
                    saved["last_used_at"] = runtime.last_used_at
                # The router's account-result write persists this timestamp;
                # avoid an extra disk fsync on every warm completion.
                runtime.lock.release()
            runtime.active -= 1
            if not runtime.active:
                if settings.vm_idle_seconds == 0 and runtime.endpoint:
                    await self._expire(runtime)
                elif not runtime.endpoint and not runtime.notebook_hash:
                    if runtime.client:
                        await runtime.client.aclose()
                    self._runtimes.pop(account.id, None)
                else:
                    self._schedule_expiry(runtime)

    def _schedule_expiry(self, runtime: _Runtime, retry_after: Optional[float] = None) -> None:
        if self._closed or runtime.active or not (runtime.endpoint or runtime.notebook_hash):
            return
        if runtime.timer:
            runtime.timer.cancel()
        now = time.time()
        deadlines = []
        if runtime.settings.vm_idle_seconds >= 0:
            deadlines.append(runtime.last_used_at + runtime.settings.vm_idle_seconds)
        if runtime.settings.vm_max_lifetime_seconds > 0:
            deadlines.append(runtime.created_at + runtime.settings.vm_max_lifetime_seconds)
        if retry_after is not None:
            delay = retry_after
        elif deadlines:
            delay = max(0, min(deadlines) - now)
        else:
            return
        def expire() -> None:
            runtime.timer = None
            task = asyncio.create_task(self._expire(runtime))
            self._cleanup_tasks.add(task)
            def finished(completed: asyncio.Task) -> None:
                self._cleanup_tasks.discard(completed)
                if not completed.cancelled() and completed.exception() is not None:
                    logger.warning("Colab cleanup failed for account %s; owned endpoint remains recoverable", runtime.account.id)
            task.add_done_callback(finished)
        runtime.timer = asyncio.get_running_loop().call_later(delay, expire)

    async def _stop_runtime(self, runtime: _Runtime) -> None:
        if runtime.socket:
            await runtime.socket.aclose()
            runtime.socket = None
        if not runtime.endpoint and runtime.notebook_hash:
            # GET reconciles an uncertain POST without creating a fresh VM just
            # to clean up a failed startup.
            params = {"nbh": runtime.notebook_hash, "variant": "DEFAULT" if runtime.hardware[0] == "CPU" else "GPU",
                      "accelerator": "NONE" if runtime.hardware[0] == "CPU" else runtime.hardware[0]}
            if runtime.hardware[1]:
                params["shape"] = "hm"
            recovered = await self._control(runtime, "GET", "/tun/m/assign", params=params)
            if recovered.get("endpoint"):
                self._apply_assignment(runtime, recovered)
        if runtime.endpoint:
            endpoint = quote(runtime.endpoint, safe="")
            try:
                preflight = await self._control(runtime, "GET", "/tun/m/unassign/" + endpoint)
                token = preflight.get("token")
                if not token:
                    raise ColabError("Release preflight omitted its XSRF token", 502)
                await self._control(runtime, "POST", "/tun/m/unassign/" + endpoint,
                                    headers={"X-Goog-Colab-Token": str(token)})
            except ColabError as exc:
                if exc.status_code != 404:
                    raise
        await self._forget_assignment(runtime)

    async def _expire(self, runtime: _Runtime) -> None:
        if runtime.active:
            return
        async with runtime.lock:
            if self._closed or runtime.active or self._runtimes.get(runtime.account.id) is not runtime:
                return
            try:
                await self._stop_runtime(runtime)
            except Exception as exc:
                # Keep the owned endpoint durable and retry release later. Never
                # lose a billable assignment merely because cleanup failed.
                logger.warning("Colab VM release failed for account %s (%s); retrying", runtime.account.id, type(exc).__name__)
                if runtime.client:
                    await runtime.client.aclose()
                    runtime.client = None
                self._schedule_expiry(runtime, retry_after=30)
                return
            if runtime.client:
                await runtime.client.aclose()
                runtime.client = None
            if not runtime.active:
                self._runtimes.pop(runtime.account.id, None)

    async def aclose(self, *, release_resources: bool = True) -> None:
        """Close connections; hot reload preserves owned VMs for the next instance."""
        self._closed = True
        for runtime in list(self._runtimes.values()):
            if runtime.timer:
                runtime.timer.cancel()
                runtime.timer = None
        if self._cleanup_tasks:
            await asyncio.gather(*self._cleanup_tasks, return_exceptions=True)
        async def close(runtime: _Runtime) -> None:
            async with runtime.lock:
                try:
                    if runtime.endpoint or runtime.notebook_hash:
                        self._save_runtime(runtime)  # retain latest idle time for reconnect
                    if release_resources and runtime.settings.release_vms_on_shutdown:
                        await self._stop_runtime(runtime)
                finally:
                    if runtime.socket:
                        await runtime.socket.aclose()
                        runtime.socket = None
                    if runtime.client:
                        await runtime.client.aclose()
                        runtime.client = None
        results = await asyncio.gather(*(close(runtime) for runtime in self._runtimes.values()), return_exceptions=True)
        self._runtimes.clear()
        self._flows.clear()
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            raise ColabError("Could not release every owned VM; saved endpoints remain available for recovery", 502)

    # ----------------------------------------------------------- Ollama wire
    @staticmethod
    def _validate_request(request: ChatCompletionRequest, model: PluginModelConfig) -> None:
        if request.n not in (None, 1):
            raise ValueError("Ollama supports one choice per request (n=1)")
        if request.tool_choice not in (None, "auto", "none") or request.function_call not in (None, "auto", "none"):
            raise ValueError("Ollama supports automatic tools or tool_choice=none, not forced/named tool choice")
        if request.parallel_tool_calls is False and (request.tools or request.functions):
            raise ValueError("Ollama cannot enforce parallel_tool_calls=false; omit it for automatic tool calls")
        options = model.parameters.get("options", {})
        if not isinstance(options, dict):
            raise ValueError("Model parameters.options must be an Ollama options object")
        _number(options, "num_ctx", model.context_window or 8192, 1, 1048576, integer=True)

    @staticmethod
    def _reasoning(request: ChatCompletionRequest, variables: Dict[str, Any],
                   model: PluginModelConfig, info: Dict[str, Any]) -> Any:
        extra = request.extra_body or {}
        native = getattr(request, "think", extra.get("think"))
        if native is not None:
            if not isinstance(native, (str, bool)):
                raise ValueError("Ollama think must be a boolean or native level")
            return native
        level = request.reasoning_effort or extra.get("reasoning_effort")
        if level is None:
            reasoning = extra.get("reasoning", getattr(request, "reasoning", None))
            level = reasoning.get("effort") if isinstance(reasoning, dict) else reasoning
        level = str(level or variables.get("reasoning_effort", "default")).lower()
        if level in ("", "default", "auto"):
            return None
        if level in ("off", "disabled"):
            level = "none"
        if model.reasoning_levels and level not in model.reasoning_levels:
            raise ValueError(f"{model.id} accepts reasoning levels: {', '.join(model.reasoning_levels)}")
        thinking = info.get("thinking")
        values = thinking.get("values", []) if isinstance(thinking, dict) else []
        if values:
            if level == "none" and False in values:
                return False
            if level in values:
                return level
            if True in values and not any(isinstance(value, str) for value in values):
                if level == "none":
                    raise ValueError("This Ollama model cannot disable thinking")
                return True
            raise ValueError("Requested reasoning level is not supported by this Ollama model")
        if "thinking" not in info.get("capabilities", []) and not model.reasoning_levels:
            if level == "none":
                return None
            raise ValueError("This Ollama model does not support thinking")
        name = model.upstream_model or model.id
        if name.split(":", 1)[0].split("/")[-1] == "gpt-oss":
            if level not in ("low", "medium", "high"):
                raise ValueError("GPT-OSS accepts low, medium and high; thinking cannot be disabled")
            return level
        # Qwen3 and other boolean-thinking models use the native on/off switch.
        return level != "none"

    async def _build_payload(self, request: ChatCompletionRequest, model: PluginModelConfig,
                             variables: Dict[str, Any], settings: _Settings,
                             info: Dict[str, Any]) -> Dict[str, Any]:
        extra = request.extra_body if isinstance(request.extra_body, dict) else {}
        self._validate_request(request, model)
        messages = await materialize_message_images(request.messages, timeout=settings.request_timeout,
                                                     proxy=settings.proxy)
        translated = []
        tool_names = {call.get("id"): call.get("function", {}).get("name")
                      for message in request.messages for call in (message.tool_calls or [])}
        for message in messages:
            content = message.content
            result: Dict[str, Any] = {"role": "tool" if message.role == "function" else message.role,
                                      "content": content_text(content)}
            if isinstance(content, list):
                images = []
                for part in content:
                    if not is_image_part(part):
                        continue
                    normalized = to_openai_image_part(part)
                    image = parse_data_url(normalized["image_url"]) if normalized else None
                    if not image:
                        raise ValueError("Ollama requires inline image data")
                    images.append(image[1])
                if images:
                    if "vision" not in info.get("capabilities", []) and not model.supports_vision:
                        raise ValueError("Selected Ollama model does not support images")
                    result["images"] = images
            if message.tool_calls:
                calls = []
                for index, call in enumerate(message.tool_calls):
                    function = dict(call.get("function", {}))
                    arguments = function.get("arguments", {})
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except ValueError:
                            raise ValueError("Tool call arguments must contain a JSON object") from None
                    if not isinstance(arguments, dict):
                        raise ValueError("Tool call arguments must be a JSON object")
                    function.update(index=index, arguments=arguments)
                    calls.append({"id": call.get("id", ""), "function": function})
                result["tool_calls"] = calls
            if message.role in ("tool", "function"):
                result["tool_call_id"] = message.tool_call_id or ""
                result["tool_name"] = message.name or tool_names.get(message.tool_call_id) or ""
            reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
            if reasoning:
                result["thinking"] = reasoning
            # Preserve provider cache extensions even though current Ollama's
            # cache is implicit. Never pretend these create a durable cache.
            if getattr(message, "cache_control", None) is not None:
                result["cache_control"] = message.cache_control
            if isinstance(content, list) and any(isinstance(part, dict) and part.get("cache_control") for part in content):
                result["content_parts"] = content
            translated.append(result)

        options = dict(model.parameters.get("options", {}))
        for key in ("temperature", "top_p", "presence_penalty", "frequency_penalty"):
            value = getattr(request, key)
            if value is not None:
                options[key] = value
        options.setdefault("num_ctx", model.context_window or 8192)
        options["num_predict"] = (request.max_tokens if request.max_tokens is not None else
                                  request.max_completion_tokens if request.max_completion_tokens is not None else model.max_tokens or 4096)
        if request.stop is not None:
            options["stop"] = [request.stop] if isinstance(request.stop, str) else request.stop
        seed = getattr(request, "seed", extra.get("seed"))
        if seed is not None:
            options["seed"] = seed
        native_options = extra.get("options", getattr(request, "options", {}))
        if not isinstance(native_options, dict):
            raise ValueError("Ollama options must be an object")
        options.update(native_options)
        payload: Dict[str, Any] = {
            "model": self._model_name(model), "messages": translated,
            "stream": bool(request.stream), "options": options,
            "keep_alive": getattr(request, "keep_alive", extra.get("keep_alive", settings.model_keep_alive_seconds)),
        }
        tools = request.tools or ([{"type": "function", "function": function} for function in request.functions]
                                  if request.functions else None)
        if tools and request.tool_choice != "none" and request.function_call != "none":
            if info.get("capabilities") and "tools" not in info["capabilities"]:
                raise ValueError("Selected Ollama model does not support tools")
            payload["tools"] = tools
        if request.response_format:
            kind = request.response_format.get("type")
            if kind == "json_object":
                payload["format"] = "json"
            elif kind == "json_schema":
                payload["format"] = request.response_format.get("json_schema", {}).get("schema", {})
            elif kind != "text":
                raise ValueError("Unsupported Ollama response_format")
        think = self._reasoning(request, variables, model, info)
        if think is not None:
            payload["think"] = think
        for key in ("format", "truncate", "shift", "logprobs", "top_logprobs", "runner",
                    "prompt_cache_key", "prompt_cache_retention", "cache_control", "cachedContent",
                    "cached_content", "cache", "session_id", "affinity"):
            value = getattr(request, key, extra.get(key))
            if value is not None:
                payload[key] = value
        return payload

    @staticmethod
    def _usage(data: Dict[str, Any]) -> Optional[CompletionUsage]:
        # /api/chat always reports usage in its terminal event without a stream
        # option, including when the downstream client didn't request usage.
        if "prompt_eval_count" not in data and "eval_count" not in data:
            return None
        prompt = max(0, int(data.get("prompt_eval_count", 0)))
        completion = max(0, int(data.get("eval_count", 0)))
        details = None
        if data.get("prompt_eval_cached_count") is not None:
            details = {"cached_tokens": max(0, int(data["prompt_eval_cached_count"]))}
        # Future Ollama versions/gateways may add separate reasoning counters.
        completion_details = data.get("completion_tokens_details")
        extra = {"completion_tokens_details": completion_details} if isinstance(completion_details, dict) else {}
        return CompletionUsage(prompt_tokens=prompt, completion_tokens=completion,
                               total_tokens=prompt + completion, prompt_tokens_details=details, **extra)

    @staticmethod
    def _tool_calls(calls: List[Dict[str, Any]], *, streaming: bool = False) -> List[Dict[str, Any]]:
        result = []
        for index, call in enumerate(calls):
            function = call.get("function", {})
            arguments = function.get("arguments", {})
            translated = {"id": call.get("id") or "call_" + uuid.uuid4().hex[:12], "type": "function",
                          "function": {"name": function.get("name", ""),
                                       "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments)}}
            if streaming:
                translated["index"] = int(function.get("index", index))
            result.append(translated)
        return result

    async def _ollama_events(self, runtime: _Runtime, payload: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        request_id = uuid.uuid4().hex
        body_json = json.dumps(payload, ensure_ascii=False)
        code = f"_llmproxy_colab['chat']({body_json!r}, {request_id!r}, {runtime.settings.timeout!r})"
        async with aclosing(self._execute(runtime, code, request_id, runtime.settings.timeout)) as events:
            async for event in events:
                if event.get("type") != "response":
                    continue
                data = event.get("data")
                if not isinstance(data, dict):
                    raise ColabError("Ollama returned an invalid response", 502)
                if data.get("error"):
                    raise ColabError(_excerpt(data["error"]), 500)
                yield data

    async def _stream(self, request: ChatCompletionRequest, model: PluginModelConfig,
                       variables: Dict[str, Any], account: Account,
                       settings: _Settings) -> AsyncIterator[ChatCompletionChunk]:
        async with self._lease(account, settings) as runtime:
            info = await self._start(runtime, model)
            payload = await self._build_payload(request, model, variables, settings, info)
            chunk_id = "chatcmpl-colab-" + uuid.uuid4().hex[:12]
            created = int(time.time())
            yield ChatCompletionChunk(id=chunk_id, created=created, model=request.model,
                                      choices=[ChatChoiceDelta(index=0, delta={"role": "assistant", "content": ""})])
            terminal = None
            tools_sent = False
            iterator = self._ollama_events(runtime, payload)
            try:
                async for data in iterator:
                    message = data.get("message", {})
                    delta = {}
                    if message.get("content"):
                        delta["content"] = message["content"]
                    if message.get("thinking"):
                        delta["reasoning_content"] = message["thinking"]
                    if message.get("tool_calls"):
                        delta["tool_calls"] = self._tool_calls(message["tool_calls"], streaming=True)
                        tools_sent = True
                    if delta:
                        yield ChatCompletionChunk(id=chunk_id, created=created, model=request.model,
                                                  choices=[ChatChoiceDelta(index=0, delta=delta)])
                    if data.get("done"):
                        terminal = data
                if terminal is None:
                    raise ColabError("Ollama stream ended without a terminal response", 502)
                reason = terminal.get("done_reason") or "stop"
                if reason == "stop" and tools_sent:
                    reason = "tool_calls"
                yield ChatCompletionChunk(id=chunk_id, created=created, model=request.model,
                                          choices=[ChatChoiceDelta(index=0, delta={}, finish_reason=reason)],
                                          usage=self._usage(terminal))
            finally:
                await iterator.aclose()

    async def _start(self, runtime: _Runtime, model: PluginModelConfig) -> Dict[str, Any]:
        try:
            async with asyncio.timeout(runtime.settings.startup_timeout):
                return await self._ensure_runtime(runtime, model)
        except TimeoutError:
            raise ColabError("VM/Ollama startup exceeded startup_timeout; existing allocation is retained for recovery", 504) from None
        except httpx.TimeoutException:
            raise ColabError("Colab allocation timed out; saved notebook hash is retained for recovery", 504) from None

    def _simulation(self, request: ChatCompletionRequest, model: PluginModelConfig,
                    account: Optional[Account]) -> ChatCompletionResponse:
        content = (f"Simulated Colab/Ollama response ({model.id}); no VM was allocated.\n\n"
                   "**Execution Details:**\n"
                   f"Timestamp: {datetime.now(timezone.utc).isoformat()}\nModel: {model.id}\n"
                   "Tokens: 10 prompt, 20 completion (simulation)\n"
                   f"Account status: {'Explicit test account' if account and account.api_key else 'No account configured (Simulation Mode)'}")
        calls = None
        if request.tools and not any(message.role in ("tool", "function") for message in request.messages[-2:]):
            calls = [{"id": "call_colab_simulation", "type": "function",
                      "function": {"name": request.tools[0].get("function", {}).get("name", "test_tool"), "arguments": "{}"}}]
        return ChatCompletionResponse(id="chatcmpl-colab-" + uuid.uuid4().hex[:12], model=request.model,
                                      choices=[ChatChoice(index=0, message=ChatMessage(role="assistant", content=None if calls else content,
                                                                                     tool_calls=calls), finish_reason="tool_calls" if calls else "stop")],
                                      usage=CompletionUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30))

    async def _simulation_stream(self, response: ChatCompletionResponse) -> AsyncIterator[ChatCompletionChunk]:
        message = response.choices[0].message
        yield ChatCompletionChunk(id=response.id, model=response.model, choices=[ChatChoiceDelta(index=0, delta={"role": "assistant", "content": ""})])
        delta = {"content": message.content}
        if message.tool_calls:
            delta = {"tool_calls": [{**call, "index": index} for index, call in enumerate(message.tool_calls)]}
        yield ChatCompletionChunk(id=response.id, model=response.model, choices=[ChatChoiceDelta(index=0, delta=delta)])
        yield ChatCompletionChunk(id=response.id, model=response.model,
                                  choices=[ChatChoiceDelta(index=0, delta={}, finish_reason=response.choices[0].finish_reason)], usage=response.usage)

    async def chat_completion(self, request: ChatCompletionRequest, model_config: PluginModelConfig,
                              variables: Dict[str, Any], account: Optional[Account] = None
                              ) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionChunk]]:
        settings = _Settings.read(variables, account)
        key = (account.api_key if account else "").strip()
        if account and not account.is_available():
            raise ColabError("Selected account is disabled or in quota cooldown", 429 if account.enabled else 503)
        synthetic = account is None or account.id.startswith("default-")
        if (synthetic and not key) or key.lower().startswith(_TEST_PREFIXES):
            simulated = self._simulation(request, model_config, account)
            return self._simulation_stream(simulated) if request.stream else simulated
        if account is None:
            raise ColabError("A Google account is required", 401)
        if not key and not account.extra_config.get("refresh_token"):
            raise ColabError("Selected account has no Google credential; sign in from Accounts", 401)
        self._check_scopes(account)
        self._model_name(model_config)
        self._validate_request(request, model_config)  # reject before creating a billable VM
        await self._refresh_token(account, settings.proxy)
        if request.stream:
            return self._stream(request, model_config, variables, account, settings)
        async with self._lease(account, settings) as runtime:
            info = await self._start(runtime, model_config)
            payload = await self._build_payload(request, model_config, variables, settings, info)
            terminal = None
            async for data in self._ollama_events(runtime, payload):
                terminal = data
            if not terminal or not terminal.get("done"):
                raise ColabError("Ollama returned no completed response", 502)
            message = terminal.get("message", {})
            calls = self._tool_calls(message.get("tool_calls", [])) or None
            reason = terminal.get("done_reason") or "stop"
            if reason == "stop" and calls:
                reason = "tool_calls"
            extra = {"reasoning_content": message["thinking"]} if message.get("thinking") else {}
            return ChatCompletionResponse(id="chatcmpl-colab-" + uuid.uuid4().hex[:12], model=request.model,
                                          choices=[ChatChoice(index=0, message=ChatMessage(role="assistant", content=message.get("content", ""),
                                                                                         tool_calls=calls, **extra), finish_reason=reason)],
                                          usage=self._usage(terminal))

    # ------------------------------------------------------------ Google OAuth
    @staticmethod
    def _oauth_config() -> Dict[str, Any]:
        path = Path.home() / ".colab-cli-oauth-config.json"
        if path.is_file():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                config = loaded.get("installed", loaded)
            except (ValueError, AttributeError):
                raise ValueError("Invalid local Colab OAuth client JSON") from None
            if not config.get("client_id"):
                raise ValueError("Local Colab OAuth client has no client_id")
            return {**_PUBLIC_CLIENT, **config}
        return dict(_PUBLIC_CLIENT)

    async def oauth_start(self) -> Dict[str, Any]:
        now = time.time()
        self._flows = {key: value for key, value in self._flows.items() if now - value["created_at"] < 600}
        config = self._oauth_config()
        # Remote landing is registered only for the CLI's Cloud SDK client.
        redirect = config.get("remote_redirect_uri", _REDIRECT_URI)
        if config["client_id"] != _PUBLIC_CLIENT["client_id"] and "remote_redirect_uri" not in config:
            raise ValueError("A custom OAuth client must declare its registered remote_redirect_uri")
        state = secrets.token_urlsafe(24)
        verifier = secrets.token_urlsafe(64)
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
        flow_id = "flow_colab_" + uuid.uuid4().hex[:12]
        self._flows[flow_id] = {"created_at": now, "status": "pending", "state": state, "verifier": verifier,
                               "config": config, "redirect_uri": redirect, "lock": asyncio.Lock()}
        auth_url = config["auth_uri"] + "?" + urlencode({
            "client_id": config["client_id"], "redirect_uri": redirect, "response_type": "code",
            "scope": " ".join(_SCOPES), "access_type": "offline", "prompt": "consent",
            "token_usage": "remote", "state": state, "code_challenge": challenge, "code_challenge_method": "S256",
        })
        return {"flow_id": flow_id, "mode": "callback", "auth_url": auth_url, "state": state,
                "redirect_uri": redirect, "expires_in": 600, "supports_callback_paste": True,
                "instructions": "Approve Colab/email access using the intended Google account. Google displays an authorization code: paste that code (or the complete callback URL) below. No local callback server is needed."}

    async def oauth_status(self, flow_id: str) -> Dict[str, Any]:
        flow = self._flows.get(flow_id)
        if not flow or time.time() - flow["created_at"] >= 600:
            self._flows.pop(flow_id, None)
            return {"status": "error", "error": "Colab sign-in expired; start again"}
        if flow["status"] == "complete":
            return {"status": "complete", "account": flow["account"]}
        return {"status": flow["status"], **({"error": flow["error"]} if flow.get("error") else {})}

    async def oauth_callback(self, flow_id: str, callback_url: str) -> Dict[str, Any]:
        flow = self._flows.get(flow_id)
        if not flow or time.time() - flow["created_at"] >= 600:
            return {"status": "error", "error": "Colab sign-in expired; start again"}
        async with flow["lock"]:
            if flow["status"] != "pending":
                return await self.oauth_status(flow_id)
            value = callback_url.strip()
            if value.startswith(("http://", "https://")):
                parsed = urlparse(value)
                registered = urlparse(flow["redirect_uri"])
                params = parse_qs(parsed.query)
                if (parsed.scheme, parsed.netloc, parsed.path) != (registered.scheme, registered.netloc, registered.path):
                    return {"status": "error", "error": "OAuth callback URL does not match this flow"}
                if not secrets.compare_digest(params.get("state", [""])[0], flow["state"]):
                    return {"status": "error", "error": "OAuth state mismatch"}
                if params.get("error"):
                    return {"status": "error", "error": "Google authorization was denied"}
                code = params.get("code", [""])[0]
            else:
                code = value  # official CLI's copy/paste-code flow, protected by PKCE
            if not code or len(code) > 4096 or any(character.isspace() for character in code):
                return {"status": "error", "error": "Paste Google's authorization code or complete callback URL"}
            config = flow["config"]
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    response = await client.post(_TOKEN_URL, data={
                        "client_id": config["client_id"], "client_secret": config.get("client_secret", ""),
                        "code": code, "code_verifier": flow["verifier"], "redirect_uri": flow["redirect_uri"],
                        "grant_type": "authorization_code",
                    })
                    _check_http(response, (code, flow["verifier"], config.get("client_secret", "")))
                    token = _json_response(response)
                    account = await self._account_from_token(client, token, config)
                flow.update(status="complete", account=account)
                flow.pop("verifier", None)
                return {"status": "complete", "account": account}
            except (ColabError, httpx.HTTPError, ValueError) as exc:
                flow.update(status="error", error=_excerpt(str(exc), (code, flow["verifier"])))
                flow.pop("verifier", None)
                return {"status": "error", "error": flow["error"]}

    async def _account_from_token(self, client: httpx.AsyncClient, token: Dict[str, Any],
                                   config: Dict[str, Any]) -> Account:
        if not token.get("access_token") or not token.get("refresh_token"):
            raise ColabError("Google did not return durable access/refresh credentials; approve offline access", 401)
        response = await client.get(_USERINFO_URL, headers={"Authorization": "Bearer " + token["access_token"]})
        _check_http(response, (token["access_token"], token["refresh_token"]))
        user = _json_response(response)
        email = user.get("email")
        if not email:
            raise ColabError("Google did not identify the signed-in account", 401)
        scopes = str(token.get("scope", " ".join(_SCOPES))).split()
        account = Account(provider_id="colab", name="Colab (" + str(email) + ")", api_key=token["access_token"],
                          base_url=_COLAB_URL, extra_config={
                              "auth_type": "oauth", "email": email, "scopes": scopes,
                              "client_id": config["client_id"], "client_secret": config.get("client_secret", ""),
                              "refresh_token": token["refresh_token"], "expires_at": time.time() + int(token.get("expires_in", 3600)),
                          })
        self._check_scopes(account)
        return account

    async def _refresh_token(self, account: Account, proxy: Optional[str], *, force: bool = False) -> bool:
        lock = self._refresh_locks.setdefault(account.id, asyncio.Lock())
        async with lock:
            extra = account.extra_config
            expires_at = float(extra.get("expires_at", 0))
            if not force and account.api_key and (not expires_at or expires_at - time.time() > 120):
                return False
            if not extra.get("refresh_token"):
                if account.api_key and not expires_at:
                    return False
                raise ColabError("Google access token expired and has no refresh token; sign in again", 401)
            config = self._oauth_config()
            data = {"grant_type": "refresh_token", "refresh_token": extra["refresh_token"],
                    "client_id": extra.get("client_id") or config["client_id"],
                    "client_secret": extra.get("client_secret") or config.get("client_secret", "")}
            async with httpx.AsyncClient(timeout=30, proxy=proxy) as client:
                response = await client.post(_TOKEN_URL, data=data)
                _check_http(response, (account.api_key, data["refresh_token"], data["client_secret"]))
                token = _json_response(response)
            if not token.get("access_token"):
                raise ColabError("Google refresh omitted its access token", 401)
            account.api_key = token["access_token"]
            if token.get("refresh_token"):
                extra["refresh_token"] = token["refresh_token"]
            extra["expires_at"] = time.time() + int(token.get("expires_in", 3600))
            if token.get("scope"):
                extra["scopes"] = token["scope"].split()
            self._persist(account)  # persist every rotation before another upstream call
            self._check_scopes(account)
            return True

    async def refresh_account(self, account: Account) -> bool:
        return await self._refresh_token(account, account.proxy_url, force=True)

    async def oauth_import_local(self) -> Optional[Account]:
        # Import only user JSON credentials. No gcloud subprocess, metadata-server
        # probes, interactive keyring prompt, or ambient service-account guessing.
        paths = []
        if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            paths.append(Path(os.environ["GOOGLE_APPLICATION_CREDENTIALS"]).expanduser())
        paths.extend([
            Path.home() / ".config/gcloud/application_default_credentials.json",
            Path.home() / ".config/colab-cli/token.json",
        ])
        for path in paths:
            if not path.is_file():
                continue
            try:
                loaded = json.loads(await asyncio.to_thread(path.read_text, encoding="utf-8"))
            except ValueError:
                raise ValueError("Local Google credential file contains invalid JSON") from None
            if loaded.get("type") not in (None, "authorized_user") or not loaded.get("refresh_token"):
                raise ValueError("Local import supports only Google authorized-user credentials with a refresh token")
            config = self._oauth_config()
            config.update({key: loaded[key] for key in ("client_id", "client_secret") if loaded.get(key)})
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(_TOKEN_URL, data={
                    "grant_type": "refresh_token", "refresh_token": loaded["refresh_token"],
                    "client_id": config["client_id"], "client_secret": config.get("client_secret", ""),
                })
                _check_http(response, (loaded["refresh_token"], config.get("client_secret", "")))
                token = _json_response(response)
                token.setdefault("refresh_token", loaded["refresh_token"])
                # Validate the scopes actually granted, not the scopes we wished
                # ADC had. Userinfo alone does not establish Colab access.
                if not token.get("scope"):
                    # A bearer-only, read-only Colab probe confirms permission
                    # without putting an access token into a URL/httpx log.
                    check = await client.get(_COLAB_URL + "/tun/m/assignments", params={"authuser": "0"},
                                             headers={"Authorization": "Bearer " + str(token.get("access_token", "")),
                                                      "X-Colab-Client-Agent": "colab-cli", "Accept": "application/json"})
                    _check_http(check, (token.get("access_token", ""),))
                    if "assignments" not in _json_response(check):
                        raise ColabError("Local credential could not verify Colab access", 401)
                    token["scope"] = " ".join((_SCOPES[1], _SCOPES[3]))
                return await self._account_from_token(client, token, config)
        return None

    async def fetch_usage_quota(self, account: Optional[Account], variables: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if account is None or not account.api_key or account.api_key.lower().startswith(_TEST_PREFIXES):
            return None
        self._check_scopes(account)
        runtime = _Runtime(account=account, settings=_Settings.read(variables, account))
        try:
            info = await self._control(runtime, "GET", "/tun/m/ccu-info")
        finally:
            if runtime.client:
                await runtime.client.aclose()
        if not all(key in info for key in ("currentBalance", "consumptionRateHourly", "assignmentsCount")):
            raise ColabError("Colab quota response omitted compute-unit balance", 502)
        return {"groups": [], "balance": {"value": float(info["currentBalance"]), "label": "Compute units"},
                "summary": [{"label": "Compute units / hour", "value": str(info["consumptionRateHourly"])},
                            {"label": "Active Colab assignments", "value": str(info["assignmentsCount"])}]}


plugin = ColabPlugin()
