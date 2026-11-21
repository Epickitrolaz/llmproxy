"""Colab's real protocol and resource lifecycle, without live Google requests."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import struct
import time
from contextlib import redirect_stdout
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from fastapi.testclient import TestClient

from llmproxy.core.manager import PluginManager, plugin_manager
from llmproxy.core.security import redact_account
from llmproxy.dashboard.loader import load_plugin_from_file
from llmproxy.dashboard.templates import get_starter_templates
from llmproxy.main import app
from llmproxy.schemas import Account, ChatCompletionRequest, ChatMessage, PluginModelConfig
from templates.colab import (
    ColabError,
    ColabPlugin,
    _KernelSocket,
    _REMOTE_SOURCE,
    _RESULT_PREFIX,
    _Runtime,
    _Settings,
)

ROOT = Path(__file__).resolve().parent.parent
SCOPES = "openid https://www.googleapis.com/auth/userinfo.email https://www.googleapis.com/auth/cloud-platform https://www.googleapis.com/auth/colaboratory"


def account(name="primary", **kwargs):
    return Account(id="acc_" + name, provider_id="colab", name=name, api_key="google-access-" + name,
                   extra_config={"refresh_token": "google-refresh", "expires_at": time.time() + 3600,
                                 "scopes": SCOPES.split()}, **kwargs)


def response_data(*, done=True, tools=False, cached=80):
    message = {"role": "assistant", "content": "OK", "thinking": "Consider this"}
    if tools:
        message["tool_calls"] = [{"function": {"index": 0, "name": "weather", "arguments": {"city": "Paris"}}}]
    return {"model": "qwen3:0.6b", "message": message, "done": done, "done_reason": "stop",
            "prompt_eval_count": 100, "prompt_eval_cached_count": cached, "eval_count": 10}


class FakeSocket:
    def __init__(self, backend):
        self.backend = backend
        self.queue = asyncio.Queue()
        self.closed = False
        self.executing = 0

    async def send_json(self, message):
        assert message["content"]["allow_stdin"] is False
        assert message["content"]["store_history"] is False
        request_id = message["header"]["msg_id"]
        code = message["content"]["code"]
        self.backend.codes.append(code)
        if "['chat'](" in code:
            kind = "chat"
            events = [response_data(done=False), response_data()] if self.backend.streaming else [response_data()]
            event_kind = "response"
            self.backend.inferences += 1
        elif "['bootstrap'](" in code or "['prepare_model'](" in code:
            kind = "setup"
            self.backend.prepares += 1
            events = [{"capabilities": ["completion", "tools", "thinking", "vision"],
                       "thinking": {"values": [True, False]}}]
            event_kind = "ready"
        else:
            kind = "bridge"
            self.backend.bridges += 1
            events = []
            event_kind = "ready"
        self.backend.kinds.append(kind)
        if self.backend.fail_kernel and kind == "setup":
            self.queue.put_nowait({"header": {"msg_type": "error"}, "parent_header": {"msg_id": request_id},
                                   "content": {"ename": "RuntimeError", "evalue": "installation failed"}})
            return
        for event in events:
            line = _RESULT_PREFIX + json.dumps({"id": request_id, "type": event_kind, "data": event}) + "\n"
            # IPython may split a single output line across messages.
            midpoint = len(line) // 2
            for text in (line[:midpoint], line[midpoint:]):
                self.queue.put_nowait({"header": {"msg_type": "stream"}, "parent_header": {"msg_id": request_id},
                                       "content": {"name": "stdout", "text": text}})
        # Execution order is not guaranteed between shell and iopub channels.
        self.queue.put_nowait({"header": {"msg_type": "execute_reply"}, "parent_header": {"msg_id": request_id},
                               "content": {"status": "ok"}})
        self.queue.put_nowait({"header": {"msg_type": "status"}, "parent_header": {"msg_id": request_id},
                               "content": {"execution_state": "idle"}})

    async def receive_json(self):
        return await self.queue.get()

    async def aclose(self):
        self.closed = True


class Backend:
    def __init__(self):
        self.calls = []
        self.codes = []
        self.kinds = []
        self.allocations = self.kernels = self.prepares = self.bridges = self.inferences = 0
        self.assigned = False
        self.releases = 0
        self.streaming = False
        self.fail_kernel = False
        self.quota_status = None
        self.uncertain_post = False
        self.proxy_token = "runtime-proxy-token"
        self.interrupts = 0
        self.block_release = False
        self.stall_stream = False

    def assignment(self):
        return {"endpoint": "gpu-owned-runtime", "accelerator": "T4", "variant": 1, "machineShape": 0,
                "runtimeProxyInfo": {"url": "https://owned.googleusercontent.com", "token": self.proxy_token,
                                     "tokenExpiresInSeconds": 3600}}

    def handle(self, request):
        self.calls.append(request)
        path = request.url.path
        if request.url.host == "colab.research.google.com":
            assert request.headers["authorization"].startswith("Bearer google-")
            assert request.headers["x-colab-client-agent"] == "colab-cli"
            assert request.url.params["authuser"] == "0"
            if path == "/tun/m/assign" and request.method == "GET":
                return httpx.Response(200, text=")]}'\n" + json.dumps(self.assignment() if self.assigned else {"token": "xsrf"}))
            if path == "/tun/m/assign" and request.method == "POST":
                assert request.headers["x-goog-colab-token"] == "xsrf"
                if self.quota_status:
                    return httpx.Response(self.quota_status, json={"error": "assignment refused"})
                self.allocations += 1
                self.assigned = True
                if self.uncertain_post:
                    self.uncertain_post = False
                    raise httpx.ReadTimeout("uncertain allocation", request=request)
                return httpx.Response(200, json=self.assignment())
            if path == "/tun/m/assignments":
                return httpx.Response(200, json={"assignments": [self.assignment()] if self.assigned else []})
            if path.startswith("/tun/m/unassign/"):
                assert path == "/tun/m/unassign/gpu-owned-runtime"
                if self.block_release:
                    return httpx.Response(503, json={"error": "try later"})
                if request.method == "GET":
                    return httpx.Response(200, json={"token": "release-xsrf"})
                assert request.headers["x-goog-colab-token"] == "release-xsrf"
                self.releases += 1
                self.assigned = False
                return httpx.Response(204)
            if path == "/tun/m/ccu-info":
                return httpx.Response(200, json={"currentBalance": 12.5, "consumptionRateHourly": 1.2, "assignmentsCount": 1})
        assert request.headers["x-colab-runtime-proxy-token"] == self.proxy_token
        assert "colab-runtime-proxy-token" not in str(request.url)
        if path == "/api/kernels" and request.method == "POST":
            self.kernels += 1
            assert json.loads(request.content) == {"name": "python3"}
            return httpx.Response(201, json={"id": "kernel-owned"})
        if path == "/api/kernels/kernel-owned/interrupt":
            self.interrupts += 1
            return httpx.Response(204)
        if path == "/api/kernels/kernel-owned":
            return httpx.Response(200, json={"id": "kernel-owned", "execution_state": "idle"})
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    def install(self, monkeypatch, plugin):
        original_client = httpx.AsyncClient
        def client(*args, **kwargs):
            kwargs.pop("proxy", None)
            return original_client(*args, transport=httpx.MockTransport(self.handle), **kwargs)
        monkeypatch.setattr(httpx, "AsyncClient", client)
        async def connect(client, url, headers, timeout):
            assert headers["X-Colab-Runtime-Proxy-Token"] == self.proxy_token
            assert url.startswith("https://owned.googleusercontent.com/api/kernels/kernel-owned/channels?session_id=")
            socket = FakeSocket(self)
            if self.stall_stream:
                send = socket.send_json
                async def stall(message):
                    await send(message)
                    if "['chat'](" not in message["content"]["code"]:
                        return
                    output = []
                    while not socket.queue.empty():
                        output.append(socket.queue.get_nowait())
                    # Keep one complete data event then leave execution waiting.
                    for incoming in output[:2]:
                        socket.queue.put_nowait(incoming)
                socket.send_json = stall
            return socket
        monkeypatch.setattr(_KernelSocket, "connect", connect)
        monkeypatch.setattr(plugin, "_persist", lambda account: None)


def test_template_is_single_standalone_colab_provider(monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("Import must not create a network client")
    monkeypatch.setattr(httpx, "AsyncClient", no_network)
    plugin = load_plugin_from_file(ROOT / "templates/colab.py")
    assert plugin.metadata.id == "colab"
    assert plugin.metadata.oauth_login and plugin.metadata.oauth_local_import
    assert plugin.metadata.supports_usage_meter
    assert plugin.metadata.models[0].upstream_model == "qwen3:0.6b"
    assert plugin.metadata.models[0].id == "cl/qwen3:0.6b"
    assert plugin.metadata.models[0].reasoning_levels == ["none", "high"]
    assert plugin.validate_variables({}) == []
    assert not (ROOT / "templates/google_colab.py").exists()
    templates = [item for item in get_starter_templates() if item["id"] == "colab"]
    assert len(templates) == 1
    assert templates[0]["name"] == "Colab · Ollama"


@pytest.mark.parametrize("variables", [
    {"gpu": "A10"}, {"gpu": "V5E1"}, {"vm_idle_seconds": -2}, {"max_active_vms": 1.5},
    {"startup_timeout": 0}, {"timeout": float("nan")}, {"split_across_accounts": "maybe"},
    {"ollama_version": "latest; rm -rf /"}, {"models_directory": "../escape"},
    {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai"},
])
def test_invalid_settings_fail_before_allocation(variables):
    assert ColabPlugin().validate_variables(variables)


def test_real_protocol_hot_path_reuses_vm_kernel_and_model(monkeypatch):
    async def run():
        plugin = ColabPlugin()
        backend = Backend()
        backend.install(monkeypatch, plugin)
        acc = account()
        model = plugin.metadata.models[0]
        request = ChatCompletionRequest(model=model.id, messages=[ChatMessage(content="ping")])
        first = await plugin.chat_completion(request, model, {}, acc)
        before = len(backend.calls)
        second = await plugin.chat_completion(request, model, {}, acc)
        assert first.choices[0].message.content == second.choices[0].message.content == "OK"
        assert first.choices[0].message.reasoning_content == "Consider this"
        assert first.usage.prompt_tokens == 100
        assert first.usage.cache_read_input_tokens == 80
        assert backend.allocations == backend.kernels == backend.prepares == backend.bridges == 1
        assert backend.inferences == 2
        assert len(backend.calls) == before  # hot path: no allocation/status/download round trip
        nbh = backend.calls[0].url.params["nbh"]
        assert len(nbh) == 44 and nbh.endswith("." * 8)
        assert backend.calls[0].url.params["variant"] == "GPU"
        assert backend.calls[0].url.params["accelerator"] == "T4"
        assert "shape" not in backend.calls[0].url.params
        assert acc.extra_config["colab_runtime"]["owner"] == "llmproxy-colab"
        await plugin.aclose()
        assert backend.releases == 1
        assert "colab_runtime" not in acc.extra_config
    asyncio.run(run())


def test_streaming_native_usage_is_reported_without_client_opt_in(monkeypatch):
    async def run():
        plugin = ColabPlugin()
        backend = Backend()
        backend.streaming = True
        backend.install(monkeypatch, plugin)
        model = plugin.metadata.models[0]
        stream = await plugin.chat_completion(ChatCompletionRequest(model=model.id, messages=[ChatMessage(content="ping")], stream=True), model, {}, account())
        chunks = [chunk async for chunk in stream]
        assert chunks[0].choices[0].delta["role"] == "assistant"
        assert any(chunk.choices[0].delta.get("reasoning_content") for chunk in chunks)
        assert chunks[-1].choices[0].finish_reason == "stop"
        assert chunks[-1].usage.cache_read_input_tokens == 80
        assert not plugin._runtimes["acc_primary"].active
        await plugin.aclose()
    asyncio.run(run())


def test_uncertain_allocation_reconciles_same_hash_without_second_post(monkeypatch):
    async def run():
        plugin = ColabPlugin()
        backend = Backend()
        backend.uncertain_post = True
        backend.install(monkeypatch, plugin)
        model = plugin.metadata.models[0]
        await plugin.chat_completion(ChatCompletionRequest(model=model.id, messages=[ChatMessage(content="ping")]), model,
                                     {"assign_retry_backoff": 0}, account())
        assign_calls = [call for call in backend.calls if call.url.path == "/tun/m/assign"]
        assert [call.method for call in assign_calls] == ["GET", "POST", "GET"]
        assert len({call.url.params["nbh"] for call in assign_calls}) == 1
        assert backend.allocations == 1
        await plugin.aclose()
    asyncio.run(run())


@pytest.mark.parametrize("status", [401, 403, 412, 429])
def test_allocation_errors_are_not_simulated_or_retried(monkeypatch, status):
    async def run():
        plugin = ColabPlugin()
        backend = Backend()
        backend.quota_status = status
        backend.install(monkeypatch, plugin)
        model = plugin.metadata.models[0]
        with pytest.raises(ColabError) as error:
            await plugin.chat_completion(ChatCompletionRequest(model=model.id, messages=[ChatMessage(content="ping")]), model, {}, account())
        assert error.value.status_code == status
        assert backend.allocations == backend.prepares == backend.inferences == 0
        assert len([call for call in backend.calls if call.method == "POST"]) == 1
        assert "Simulated" not in str(error.value)
        await plugin.aclose()
    asyncio.run(run())


def test_remote_setup_error_is_not_blindly_reexecuted(monkeypatch):
    async def run():
        plugin = ColabPlugin()
        backend = Backend()
        backend.fail_kernel = True
        backend.install(monkeypatch, plugin)
        model = plugin.metadata.models[0]
        with pytest.raises(ColabError, match="installation failed"):
            await plugin.chat_completion(ChatCompletionRequest(model=model.id, messages=[ChatMessage(content="ping")]), model, {}, account())
        assert backend.prepares == 1
        assert backend.interrupts == 1
        await plugin.aclose()
    asyncio.run(run())


def test_lifecycle_idle_zero_and_max_age_release_only_owned_vm(monkeypatch):
    async def run():
        plugin = ColabPlugin()
        backend = Backend()
        backend.install(monkeypatch, plugin)
        model = plugin.metadata.models[0]
        acc = account()
        req = ChatCompletionRequest(model=model.id, messages=[ChatMessage(content="ping")])
        await plugin.chat_completion(req, model, {"vm_idle_seconds": 0}, acc)
        assert backend.releases == 1 and not plugin._runtimes
        await plugin.chat_completion(req, model, {}, acc)
        runtime = plugin._runtimes[acc.id]
        runtime.created_at -= 4000
        await plugin.chat_completion(req, model, {}, acc)
        assert backend.releases == 2
        assert backend.allocations == 3
        await plugin.aclose()
        assert backend.releases == 3
    asyncio.run(run())


def test_runtime_survives_hot_reload_but_refreshes_proxy_token(monkeypatch):
    async def run():
        old = ColabPlugin()
        backend = Backend()
        backend.install(monkeypatch, old)
        acc = account()
        model = old.metadata.models[0]
        req = ChatCompletionRequest(model=model.id, messages=[ChatMessage(content="ping")])
        await old.chat_completion(req, model, {}, acc)
        await old.aclose(release_resources=False)
        assert backend.releases == 0
        replacement = ColabPlugin()
        monkeypatch.setattr(replacement, "_persist", lambda account: None)
        backend.proxy_token = "refreshed-runtime-token"
        await replacement.chat_completion(req, model, {}, acc)
        assert backend.allocations == 1 and backend.kernels == 1
        assert acc.extra_config["colab_runtime"]["token"] == backend.proxy_token
        assert any(call.url.path == "/tun/m/assignments" for call in backend.calls)
        await replacement.aclose()
    asyncio.run(run())


def test_failed_release_keeps_endpoint_and_schedules_retry(monkeypatch):
    async def run():
        plugin = ColabPlugin()
        backend = Backend()
        backend.install(monkeypatch, plugin)
        acc = account()
        model = plugin.metadata.models[0]
        await plugin.chat_completion(ChatCompletionRequest(model=model.id, messages=[ChatMessage(content="ping")]), model, {}, acc)
        backend.block_release = True
        runtime = plugin._runtimes[acc.id]
        await plugin._expire(runtime)
        assert runtime.endpoint and runtime.timer
        assert acc.extra_config["colab_runtime"]["endpoint"] == runtime.endpoint
        assert runtime.client is None
        backend.block_release = False
        await plugin.aclose()
    asyncio.run(run())


def test_idle_timer_releases_vm_without_a_followup_request(monkeypatch):
    async def run():
        plugin = ColabPlugin()
        backend = Backend()
        backend.install(monkeypatch, plugin)
        acc = account()
        runtime = _Runtime(account=acc, settings=_Settings.read({"vm_idle_seconds": 0, "vm_max_lifetime_seconds": 0}),
                           created_at=time.time(), last_used_at=time.time(), hardware=("T4", False))
        backend.assigned = True
        plugin._apply_assignment(runtime, backend.assignment())
        plugin._runtimes[acc.id] = runtime
        released = asyncio.Event()
        original = plugin._stop_runtime
        async def stop(item):
            await original(item)
            released.set()
        monkeypatch.setattr(plugin, "_stop_runtime", stop)
        plugin._schedule_expiry(runtime)
        await asyncio.wait_for(released.wait(), 1)
        assert backend.releases == 1
        assert "colab_runtime" not in acc.extra_config
        await plugin.aclose()
    asyncio.run(run())


def test_restart_rearms_owned_timers_without_network_or_new_vm(monkeypatch):
    async def run():
        plugin = ColabPlugin()
        acc = account()
        runtime = _Runtime(account=acc, settings=_Settings.read({}), endpoint="gpu-owned-runtime",
                           url="https://owned.googleusercontent.com", token="runtime-token", token_expires_at=time.time() + 3600,
                           notebook_hash="12345678_1234_1234_1234_123456789012........", created_at=time.time(),
                           last_used_at=time.time(), hardware=("T4", False))
        acc.extra_config["colab_runtime"] = runtime.saved()
        def no_network(*args, **kwargs):
            raise AssertionError("Startup must not make network calls")
        monkeypatch.setattr(httpx, "AsyncClient", no_network)
        await plugin.astart({}, [acc, account("unowned")])
        assert list(plugin._runtimes) == [acc.id]
        assert plugin._runtimes[acc.id].timer
        await plugin.aclose(release_resources=False)
        assert acc.extra_config["colab_runtime"]["endpoint"] == "gpu-owned-runtime"
    asyncio.run(run())


def test_closing_stream_interrupts_execution_and_releases_account_lock(monkeypatch):
    async def run():
        plugin = ColabPlugin()
        backend = Backend()
        backend.streaming = True
        backend.install(monkeypatch, plugin)
        acc = account()
        model = plugin.metadata.models[0]
        stream = await plugin.chat_completion(ChatCompletionRequest(model=model.id, messages=[ChatMessage(content="ping")], stream=True), model, {}, acc)
        first = await stream.__anext__()
        assert first.choices[0].delta["role"] == "assistant"
        content = await stream.__anext__()
        assert content.choices[0].delta["content"] == "OK"
        await stream.aclose()
        runtime = plugin._runtimes[acc.id]
        assert runtime.active == 0 and not runtime.lock.locked()
        assert runtime.socket is None
        assert backend.interrupts == 1
        await plugin.aclose()
    asyncio.run(run())


def test_stream_deadline_survives_router_priming_task_change(monkeypatch):
    async def run():
        plugin = ColabPlugin()
        backend = Backend()
        backend.streaming = backend.stall_stream = True
        backend.install(monkeypatch, plugin)
        acc = account()
        model = plugin.metadata.models[0]
        stream = await plugin.chat_completion(ChatCompletionRequest(model=model.id, messages=[ChatMessage(content="ping")], stream=True), model, {}, acc)
        runtime = None
        # Prime in a separate task, matching the router's first-output handling.
        async def prime():
            await stream.__anext__()
            await stream.__anext__()
        await asyncio.create_task(prime())
        runtime = plugin._runtimes[acc.id]
        # Force the deadline forward cheaply without waiting real seconds.
        import templates.colab as colab
        original_monotonic = colab.time.monotonic
        monkeypatch.setattr(colab.time, "monotonic", lambda: original_monotonic() + 1000)
        with pytest.raises(ColabError, match="timeout"):
            await stream.__anext__()
        monkeypatch.setattr(colab.time, "monotonic", original_monotonic)
        assert runtime.active == 0 and backend.interrupts == 1
        await plugin.aclose()
    asyncio.run(run())


@pytest.mark.parametrize("stream", [False, True])
def test_full_colab_wire_path_through_router(monkeypatch, stream):
    plugin = ColabPlugin()
    backend = Backend()
    backend.streaming = stream
    backend.install(monkeypatch, plugin)
    acc = account()
    monkeypatch.setattr(plugin_manager, "plugins", {"colab": plugin})
    monkeypatch.setattr(plugin_manager, "accounts", {acc.id: acc})
    monkeypatch.setattr(plugin_manager, "configs", {"colab": {"enabled": True, "variables": {"vm_idle_seconds": 0}}})
    monkeypatch.setattr(plugin_manager, "rules", {})
    monkeypatch.setattr(plugin_manager, "api_keys", {})
    monkeypatch.setattr(plugin_manager, "logs", [])
    monkeypatch.setattr(plugin_manager, "_save_accounts", lambda: None)
    monkeypatch.setattr(plugin_manager, "_schedule_logs_save", lambda: None)
    monkeypatch.setattr(plugin_manager, "discover_and_load_all", lambda: None)
    with TestClient(app, headers={"X-Admin-Token": "test-admin-token"}) as client:
        result = client.post("/v1/chat/completions", json={"model": plugin.metadata.models[0].id,
                            "messages": [{"role": "user", "content": "ping"}], "stream": stream,
                            "stream_options": {"include_usage": True}})
        assert result.status_code == 200, result.text
        assert "Consider this" in result.text and "cached_tokens" in result.text
        assert backend.allocations == backend.prepares == backend.inferences == backend.releases == 1
        assert client.get("/api/stats").json()["cache_hit_rate"] == 80.0


def test_router_fails_over_every_colab_account_without_simulation(monkeypatch):
    plugin = ColabPlugin()
    first, second = account(), account("secondary", priority=2)
    attempts = []
    monkeypatch.setattr(plugin_manager, "plugins", {"colab": plugin})
    monkeypatch.setattr(plugin_manager, "accounts", {first.id: first, second.id: second})
    monkeypatch.setattr(plugin_manager, "configs", {"colab": {"enabled": True, "variables": {"split_across_accounts": False}}})
    monkeypatch.setattr(plugin_manager, "rules", {})
    monkeypatch.setattr(plugin_manager, "api_keys", {})
    monkeypatch.setattr(plugin_manager, "logs", [])
    monkeypatch.setattr(plugin_manager, "_save_accounts", lambda: None)
    monkeypatch.setattr(plugin_manager, "_schedule_logs_save", lambda: None)
    async def start(runtime, model):
        attempts.append(runtime.account.id)
        if runtime.account.id == first.id:
            raise ColabError("allocation quota limit", 412)
        return {"capabilities": ["tools", "thinking"]}
    async def events(runtime, payload):
        yield response_data()
    monkeypatch.setattr(plugin, "_start", start)
    monkeypatch.setattr(plugin, "_ollama_events", events)
    client = TestClient(app)
    result = client.post("/v1/chat/completions", json={"model": plugin.metadata.models[0].id, "messages": [{"role": "user", "content": "ping"}]})
    assert result.status_code == 200 and "Simulated" not in result.text
    assert attempts == [first.id, second.id]
    assert first.rate_limited_until > time.time()


def test_account_redaction_masks_nested_runtime_proxy_token():
    acc = account()
    acc.extra_config["colab_runtime"] = {"owner": "llmproxy-colab", "token": "runtime-proxy-secret"}
    redacted = redact_account(acc).model_dump()
    assert "runtime-proxy-secret" not in json.dumps(redacted)
    assert "google-refresh" not in json.dumps(redacted)


def test_requests_serialize_on_one_vm_but_parallelize_across_accounts():
    async def run():
        plugin = ColabPlugin()
        primary, secondary = account(), account("secondary")
        settings = _Settings.read({})
        primary_entered, secondary_entered, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        active, maximum = {}, {}
        async def work(acc, entered):
            async with plugin._lease(acc, settings):
                active[acc.id] = active.get(acc.id, 0) + 1
                maximum[acc.id] = max(maximum.get(acc.id, 0), active[acc.id])
                entered.set()
                await release.wait()
                active[acc.id] -= 1
        first = asyncio.create_task(work(primary, primary_entered))
        await primary_entered.wait()
        second = asyncio.create_task(work(primary, asyncio.Event()))
        third = asyncio.create_task(work(secondary, secondary_entered))
        await asyncio.wait_for(secondary_entered.wait(), 1)
        assert plugin._runtimes[primary.id].active == 2
        assert active[primary.id] == active[secondary.id] == 1
        release.set()
        await asyncio.gather(first, second, third)
        assert maximum == {primary.id: 1, secondary.id: 1}
        await plugin.aclose()
    asyncio.run(run())


def test_manager_reload_waits_for_old_state_transfer(tmp_path):
    async def run():
        manager = PluginManager(plugins_dir=tmp_path / "plugins", data_dir=tmp_path / "data")
        old, new = ColabPlugin(), ColabPlugin()
        calls = []
        closed = asyncio.Event()
        async def close(*, release_resources):
            calls.append(("close", release_resources))
            await closed.wait()
        async def start(variables, accounts):
            calls.append(("start", variables, accounts))
        old.aclose = close
        new.astart = start
        manager.plugins["colab"] = new
        manager._activate_plugin(new, old)
        assert not new._activation_task.done()
        closed.set()
        await manager.prepare_plugin(new)
        assert calls[0] == ("close", False)
        assert calls[1][0] == "start"
    asyncio.run(run())


def test_vm_cap_rejects_new_vm_not_busy_existing_vm(monkeypatch):
    async def run():
        plugin = ColabPlugin()
        first, second = account(), account("secondary")
        settings = _Settings.read({"max_active_vms": 1})
        async with plugin._lease(first, settings) as runtime:
            async def other():
                with pytest.raises(ColabError, match="max_active_vms"):
                    async with plugin._lease(second, settings):
                        raise AssertionError("must not allocate")
            await other()
            await plugin._expire(runtime)
            assert runtime.active == 1
        assert not plugin._runtimes  # no allocation occurred
        await plugin.aclose()
    asyncio.run(run())


def test_real_missing_disabled_cooldown_or_old_scope_account_never_simulates():
    async def run():
        plugin = ColabPlugin()
        model = plugin.metadata.models[0]
        req = ChatCompletionRequest(model=model.id, messages=[ChatMessage(content="ping")])
        for acc in (
            Account(provider_id="colab", name="Missing"),
            account("disabled", enabled=False),
            account("cooldown", rate_limited_until=time.time() + 60),
        ):
            with pytest.raises(ColabError):
                await plugin.chat_completion(req, model, {}, acc)
        old = account("old")
        old.extra_config["scopes"] = ["https://www.googleapis.com/auth/cloud-platform"]
        with pytest.raises(ColabError, match="lacks Colab/email"):
            await plugin.chat_completion(req, model, {}, old)
        assert not plugin._runtimes
        simulated = await plugin.chat_completion(req, model, {}, None)
        assert "no VM was allocated" in simulated.choices[0].message.content
    asyncio.run(run())


def test_native_payload_translates_tools_images_reasoning_and_cache():
    async def run():
        plugin = ColabPlugin()
        model = PluginModelConfig(id="cl/vision", name="Vision", upstream_model="qwen3-vl:2b", supports_vision=True,
                                  context_window=4096, reasoning_levels=["none", "high"])
        request = ChatCompletionRequest(model=model.id, reasoning_effort="high", max_completion_tokens=50,
            messages=[ChatMessage(role="system", content=[{"type": "text", "text": "Be clear", "cache_control": {"type": "ephemeral"}}]),
                      ChatMessage(role="user", content=[{"type": "text", "text": "Look"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,YWJj"}}]),
                      ChatMessage(role="assistant", content=None, tool_calls=[{"id": "call1", "type": "function", "function": {"name": "weather", "arguments": '{"city":"Paris"}'}}]),
                      ChatMessage(role="tool", tool_call_id="call1", content="Sunny")],
            tools=[{"type": "function", "function": {"name": "weather", "parameters": {"type": "object"}}}],
            parallel_tool_calls=True, extra_body={"options": {"num_ctx": 2048}, "prompt_cache_key": "affinity",
                                                 "cache_control": {"type": "ephemeral"}, "seed": 7})
        info = {"capabilities": ["tools", "vision", "thinking"], "thinking": {"values": [True, False]}}
        payload = await plugin._build_payload(request, model, {}, _Settings.read({}), info)
        assert payload["model"] == "qwen3-vl:2b"
        assert payload["messages"][0]["content"] == "Be clear"
        assert payload["messages"][0]["content_parts"][0]["cache_control"] == {"type": "ephemeral"}
        assert payload["messages"][1]["images"] == ["YWJj"]
        assert payload["messages"][2]["tool_calls"][0]["function"]["arguments"] == {"city": "Paris"}
        assert payload["messages"][3]["tool_name"] == "weather"
        assert payload["messages"][3]["tool_call_id"] == "call1"
        assert payload["options"]["num_ctx"] == 2048
        assert payload["options"]["num_predict"] == 50
        assert payload["options"]["seed"] == 7
        assert payload["think"] is True and payload["keep_alive"] == -1
        assert payload["prompt_cache_key"] == "affinity"
        request.reasoning_effort = "none"
        assert (await plugin._build_payload(request, model, {}, _Settings.read({}), info))["think"] is False
    asyncio.run(run())


@pytest.mark.parametrize("updates", [{"n": 2}, {"tool_choice": "required"}, {"tool_choice": {"type": "function", "function": {"name": "weather"}}},
                                     {"parallel_tool_calls": False}])
def test_unsupported_ollama_features_fail_explicitly(updates):
    async def run():
        plugin = ColabPlugin()
        request = ChatCompletionRequest(model="cl/qwen3:0.6b", messages=[ChatMessage(content="ping")],
                                         tools=[{"type": "function", "function": {"name": "weather"}}], **updates)
        with pytest.raises(ValueError):
            await plugin._build_payload(request, plugin.metadata.models[0], {}, _Settings.read({}), {"capabilities": ["tools"]})
    asyncio.run(run())


def test_oauth_pkce_remote_code_flow_and_state_validation(monkeypatch, tmp_path):
    async def run():
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        plugin = ColabPlugin()
        flow = await plugin.oauth_start()
        params = parse_qs(urlparse(flow["auth_url"]).query)
        assert flow["mode"] == "callback" and flow["supports_callback_paste"]
        assert params["token_usage"] == ["remote"]
        assert params["redirect_uri"] == ["https://sdk.cloud.google.com/applicationdefaultauthcode.html"]
        assert params["code_challenge_method"] == ["S256"]
        assert "https://www.googleapis.com/auth/colaboratory" in params["scope"][0]
        calls = []
        def handle(request):
            calls.append(request)
            if request.url.path == "/token":
                data = parse_qs(request.content.decode())
                assert data["code"] == ["remote-code"]
                verifier = data["code_verifier"][0]
                challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
                assert params["code_challenge"] == [challenge]
                return httpx.Response(200, json={"access_token": "google-access", "refresh_token": "durable-refresh", "scope": SCOPES, "expires_in": 3600})
            assert request.headers["authorization"] == "Bearer google-access"
            return httpx.Response(200, json={"email": "me@example.com"})
        original = httpx.AsyncClient
        monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs))
        rejected = await plugin.oauth_callback(flow["flow_id"], flow["redirect_uri"] + "?code=remote-code&state=wrong")
        assert rejected["status"] == "error" and not calls
        result = await plugin.oauth_callback(flow["flow_id"], "remote-code")
        assert result["status"] == "complete"
        assert result["account"].extra_config["refresh_token"] == "durable-refresh"
        assert result["account"].provider_id == "colab"
        assert result["account"].base_url == "https://colab.research.google.com"
        assert (await plugin.oauth_status(flow["flow_id"]))["status"] == "complete"
        assert (await plugin.oauth_callback(flow["flow_id"], "remote-code"))["status"] == "complete"
        assert len(calls) == 2  # auth code not exchanged twice
    asyncio.run(run())


def test_refresh_rotation_is_serialized_and_persisted_immediately(monkeypatch):
    async def run():
        plugin = ColabPlugin()
        acc = account()
        acc.extra_config["expires_at"] = time.time() - 1
        calls = []
        snapshots = []
        def handle(request):
            calls.append(request)
            assert parse_qs(request.content.decode())["refresh_token"] == ["google-refresh"]
            return httpx.Response(200, json={"access_token": "google-new-access", "refresh_token": "rotated-refresh", "scope": SCOPES, "expires_in": 3600})
        original = httpx.AsyncClient
        monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs))
        monkeypatch.setattr(plugin, "_persist", lambda item: snapshots.append(item.model_dump()))
        await asyncio.gather(plugin._refresh_token(acc, None), plugin._refresh_token(acc, None))
        assert len(calls) == len(snapshots) == 1
        assert snapshots[0]["api_key"] == "google-new-access"
        assert snapshots[0]["extra_config"]["refresh_token"] == "rotated-refresh"
    asyncio.run(run())


def test_adc_import_requires_user_credential_and_real_colab_scopes(monkeypatch, tmp_path):
    async def run():
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        path = tmp_path / ".config/gcloud/application_default_credentials.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"type": "authorized_user", "client_id": "client", "client_secret": "public-client", "refresh_token": "refresh"}))
        original = httpx.AsyncClient
        def handle(request):
            if request.url.path == "/token":
                return httpx.Response(200, json={"access_token": "access", "scope": SCOPES, "expires_in": 3600})
            return httpx.Response(200, json={"email": "adc@example.com"})
        monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs))
        acc = await ColabPlugin().oauth_import_local()
        assert acc.extra_config["refresh_token"] == "refresh"
        assert acc.name == "Colab (adc@example.com)"
        assert json.loads(path.read_text())["refresh_token"] == "refresh"  # original CLI data is never changed
        path.write_text(json.dumps({"type": "service_account", "refresh_token": "refresh"}))
        with pytest.raises(ValueError, match="authorized-user"):
            await ColabPlugin().oauth_import_local()
    asyncio.run(run())


def test_quota_meter_uses_real_ccu_balance_without_allocating(monkeypatch):
    async def run():
        plugin = ColabPlugin()
        backend = Backend()
        backend.install(monkeypatch, plugin)
        usage = await plugin.fetch_usage_quota(account(), {})
        assert usage["balance"] == {"value": 12.5, "label": "Compute units"}
        assert usage["groups"] == []  # no made-up quota percentages
        assert len(backend.calls) == 1 and not backend.allocations
    asyncio.run(run())


def test_account_split_switch_preserves_router_owned_selection(tmp_path):
    manager = PluginManager(plugins_dir=tmp_path / "plugins", data_dir=tmp_path / "data")
    manager.plugins["colab"] = ColabPlugin()
    manager.configs["colab"] = {"variables": {"split_across_accounts": False}}
    manager.accounts = {"a": account("a", priority=1), "b": account("b", priority=2)}
    assert manager.select_account("colab", "round_robin").id == "acc_a"
    assert manager.select_account("colab", "round_robin").id == "acc_a"
    assert [item.id for item in manager.get_account_candidates("colab", "round_robin")] == ["acc_a", "acc_b"]
    manager.configs["colab"]["variables"]["split_across_accounts"] = True
    assert manager.select_account("colab", "round_robin").id == "acc_a"
    assert manager.select_account("colab", "round_robin").id == "acc_b"


@pytest.mark.parametrize("stream", [False, True])
def test_colab_cache_usage_reaches_logs_and_dashboard(monkeypatch, stream):
    plugin = ColabPlugin()
    model = plugin.metadata.models[0]
    acc = account()
    monkeypatch.setattr(plugin_manager, "plugins", {"colab": plugin})
    monkeypatch.setattr(plugin_manager, "accounts", {acc.id: acc})
    monkeypatch.setattr(plugin_manager, "configs", {"colab": {"enabled": True, "variables": {}}})
    monkeypatch.setattr(plugin_manager, "rules", {})
    monkeypatch.setattr(plugin_manager, "api_keys", {})
    monkeypatch.setattr(plugin_manager, "logs", [])
    monkeypatch.setattr(plugin_manager, "_save_accounts", lambda: None)
    monkeypatch.setattr(plugin_manager, "_schedule_logs_save", lambda: None)
    async def start(runtime, config):
        return {"capabilities": ["tools", "thinking"]}
    async def events(runtime, payload):
        yield response_data()
    monkeypatch.setattr(plugin, "_start", start)
    monkeypatch.setattr(plugin, "_ollama_events", events)
    client = TestClient(app, headers={"X-Admin-Token": "test-admin-token"})
    response = client.post("/v1/chat/completions", json={"model": model.id, "messages": [{"role": "user", "content": "ping"}], "stream": stream})
    assert response.status_code == 200
    if stream:
        assert "reasoning_content" in response.text and "cache_read_input_tokens" not in response.text
    else:
        assert response.json()["usage"]["cache_read_input_tokens"] == 80
    logs = client.get("/api/logs").json()
    assert logs[0]["prompt_tokens"] == 100 and logs[0]["cache_read_input_tokens"] == 80
    assert client.get("/api/stats").json()["cache_hit_rate"] == 80.0


class RawStream:
    def __init__(self, data=b""):
        self.data = bytearray(data)
        self.writes = []
        self.closed = False

    async def read(self, max_bytes, timeout):
        part = bytes(self.data[:min(max_bytes, 3)])  # force partial header/payload reads
        del self.data[:len(part)]
        return part

    async def write(self, data, timeout):
        self.writes.append(data)


def frame(opcode, payload, final=True):
    header = bytes([(128 if final else 0) | opcode])
    if len(payload) < 126:
        return header + bytes([len(payload)]) + payload
    return header + bytes([126]) + struct.pack("!H", len(payload)) + payload


def test_websocket_frames_fragmentation_ping_and_jupyter_binary():
    async def run():
        message = json.dumps({"channel": "iopub", "content": {"text": "hello"}}).encode()
        raw = RawStream(frame(1, message[:10], False) + frame(9, b"ping") + frame(0, message[10:]))
        response = httpx.Response(101, extensions={"network_stream": raw})
        socket = _KernelSocket(response, 1)
        assert (await socket.receive_json())["content"]["text"] == "hello"
        assert raw.writes[0][0] == 0x8A and raw.writes[0][1] & 0x80  # client pong masked
        binary = struct.pack("!II", 1, 8) + message
        raw.data.extend(frame(2, binary))
        assert (await socket.receive_json())["channel"] == "iopub"
        await socket.send_json({"content": "x" * 200})
        sent = raw.writes[-1]
        assert sent[0] == 0x81 and sent[1] == (0x80 | 126)
        size = struct.unpack("!H", sent[2:4])[0]
        mask = sent[4:8]
        decoded = bytes(value ^ mask[index % 4] for index, value in enumerate(sent[8:]))
        assert len(decoded) == size and json.loads(decoded)["content"] == "x" * 200
        await socket.aclose()
    asyncio.run(run())


def test_websocket_upgrade_validates_accept_and_uses_header_token():
    async def run():
        raw = RawStream()
        def handle(request):
            expected = base64.b64encode(hashlib.sha1((request.headers["sec-websocket-key"] + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()).decode()
            return httpx.Response(101, headers={"connection": "Upgrade", "upgrade": "websocket", "sec-websocket-accept": expected},
                                  extensions={"network_stream": raw})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            socket = await _KernelSocket.connect(client, "https://owned.googleusercontent.com/api/kernels/k/channels?session_id=s",
                                                  {"X-Colab-Runtime-Proxy-Token": "private-token"}, 1)
            await socket.aclose()
    asyncio.run(run())


def test_websocket_over_httpx_real_upgrade_and_trailing_bytes():
    """MockTransport can't prove httpx's upgraded stream actually works."""
    async def run():
        received = asyncio.Event()
        async def server(reader, writer):
            try:
                headers = await reader.readuntil(b"\r\n\r\n")
                text = headers.decode("ascii")
                key = next(line.split(":", 1)[1].strip() for line in text.split("\r\n") if line.lower().startswith("sec-websocket-key:"))
                assert "x-colab-runtime-proxy-token: private-token" in text.lower()
                expected = base64.b64encode(hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()).decode()
                message = json.dumps({"channel": "iopub", "content": {"text": "network"}}).encode()
                # Send WebSocket data in the same packet as the upgrade response.
                writer.write(("HTTP/1.1 101 Switching Protocols\r\nConnection: Upgrade\r\nUpgrade: websocket\r\nSec-WebSocket-Accept: " + expected + "\r\n\r\n").encode() + frame(1, message))
                await writer.drain()
                header = await reader.readexactly(2)
                assert header == bytes([0x81, 0x80 | len(b'{"ping": true}')])
                mask = await reader.readexactly(4)
                data = await reader.readexactly(header[1] & 127)
                decoded = bytes(value ^ mask[index % 4] for index, value in enumerate(data))
                assert json.loads(decoded) == {"ping": True}
                received.set()
            finally:
                writer.close()
                await writer.wait_closed()
        listener = await asyncio.start_server(server, "127.0.0.1", 0)
        port = listener.sockets[0].getsockname()[1]
        try:
            async with httpx.AsyncClient() as client:
                socket = await _KernelSocket.connect(client, f"http://127.0.0.1:{port}/api/kernels/k/channels",
                                                      {"X-Colab-Runtime-Proxy-Token": "private-token"}, 2)
                assert (await socket.receive_json())["content"]["text"] == "network"
                await socket.send_json({"ping": True})
                await asyncio.wait_for(received.wait(), 2)
                await socket.aclose()
        finally:
            listener.close()
            await listener.wait_closed()
    asyncio.run(run())


def test_remote_bridge_is_syntax_valid_and_passes_json_not_shell_code(monkeypatch):
    namespace = {}
    exec(compile(_REMOTE_SOURCE, "<colab-bridge>", "exec"), namespace)
    seen = []
    class Reply(io.BytesIO):
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.close()
    def api(path, payload=None, timeout=30):
        seen.append((path, payload))
        return Reply(json.dumps(response_data()).encode())
    namespace["api"] = api
    payload = {"model": "qwen3:0.6b", "messages": [{"role": "user", "content": "'); __import__('os').system('danger'); #"}], "stream": False}
    output = io.StringIO()
    with redirect_stdout(output):
        namespace["chat"](json.dumps(payload), "req", 3)
    assert seen == [("/api/chat", payload)]
    assert json.loads(output.getvalue()[len(_RESULT_PREFIX):])["data"]["prompt_eval_cached_count"] == 80
