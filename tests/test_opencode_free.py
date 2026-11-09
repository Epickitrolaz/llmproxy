"""
Tests for OpenCode Free Models Provider Plugin and Starter Template:
- Discovered in templates/ gallery and loadable via /api/templates/opencode_free/add
- Zero authentication / anonymous execution (no account or API key required)
- Automatic model exposure: scans all models and strictly filters for those ending with '-free'
- Manual model exposure: allows explicitly configured exposed models
- Rate limit handling and reset requirement documentation
- Limit bypassing controlled by ENABLE_LIMIT_BYPASS const variable (disabled by default)
- Session and project rotation when limit bypass is enabled
- Streaming SSE and non-streaming execution
"""

import asyncio
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import httpx
from fastapi.testclient import TestClient

from llmproxy.main import app
from llmproxy.core.manager import plugin_manager
from llmproxy.schemas import (
    ChatMessage,
    ChatCompletionRequest,
    PluginModelConfig,
)
from templates.opencode_free import (
    ENABLE_LIMIT_BYPASS,
    DEFAULT_FREE_MODELS,
    OpenCodeFreePlugin,
    generate_session_id,
    generate_request_id,
    generate_project_id,
    scan_upstream_models,
)


def test_const_limit_bypass_disabled_by_default():
    """Verify ENABLE_LIMIT_BYPASS is a const variable and disabled by default."""
    import templates.opencode_free as ocf
    assert hasattr(ocf, "ENABLE_LIMIT_BYPASS")
    assert ocf.ENABLE_LIMIT_BYPASS is False


def test_opencode_free_template_discovered_in_gallery():
    """Verify that the opencode_free template appears in /api/templates."""
    client = TestClient(app)
    res = client.get("/api/templates")
    assert res.status_code == 200
    templates = res.json()
    tpl_ids = {t["id"] for t in templates}
    assert "opencode_free" in tpl_ids

    oc_tpl = next(t for t in templates if t["id"] == "opencode_free")
    assert oc_tpl["name"] == "OpenCode Free"
    assert oc_tpl["filename"] == "opencode_free.py"
    assert "ENABLE_LIMIT_BYPASS" in oc_tpl["code"]
    assert "automatic" in oc_tpl["code"]
    assert "manual" in oc_tpl["code"]


def test_install_template_to_plugins(tmp_path, monkeypatch):
    """Verify installing opencode_free template adds it to active plugins."""
    temp_plugins = tmp_path / "plugins"
    temp_data = tmp_path / "data"
    temp_plugins.mkdir(parents=True, exist_ok=True)
    temp_data.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(plugin_manager, "plugins_dir", temp_plugins)
    monkeypatch.setattr(plugin_manager, "data_dir", temp_data)
    monkeypatch.setattr(plugin_manager, "plugins", {})
    monkeypatch.setattr(plugin_manager, "configs", {})

    client = TestClient(app)
    res_add = client.post("/api/templates/opencode_free/add")
    assert res_add.status_code == 200
    added = res_add.json()
    assert added["id"] == "opencode_free"
    assert (temp_plugins / "opencode_free.py").exists()

    # Provider list should now contain opencode_free
    res_list = client.get("/api/plugins")
    assert res_list.status_code == 200
    p_ids = [p["id"] for p in res_list.json()]
    assert "opencode_free" in p_ids


def test_zero_auth_no_credentials_required():
    """Verify zero authentication is required for opencode_free."""
    plugin = OpenCodeFreePlugin()
    # Variables validation passes without any API key or secret
    errors = plugin.validate_variables({})
    assert len(errors) == 0

    # Headers contain anonymous Authorization: Bearer public
    headers = plugin._build_upstream_headers(
        session_id="ses_123",
        project_id="global",
    )
    assert headers["Authorization"] == "Bearer public"
    assert "opencode" in headers["User-Agent"]
    assert headers["x-opencode-client"] == "cli"
    assert headers["x-opencode-session"] == "ses_123"
    assert headers["x-opencode-project"] == "global"
    assert headers["x-opencode-request"].startswith("msg_")


def test_automatic_model_exposure_filters_free_suffix():
    """Verify automatic mode scans models and strictly exposes ONLY those ending with -free."""
    mock_upstream_models = [
        {"id": "deepseek-v4-flash-free"},
        {"id": "gpt-4o"},                     # not free, should be excluded
        {"id": "minimax-m2.5-free"},
        {"id": "claude-3-7-sonnet"},           # not free, should be excluded
        {"id": "qwen3.6-plus-free"},
        {"id": "big-pickle"},                 # does not end with -free, excluded
        {"id": "nemotron-3-super-free"},
    ]

    def mock_handler(request: httpx.Request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"object": "list", "data": mock_upstream_models})
        return httpx.Response(404)

    transport = httpx.MockTransport(mock_handler)
    with patch("httpx.Client", return_value=httpx.Client(transport=transport)):
        scanned = scan_upstream_models("https://opencode.ai/zen/v1")
        scanned_ids = [m.id for m in scanned]

        # Verify only models ending with -free are retained
        assert "deepseek-v4-flash-free" in scanned_ids
        assert "minimax-m2.5-free" in scanned_ids
        assert "qwen3.6-plus-free" in scanned_ids
        assert "nemotron-3-super-free" in scanned_ids
        assert "gpt-4o" not in scanned_ids
        assert "claude-3-7-sonnet" not in scanned_ids
        assert "big-pickle" not in scanned_ids

        # Ensure all exposed models end with -free
        for mid in scanned_ids:
            assert mid.endswith("-free")


def test_automatic_model_exposure_on_plugin(monkeypatch):
    """Verify plugin.metadata in automatic mode dynamically exposes -free models."""
    plugin = OpenCodeFreePlugin()
    plugin.set_exposure_mode("automatic")

    mock_models = [
        PluginModelConfig(id="m1-free", name="M1 Free"),
        PluginModelConfig(id="m2-free", name="M2 Free"),
    ]
    monkeypatch.setattr(plugin, "scan_models", lambda base_url=None: mock_models)

    models = plugin.get_exposed_models(exposure_mode="automatic")
    assert [m.id for m in models] == ["m1-free", "m2-free"]

    # Also reflected in metadata.models
    assert [m.id for m in plugin.metadata.models] == ["m1-free", "m2-free"]


def test_manual_model_exposure_allows_custom_models():
    """Verify manual mode allows setting and exposing specific models."""
    plugin = OpenCodeFreePlugin()
    plugin.set_exposure_mode("manual")

    # 1. Via manual_models variable string
    exposed = plugin.get_exposed_models(
        exposure_mode="manual",
        variables={"manual_models": "deepseek-v4-flash-free, custom-enterprise-model, special-preview"},
    )
    exposed_ids = [m.id for m in exposed]
    assert exposed_ids == ["deepseek-v4-flash-free", "custom-enterprise-model", "special-preview"]

    # 2. Via programmatic set_manual_models
    plugin.set_manual_models(["model-alpha", "model-beta"])
    exposed_prog = plugin.get_exposed_models(exposure_mode="manual")
    assert [m.id for m in exposed_prog] == ["model-alpha", "model-beta"]
    assert [m.id for m in plugin.metadata.models] == ["model-alpha", "model-beta"]


@pytest.mark.anyio
async def test_non_streaming_upstream_success():
    """Verify successful non-streaming execution against OpenCode wire protocol."""
    plugin = OpenCodeFreePlugin()

    mock_response_payload = {
        "id": "chatcmpl-oc-test123",
        "object": "chat.completion",
        "created": 1779000000,
        "model": "deepseek-v4-flash-free",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello from OpenCode free tier!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
    }

    recorded_headers = {}

    def mock_handler(request: httpx.Request):
        nonlocal recorded_headers
        recorded_headers = dict(request.headers)
        return httpx.Response(200, json=mock_response_payload)

    transport = httpx.MockTransport(mock_handler)
    orig_client = httpx.AsyncClient
    with patch("httpx.AsyncClient", side_effect=lambda **kwargs: orig_client(transport=transport, **kwargs)):
        req = ChatCompletionRequest(
            model="deepseek-v4-flash-free",
            messages=[ChatMessage(role="user", content="hello")],
            stream=False,
        )
        model_cfg = PluginModelConfig(id="deepseek-v4-flash-free", name="DeepSeek Free")
        res = await plugin.chat_completion(req, model_cfg, variables={"base_url": "https://opencode.ai/zen/v1"})

        assert res.id == "chatcmpl-oc-test123"
        assert res.choices[0].message.content == "Hello from OpenCode free tier!"
        assert recorded_headers["authorization"] == "Bearer public"
        assert recorded_headers["x-opencode-client"] == "cli"
        assert "ses_" in recorded_headers["x-opencode-session"]


@pytest.mark.anyio
async def test_streaming_upstream_success():
    """Verify successful streaming SSE execution against OpenCode wire protocol."""
    plugin = OpenCodeFreePlugin()

    sse_lines = [
        'data: {"id":"chatcmpl-s1","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n\n',
        'data: {"id":"chatcmpl-s1","choices":[{"index":0,"delta":{"content":"Hello "},"finish_reason":null}]}\n\n',
        'data: {"id":"chatcmpl-s1","choices":[{"index":0,"delta":{"content":"world!"},"finish_reason":null}]}\n\n',
        'data: [DONE]\n\n',
    ]

    def mock_handler(request: httpx.Request):
        return httpx.Response(200, text="".join(sse_lines))

    transport = httpx.MockTransport(mock_handler)
    orig_client = httpx.AsyncClient
    with patch("httpx.AsyncClient", side_effect=lambda **kwargs: orig_client(transport=transport, **kwargs)):
        req = ChatCompletionRequest(
            model="minimax-m2.5-free",
            messages=[ChatMessage(role="user", content="stream test")],
            stream=True,
        )
        model_cfg = PluginModelConfig(id="minimax-m2.5-free", name="MiniMax Free")
        stream_gen = await plugin.chat_completion(req, model_cfg, variables={})

        chunks = []
        async for chunk in stream_gen:
            chunks.append(chunk)

        assert len(chunks) == 3
        full_content = "".join(c.choices[0].delta.get("content", "") for c in chunks)
        assert full_content == "Hello world!"


@pytest.mark.anyio
async def test_rate_limit_without_bypass_raises_429():
    """Verify that when ENABLE_LIMIT_BYPASS is False, 429 raises limit error with reset details."""
    import templates.opencode_free as ocf
    ocf.ENABLE_LIMIT_BYPASS = False

    plugin = OpenCodeFreePlugin()

    def mock_handler(request: httpx.Request):
        return httpx.Response(
            429,
            json={"type": "FreeUsageLimitError", "message": "Rate limit exceeded. Please try again later."},
        )

    transport = httpx.MockTransport(mock_handler)
    orig_client = httpx.AsyncClient
    with patch("httpx.AsyncClient", side_effect=lambda **kwargs: orig_client(transport=transport, **kwargs)):
        req = ChatCompletionRequest(
            model="deepseek-v4-flash-free",
            messages=[ChatMessage(role="user", content="hi")],
            stream=False,
        )
        model_cfg = PluginModelConfig(id="deepseek-v4-flash-free", name="DeepSeek Free")

        with pytest.raises(RuntimeError) as exc_info:
            await plugin.chat_completion(req, model_cfg, variables={"bypass_rate_limits": False})

        assert "OpenCode Free Tier Limit Exceeded (HTTP 429)" in str(exc_info.value)
        assert "Standard reset requires waiting for quota window cooldown" in str(exc_info.value)


@pytest.mark.anyio
async def test_rate_limit_bypass_rotates_session_and_retries_non_streaming():
    """Verify that when ENABLE_LIMIT_BYPASS is True, session is rotated and request retries successfully."""
    import templates.opencode_free as ocf
    ocf.ENABLE_LIMIT_BYPASS = True

    plugin = OpenCodeFreePlugin()
    attempts = 0
    sessions_seen = []

    def mock_handler(request: httpx.Request):
        nonlocal attempts
        attempts += 1
        sessions_seen.append(request.headers.get("x-opencode-session"))

        if attempts == 1:
            # First attempt hits 429 FreeUsageLimitError
            return httpx.Response(
                429,
                json={"type": "FreeUsageLimitError", "message": "Rate limit exceeded"},
            )
        # Second attempt after session rotation succeeds
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-bypassed",
                "object": "chat.completion",
                "model": "deepseek-v4-flash-free",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "Bypass success!"}}],
            },
        )

    transport = httpx.MockTransport(mock_handler)
    orig_client = httpx.AsyncClient
    with patch("httpx.AsyncClient", side_effect=lambda **kwargs: orig_client(transport=transport, **kwargs)):
        req = ChatCompletionRequest(
            model="deepseek-v4-flash-free",
            messages=[ChatMessage(role="user", content="bypass test")],
            stream=False,
        )
        model_cfg = PluginModelConfig(id="deepseek-v4-flash-free", name="DeepSeek Free")

        res = await plugin.chat_completion(req, model_cfg, variables={})
        assert res.id == "chatcmpl-bypassed"
        assert res.choices[0].message.content == "Bypass success!"
        assert attempts == 2
        # Verify session ID was rotated between attempt 1 and attempt 2
        assert len(sessions_seen) == 2
        assert sessions_seen[0] != sessions_seen[1]

    # Reset constant
    ocf.ENABLE_LIMIT_BYPASS = False


@pytest.mark.anyio
async def test_rate_limit_bypass_rotates_session_and_retries_streaming():
    """Verify streaming limit bypass on 429 rotates session and yields chunks."""
    import templates.opencode_free as ocf
    ocf.ENABLE_LIMIT_BYPASS = True

    plugin = OpenCodeFreePlugin()
    stream_attempts = 0

    def mock_handler(request: httpx.Request):
        nonlocal stream_attempts
        stream_attempts += 1
        if stream_attempts == 1:
            return httpx.Response(429, text='{"type":"FreeUsageLimitError","message":"rate limit"}')

        sse_data = (
            'data: {"id":"c1","choices":[{"index":0,"delta":{"content":"Bypassed stream!"}}]}\n\n'
            'data: [DONE]\n\n'
        )
        return httpx.Response(200, text=sse_data)

    transport = httpx.MockTransport(mock_handler)
    orig_client = httpx.AsyncClient
    with patch("httpx.AsyncClient", side_effect=lambda **kwargs: orig_client(transport=transport, **kwargs)):
        req = ChatCompletionRequest(
            model="qwen3.6-plus-free",
            messages=[ChatMessage(role="user", content="stream bypass")],
            stream=True,
        )
        model_cfg = PluginModelConfig(id="qwen3.6-plus-free", name="Qwen Free")
        stream_gen = await plugin.chat_completion(req, model_cfg, variables={})

        chunks = [c async for c in stream_gen]
        assert len(chunks) == 1
        assert chunks[0].choices[0].delta.get("content") == "Bypassed stream!"
        assert stream_attempts == 2

    ocf.ENABLE_LIMIT_BYPASS = False


@pytest.mark.anyio
async def test_simulation_mode_with_test_key():
    """Verify offline simulation mode triggers when test API key is provided."""
    plugin = OpenCodeFreePlugin()
    req = ChatCompletionRequest(
        model="deepseek-v4-flash-free",
        messages=[ChatMessage(role="user", content="sim test")],
        stream=False,
    )
    model_cfg = PluginModelConfig(id="deepseek-v4-flash-free", name="DeepSeek Free")

    # Non-streaming simulation
    res = await plugin.chat_completion(req, model_cfg, variables={"api_key": "test-key-mock"})
    assert "Simulated response from OpenCode Free Models" in res.choices[0].message.content
    assert res.choices[0].finish_reason == "stop"

    # Streaming simulation
    req_stream = ChatCompletionRequest(
        model="deepseek-v4-flash-free",
        messages=[ChatMessage(role="user", content="sim test stream")],
        stream=True,
    )
    stream_gen = await plugin.chat_completion(req_stream, model_cfg, variables={"api_key": "test-key-mock"})
    chunks = [c async for c in stream_gen]
    assert len(chunks) > 0
    full_text = "".join(c.choices[0].delta.get("content", "") for c in chunks)
    assert "Simulated OpenCode Free response" in full_text
