"""
Tests for OpenCode Free Models Provider Plugin and Starter Template:
- Discovered in templates/ gallery and loadable via /api/templates/opencode_free/add
- Zero authentication / anonymous execution (no account or API key required)
- Model list managed via Models tab (not in configuration variables)
- Automatic model exposure mode removed
- Outbound proxy configuration support
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
)


def test_const_limit_bypass_disabled_by_default():
    """Verify ENABLE_LIMIT_BYPASS is a const variable and disabled by default."""
    import templates.opencode_free as ocf
    assert hasattr(ocf, "ENABLE_LIMIT_BYPASS")
    assert ocf.ENABLE_LIMIT_BYPASS is False


def test_opencode_free_template_discovered_in_gallery():
    """Verify that the opencode_free template appears in /api/templates without model config variables."""
    client = TestClient(app, headers={"X-Admin-Token": "test-admin-token"})
    res = client.get("/api/templates")
    assert res.status_code == 200
    templates = res.json()
    tpl_ids = {t["id"] for t in templates}
    assert "opencode_free" in tpl_ids

    oc_tpl = next(t for t in templates if t["id"] == "opencode_free")
    assert oc_tpl["name"] == "OpenCode Free"
    assert oc_tpl["filename"] == "opencode_free.py"
    assert "ENABLE_LIMIT_BYPASS" in oc_tpl["code"]
    # Model exposure modes must not be in variables/configuration
    assert 'name="model_exposure"' not in oc_tpl["code"]
    assert 'name="manual_models"' not in oc_tpl["code"]


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

    client = TestClient(app, headers={"X-Admin-Token": "test-admin-token"})
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
    assert headers["x-opencode-session-id"] == "ses_123"
    assert headers["x-session-affinity"] == "ses_123"
    assert headers["x-session-id"] == "ses_123"
    assert headers["x-opencode-project"] == "global"
    assert headers["x-opencode-request"].startswith("msg_")


def test_models_managed_via_models_tab(tmp_path, monkeypatch):
    """Verify models are managed directly via Models tab, not in configuration variables."""
    plugin = OpenCodeFreePlugin()

    # 1. Variables do NOT contain model_exposure or manual_models
    var_names = {v.name for v in plugin.metadata.variables}
    assert "model_exposure" not in var_names
    assert "manual_models" not in var_names
    assert "proxy" in var_names

    # 2. Default models are populated in metadata.models
    default_ids = {m.id for m in plugin.metadata.models}
    assert "mimo-v2.6-flash-free" in default_ids
    assert "deepseek-v4.1-flash-free" in default_ids
    assert "glm-5.3-flash-free" in default_ids
    assert "big-pickle" in default_ids

    # 3. Models tab integration: adding and editing custom models in plugin_manager
    temp_plugins = tmp_path / "plugins"
    temp_data = tmp_path / "data"
    temp_plugins.mkdir(parents=True, exist_ok=True)
    temp_data.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(plugin_manager, "plugins_dir", temp_plugins)
    monkeypatch.setattr(plugin_manager, "data_dir", temp_data)
    monkeypatch.setattr(plugin_manager, "plugins", {"opencode_free": plugin})
    monkeypatch.setattr(plugin_manager, "configs", {})

    # User adds a new model in Models tab
    plugin_manager.add_custom_model(
        "opencode_free",
        PluginModelConfig(
            id="my-custom-model",
            name="My Custom Model",
            upstream_model="mimo-v2.6-flash-free",
        ),
    )
    active_models = plugin_manager.get_plugin_models("opencode_free")
    active_ids = {m.id for m in active_models}
    assert "my-custom-model" in active_ids
    assert "mimo-v2.6-flash-free" in active_ids


def test_proxy_configuration_passed_to_client():
    """Verify outbound proxy URL is configured and passed to httpx client."""
    plugin = OpenCodeFreePlugin()

    # Empty proxy returns no proxy in kwargs
    kwargs_no_proxy = plugin._get_client_kwargs(timeout=30.0, variables={})
    assert "proxy" not in kwargs_no_proxy
    assert kwargs_no_proxy["timeout"] == 30.0

    # Configured proxy is included in client kwargs
    kwargs_with_proxy = plugin._get_client_kwargs(
        timeout=45.0,
        variables={"proxy": "http://127.0.0.1:8888"},
    )
    assert kwargs_with_proxy["proxy"] == "http://127.0.0.1:8888"
    assert kwargs_with_proxy["timeout"] == 45.0


@pytest.mark.anyio
async def test_non_streaming_upstream_success():
    """Verify successful non-streaming execution against OpenCode wire protocol."""
    plugin = OpenCodeFreePlugin()

    mock_response_payload = {
        "id": "chatcmpl-oc-test123",
        "object": "chat.completion",
        "created": 1779000000,
        "model": "mimo-v2.6-flash-free",
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
            model="mimo-v2.6-flash-free",
            messages=[ChatMessage(role="user", content="hello")],
            stream=False,
        )
        model_cfg = PluginModelConfig(id="mimo-v2.6-flash-free", name="MiMo Free")
        res = await plugin.chat_completion(req, model_cfg, variables={"base_url": "https://opencode.ai/inference/openai/v1"})

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
            model="deepseek-v4.1-flash-free",
            messages=[ChatMessage(role="user", content="stream test")],
            stream=True,
        )
        model_cfg = PluginModelConfig(id="deepseek-v4.1-flash-free", name="DeepSeek Free")
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
            model="mimo-v2.6-flash-free",
            messages=[ChatMessage(role="user", content="hi")],
            stream=False,
        )
        model_cfg = PluginModelConfig(id="mimo-v2.6-flash-free", name="MiMo Free")

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
                "model": "mimo-v2.6-flash-free",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "Bypass success!"}}],
            },
        )

    transport = httpx.MockTransport(mock_handler)
    orig_client = httpx.AsyncClient
    with patch("httpx.AsyncClient", side_effect=lambda **kwargs: orig_client(transport=transport, **kwargs)):
        req = ChatCompletionRequest(
            model="mimo-v2.6-flash-free",
            messages=[ChatMessage(role="user", content="bypass test")],
            stream=False,
        )
        model_cfg = PluginModelConfig(id="mimo-v2.6-flash-free", name="MiMo Free")

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
            model="glm-5.3-flash-free",
            messages=[ChatMessage(role="user", content="stream bypass")],
            stream=True,
        )
        model_cfg = PluginModelConfig(id="glm-5.3-flash-free", name="GLM Free")
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
        model="mimo-v2.6-flash-free",
        messages=[ChatMessage(role="user", content="sim test")],
        stream=False,
    )
    model_cfg = PluginModelConfig(id="mimo-v2.6-flash-free", name="MiMo Free")

    # Non-streaming simulation
    res = await plugin.chat_completion(req, model_cfg, variables={"api_key": "test-key-mock"})
    assert "Simulated response from OpenCode Free Models" in res.choices[0].message.content
    assert res.choices[0].finish_reason == "stop"

    # Streaming simulation
    req_stream = ChatCompletionRequest(
        model="mimo-v2.6-flash-free",
        messages=[ChatMessage(role="user", content="sim test stream")],
        stream=True,
    )
    stream_gen = await plugin.chat_completion(req_stream, model_cfg, variables={"api_key": "test-key-mock"})
    chunks = [c async for c in stream_gen]
    assert len(chunks) > 0
    full_text = "".join(c.choices[0].delta.get("content", "") for c in chunks)
    assert "Simulated OpenCode Free response" in full_text


def test_reasoning_variable_and_validation():
    """Verify reasoning effort is a plugin variable with validation."""
    plugin = OpenCodeFreePlugin()
    var = next(v for v in plugin.metadata.variables if v.name == "reasoning_effort")
    assert var.type == "select"
    assert var.default == "default"
    assert set(var.options) == {"default", "none", "low", "medium", "high", "xhigh", "max"}

    assert plugin.validate_variables({}) == []
    assert plugin.validate_variables({"reasoning_effort": "high"}) == []
    errors = plugin.validate_variables({"reasoning_effort": "extreme"})
    assert errors and "Reasoning effort must be one of" in errors[0]


def test_reasoning_suffix_and_plugin_default_in_payload():
    """Verify reasoning levels flow into the upstream payload and are stripped from the model."""
    plugin = OpenCodeFreePlugin()
    model = PluginModelConfig(
        id="deepseek-v4.1-flash-free",
        name="DeepSeek V4.1 Flash Free",
        upstream_model="deepseek-v4.1-flash-free",
    )

    # Plugin-level default applies when the request does not specify a level.
    default_payload = plugin._build_upstream_payload(
        ChatCompletionRequest(
            model=model.id,
            messages=[ChatMessage(role="user", content="hi")],
        ),
        model,
        {"reasoning_effort": "medium"},
    )
    assert default_payload["model"] == "deepseek-v4.1-flash-free"
    assert default_payload["reasoning_effort"] == "medium"

    # Model suffix wins over the plugin default and is removed from the model ID.
    suffix_payload = plugin._build_upstream_payload(
        ChatCompletionRequest(
            model="deepseek-v4.1-flash-free:high",
            messages=[ChatMessage(role="user", content="hi")],
        ),
        model,
        {"reasoning_effort": "medium"},
    )
    assert suffix_payload["model"] == "deepseek-v4.1-flash-free"
    assert suffix_payload["reasoning_effort"] == "high"

    # Explicit request-level reasoning_effort wins over the suffix.
    request_payload = plugin._build_upstream_payload(
        ChatCompletionRequest(
            model="deepseek-v4.1-flash-free:low",
            messages=[ChatMessage(role="user", content="hi")],
            reasoning_effort="max",
        ),
        model,
        {},
    )
    assert request_payload["model"] == "deepseek-v4.1-flash-free"
    assert request_payload["reasoning_effort"] == "max"

    # "none"/"off" disable reasoning explicitly instead of being dropped.
    off_payload = plugin._build_upstream_payload(
        ChatCompletionRequest(
            model=model.id,
            messages=[ChatMessage(role="user", content="hi")],
            reasoning_effort="off",
        ),
        model,
    )
    assert off_payload["reasoning_effort"] == "none"

    # No configured or requested level means no reasoning field is sent.
    plain_payload = plugin._build_upstream_payload(
        ChatCompletionRequest(
            model=model.id,
            messages=[ChatMessage(role="user", content="hi")],
        ),
        model,
    )
    assert "reasoning_effort" not in plain_payload


@pytest.mark.anyio
async def test_reasoning_effort_sent_upstream_non_streaming():
    """Verify the resolved reasoning level reaches the upstream request body."""
    plugin = OpenCodeFreePlugin()
    recorded_body = {}

    def mock_handler(request: httpx.Request):
        recorded_body.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-reasoning",
                "object": "chat.completion",
                "model": "deepseek-v4.1-flash-free",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "ok"}}
                ],
            },
        )

    transport = httpx.MockTransport(mock_handler)
    orig_client = httpx.AsyncClient
    with patch(
        "httpx.AsyncClient",
        side_effect=lambda **kwargs: orig_client(transport=transport, **kwargs),
    ):
        req = ChatCompletionRequest(
            model="deepseek-v4.1-flash-free:high",
            messages=[ChatMessage(role="user", content="reason")],
            stream=False,
        )
        model_cfg = PluginModelConfig(
            id="deepseek-v4.1-flash-free",
            name="DeepSeek Free",
        )
        await plugin.chat_completion(req, model_cfg, variables={})

    assert recorded_body["model"] == "deepseek-v4.1-flash-free"
    assert recorded_body["reasoning_effort"] == "high"

