"""
Tests for Feature 1 (Example Response with Useful Details) and Feature 2 (New Codex Template & Auth Flow).
========================================================================================================
1. Feature 1:
   - Messaging models without an account configured returns the example response with useful details:
     - Original example response text preserved ("Leave that in")
     - Time taken (latency in ms)
     - Timestamp (UTC)
     - Model identifier
     - Token count breakdown
     - Account status ("No account configured (Simulation Mode)")
   - Verified for both non-streaming and streaming completions.
2. Feature 2:
   - The new Codex template is NOT an OpenAI compatible API wrapper.
   - Removed old OpenAI-compatible template and added authentic Codex template.
   - Supports authentic ChatGPT OAuth 2.0 PKCE flow (CODEX_CLIENT_ID, auth.openai.com).
   - Rich options matching and extending Antigravity:
     - load_balancing_strategy, account_switch_cooldown, account_switch_wait, max_retries, base_url
     - reasoning_effort (low, medium, high, xhigh, max, ultra, none, default)
     - sandbox_mode (read-only, workspace-write, danger-full-access)
     - approval_policy (never, on-request)
     - execution_mode (cli, api, auto)
   - Frontier models (gpt-6-luna, gpt-5.6-terra, gpt-5.6-luna, gpt-reserve, codex-auto-review, gpt-6-astra, gpt-6.1-sol)
   - Deterministic SHA-256 UUIDs and session ID derivation
   - Telemetry headers: ChatGPT-Account-Id, User-Agent: codex/cli/...
   - Local session auto-import from ~/.codex/auth.json
   - OAuth endpoints: /api/codex/oauth/start, /api/codex/oauth/status, /api/codex/oauth/callback, /api/codex/oauth/import-local, /api/codex/status
"""

import json
import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from llmproxy.main import app
from llmproxy.core.manager import plugin_manager
from llmproxy.core.codex_oauth import (
    CODEX_BACKEND_API,
    CODEX_CLIENT_ID,
    CODEX_SCOPES,
    CODEX_TOKEN_URI,
    codex_oauth,
    decode_jwt_payload,
)
from llmproxy.schemas import (
    Account,
    ChatMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
    PluginModelConfig,
)
from plugins.codex import CodexPlugin
from plugins.antigravity import AntigravityPlugin


client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_test_state():
    orig_accounts = dict(plugin_manager.accounts)
    orig_keys = dict(plugin_manager.api_keys)
    plugin_manager.accounts.clear()
    plugin_manager.api_keys.clear()
    yield
    plugin_manager.accounts = orig_accounts
    plugin_manager.api_keys = orig_keys
    plugin_manager._save_accounts()
    plugin_manager._save_api_keys()


# =========================================================================
# FEATURE 1: Useful Details in Example Responses Without Configured Account
# =========================================================================

def test_feature1_non_streaming_example_response_details_antigravity():
    """Verify Antigravity non-streaming response retains example text and includes time taken, timestamp, tokens, and account status."""
    plugin_manager.accounts.clear()
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "gemini-3.8-flash",
            "messages": [{"role": "user", "content": "Hello without account"}],
            "stream": False,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    content = data["choices"][0]["message"]["content"]

    # 1. Base example response must remain intact ("Leave that in")
    assert "Simulated response from 9Router Antigravity (gemini-3.8-flash)" in content

    # 2. Useful execution details must be present
    assert "Execution Details:" in content
    assert "Time taken:" in content
    assert "ms" in content
    assert "Timestamp:" in content
    assert "UTC" in content
    assert "gemini-3.8-flash" in content
    assert "Tokens:" in content
    assert "No account configured (Simulation Mode)" in content


def test_feature1_non_streaming_example_response_details_codex():
    """Verify Codex non-streaming response retains example text and includes time taken, timestamp, tokens, and account status."""
    plugin_manager.accounts.clear()
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-6-luna",
            "messages": [{"role": "user", "content": "Explain quantum computing briefly"}],
            "stream": False,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    content = data["choices"][0]["message"]["content"]

    # 1. Base example response preserved
    assert "Simulated response from Codex (gpt-6-luna)" in content

    # 2. Useful details included
    assert "Execution Details:" in content
    assert "Time taken:" in content
    assert "ms" in content
    assert "Timestamp:" in content
    assert "gpt-6-luna" in content
    assert "Tokens:" in content
    assert "No account configured (Simulation Mode)" in content


def test_feature1_streaming_example_response_details():
    """Verify streaming response retains example text and streams execution details."""
    plugin_manager.accounts.clear()
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "gpt-6-luna",
            "messages": [{"role": "user", "content": "Stream test without account"}],
            "stream": True,
        },
    ) as resp:
        assert resp.status_code == 200
        full_text = ""
        for line in resp.iter_lines():
            line = line.strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                chunk = json.loads(line[6:])
                delta = chunk["choices"][0]["delta"]
                if "content" in delta and delta["content"]:
                    full_text += delta["content"]

        # 1. Base example stream preserved
        assert "Simulated stream response from Codex for gpt-6-luna" in full_text

        # 2. Useful details streamed
        assert "Execution Details:" in full_text
        assert "Time taken:" in full_text
        assert "ms" in full_text
        assert "Timestamp:" in full_text
        assert "No account configured (Simulation Mode)" in full_text


def test_feature1_test_chat_playground_endpoint_details():
    """Verify POST /api/test-chat without configured account returns useful details."""
    plugin_manager.accounts.clear()
    resp = client.post(
        "/api/test-chat",
        json={
            "model": "gpt-6-luna",
            "prompt": "Test playground prompt",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "choices" in data
    content = data["choices"][0]["message"]["content"]
    assert "Simulated response from Codex (gpt-6-luna)" in content
    assert "Execution Details:" in content
    assert "Time taken:" in content
    assert "latency_ms" in data
    assert data["latency_ms"] >= 0.0


# =========================================================================
# FEATURE 2: New Codex Template & Updated Auth Flow
# =========================================================================

def test_feature2_codex_template_not_openai_compatible():
    """Verify the new Codex template is NOT an OpenAI compatible API wrapper and has options like Antigravity."""
    tpl_path = Path("templates/codex.py")
    assert tpl_path.exists()
    code = tpl_path.read_text(encoding="utf-8")

    # Should NOT be standard OpenAI chat/completions client
    assert "https://api.openai.com/v1" not in code
    assert "OpenAI Codex CLI & frontier models provider" in code

    # Should have ChatGPT OAuth / Codex credentials
    assert "CODEX_CLIENT_ID" in code
    assert "CODEX_BACKEND_API" in code
    assert "codex_oauth" in code
    assert "ChatGPT-Account-Id" in code

    # Options like Antigravity
    assert 'name="load_balancing_strategy"' in code
    assert 'name="account_switch_cooldown"' in code
    assert 'name="account_switch_wait"' in code
    assert 'name="max_retries"' in code
    assert 'name="reasoning_effort"' in code
    assert 'name="sandbox_mode"' in code
    assert 'name="approval_policy"' in code
    assert 'name="execution_mode"' in code

    # Frontier models
    assert "gpt-6-luna" in code
    assert "gpt-5.6-terra" in code
    assert "gpt-5.6-luna" in code
    assert "gpt-reserve" in code
    assert "codex-auto-review" in code


def test_feature2_codex_plugin_instantiation_and_metadata():
    """Verify CodexPlugin metadata, models, and variables."""
    plugin = CodexPlugin()
    meta = plugin.metadata
    assert meta.id == "codex"
    assert meta.name == "Codex"
    assert meta.version == "3.0.0"

    model_ids = {m.id for m in meta.models}
    assert "gpt-6-luna" in model_ids
    assert "gpt-5.6-terra" in model_ids
    assert "gpt-5.6-luna" in model_ids
    assert "gpt-reserve" in model_ids
    assert "codex-auto-review" in model_ids
    assert "gpt-4o" in model_ids

    var_names = {v.name for v in meta.variables}
    assert "timeout" in var_names
    assert "load_balancing_strategy" in var_names
    assert "account_switch_cooldown" in var_names
    assert "account_switch_wait" in var_names
    assert "max_retries" in var_names
    assert "reasoning_effort" in var_names
    assert "sandbox_mode" in var_names
    assert "approval_policy" in var_names


def test_feature2_codex_deterministic_uuid_and_session_id():
    """Verify deterministic SHA-256 UUID and session ID generation for Codex."""
    plugin = CodexPlugin()
    uuid1 = plugin.generate_sha256_uuid("session_seed_abc")
    uuid2 = plugin.generate_sha256_uuid("session_seed_abc")
    assert uuid1 == uuid2
    assert len(uuid1) == 36
    assert uuid1.count("-") == 4

    # Session ID generation
    sess_id = plugin.generate_session_id()
    assert len(sess_id) == 36


def test_feature2_codex_headers_building():
    """Verify Codex HTTP headers include official User-Agent, Bearer token, and ChatGPT-Account-Id."""
    plugin = CodexPlugin()
    account = Account(
        id="acc_codex_test",
        provider_id="codex",
        name="Test Codex Account",
        api_key="codex-jwt-access-token-xyz",
        extra_config={"account_id": "395202d6-cdfe-4094-8123-61a3f175d1c7"},
    )
    headers = plugin.build_headers(account=account)
    assert headers["Authorization"] == "Bearer codex-jwt-access-token-xyz"
    assert headers["ChatGPT-Account-Id"] == "395202d6-cdfe-4094-8123-61a3f175d1c7"
    assert "codex/cli" in headers["User-Agent"]


def test_feature2_codex_reasoning_effort_levels():
    """Verify reasoning levels including low, medium, high, xhigh, max, ultra, none."""
    codex = CodexPlugin()
    model_cfg = PluginModelConfig(id="gpt-6-luna", name="GPT-6 Luna", upstream_model="gpt-6-luna")

    # Standard levels
    assert codex._parse_reasoning_effort(
        ChatCompletionRequest(model="gpt-6-luna", messages=[ChatMessage(role="user", content="hi")], reasoning_effort="high"),
        model_cfg, {}
    ) == "high"

    assert codex._parse_reasoning_effort(
        ChatCompletionRequest(model="gpt-6-luna", messages=[ChatMessage(role="user", content="hi")], reasoning_effort="low"),
        model_cfg, {}
    ) == "low"

    # Frontier high reasoning levels
    assert codex._parse_reasoning_effort(
        ChatCompletionRequest(model="gpt-6-luna", messages=[ChatMessage(role="user", content="hi")], reasoning_effort="xhigh"),
        model_cfg, {}
    ) == "xhigh"

    assert codex._parse_reasoning_effort(
        ChatCompletionRequest(model="gpt-6-luna", messages=[ChatMessage(role="user", content="hi")], reasoning_effort="max"),
        model_cfg, {}
    ) == "max"

    assert codex._parse_reasoning_effort(
        ChatCompletionRequest(model="gpt-5.6-terra", messages=[ChatMessage(role="user", content="hi")], reasoning_effort="ultra"),
        model_cfg, {}
    ) == "ultra"

    # Suffixes
    assert codex._parse_reasoning_effort(
        ChatCompletionRequest(model="gpt-6-luna:max", messages=[ChatMessage(role="user", content="hi")]),
        model_cfg, {}
    ) == "max"

    assert codex._parse_reasoning_effort(
        ChatCompletionRequest(model="gpt-6-luna-none", messages=[ChatMessage(role="user", content="hi")]),
        model_cfg, {}
    ) == "none"


@pytest.mark.anyio
async def test_feature2_codex_local_session_import():
    """Verify auto-import of local ~/.codex/auth.json session in this environment."""
    acc = await codex_oauth.import_local_session()
    assert acc is not None
    assert acc.provider_id == "codex"
    assert acc.api_key != ""
    assert acc.extra_config.get("account_id") == "395202d6-cdfe-4094-8123-61a3f175d1c7"
    assert "k.kuczewski10@gmail.com" in acc.name or acc.extra_config.get("email") == "k.kuczewski10@gmail.com"


def test_feature2_codex_oauth_endpoints():
    """Verify Codex dashboard OAuth endpoints: start, status, and import-local."""
    # 1. Start OAuth Flow
    res_start = client.post("/api/codex/oauth/start")
    assert res_start.status_code == 200
    start_data = res_start.json()
    assert "flow_id" in start_data
    assert "auth_url" in start_data
    assert "client_id=" in start_data["auth_url"]
    assert "code_challenge=" in start_data["auth_url"]
    assert "code_challenge_method=S256" in start_data["auth_url"]

    flow_id = start_data["flow_id"]

    # 2. Check Flow Status
    res_status = client.get(f"/api/codex/oauth/status/{flow_id}")
    assert res_status.status_code == 200
    assert res_status.json()["status"] in ("waiting", "complete")

    # 3. Auto-Import Local Session
    res_import = client.post("/api/codex/oauth/import-local")
    assert res_import.status_code == 200
    import_data = res_import.json()
    assert import_data["status"] == "ok"
    assert import_data["account"]["provider_id"] == "codex"

    # 4. Check Codex status endpoint
    res_all_status = client.get("/api/codex/status")
    assert res_all_status.status_code == 200
    all_status = res_all_status.json()
    assert "accounts" in all_status
    assert len(all_status["accounts"]) >= 1


@pytest.mark.anyio
async def test_feature2_codex_cli_execution_live():
    """Verify real execution using official codex CLI installed in this environment."""
    plugin = CodexPlugin()
    model_cfg = next(m for m in plugin.metadata.models if m.id == "gpt-6-luna")

    # Import authenticated account
    account = await codex_oauth.import_local_session()
    assert account is not None

    req = ChatCompletionRequest(
        model="gpt-6-luna",
        messages=[ChatMessage(role="user", content="Calculate 9 * 9. Answer with only the number.")],
        stream=False,
    )

    resp = await plugin.chat_completion(req, model_cfg, {}, account=account)
    assert isinstance(resp, ChatCompletionResponse)
    assert resp.model == "gpt-6-luna"
    assert len(resp.choices) >= 1
    content = resp.choices[0].message.content
    assert "81" in content


# =========================================================================
# Additional Follow-Up Tests (URL detection, Codex scopes, test-model)
# =========================================================================

def test_codex_oauth_scopes_no_model_request():
    """Verify CODEX_SCOPES does not request illegal 'model.request' and matches official CLI."""
    assert "model.request" not in CODEX_SCOPES
    assert "openid" in CODEX_SCOPES
    assert "profile" in CODEX_SCOPES
    assert "email" in CODEX_SCOPES
    assert "offline_access" in CODEX_SCOPES

    flow = codex_oauth.start_flow()
    assert "model.request" not in flow["auth_url"]
    assert "id_token_add_organizations=true" in flow["auth_url"]
    assert "codex_cli_simplified_flow=true" in flow["auth_url"]
    assert "originator=codex_cli_rs" in flow["auth_url"]


def test_test_model_completion_non_streaming():
    """Verify 'test-model' returns test details without being in /models or provider settings."""
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Quick test"}],
            "stream": False,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "test-model"
    content = data["choices"][0]["message"]["content"]
    assert "Simulated response from LLMProxy (test-model)" in content
    assert "Execution Details:" in content
    assert "Time taken:" in content
    assert "test-model" in content
    assert "No account configured (Simulation Mode)" in content


def test_test_model_completion_streaming():
    """Verify 'test-model' streams test details via SSE."""
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Quick stream test"}],
            "stream": True,
        },
    ) as resp:
        assert resp.status_code == 200
        full_text = ""
        for line in resp.iter_lines():
            line = line.strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                chunk = json.loads(line[6:])
                delta = chunk["choices"][0]["delta"]
                if "content" in delta and delta["content"]:
                    full_text += delta["content"]

        assert "Simulated response from LLMProxy (test-model)" in full_text
        assert "Execution Details:" in full_text
        assert "test-model" in full_text


def test_test_model_not_in_models_list_or_provider_settings():
    """Verify 'test-model' is hidden from /v1/models and provider metadata."""
    models_resp = client.get("/v1/models")
    assert models_resp.status_code == 200
    listed_model_ids = {m["id"] for m in models_resp.json()["data"]}
    assert "test-model" not in listed_model_ids

    # GET /v1/models/test-model should return 404
    single_resp = client.get("/v1/models/test-model")
    assert single_resp.status_code == 404

    # None of the loaded plugins should have test-model in their models list
    for p in plugin_manager.plugins.values():
        plugin_model_ids = {m.id for m in p.metadata.models}
        assert "test-model" not in plugin_model_ids


def test_test_chat_supports_test_model():
    """Verify /api/test-chat endpoint handles test-model and returns test details."""
    resp = client.post(
        "/api/test-chat",
        json={
            "model": "test-model",
            "prompt": "Test from dashboard playground",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "test-model"
    content = data["choices"][0]["message"]["content"]
    assert "Simulated response from LLMProxy (test-model)" in content
    assert "Execution Details:" in content


def test_ui_dynamic_url_detection_and_test_model_default():
    """Verify index.html and app.js implement dynamic URL detection and default to test-model."""
    res_html = client.get("/static/index.html")
    assert res_html.status_code == 200
    html_text = res_html.text
    assert 'id="header-endpoint-display"' in html_text
    assert "window.location.origin" in html_text

    res_js = client.get("/static/app.js")
    assert res_js.status_code == 200
    js_text = res_js.text
    assert "getDetectedBaseUrl" in js_text
    assert "window.location.origin" in js_text
    assert 'test-model' in js_text
    assert "gemini-3.8-flash-high" not in js_text or "allModels.push(\"gemini-3.8-flash-high\")" not in js_text


def test_provider_menu_model_test_button():
    """Verify provider models menu includes a Test button with success and error visual states."""
    res_css = client.get("/static/style.css")
    assert res_css.status_code == 200
    css_text = res_css.text
    assert ".btn-test" in css_text
    assert ".btn-test.is-success" in css_text
    assert ".btn-test.is-error" in css_text

    res_js = client.get("/static/app.js")
    assert res_js.status_code == 200
    js_text = res_js.text
    assert "btn-fs-test-model" in js_text
    assert "is-success" in js_text
    assert "is-error" in js_text
    assert "/api/test-chat" in js_text


