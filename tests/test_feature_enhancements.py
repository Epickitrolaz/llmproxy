import time
import pytest
from fastapi.testclient import TestClient

from llmproxy.main import app
from llmproxy.core.manager import plugin_manager
from llmproxy.schemas import (
    Account,
    AccountCreate,
    ChatMessage,
    ChatCompletionRequest,
    LoadBalancingRuleCreate,
    PluginModelConfig,
)
from plugins.codex import CodexPlugin
from plugins.antigravity import AntigravityPlugin

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_state():
    """Ensure clean isolated state around tests."""
    original_keys = dict(plugin_manager.api_keys)
    original_rules = dict(plugin_manager.rules)
    original_accounts = dict(plugin_manager.accounts)
    original_logs = list(plugin_manager.logs)

    plugin_manager.api_keys.clear()
    plugin_manager.rules.clear()
    plugin_manager._save_api_keys()
    plugin_manager._save_rules()

    yield

    plugin_manager.api_keys = original_keys
    plugin_manager.rules = original_rules
    plugin_manager.accounts = original_accounts
    plugin_manager.logs = original_logs
    plugin_manager._save_api_keys()
    plugin_manager._save_rules()
    plugin_manager._save_accounts()
    plugin_manager._save_logs()


def test_account_pin_toggle():
    """Verify POST /api/accounts/{account_id}/pin toggles account pinned status."""
    acc = plugin_manager.add_account(
        AccountCreate(
            provider_id="codex",
            name="Primary Account",
            api_key="test-key-123",
            pinned=False,
        )
    )
    assert acc.pinned is False

    # Toggle to pinned
    res = client.post(f"/api/accounts/{acc.id}/pin")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["pinned"] is True
    assert plugin_manager.get_account(acc.id).pinned is True

    # Toggle to unpinned
    res2 = client.post(f"/api/accounts/{acc.id}/pin")
    assert res2.status_code == 200
    assert res2.json()["pinned"] is False
    assert plugin_manager.get_account(acc.id).pinned is False

    # Cleanup
    plugin_manager.delete_account(acc.id)


def test_temporary_api_key_lifecycle_and_cutoff():
    """Verify GET /api/keys/temporary creates a temporary key expiring in 30min / 10 requests."""
    res = client.get("/api/keys/temporary")
    assert res.status_code == 200
    key_data = res.json()
    assert key_data["is_temporary"] is True
    assert key_data["max_requests"] == 10
    assert key_data["expires_at"] > time.time() + 1700
    assert key_data["key"].startswith("sk-proxy-test-")

    key_id = key_data["id"]
    temp_key = key_data["key"]

    # Verify usage cut-off after 10 requests
    for i in range(10):
        plugin_manager.record_api_key_usage(key_id, prompt_tokens=5, completion_tokens=5)

    stored_key = plugin_manager.get_api_key(key_id)
    assert stored_key.total_requests == 10
    assert stored_key.is_valid() is False

    # 11th request with this key should be rejected with 401
    resp = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {temp_key}"},
        json={
            "model": "gemini-3.8-flash",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 401
    assert "expired or reached its maximum request limit" in resp.json()["detail"]["error"]["message"]

    # Cleanup
    plugin_manager.delete_api_key(key_id)


def test_custom_load_balancing_rules_add_to_active_models_stats():
    """Verify custom load balancing rules increase active_models_count on the dashboard."""
    stats_before = client.get("/api/stats").json()
    base_count = stats_before["active_models_count"]

    rule = plugin_manager.add_load_balancing_rule(
        LoadBalancingRuleCreate(
            model_id="test-hybrid-virtual-model",
            name="Hybrid Test Rule",
            enabled=True,
            strategy="round_robin",
            targets=["gemini-3.8-flash", "gpt-4o"],
        )
    )

    stats_after = client.get("/api/stats").json()
    assert stats_after["active_models_count"] == base_count + 1

    # Disable rule -> count decreases
    plugin_manager.update_load_balancing_rule(rule.id, {"enabled": False})
    stats_disabled = client.get("/api/stats").json()
    assert stats_disabled["active_models_count"] == base_count

    # Cleanup
    plugin_manager.delete_load_balancing_rule(rule.id)


def test_codex_reasoning_levels():
    """Verify Codex provider parses reasoning levels from request, model name, or variables."""
    codex = CodexPlugin()
    model_cfg = PluginModelConfig(id="gpt-6-luna", name="GPT-6 Luna", upstream_model="gpt-6-luna")

    # 1. Explicit request level: high
    req_high = ChatCompletionRequest(
        model="gpt-6-luna",
        messages=[ChatMessage(role="user", content="hi")],
        reasoning_effort="high",
    )
    assert codex._parse_reasoning_effort(req_high, model_cfg, {}) == "high"

    # 2. Explicit request level: low
    req_low = ChatCompletionRequest(
        model="gpt-6-luna",
        messages=[ChatMessage(role="user", content="hi")],
        reasoning_effort="low",
    )
    assert codex._parse_reasoning_effort(req_low, model_cfg, {}) == "low"

    # 3. Disabling reasoning altogether: none / off
    req_none = ChatCompletionRequest(
        model="gpt-6-luna",
        messages=[ChatMessage(role="user", content="hi")],
        reasoning_effort="none",
    )
    assert codex._parse_reasoning_effort(req_none, model_cfg, {}) == "none"

    req_off = ChatCompletionRequest(
        model="gpt-6-luna",
        messages=[ChatMessage(role="user", content="hi")],
        reasoning_effort="off",
    )
    assert codex._parse_reasoning_effort(req_off, model_cfg, {}) == "none"

    # 4. Model suffix parsing
    req_suffix_high = ChatCompletionRequest(
        model="gpt-6-luna:high",
        messages=[ChatMessage(role="user", content="hi")],
    )
    assert codex._parse_reasoning_effort(req_suffix_high, model_cfg, {}) == "high"

    req_suffix_none = ChatCompletionRequest(
        model="gpt-6-luna-none",
        messages=[ChatMessage(role="user", content="hi")],
    )
    assert codex._parse_reasoning_effort(req_suffix_none, model_cfg, {}) == "none"

    # 5. Default
    req_default = ChatCompletionRequest(
        model="gpt-6-luna",
        messages=[ChatMessage(role="user", content="hi")],
    )
    assert codex._parse_reasoning_effort(req_default, model_cfg, {}) == "default"


def test_antigravity_reasoning_levels():
    """Verify Antigravity provider parses reasoning levels and adjusts the upstream model name."""
    agy = AntigravityPlugin()
    model_cfg = PluginModelConfig(
        id="gemini-3.8-flash-high",
        name="Gemini 3.8 Flash High",
        upstream_model="gemini-3.8-flash-high",
    )

    # 1. Request reasoning_effort='high' -> gemini-3.8-flash-high, enabled=True
    req_high = ChatCompletionRequest(
        model="gemini-3.8-flash",
        messages=[ChatMessage(role="user", content="hi")],
        reasoning_effort="high",
    )
    actual_model, level, enabled = agy._parse_reasoning_level(req_high, model_cfg, {})
    assert actual_model == "gemini-3.8-flash-high"
    assert level == "high"
    assert enabled is True

    # 2. Request reasoning_effort='low' -> gemini-3.8-flash-low, enabled=True
    req_low = ChatCompletionRequest(
        model="gemini-3.8-flash",
        messages=[ChatMessage(role="user", content="hi")],
        reasoning_effort="low",
    )
    actual_model, level, enabled = agy._parse_reasoning_level(req_low, model_cfg, {})
    assert actual_model == "gemini-3.8-flash-low"
    assert level == "low"
    assert enabled is True

    # 3. Disabling reasoning altogether: 'none' -> gemini-3.8-flash, enabled=False
    req_none = ChatCompletionRequest(
        model="gemini-3.8-flash-high",
        messages=[ChatMessage(role="user", content="hi")],
        reasoning_effort="none",
    )
    actual_model, level, enabled = agy._parse_reasoning_level(req_none, model_cfg, {})
    assert actual_model == "gemini-3.8-flash"
    assert level == "none"
    assert enabled is False

    # 4. Model suffix: gemini-3.8-flash:low
    req_suffix = ChatCompletionRequest(
        model="gemini-3.8-flash:low",
        messages=[ChatMessage(role="user", content="hi")],
    )
    actual_model, level, enabled = agy._parse_reasoning_level(req_suffix, model_cfg, {})
    assert actual_model == "gemini-3.8-flash-low"
    assert level == "low"
    assert enabled is True

    # 5. Claude models in Antigravity: claude-opus-4-6-thinking
    claude_cfg = PluginModelConfig(
        id="claude-opus-4-6-thinking",
        name="Claude Opus 4.6 Thinking",
        upstream_model="claude-opus-4-6-thinking",
    )
    # Disabled altogether
    req_claude_none = ChatCompletionRequest(
        model="claude-opus-4-6-thinking",
        messages=[ChatMessage(role="user", content="hi")],
        reasoning_effort="none",
    )
    actual_model, level, enabled = agy._parse_reasoning_level(req_claude_none, claude_cfg, {})
    assert actual_model == "claude-opus-4-6"
    assert enabled is False

    # Enabled high
    req_claude_high = ChatCompletionRequest(
        model="claude-opus-4-6",
        messages=[ChatMessage(role="user", content="hi")],
        reasoning_effort="high",
    )
    actual_model, level, enabled = agy._parse_reasoning_level(req_claude_high, claude_cfg, {})
    assert actual_model == "claude-opus-4-6-thinking"
    assert enabled is True
