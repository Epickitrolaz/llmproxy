"""Tests verifying all fixes for GitHub Issue #28:
1. Model 'Test' button routing across accounts with priority > min_priority on failure/quota exhaustion.
2. Proxy restart does not re-enable disabled models or add back deleted models.
3. Account priority uniqueness rule per provider.
4. Redirect URL input matching full width.
5. Model test button status persisting until window closed.
6. Quota time left to reset supporting days, hours, and minutes.
7. Codex usage dashboard endpoint and meter reporting.
8. Provider configuration options (timeout) persisting across server restarts without resetting to blank.
"""

import asyncio
import json
import time
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from llmproxy.main import app
from llmproxy.core.manager import PluginManager, plugin_manager
from llmproxy.schemas import (
    Account,
    AccountCreate,
    AccountUpdate,
    ChatChoice,
    ChatMessage,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionUsage,
    PluginMetadata,
    PluginModelConfig,
    PluginVariable,
)
from llmproxy.dashboard.router import fetch_codex_quota


# =========================================================================
# Issue 1: Test chat routing and failover across accounts
# =========================================================================

@pytest.mark.asyncio
async def test_issue1_test_chat_failover_to_higher_priority_account(monkeypatch):
    """Verify /api/test-chat routes to account with priority > min_priority when priority 1 fails."""
    client = TestClient(app)

    # Clear existing accounts for mock_provider
    for a in list(plugin_manager.get_accounts("mock_provider")):
        plugin_manager.delete_account(a.id)

    # Create two accounts with priorities 1 and 2
    acc1 = plugin_manager.add_account(
        AccountCreate(
            provider_id="mock_provider",
            name="Mock Account 1 (Failing/Quota Exhausted)",
            api_key="key-1",
            priority=1,
            weight=1,
        )
    )
    acc2 = plugin_manager.add_account(
        AccountCreate(
            provider_id="mock_provider",
            name="Mock Account 2 (Backup)",
            api_key="key-2",
            priority=2,
            weight=1,
        )
    )

    # Ensure strategy is fallback / priority
    plugin_manager.update_plugin_config(
        "mock_provider",
        enabled=True,
        variables={"load_balancing_strategy": "fallback", "max_retries": 2},
    )

    plugin = plugin_manager.get_plugin("mock_provider")
    attempted_accounts = []

    async def mock_chat_completion(request, model_config, variables, account=None):
        acc_id = account.id if account else None
        attempted_accounts.append(acc_id)
        if acc_id == acc1.id:
            # Simulate 429 quota exhaustion on account 1
            raise RuntimeError("HTTP 429: rate limit / quota exceeded for account 1")
        return ChatCompletionResponse(
            id="chatcmpl-test-ok",
            model=request.model,
            choices=[ChatChoice(index=0, message=ChatMessage(role="assistant", content="OK from acc2"))],
            usage=CompletionUsage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
        )

    monkeypatch.setattr(plugin, "chat_completion", mock_chat_completion)

    resp = client.post(
        "/api/test-chat",
        json={"model": "mock-gpt-4o", "prompt": "ping", "max_tokens": 5},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["message"]["content"] == "OK from acc2"

    # Verify both accounts were attempted in order of priority (acc1 first, then acc2)
    assert attempted_accounts == [acc1.id, acc2.id]

    # Verify acc1 was marked rate limited and acc2 was marked successful
    assert acc1.rate_limited_until is not None
    assert acc2.success_count >= 1

    # Clean up
    plugin_manager.delete_account(acc1.id)
    plugin_manager.delete_account(acc2.id)


# =========================================================================
# Issue 2: Proxy restart preserves disabled and deleted models
# =========================================================================

def test_issue2_proxy_restart_preserves_disabled_and_deleted_models(tmp_path):
    """Verify that a proxy restart does not re-enable disabled models or restore deleted models."""
    plugins_dir = tmp_path / "plugins"
    data_dir = tmp_path / "data"
    plugins_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Create a plugin with 3 default models
    code = """
from llmproxy.core.interface import BasePlugin
from llmproxy.schemas import PluginMetadata, PluginModelConfig, PluginVariable

class MyPlugin(BasePlugin):
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            id="test_restart_provider",
            name="Restart Provider",
            enabled=True,
            filename="test_restart.py",
            models=[
                PluginModelConfig(id="model-active", name="Active Model", enabled=True),
                PluginModelConfig(id="model-to-disable", name="To Disable", enabled=True),
                PluginModelConfig(id="model-to-delete", name="To Delete", enabled=True),
            ]
        )
    async def chat_completion(self, *args, **kwargs):
        pass

plugin = MyPlugin()
"""
    (plugins_dir / "test_restart.py").write_text(code, encoding="utf-8")

    # Initial manager load
    mgr1 = PluginManager(plugins_dir=plugins_dir, data_dir=data_dir)
    mgr1.discover_and_load_all()

    models_initial = {m.id: m.enabled for m in mgr1.get_plugin_models("test_restart_provider")}
    assert "model-active" in models_initial
    assert "model-to-disable" in models_initial
    assert "model-to-delete" in models_initial

    # Disable model-to-disable
    mgr1.update_custom_model(
        "test_restart_provider",
        "model-to-disable",
        PluginModelConfig(id="model-to-disable", name="To Disable", enabled=False),
    )

    # Delete model-to-delete
    mgr1.delete_custom_model("test_restart_provider", "model-to-delete")

    # Verify state before restart
    models_before = {m.id: m.enabled for m in mgr1.get_plugin_models("test_restart_provider")}
    assert models_before["model-active"] is True
    assert models_before["model-to-disable"] is False
    assert "model-to-delete" not in models_before

    # Now simulate complete proxy restart with fresh manager instance reading same disk files
    mgr2 = PluginManager(plugins_dir=plugins_dir, data_dir=data_dir)
    mgr2.discover_and_load_all()

    models_after = {m.id: m.enabled for m in mgr2.get_plugin_models("test_restart_provider")}

    # Model to delete MUST NOT be restored
    assert "model-to-delete" not in models_after

    # Model to disable MUST NOT be re-enabled
    assert "model-to-disable" in models_after
    assert models_after["model-to-disable"] is False

    # Active model remains active
    assert models_after["model-active"] is True

    # Check active models exposed to API
    active_models = [m.id for m, _ in mgr2.get_all_active_models()]
    assert "model-active" in active_models
    assert "model-to-disable" not in active_models
    assert "model-to-delete" not in active_models


# =========================================================================
# Issue 3: Priority uniqueness rule per provider
# =========================================================================

def test_issue3_account_priority_uniqueness_per_provider():
    """Verify that two accounts connected to the same provider cannot share the same priority."""
    prov_id = "test_priority_provider"

    # Add account 1 with priority 1
    acc1 = plugin_manager.add_account(
        AccountCreate(
            provider_id=prov_id,
            name="Account A",
            api_key="key-a",
            priority=1,
        )
    )
    assert acc1.priority == 1

    # Add account 2 with priority 1 (conflict)
    acc2 = plugin_manager.add_account(
        AccountCreate(
            provider_id=prov_id,
            name="Account B",
            api_key="key-b",
            priority=1,
        )
    )

    # Priorities must be strictly different
    acc1_refreshed = plugin_manager.get_account(acc1.id)
    acc2_refreshed = plugin_manager.get_account(acc2.id)
    assert acc1_refreshed.priority != acc2_refreshed.priority

    # Updating account 2 to have priority equal to account 1 must be auto-adjusted
    plugin_manager.update_account(acc2.id, AccountUpdate(priority=acc1_refreshed.priority))
    acc1_after = plugin_manager.get_account(acc1.id)
    acc2_after = plugin_manager.get_account(acc2.id)
    assert acc1_after.priority != acc2_after.priority

    # Accounts belonging to different providers CAN share the same priority
    other_prov_acc = plugin_manager.add_account(
        AccountCreate(
            provider_id="other_provider",
            name="Other Account",
            api_key="other-key",
            priority=acc1_after.priority,
        )
    )
    assert other_prov_acc.priority == acc1_after.priority

    # Clean up
    plugin_manager.delete_account(acc1.id)
    plugin_manager.delete_account(acc2.id)
    plugin_manager.delete_account(other_prov_acc.id)


# =========================================================================
# Issue 4: Redirect URL input full width styling
# =========================================================================

def test_issue4_redirect_url_input_styling():
    """Verify Google and Codex OAuth callback paste inputs use flex: 1 to fill dialog width."""
    html = Path("llmproxy/static/index.html").read_text(encoding="utf-8")
    assert 'id="google-callback-url-paste"' in html
    assert 'id="codex-callback-url-paste"' in html

    # Look for flex: 1 styling on callback inputs
    for line in html.splitlines():
        if "google-callback-url-paste" in line or "codex-callback-url-paste" in line:
            assert "flex: 1" in line


# =========================================================================
# Issue 5: Model test button status does not reset automatically
# =========================================================================

def test_issue5_model_test_button_no_auto_reset():
    """Verify that model test button click handler does not reset state back with setTimeout."""
    js = Path("llmproxy/static/app.js").read_text(encoding="utf-8")
    # Verify the testBtn click block in app.js does not contain setTimeout to remove is-success / is-error
    assert "btn-fs-test-model" in js
    # The old pattern was setTimeout(..., 4000) inside finally block of testBtn
    assert "testBtn.textContent = \"Test\";\n                testBtn.title = \"Test model response\";" not in js


# =========================================================================
# Issue 6: Quota reset display shows time left supporting days
# =========================================================================

def test_issue6_quota_reset_display_supports_days_hours_minutes():
    """Verify formatTimeRemaining in app.js correctly computes time remaining with days."""
    js = Path("llmproxy/static/app.js").read_text(encoding="utf-8")
    assert "function formatTimeRemaining" in js
    assert "days > 0" in js
    assert "${days}d" in js
    assert "${hours}h" in js
    assert "${minutes}m" in js


# =========================================================================
# Issue 7: Codex usage dashboard meter and endpoint
# =========================================================================

@pytest.mark.asyncio
async def test_issue7_codex_usage_dashboard_quota_endpoint(monkeypatch):
    """Verify Codex usage quota endpoint fetches /wham/usage and parses used_percent."""
    import httpx

    acc = Account(
        id="acc_codex_test_usage",
        provider_id="codex",
        name="Codex (test@example.com)",
        api_key="test-access-token",
        base_url="https://chatgpt.com/backend-api",
        extra_config={"account_id": "chatgpt-acc-123", "email": "test@example.com", "plan_type": "free"},
    )

    mock_wham_response = {
        "user_id": "user-123",
        "account_id": "chatgpt-acc-123",
        "email": "test@example.com",
        "plan_type": "free",
        "rate_limit": {
            "allowed": True,
            "limit_reached": False,
            "primary_window": {
                "used_percent": 0,
                "limit_window_seconds": 18000,
                "reset_after_seconds": 12000,
                "reset_at": int(time.time() + 12000),
            },
            "secondary_window": {
                "used_percent": 25,
                "limit_window_seconds": 604800,
                "reset_after_seconds": 400000,
                "reset_at": int(time.time() + 400000),
            },
        },
    }

    async def mock_get(self, url, headers=None):
        class MockResp:
            status_code = 200
            text = json.dumps(mock_wham_response)
            def json(self):
                return mock_wham_response
        return MockResp()

    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

    quota = await fetch_codex_quota(acc)
    assert "groups" in quota
    group = quota["groups"][0]
    assert group["displayName"] == "Codex Usage"
    assert len(group["buckets"]) == 2

    # Primary window: 5-Hour Limit with usedPercent=0 (matches official 0%)
    pw = group["buckets"][0]
    assert pw["displayName"] == "5-Hour Limit"
    assert pw["usedPercent"] == 0
    assert pw["remainingFraction"] == 1.0
    assert pw["resetTime"] is not None

    # Secondary window: Weekly Limit with usedPercent=25
    sw = group["buckets"][1]
    assert sw["displayName"] == "Weekly Limit"
    assert sw["usedPercent"] == 25
    assert sw["remainingFraction"] == 0.75


# =========================================================================
# Issue 8: Antigravity provider options persist across restart
# =========================================================================

def test_issue8_antigravity_timeout_persists_across_restart(tmp_path):
    """Verify that Antigravity timeout variable retains default or user value and does not become blank on restart."""
    plugins_dir = Path("plugins")
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Initial manager load
    mgr1 = PluginManager(plugins_dir=plugins_dir, data_dir=data_dir)
    mgr1.discover_and_load_all()

    vars1 = mgr1.get_plugin_variables("antigravity")
    # Timeout must not be empty string or None
    assert vars1.get("timeout") == 60

    # User modifies timeout to 120
    mgr1.update_plugin_config("antigravity", variables={"timeout": 120})
    assert mgr1.get_plugin_variables("antigravity")["timeout"] == 120

    # Simulate restart
    mgr2 = PluginManager(plugins_dir=plugins_dir, data_dir=data_dir)
    mgr2.discover_and_load_all()

    vars2 = mgr2.get_plugin_variables("antigravity")
    assert vars2.get("timeout") == 120
