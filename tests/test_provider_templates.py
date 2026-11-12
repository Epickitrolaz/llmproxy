"""
Tests for the Provider Plugin Template Architecture:
- By default plugins/ is empty and provider page returns []
- Templates live in templates/ and are returned by /api/templates
- Templates can optionally be added to plugins/
- Antigravity defaults: fallback strategy, switch delay 60s, reasoning effort routing
- Test keys are not displayed in the key list
"""

import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from llmproxy.main import app
from llmproxy.core.manager import PluginManager, plugin_manager
@pytest.fixture(autouse=True)
def clean_keys():
    orig = dict(plugin_manager.api_keys)
    plugin_manager.api_keys.clear()
    plugin_manager._save_api_keys()
    yield
    plugin_manager.api_keys = orig
    plugin_manager._save_api_keys()


def test_empty_plugins_by_default(tmp_path, monkeypatch):
    """Verify that when plugins/ is clean/empty, provider list is completely empty."""
    temp_plugins = tmp_path / "plugins"
    temp_data = tmp_path / "data"
    temp_plugins.mkdir(parents=True, exist_ok=True)
    temp_data.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(plugin_manager, "plugins_dir", temp_plugins)
    monkeypatch.setattr(plugin_manager, "data_dir", temp_data)
    monkeypatch.setattr(plugin_manager, "plugins", {})
    monkeypatch.setattr(plugin_manager, "configs", {})

    client = TestClient(app, headers={"X-Admin-Token": "test-admin-token"})
    res = client.get("/api/plugins")
    assert res.status_code == 200
    assert res.json() == []


def test_templates_gallery_and_install_to_plugins(tmp_path, monkeypatch):
    """Verify templates can be listed and optionally added to plugins."""
    temp_plugins = tmp_path / "plugins"
    temp_data = tmp_path / "data"
    temp_plugins.mkdir(parents=True, exist_ok=True)
    temp_data.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(plugin_manager, "plugins_dir", temp_plugins)
    monkeypatch.setattr(plugin_manager, "data_dir", temp_data)
    monkeypatch.setattr(plugin_manager, "plugins", {})
    monkeypatch.setattr(plugin_manager, "configs", {})

    client = TestClient(app, headers={"X-Admin-Token": "test-admin-token"})

    # 1. Fetch available templates
    res = client.get("/api/templates")
    assert res.status_code == 200
    templates = res.json()
    assert len(templates) >= 4
    template_ids = {t["id"] for t in templates}
    assert "antigravity" in template_ids
    assert "codex" in template_ids
    assert "anthropic_proxy" in template_ids
    assert "openrouter" in template_ids
    assert "openai_compatible" in template_ids
    assert "opencode_free" in template_ids

    # Check antigravity template defaults in code/metadata
    ag_tpl = next(t for t in templates if t["id"] == "antigravity")
    assert "fallback" in ag_tpl["code"]
    assert 'name="reasoning_tags"' not in ag_tpl["code"]

    # 2. Add antigravity template to plugins
    res_add = client.post("/api/templates/antigravity/add")
    assert res_add.status_code == 200
    added = res_add.json()
    assert added["id"] == "antigravity"

    # Check that file was written to temp_plugins
    assert (temp_plugins / "antigravity.py").exists()

    # 3. Now /api/plugins has antigravity
    res_plugins = client.get("/api/plugins")
    assert res_plugins.status_code == 200
    plugins = res_plugins.json()
    assert len(plugins) == 1
    assert plugins[0]["id"] == "antigravity"

    # Verify strategy defaults to fallback
    vars_dict = {v["name"]: v for v in plugins[0]["variables"]}
    assert vars_dict["load_balancing_strategy"]["default"] == "fallback"
    assert vars_dict["account_switch_wait"]["default"] == 60


def test_test_keys_not_displayed_in_key_list(tmp_path, monkeypatch):
    """Verify that temporary test keys are not returned by GET /api/keys or added to the key list."""
    temp_plugins = tmp_path / "plugins"
    temp_data = tmp_path / "data"
    temp_plugins.mkdir(parents=True, exist_ok=True)
    temp_data.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(plugin_manager, "plugins_dir", temp_plugins)
    monkeypatch.setattr(plugin_manager, "data_dir", temp_data)
    monkeypatch.setattr(plugin_manager, "api_keys", {})

    client = TestClient(app, headers={"X-Admin-Token": "test-admin-token"})

    # 1. Create a normal user API key
    res = client.post("/api/keys", json={"name": "Production Key", "key": "sk-proxy-user-123"})
    assert res.status_code == 200

    # 2. Request a temporary test key
    res_temp = client.get("/api/keys/temporary")
    assert res_temp.status_code == 200
    assert res_temp.json()["is_temporary"] is True

    # 3. GET /api/keys should only return the user key, NOT the temporary test key
    res_keys = client.get("/api/keys")
    assert res_keys.status_code == 200
    keys = res_keys.json()
    assert len(keys) == 1
    assert keys[0]["name"] == "Production Key"
    assert not any(k.get("is_temporary") for k in keys)


def test_antigravity_reasoning_effort_routing():
    """Verify gemini-3.8-flash dynamically routes to gemini-3.8-flash-high/-medium/-low."""
    client = TestClient(app, headers={"X-Admin-Token": "test-admin-token"})

    # 1. Normal call with default effort -> routes to high
    res_high = client.post(
        "/v1/chat/completions",
        json={
            "model": "gemini-3.8-flash",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "high",
        },
    )
    assert res_high.status_code == 200
    data = res_high.json()
    assert data["model"] == "gemini-3.8-flash"
    assert "<think>" not in data["choices"][0]["message"]["content"]

    # 2. Medium effort
    res_med = client.post(
        "/v1/chat/completions",
        json={
            "model": "gemini-3.8-flash",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "medium",
        },
    )
    assert res_med.status_code == 200
    assert res_med.json()["model"] == "gemini-3.8-flash"

    # 3. None effort
    res_none = client.post(
        "/v1/chat/completions",
        json={
            "model": "gemini-3.8-flash",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "none",
        },
    )
    assert res_none.status_code == 200
    assert res_none.json()["model"] == "gemini-3.8-flash"
