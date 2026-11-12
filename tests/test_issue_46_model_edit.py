"""Regression tests for issue #46: built-in model edits must take effect."""

import json

import pytest
from fastapi.testclient import TestClient

from llmproxy.core.manager import PluginManager, plugin_manager
from llmproxy.main import app
from llmproxy.schemas import PluginModelConfig


PLUGIN_ID = "issue46_provider"

PLUGIN_CODE = '''
from llmproxy.core.interface import BasePlugin
from llmproxy.schemas import PluginMetadata, PluginModelConfig


class Issue46Plugin(BasePlugin):
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            id="issue46_provider",
            name="Issue 46 Provider",
            version="1.0.0",
            models=[
                PluginModelConfig(
                    id="model-a",
                    name="Model A",
                    upstream_model="model-a-upstream",
                    context_window=200000,
                    max_tokens=64000,
                ),
                PluginModelConfig(id="model-b", name="Model B"),
            ],
        )

    async def chat_completion(self, request, model_config, variables, account=None):
        raise NotImplementedError

plugin = Issue46Plugin()
'''


def _write_plugin(plugins_dir):
    plugins_dir.mkdir(parents=True, exist_ok=True)
    (plugins_dir / f"{PLUGIN_ID}.py").write_text(PLUGIN_CODE, encoding="utf-8")


def _make_manager(tmp_path):
    plugins_dir = tmp_path / "plugins"
    data_dir = tmp_path / "data"
    _write_plugin(plugins_dir)
    manager = PluginManager(plugins_dir=plugins_dir, data_dir=data_dir)
    manager.discover_and_load_all()
    return manager, plugins_dir, data_dir


def _models_by_id(manager):
    return {model.id: model for model in manager.get_plugin_models(PLUGIN_ID)}


def test_delete_then_readd_builtin_persists_edit(tmp_path):
    """Delete + re-add of a built-in ID must persist the edited limits."""
    manager, plugins_dir, data_dir = _make_manager(tmp_path)

    assert manager.delete_custom_model(PLUGIN_ID, "model-a") is True
    assert "model-a" not in _models_by_id(manager)

    replacement = PluginModelConfig(
        id="model-a",
        name="Edited Model A",
        upstream_model="edited-upstream",
        context_window=512000,
        max_tokens=128000,
    )
    assert manager.add_custom_model(PLUGIN_ID, replacement) is True

    cfg = manager.configs[PLUGIN_ID]
    assert "model-a" in cfg["edited_models"]
    assert cfg["custom_models"] == []

    model = _models_by_id(manager)["model-a"]
    assert model.context_window == 512000
    assert model.max_tokens == 128000
    assert model.upstream_model == "edited-upstream"

    # Survives a plugin hot reload.
    manager.reload_plugin(PLUGIN_ID)
    model = _models_by_id(manager)["model-a"]
    assert model.context_window == 512000
    assert model.max_tokens == 128000

    # Survives a full restart with fresh manager state from disk.
    restarted = PluginManager(plugins_dir=plugins_dir, data_dir=data_dir)
    restarted.discover_and_load_all()
    model = _models_by_id(restarted)["model-a"]
    assert model.context_window == 512000
    assert model.max_tokens == 128000
    assert model.upstream_model == "edited-upstream"
    assert restarted.configs[PLUGIN_ID]["custom_models"] == []


def test_stale_custom_collision_is_migrated_to_edit(tmp_path):
    """A custom entry with a built-in ID is cleaned up without shadowing the model."""
    plugins_dir = tmp_path / "plugins"
    data_dir = tmp_path / "data"
    _write_plugin(plugins_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    stale = PluginModelConfig(
        id="model-a",
        name="Stale Custom A",
        upstream_model="stale-upstream",
        context_window=512000,
        max_tokens=128000,
    )
    config = {
        PLUGIN_ID: {
            "enabled": True,
            "variables": {},
            "custom_models": [stale.model_dump()],
            "edited_models": {},
            "deleted_models": [],
        }
    }
    (data_dir / "plugin_configs.json").write_text(json.dumps(config), encoding="utf-8")

    manager = PluginManager(plugins_dir=plugins_dir, data_dir=data_dir)
    manager.discover_and_load_all()

    models = manager.get_plugin_models(PLUGIN_ID)
    matches = [model for model in models if model.id == "model-a"]
    assert len(matches) == 1
    assert matches[0].context_window == 512000
    assert matches[0].max_tokens == 128000
    assert manager.configs[PLUGIN_ID]["custom_models"] == []
    assert "model-a" in manager.configs[PLUGIN_ID]["edited_models"]

    # The cleanup is persisted, so a restart stays consistent.
    restarted = PluginManager(plugins_dir=plugins_dir, data_dir=data_dir)
    restarted.discover_and_load_all()
    assert restarted.configs[PLUGIN_ID]["custom_models"] == []
    model = _models_by_id(restarted)["model-a"]
    assert model.context_window == 512000
    assert model.max_tokens == 128000


def test_custom_model_crud_still_works(tmp_path):
    """Existing custom model add/update/delete behaviour is unchanged."""
    manager, plugins_dir, data_dir = _make_manager(tmp_path)

    custom = PluginModelConfig(
        id="custom-model",
        name="Custom Model",
        upstream_model="custom-upstream",
        context_window=32000,
        max_tokens=8192,
    )
    assert manager.add_custom_model(PLUGIN_ID, custom) is True
    assert manager.configs[PLUGIN_ID]["custom_models"][0]["id"] == "custom-model"
    assert _models_by_id(manager)["custom-model"].context_window == 32000

    updated = custom.model_copy(update={"name": "Custom Model Updated", "max_tokens": 16384})
    assert manager.update_custom_model(PLUGIN_ID, "custom-model", updated) is True
    model = _models_by_id(manager)["custom-model"]
    assert model.name == "Custom Model Updated"
    assert model.max_tokens == 16384

    restarted = PluginManager(plugins_dir=plugins_dir, data_dir=data_dir)
    restarted.discover_and_load_all()
    assert _models_by_id(restarted)["custom-model"].max_tokens == 16384

    assert manager.delete_custom_model(PLUGIN_ID, "custom-model") is True
    assert "custom-model" not in _models_by_id(manager)


@pytest.fixture
def client(tmp_path, monkeypatch):
    """TestClient backed by an isolated plugins and data directory."""
    temp_plugins_dir = tmp_path / "plugins"
    temp_data_dir = tmp_path / "data"
    temp_plugins_dir.mkdir(parents=True, exist_ok=True)
    temp_data_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(plugin_manager, "plugins_dir", temp_plugins_dir)
    monkeypatch.setattr(plugin_manager, "data_dir", temp_data_dir)
    monkeypatch.setattr(plugin_manager, "plugins", {})
    monkeypatch.setattr(plugin_manager, "plugin_files", {})
    monkeypatch.setattr(plugin_manager, "configs", {})
    monkeypatch.setattr(plugin_manager, "rules", {})
    monkeypatch.setattr(plugin_manager, "logs", [])
    monkeypatch.setattr(plugin_manager, "total_requests_count", 0)

    return TestClient(app, headers={"X-Admin-Token": "test-admin-token"})


def _create_plugin(client):
    response = client.post(
        "/api/plugins",
        json={"filename": f"{PLUGIN_ID}.py", "code": PLUGIN_CODE},
    )
    assert response.status_code == 200


def test_put_builtin_model_writes_edited_models_and_survives_reload(client):
    """PUT on a built-in model must persist as an edit and survive a reload."""
    _create_plugin(client)

    response = client.put(
        f"/api/plugins/{PLUGIN_ID}/models/model-a",
        json={
            "name": "Edited Model A",
            "upstream_model": "edited-upstream",
            "context_window": 512000,
            "max_tokens": 128000,
            "enabled": False,
            "parameters": {"temperature": 0.2},
        },
    )
    assert response.status_code == 200
    assert response.json()["model"]["context_window"] == 512000
    assert response.json()["model"]["max_tokens"] == 128000

    cfg = plugin_manager.configs[PLUGIN_ID]
    assert "model-a" in cfg["edited_models"]
    assert cfg["custom_models"] == []

    reload_response = client.post(f"/api/plugins/{PLUGIN_ID}/reload")
    assert reload_response.status_code == 200

    plugins = client.get("/api/plugins").json()
    model = next(m for p in plugins for m in p["models"] if m["id"] == "model-a")
    assert model["context_window"] == 512000
    assert model["max_tokens"] == 128000
    assert model["upstream_model"] == "edited-upstream"
    assert model["enabled"] is False
    assert model["parameters"] == {"temperature": 0.2}


def test_model_cards_include_configured_limits(client):
    """GET /v1/models and /v1/models/{id} expose context_window and max_tokens."""
    _create_plugin(client)

    list_response = client.get("/v1/models")
    assert list_response.status_code == 200
    card = next(m for m in list_response.json()["data"] if m["id"] == "model-a")
    assert card["context_window"] == 200000
    assert card["max_tokens"] == 64000

    single_response = client.get("/v1/models/model-a")
    assert single_response.status_code == 200
    single = single_response.json()
    assert single["context_window"] == 200000
    assert single["max_tokens"] == 64000

    client.put(
        f"/api/plugins/{PLUGIN_ID}/models/model-a",
        json={"context_window": 512000, "max_tokens": 128000},
    )
    updated = client.get("/v1/models/model-a").json()
    assert updated["context_window"] == 512000
    assert updated["max_tokens"] == 128000
