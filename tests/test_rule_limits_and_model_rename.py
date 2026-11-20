"""Custom rule limits and editable exposed model IDs."""

import pytest
from fastapi.testclient import TestClient

from llmproxy.core.manager import PluginManager, plugin_manager
from llmproxy.main import app


PLUGIN_ID = "limits_provider"

PLUGIN_CODE = '''
from llmproxy.core.interface import BasePlugin
from llmproxy.schemas import PluginMetadata, PluginModelConfig


class LimitsPlugin(BasePlugin):
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            id="limits_provider",
            name="Limits Provider",
            version="1.0.0",
            models=[
                PluginModelConfig(
                    id="model-a",
                    name="Model A",
                    upstream_model="model-a-upstream",
                    context_window=200000,
                    max_tokens=64000,
                ),
                PluginModelConfig(
                    id="model-b",
                    name="Model B",
                    context_window=32000,
                    max_tokens=8000,
                ),
            ],
        )

    async def chat_completion(self, request, model_config, variables, account=None):
        raise NotImplementedError

plugin = LimitsPlugin()
'''


@pytest.fixture
def client(tmp_path, monkeypatch):
    temp_plugins_dir = tmp_path / "plugins"
    temp_data_dir = tmp_path / "data"
    temp_plugins_dir.mkdir(parents=True, exist_ok=True)
    temp_data_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(plugin_manager, "plugins_dir", temp_plugins_dir)
    monkeypatch.setattr(plugin_manager, "data_dir", temp_data_dir)
    monkeypatch.setattr(plugin_manager, "config_file", temp_data_dir / "plugin_configs.json")
    monkeypatch.setattr(plugin_manager, "api_keys_file", temp_data_dir / "api_keys.json")
    monkeypatch.setattr(plugin_manager, "rules_file", temp_data_dir / "load_balancing_rules.json")
    monkeypatch.setattr(plugin_manager, "logs_file", temp_data_dir / "request_logs.json")
    monkeypatch.setattr(plugin_manager, "plugins", {})
    monkeypatch.setattr(plugin_manager, "plugin_files", {})
    monkeypatch.setattr(plugin_manager, "configs", {})
    monkeypatch.setattr(plugin_manager, "rules", {})
    monkeypatch.setattr(plugin_manager, "logs", [])
    monkeypatch.setattr(plugin_manager, "total_requests_count", 0)

    client = TestClient(app, headers={"X-Admin-Token": "test-admin-token"})
    response = client.post(
        "/api/plugins",
        json={"filename": f"{PLUGIN_ID}.py", "code": PLUGIN_CODE},
    )
    assert response.status_code == 200
    return client


def _model_card(client, model_id):
    response = client.get(f"/v1/models/{model_id}")
    assert response.status_code == 200, response.text
    return response.json()


def test_rule_limits_are_stored_and_advertised(client):
    response = client.post(
        "/api/load-balancing-rules",
        json={
            "model_id": "virtual-a",
            "name": "Virtual A",
            "targets": ["model-a", "model-b"],
            "context_window": 128000,
            "max_tokens": 16000,
        },
    )
    assert response.status_code == 200, response.text
    rule = response.json()
    assert rule["context_window"] == 128000
    assert rule["max_tokens"] == 16000

    card = _model_card(client, "virtual-a")
    assert card["context_window"] == 128000
    assert card["max_tokens"] == 16000

    # Updating the limits flows through to the model card.
    response = client.put(
        f"/api/load-balancing-rules/{rule['id']}",
        json={"context_window": 64000, "max_tokens": 12000},
    )
    assert response.status_code == 200
    card = _model_card(client, "virtual-a")
    assert card["context_window"] == 64000
    assert card["max_tokens"] == 12000

    # Clearing the limits reverts to deriving them from the targets.
    response = client.put(
        f"/api/load-balancing-rules/{rule['id']}",
        json={"context_window": None, "max_tokens": None},
    )
    assert response.status_code == 200
    card = _model_card(client, "virtual-a")
    assert card["context_window"] == 32000
    assert card["max_tokens"] == 8000


def test_rule_without_limits_derives_smallest_target(client):
    response = client.post(
        "/api/load-balancing-rules",
        json={"model_id": "virtual-auto", "name": "Auto", "targets": ["model-a", "model-b"]},
    )
    assert response.status_code == 200, response.text
    card = _model_card(client, "virtual-auto")
    assert card["context_window"] == 32000
    assert card["max_tokens"] == 8000


def test_rule_derives_limits_from_suffixed_targets(client):
    response = client.post(
        "/api/load-balancing-rules",
        json={
            "model_id": "virtual-suffixed",
            "name": "Suffixed",
            "targets": ["model-a:high", "model-b:low"],
        },
    )
    assert response.status_code == 200, response.text
    card = _model_card(client, "virtual-suffixed")
    assert card["context_window"] == 32000
    assert card["max_tokens"] == 8000


def test_rule_exposed_id_can_be_renamed(client):
    response = client.post(
        "/api/load-balancing-rules",
        json={"model_id": "virtual-old", "name": "Virtual", "targets": ["model-a"]},
    )
    rule_id = response.json()["id"]

    response = client.put(
        f"/api/load-balancing-rules/{rule_id}", json={"model_id": "virtual-new"}
    )
    assert response.status_code == 200
    assert response.json()["model_id"] == "virtual-new"

    assert client.get("/v1/models/virtual-old").status_code == 404
    assert _model_card(client, "virtual-new")["id"] == "virtual-new"

    # Duplicate exposed IDs are rejected.
    client.post(
        "/api/load-balancing-rules",
        json={"model_id": "virtual-other", "name": "Other", "targets": ["model-a"]},
    )
    duplicate = client.put(
        f"/api/load-balancing-rules/{rule_id}", json={"model_id": "virtual-other"}
    )
    assert duplicate.status_code == 400


def test_builtin_model_id_can_be_renamed_and_survives_reload(client):
    response = client.put(
        f"/api/plugins/{PLUGIN_ID}/models/model-a",
        json={"id": "model-a-renamed", "context_window": 100000},
    )
    assert response.status_code == 200, response.text
    assert response.json()["model"]["id"] == "model-a-renamed"

    models = client.get("/v1/models").json()["data"]
    ids = {m["id"] for m in models}
    assert "model-a-renamed" in ids
    assert "model-a" not in ids
    assert _model_card(client, "model-a-renamed")["context_window"] == 100000

    # Editing the renamed model again updates the same entry instead of
    # appending a duplicate custom model.
    response = client.put(
        f"/api/plugins/{PLUGIN_ID}/models/model-a-renamed",
        json={"id": "model-a-renamed", "max_tokens": 32000},
    )
    assert response.status_code == 200
    cfg = plugin_manager.configs[PLUGIN_ID]
    assert cfg["custom_models"] == []
    model = next(m for m in plugin_manager.get_plugin_models(PLUGIN_ID) if m.id == "model-a-renamed")
    assert model.context_window == 100000
    assert model.max_tokens == 32000

    # Survives hot reload.
    assert client.post(f"/api/plugins/{PLUGIN_ID}/reload").status_code == 200
    card = _model_card(client, "model-a-renamed")
    assert card["context_window"] == 100000
    assert card["max_tokens"] == 32000

    # Survives a fresh manager reading the persisted config.
    restarted = PluginManager(
        plugins_dir=plugin_manager.plugins_dir, data_dir=plugin_manager.data_dir
    )
    restarted.discover_and_load_all()
    renamed = [m for m in restarted.get_plugin_models(PLUGIN_ID) if m.id == "model-a-renamed"]
    assert len(renamed) == 1
    assert renamed[0].context_window == 100000


def test_custom_model_id_can_be_renamed(client):
    response = client.post(
        f"/api/plugins/{PLUGIN_ID}/models",
        json={"id": "custom-old", "name": "Custom", "upstream_model": "custom-up"},
    )
    assert response.status_code == 200

    response = client.put(
        f"/api/plugins/{PLUGIN_ID}/models/custom-old", json={"id": "custom-new"}
    )
    assert response.status_code == 200, response.text
    assert response.json()["model"]["id"] == "custom-new"

    models = {m.id: m for m in plugin_manager.get_plugin_models(PLUGIN_ID)}
    assert "custom-new" in models
    assert "custom-old" not in models
    assert client.get("/v1/models/custom-old").status_code == 404
    assert _model_card(client, "custom-new")["id"] == "custom-new"


def test_model_rename_collision_is_rejected(client):
    response = client.put(
        f"/api/plugins/{PLUGIN_ID}/models/model-a", json={"id": "model-b"}
    )
    assert response.status_code == 400
