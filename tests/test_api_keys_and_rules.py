import json
import pytest
from fastapi.testclient import TestClient

from llmproxy.main import app
from llmproxy.core.manager import plugin_manager
from llmproxy.schemas import (
    APIKeyCreate,
    APIKeyUpdate,
    LoadBalancingRuleCreate,
    LoadBalancingRuleUpdate,
    RequestLogEntry,
)

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_keys_and_rules():
    """Ensure clean state for keys and rules around each test."""
    original_keys = dict(plugin_manager.api_keys)
    original_rules = dict(plugin_manager.rules)
    original_logs = list(plugin_manager.logs)
    original_req_count = plugin_manager.total_requests_count

    # Clear keys and rules for isolated test run
    plugin_manager.api_keys.clear()
    plugin_manager.rules.clear()
    plugin_manager.logs.clear()
    plugin_manager.total_requests_count = 0
    plugin_manager._save_api_keys()
    plugin_manager._save_rules()
    plugin_manager._save_logs()

    yield

    # Restore
    plugin_manager.api_keys = original_keys
    plugin_manager.rules = original_rules
    plugin_manager.logs = original_logs
    plugin_manager.total_requests_count = original_req_count
    plugin_manager._save_api_keys()
    plugin_manager._save_rules()
    plugin_manager._save_logs()


def test_api_keys_crud_endpoints():
    # 1. Initially empty
    res = client.get("/api/keys")
    assert res.status_code == 200
    assert res.json() == []

    # 2. Create API key
    res = client.post("/api/keys", json={"name": "Dev Key", "key": "sk-proxy-test12345"})
    assert res.status_code == 200
    key_data = res.json()
    assert key_data["name"] == "Dev Key"
    assert key_data["key"] == "sk-proxy-test12345"
    assert key_data["enabled"] is True
    assert key_data["total_requests"] == 0
    key_id = key_data["id"]

    # 3. Get key
    res = client.get(f"/api/keys/{key_id}")
    assert res.status_code == 200
    assert res.json()["id"] == key_id

    # 4. Update key
    res = client.put(f"/api/keys/{key_id}", json={"name": "Dev Key Renamed", "enabled": False})
    assert res.status_code == 200
    assert res.json()["name"] == "Dev Key Renamed"
    assert res.json()["enabled"] is False

    # 5. Delete key
    res = client.delete(f"/api/keys/{key_id}")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"

    res = client.get(f"/api/keys/{key_id}")
    assert res.status_code == 404


def test_api_key_auth_and_usage_tracking():
    # When no keys configured, completion works without key
    res = client.post(
        "/v1/chat/completions",
        json={"model": "gemini-3.8-flash", "messages": [{"role": "user", "content": "Hello"}]},
    )
    # Could succeed or mock error, but should NOT be 401
    assert res.status_code != 401

    # Create a local API key
    created_key = plugin_manager.add_api_key(
        APIKeyCreate(name="Production Key", key="sk-proxy-secret-999")
    )
    assert created_key.key == "sk-proxy-secret-999"

    # Now request without key should fail with 401
    res = client.post(
        "/v1/chat/completions",
        json={"model": "gemini-3.8-flash", "messages": [{"role": "user", "content": "Hello"}]},
    )
    assert res.status_code == 401
    assert res.json()["detail"]["error"]["code"] == "missing_api_key"

    # Request with invalid key should fail with 401
    res = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer sk-proxy-wrong"},
        json={"model": "gemini-3.8-flash", "messages": [{"role": "user", "content": "Hello"}]},
    )
    assert res.status_code == 401
    assert res.json()["detail"]["error"]["code"] == "invalid_api_key"

    # Record usage directly on key
    plugin_manager.record_api_key_usage(created_key.id, prompt_tokens=15, completion_tokens=25)
    updated_key = plugin_manager.get_api_key(created_key.id)
    assert updated_key.total_requests == 1
    assert updated_key.prompt_tokens == 15
    assert updated_key.completion_tokens == 25
    assert updated_key.total_tokens == 40
    assert updated_key.last_used_at is not None


def test_logs_persistence_across_restarts(tmp_path):
    """Test that request logs persist to file and reload."""
    test_entry = RequestLogEntry(
        id="test-log-persisted-1",
        model="gpt-4o",
        status="success",
        status_code=200,
        latency_ms=123.4,
        prompt_tokens=10,
        completion_tokens=20,
    )
    plugin_manager.add_log(test_entry)
    plugin_manager.flush()

    # Check file exists on disk
    assert plugin_manager.logs_file.exists()
    content = json.loads(plugin_manager.logs_file.read_text(encoding="utf-8"))
    assert any(item["id"] == "test-log-persisted-1" for item in content)

    # Re-instantiate manager or simulate restart
    plugin_manager.logs.clear()
    assert len(plugin_manager.logs) == 0
    plugin_manager._load_logs()
    assert any(item.id == "test-log-persisted-1" for item in plugin_manager.logs)


def test_load_balancing_rules_crud_and_routing():
    # 1. Create a load balancing rule
    rule_payload = {
        "model_id": "custom-fast-router",
        "name": "Custom Fast Router",
        "enabled": True,
        "strategy": "round_robin",
        "targets": ["gemini-3.8-flash", "gpt-6-luna"],
        "description": "Load balances fast models",
    }
    res = client.post("/api/load-balancing-rules", json=rule_payload)
    assert res.status_code == 200
    rule_data = res.json()
    rule_id = rule_data["id"]
    assert rule_data["model_id"] == "custom-fast-router"
    assert rule_data["strategy"] == "round_robin"
    assert len(rule_data["targets"]) == 2

    # 2. List rules
    res = client.get("/api/load-balancing-rules")
    assert res.status_code == 200
    assert any(r["id"] == rule_id for r in res.json())

    # 3. Rule model is exposed in GET /v1/models
    res = client.get("/v1/models")
    assert res.status_code == 200
    models_list = res.json()["data"]
    custom_model = next((m for m in models_list if m["id"] == "custom-fast-router"), None)
    assert custom_model is not None
    assert custom_model["owned_by"] == "load_balancer"

    # 4. Rule model info via GET /v1/models/{model_id}
    res = client.get("/v1/models/custom-fast-router")
    assert res.status_code == 200
    assert res.json()["id"] == "custom-fast-router"
    assert res.json()["owned_by"] == "load_balancer"

    # 5. Model resolution routes to target
    resolved = plugin_manager.resolve_model("custom-fast-router")
    assert resolved is not None
    model_config, plugin, vars_dict = resolved
    # Should resolve to one of the targets: gemini-3.8-flash or gpt-6-luna
    assert model_config.id in ["gemini-3.8-flash", "gpt-6-luna"]

    # 6. Update rule
    res = client.put(
        f"/api/load-balancing-rules/{rule_id}",
        json={"strategy": "random", "name": "Random Fast Router"},
    )
    assert res.status_code == 200
    assert res.json()["strategy"] == "random"

    # 7. Delete rule
    res = client.delete(f"/api/load-balancing-rules/{rule_id}")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"

    # Now custom model is gone
    res = client.get("/v1/models/custom-fast-router")
    assert res.status_code == 404


def test_plugin_pinning():
    # Check plugins
    plugins = plugin_manager.list_plugins()
    if not plugins:
        return
    first_plugin = plugins[0]
    plugin_id = first_plugin.id

    # Toggle pin
    res = client.post(f"/api/plugins/{plugin_id}/pin")
    assert res.status_code == 200
    assert res.json()["pinned"] is True

    # Pinned plugin must now be at index 0
    updated_plugins = plugin_manager.list_plugins()
    assert updated_plugins[0].id == plugin_id
    assert updated_plugins[0].pinned is True

    # Toggle unpin
    res = client.post(f"/api/plugins/{plugin_id}/pin")
    assert res.status_code == 200
    assert res.json()["pinned"] is False
