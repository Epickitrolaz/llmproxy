import json
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from llmproxy.main import app
from llmproxy.core.manager import plugin_manager
from llmproxy.schemas import PluginMetadata, PluginModelConfig, PluginVariable, RequestLogEntry


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Fixture providing TestClient with an isolated plugins directory and data directory."""
    temp_plugins_dir = tmp_path / "plugins"
    temp_data_dir = tmp_path / "data"
    temp_plugins_dir.mkdir(parents=True, exist_ok=True)
    temp_data_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(plugin_manager, "plugins_dir", temp_plugins_dir)
    monkeypatch.setattr(plugin_manager, "data_dir", temp_data_dir)
    monkeypatch.setattr(plugin_manager, "plugins", {})
    monkeypatch.setattr(plugin_manager, "configs", {})
    monkeypatch.setattr(plugin_manager, "logs", [])

    return TestClient(app)


SAMPLE_PLUGIN_CODE = '''
import time
import uuid
from llmproxy.core.interface import BasePlugin
from llmproxy.schemas import (
    PluginMetadata,
    PluginVariable,
    PluginModelConfig,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatChoice,
    ChatMessage,
    CompletionUsage,
)

class SamplePlugin(BasePlugin):
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            id="test_plugin",
            name="Test Provider",
            description="Plugin for automated testing",
            version="1.0.0",
            author="tester",
            enabled=True,
            variables=[
                PluginVariable(name="api_token", label="API Token", type="secret", default="initial_token", value="initial_token")
            ],
            models=[
                PluginModelConfig(id="test-model-1", name="Test Model 1", upstream_model="upstream-1")
            ],
        )

    async def chat_completion(self, request: ChatCompletionRequest, model_config: PluginModelConfig, variables: dict):
        reply = f"Echo from {model_config.id}: {request.messages[-1].content}"
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            model=request.model,
            choices=[
                ChatChoice(index=0, message=ChatMessage(role="assistant", content=reply))
            ],
            usage=CompletionUsage(prompt_tokens=5, completion_tokens=10, total_tokens=15)
        )

plugin = SamplePlugin()
'''

SAMPLE_PLUGIN_CODE_UPDATED = '''
import time
import uuid
from llmproxy.core.interface import BasePlugin
from llmproxy.schemas import (
    PluginMetadata,
    PluginVariable,
    PluginModelConfig,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatChoice,
    ChatMessage,
    CompletionUsage,
)

class SamplePlugin(BasePlugin):
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            id="test_plugin",
            name="Updated Test Provider",
            description="Updated description",
            version="2.0.0",
            author="tester",
            enabled=True,
            variables=[
                PluginVariable(name="api_token", label="API Token", type="secret", default="new_token", value="new_token")
            ],
            models=[
                PluginModelConfig(id="test-model-1", name="Test Model 1 Updated", upstream_model="upstream-1-v2")
            ],
        )

    async def chat_completion(self, request: ChatCompletionRequest, model_config: PluginModelConfig, variables: dict):
        reply = f"Updated echo from {model_config.id}: {request.messages[-1].content}"
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            model=request.model,
            choices=[
                ChatChoice(index=0, message=ChatMessage(role="assistant", content=reply))
            ],
            usage=CompletionUsage(prompt_tokens=6, completion_tokens=12, total_tokens=18)
        )

plugin = SamplePlugin()
'''


def test_static_file_serving(client):
    """Test that the web dashboard HTML and static files are served properly."""
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert "text/html" in res_root.headers.get("content-type", "")
    assert "LLMPROXY" in res_root.text

    res_static = client.get("/static/index.html")
    assert res_static.status_code == 200

    res_css = client.get("/static/style.css")
    assert res_css.status_code == 200
    assert "--bg-canvas" in res_css.text

    res_js = client.get("/static/app.js")
    assert res_js.status_code == 200
    assert "LLMProxy" in res_js.text


def test_plugin_lifecycle_crud(client):
    """Test full CRUD lifecycle for plugins via management API."""
    # 1. Initial list should be empty
    res = client.get("/api/plugins")
    assert res.status_code == 200
    assert res.json() == []

    # 2. Create/upload a new plugin
    create_res = client.post(
        "/api/plugins",
        json={"filename": "test_plugin.py", "code": SAMPLE_PLUGIN_CODE},
    )
    assert create_res.status_code == 200
    created_data = create_res.json()
    assert created_data["status"] == "ok"
    assert created_data["plugin"]["id"] == "test_plugin"
    assert created_data["plugin"]["name"] == "Test Provider"

    # 3. List plugins should now contain test_plugin
    res_list = client.get("/api/plugins")
    assert res_list.status_code == 200
    plugins = res_list.json()
    assert len(plugins) == 1
    assert plugins[0]["id"] == "test_plugin"
    assert plugins[0]["filename"] == "test_plugin.py"

    # 4. Get plugin raw code
    res_code = client.get("/api/plugins/test_plugin/code")
    assert res_code.status_code == 200
    code_data = res_code.json()
    assert code_data["plugin_id"] == "test_plugin"
    assert "SamplePlugin" in code_data["code"]

    # 5. Update/overwrite code live and trigger hot-reload
    update_res = client.put(
        "/api/plugins/test_plugin/code",
        json={"code": SAMPLE_PLUGIN_CODE_UPDATED},
    )
    assert update_res.status_code == 200
    assert update_res.json()["plugin"]["version"] == "2.0.0"
    assert update_res.json()["plugin"]["name"] == "Updated Test Provider"

    # 6. Syntax error handling
    bad_syntax_res = client.put(
        "/api/plugins/test_plugin/code",
        json={"code": "def bad_syntax(:\n   broken"},
    )
    assert bad_syntax_res.status_code == 400
    assert "Syntax error" in bad_syntax_res.json()["detail"]

    # 7. Explicit hot-reload
    reload_res = client.post("/api/plugins/test_plugin/reload")
    assert reload_res.status_code == 200
    assert reload_res.json()["status"] == "ok"


def test_plugin_config_update(client):
    """Test updating variables and enabled status."""
    client.post(
        "/api/plugins",
        json={"filename": "test_plugin.py", "code": SAMPLE_PLUGIN_CODE},
    )

    # Update enabled and variable
    res = client.put(
        "/api/plugins/test_plugin/config",
        json={
            "enabled": False,
            "variables": {"api_token": "updated_secret_123"},
        },
    )
    assert res.status_code == 200
    updated_meta = res.json()["plugin"]
    assert updated_meta["enabled"] is False
    assert updated_meta["variables"][0]["value"] == "updated_secret_123"

    # Re-enable
    res_enable = client.put(
        "/api/plugins/test_plugin/config",
        json={"enabled": True},
    )
    assert res_enable.status_code == 200
    assert res_enable.json()["plugin"]["enabled"] is True


def test_models_management(client):
    """Test adding, editing, and deleting custom models."""
    client.post(
        "/api/plugins",
        json={"filename": "test_plugin.py", "code": SAMPLE_PLUGIN_CODE},
    )

    # 1. Add custom model
    add_res = client.post(
        "/api/plugins/test_plugin/models",
        json={
            "id": "custom-gpt-4o",
            "name": "Custom GPT-4o",
            "upstream_model": "gpt-4o-2024-08-06",
            "enabled": True,
            "context_window": 128000,
            "max_tokens": 4096,
            "parameters": {"temperature": 0.5},
        },
    )
    assert add_res.status_code == 200
    assert add_res.json()["model"]["id"] == "custom-gpt-4o"

    # Duplicate model ID should fail
    dup_res = client.post(
        "/api/plugins/test_plugin/models",
        json={
            "id": "custom-gpt-4o",
            "name": "Custom GPT-4o Duplicate",
        },
    )
    assert dup_res.status_code == 400

    # 2. Edit model
    edit_res = client.put(
        "/api/plugins/test_plugin/models/custom-gpt-4o",
        json={
            "name": "Custom GPT-4o Renamed",
            "context_window": 64000,
            "enabled": False,
        },
    )
    assert edit_res.status_code == 200
    assert edit_res.json()["model"]["name"] == "Custom GPT-4o Renamed"
    assert edit_res.json()["model"]["enabled"] is False

    # 3. Delete model
    del_res = client.delete("/api/plugins/test_plugin/models/custom-gpt-4o")
    assert del_res.status_code == 200

    # Deleting again should return 404
    del_404 = client.delete("/api/plugins/test_plugin/models/custom-gpt-4o")
    assert del_404.status_code == 404


def test_playground_chat_and_telemetry(client):
    """Test interactive chat testing, logs collection, and stats metrics."""
    client.post(
        "/api/plugins",
        json={"filename": "test_plugin.py", "code": SAMPLE_PLUGIN_CODE},
    )

    # Check initial stats
    stats_res = client.get("/api/stats")
    assert stats_res.status_code == 200
    initial_stats = stats_res.json()
    assert initial_stats["total_requests"] == 0
    assert initial_stats["active_plugins_count"] == 1
    assert initial_stats["active_models_count"] == 1

    # Send a test chat request
    chat_res = client.post(
        "/api/test-chat",
        json={
            "model": "test-model-1",
            "prompt": "Hello test router!",
            "temperature": 0.7,
        },
    )
    assert chat_res.status_code == 200
    chat_data = chat_res.json()
    assert "choices" in chat_data
    assert "Echo from test-model-1: Hello test router!" in chat_data["choices"][0]["message"]["content"]
    assert "latency_ms" in chat_data

    # Check logs
    logs_res = client.get("/api/logs")
    assert logs_res.status_code == 200
    logs = logs_res.json()
    assert len(logs) == 1
    assert logs[0]["model"] == "test-model-1"
    assert logs[0]["status"] == "success"
    assert logs[0]["prompt_tokens"] > 0

    # Check updated stats
    stats_after = client.get("/api/stats").json()
    assert stats_after["total_requests"] == 1
    assert stats_after["success_rate"] == 100.0
    assert stats_after["avg_latency_ms"] >= 0.0


def test_delete_plugin(client):
    """Test plugin deletion and cleanup."""
    client.post(
        "/api/plugins",
        json={"filename": "to_delete.py", "code": SAMPLE_PLUGIN_CODE},
    )
    assert len(client.get("/api/plugins").json()) == 1

    del_res = client.delete("/api/plugins/test_plugin")
    assert del_res.status_code == 200

    # List should be empty now
    assert len(client.get("/api/plugins").json()) == 0

    # Second delete returns 404
    del_404 = client.delete("/api/plugins/test_plugin")
    assert del_404.status_code == 404


def test_templates_endpoint(client):
    """Test that starter templates are available."""
    res = client.get("/api/templates")
    assert res.status_code == 200
    templates = res.json()
    assert len(templates) >= 2
    template_ids = [t["id"] for t in templates]
    assert "mock_provider" in template_ids
    assert "openai_compatible" in template_ids


def test_test_chat_errors_and_validation(client):
    """Test error conditions for the chat playground."""
    # 1. Non-existent model
    res_404 = client.post("/api/test-chat", json={"model": "non-existent-model", "prompt": "Hi"})
    assert res_404.status_code == 404

    # 2. Missing prompt and messages
    client.post("/api/plugins", json={"filename": "test_plugin.py", "code": SAMPLE_PLUGIN_CODE})
    res_400 = client.post("/api/test-chat", json={"model": "test-model-1"})
    assert res_400.status_code == 400

    # 3. Disabled model
    client.put("/api/plugins/test_plugin/models/test-model-1", json={"enabled": False})
    res_disabled = client.post("/api/test-chat", json={"model": "test-model-1", "prompt": "Hi"})
    assert res_disabled.status_code == 400
    assert "disabled" in res_disabled.json()["detail"].lower()


def test_invalid_input_handling(client):
    """Test validation errors for empty filenames or invalid plugin IDs."""
    res_empty_file = client.post("/api/plugins", json={"filename": "", "code": "code"})
    assert res_empty_file.status_code == 400

    res_empty_code = client.post("/api/plugins", json={"filename": "empty.py", "code": ""})
    assert res_empty_code.status_code == 400

    res_nonexistent_code = client.get("/api/plugins/nonexistent/code")
    assert res_nonexistent_code.status_code == 404

    res_nonexistent_config = client.put("/api/plugins/nonexistent/config", json={"enabled": True})
    assert res_nonexistent_config.status_code == 404


def test_streaming_chat_completion(client):
    """Test that streaming chat requests return valid Server-Sent Events."""
    client.post("/api/plugins", json={"filename": "test_plugin.py", "code": SAMPLE_PLUGIN_CODE})
    res_stream = client.post(
        "/api/test-chat",
        json={"model": "test-model-1", "prompt": "Streaming test", "stream": True},
    )
    assert res_stream.status_code == 200
    assert "text/event-stream" in res_stream.headers.get("content-type", "")
    content = res_stream.text
    assert "data:" in content
    assert "[DONE]" in content


def test_path_traversal_protection(client):
    """Test that directory traversal attempts are blocked."""
    # Attempt to write outside plugins_dir
    res_traversal_create = client.post(
        "/api/plugins",
        json={"filename": "../../escaped.py", "code": SAMPLE_PLUGIN_CODE},
    )
    # Filename gets sanitized to escaped.py inside plugins_dir or rejected
    if res_traversal_create.status_code == 200:
        plugin_file = res_traversal_create.json()["plugin"]["filename"]
        assert "/" not in plugin_file
        assert ".." not in plugin_file

    # Attempt to read non-existent path
    res_traversal_code = client.get("/api/plugins/..%2F..%2Fetc/code")
    assert res_traversal_code.status_code in (400, 404)


def test_model_deletion_persistence(client):
    """Test that deleting a model persists and does not resurrect on reload."""
    client.post("/api/plugins", json={"filename": "test_plugin.py", "code": SAMPLE_PLUGIN_CODE})

    # Add a model
    client.post(
        "/api/plugins/test_plugin/models",
        json={"id": "persist-model", "name": "Persist Model"},
    )
    assert any(m["id"] == "persist-model" for m in client.get("/api/plugins").json()[0]["models"])

    # Delete the model
    del_res = client.delete("/api/plugins/test_plugin/models/persist-model")
    assert del_res.status_code == 200

    # Reload the plugin from disk
    reload_res = client.post("/api/plugins/test_plugin/reload")
    assert reload_res.status_code == 200

    # Model should still be gone!
    plugins = client.get("/api/plugins").json()
    model_ids = [m["id"] for m in plugins[0]["models"]]
    assert "persist-model" not in model_ids


def test_dashboard_emoji_and_exit_clean_modal(client):
    """Test that lightning emoji is removed and exit confirmation checks dirty state."""
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert "⚡" not in res_root.text

    res_static = client.get("/static/index.html")
    assert res_static.status_code == 200
    assert "⚡" not in res_static.text
    assert "Test Key (Cutoff: 30m / 10 reqs)" in res_static.text

    res_js = client.get("/static/app.js")
    assert res_js.status_code == 200
    assert "isFullscreenDirty" in res_js.text
    assert "if (isFullscreenDirty()) {" in res_js.text



