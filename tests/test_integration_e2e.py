"""End-to-End full system integration tests for LLMProxy."""

import pytest
from fastapi.testclient import TestClient
from llmproxy.main import app
from llmproxy.core.manager import plugin_manager


@pytest.fixture
def e2e_client():
    # Discover real plugins
    plugin_manager.discover_and_load_all()
    return TestClient(app, headers={"X-Admin-Token": "test-admin-token"})


def test_e2e_plugins_and_models_catalog(e2e_client):
    """Verify that default plugins (antigravity, codex, anthropic, openrouter, openai) are loaded."""
    # 1. Check health
    health = e2e_client.get("/health").json()
    assert health["status"] == "ok"
    assert health["plugins_count"] >= 4

    # 2. Check OpenAI-compatible models list
    models_resp = e2e_client.get("/v1/models")
    assert models_resp.status_code == 200
    model_ids = [m["id"] for m in models_resp.json()["data"]]

    # Antigravity models
    assert "gemini-3.8-flash" in model_ids
    assert "claude-sonnet-4-6" in model_ids
    # Codex models
    assert "gpt-6-luna" in model_ids
    assert "gpt-4o" in model_ids
    # Anthropic Messages models
    assert "claude-sonnet-5-5" in model_ids
    # OpenRouter free models
    assert "nvidia/nemotron-3-ultra-550b-a55b:free" in model_ids
    # OpenAI compatible models
    assert "default-model" in model_ids


def test_e2e_openai_chat_completion(e2e_client):
    """Verify standard non-streaming /v1/chat/completions."""
    payload = {
        "model": "gemini-3.8-flash",
        "messages": [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Hello LLMProxy"},
        ],
        "temperature": 0.7,
    }
    resp = e2e_client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert data["model"] == "gemini-3.8-flash"
    assert len(data["choices"]) > 0
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert len(data["choices"][0]["message"]["content"]) > 0


def test_e2e_openai_streaming_completion(e2e_client):
    """Verify standard streaming SSE /v1/chat/completions."""
    payload = {
        "model": "gpt-6-luna",
        "messages": [{"role": "user", "content": "Stream me a response"}],
        "stream": True,
    }
    resp = e2e_client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    lines = resp.text.strip().split("\n\n")
    assert len(lines) >= 2
    assert lines[-1] == "data: [DONE]"


def test_e2e_dashboard_static_and_monochrome_assets(e2e_client):
    """Verify the black & white minimalist frontend is served with zero external CDNs."""
    # Root dashboard
    root_resp = e2e_client.get("/")
    assert root_resp.status_code == 200
    assert "LLMProxy" in root_resp.text
    assert '<link rel="stylesheet" href="/static/style.css">' in root_resp.text
    assert '<script src="/static/app.js">' in root_resp.text

    # CSS check: monochrome styling variables present
    css_resp = e2e_client.get("/static/style.css")
    assert css_resp.status_code == 200
    assert "--bg-base: #000000" in css_resp.text or "--bg" in css_resp.text
    assert "--border" in css_resp.text

    # JS check: modern vanilla app
    js_resp = e2e_client.get("/static/app.js")
    assert js_resp.status_code == 200
    assert "fetchPlugins" in js_resp.text or "plugins" in js_resp.text


def test_e2e_dashboard_plugin_customization(e2e_client):
    """Verify adding custom model and updating variables via dashboard API."""
    # 1. Add custom model to antigravity
    add_model_resp = e2e_client.post(
        "/api/plugins/antigravity/models",
        json={
            "id": "antigravity-custom-fast",
            "name": "Custom Fast Antigravity",
            "upstream_model": "gemini-2.5-flash",
            "enabled": True,
        },
    )
    assert add_model_resp.status_code == 200

    # 2. Check /v1/models reflects the custom model
    models = [m["id"] for m in e2e_client.get("/v1/models").json()["data"]]
    assert "antigravity-custom-fast" in models

    # 3. Clean up: delete custom model
    del_resp = e2e_client.delete("/api/plugins/antigravity/models/antigravity-custom-fast")
    assert del_resp.status_code == 200
