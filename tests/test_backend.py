import json
import pytest
from fastapi.testclient import TestClient

from llmproxy.core.manager import plugin_manager
from llmproxy.main import app
from llmproxy.schemas import (
    ChatChoiceDelta,
    ChatCompletionChunk,
    ChatCompletionResponse,
    ModelCard,
    ModelListResponse,
)


@pytest.fixture(autouse=True)
def setup_backend():
    """Ensure plugins and test credentials are initialized for backend tests."""
    plugin_manager.discover_and_load_all()
    plugin_manager.accounts.clear()
    plugin_manager.logs.clear()
    plugin_manager.total_requests_count = 0
    # Configure test API keys so variable validation succeeds
    plugin_manager.update_plugin_config("antigravity", enabled=True, variables={"api_key": "test-key", "timeout": 60})
    plugin_manager.update_plugin_config("codex", enabled=True, variables={"api_key": "test-key"})
    plugin_manager.update_plugin_config("anthropic_proxy", enabled=True, variables={"api_key": "test-key"})
    plugin_manager.update_plugin_config("openrouter", enabled=True, variables={"api_key": "test-key"})
    plugin_manager.update_plugin_config("openai_compatible", enabled=True, variables={"api_key": ""})


def test_models_endpoint_list():
    """Verify GET /v1/models returns active models from all enabled plugins."""
    with TestClient(app) as client:
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()

        assert data["object"] == "list"
        model_ids = {m["id"] for m in data["data"]}

        # All provider plugins' default models should be listed
        assert "gemini-3.8-flash" in model_ids
        assert "claude-sonnet-4-6" in model_ids
        assert "gpt-4o" in model_ids
        assert "gpt-6-luna" in model_ids
        assert "claude-sonnet-5-5" in model_ids
        # Anthropic reasoning variants are discoverable through the API.
        assert "claude-sonnet-5-5:high" in model_ids
        assert "claude-fable-5-2:max" in model_ids
        assert "nvidia/nemotron-3-ultra-550b-a55b:free" in model_ids
        assert "default-model" in model_ids


def test_models_endpoint_get_single():
    """Verify GET /v1/models/{model_id} returns single ModelCard or 404."""
    with TestClient(app) as client:
        resp = client.get("/v1/models/gpt-4o")
        assert resp.status_code == 200
        card = resp.json()
        assert card["id"] == "gpt-4o"
        assert card["object"] == "model"
        assert card["owned_by"] == "codex"

        # Anthropic reasoning variants are addressable as models too.
        resp_variant = client.get("/v1/models/claude-sonnet-5-5:high")
        assert resp_variant.status_code == 200
        assert resp_variant.json()["owned_by"] == "anthropic_proxy"

        # Non-existent model
        resp_404 = client.get("/v1/models/non-existent-model")
        assert resp_404.status_code == 404
        assert "not found" in resp_404.json()["detail"]["error"]["message"].lower()


def test_chat_completions_non_streaming():
    """Verify POST /v1/chat/completions non-streaming response."""
    with TestClient(app) as client:
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"},
            ],
            "temperature": 0.7,
            "stream": False,
        }
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        assert data["object"] == "chat.completion"
        assert data["model"] == "gpt-4o"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["finish_reason"] == "stop"
        assert len(data["choices"][0]["message"]["content"]) > 0
        assert data["usage"]["total_tokens"] > 0


def test_chat_completions_streaming_sse():
    """Verify POST /v1/chat/completions SSE streaming with data: <json> and data: [DONE]."""
    with TestClient(app) as client:
        payload = {
            "model": "gemini-3.8-flash",
            "messages": [{"role": "user", "content": "Stream me a response"}],
            "stream": True,
        }
        with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")

            lines = [line.strip() for line in resp.iter_lines() if line.strip()]
            assert len(lines) >= 3

            # Each event line starts with data:
            data_lines = [l for l in lines if l.startswith("data: ")]
            assert len(data_lines) == len(lines)

            # Last line must be [DONE]
            assert data_lines[-1] == "data: [DONE]"

            # Earlier lines must be valid ChatCompletionChunk JSON
            chunks = []
            for line in data_lines[:-1]:
                raw_json = line[6:]
                chunk = json.loads(raw_json)
                assert chunk["object"] == "chat.completion.chunk"
                assert chunk["model"] == "gemini-3.8-flash"
                chunks.append(chunk)

            # Check chunk content aggregation
            content_tokens = [
                c["choices"][0]["delta"].get("content", "")
                for c in chunks
                if "content" in c["choices"][0]["delta"]
            ]
            full_text = "".join(content_tokens)
            assert "Antigravity" in full_text


def test_routing_across_plugins():
    """Verify model routing sends requests to the appropriate plugin."""
    with TestClient(app) as client:
        test_routes = [
            ("gemini-3.8-flash", "Antigravity"),
            ("gpt-6-luna", "Codex"),
            ("claude-sonnet-5-5", "Anthropic"),
            ("nvidia/nemotron-3-ultra-550b-a55b:free", "OpenRouter"),
            ("default-model", "OpenAI Compatible"),
        ]

        for model_id, provider_keyword in test_routes:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": model_id,
                    "messages": [{"role": "user", "content": "Ping"}],
                    "stream": False,
                },
            )
            assert resp.status_code == 200, f"Failed for model {model_id}: {resp.text}"
            content = resp.json()["choices"][0]["message"]["content"]
            assert provider_keyword.lower() in content.lower()


def test_error_handling_unknown_model():
    """Verify 404 is returned when requested model does not exist."""
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "non-existent-gpt-999",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert resp.status_code == 404
        data = resp.json()
        assert "not found" in data["detail"]["error"]["message"].lower()


def test_error_handling_disabled_plugin():
    """Verify models from disabled plugins return 404."""
    with TestClient(app) as client:
        # Disable codex
        plugin_manager.update_plugin_config("codex", enabled=False)

        resp = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]},
        )
        assert resp.status_code == 404

        # Re-enable codex
        plugin_manager.update_plugin_config("codex", enabled=True)
        resp2 = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]},
        )
        assert resp2.status_code == 200


def test_error_handling_variable_validation():
    """Verify 400 is returned when required operational plugin variables are missing."""
    with TestClient(app) as client:
        # Clear timeout for antigravity
        plugin_manager.update_plugin_config("antigravity", variables={"timeout": ""})

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "gemini-3.8-flash",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert resp.status_code == 400
        data = resp.json()
        assert "required" in data["detail"]["error"]["message"].lower()

        # Restore timeout
        plugin_manager.update_plugin_config("antigravity", variables={"timeout": 60})


def test_telemetry_logs_recording():
    """Verify request telemetry logs capture latency, status, and token estimations."""
    with TestClient(app) as client:
        initial_log_count = len(plugin_manager.get_logs())

        # Successful request
        client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Telemetry test prompt"}],
                "stream": False,
            },
        )

        logs = plugin_manager.get_logs()
        assert len(logs) > initial_log_count
        latest_log = logs[0]
        assert latest_log.model == "gpt-4o"
        assert latest_log.plugin_id == "codex"
        assert latest_log.status == "success"
        assert latest_log.status_code == 200
        assert latest_log.latency_ms >= 0
        assert latest_log.prompt_tokens > 0
        assert latest_log.completion_tokens > 0


def test_streaming_error_handling(monkeypatch):
    """Verify streaming yields error JSON payload and [DONE] on plugin exception."""
    with TestClient(app) as client:
        plugin = plugin_manager.get_plugin("openai_compatible")
        assert plugin is not None

        async def broken_chat_completion(*args, **kwargs):
            async def fail_stream():
                yield ChatCompletionChunk(
                    id="chunk-1",
                    model="default-model",
                    choices=[ChatChoiceDelta(index=0, delta={"content": "Start"})],
                )
                raise RuntimeError("Mid-stream connection dropped")

            return fail_stream()

        monkeypatch.setattr(plugin, "chat_completion", broken_chat_completion)

        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={"model": "default-model", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
        ) as resp:
            lines = [l.strip() for l in resp.iter_lines() if l.strip()]
            assert any("Mid-stream connection dropped" in l for l in lines)
            assert lines[-1] == "data: [DONE]"


def test_health_check_endpoint():
    """Verify /health endpoint returns status ok and correct plugin count."""
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["app"] == "llmproxy"
        assert data["plugins_count"] >= 4
