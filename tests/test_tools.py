"""
Unit and integration tests for Tool Calling (function calling) support across
all providers (Antigravity, Codex, Anthropic, OpenRouter, OpenAI Compatible) and local endpoints.
"""

import json
import pytest
from fastapi.testclient import TestClient

from llmproxy.core.manager import plugin_manager
from llmproxy.main import app
from llmproxy.schemas import (
    ChatMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
)


SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. San Francisco",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["location"],
            },
        },
    }
]


@pytest.fixture(autouse=True)
def setup_providers():
    plugin_manager.discover_and_load_all()
    plugin_manager.accounts.clear()
    plugin_manager.update_plugin_config("antigravity", enabled=True, variables={"api_key": "test-key", "timeout": 60})
    plugin_manager.update_plugin_config("codex", enabled=True, variables={"api_key": "test-key"})
    plugin_manager.update_plugin_config("anthropic_proxy", enabled=True, variables={"api_key": "test-key"})
    plugin_manager.update_plugin_config("openrouter", enabled=True, variables={"api_key": "test-key"})
    plugin_manager.update_plugin_config("openai_compatible", enabled=True, variables={"api_key": ""})


def test_schema_tool_fields():
    """Verify ChatCompletionRequest and ChatMessage support tool fields without validation errors."""
    msg_user = ChatMessage(role="user", content="What is the weather in Paris?")
    msg_assist = ChatMessage(
        role="assistant",
        content=None,
        tool_calls=[
            {
                "id": "call_123",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"location": "Paris"}'},
            }
        ],
    )
    msg_tool = ChatMessage(
        role="tool",
        tool_call_id="call_123",
        name="get_weather",
        content='{"temperature": 20, "unit": "celsius"}',
    )

    req = ChatCompletionRequest(
        model="gemini-3.8-flash",
        messages=[msg_user, msg_assist, msg_tool],
        tools=SAMPLE_TOOLS,
        tool_choice="auto",
    )
    assert len(req.messages) == 3
    assert req.messages[1].tool_calls[0]["function"]["name"] == "get_weather"
    assert req.messages[2].tool_call_id == "call_123"
    assert req.tools[0]["function"]["name"] == "get_weather"


@pytest.mark.parametrize("model_id", ["default-model", "gpt-4o", "claude-sonnet-5-5", "gemini-3.8-flash", "nvidia/nemotron-3-ultra-550b-a55b:free"])
def test_all_providers_mock_tool_calling_non_streaming(model_id):
    """Verify all providers return valid tool_calls when tools are supplied."""
    with TestClient(app) as client:
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
            "tools": SAMPLE_TOOLS,
            "tool_choice": "auto",
        }
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200, f"Failed for {model_id}: {resp.text}"
        data = resp.json()

        assert data["object"] == "chat.completion"
        choice = data["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        message = choice["message"]
        assert message["role"] == "assistant"
        assert message["tool_calls"] is not None
        assert len(message["tool_calls"]) > 0
        assert message["tool_calls"][0]["function"]["name"] == "get_weather"
        assert "location" in message["tool_calls"][0]["function"]["arguments"]


@pytest.mark.parametrize("model_id", ["default-model", "gpt-4o", "claude-sonnet-5-5", "gemini-3.8-flash", "nvidia/nemotron-3-ultra-550b-a55b:free"])
def test_all_providers_mock_tool_calling_streaming(model_id):
    """Verify all providers stream tool_calls via SSE chunks properly."""
    with TestClient(app) as client:
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
            "tools": SAMPLE_TOOLS,
            "tool_choice": "auto",
            "stream": True,
        }
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        lines = resp.text.strip().split("\n")
        data_lines = [line[6:].strip() for line in lines if line.startswith("data: ") and line.strip() != "data: [DONE]"]
        
        has_tool_call_delta = False
        final_finish_reason = None

        for raw in data_lines:
            chunk = json.loads(raw)
            for c in chunk.get("choices", []):
                if c.get("delta", {}).get("tool_calls"):
                    has_tool_call_delta = True
                if c.get("finish_reason"):
                    final_finish_reason = c["finish_reason"]

        assert has_tool_call_delta, f"No tool_calls delta found in stream for {model_id}"
        assert final_finish_reason == "tool_calls", f"Expected finish_reason tool_calls for {model_id}, got {final_finish_reason}"


@pytest.mark.parametrize("model_id", ["default-model", "gpt-4o", "claude-sonnet-5-5", "gemini-3.8-flash", "nvidia/nemotron-3-ultra-550b-a55b:free"])
def test_all_providers_tool_followup_roundtrip(model_id):
    """Verify follow-up message with tool role completes conversation with finish_reason stop."""
    with TestClient(app) as client:
        payload = {
            "model": model_id,
            "messages": [
                {"role": "user", "content": "What is the weather in Paris?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"location": "Paris"}'},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_abc123",
                    "name": "get_weather",
                    "content": json.dumps({"temperature": 21, "condition": "sunny"}),
                },
            ],
            "tools": SAMPLE_TOOLS,
        }
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        choice = data["choices"][0]
        assert choice["finish_reason"] == "stop"
        assert choice["message"]["content"] is not None
        assert len(choice["message"]["content"]) > 0


def test_antigravity_gemini_schema_conversion():
    """Verify Antigravity tool conversion transforms JSON Schema to Gemini uppercase types."""
    from plugins.antigravity import _convert_openai_tools_to_gemini, _convert_tool_choice_to_gemini

    gemini_tools = _convert_openai_tools_to_gemini(SAMPLE_TOOLS)
    assert gemini_tools is not None
    assert len(gemini_tools) == 1
    decl = gemini_tools[0]["functionDeclarations"][0]
    assert decl["name"] == "get_weather"
    params = decl["parameters"]
    assert params["type"] == "OBJECT"
    assert params["properties"]["location"]["type"] == "STRING"
    assert "location" in params["required"]

    tool_cfg_auto = _convert_tool_choice_to_gemini("auto")
    assert tool_cfg_auto == {"functionCallingConfig": {"mode": "AUTO"}}

    tool_cfg_specific = _convert_tool_choice_to_gemini({"type": "function", "function": {"name": "get_weather"}})
    assert tool_cfg_specific == {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["get_weather"]}}


def test_anthropic_proxy_tool_conversion():
    """Verify the Anthropic Messages template converts tools and messages."""
    from templates.anthropic_proxy import AnthropicProxyPlugin
    plugin = AnthropicProxyPlugin()

    anthropic_tools = plugin._format_anthropic_tools(SAMPLE_TOOLS)
    assert anthropic_tools is not None
    assert anthropic_tools[0]["name"] == "get_weather"
    assert anthropic_tools[0]["input_schema"]["type"] == "object"

    req_messages = [
        ChatMessage(role="system", content="System instruction"),
        ChatMessage(role="user", content="Call tool"),
        ChatMessage(
            role="assistant",
            content=None,
            tool_calls=[
                {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"location": "Rome"}'}}
            ],
        ),
        ChatMessage(role="tool", tool_call_id="call_1", content="25C"),
    ]
    sys_prompts, a_messages = plugin._build_anthropic_messages(req_messages)
    assert sys_prompts == ["System instruction"]
    assert len(a_messages) == 3
    assert a_messages[0]["role"] == "user"
    assert a_messages[1]["role"] == "assistant"
    assert a_messages[1]["content"][0]["type"] == "tool_use"
    assert a_messages[2]["role"] == "user"
    assert a_messages[2]["content"][0]["type"] == "tool_result"
