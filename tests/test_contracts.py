import pytest
from llmproxy.schemas import (
    ChatMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
    PluginMetadata,
    PluginVariable,
    PluginModelConfig,
    ModelCard,
    ModelListResponse,
)
from llmproxy.core.interface import BasePlugin


def test_schema_instantiation():
    req = ChatCompletionRequest(
        model="test-model",
        messages=[ChatMessage(role="user", content="Hello")],
        temperature=0.7,
    )
    assert req.model == "test-model"
    assert len(req.messages) == 1
    assert req.messages[0].content == "Hello"


def test_plugin_metadata_schema():
    meta = PluginMetadata(
        id="test_provider",
        name="Test Provider",
        variables=[
            PluginVariable(name="api_key", label="API Key", type="secret", required=True)
        ],
        models=[
            PluginModelConfig(id="test-gpt", name="Test GPT", upstream_model="gpt-4o")
        ],
    )
    assert meta.id == "test_provider"
    assert len(meta.variables) == 1
    assert meta.variables[0].name == "api_key"
    assert len(meta.models) == 1


class DummyPlugin(BasePlugin):
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            id="dummy",
            name="Dummy",
            variables=[PluginVariable(name="token", label="Token", required=True)],
            models=[PluginModelConfig(id="dummy-1", name="Dummy 1")],
        )

    async def chat_completion(self, request, model_config, variables):
        return ChatCompletionResponse(
            id="chatcmpl-123",
            model=request.model,
            choices=[],
        )


def test_base_plugin_contract():
    plugin = DummyPlugin()
    assert plugin.metadata.id == "dummy"
    errors = plugin.validate_variables({})
    assert len(errors) == 1
    assert "token" in errors[0]

    valid = plugin.validate_variables({"token": "secret123"})
    assert len(valid) == 0
