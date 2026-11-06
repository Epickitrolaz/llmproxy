import importlib.util
import json
from pathlib import Path
import pytest

from llmproxy.core.interface import BasePlugin
from llmproxy.core.manager import PluginManager, plugin_manager
from llmproxy.schemas import (
    ChatMessage,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    PluginMetadata,
    PluginModelConfig,
    PluginVariable,
    RequestLogEntry,
)


@pytest.fixture(autouse=True)
def init_manager():
    """Ensure plugin_manager has loaded all default plugins before each test."""
    plugin_manager.discover_and_load_all()
    # Reset default test keys
    for pid in ["antigravity", "codex", "claude_code"]:
        plugin_manager.update_plugin_config(pid, enabled=True, variables={"api_key": "test-key"})
    plugin_manager.update_plugin_config("openai_compatible", enabled=True, variables={"api_key": ""})


def test_plugin_discovery_and_metadata():
    """Verify built-in plugins are discovered and have correct metadata."""
    plugins = plugin_manager.list_plugins()
    plugin_ids = {p.id for p in plugins}

    assert "antigravity" in plugin_ids
    assert "codex" in plugin_ids
    assert "claude_code" in plugin_ids
    assert "openai_compatible" in plugin_ids

    # Check antigravity metadata
    ag = plugin_manager.get_plugin("antigravity")
    assert ag is not None
    assert ag.metadata.id == "antigravity"
    assert ag.metadata.name == "Antigravity"
    assert ag.metadata.filename == "antigravity.py"
    ag_model_ids = {m.id for m in ag.metadata.models}
    assert "gemini-3.8-flash" in ag_model_ids
    assert "claude-sonnet-4-6" in ag_model_ids
    assert "claude-opus-4-6-thinking" in ag_model_ids

    # Check codex metadata
    codex = plugin_manager.get_plugin("codex")
    assert codex is not None
    assert codex.metadata.id == "codex"
    codex_models = {m.id for m in codex.metadata.models}
    assert "gpt-6-luna" in codex_models
    assert "gpt-4o" in codex_models
    assert "gpt-4o-mini" in codex_models

    # Check claude_code metadata
    claude = plugin_manager.get_plugin("claude_code")
    assert claude is not None
    claude_models = {m.id for m in claude.metadata.models}
    assert "claude-3-7-sonnet" in claude_models
    assert "claude-3-5-sonnet" in claude_models
    assert "claude-3-5-haiku" in claude_models

    # Check openai_compatible metadata
    oai = plugin_manager.get_plugin("openai_compatible")
    assert oai is not None
    oai_models = {m.id for m in oai.metadata.models}
    assert "default-model" in oai_models


def test_variable_validation():
    """Verify validation of required vs optional variables."""
    ag = plugin_manager.get_plugin("antigravity")
    assert ag is not None

    # Missing api_key
    errs = ag.validate_variables({})
    assert len(errs) == 1
    assert "api_key" in errs[0]

    # Valid api_key
    assert len(ag.validate_variables({"api_key": "sk-antigravity-123"})) == 0

    # OpenAI compatible has no required variables by default
    oai = plugin_manager.get_plugin("openai_compatible")
    assert oai is not None
    assert len(oai.validate_variables({})) == 0


def test_config_persistence_and_isolation(tmp_path: Path):
    """Verify that configuration changes persist to disk and reload accurately."""
    plugins_dir = tmp_path / "plugins"
    data_dir = tmp_path / "data"
    plugins_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)

    # Create a dummy plugin file in tmp plugins dir
    dummy_code = """
from llmproxy.core.interface import BasePlugin
from llmproxy.schemas import PluginMetadata, PluginVariable, PluginModelConfig, ChatCompletionResponse

class CustomDummy(BasePlugin):
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            id="dummy_p",
            name="Dummy Plugin",
            variables=[PluginVariable(name="secret", label="Secret", type="secret", default="init_sec")],
            models=[PluginModelConfig(id="dummy-model-1", name="Dummy 1")],
        )

    async def chat_completion(self, request, model_config, variables):
        return ChatCompletionResponse(id="test", model=request.model, choices=[])

plugin = CustomDummy()
"""
    (plugins_dir / "dummy_p.py").write_text(dummy_code)

    mgr = PluginManager(plugins_dir=plugins_dir, data_dir=data_dir)
    mgr.discover_and_load_all()

    assert mgr.get_plugin("dummy_p") is not None
    models = mgr.get_plugin_models("dummy_p")
    assert len(models) == 1
    assert models[0].id == "dummy-model-1"

    # Update config: change variable, disable plugin
    mgr.update_plugin_config("dummy_p", enabled=False, variables={"secret": "updated_secret"})

    # Check persistence file on disk
    config_file = data_dir / "plugin_configs.json"
    assert config_file.exists()
    saved_data = json.loads(config_file.read_text())
    assert saved_data["dummy_p"]["enabled"] is False
    assert saved_data["dummy_p"]["variables"]["secret"] == "updated_secret"

    # New manager instance pointing to the same data_dir
    mgr2 = PluginManager(plugins_dir=plugins_dir, data_dir=data_dir)
    mgr2.discover_and_load_all()

    vars2 = mgr2.get_plugin_variables("dummy_p")
    assert vars2["secret"] == "updated_secret"
    active_models = mgr2.get_all_active_models()
    assert len(active_models) == 0  # because plugin was disabled


def test_custom_model_crud():
    """Verify adding, updating, and deleting custom and default models."""
    # 1. Add custom model to codex
    new_model = PluginModelConfig(
        id="custom-o3-mini",
        name="Custom O3 Mini",
        upstream_model="o3-mini",
        context_window=64000,
        max_tokens=8192,
    )
    assert plugin_manager.add_custom_model("codex", new_model) is True

    resolved = plugin_manager.resolve_model("custom-o3-mini")
    assert resolved is not None
    m_cfg, plugin, _ = resolved
    assert m_cfg.id == "custom-o3-mini"
    assert m_cfg.upstream_model == "o3-mini"
    assert m_cfg.context_window == 64000

    # 2. Update custom model
    updated_model = PluginModelConfig(
        id="custom-o3-mini",
        name="Custom O3 Mini Pro",
        upstream_model="o3-mini-pro",
        context_window=128000,
        max_tokens=16384,
    )
    assert plugin_manager.update_custom_model("codex", "custom-o3-mini", updated_model) is True

    resolved2 = plugin_manager.resolve_model("custom-o3-mini")
    assert resolved2 is not None
    m_cfg2, _, _ = resolved2
    assert m_cfg2.name == "Custom O3 Mini Pro"
    assert m_cfg2.upstream_model == "o3-mini-pro"
    assert m_cfg2.context_window == 128000

    # 3. Update an existing built-in default model
    default_edit = PluginModelConfig(
        id="gpt-4o",
        name="GPT-4o Custom Edited",
        upstream_model="gpt-4o-2024-11-20",
        context_window=256000,
    )
    assert plugin_manager.update_custom_model("codex", "gpt-4o", default_edit) is True
    resolved_default = plugin_manager.resolve_model("gpt-4o")
    assert resolved_default is not None
    assert resolved_default[0].name == "GPT-4o Custom Edited"
    assert resolved_default[0].upstream_model == "gpt-4o-2024-11-20"

    # 4. Delete the custom model
    assert plugin_manager.delete_custom_model("codex", "custom-o3-mini") is True
    assert plugin_manager.resolve_model("custom-o3-mini") is None

    # 5. Delete a built-in default model
    assert plugin_manager.delete_custom_model("codex", "gpt-4o-mini") is True
    assert plugin_manager.resolve_model("gpt-4o-mini") is None

    # Reset codex models for subsequent tests
    plugin_manager.configs["codex"]["edited_models"] = {}
    plugin_manager.configs["codex"]["deleted_models"] = []
    plugin_manager._save_configs()


def test_plugin_hot_reload(tmp_path: Path):
    """Verify hot-reloading a plugin dynamically updates code and metadata without restarts."""
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir(parents=True)

    plugin_file = plugins_dir / "dynamic_test.py"
    version_1_code = """
from llmproxy.core.interface import BasePlugin
from llmproxy.schemas import PluginMetadata, PluginModelConfig, ChatCompletionResponse

class DynamicPlugin(BasePlugin):
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            id="dynamic_test",
            name="Dynamic Test v1",
            version="1.0.0",
            models=[PluginModelConfig(id="v1-model", name="Model V1")],
        )

    async def chat_completion(self, request, model_config, variables):
        return ChatCompletionResponse(id="v1", model=request.model, choices=[])

plugin = DynamicPlugin()
"""
    plugin_file.write_text(version_1_code)

    mgr = PluginManager(plugins_dir=plugins_dir, data_dir=tmp_path / "data")
    mgr.discover_and_load_all()

    p = mgr.get_plugin("dynamic_test")
    assert p is not None
    assert p.metadata.version == "1.0.0"
    assert p.metadata.name == "Dynamic Test v1"

    # Update the plugin file on disk
    version_2_code = """
from llmproxy.core.interface import BasePlugin
from llmproxy.schemas import PluginMetadata, PluginModelConfig, ChatCompletionResponse

class DynamicPlugin(BasePlugin):
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            id="dynamic_test",
            name="Dynamic Test v2",
            version="2.0.0",
            models=[PluginModelConfig(id="v2-model", name="Model V2")],
        )

    async def chat_completion(self, request, model_config, variables):
        return ChatCompletionResponse(id="v2", model=request.model, choices=[])

plugin = DynamicPlugin()
"""
    plugin_file.write_text(version_2_code)

    # Perform hot-reload
    reloaded = mgr.reload_plugin("dynamic_test")
    assert reloaded is not None
    assert reloaded.metadata.version == "2.0.0"
    assert reloaded.metadata.name == "Dynamic Test v2"
    assert mgr.get_plugin("dynamic_test").metadata.version == "2.0.0"


def test_reload_plugin_nonexistent_and_syntax_error(tmp_path: Path):
    """Verify reload_plugin returns None on non-existent or syntax-broken files."""
    # Nonexistent
    assert plugin_manager.reload_plugin("non_existent_plugin_xyz") is None

    # Broken syntax on reload
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir(parents=True)
    bad_file = plugins_dir / "bad_syntax.py"
    bad_file.write_text("def foo(", encoding="utf-8")

    mgr = PluginManager(plugins_dir=plugins_dir, data_dir=tmp_path / "data")
    mgr.plugin_files["bad_syntax"] = bad_file
    assert mgr.reload_plugin("bad_syntax") is None


def test_log_retention_cap():
    """Verify add_log caps in-memory logs to 1000 entries."""
    mgr = PluginManager()
    mgr.logs.clear()

    for i in range(1050):
        mgr.add_log(
            RequestLogEntry(
                id=f"log-{i}",
                model="test-model",
                status="success",
                status_code=200,
            )
        )

    assert len(mgr.logs) == 1000
    assert mgr.logs[0].id == "log-1049"  # newest first
    assert mgr.logs[-1].id == "log-50"


def test_corrupted_config_and_invalid_plugin_file(tmp_path: Path):
    """Verify resilience when config is corrupted or a plugin file has syntax errors."""
    plugins_dir = tmp_path / "plugins"
    data_dir = tmp_path / "data"
    plugins_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)

    # 1. Corrupted config file
    (data_dir / "plugin_configs.json").write_text("{invalid_json: 123", encoding="utf-8")

    # 2. Syntax error in a plugin file
    (plugins_dir / "broken_plugin.py").write_text("def broken syntax (: :", encoding="utf-8")

    # 3. Valid plugin alongside broken one
    valid_code = """
from llmproxy.core.interface import BasePlugin
from llmproxy.schemas import PluginMetadata

class ValidPlugin(BasePlugin):
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(id="valid_p", name="Valid Plugin")
    async def chat_completion(self, request, model_config, variables):
        pass

plugin = ValidPlugin()
"""
    (plugins_dir / "valid_plugin.py").write_text(valid_code, encoding="utf-8")

    mgr = PluginManager(plugins_dir=plugins_dir, data_dir=data_dir)
    # Should not raise exception
    mgr.discover_and_load_all()

    assert mgr.get_plugin("valid_p") is not None
    assert mgr.get_plugin("broken_plugin") is None


def test_disabled_model_resolution():
    """Verify that a model explicitly disabled is not resolved."""
    # Disable gpt-4o by updating its config
    edit = PluginModelConfig(
        id="gpt-4o",
        name="GPT-4o",
        enabled=False,
    )
    plugin_manager.update_custom_model("codex", "gpt-4o", edit)
    assert plugin_manager.resolve_model("gpt-4o") is None

    # Re-enable gpt-4o
    edit.enabled = True
    plugin_manager.update_custom_model("codex", "gpt-4o", edit)
    assert plugin_manager.resolve_model("gpt-4o") is not None


def test_plugin_loader_subclass_and_factory(tmp_path: Path):
    """Test loading plugins that expose create_plugin() or just a class."""
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir(parents=True)

    # 1. Factory create_plugin
    factory_code = """
from llmproxy.core.interface import BasePlugin
from llmproxy.schemas import PluginMetadata

class FactoryPlugin(BasePlugin):
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(id="factory_p", name="Factory Plugin")

    async def chat_completion(self, request, model_config, variables):
        pass

def create_plugin():
    return FactoryPlugin()
"""
    (plugins_dir / "factory_p.py").write_text(factory_code)

    # 2. Bare class subclassing BasePlugin
    class_code = """
from llmproxy.core.interface import BasePlugin
from llmproxy.schemas import PluginMetadata

class BareClassPlugin(BasePlugin):
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(id="bare_class_p", name="Bare Class Plugin")

    async def chat_completion(self, request, model_config, variables):
        pass
"""
    (plugins_dir / "bare_p.py").write_text(class_code)

    mgr = PluginManager(plugins_dir=plugins_dir, data_dir=tmp_path / "data")
    mgr.discover_and_load_all()

    assert mgr.get_plugin("factory_p") is not None
    assert mgr.get_plugin("bare_class_p") is not None


@pytest.mark.anyio
async def test_all_plugins_chat_completion_non_streaming():
    """Verify all 4 provider plugins handle non-streaming chat completions with fallback."""
    cases = [
        ("antigravity", "gemini-3.8-flash"),
        ("codex", "gpt-4o"),
        ("claude_code", "claude-3-7-sonnet"),
        ("openai_compatible", "default-model"),
    ]

    for plugin_id, model_id in cases:
        plugin = plugin_manager.get_plugin(plugin_id)
        assert plugin is not None

        model_cfg = next(m for m in plugin.metadata.models if m.id == model_id)
        req = ChatCompletionRequest(
            model=model_id,
            messages=[ChatMessage(role="user", content="Hello test")],
            stream=False,
        )
        vars_dict = {"api_key": "test-key"}
        resp = await plugin.chat_completion(req, model_cfg, vars_dict)

        assert isinstance(resp, ChatCompletionResponse)
        assert resp.model == model_id
        assert len(resp.choices) >= 1
        assert resp.choices[0].finish_reason == "stop"
        assert len(resp.choices[0].message.content) > 0


@pytest.mark.anyio
async def test_all_plugins_chat_completion_streaming():
    """Verify all 4 provider plugins handle streaming chat completions with fallback."""
    cases = [
        ("antigravity", "gemini-3.8-flash"),
        ("codex", "gpt-4o"),
        ("claude_code", "claude-3-7-sonnet"),
        ("openai_compatible", "default-model"),
    ]

    for plugin_id, model_id in cases:
        plugin = plugin_manager.get_plugin(plugin_id)
        assert plugin is not None

        model_cfg = next(m for m in plugin.metadata.models if m.id == model_id)
        req = ChatCompletionRequest(
            model=model_id,
            messages=[ChatMessage(role="user", content="Stream test")],
            stream=True,
        )
        vars_dict = {"api_key": "test-key"}
        stream_iter = await plugin.chat_completion(req, model_cfg, vars_dict)

        chunks = []
        async for chunk in stream_iter:
            assert isinstance(chunk, ChatCompletionChunk)
            assert chunk.model == model_id
            chunks.append(chunk)

        assert len(chunks) >= 3
        # First chunk delta announces assistant role
        assert chunks[0].choices[0].delta.get("role") == "assistant"
        # Last chunk has finish_reason="stop"
        assert chunks[-1].choices[0].finish_reason == "stop"
