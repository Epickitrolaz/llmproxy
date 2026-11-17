"""Built-in templates expose namespaced model IDs and reasoning metadata."""

from __future__ import annotations

from pathlib import Path

from llmproxy.api.router import _model_card
from llmproxy.core.manager import plugin_manager
from llmproxy.dashboard.loader import load_plugin_from_file
from llmproxy.schemas import (
    PluginMetadata,
    PluginModelConfig,
    PluginVariable,
)

ROOT = Path(__file__).resolve().parent.parent

EXPECTED_PREFIXES = {
    "anthropic_proxy": "ap",
    "antigravity": "ag",
    "codex": "cx",
    "gemini": "gm",
    "mock_provider": "mock",
    "openai_compatible": "compat",
    "opencode_free": "oc",
    "opencode_go": "ocg",
    "openrouter": "or",
}

REASONING_TEMPLATES = {
    "anthropic_proxy",
    "antigravity",
    "codex",
    "gemini",
    "opencode_free",
    "opencode_go",
    "openrouter",
}


def test_builtin_templates_use_provider_prefixed_model_ids():
    for plugin_id, prefix in EXPECTED_PREFIXES.items():
        plugin = load_plugin_from_file(ROOT / "templates" / f"{plugin_id}.py")
        assert plugin.metadata.id == plugin_id
        assert plugin.metadata.models

        for model in plugin.metadata.models:
            assert model.id.startswith(f"{prefix}/"), (plugin_id, model.id)
            assert model.upstream_model, (plugin_id, model.id)
            assert not model.upstream_model.startswith(f"{prefix}/"), (plugin_id, model.id)


def test_reasoning_templates_expose_levels_on_model_cards():
    for plugin_id in REASONING_TEMPLATES:
        plugin = load_plugin_from_file(ROOT / "templates" / f"{plugin_id}.py")
        assert any(model.reasoning_levels for model in plugin.metadata.models), plugin_id


def test_model_card_uses_provider_reasoning_fallback():
    class _Provider:
        metadata = PluginMetadata(
            id="custom_provider",
            name="Custom Provider",
            variables=[
                PluginVariable(
                    name="reasoning_effort",
                    label="Reasoning Effort",
                    type="select",
                    default="default",
                    options=["default", "low", "medium", "high", "max"],
                )
            ],
        )

    model = PluginModelConfig(
        id="custom/plain-model",
        name="Plain Model",
        upstream_model="plain-model",
        reasoning_levels=[],
    )
    card = _model_card(model, "custom_provider", _Provider())
    assert card.reasoning_levels == ["low", "medium", "high", "max"]

    explicit = PluginModelConfig(
        id="custom/explicit-model",
        name="Explicit Model",
        upstream_model="explicit-model",
        reasoning_levels=["none", "high"],
    )
    explicit_card = _model_card(explicit, "custom_provider", _Provider())
    assert explicit_card.reasoning_levels == ["none", "high"]


def test_prefixed_reasoning_suffix_resolves_to_base_model():
    plugin_manager.discover_and_load_all()
    resolved = plugin_manager.resolve_model("cx/gpt-6-luna:high")
    assert resolved is not None
    model, _plugin, variables = resolved
    assert model.id == "cx/gpt-6-luna"
    assert model.upstream_model == "gpt-6-luna"
    assert variables["reasoning_effort"] == "high"

    resolved_free = plugin_manager.resolve_model("or/google/gemma-4-31b-it:free:high")
    assert resolved_free is not None
    free_model, _plugin, free_variables = resolved_free
    assert free_model.id == "or/google/gemma-4-31b-it:free"
    assert free_model.upstream_model == "google/gemma-4-31b-it:free"
    assert free_variables["reasoning_effort"] == "high"
