"""Model snapshots must not hide defaults added by a plugin update."""

import json
from pathlib import Path

from llmproxy.core.manager import PluginManager


PLUGIN_TEMPLATE = '''
from llmproxy.core.interface import BasePlugin
from llmproxy.schemas import PluginMetadata, PluginModelConfig

class SnapshotPlugin(BasePlugin):
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            id="snapshot_test",
            name="Snapshot Test",
            version="{version}",
            models=[{models}],
        )

    async def chat_completion(self, request, model_config, variables, account=None):
        raise NotImplementedError

plugin = SnapshotPlugin()
'''


def _write_plugin(path: Path, version: str, model_ids: list[str]) -> None:
    models = ", ".join(
        f'PluginModelConfig(id="{model_id}", name="{model_id}")' for model_id in model_ids
    )
    path.write_text(PLUGIN_TEMPLATE.format(version=version, models=models), encoding="utf-8")


def _write_config(data_dir: Path, config: dict) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "plugin_configs.json").write_text(
        json.dumps({"snapshot_test": config}, indent=2), encoding="utf-8"
    )


def test_models_added_by_plugin_update_stay_visible(tmp_path: Path):
    """A model introduced in a new template version survives the old snapshot logic."""
    plugins_dir = tmp_path / "plugins"
    data_dir = tmp_path / "data"
    plugins_dir.mkdir()
    plugin_file = plugins_dir / "snapshot_test.py"

    _write_plugin(plugin_file, "1.0.0", ["model-a"])

    manager = PluginManager(plugins_dir=plugins_dir, data_dir=data_dir)
    manager.discover_and_load_all()
    manager._save_configs()

    saved = json.loads((data_dir / "plugin_configs.json").read_text(encoding="utf-8"))
    assert saved["snapshot_test"]["default_model_ids"] == ["model-a"]
    assert [m["id"] for m in saved["snapshot_test"]["models"]] == ["model-a"]

    _write_plugin(plugin_file, "2.0.0", ["model-a", "model-a:high"])
    manager.reload_plugin("snapshot_test")

    model_ids = [m.id for m in manager.get_plugin_models("snapshot_test")]
    assert model_ids == ["model-a", "model-a:high"]


def test_reasoning_levels_are_refreshed_after_template_update(tmp_path: Path):
    """Capability metadata added to a default model is not hidden by a stale edit."""
    plugins_dir = tmp_path / "plugins"
    data_dir = tmp_path / "data"
    plugins_dir.mkdir()
    plugin_file = plugins_dir / "snapshot_test.py"
    _write_plugin(plugin_file, "1.0.0", ["model-a"])

    manager = PluginManager(plugins_dir=plugins_dir, data_dir=data_dir)
    manager.discover_and_load_all()
    manager._save_configs()

    plugin_file.write_text(
        PLUGIN_TEMPLATE.format(version="2.0.0", models=(
            'PluginModelConfig(id="model-a", name="model-a", '
            'reasoning_levels=["low", "high"])'
        )),
        encoding="utf-8",
    )
    manager.reload_plugin("snapshot_test")

    models = {m.id: m for m in manager.get_plugin_models("snapshot_test")}
    assert models["model-a"].reasoning_levels == ["low", "high"]


def test_legacy_generated_deletions_are_repaired(tmp_path: Path):
    """The generated state that hid newly added defaults is cleared once."""
    plugins_dir = tmp_path / "plugins"
    data_dir = tmp_path / "data"
    plugins_dir.mkdir()
    plugin_file = plugins_dir / "snapshot_test.py"
    _write_plugin(plugin_file, "2.0.0", ["model-a", "model-a:high"])

    defaults = PluginManager(plugins_dir=plugins_dir, data_dir=tmp_path / "scratch")
    defaults.discover_and_load_all()
    model_a = defaults.get_plugin("snapshot_test").metadata.models[0].model_dump()

    # Exactly what the previous release wrote after seeing the old model-a-only snapshot.
    _write_config(
        data_dir,
        {
            "enabled": True,
            "variables": {},
            "custom_models": [],
            "models": [model_a],
            "edited_models": {"model-a": model_a},
            "deleted_models": ["model-a:high"],
        },
    )

    manager = PluginManager(plugins_dir=plugins_dir, data_dir=data_dir)
    manager.discover_and_load_all()

    model_ids = [m.id for m in manager.get_plugin_models("snapshot_test")]
    assert model_ids == ["model-a", "model-a:high"]
    assert manager.configs["snapshot_test"]["deleted_models"] == []
    assert manager.configs["snapshot_test"]["edited_models"] == {}

    # The repair is persisted, so a restart sees the full default set directly.
    saved = json.loads((data_dir / "plugin_configs.json").read_text(encoding="utf-8"))
    saved_cfg = saved["snapshot_test"]
    assert saved_cfg["deleted_models"] == []
    assert saved_cfg["default_model_ids"] == ["model-a", "model-a:high"]
    assert [m["id"] for m in saved_cfg["models"]] == ["model-a", "model-a:high"]


def test_genuine_user_deletion_is_preserved(tmp_path: Path):
    """Repair only targets generated snapshots, not real deleted or edited models."""
    plugins_dir = tmp_path / "plugins"
    data_dir = tmp_path / "data"
    plugins_dir.mkdir()
    plugin_file = plugins_dir / "snapshot_test.py"
    _write_plugin(plugin_file, "2.0.0", ["model-a", "model-b"])

    defaults = PluginManager(plugins_dir=plugins_dir, data_dir=tmp_path / "scratch")
    defaults.discover_and_load_all()
    model_a = defaults.get_plugin("snapshot_test").metadata.models[0].model_dump()

    # User kept the default for model-a but deliberately deleted model-b.
    _write_config(
        data_dir,
        {
            "enabled": True,
            "variables": {},
            "custom_models": [],
            "models": [model_a],
            "edited_models": {},
            "deleted_models": ["model-b"],
        },
    )

    manager = PluginManager(plugins_dir=plugins_dir, data_dir=data_dir)
    manager.discover_and_load_all()

    model_ids = [m.id for m in manager.get_plugin_models("snapshot_test")]
    assert model_ids == ["model-a"]
