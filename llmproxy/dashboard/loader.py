from __future__ import annotations

import importlib.util
import inspect
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from llmproxy.core.interface import BasePlugin
from llmproxy.schemas import (
    PluginMetadata,
    PluginModelConfig,
    PluginVariable,
)

logger = logging.getLogger("llmproxy.dashboard.loader")

CONFIG_FILE_NAME = "plugin_configs.json"


def is_safe_path(base_dir: Path, target_path: Path) -> bool:
    """Check if target_path resolves strictly within base_dir to prevent path traversal."""
    try:
        target_path.resolve().relative_to(base_dir.resolve())
        return True
    except (ValueError, RuntimeError):
        return False


def sanitize_plugin_id(plugin_id: str) -> str:
    """Strip dangerous characters from plugin ID or filename."""
    import re
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "_", plugin_id).strip(".")


def validate_python_syntax(code: str, filename: str = "<plugin>") -> None:
    """Validate that the provided code string is syntactically valid Python."""
    try:
        compile(code, filename, "exec")
    except SyntaxError as e:
        raise ValueError(f"Syntax error on line {e.lineno}: {e.msg}") from e


def get_configs_file_path(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / CONFIG_FILE_NAME


def read_persisted_configs(data_dir: Path) -> Dict[str, Any]:
    config_file = get_configs_file_path(data_dir)
    if not config_file.exists():
        return {}
    try:
        return json.loads(config_file.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to read persisted configs from %s: %s", config_file, e)
        return {}


def write_persisted_configs(data_dir: Path, configs: Dict[str, Any]) -> None:
    config_file = get_configs_file_path(data_dir)
    try:
        temp_file = config_file.with_suffix(".tmp")
        temp_file.write_text(json.dumps(configs, indent=2), encoding="utf-8")
        temp_file.replace(config_file)
    except Exception as e:
        logger.error("Failed to write persisted configs: %s", e)


def apply_persisted_config_to_plugin(plugin: BasePlugin, saved_config: Dict[str, Any]) -> None:
    """Apply persisted variables, enabled status, and custom models to a plugin instance."""
    if not saved_config:
        return

    meta = plugin.metadata

    if "enabled" in saved_config and saved_config["enabled"] is not None:
        meta.enabled = bool(saved_config["enabled"])

    saved_vars = saved_config.get("variables", {})
    if isinstance(saved_vars, dict):
        for var in meta.variables:
            if var.name in saved_vars:
                var.value = saved_vars[var.name]

    if "models" in saved_config and isinstance(saved_config["models"], list):
        parsed_models = []
        for model_dict in saved_config["models"]:
            try:
                parsed_models.append(PluginModelConfig.model_validate(model_dict))
            except Exception as e:
                logger.warning("Failed to deserialize saved model: %s", e)
        meta.models = parsed_models


def load_plugin_from_file(file_path: Path, data_dir: Optional[Path] = None) -> BasePlugin:
    """Dynamically load an isolated python plugin file, instantiate it, and apply configs."""
    if not file_path.exists():
        raise FileNotFoundError(f"Plugin file not found: {file_path}")

    code = file_path.read_text(encoding="utf-8")
    validate_python_syntax(code, file_path.name)

    base_module_name = f"llmproxy_plugin_{file_path.stem}"
    to_remove = [k for k in sys.modules if k.startswith(base_module_name)]
    for k in to_remove:
        sys.modules.pop(k, None)

    module_name = f"{base_module_name}_{int(time.time() * 1000)}"
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if not spec or not spec.loader:
        raise ImportError(f"Cannot load module spec from {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        sys.modules.pop(module_name, None)
        raise RuntimeError(f"Error executing plugin file {file_path.name}: {e}") from e

    # Discover BasePlugin instance or subclass
    plugin_instance: Optional[BasePlugin] = None

    # 1. Look for explicit instance variable 'plugin'
    if hasattr(module, "plugin") and isinstance(module.plugin, BasePlugin):
        plugin_instance = module.plugin

    # 2. Look for any BasePlugin instance
    if not plugin_instance:
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, BasePlugin):
                plugin_instance = attr
                break

    # 3. Look for BasePlugin subclass and instantiate
    if not plugin_instance:
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if inspect.isclass(attr) and issubclass(attr, BasePlugin) and attr is not BasePlugin:
                try:
                    plugin_instance = attr()
                    break
                except Exception as e:
                    raise RuntimeError(f"Failed to instantiate {attr_name}: {e}") from e

    if not plugin_instance:
        sys.modules.pop(module_name, None)
        raise ValueError(
            f"Plugin file {file_path.name} does not export a BasePlugin subclass or instance"
        )

    meta = plugin_instance.metadata
    meta.filename = file_path.name
    plugin_instance._cached_metadata = meta

    class _PluginWrapper(plugin_instance.__class__):
        @property
        def metadata(self) -> PluginMetadata:
            return self._cached_metadata

    plugin_instance.__class__ = _PluginWrapper

    if data_dir:
        saved_configs = read_persisted_configs(data_dir)
        plugin_id = plugin_instance.metadata.id
        if plugin_id in saved_configs:
            apply_persisted_config_to_plugin(plugin_instance, saved_configs[plugin_id])

    return plugin_instance


def find_plugin_file_path(plugin_manager: Any, plugin_id: str) -> Optional[Path]:
    """Find the path of a plugin file on disk, ensuring it resolves safely within plugins_dir."""
    plugins_dir = Path(plugin_manager.plugins_dir)
    safe_id = sanitize_plugin_id(plugin_id)

    plugin = plugin_manager.plugins.get(plugin_id)
    if plugin and plugin.metadata.filename:
        direct_path = (plugins_dir / plugin.metadata.filename).resolve()
        if is_safe_path(plugins_dir, direct_path) and direct_path.exists():
            return direct_path

    # Try matching filename: {safe_id}.py
    named_path = (plugins_dir / f"{safe_id}.py").resolve()
    if is_safe_path(plugins_dir, named_path) and named_path.exists():
        return named_path

    # Scan all .py files in plugins_dir to match metadata id
    for py_file in plugins_dir.glob("*.py"):
        if py_file.name.startswith("__"):
            continue
        try:
            resolved_file = py_file.resolve()
            if not is_safe_path(plugins_dir, resolved_file):
                continue
            temp_plugin = load_plugin_from_file(resolved_file)
            if temp_plugin.metadata.id == plugin_id:
                return resolved_file
        except Exception:
            continue

    return None
