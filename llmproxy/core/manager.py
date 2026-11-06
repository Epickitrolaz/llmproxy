from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from llmproxy.core.interface import BasePlugin
from llmproxy.schemas import (
    PluginMetadata,
    PluginModelConfig,
    PluginVariable,
    RequestLogEntry,
)


class PluginManager:
    """
    Manages provider plugins loaded dynamically from the plugins/ directory.
    Contract specification for plugin discovery, persistence, model management,
    and routing.
    """

    def __init__(self, plugins_dir: Optional[Path] = None, data_dir: Optional[Path] = None):
        self.plugins_dir = plugins_dir or (Path(__file__).resolve().parent.parent.parent / "plugins")
        self.data_dir = data_dir or (Path(__file__).resolve().parent.parent.parent / "data")
        self.plugins_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.plugins: Dict[str, BasePlugin] = {}
        self.configs: Dict[str, Dict[str, Any]] = {}
        self.logs: List[RequestLogEntry] = []

    def discover_and_load_all(self) -> None:
        """Scan plugins_dir and load all valid python plugin files."""
        pass

    def load_plugin_file(self, file_path: Path) -> Optional[BasePlugin]:
        """Load or reload a single python plugin file."""
        pass

    def reload_plugin(self, plugin_id: str) -> Optional[BasePlugin]:
        """Hot-reload a plugin by its ID."""
        pass

    def get_plugin(self, plugin_id: str) -> Optional[BasePlugin]:
        """Retrieve a loaded plugin by ID."""
        return self.plugins.get(plugin_id)

    def list_plugins(self) -> List[PluginMetadata]:
        """List all loaded plugins with their current merged configuration."""
        return [p.metadata for p in self.plugins.values()]

    def update_plugin_config(
        self,
        plugin_id: str,
        enabled: Optional[bool] = None,
        variables: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update plugin variables and/or enabled status."""
        pass

    def add_custom_model(self, plugin_id: str, model_config: PluginModelConfig) -> bool:
        """Add a custom model to a plugin."""
        pass

    def update_custom_model(
        self, plugin_id: str, model_id: str, model_config: PluginModelConfig
    ) -> bool:
        """Update an existing model in a plugin."""
        pass

    def delete_custom_model(self, plugin_id: str, model_id: str) -> bool:
        """Delete/remove a model from a plugin."""
        pass

    def get_all_active_models(self) -> List[Tuple[PluginModelConfig, BasePlugin]]:
        """Return list of (model_config, plugin) for all enabled plugins and models."""
        return []

    def resolve_model(
        self, model_id: str
    ) -> Optional[Tuple[PluginModelConfig, BasePlugin, Dict[str, Any]]]:
        """Resolve a model ID to its plugin, model config, and active variables."""
        return None

    def add_log(self, entry: RequestLogEntry) -> None:
        """Add a request log entry."""
        self.logs.insert(0, entry)
        if len(self.logs) > 1000:
            self.logs.pop()

    def get_logs(self, limit: int = 100) -> List[RequestLogEntry]:
        """Retrieve recent request logs."""
        return self.logs[:limit]


# Global singleton instance
plugin_manager = PluginManager()
