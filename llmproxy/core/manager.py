from __future__ import annotations

import importlib.util
import json
import logging
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from llmproxy.core.interface import BasePlugin
from llmproxy.schemas import (
    Account,
    AccountCreate,
    AccountUpdate,
    PluginMetadata,
    PluginModelConfig,
    PluginVariable,
    RequestLogEntry,
)

logger = logging.getLogger(__name__)


class PluginManager:
    """
    Manages provider plugins loaded dynamically from the plugins/ directory.
    Contract specification for plugin discovery, persistence, model management,
    and routing.
    """

    def __init__(self, plugins_dir: Optional[Path] = None, data_dir: Optional[Path] = None):
        self.plugins_dir = Path(plugins_dir) if plugins_dir else (Path(__file__).resolve().parent.parent.parent / "plugins")
        self.data_dir = Path(data_dir) if data_dir else (Path(__file__).resolve().parent.parent.parent / "data")
        self.plugins_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.data_dir / "plugin_configs.json"
        self.accounts_file = self.data_dir / "accounts.json"

        self.plugins: Dict[str, BasePlugin] = {}
        self.plugin_files: Dict[str, Path] = {}
        self.configs: Dict[str, Dict[str, Any]] = {}
        self.accounts: Dict[str, Account] = {}
        self.rr_indices: Dict[str, int] = {}
        self.logs: List[RequestLogEntry] = []

        self._load_configs()
        self._load_accounts()

    def _load_configs(self) -> None:
        """Load persisted plugin configurations from disk."""
        if self.config_file.exists():
            try:
                content = self.config_file.read_text(encoding="utf-8")
                self.configs = json.loads(content)
            except Exception as e:
                logger.warning("Failed to load plugin configs from %s: %s", self.config_file, e)
                self.configs = {}
        else:
            self.configs = {}

    def _save_configs(self) -> None:
        """Persist current plugin configurations to disk."""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            self.config_file.write_text(json.dumps(self.configs, indent=2), encoding="utf-8")
        except Exception as e:
            logger.error("Failed to save plugin configs to %s: %s", self.config_file, e)

    def _load_accounts(self) -> None:
        """Load connected accounts from disk."""
        if self.accounts_file.exists():
            try:
                data = json.loads(self.accounts_file.read_text(encoding="utf-8"))
                self.accounts = {
                    item["id"]: Account.model_validate(item)
                    for item in data
                }
            except Exception as e:
                logger.warning("Failed to load accounts from %s: %s", self.accounts_file, e)
                self.accounts = {}
        else:
            self.accounts = {}

    def _save_accounts(self) -> None:
        """Persist accounts to disk."""
        try:
            self.accounts_file.parent.mkdir(parents=True, exist_ok=True)
            serialized = [acc.model_dump() for acc in self.accounts.values()]
            self.accounts_file.write_text(json.dumps(serialized, indent=2), encoding="utf-8")
        except Exception as e:
            logger.error("Failed to save accounts to %s: %s", self.accounts_file, e)

    # ==========================================
    # Accounts Management & Load Balancing
    # ==========================================

    def get_accounts(self, provider_id: Optional[str] = None) -> List[Account]:
        """Return all accounts, optionally filtered by provider_id."""
        if provider_id:
            return [acc for acc in self.accounts.values() if acc.provider_id == provider_id]
        return list(self.accounts.values())

    def get_account(self, account_id: str) -> Optional[Account]:
        """Retrieve an account by ID."""
        return self.accounts.get(account_id)

    def add_account(self, account_data: Union[Account, AccountCreate, Dict[str, Any]]) -> Account:
        """Add a new connected account."""
        if isinstance(account_data, Account):
            account = account_data
        elif isinstance(account_data, AccountCreate):
            account = Account(
                id=f"acc_{uuid.uuid4().hex[:8]}",
                provider_id=account_data.provider_id,
                name=account_data.name,
                api_key=account_data.api_key,
                base_url=account_data.base_url,
                priority=account_data.priority,
                weight=account_data.weight,
                enabled=account_data.enabled,
                extra_config=account_data.extra_config,
            )
        else:
            acc_dict = dict(account_data)
            if "id" not in acc_dict:
                acc_dict["id"] = f"acc_{uuid.uuid4().hex[:8]}"
            account = Account.model_validate(acc_dict)

        self.accounts[account.id] = account
        self._save_accounts()
        return account

    def update_account(
        self, account_id: str, updates: Union[AccountUpdate, Dict[str, Any]]
    ) -> Optional[Account]:
        """Update account properties."""
        account = self.accounts.get(account_id)
        if not account:
            return None

        upd = updates.model_dump(exclude_unset=True) if isinstance(updates, AccountUpdate) else updates
        for k, v in upd.items():
            if v is not None and hasattr(account, k):
                setattr(account, k, v)

        self._save_accounts()
        return account

    def delete_account(self, account_id: str) -> bool:
        """Delete an account."""
        if account_id in self.accounts:
            del self.accounts[account_id]
            self._save_accounts()
            return True
        return False

    def reset_account_cooldown(self, account_id: str) -> bool:
        """Clear rate limit / cooldown status on an account."""
        account = self.accounts.get(account_id)
        if not account:
            return False
        account.rate_limited_until = None
        self._save_accounts()
        return True

    def select_account(
        self, provider_id: str, strategy: Optional[str] = None
    ) -> Optional[Account]:
        """
        Select an available account based on load balancing strategy:
        - round_robin: cyclically cycles through active, non-rate-limited accounts
        - least_used: picks account with lowest success count
        - random: picks a random active account
        - fallback/priority: picks highest priority (lowest number) available
        """
        all_accounts = [a for a in self.accounts.values() if a.provider_id == provider_id]
        if not all_accounts:
            return None

        available = [a for a in all_accounts if a.is_available()]
        if not available:
            return None

        strat = (strategy or "round_robin").lower().replace("-", "_")

        if strat == "least_used":
            return min(available, key=lambda a: (a.success_count, a.last_used_at or 0))
        elif strat == "random":
            return random.choice(available)
        elif strat in ("fallback", "priority"):
            return sorted(available, key=lambda a: (a.priority, a.last_used_at or 0))[0]
        else:
            # Default: round_robin
            idx = self.rr_indices.get(provider_id, 0)
            selected = available[idx % len(available)]
            self.rr_indices[provider_id] = (idx + 1) % len(available)
            return selected

    def mark_account_result(
        self,
        account_id: str,
        success: bool,
        is_rate_limit: bool = False,
        cooldown_seconds: float = 60.0,
    ) -> None:
        """Record success or error/rate-limit result for an account."""
        account = self.accounts.get(account_id)
        if not account:
            return
        if success:
            account.success_count += 1
            account.last_used_at = time.time()
        else:
            account.error_count += 1
            if is_rate_limit:
                account.rate_limited_until = time.time() + cooldown_seconds
        self._save_accounts()

    def discover_and_load_all(self) -> None:
        """Scan plugins_dir and load all valid python plugin files."""
        if not self.plugins_dir.exists():
            return

        for file_path in sorted(self.plugins_dir.glob("*.py")):
            if file_path.name.startswith("_") or file_path.name.startswith("."):
                continue
            try:
                self.load_plugin_file(file_path)
            except Exception as e:
                logger.error("Failed to load plugin from %s: %s", file_path, e)

    def load_plugin_file(self, file_path: Path) -> Optional[BasePlugin]:
        """Load or reload a single python plugin file."""
        if not file_path.exists() or not file_path.is_file():
            return None

        importlib.invalidate_caches()
        module_name = f"llmproxy_plugins.{file_path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, str(file_path))
        if spec is None:
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        code = compile(file_path.read_text(encoding="utf-8"), str(file_path), "exec")
        exec(code, module.__dict__)

        # 1. Check for `plugin` instance
        plugin_instance: Optional[BasePlugin] = None
        if hasattr(module, "plugin") and isinstance(module.plugin, BasePlugin):
            plugin_instance = module.plugin
        # 2. Check for `create_plugin()` factory function
        elif hasattr(module, "create_plugin") and callable(module.create_plugin):
            created = module.create_plugin()
            if isinstance(created, BasePlugin):
                plugin_instance = created
        # 3. Check for any class inheriting BasePlugin defined in this module
        if plugin_instance is None:
            for attr_name in dir(module):
                val = getattr(module, attr_name)
                if (
                    isinstance(val, type)
                    and issubclass(val, BasePlugin)
                    and val is not BasePlugin
                    and getattr(val, "__module__", None) == module.__name__
                ):
                    try:
                        plugin_instance = val()
                        break
                    except Exception:
                        continue

        # Fallback if class was defined without matching __module__
        if plugin_instance is None:
            for attr_name in dir(module):
                val = getattr(module, attr_name)
                if isinstance(val, type) and issubclass(val, BasePlugin) and val is not BasePlugin:
                    try:
                        plugin_instance = val()
                        break
                    except Exception:
                        continue

        if plugin_instance is None:
            logger.warning("No BasePlugin instance or subclass found in %s", file_path)
            return None

        # Set filename in metadata if writable
        try:
            plugin_instance.metadata.filename = file_path.name
        except Exception:
            pass
        setattr(plugin_instance, "_filename", file_path.name)
        plugin_id = plugin_instance.metadata.id

        self.plugins[plugin_id] = plugin_instance
        self.plugin_files[plugin_id] = file_path

        # Initialize config entry if missing
        if plugin_id not in self.configs:
            default_vars = {
                v.name: v.default
                for v in plugin_instance.metadata.variables
                if v.default is not None
            }
            self.configs[plugin_id] = {
                "enabled": plugin_instance.metadata.enabled,
                "variables": default_vars,
                "custom_models": [],
                "edited_models": {},
                "deleted_models": [],
            }
            self._save_configs()
        else:
            cfg = self.configs[plugin_id]
            cfg.setdefault("enabled", plugin_instance.metadata.enabled)
            cfg.setdefault("variables", {})
            cfg.setdefault("custom_models", [])
            cfg.setdefault("edited_models", {})
            cfg.setdefault("deleted_models", [])

        return plugin_instance

    def reload_plugin(self, plugin_id: str) -> Optional[BasePlugin]:
        """Hot-reload a plugin by its ID."""
        file_path = self.plugin_files.get(plugin_id)
        if not file_path or not file_path.exists():
            file_path = self.plugins_dir / f"{plugin_id}.py"

        if file_path.exists():
            try:
                return self.load_plugin_file(file_path)
            except Exception as e:
                logger.error("Failed to reload plugin %s from %s: %s", plugin_id, file_path, e)
                return None
        return None

    def get_plugin(self, plugin_id: str) -> Optional[BasePlugin]:
        """Retrieve a loaded plugin by ID."""
        return self.plugins.get(plugin_id)

    def get_plugin_models(self, plugin_id: str) -> List[PluginModelConfig]:
        """Retrieve configured models for a plugin (defaults - deleted + edited + custom)."""
        plugin = self.plugins.get(plugin_id)
        if not plugin:
            return []

        cfg = self.configs.get(plugin_id, {})
        deleted = set(cfg.get("deleted_models", []))
        edited = cfg.get("edited_models", {})
        custom = cfg.get("custom_models", [])

        result: List[PluginModelConfig] = []
        seen_ids = set()

        for default_m in plugin.metadata.models:
            if default_m.id in deleted:
                continue
            if default_m.id in edited:
                model_obj = PluginModelConfig.model_validate(edited[default_m.id])
            else:
                model_obj = default_m.model_copy()
            result.append(model_obj)
            seen_ids.add(model_obj.id)

        for c in custom:
            model_obj = PluginModelConfig.model_validate(c)
            if model_obj.id not in seen_ids:
                result.append(model_obj)
                seen_ids.add(model_obj.id)

        return result

    def get_plugin_variables(self, plugin_id: str) -> Dict[str, Any]:
        """Return merged dictionary of variable values for a plugin."""
        plugin = self.plugins.get(plugin_id)
        if not plugin:
            return {}

        cfg = self.configs.get(plugin_id, {})
        stored_vars = cfg.get("variables", {})
        merged: Dict[str, Any] = {}

        for var in plugin.metadata.variables:
            if var.name in stored_vars and stored_vars[var.name] is not None:
                merged[var.name] = stored_vars[var.name]
            elif var.value != "":
                merged[var.name] = var.value
            else:
                merged[var.name] = var.default

        for k, v in stored_vars.items():
            if k not in merged:
                merged[k] = v

        return merged

    def list_plugins(self) -> List[PluginMetadata]:
        """List all loaded plugins with their current merged configuration."""
        result: List[PluginMetadata] = []
        for plugin in self.plugins.values():
            cfg = self.configs.get(plugin.metadata.id, {})
            enabled = cfg.get("enabled", plugin.metadata.enabled)
            models = self.get_plugin_models(plugin.metadata.id)
            vars_dict = self.get_plugin_variables(plugin.metadata.id)

            variables: List[PluginVariable] = []
            declared_names = set()
            for v in plugin.metadata.variables:
                v_copy = v.model_copy()
                v_copy.value = vars_dict.get(v.name, v.default)
                variables.append(v_copy)
                declared_names.add(v.name)

            for k, v in vars_dict.items():
                if k not in declared_names:
                    variables.append(PluginVariable(name=k, label=k, value=v, default=v))

            file_path = self.plugin_files.get(plugin.metadata.id)
            filename = file_path.name if file_path else plugin.metadata.filename

            meta_copy = plugin.metadata.model_copy(update={
                "enabled": enabled,
                "filename": filename,
                "variables": variables,
                "models": models,
            })
            result.append(meta_copy)
        return result

    def update_plugin_config(
        self,
        plugin_id: str,
        enabled: Optional[bool] = None,
        variables: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update plugin variables and/or enabled status."""
        if plugin_id not in self.plugins and plugin_id not in self.configs:
            return False

        cfg = self.configs.setdefault(
            plugin_id,
            {
                "enabled": True,
                "variables": {},
                "custom_models": [],
                "edited_models": {},
                "deleted_models": [],
            },
        )
        if enabled is not None:
            cfg["enabled"] = enabled
        if variables is not None:
            cfg.setdefault("variables", {}).update(variables)

        self._save_configs()
        return True

    def add_custom_model(self, plugin_id: str, model_config: PluginModelConfig) -> bool:
        """Add a custom model to a plugin, or un-delete / edit a default model with that ID."""
        if plugin_id not in self.plugins:
            return False

        cfg = self.configs.setdefault(
            plugin_id,
            {
                "enabled": True,
                "variables": {},
                "custom_models": [],
                "edited_models": {},
                "deleted_models": [],
            },
        )
        custom_models = cfg.setdefault("custom_models", [])
        deleted = cfg.setdefault("deleted_models", [])
        if model_config.id in deleted:
            deleted.remove(model_config.id)

        default_model_ids = {m.id for m in self.plugins[plugin_id].metadata.models}
        if model_config.id in default_model_ids:
            cfg.setdefault("edited_models", {})[model_config.id] = model_config.model_dump()
            self._save_configs()
            return True

        for i, m in enumerate(custom_models):
            if m.get("id") == model_config.id:
                custom_models[i] = model_config.model_dump()
                break
        else:
            custom_models.append(model_config.model_dump())

        self._save_configs()
        return True

    def update_custom_model(
        self, plugin_id: str, model_id: str, model_config: PluginModelConfig
    ) -> bool:
        """Update an existing model in a plugin."""
        if plugin_id not in self.plugins:
            return False

        cfg = self.configs.setdefault(
            plugin_id,
            {
                "enabled": True,
                "variables": {},
                "custom_models": [],
                "edited_models": {},
                "deleted_models": [],
            },
        )
        custom_models = cfg.setdefault("custom_models", [])
        for i, m in enumerate(custom_models):
            if m.get("id") == model_id:
                custom_models[i] = model_config.model_dump()
                self._save_configs()
                return True

        default_model_ids = {m.id for m in self.plugins[plugin_id].metadata.models}
        if model_id in default_model_ids:
            cfg.setdefault("edited_models", {})[model_id] = model_config.model_dump()
            deleted = cfg.setdefault("deleted_models", [])
            if model_id in deleted:
                deleted.remove(model_id)
            self._save_configs()
            return True

        custom_models.append(model_config.model_dump())
        self._save_configs()
        return True

    def delete_custom_model(self, plugin_id: str, model_id: str) -> bool:
        """Delete/remove a model from a plugin."""
        if plugin_id not in self.plugins:
            return False

        cfg = self.configs.setdefault(
            plugin_id,
            {
                "enabled": True,
                "variables": {},
                "custom_models": [],
                "edited_models": {},
                "deleted_models": [],
            },
        )
        custom_models = cfg.setdefault("custom_models", [])
        cfg["custom_models"] = [m for m in custom_models if m.get("id") != model_id]

        if model_id in cfg.get("edited_models", {}):
            del cfg["edited_models"][model_id]

        default_model_ids = {m.id for m in self.plugins[plugin_id].metadata.models}
        if model_id in default_model_ids:
            deleted = cfg.setdefault("deleted_models", [])
            if model_id not in deleted:
                deleted.append(model_id)

        self._save_configs()
        return True

    def get_all_active_models(self) -> List[Tuple[PluginModelConfig, BasePlugin]]:
        """Return list of (model_config, plugin) for all enabled plugins and models."""
        active: List[Tuple[PluginModelConfig, BasePlugin]] = []
        for plugin_id, plugin in self.plugins.items():
            cfg = self.configs.get(plugin_id, {})
            if not cfg.get("enabled", plugin.metadata.enabled):
                continue
            models = self.get_plugin_models(plugin_id)
            for m in models:
                if m.enabled:
                    active.append((m, plugin))
        return active

    def resolve_model(
        self, model_id: str
    ) -> Optional[Tuple[PluginModelConfig, BasePlugin, Dict[str, Any]]]:
        """Resolve a model ID to its plugin, model config, and active variables."""
        for model_config, plugin in self.get_all_active_models():
            if model_config.id == model_id:
                vars_dict = self.get_plugin_variables(plugin.metadata.id)
                return (model_config, plugin, vars_dict)
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
