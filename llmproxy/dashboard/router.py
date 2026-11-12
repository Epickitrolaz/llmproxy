from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from llmproxy.core.antigravity_oauth import (
    ANTIGRAVITY_CLIENT_ID,
    ANTIGRAVITY_SCOPES,
    CLOUDCODE_PA_ENDPOINTS,
    antigravity_oauth,
)
from llmproxy.core.codex_oauth import (
    CODEX_BACKEND_API,
    CODEX_CLIENT_ID,
    CODEX_SCOPES,
    codex_oauth,
)
from llmproxy.core.manager import plugin_manager
from llmproxy.core.security import redact_account, redact_api_key, require_admin
from llmproxy.dashboard.loader import (
    apply_persisted_config_to_plugin,
    find_plugin_file_path,
    is_safe_path,
    load_plugin_from_file,
    read_persisted_configs,
    validate_python_syntax,
    write_persisted_configs,
)
from llmproxy.dashboard.templates import STARTER_TEMPLATES, get_starter_templates
from llmproxy.schemas import (
    Account,
    AccountCreate,
    AccountUpdate,
    APIKey,
    APIKeyCreate,
    APIKeyUpdate,
    ChatMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    LoadBalancingRule,
    LoadBalancingRuleCreate,
    LoadBalancingRuleUpdate,
    PluginConfigUpdate,
    PluginMetadata,
    PluginModelConfig,
    PluginSourceUpdate,
    Proxy,
    ProxyCreate,
    ProxyUpdate,
    RequestLogEntry,
)

router = APIRouter(
    prefix="/api",
    tags=["dashboard"],
    dependencies=[Depends(require_admin)],
)


class PluginCreateRequest(BaseModel):
    filename: str
    code: str


class PluginModelUpdate(BaseModel):
    name: Optional[str] = None
    upstream_model: Optional[str] = None
    enabled: Optional[bool] = None
    context_window: Optional[int] = None
    max_tokens: Optional[int] = None
    parameters: Optional[Dict[str, Any]] = None


class TestChatRequest(BaseModel):
    model: str
    prompt: Optional[str] = None
    messages: Optional[List[ChatMessage]] = None
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False


def _ensure_plugins_loaded() -> None:
    """Ensure plugins from plugins_dir are discovered if plugins dictionary is empty."""
    plugins_dir = Path(plugin_manager.plugins_dir)
    if not plugins_dir.exists():
        return

    # If plugin_manager.discover_and_load_all() works, it will populate plugins
    if not plugin_manager.plugins:
        try:
            plugin_manager.discover_and_load_all()
        except Exception:
            pass

    # Fallback discovery if still empty
    if not plugin_manager.plugins:
        for py_file in plugins_dir.glob("*.py"):
            if py_file.name.startswith("__"):
                continue
            try:
                plugin = load_plugin_from_file(py_file, data_dir=plugin_manager.data_dir)
                plugin_manager.plugins[plugin.metadata.id] = plugin
            except Exception:
                continue


def _persist_plugin_metadata(plugin_id: str, meta: PluginMetadata) -> None:
    """Save plugin configuration to persistent storage in sync with plugin_manager."""
    cfg = plugin_manager.configs.setdefault(plugin_id, {
        "enabled": meta.enabled,
        "pinned": getattr(meta, "pinned", False),
        "variables": {},
        "custom_models": [],
        "edited_models": {},
        "deleted_models": [],
    })
    cfg["enabled"] = meta.enabled
    cfg["pinned"] = getattr(meta, "pinned", False)

    vars_dict = cfg.setdefault("variables", {})
    for v in meta.variables:
        if v.value is not None and v.value != "":
            vars_dict[v.name] = v.value
        elif v.name not in vars_dict and v.default is not None and v.default != "":
            vars_dict[v.name] = v.default

    cfg["models"] = [m.model_dump() if hasattr(m, "model_dump") else m.dict() for m in meta.models]
    plugin_manager._save_configs()

    try:
        saved_configs = read_persisted_configs(plugin_manager.data_dir)
        saved_configs[plugin_id] = cfg
        write_persisted_configs(plugin_manager.data_dir, saved_configs)
    except Exception:
        pass


# =========================================================================
# Plugins Management
# =========================================================================

@router.get("/plugins", response_model=List[PluginMetadata])
async def list_plugins():
    """List all loaded plugins with their metadata, variables, models, and enabled status."""
    _ensure_plugins_loaded()
    plugins = plugin_manager.list_plugins()
    if not plugins and plugin_manager.plugins:
        plugins = [p.metadata for p in plugin_manager.plugins.values()]
    for p in plugins:
        if not p.filename:
            fp = find_plugin_file_path(plugin_manager, p.id)
            if fp:
                p.filename = fp.name
    return plugins


@router.get("/plugins/{plugin_id}/code")
async def get_plugin_code(plugin_id: str):
    """Retrieve raw Python source code of the plugin file."""
    _ensure_plugins_loaded()
    file_path = find_plugin_file_path(plugin_manager, plugin_id)
    if not file_path or not file_path.exists():
        raise HTTPException(
            status_code=404, detail=f"Source file for plugin '{plugin_id}' not found"
        )

    code = file_path.read_text(encoding="utf-8")
    return {
        "plugin_id": plugin_id,
        "filename": file_path.name,
        "code": code,
    }


@router.put("/plugins/{plugin_id}/code")
async def update_plugin_code(plugin_id: str, body: PluginSourceUpdate):
    """Update/overwrite plugin Python source code and trigger hot reload."""
    _ensure_plugins_loaded()
    code = body.code
    if not code or not code.strip():
        raise HTTPException(status_code=400, detail="Plugin code cannot be empty")

    file_path = find_plugin_file_path(plugin_manager, plugin_id)
    plugins_dir = Path(plugin_manager.plugins_dir)
    filename = file_path.name if file_path else f"{plugin_id}.py"
    target_path = (plugins_dir / filename).resolve()

    if not is_safe_path(plugins_dir, target_path):
        raise HTTPException(status_code=400, detail="Invalid plugin file path")

    try:
        validate_python_syntax(code, filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    # Write source code to disk
    target_path.write_text(code, encoding="utf-8")

    # Trigger hot-reload
    reloaded_plugin = None
    try:
        reloaded_plugin = plugin_manager.reload_plugin(plugin_id)
    except Exception:
        pass

    if reloaded_plugin is None:
        try:
            reloaded_plugin = load_plugin_from_file(target_path, data_dir=plugin_manager.data_dir)
            plugin_manager.plugins[reloaded_plugin.metadata.id] = reloaded_plugin
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Plugin saved, but reload failed: {e}",
            ) from e

    meta = reloaded_plugin.metadata.model_copy(update={"filename": filename})
    _persist_plugin_metadata(reloaded_plugin.metadata.id, meta)

    return {
        "status": "ok",
        "message": f"Plugin '{plugin_id}' code updated and hot-reloaded successfully",
        "plugin": meta,
    }


@router.post("/plugins")
async def create_plugin(body: PluginCreateRequest):
    """Create or upload a new plugin Python file and load it."""
    filename = Path(body.filename).name.strip()
    if not filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    if not filename.endswith(".py"):
        filename = f"{filename}.py"

    code = body.code
    if not code or not code.strip():
        raise HTTPException(status_code=400, detail="Plugin code cannot be empty")

    try:
        validate_python_syntax(code, filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    plugins_dir = Path(plugin_manager.plugins_dir)
    target_path = (plugins_dir / filename).resolve()
    if not is_safe_path(plugins_dir, target_path):
        raise HTTPException(status_code=400, detail="Invalid plugin filename")

    target_path.write_text(code, encoding="utf-8")

    plugin = None
    try:
        plugin = plugin_manager.load_plugin_file(target_path)
    except Exception:
        pass

    if plugin is None:
        try:
            plugin = load_plugin_from_file(target_path, data_dir=plugin_manager.data_dir)
            plugin_manager.plugins[plugin.metadata.id] = plugin
        except Exception as e:
            if target_path.exists():
                target_path.unlink()
            raise HTTPException(
                status_code=400,
                detail=f"Failed to load new plugin: {e}",
            ) from e

    meta = plugin.metadata.model_copy(update={"filename": filename})
    _persist_plugin_metadata(plugin.metadata.id, meta)

    return {
        "status": "ok",
        "message": f"Plugin '{plugin.metadata.id}' created and loaded successfully",
        "plugin": meta,
    }


@router.delete("/plugins/{plugin_id}")
async def delete_plugin(plugin_id: str):
    """Delete the plugin file from disk and unregister it."""
    _ensure_plugins_loaded()
    plugin = plugin_manager.plugins.get(plugin_id)
    file_path = find_plugin_file_path(plugin_manager, plugin_id)

    if not plugin and not file_path:
        raise HTTPException(status_code=404, detail=f"Plugin '{plugin_id}' not found")

    if file_path and file_path.exists():
        try:
            file_path.unlink()
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to delete plugin file {file_path.name}: {e}"
            ) from e

    plugin_manager.plugins.pop(plugin_id, None)
    plugin_manager.configs.pop(plugin_id, None)

    # Clean up persisted configs
    saved = read_persisted_configs(plugin_manager.data_dir)
    if plugin_id in saved:
        saved.pop(plugin_id, None)
        write_persisted_configs(plugin_manager.data_dir, saved)

    return {
        "status": "ok",
        "message": f"Plugin '{plugin_id}' deleted successfully",
    }


@router.post("/plugins/{plugin_id}/reload")
async def reload_plugin_endpoint(plugin_id: str):
    """Explicitly trigger hot-reload of a plugin."""
    _ensure_plugins_loaded()
    file_path = find_plugin_file_path(plugin_manager, plugin_id)
    if not file_path or not file_path.exists():
        raise HTTPException(
            status_code=404, detail=f"Plugin file for '{plugin_id}' not found on disk"
        )

    plugin = None
    try:
        plugin = plugin_manager.reload_plugin(plugin_id)
    except Exception:
        pass

    if plugin is None:
        try:
            plugin = load_plugin_from_file(file_path, data_dir=plugin_manager.data_dir)
            plugin_manager.plugins[plugin.metadata.id] = plugin
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Failed to hot-reload plugin '{plugin_id}': {e}"
            ) from e

    meta = plugin.metadata.model_copy(update={"filename": file_path.name})
    _persist_plugin_metadata(plugin.metadata.id, meta)

    return {
        "status": "ok",
        "message": f"Plugin '{plugin_id}' hot-reloaded successfully",
        "plugin": meta,
    }


@router.put("/plugins/{plugin_id}/config")
async def update_plugin_config(plugin_id: str, body: PluginConfigUpdate):
    """Update custom variables and/or enabled status of a plugin."""
    _ensure_plugins_loaded()
    plugin = plugin_manager.plugins.get(plugin_id)
    if not plugin:
        raise HTTPException(status_code=404, detail=f"Plugin '{plugin_id}' not found")

    try:
        plugin_manager.update_plugin_config(
            plugin_id, enabled=body.enabled, variables=body.variables
        )
    except Exception:
        pass

    meta = plugin.metadata
    if body.enabled is not None:
        meta.enabled = body.enabled

    if body.pinned is not None:
        meta.pinned = body.pinned
        try:
            plugin_manager.update_plugin_config(plugin_id, pinned=body.pinned)
        except Exception:
            pass

    if body.variables is not None:
        for var in meta.variables:
            if var.name in body.variables:
                var.value = body.variables[var.name]

    _persist_plugin_metadata(plugin_id, meta)

    return {
        "status": "ok",
        "message": f"Configuration for plugin '{plugin_id}' updated",
        "plugin": meta,
    }


@router.post("/plugins/{plugin_id}/pin")
async def toggle_plugin_pin(plugin_id: str):
    """Toggle or set pinned status for a provider plugin."""
    _ensure_plugins_loaded()
    plugin = plugin_manager.plugins.get(plugin_id)
    if not plugin:
        raise HTTPException(status_code=404, detail=f"Plugin '{plugin_id}' not found")
    cfg = plugin_manager.configs.setdefault(plugin_id, {})
    current_pinned = cfg.get("pinned", getattr(plugin.metadata, "pinned", False))
    new_pinned = not current_pinned
    plugin_manager.update_plugin_config(plugin_id, pinned=new_pinned)
    plugin.metadata.pinned = new_pinned
    return {"status": "ok", "pinned": new_pinned, "plugin_id": plugin_id}


# =========================================================================
# Local API Keys Management & Usage Tracking
# =========================================================================

@router.get("/keys", response_model=List[APIKey])
async def list_api_keys(include_temporary: bool = False):
    """List all local API keys and their usage metrics.

    Secrets are masked. Use ``GET /api/keys/{key_id}/reveal`` when the literal
    key is needed for a copy action.
    """
    keys = plugin_manager.get_api_keys()
    if not include_temporary:
        keys = [k for k in keys if not k.is_temporary]
    return [redact_api_key(key) for key in keys]


@router.get("/keys/temporary", response_model=APIKey)
async def get_temporary_api_key():
    """Get or create an active temporary test API key (expires in 30 min or 10 requests)."""
    return plugin_manager.get_or_create_temporary_key()


@router.post("/keys", response_model=APIKey)
async def create_api_key(body: APIKeyCreate):
    """Create a new local API key with usage tracking."""
    return plugin_manager.add_api_key(body)


@router.get("/keys/{key_id}", response_model=APIKey)
async def get_api_key(key_id: str):
    """Get details and usage for a specific API key (secret masked)."""
    key = plugin_manager.get_api_key(key_id)
    if not key:
        raise HTTPException(status_code=404, detail=f"API key '{key_id}' not found")
    return redact_api_key(key)


@router.get("/keys/{key_id}/reveal", response_model=APIKey)
async def reveal_api_key(key_id: str):
    """Return the literal secret for a key. Requires admin authentication."""
    key = plugin_manager.get_api_key(key_id)
    if not key:
        raise HTTPException(status_code=404, detail=f"API key '{key_id}' not found")
    return key


@router.put("/keys/{key_id}", response_model=APIKey)
async def update_api_key(key_id: str, body: APIKeyUpdate):
    """Update name or enabled status of an API key."""
    updated = plugin_manager.update_api_key(key_id, body)
    if not updated:
        raise HTTPException(status_code=404, detail=f"API key '{key_id}' not found")
    return updated


@router.delete("/keys/{key_id}")
async def delete_api_key(key_id: str):
    """Delete a local API key."""
    success = plugin_manager.delete_api_key(key_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"API key '{key_id}' not found")
    return {"status": "ok", "message": f"API key '{key_id}' deleted successfully"}


# =========================================================================
# Custom Load Balancing Rules Management
# =========================================================================

@router.get("/load-balancing-rules", response_model=List[LoadBalancingRule])
async def list_load_balancing_rules():
    """List all custom load balancing rules."""
    return plugin_manager.get_load_balancing_rules()


@router.post("/load-balancing-rules", response_model=LoadBalancingRule)
async def create_load_balancing_rule(body: LoadBalancingRuleCreate):
    """Create a new custom load balancing rule model exposed to /v1."""
    if not body.model_id or not body.model_id.strip():
        raise HTTPException(status_code=400, detail="Custom model ID is required")
    if not body.targets:
        raise HTTPException(status_code=400, detail="At least one target model must be selected")
    return plugin_manager.add_load_balancing_rule(body)


@router.get("/load-balancing-rules/{rule_id}", response_model=LoadBalancingRule)
async def get_load_balancing_rule(rule_id: str):
    """Get a custom load balancing rule by ID."""
    rule = plugin_manager.get_load_balancing_rule(rule_id)
    if not rule:
        raise HTTPException(status_code=404, detail=f"Load balancing rule '{rule_id}' not found")
    return rule


@router.put("/load-balancing-rules/{rule_id}", response_model=LoadBalancingRule)
async def update_load_balancing_rule(rule_id: str, body: LoadBalancingRuleUpdate):
    """Update a custom load balancing rule."""
    updated = plugin_manager.update_load_balancing_rule(rule_id, body)
    if not updated:
        raise HTTPException(status_code=404, detail=f"Load balancing rule '{rule_id}' not found")
    return updated


@router.delete("/load-balancing-rules/{rule_id}")
async def delete_load_balancing_rule(rule_id: str):
    """Delete a custom load balancing rule."""
    success = plugin_manager.delete_load_balancing_rule(rule_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Load balancing rule '{rule_id}' not found")
    return {"status": "ok", "message": f"Load balancing rule '{rule_id}' deleted successfully"}


# =========================================================================
# Models Management
# =========================================================================

@router.post("/plugins/{plugin_id}/models")
async def add_custom_model(plugin_id: str, model_config: PluginModelConfig):
    """Add a new custom model to a plugin."""
    _ensure_plugins_loaded()
    plugin = plugin_manager.plugins.get(plugin_id)
    if not plugin:
        raise HTTPException(status_code=404, detail=f"Plugin '{plugin_id}' not found")

    current_models = plugin_manager.get_plugin_models(plugin_id)
    if not current_models:
        current_models = plugin.metadata.models

    existing = [m for m in current_models if m.id == model_config.id]
    if existing:
        raise HTTPException(
            status_code=400,
            detail=f"Model with ID '{model_config.id}' already exists in plugin '{plugin_id}'",
        )

    try:
        plugin_manager.add_custom_model(plugin_id, model_config)
    except Exception:
        pass

    if model_config.id not in [m.id for m in plugin.metadata.models]:
        plugin.metadata.models.append(model_config)

    _persist_plugin_metadata(plugin_id, plugin.metadata)

    return {
        "status": "ok",
        "message": f"Model '{model_config.id}' added to plugin '{plugin_id}'",
        "model": model_config,
        "plugin": plugin.metadata,
    }


@router.put("/plugins/{plugin_id}/models/{model_id}")
async def update_custom_model(plugin_id: str, model_id: str, body: PluginModelUpdate):
    """Edit model properties (name, upstream_model, enabled, parameters, context window, max tokens)."""
    _ensure_plugins_loaded()
    plugin = plugin_manager.plugins.get(plugin_id)
    if not plugin:
        raise HTTPException(status_code=404, detail=f"Plugin '{plugin_id}' not found")

    current_models = plugin_manager.get_plugin_models(plugin_id)
    if not current_models:
        current_models = plugin.metadata.models

    target_model: Optional[PluginModelConfig] = None
    for m in current_models:
        if m.id == model_id:
            target_model = m
            break

    if not target_model:
        raise HTTPException(
            status_code=404, detail=f"Model '{model_id}' not found in plugin '{plugin_id}'"
        )

    if body.name is not None:
        target_model.name = body.name
    if body.upstream_model is not None:
        target_model.upstream_model = body.upstream_model
    if body.enabled is not None:
        target_model.enabled = body.enabled
    if body.context_window is not None:
        target_model.context_window = body.context_window
    if body.max_tokens is not None:
        target_model.max_tokens = body.max_tokens
    if body.parameters is not None:
        target_model.parameters = body.parameters

    try:
        plugin_manager.update_custom_model(plugin_id, model_id, target_model)
    except Exception:
        pass

    for m in plugin.metadata.models:
        if m.id == model_id:
            if body.name is not None:
                m.name = body.name
            if body.upstream_model is not None:
                m.upstream_model = body.upstream_model
            if body.enabled is not None:
                m.enabled = body.enabled
            if body.context_window is not None:
                m.context_window = body.context_window
            if body.max_tokens is not None:
                m.max_tokens = body.max_tokens
            if body.parameters is not None:
                m.parameters = body.parameters

    _persist_plugin_metadata(plugin_id, plugin.metadata)

    return {
        "status": "ok",
        "message": f"Model '{model_id}' updated",
        "model": target_model,
        "plugin": plugin.metadata,
    }


@router.delete("/plugins/{plugin_id}/models/{model_id}")
async def delete_custom_model(plugin_id: str, model_id: str):
    """Remove/delete a model from a plugin."""
    _ensure_plugins_loaded()
    plugin = plugin_manager.plugins.get(plugin_id)
    if not plugin:
        raise HTTPException(status_code=404, detail=f"Plugin '{plugin_id}' not found")

    current_models = plugin_manager.get_plugin_models(plugin_id)
    if not current_models:
        current_models = plugin.metadata.models

    target = next((m for m in current_models if m.id == model_id), None)
    if target is None:
        raise HTTPException(
            status_code=404, detail=f"Model '{model_id}' not found in plugin '{plugin_id}'"
        )

    try:
        plugin_manager.delete_custom_model(plugin_id, model_id)
    except Exception:
        pass

    idx = next((i for i, m in enumerate(plugin.metadata.models) if m.id == model_id), None)
    if idx is not None:
        plugin.metadata.models.pop(idx)

    _persist_plugin_metadata(plugin_id, plugin.metadata)

    return {
        "status": "ok",
        "message": f"Model '{model_id}' removed from plugin '{plugin_id}'",
    }


# =========================================================================
# Accounts Management & Multi-Account Load Balancing
# =========================================================================

@router.get("/accounts", response_model=List[Account])
async def list_accounts(provider_id: Optional[str] = Query(default=None)):
    """List connected accounts, optionally filtered by provider_id.

    Credential fields are masked. Use the account reveal endpoint when an
    edit flow genuinely needs the literal secret.
    """
    accounts = plugin_manager.get_accounts(provider_id=provider_id)
    return [redact_account(account) for account in accounts]


@router.post("/accounts", response_model=Account)
async def create_account(account_in: AccountCreate):
    """Connect a new account to a provider."""
    _ensure_plugins_loaded()
    if account_in.provider_id not in plugin_manager.plugins:
        # Check if provider exists on disk or in configs
        pass
    account = plugin_manager.add_account(account_in)
    return account


@router.get("/accounts/{account_id}", response_model=Account)
async def get_account(account_id: str):
    """Get detail for a specific account (credentials masked)."""
    account = plugin_manager.get_account(account_id)
    if not account:
        raise HTTPException(status_code=404, detail=f"Account '{account_id}' not found")
    return redact_account(account)


@router.get("/accounts/{account_id}/reveal", response_model=Account)
async def reveal_account(account_id: str):
    """Return an account with literal credentials. Requires admin authentication."""
    account = plugin_manager.get_account(account_id)
    if not account:
        raise HTTPException(status_code=404, detail=f"Account '{account_id}' not found")
    return account


@router.put("/accounts/{account_id}", response_model=Account)
async def update_account(account_id: str, updates: AccountUpdate):
    """Update properties of a connected account."""
    account = plugin_manager.update_account(account_id, updates)
    if not account:
        raise HTTPException(status_code=404, detail=f"Account '{account_id}' not found")
    return account


@router.delete("/accounts/{account_id}")
async def delete_account(account_id: str):
    """Delete a connected account."""
    success = plugin_manager.delete_account(account_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Account '{account_id}' not found")
    return {"status": "ok", "message": f"Account '{account_id}' deleted successfully"}


@router.post("/accounts/{account_id}/reset-cooldown")
async def reset_account_cooldown(account_id: str):
    """Reset rate-limit / cooldown status for an account."""
    success = plugin_manager.reset_account_cooldown(account_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Account '{account_id}' not found")
    account = plugin_manager.get_account(account_id)
    return {
        "status": "ok",
        "message": f"Cooldown reset for account '{account_id}'",
        "account": account,
    }


@router.post("/accounts/{account_id}/pin")
async def toggle_account_pin(account_id: str):
    """Toggle or set pinned status for an account."""
    account = plugin_manager.get_account(account_id)
    if not account:
        raise HTTPException(status_code=404, detail=f"Account '{account_id}' not found")
    new_pinned = not getattr(account, "pinned", False)
    account.pinned = new_pinned
    plugin_manager._save_accounts()
    return {"status": "ok", "pinned": new_pinned, "account_id": account_id}


# =========================================================================
# External Proxy Management Endpoints
# =========================================================================

@router.get("/proxies", response_model=List[Proxy])
async def list_proxies(tag: Optional[str] = Query(default=None)):
    """List all external proxies in the proxy pool, optionally filtered by tag."""
    return plugin_manager.get_proxies(tag=tag)


@router.post("/proxies", response_model=Proxy)
async def create_proxy(proxy_in: ProxyCreate):
    """Add a new external proxy to the pool."""
    return plugin_manager.add_proxy(proxy_in)


@router.get("/proxies/{proxy_id}", response_model=Proxy)
async def get_proxy_detail(proxy_id: str):
    """Get details for a specific proxy."""
    p = plugin_manager.get_proxy(proxy_id)
    if not p:
        raise HTTPException(status_code=404, detail=f"Proxy '{proxy_id}' not found")
    return p


@router.put("/proxies/{proxy_id}", response_model=Proxy)
async def update_proxy_endpoint(proxy_id: str, updates: ProxyUpdate):
    """Update properties of an existing proxy."""
    p = plugin_manager.update_proxy(proxy_id, updates)
    if not p:
        raise HTTPException(status_code=404, detail=f"Proxy '{proxy_id}' not found")
    return p


@router.delete("/proxies/{proxy_id}")
async def delete_proxy_endpoint(proxy_id: str):
    """Delete a proxy from the pool."""
    success = plugin_manager.delete_proxy(proxy_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Proxy '{proxy_id}' not found")
    return {"status": "ok", "message": f"Proxy '{proxy_id}' deleted successfully"}


@router.post("/proxies/{proxy_id}/reset-cooldown")
async def reset_proxy_cooldown_endpoint(proxy_id: str):
    """Reset rate limit status / cooldown on a proxy."""
    success = plugin_manager.reset_proxy_cooldown(proxy_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Proxy '{proxy_id}' not found")
    p = plugin_manager.get_proxy(proxy_id)
    return {"status": "ok", "message": f"Cooldown reset for proxy '{proxy_id}'", "proxy": p}


class ProxyTestRequest(BaseModel):
    url: str
    target_url: Optional[str] = "https://httpbin.org/ip"


@router.post("/proxies/test")
async def test_proxy_connectivity(req: ProxyTestRequest):
    """Test connectivity through an external proxy URL."""
    start_t = time.perf_counter()
    try:
        async with httpx.AsyncClient(proxy=req.url, timeout=10.0) as client:
            resp = await client.get(req.target_url or "https://httpbin.org/ip")
            latency_ms = round((time.perf_counter() - start_t) * 1000, 2)
            return {
                "status": "ok",
                "status_code": resp.status_code,
                "latency_ms": latency_ms,
                "response": resp.text[:200],
            }
    except Exception as exc:
        latency_ms = round((time.perf_counter() - start_t) * 1000, 2)
        return {
            "status": "error",
            "error": str(exc),
            "latency_ms": latency_ms,
        }


class AntigravityCallbackRequest(BaseModel):
    callback_url: Optional[str] = None
    code: Optional[str] = None
    redirect_uri: Optional[str] = None
    flow_id: Optional[str] = None


@router.post("/antigravity/oauth/start")
async def start_antigravity_oauth():
    """Start Google OAuth consent flow with loopback redirect listener."""
    flow = antigravity_oauth.start_flow()
    return flow


@router.get("/antigravity/oauth/status/{flow_id}")
async def check_antigravity_oauth_status(flow_id: str):
    """Check loopback callback status for a pending Google OAuth flow."""
    res = await antigravity_oauth.check_flow_status(flow_id)
    if res.get("status") == "complete" and res.get("account"):
        # Ensure account is persisted in manager
        saved = plugin_manager.add_account(res["account"])
        res["account"] = saved
    return res


@router.post("/antigravity/oauth/callback")
async def complete_antigravity_oauth_callback(payload: AntigravityCallbackRequest):
    """
    Complete Google OAuth via manual callback URL or authorization code paste
    (useful for remote/docker/headless instances where loopback isn't reachable).
    """
    code = payload.code
    redirect_uri = payload.redirect_uri or "http://127.0.0.1:45123/"

    if payload.callback_url:
        import urllib.parse
        parsed = urllib.parse.urlparse(payload.callback_url)
        params = urllib.parse.parse_qs(parsed.query)
        code = params.get("code", [None])[0]
        # Reconstruct redirect_uri from the pasted URL
        redirect_uri = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

    if not code:
        raise HTTPException(
            status_code=400,
            detail="Could not extract Google authorization code from input",
        )

    try:
        account = await antigravity_oauth.exchange_code_and_create_account(
            code=code, redirect_uri=redirect_uri
        )
        saved = plugin_manager.add_account(account)
        return {"status": "complete", "account": saved}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Google OAuth completion failed: {e}")


@router.post("/antigravity/oauth/import-local")
async def import_local_antigravity_session():
    """Auto-detect and import local Antigravity acp_token.json credentials from the environment."""
    account = await antigravity_oauth.import_local_session()
    if not account:
        raise HTTPException(
            status_code=404,
            detail="No local Antigravity session (acp_token.json) found on this machine",
        )
    saved = plugin_manager.add_account(account)
    return {
        "status": "ok",
        "message": f"Successfully imported Google account '{saved.name}'",
        "account": saved,
    }


async def fetch_antigravity_quota(account: Account) -> Dict[str, Any]:
    """Fetch live quota and usage summary for an Antigravity account using configured endpoints and token refresh."""
    import httpx
    antigravity_oauth.sanitize_ssl_env()

    # Pre-emptively refresh token if expiring within 2 minutes or missing
    if account.extra_config.get("refresh_token"):
        expires_at = account.extra_config.get("expires_at", 0)
        if expires_at - time.time() < 120:
            refreshed = await antigravity_oauth.refresh_account_token(account)
            if refreshed:
                plugin_manager._save_accounts()

    # Determine endpoints in order of priority: account.base_url first, then fallbacks
    endpoints: List[str] = []
    if account.base_url:
        endpoints.append(account.base_url.rstrip("/"))
    for ep in CLOUDCODE_PA_ENDPOINTS:
        if ep not in endpoints:
            endpoints.append(ep)

    project_id = account.extra_config.get("project_id", "aicode-consumers")
    headers = {
        "Authorization": f"Bearer {account.api_key}",
        "User-Agent": "antigravity/hub/2.24.0 (aidev_client; os_type=linux; arch=x64; cl=963137146)",
        "x-goog-api-client": "ide/antigravity",
        "Content-Type": "application/json",
    }

    last_error = "No endpoints attempted"
    async with httpx.AsyncClient(timeout=15.0) as client:
        for ep in endpoints:
            url = f"{ep}/v1internal:retrieveUserQuotaSummary"
            bodies = [{"project": project_id}] if project_id else [{}]
            if {} not in bodies:
                bodies.append({})

            for body in bodies:
                try:
                    resp = await client.post(url, headers=headers, json=body)
                    # If 401 Unauthorized, refresh token once and retry
                    if resp.status_code == 401 and account.extra_config.get("refresh_token"):
                        refreshed = await antigravity_oauth.refresh_account_token(account)
                        if refreshed:
                            plugin_manager._save_accounts()
                            headers["Authorization"] = f"Bearer {account.api_key}"
                            resp = await client.post(url, headers=headers, json=body)

                    if resp.status_code == 200:
                        data = resp.json()
                        data["account_id"] = account.id
                        data["account_name"] = account.name
                        data["tier"] = account.extra_config.get("tier", "free-tier")
                        data["email"] = account.extra_config.get("email", "")
                        return data
                    last_error = f"Upstream error ({resp.status_code}) from {ep}: {resp.text}"
                except Exception as e:
                    last_error = f"Connection error to {ep}: {e}"

    return {"error": last_error}


@router.get("/antigravity/quota/{account_id}")
async def get_antigravity_account_quota(account_id: str):
    """Fetch live 5-hour and weekly Google Antigravity quota usage summary."""
    account = plugin_manager.get_account(account_id)
    if not account:
        raise HTTPException(status_code=404, detail=f"Account '{account_id}' not found")
    if account.provider_id != "antigravity":
        raise HTTPException(status_code=400, detail="Account is not an Antigravity account")

    return await fetch_antigravity_quota(account)


@router.get("/antigravity/status/{account_id}")
async def get_antigravity_account_status(account_id: str):
    """Alias for Google Antigravity account status & quota usage summary."""
    return await get_antigravity_account_quota(account_id)


@router.get("/antigravity/status")
async def get_antigravity_all_status():
    """Fetch live status and quota usage for all connected Google Antigravity accounts."""
    accounts = [a for a in plugin_manager.get_accounts() if a.provider_id == "antigravity"]
    if not accounts:
        return {"accounts": [], "message": "No Google Antigravity accounts configured"}

    async def build(acc: Account) -> Dict[str, Any]:
        quota_data = await fetch_antigravity_quota(acc)
        return {
            "account_id": acc.id,
            "account_name": acc.name,
            "enabled": acc.enabled,
            "email": acc.extra_config.get("email", ""),
            "tier": acc.extra_config.get("tier", "free-tier"),
            "rate_limited": not acc.is_available(),
            "quota": quota_data,
        }

    results = await asyncio.gather(*(build(acc) for acc in accounts), return_exceptions=True)
    items: List[Dict[str, Any]] = []
    for index, item in enumerate(results):
        if isinstance(item, Exception):
            acc = accounts[index]
            items.append({
                "account_id": acc.id,
                "account_name": acc.name,
                "enabled": acc.enabled,
                "email": acc.extra_config.get("email", ""),
                "tier": acc.extra_config.get("tier", "free-tier"),
                "rate_limited": not acc.is_available(),
                "quota": {"error": str(item)},
            })
        else:
            items.append(item)
    return {"accounts": items}


# =========================================================================
# Codex OAuth & Account Management
# =========================================================================

class CodexCallbackRequest(BaseModel):
    callback_url: Optional[str] = None
    code: Optional[str] = None
    redirect_uri: Optional[str] = None
    flow_id: Optional[str] = None


@router.post("/codex/oauth/start")
async def start_codex_oauth():
    """Start OpenAI Codex OAuth consent flow with PKCE and loopback redirect listener."""
    flow = codex_oauth.start_flow()
    return flow


@router.get("/codex/oauth/status/{flow_id}")
async def check_codex_oauth_status(flow_id: str):
    """Check loopback callback status for a pending Codex OAuth flow."""
    res = await codex_oauth.check_flow_status(flow_id)
    if res.get("status") == "complete" and res.get("account"):
        saved = plugin_manager.add_account(res["account"])
        res["account"] = saved
    return res


@router.post("/codex/oauth/callback")
async def complete_codex_oauth_callback(payload: CodexCallbackRequest):
    """
    Complete Codex OAuth via manual callback URL or authorization code paste
    (useful for remote/docker/headless instances where loopback isn't reachable).
    """
    code = payload.code
    redirect_uri = payload.redirect_uri or "http://127.0.0.1:45124/auth/callback"
    code_verifier = None

    if payload.flow_id and payload.flow_id in codex_oauth.active_flows:
        flow = codex_oauth.active_flows[payload.flow_id]
        code_verifier = flow.get("code_verifier")
        redirect_uri = flow.get("redirect_uri", redirect_uri)

    if payload.callback_url:
        import urllib.parse
        parsed = urllib.parse.urlparse(payload.callback_url)
        params = urllib.parse.parse_qs(parsed.query)
        code = params.get("code", [None])[0]
        redirect_uri = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        state = params.get("state", [None])[0]
        if state:
            for fid, f in codex_oauth.active_flows.items():
                if f.get("state") == state:
                    code_verifier = f.get("code_verifier")
                    break

    if not code:
        raise HTTPException(
            status_code=400,
            detail="Could not extract Codex authorization code from input",
        )

    try:
        account = await codex_oauth.exchange_code_and_create_account(
            code=code, redirect_uri=redirect_uri, code_verifier=code_verifier
        )
        saved = plugin_manager.add_account(account)
        return {"status": "complete", "account": saved}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Codex OAuth completion failed: {e}")


@router.post("/codex/oauth/import-local")
async def import_local_codex_session():
    """Auto-detect and import local Codex session (~/.codex/auth.json) credentials from the environment."""
    account = await codex_oauth.import_local_session()
    if not account:
        raise HTTPException(
            status_code=404,
            detail="No local Codex session (~/.codex/auth.json) found on this machine",
        )
    saved = plugin_manager.add_account(account)
    return {
        "status": "ok",
        "message": f"Successfully imported Codex account '{saved.name}'",
        "account": saved,
    }


async def fetch_codex_quota(account: Account) -> Dict[str, Any]:
    """Fetch live usage and rate limits for a Codex account from /backend-api/wham/usage."""
    import httpx
    from datetime import datetime, timezone

    if account.extra_config.get("refresh_token"):
        expires_at = account.extra_config.get("expires_at", 0)
        if expires_at - time.time() < 120:
            refreshed = await codex_oauth.refresh_account_token(account)
            if refreshed:
                plugin_manager._save_accounts()

    base_url = (account.base_url or "https://chatgpt.com/backend-api").rstrip("/")
    if base_url.endswith("/responses") or base_url.endswith("/codex/responses"):
        base_url = "https://chatgpt.com/backend-api"
    url = f"{base_url}/wham/usage"

    headers = {
        "Authorization": f"Bearer {account.api_key}",
        "User-Agent": "codex/cli/0.167.0 (linux; x86_64)",
        "Accept": "application/json",
    }
    chatgpt_acc_id = account.extra_config.get("account_id")
    if chatgpt_acc_id:
        headers["ChatGPT-Account-Id"] = chatgpt_acc_id

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code == 401 and account.extra_config.get("refresh_token"):
                refreshed = await codex_oauth.refresh_account_token(account)
                if refreshed:
                    plugin_manager._save_accounts()
                    headers["Authorization"] = f"Bearer {account.api_key}"
                    resp = await client.get(url, headers=headers)

            if resp.status_code == 200:
                data = resp.json()
                rate_limit = data.get("rate_limit") or {}
                buckets = []

                def _build_bucket(window_data: Optional[Dict[str, Any]], default_name: str, bucket_id: str):
                    if not window_data:
                        return None
                    used_pct = window_data.get("used_percent", 0)
                    if used_pct is None:
                        used_pct = 0
                    window_sec = window_data.get("limit_window_seconds")
                    name = default_name
                    if window_sec:
                        if abs(window_sec - 18000) < 1800:
                            name = "5-Hour Limit"
                        elif abs(window_sec - 604800) < 36000:
                            name = "Weekly Limit"
                        elif abs(window_sec - 2592000) < 86400:
                            name = "Monthly Limit"

                    reset_at = window_data.get("reset_at")
                    reset_after = window_data.get("reset_after_seconds")
                    reset_time_str = None
                    if reset_at:
                        reset_time_str = datetime.fromtimestamp(reset_at, tz=timezone.utc).isoformat()
                    elif reset_after is not None:
                        reset_time_str = datetime.fromtimestamp(time.time() + reset_after, tz=timezone.utc).isoformat()

                    return {
                        "bucketId": bucket_id,
                        "displayName": name,
                        "usedPercent": used_pct,
                        "remainingFraction": round(max(0.0, (100 - used_pct) / 100.0), 4),
                        "resetTime": reset_time_str,
                    }

                pw = _build_bucket(rate_limit.get("primary_window"), "Primary Limit", "primary_window")
                if pw:
                    buckets.append(pw)

                sw = _build_bucket(rate_limit.get("secondary_window"), "Secondary Limit", "secondary_window")
                if sw:
                    buckets.append(sw)

                plan_type = data.get("plan_type") or account.extra_config.get("plan_type", "standard")
                email = data.get("email") or account.extra_config.get("email", "")

                return {
                    "account_id": account.id,
                    "account_name": account.name,
                    "email": email,
                    "plan_type": plan_type,
                    "groups": [
                        {
                            "displayName": "Codex Usage",
                            "buckets": buckets,
                        }
                    ] if buckets else [],
                    "raw": data,
                }
            else:
                return {"error": f"Upstream error ({resp.status_code}): {resp.text[:200]}"}
    except Exception as e:
        return {"error": f"Failed to fetch Codex usage: {e}"}


async def fetch_codex_status(account: Account) -> Dict[str, Any]:
    """Fetch status, plan, token validity, and usage for a Codex account."""
    if account.extra_config.get("refresh_token"):
        expires_at = account.extra_config.get("expires_at", 0)
        if expires_at - time.time() < 120:
            refreshed = await codex_oauth.refresh_account_token(account)
            if refreshed:
                plugin_manager._save_accounts()

    quota_data = await fetch_codex_quota(account)
    return {
        "account_id": account.id,
        "account_name": account.name,
        "email": account.extra_config.get("email", ""),
        "plan_type": account.extra_config.get("plan_type", "standard"),
        "chatgpt_account_id": account.extra_config.get("account_id", ""),
        "expires_at": account.extra_config.get("expires_at"),
        "authenticated": bool(account.api_key),
        "quota": quota_data,
    }


@router.get("/codex/status/{account_id}")
async def get_codex_account_status(account_id: str):
    """Fetch live status for a specific Codex account."""
    account = plugin_manager.get_account(account_id)
    if not account:
        raise HTTPException(status_code=404, detail=f"Account '{account_id}' not found")
    if account.provider_id != "codex":
        raise HTTPException(status_code=400, detail="Account is not a Codex account")
    return await fetch_codex_status(account)


@router.get("/codex/quota/{account_id}")
async def get_codex_account_quota(account_id: str):
    """Fetch live usage and rate limits for a specific Codex account."""
    account = plugin_manager.get_account(account_id)
    if not account:
        raise HTTPException(status_code=404, detail=f"Account '{account_id}' not found")
    if account.provider_id != "codex":
        raise HTTPException(status_code=400, detail="Account is not a Codex account")
    return await fetch_codex_quota(account)


@router.get("/codex/status")
async def get_codex_all_status():
    """Fetch live status for all connected Codex accounts."""
    accounts = [a for a in plugin_manager.get_accounts() if a.provider_id == "codex"]
    if not accounts:
        return {"accounts": [], "message": "No Codex accounts configured"}

    async def build(acc: Account) -> Dict[str, Any]:
        status_data = await fetch_codex_status(acc)
        return {
            "account_id": acc.id,
            "account_name": acc.name,
            "enabled": acc.enabled,
            "email": acc.extra_config.get("email", ""),
            "plan_type": acc.extra_config.get("plan_type", "standard"),
            "rate_limited": not acc.is_available(),
            "quota": status_data.get("quota"),
            "status": status_data,
        }

    results = await asyncio.gather(*(build(acc) for acc in accounts), return_exceptions=True)
    items: List[Dict[str, Any]] = []
    for index, item in enumerate(results):
        if isinstance(item, Exception):
            acc = accounts[index]
            items.append({
                "account_id": acc.id,
                "account_name": acc.name,
                "enabled": acc.enabled,
                "email": acc.extra_config.get("email", ""),
                "plan_type": acc.extra_config.get("plan_type", "standard"),
                "rate_limited": not acc.is_available(),
                "quota": {"error": str(item)},
                "status": {"error": str(item)},
            })
        else:
            items.append(item)
    return {"accounts": items}


# =========================================================================
# Telemetry, Stats, and Logs
# =========================================================================

@router.get("/logs", response_model=List[RequestLogEntry])
async def get_logs(limit: int = Query(default=100, ge=1, le=1000)):
    """Retrieve recent request logs."""
    return plugin_manager.get_logs(limit=limit)


@router.get("/stats")
async def get_stats():
    """Return summary metrics for the router and dashboard."""
    _ensure_plugins_loaded()
    logs = plugin_manager.get_logs(limit=1000)
    sample_requests = len(logs)
    total_requests = plugin_manager.get_total_requests()
    success_requests = sum(1 for log in logs if log.status == "success")
    success_rate = (
        round((success_requests / sample_requests) * 100.0, 1) if sample_requests > 0 else 100.0
    )

    active_plugins = [p for p in plugin_manager.plugins.values() if p.metadata.enabled]
    active_plugins_count = len(active_plugins)

    active_rules_count = sum(1 for r in plugin_manager.get_load_balancing_rules() if r.enabled)
    active_models_count = sum(
        sum(1 for m in p.metadata.models if m.enabled) for p in active_plugins
    ) + active_rules_count

    avg_latency = (
        round(sum(log.latency_ms for log in logs) / sample_requests, 1)
        if sample_requests > 0
        else 0.0
    )

    all_accounts = plugin_manager.get_accounts()
    total_accounts = len(all_accounts)
    active_accounts = len([a for a in all_accounts if a.is_available()])

    return {
        "total_requests": total_requests,
        "success_rate": success_rate,
        "active_models_count": active_models_count,
        "active_plugins_count": active_plugins_count,
        "total_accounts": total_accounts,
        "active_accounts": active_accounts,
        "avg_latency_ms": avg_latency,
    }


@router.get("/management")
async def get_management_info(request: Request):
    """Return the CLIProxyAPI-compatible hub URL and management key for T3 Code/OpenCode."""
    base_url = str(request.base_url).rstrip("/")
    return {
        "base_url": base_url,
        "management_url": f"{base_url}/v0/management",
        "management_key": plugin_manager.get_management_key(),
    }


# =========================================================================
# Chat Playground Test Endpoint
# =========================================================================

@router.post("/test-chat")
async def test_chat(req: TestChatRequest):
    """Test endpoint that routes a test prompt through the router to a specific model."""
    _ensure_plugins_loaded()

    # Handle example test-model (not exposed in /models or provider settings)
    if req.model == "test-model":
        messages: List[ChatMessage] = []
        if req.messages:
            messages = req.messages
        elif req.prompt:
            messages = [ChatMessage(role="user", content=req.prompt)]
        else:
            raise HTTPException(status_code=400, detail="Either 'prompt' or 'messages' must be provided")

        latency_ms = 1.00
        prompt_tokens = sum(len(str(m.content).split()) for m in messages)
        completion_tokens = 22
        now_str = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        reply_content = (
            f"Simulated response from LLMProxy (test-model).\n\n---\n"
            f"**Execution Details:**\n"
            f"• **Time taken:** {latency_ms:.2f} ms\n"
            f"• **Timestamp:** {now_str}\n"
            f"• **Model:** test-model\n"
            f"• **Tokens:** {prompt_tokens} prompt, {completion_tokens} completion ({prompt_tokens + completion_tokens} total)\n"
            f"• **Account status:** No account configured (Simulation Mode)"
        )

        if req.stream:
            async def event_generator():
                chunk_obj = {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                    "object": "chat.completion.chunk",
                    "model": "test-model",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": reply_content},
                            "finish_reason": "stop",
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk_obj)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(event_generator(), media_type="text/event-stream")

        log_entry = RequestLogEntry(
            id=f"log-{uuid.uuid4().hex[:8]}",
            timestamp=time.time(),
            model="test-model",
            plugin_id=None,
            status="success",
            status_code=200,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        plugin_manager.add_log(log_entry)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "model": "test-model",
            "created": int(time.time()),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": reply_content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "latency_ms": latency_ms,
        }

    # Model resolution
    found_model: Optional[PluginModelConfig] = None
    found_plugin = None

    for p_id, p in plugin_manager.plugins.items():
        p_models = plugin_manager.get_plugin_models(p_id)
        if not p_models:
            p_models = p.metadata.models
        for m in p_models:
            if m.id == req.model:
                found_model = m
                found_plugin = p
                break
        if found_model:
            break

    if not found_plugin or not found_model:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{req.model}' not found in any provider plugin",
        )

    cfg = plugin_manager.configs.get(found_plugin.metadata.id, {})
    p_enabled = cfg.get("enabled", found_plugin.metadata.enabled)
    if not p_enabled or not found_model.enabled:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{req.model}' or its provider is disabled",
        )

    matched_plugin = found_plugin
    matched_model = found_model
    matched_vars = plugin_manager.get_plugin_variables(found_plugin.metadata.id)
    if not matched_vars:
        matched_vars = {
            v.name: (v.value if (v.value != "" and v.value is not None) else v.default)
            for v in matched_plugin.metadata.variables
        }

    # Prepare chat messages
    messages: List[ChatMessage] = []
    if req.messages:
        messages = req.messages
    elif req.prompt:
        messages = [ChatMessage(role="user", content=req.prompt)]
    else:
        raise HTTPException(status_code=400, detail="Either 'prompt' or 'messages' must be provided")

    chat_request = ChatCompletionRequest(
        model=req.model,
        messages=messages,
        temperature=req.temperature or 1.0,
        max_tokens=req.max_tokens,
        stream=bool(req.stream),
    )

    # Select Account for test chat with routing & failover rules
    strategy = str(matched_vars.get("load_balancing_strategy", "round_robin"))
    accounts = plugin_manager.get_accounts(matched_plugin.metadata.id)
    max_retries = max(len(accounts), int(matched_vars.get("max_retries", 2) or 2)) if accounts else 1
    cooldown_sec = float(matched_vars.get("account_switch_cooldown", 60.0) or 60.0)
    switch_wait = float(matched_vars.get("account_switch_wait", 0.0) or 0.0)

    excluded_accounts: Set[str] = set()
    current_account: Optional[Account] = None
    if accounts:
        current_account = plugin_manager.select_account(
            matched_plugin.metadata.id,
            strategy=strategy,
            exclude_account_ids=excluded_accounts,
        )
    if not current_account:
        current_account = Account(
            id=f"default-{matched_plugin.metadata.id}",
            provider_id=matched_plugin.metadata.id,
            name="Default Account",
            api_key=str(matched_vars.get("api_key", "")),
            base_url=matched_vars.get("base_url"),
        )

    start_time = time.perf_counter()
    last_exception = None

    for attempt in range(max_retries):
        try:
            init_proxy_url, current_p = plugin_manager.resolve_request_proxy(
                variables=matched_vars,
                account=current_account,
                provider_id=matched_plugin.metadata.id,
            )
            attempt_vars = dict(matched_vars)
            if init_proxy_url:
                attempt_vars["proxy"] = init_proxy_url

            import inspect
            sig = inspect.signature(matched_plugin.chat_completion)
            if "account" in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                result = await matched_plugin.chat_completion(
                    chat_request, matched_model, attempt_vars, account=current_account
                )
            else:
                result = await matched_plugin.chat_completion(
                    chat_request, matched_model, attempt_vars
                )
            latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

            if current_account and not current_account.id.startswith("default-"):
                plugin_manager.mark_account_result(current_account.id, success=True)
            if current_p:
                plugin_manager.mark_proxy_result(current_p.id, success=True)

            # Handle streaming responses
            if req.stream:
                async def event_generator():
                    if hasattr(result, "__aiter__"):
                        async for chunk in result:
                            chunk_json = (
                                chunk.model_dump_json()
                                if hasattr(chunk, "model_dump_json")
                                else json.dumps(chunk)
                            )
                            yield f"data: {chunk_json}\n\n"
                    else:
                        content = (
                            result.choices[0].message.content
                            if result.choices and result.choices[0].message
                            else ""
                        )
                        chunk_obj = {
                            "id": getattr(result, "id", f"chatcmpl-{uuid.uuid4().hex[:8]}"),
                            "object": "chat.completion.chunk",
                            "model": req.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"role": "assistant", "content": content},
                                    "finish_reason": "stop",
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk_obj)}\n\n"
                    yield "data: [DONE]\n\n"

                prompt_tokens = sum(len(str(m.content).split()) for m in messages)
                log_entry = RequestLogEntry(
                    id=f"log-{uuid.uuid4().hex[:8]}",
                    timestamp=time.time(),
                    model=req.model,
                    plugin_id=matched_plugin.metadata.id,
                    status="success",
                    status_code=200,
                    latency_ms=latency_ms,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=20,
                )
                plugin_manager.add_log(log_entry)

                return StreamingResponse(event_generator(), media_type="text/event-stream")

            # Non-streaming response
            prompt_tokens = result.usage.prompt_tokens if result.usage else sum(len(str(m.content).split()) for m in messages)
            completion_tokens = result.usage.completion_tokens if result.usage else 20

            log_entry = RequestLogEntry(
                id=f"log-{uuid.uuid4().hex[:8]}",
                timestamp=time.time(),
                model=req.model,
                plugin_id=matched_plugin.metadata.id,
                status="success",
                status_code=200,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            plugin_manager.add_log(log_entry)

            resp_dict = result.model_dump() if hasattr(result, "model_dump") else result.dict()
            resp_dict["latency_ms"] = latency_ms
            return resp_dict

        except Exception as e:
            last_exception = e
            msg = str(e).lower()
            is_rate_limit = (
                "rate limit" in msg
                or "429" in msg
                or "quota" in msg
                or "too many requests" in msg
                or "resource_exhausted" in msg
            )
            if current_account and not current_account.id.startswith("default-"):
                plugin_manager.mark_account_result(
                    current_account.id,
                    success=False,
                    is_rate_limit=is_rate_limit,
                    cooldown_seconds=cooldown_sec,
                )
                excluded_accounts.add(current_account.id)

            if attempt < max_retries - 1 and accounts:
                if switch_wait > 0:
                    await asyncio.sleep(switch_wait)
                next_acc = plugin_manager.select_account(
                    matched_plugin.metadata.id,
                    strategy=strategy,
                    exclude_account_ids=excluded_accounts,
                )
                if not next_acc and len(excluded_accounts) < len(accounts):
                    rem = [a for a in accounts if a.id not in excluded_accounts]
                    if rem:
                        next_acc = rem[0]
                if next_acc and next_acc.id != current_account.id:
                    current_account = next_acc
                    continue
            break

    # If all attempts failed
    latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
    plugin_id = getattr(matched_plugin, "metadata", None) and matched_plugin.metadata.id
    log_entry = RequestLogEntry(
        id=f"log-{uuid.uuid4().hex[:8]}",
        timestamp=time.time(),
        model=req.model,
        plugin_id=plugin_id,
        status="error",
        status_code=500,
        latency_ms=latency_ms,
        error_message=str(last_exception),
    )
    plugin_manager.add_log(log_entry)
    raise HTTPException(status_code=500, detail=f"Chat completion error: {last_exception}") from last_exception


# =========================================================================
# Templates Gallery
# =========================================================================

@router.get("/templates")
async def get_templates():
    """Return starter provider plugin templates for quick creation and swapping."""
    return get_starter_templates()


@router.post("/templates/{template_id}/add", response_model=PluginMetadata)
async def add_provider_template(template_id: str):
    """Install and hot-reload a provider template into the runtime plugins directory."""
    templates = get_starter_templates()
    matched = next((t for t in templates if t["id"] == template_id or t["filename"] == f"{template_id}.py" or t["filename"] == template_id), None)
    if not matched:
        raise HTTPException(status_code=404, detail=f"Template '{template_id}' not found")

    target_file = plugin_manager.plugins_dir / matched["filename"]
    target_file.write_text(matched["code"], encoding="utf-8")
    loaded = plugin_manager.load_plugin_file(target_file)
    if not loaded:
        raise HTTPException(status_code=400, detail="Failed to instantiate plugin from template")
    return loaded.metadata
