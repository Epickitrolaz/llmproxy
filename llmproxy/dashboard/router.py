from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from llmproxy.core.antigravity_oauth import (
    ANTIGRAVITY_CLIENT_ID,
    ANTIGRAVITY_SCOPES,
    CLOUDCODE_PA_ENDPOINTS,
    antigravity_oauth,
)
from llmproxy.core.manager import plugin_manager
from llmproxy.dashboard.loader import (
    apply_persisted_config_to_plugin,
    find_plugin_file_path,
    is_safe_path,
    load_plugin_from_file,
    read_persisted_configs,
    validate_python_syntax,
    write_persisted_configs,
)
from llmproxy.dashboard.templates import STARTER_TEMPLATES
from llmproxy.schemas import (
    Account,
    AccountCreate,
    AccountUpdate,
    ChatMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    PluginConfigUpdate,
    PluginMetadata,
    PluginModelConfig,
    PluginSourceUpdate,
    RequestLogEntry,
)

router = APIRouter(prefix="/api", tags=["dashboard"])


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
    """Save plugin configuration to persistent storage."""
    saved_configs = read_persisted_configs(plugin_manager.data_dir)
    saved_configs[plugin_id] = {
        "enabled": meta.enabled,
        "variables": {v.name: v.value for v in meta.variables},
        "models": [m.model_dump() if hasattr(m, "model_dump") else m.dict() for m in meta.models],
    }
    write_persisted_configs(plugin_manager.data_dir, saved_configs)


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
    """List connected accounts, optionally filtered by provider_id."""
    return plugin_manager.get_accounts(provider_id=provider_id)


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
    """Get detail for a specific account."""
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


@router.get("/antigravity/quota/{account_id}")
async def get_antigravity_account_quota(account_id: str):
    """Fetch live 5-hour and weekly Google Antigravity quota usage summary."""
    account = plugin_manager.get_account(account_id)
    if not account:
        raise HTTPException(status_code=404, detail=f"Account '{account_id}' not found")
    if account.provider_id != "antigravity":
        raise HTTPException(status_code=400, detail="Account is not an Antigravity account")

    # Refresh token if needed
    if account.extra_config.get("refresh_token"):
        await antigravity_oauth.refresh_account_token(account)
        plugin_manager._save_accounts()

    access_token = account.api_key
    project_id = account.extra_config.get("project_id", "aicode-consumers")

    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": "antigravity/hub/2.24.0 (aidev_client; os_type=linux; arch=x64; cl=963137146)",
        "x-goog-api-client": "ide/antigravity",
        "Content-Type": "application/json",
    }
    body = {"project": project_id}

    import httpx
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                "https://cloudcode-pa.googleapis.com/v1internal:retrieveUserQuotaSummary",
                headers=headers,
                json=body,
            )
            if resp.status_code == 200:
                return resp.json()
            return {"error": f"Upstream error {resp.status_code}: {resp.text}"}
    except Exception as e:
        return {"error": str(e)}


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
    total_requests = len(logs)
    success_requests = sum(1 for log in logs if log.status == "success")
    success_rate = (
        round((success_requests / total_requests) * 100.0, 1) if total_requests > 0 else 100.0
    )

    active_plugins = [p for p in plugin_manager.plugins.values() if p.metadata.enabled]
    active_plugins_count = len(active_plugins)

    active_models_count = sum(
        sum(1 for m in p.metadata.models if m.enabled) for p in active_plugins
    )

    avg_latency = (
        round(sum(log.latency_ms for log in logs) / total_requests, 1)
        if total_requests > 0
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


# =========================================================================
# Chat Playground Test Endpoint
# =========================================================================

@router.post("/test-chat")
async def test_chat(req: TestChatRequest):
    """Test endpoint that routes a test prompt through the router to a specific model."""
    _ensure_plugins_loaded()

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

    # Select Account for test chat
    strategy = str(matched_vars.get("load_balancing_strategy", "round_robin"))
    accounts = plugin_manager.get_accounts(matched_plugin.metadata.id)
    account: Optional[Account] = None
    if accounts:
        account = plugin_manager.select_account(matched_plugin.metadata.id, strategy=strategy)
    if not account:
        account = Account(
            id=f"default-{matched_plugin.metadata.id}",
            provider_id=matched_plugin.metadata.id,
            name="Default Account",
            api_key=str(matched_vars.get("api_key", "")),
            base_url=matched_vars.get("base_url"),
        )

    start_time = time.perf_counter()

    try:
        import inspect
        sig = inspect.signature(matched_plugin.chat_completion)
        if "account" in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            result = await matched_plugin.chat_completion(
                chat_request, matched_model, matched_vars, account=account
            )
        else:
            result = await matched_plugin.chat_completion(
                chat_request, matched_model, matched_vars
            )
        latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

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
            error_message=str(e),
        )
        plugin_manager.add_log(log_entry)
        raise HTTPException(status_code=500, detail=f"Chat completion error: {e}") from e


# =========================================================================
# Templates Gallery
# =========================================================================

@router.get("/templates")
async def get_templates():
    """Return starter provider plugin templates for quick creation and swapping."""
    return STARTER_TEMPLATES
