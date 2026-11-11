from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, AsyncIterator, List, Optional

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from llmproxy.core.manager import plugin_manager
from llmproxy.schemas import (
    Account,
    ChatChoice,
    ChatChoiceDelta,
    ChatMessage,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionUsage,
    ModelCard,
    ModelListResponse,
    RequestLogEntry,
)

router = APIRouter(prefix="/v1")


def _estimate_tokens(content: Any) -> int:
    """Simple token estimation based on character count / standard ratio."""
    if not content:
        return 0
    if isinstance(content, str):
        return max(1, len(content) // 4)
    if isinstance(content, list):
        total = 0
        for item in content:
            if isinstance(item, dict):
                total += _estimate_tokens(item.get("text", json.dumps(item)))
            else:
                total += _estimate_tokens(str(item))
        return max(1, total)
    if isinstance(content, dict):
        return max(1, len(json.dumps(content)) // 4)
    return 1


def _calc_prompt_tokens(messages: list, tools: Optional[list] = None) -> int:
    total = 0
    for m in messages:
        if m.content:
            total += _estimate_tokens(m.content)
        if getattr(m, "tool_calls", None):
            total += _estimate_tokens(m.tool_calls)
    if tools:
        total += _estimate_tokens(tools)
    return max(1, total)


def _safe_float(val: Any, default: float) -> float:
    try:
        if val is None or val == "":
            return default
        return float(val)
    except (ValueError, TypeError):
        return default


def _safe_int(val: Any, default: int) -> int:
    try:
        if val is None or val == "":
            return default
        return int(val)
    except (ValueError, TypeError):
        return default


def _is_rate_limit_error(exc: Exception) -> bool:
    """Check if exception represents a rate limit / 429 quota error."""
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
        return True
    msg = str(exc).lower()
    return "rate limit" in msg or "429" in msg or "quota" in msg or "too many requests" in msg


async def _invoke_plugin_completion(plugin, request, model_config, variables, account):
    import inspect
    sig = inspect.signature(plugin.chat_completion)
    if "account" in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return await plugin.chat_completion(request, model_config, variables, account=account)
    return await plugin.chat_completion(request, model_config, variables)


def _extract_api_key(req: Request) -> Optional[str]:
    """Extract local API key from Authorization header or custom headers."""
    auth = req.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:].strip()
    if auth.startswith("bearer "):
        return auth[7:].strip()
    x_key = req.headers.get("x-api-key") or req.headers.get("api-key")
    if x_key:
        return x_key.strip()
    return None


@router.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest, http_request: Request):
    start_time = time.time()
    prompt_tokens = _calc_prompt_tokens(request.messages, tools=request.tools)

    # 0. Local API Key Authentication & Verification
    matched_api_key = None
    all_keys = plugin_manager.get_api_keys()
    permanent_active_keys = [k for k in all_keys if k.enabled and k.is_valid() and not k.is_temporary]
    token = _extract_api_key(http_request)

    if token:
        matched = next((k for k in all_keys if k.key == token), None)
        if not matched:
            if permanent_active_keys:
                raise HTTPException(
                    status_code=401,
                    detail={
                        "error": {
                            "message": "Invalid API key or key is disabled.",
                            "type": "invalid_request_error",
                            "code": "invalid_api_key",
                        }
                    },
                )
        else:
            if not matched.is_valid():
                raise HTTPException(
                    status_code=401,
                    detail={
                        "error": {
                            "message": "API key has expired or reached its maximum request limit.",
                            "type": "invalid_request_error",
                            "code": "api_key_expired",
                        }
                    },
                )
            matched_api_key = matched
    elif permanent_active_keys:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "Missing API key. Pass 'Authorization: Bearer <key>' or 'x-api-key'.",
                    "type": "invalid_request_error",
                    "code": "missing_api_key",
                }
            },
        )

    # 1. Check for example test-model (not exposed in /models or provider settings)
    if request.model == "test-model":
        now_str = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        latency_ms = max(0.5, round((time.time() - start_time) * 1000, 2))
        completion_tokens = 22
        total_tokens = prompt_tokens + completion_tokens

        reply_content = (
            f"Simulated response from LLMProxy (test-model).\n\n---\n"
            f"**Execution Details:**\n"
            f"• **Time taken:** {latency_ms:.2f} ms\n"
            f"• **Timestamp:** {now_str}\n"
            f"• **Model:** test-model\n"
            f"• **Tokens:** {prompt_tokens} prompt, {completion_tokens} completion ({total_tokens} total)\n"
            f"• **Account status:** No account configured (Simulation Mode)"
        )

        if matched_api_key:
            plugin_manager.record_api_key_usage(
                matched_api_key.id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

        plugin_manager.add_log(
            RequestLogEntry(
                id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
                model=request.model,
                plugin_id=None,
                api_key_id=matched_api_key.id if matched_api_key else None,
                status="success",
                status_code=200,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        )

        if request.stream:
            async def test_model_sse_stream() -> AsyncIterator[str]:
                chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
                created = int(time.time())

                init_chunk = ChatCompletionChunk(
                    id=chunk_id,
                    created=created,
                    model="test-model",
                    choices=[ChatChoiceDelta(index=0, delta={"role": "assistant", "content": ""})],
                )
                yield f"data: {init_chunk.model_dump_json()}\n\n"

                base_words = ["Simulated ", "response ", "from ", "LLMProxy ", "(test-model).\n\n"]
                for w in base_words:
                    c = ChatCompletionChunk(
                        id=chunk_id,
                        created=created,
                        model="test-model",
                        choices=[ChatChoiceDelta(index=0, delta={"content": w})],
                    )
                    yield f"data: {c.model_dump_json()}\n\n"

                details_text = (
                    f"---\n"
                    f"**Execution Details:**\n"
                    f"• **Time taken:** {latency_ms:.2f} ms\n"
                    f"• **Timestamp:** {now_str}\n"
                    f"• **Model:** test-model\n"
                    f"• **Tokens:** {prompt_tokens} prompt, {completion_tokens} completion ({total_tokens} total)\n"
                    f"• **Account status:** No account configured (Simulation Mode)"
                )
                details_chunk = ChatCompletionChunk(
                    id=chunk_id,
                    created=created,
                    model="test-model",
                    choices=[ChatChoiceDelta(index=0, delta={"content": details_text})],
                )
                yield f"data: {details_chunk.model_dump_json()}\n\n"

                final_chunk = ChatCompletionChunk(
                    id=chunk_id,
                    created=created,
                    model="test-model",
                    choices=[ChatChoiceDelta(index=0, delta={}, finish_reason="stop")],
                )
                yield f"data: {final_chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                test_model_sse_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model="test-model",
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=reply_content),
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ),
        )

    # 1. Resolve model to plugin, model config, and variables
    resolved = plugin_manager.resolve_model(request.model)
    if resolved is None:
        latency_ms = round((time.time() - start_time) * 1000, 2)
        plugin_manager.add_log(
            RequestLogEntry(
                id=f"req-{uuid.uuid4().hex[:12]}",
                model=request.model,
                plugin_id=None,
                api_key_id=matched_api_key.id if matched_api_key else None,
                status="error",
                status_code=404,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
                error_message=f"Model '{request.model}' not found",
            )
        )
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "message": f"Model '{request.model}' not found",
                    "type": "invalid_request_error",
                    "code": "model_not_found",
                }
            },
        )

    model_config, plugin, variables = resolved

    # 2. Validate variables for the plugin
    var_errors = plugin.validate_variables(variables)
    if var_errors:
        latency_ms = round((time.time() - start_time) * 1000, 2)
        err_msg = "; ".join(var_errors)
        plugin_manager.add_log(
            RequestLogEntry(
                id=f"req-{uuid.uuid4().hex[:12]}",
                model=request.model,
                plugin_id=plugin.metadata.id,
                status="error",
                status_code=400,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
                error_message=err_msg,
            )
        )
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": f"Plugin configuration invalid: {err_msg}",
                    "type": "invalid_request_error",
                    "code": "invalid_plugin_config",
                }
            },
        )

    # 3. Select Account with Load Balancing & Cooldown checks
    strategy = str(variables.get("load_balancing_strategy", "round_robin"))
    accounts = plugin_manager.get_accounts(plugin.metadata.id)
    account: Optional[Account] = None

    if accounts:
        account = plugin_manager.select_account(plugin.metadata.id, strategy=strategy)
        if not account:
            cooling_down = [
                a for a in accounts
                if a.rate_limited_until and a.rate_limited_until > time.time()
            ]
            if cooling_down:
                min_wait = max(1, int(min(a.rate_limited_until - time.time() for a in cooling_down)))
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": {
                            "message": f"All accounts for provider '{plugin.metadata.id}' are in cooldown. Retry in {min_wait}s.",
                            "type": "rate_limit_error",
                            "code": "accounts_rate_limited",
                        }
                    },
                )
            raise HTTPException(
                status_code=503,
                detail={
                    "error": {
                        "message": f"No active accounts available for provider '{plugin.metadata.id}'",
                        "type": "service_unavailable_error",
                        "code": "no_active_accounts",
                    }
                },
            )
    else:
        # Fallback account when no dedicated accounts have been configured yet
        account = Account(
            id=f"default-{plugin.metadata.id}",
            provider_id=plugin.metadata.id,
            name="Default Account",
            api_key=str(variables.get("api_key", "")),
            base_url=variables.get("base_url"),
        )

    cooldown_sec = _safe_float(variables.get("account_switch_cooldown"), 10.0)

    # 4. Resolve External Proxy (off by default)
    init_proxy_url, current_proxy = plugin_manager.resolve_request_proxy(
        variables=variables,
        account=account,
        provider_id=plugin.metadata.id,
    )
    if init_proxy_url:
        variables["proxy"] = init_proxy_url

    # 5. Handle Streaming vs Non-Streaming
    if request.stream:
        async def sse_event_stream() -> AsyncIterator[str]:
            completion_tokens = 0
            log_id = f"chatcmpl-stream-{uuid.uuid4().hex[:12]}"
            current_account = account
            try:
                stream_result = await _invoke_plugin_completion(
                    plugin, request, model_config, variables, current_account
                )
                async for chunk in stream_result:
                    if isinstance(chunk, ChatCompletionChunk):
                        if chunk.id:
                            log_id = chunk.id
                        for choice in chunk.choices:
                            if choice.delta.get("content"):
                                completion_tokens += _estimate_tokens(choice.delta["content"])
                            if choice.delta.get("tool_calls"):
                                completion_tokens += _estimate_tokens(choice.delta["tool_calls"])
                        yield f"data: {chunk.model_dump_json()}\n\n"
                    elif isinstance(chunk, str):
                        yield f"data: {chunk}\n\n"

                include_usage = bool((request.stream_options or {}).get("include_usage"))
                if include_usage:
                    usage_chunk = ChatCompletionChunk(
                        id=log_id,
                        created=int(time.time()),
                        model=request.model,
                        choices=[],
                        usage=CompletionUsage(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=prompt_tokens + completion_tokens,
                        ),
                    )
                    yield f"data: {usage_chunk.model_dump_json()}\n\n"

                yield "data: [DONE]\n\n"

                latency_ms = round((time.time() - start_time) * 1000, 2)
                if current_account:
                    plugin_manager.mark_account_result(current_account.id, success=True)
                if current_proxy:
                    plugin_manager.mark_proxy_result(current_proxy.id, success=True)
                if matched_api_key:
                    plugin_manager.record_api_key_usage(
                        matched_api_key.id,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                    )

                plugin_manager.add_log(
                    RequestLogEntry(
                        id=log_id,
                        model=request.model,
                        plugin_id=plugin.metadata.id,
                        api_key_id=matched_api_key.id if matched_api_key else None,
                        status="success",
                        status_code=200,
                        latency_ms=latency_ms,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                    )
                )
            except Exception as e:
                latency_ms = round((time.time() - start_time) * 1000, 2)
                is_rate_limit = _is_rate_limit_error(e)
                if current_account:
                    plugin_manager.mark_account_result(
                        current_account.id,
                        success=False,
                        is_rate_limit=is_rate_limit,
                        cooldown_seconds=cooldown_sec,
                    )
                if current_proxy:
                    plugin_manager.mark_proxy_result(
                        current_proxy.id,
                        success=False,
                        is_rate_limit=is_rate_limit,
                        cooldown_seconds=cooldown_sec,
                    )

                plugin_manager.add_log(
                    RequestLogEntry(
                        id=log_id,
                        model=request.model,
                        plugin_id=plugin.metadata.id,
                        api_key_id=matched_api_key.id if matched_api_key else None,
                        status="error",
                        status_code=429 if is_rate_limit else 500,
                        latency_ms=latency_ms,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        error_message=str(e),
                    )
                )
                err_payload = json.dumps({"error": {"message": str(e), "type": "server_error"}})
                yield f"data: {err_payload}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            sse_event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming with account failover and proxy switching on limit
    max_retries = max(1, _safe_int(variables.get("max_retries"), 2))
    switch_wait = _safe_float(variables.get("account_switch_wait"), 0.0)
    current_account = account
    last_exception: Optional[Exception] = None
    excluded_proxies: Set[str] = set()
    excluded_accounts: Set[str] = set()

    for attempt in range(max_retries):
        # Resolve proxy for current attempt
        proxy_url, attempt_proxy = plugin_manager.resolve_request_proxy(
            variables=variables,
            account=current_account,
            exclude_proxy_ids=excluded_proxies,
            provider_id=plugin.metadata.id,
        )
        if proxy_url:
            variables["proxy"] = proxy_url
        elif "proxy" in variables and not bool(variables.get("proxy_enabled", False)):
            pass
        else:
            variables.pop("proxy", None)

        try:
            response = await _invoke_plugin_completion(
                plugin, request, model_config, variables, current_account
            )
            latency_ms = round((time.time() - start_time) * 1000, 2)

            completion_tokens = 0
            if isinstance(response, ChatCompletionResponse):
                if response.usage and response.usage.prompt_tokens:
                    prompt_tokens = response.usage.prompt_tokens
                if response.usage and response.usage.completion_tokens:
                    completion_tokens = response.usage.completion_tokens
                elif response.choices:
                    msg = response.choices[0].message
                    if msg.content:
                        completion_tokens += _estimate_tokens(msg.content)
                    if getattr(msg, "tool_calls", None):
                        completion_tokens += _estimate_tokens(msg.tool_calls)
                    response.usage = CompletionUsage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                    )

            if current_account:
                plugin_manager.mark_account_result(current_account.id, success=True)
            if attempt_proxy:
                plugin_manager.mark_proxy_result(attempt_proxy.id, success=True)
            if matched_api_key:
                plugin_manager.record_api_key_usage(
                    matched_api_key.id,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )

            plugin_manager.add_log(
                RequestLogEntry(
                    id=getattr(response, "id", f"req-{uuid.uuid4().hex[:12]}"),
                    model=request.model,
                    plugin_id=plugin.metadata.id,
                    api_key_id=matched_api_key.id if matched_api_key else None,
                    status="success",
                    status_code=200,
                    latency_ms=latency_ms,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            )
            return response
        except Exception as e:
            last_exception = e
            is_rate_limit = _is_rate_limit_error(e)
            if attempt_proxy:
                plugin_manager.mark_proxy_result(
                    attempt_proxy.id,
                    success=False,
                    is_rate_limit=is_rate_limit,
                    cooldown_seconds=cooldown_sec,
                )
                excluded_proxies.add(attempt_proxy.id)
            if current_account:
                plugin_manager.mark_account_result(
                    current_account.id,
                    success=False,
                    is_rate_limit=is_rate_limit,
                    cooldown_seconds=cooldown_sec,
                )
                excluded_accounts.add(current_account.id)

            # Try to failover to next proxy or next account if retries remain
            if attempt < max_retries - 1:
                # Check if proxy switching on limit applies
                proxy_enabled = bool(variables.get("proxy_enabled", False))
                if proxy_enabled:
                    next_proxy_url, next_p = plugin_manager.resolve_request_proxy(
                        variables=variables,
                        account=current_account,
                        exclude_proxy_ids=excluded_proxies,
                        provider_id=plugin.metadata.id,
                    )
                    if next_p:
                        if switch_wait > 0:
                            await asyncio.sleep(switch_wait)
                        continue

                if accounts:
                    if switch_wait > 0:
                        await asyncio.sleep(switch_wait)
                    next_acc = plugin_manager.select_account(
                        plugin.metadata.id,
                        strategy=strategy,
                        exclude_account_ids=excluded_accounts,
                    )
                    if not next_acc:
                        next_acc = plugin_manager.select_account(plugin.metadata.id, strategy=strategy)
                    if next_acc and next_acc.id != current_account.id:
                        current_account = next_acc
                        continue
            break

    # If all attempts failed
    latency_ms = round((time.time() - start_time) * 1000, 2)
    is_rate_limit = _is_rate_limit_error(last_exception)
    status_code = 429 if is_rate_limit else 500
    plugin_manager.add_log(
        RequestLogEntry(
            id=f"req-{uuid.uuid4().hex[:12]}",
            model=request.model,
            plugin_id=plugin.metadata.id,
            api_key_id=matched_api_key.id if matched_api_key else None,
            status="error",
            status_code=status_code,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=0,
            error_message=str(last_exception),
        )
    )
    if is_rate_limit:
        raise HTTPException(
            status_code=429,
            detail={
                "error": {
                    "message": f"Usage limit reached: {last_exception}",
                    "type": "rate_limit_error",
                    "code": "usage_limit_reached",
                }
            },
        )
    raise HTTPException(
        status_code=status_code,
        detail={
            "error": {
                "message": f"Plugin execution error: {last_exception}",
                "type": "server_error",
                "code": "plugin_execution_failed",
            }
        },
    )


@router.get("/models", response_model=ModelListResponse)
async def list_models():
    """List all available models from active plugins and load balancing rules in OpenAI format."""
    cards: List[ModelCard] = []
    seen = set()

    # Include custom load balancing rule models
    for rule in plugin_manager.get_load_balancing_rules():
        if rule.enabled and rule.model_id not in seen:
            cards.append(ModelCard(id=rule.model_id, owned_by="load_balancer"))
            seen.add(rule.model_id)

    active_models = plugin_manager.get_all_active_models()
    for model_config, plugin in active_models:
        if model_config.id not in seen:
            cards.append(
                ModelCard(
                    id=model_config.id,
                    owned_by=plugin.metadata.id,
                )
            )
            seen.add(model_config.id)
    return ModelListResponse(data=cards)


@router.get("/models/{model_id}", response_model=ModelCard)
async def get_model(model_id: str):
    """Get information about a specific model."""
    for rule in plugin_manager.get_load_balancing_rules():
        if rule.enabled and rule.model_id == model_id:
            return ModelCard(id=rule.model_id, owned_by="load_balancer")

    active_models = plugin_manager.get_all_active_models()
    for model_config, plugin in active_models:
        if model_config.id == model_id:
            return ModelCard(id=model_config.id, owned_by=plugin.metadata.id)
    raise HTTPException(
        status_code=404,
        detail={
            "error": {
                "message": f"Model '{model_id}' not found",
                "type": "invalid_request_error",
                "code": "model_not_found",
            }
        },
    )
