from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, AsyncIterator, List, Optional, Set, Tuple

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


def _usage_limit_error(provider_id: str) -> HTTPException:
    message = f"Usage limit reached for all accounts of provider '{provider_id}'."
    return HTTPException(
        status_code=429,
        detail={
            "error": {
                "message": message,
                "type": "rate_limit_error",
                "code": "usage_limit_reached",
            }
        },
    )


def _is_meaningful_stream_chunk(chunk: Any) -> bool:
    """Return True once a streamed chunk contains actual output rather than a role-only preamble."""
    if isinstance(chunk, str):
        return True
    if isinstance(chunk, ChatCompletionChunk):
        for choice in chunk.choices:
            delta = choice.delta or {}
            if (
                delta.get("content")
                or delta.get("reasoning_content")
                or delta.get("reasoning")
                or delta.get("tool_calls")
            ):
                return True
            if choice.finish_reason is not None:
                return True
        return False
    return True


async def _single_chunk_stream(chunk: ChatCompletionChunk) -> AsyncIterator[ChatCompletionChunk]:
    yield chunk


async def _prime_stream(result: Any) -> Tuple[Any, List[Any]]:
    """
    Consume a stream until the first meaningful chunk (or the end) so upstream
    failures can be retried on another account before any bytes reach the client.
    """
    if hasattr(result, "__aiter__"):
        iterator = result.__aiter__()
        buffered: List[Any] = []
        try:
            while True:
                try:
                    chunk = await iterator.__anext__()
                except StopAsyncIteration:
                    break
                buffered.append(chunk)
                if _is_meaningful_stream_chunk(chunk):
                    break
        except Exception:
            await _close_stream(iterator)
            raise
        return iterator, buffered

    if isinstance(result, ChatCompletionResponse):
        message = result.choices[0].message if result.choices else None
        chunk = ChatCompletionChunk(
            id=result.id,
            created=result.created,
            model=result.model,
            choices=[
                ChatChoiceDelta(
                    index=0,
                    delta={
                        "role": "assistant",
                        "content": message.content if message else "",
                    },
                    finish_reason="stop",
                )
            ],
            usage=result.usage,
        )
        return _single_chunk_stream(chunk).__aiter__(), [chunk]

    raise TypeError(f"Plugin returned an unsupported streaming result: {type(result)!r}")


async def _close_stream(iterator: Any) -> None:
    aclose = getattr(iterator, "aclose", None)
    if aclose is not None:
        try:
            await aclose()
        except Exception:
            pass


def _prepare_attempt_variables(
    variables: dict,
    account: Account,
    provider_id: str,
    excluded_proxies: Set[str],
) -> Tuple[dict, Optional[Any]]:
    proxy_url, attempt_proxy = plugin_manager.resolve_request_proxy(
        variables=variables,
        account=account,
        exclude_proxy_ids=excluded_proxies,
        provider_id=provider_id,
    )
    attempt_vars = dict(variables)
    if proxy_url:
        attempt_vars["proxy"] = proxy_url
    elif not bool(variables.get("proxy_enabled", False)):
        attempt_vars.pop("proxy", None)
    return attempt_vars, attempt_proxy


def _record_attempt_failure(
    account: Optional[Account],
    proxy: Optional[Any],
    exc: Exception,
    cooldown_sec: float,
) -> bool:
    is_rate_limit = _is_rate_limit_error(exc)
    if account and not account.id.startswith("default-"):
        plugin_manager.mark_account_result(
            account.id,
            success=False,
            is_rate_limit=is_rate_limit,
            cooldown_seconds=cooldown_sec,
        )
    if proxy:
        plugin_manager.mark_proxy_result(
            proxy.id,
            success=False,
            is_rate_limit=is_rate_limit,
            cooldown_seconds=cooldown_sec,
        )
    return is_rate_limit


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

    # 3. Build the ordered account candidates for this request.
    strategy = str(variables.get("load_balancing_strategy", "round_robin"))
    accounts = plugin_manager.get_accounts(plugin.metadata.id)
    default_account = Account(
        id=f"default-{plugin.metadata.id}",
        provider_id=plugin.metadata.id,
        name="Default Account",
        api_key=str(variables.get("api_key", "")),
        base_url=variables.get("base_url"),
    )

    if accounts:
        candidates = plugin_manager.get_account_candidates(
            plugin.metadata.id, strategy=strategy
        )
        if not candidates:
            cooling_down = [
                a for a in accounts
                if a.rate_limited_until and a.rate_limited_until > time.time()
            ]
            if cooling_down:
                min_wait = max(1, int(min(a.rate_limited_until - time.time() for a in cooling_down)))
                latency_ms = round((time.time() - start_time) * 1000, 2)
                plugin_manager.add_log(
                    RequestLogEntry(
                        id=f"req-{uuid.uuid4().hex[:12]}",
                        model=request.model,
                        plugin_id=plugin.metadata.id,
                        api_key_id=matched_api_key.id if matched_api_key else None,
                        status="error",
                        status_code=429,
                        latency_ms=latency_ms,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=0,
                        error_message=f"All accounts for provider '{plugin.metadata.id}' are in cooldown.",
                    )
                )
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": {
                            "message": f"All accounts for provider '{plugin.metadata.id}' are in cooldown. Retry in {min_wait}s.",
                            "type": "rate_limit_error",
                            "code": "accounts_rate_limited",
                        }
                    },
                    headers={"Retry-After": str(min_wait)},
                )

            latency_ms = round((time.time() - start_time) * 1000, 2)
            plugin_manager.add_log(
                RequestLogEntry(
                    id=f"req-{uuid.uuid4().hex[:12]}",
                    model=request.model,
                    plugin_id=plugin.metadata.id,
                    api_key_id=matched_api_key.id if matched_api_key else None,
                    status="error",
                    status_code=503,
                    latency_ms=latency_ms,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=0,
                    error_message=f"No active accounts available for provider '{plugin.metadata.id}'",
                )
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
        candidates = [default_account]

    cooldown_sec = _safe_float(variables.get("account_switch_cooldown"), 10.0)
    switch_wait = _safe_float(variables.get("account_switch_wait"), 0.0)

    # 4. Handle Streaming vs Non-Streaming
    if request.stream:
        include_usage = bool((request.stream_options or {}).get("include_usage"))
        excluded_proxies: Set[str] = set()
        last_exception: Optional[Exception] = None
        last_rate_limit = False
        stream_state: Optional[Tuple[Any, List[Any], Account, Optional[Any]]] = None

        # Prime each candidate stream until it produces real output. This lets us
        # fail over to another account when the current one is rate limited before
        # any bytes have been sent to the client.
        for idx, candidate in enumerate(candidates):
            attempt_vars, attempt_proxy = _prepare_attempt_variables(
                variables, candidate, plugin.metadata.id, excluded_proxies
            )
            iterator = None
            try:
                result = await _invoke_plugin_completion(
                    plugin, request, model_config, attempt_vars, candidate
                )
                iterator, buffered = await _prime_stream(result)
                stream_state = (iterator, buffered, candidate, attempt_proxy)
                break
            except Exception as e:
                last_exception = e
                last_rate_limit = _record_attempt_failure(
                    candidate, attempt_proxy, e, cooldown_sec
                )
                if attempt_proxy:
                    excluded_proxies.add(attempt_proxy.id)
                if iterator is not None:
                    await _close_stream(iterator)
                if switch_wait > 0 and not last_rate_limit and idx < len(candidates) - 1:
                    await asyncio.sleep(switch_wait)

        if stream_state is None:
            latency_ms = round((time.time() - start_time) * 1000, 2)
            status_code = 429 if last_rate_limit else 500
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
            if last_rate_limit:
                raise _usage_limit_error(plugin.metadata.id)
            raise HTTPException(
                status_code=500,
                detail={
                    "error": {
                        "message": f"Plugin execution error: {last_exception}",
                        "type": "server_error",
                        "code": "plugin_execution_failed",
                    }
                },
            )

        iterator, buffered, current_account, current_proxy = stream_state
        log_id = f"chatcmpl-stream-{uuid.uuid4().hex[:12]}"

        async def sse_event_stream() -> AsyncIterator[str]:
            nonlocal log_id, prompt_tokens
            completion_tokens = 0
            last_usage: Optional[CompletionUsage] = None

            def handle_chunk(chunk: Any) -> Optional[str]:
                nonlocal log_id, prompt_tokens, completion_tokens, last_usage
                if isinstance(chunk, ChatCompletionChunk):
                    if chunk.id:
                        log_id = chunk.id
                    if chunk.usage is not None:
                        last_usage = chunk.usage
                        if chunk.usage.prompt_tokens:
                            prompt_tokens = chunk.usage.prompt_tokens
                        if chunk.usage.completion_tokens:
                            completion_tokens = chunk.usage.completion_tokens
                    else:
                        for choice in chunk.choices:
                            if choice.delta.get("content"):
                                completion_tokens += _estimate_tokens(choice.delta["content"])
                            if choice.delta.get("tool_calls"):
                                completion_tokens += _estimate_tokens(choice.delta["tool_calls"])
                    if include_usage and chunk.usage is not None:
                        # OpenAI clients expect usage only in the terminal usage chunk.
                        chunk = chunk.model_copy(update={"usage": None})
                    return f"data: {chunk.model_dump_json()}\n\n"
                if isinstance(chunk, str):
                    return f"data: {chunk}\n\n"
                return None

            try:
                for chunk in buffered:
                    event = handle_chunk(chunk)
                    if event:
                        yield event

                async for chunk in iterator:
                    event = handle_chunk(chunk)
                    if event:
                        yield event

                if include_usage:
                    usage_payload = last_usage or CompletionUsage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                    )
                    usage_chunk = ChatCompletionChunk(
                        id=log_id,
                        model=request.model,
                        choices=[],
                        usage=usage_payload,
                    )
                    yield f"data: {usage_chunk.model_dump_json()}\n\n"

                yield "data: [DONE]\n\n"

                latency_ms = round((time.time() - start_time) * 1000, 2)
                if current_account and not current_account.id.startswith("default-"):
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
                if current_account and not current_account.id.startswith("default-"):
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
            finally:
                await _close_stream(iterator)

        return StreamingResponse(
            sse_event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming: try every available account before returning a limit error.
    last_exception: Optional[Exception] = None
    last_rate_limit = False
    excluded_proxies: Set[str] = set()

    for idx, candidate in enumerate(candidates):
        attempt_vars, attempt_proxy = _prepare_attempt_variables(
            variables, candidate, plugin.metadata.id, excluded_proxies
        )
        try:
            response = await _invoke_plugin_completion(
                plugin, request, model_config, attempt_vars, candidate
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

            if candidate and not candidate.id.startswith("default-"):
                plugin_manager.mark_account_result(candidate.id, success=True)
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
            last_rate_limit = _record_attempt_failure(
                candidate, attempt_proxy, e, cooldown_sec
            )
            if attempt_proxy:
                excluded_proxies.add(attempt_proxy.id)
            if switch_wait > 0 and not last_rate_limit and idx < len(candidates) - 1:
                await asyncio.sleep(switch_wait)

    # If every candidate failed, return a usage limit error for rate limits.
    latency_ms = round((time.time() - start_time) * 1000, 2)
    status_code = 429 if last_rate_limit else 500
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
    if last_rate_limit:
        raise _usage_limit_error(plugin.metadata.id)
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
                    reasoning_levels=list(model_config.reasoning_levels),
                    context_window=model_config.context_window,
                    max_tokens=model_config.max_tokens,
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
            return ModelCard(
                id=model_config.id,
                owned_by=plugin.metadata.id,
                reasoning_levels=list(model_config.reasoning_levels),
                context_window=model_config.context_window,
                max_tokens=model_config.max_tokens,
            )
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
