from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncIterator, List

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from llmproxy.core.manager import plugin_manager
from llmproxy.schemas import (
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
    return 1


def _calc_prompt_tokens(messages: list) -> int:
    return sum(_estimate_tokens(m.content) for m in messages)


@router.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    start_time = time.time()
    prompt_tokens = _calc_prompt_tokens(request.messages)

    # 1. Resolve model to plugin, model config, and variables
    resolved = plugin_manager.resolve_model(request.model)
    if resolved is None:
        latency_ms = round((time.time() - start_time) * 1000, 2)
        plugin_manager.add_log(
            RequestLogEntry(
                id=f"req-{uuid.uuid4().hex[:12]}",
                model=request.model,
                plugin_id=None,
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

    # 3. Handle Streaming vs Non-Streaming
    if request.stream:
        async def sse_event_stream() -> AsyncIterator[str]:
            completion_tokens = 0
            log_id = f"chatcmpl-stream-{uuid.uuid4().hex[:12]}"
            try:
                stream_result = await plugin.chat_completion(request, model_config, variables)
                async for chunk in stream_result:
                    if isinstance(chunk, ChatCompletionChunk):
                        if chunk.id:
                            log_id = chunk.id
                        for choice in chunk.choices:
                            if choice.delta.get("content"):
                                completion_tokens += _estimate_tokens(choice.delta["content"])
                        yield f"data: {chunk.model_dump_json()}\n\n"
                    elif isinstance(chunk, str):
                        yield f"data: {chunk}\n\n"

                yield "data: [DONE]\n\n"

                latency_ms = round((time.time() - start_time) * 1000, 2)
                plugin_manager.add_log(
                    RequestLogEntry(
                        id=log_id,
                        model=request.model,
                        plugin_id=plugin.metadata.id,
                        status="success",
                        status_code=200,
                        latency_ms=latency_ms,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                    )
                )
            except Exception as e:
                latency_ms = round((time.time() - start_time) * 1000, 2)
                plugin_manager.add_log(
                    RequestLogEntry(
                        id=log_id,
                        model=request.model,
                        plugin_id=plugin.metadata.id,
                        status="error",
                        status_code=500,
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
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    # Non-streaming
    try:
        response = await plugin.chat_completion(request, model_config, variables)
        latency_ms = round((time.time() - start_time) * 1000, 2)

        completion_tokens = 0
        if isinstance(response, ChatCompletionResponse):
            if response.usage and response.usage.prompt_tokens:
                prompt_tokens = response.usage.prompt_tokens
            if response.usage and response.usage.completion_tokens:
                completion_tokens = response.usage.completion_tokens
            elif response.choices and response.choices[0].message.content:
                completion_tokens = _estimate_tokens(response.choices[0].message.content)
                response.usage = CompletionUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                )

        plugin_manager.add_log(
            RequestLogEntry(
                id=getattr(response, "id", f"req-{uuid.uuid4().hex[:12]}"),
                model=request.model,
                plugin_id=plugin.metadata.id,
                status="success",
                status_code=200,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        )
        return response
    except Exception as e:
        latency_ms = round((time.time() - start_time) * 1000, 2)
        plugin_manager.add_log(
            RequestLogEntry(
                id=f"req-{uuid.uuid4().hex[:12]}",
                model=request.model,
                plugin_id=plugin.metadata.id,
                status="error",
                status_code=500,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
                error_message=str(e),
            )
        )
        raise HTTPException(
            status_code=500,
            detail={"error": {"message": str(e), "type": "server_error"}},
        )


@router.get("/models", response_model=ModelListResponse)
async def list_models() -> ModelListResponse:
    active_models = plugin_manager.get_all_active_models()
    cards = [
        ModelCard(
            id=model_cfg.id,
            owned_by=plugin.metadata.id,
        )
        for model_cfg, plugin in active_models
    ]
    return ModelListResponse(object="list", data=cards)


@router.get("/models/{model_id}", response_model=ModelCard)
async def get_model(model_id: str) -> ModelCard:
    resolved = plugin_manager.resolve_model(model_id)
    if not resolved:
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
    model_cfg, plugin, _ = resolved
    return ModelCard(id=model_cfg.id, owned_by=plugin.metadata.id)
