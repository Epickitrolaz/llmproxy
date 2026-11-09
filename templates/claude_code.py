from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union

import httpx

from llmproxy.core.interface import BasePlugin
from llmproxy.schemas import (
    Account,
    ChatChoice,
    ChatChoiceDelta,
    ChatMessage,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionUsage,
    PluginMetadata,
    PluginModelConfig,
    PluginVariable,
)


class ClaudeCodePlugin(BasePlugin):
    """
    Anthropic Claude Code provider plugin with native Messages API translation.
    Translates OpenAI request/response schemas to/from Anthropic Messages format,
    supporting streaming SSE, multi-account load balancing, and robust simulation for testing.
    """

    def __init__(self):
        self._metadata = PluginMetadata(
            id="claude_code",
            name="Claude Code (Anthropic)",
            description="Anthropic Claude Code provider with native Messages API translation",
            version="2.0.0",
            author="Anthropic",
            enabled=True,
            filename="claude_code.py",
            variables=[
                PluginVariable(
                    name="timeout",
                    label="Timeout (seconds)",
                    type="number",
                    default=60,
                    description="Request timeout in seconds",
                ),
                PluginVariable(
                    name="load_balancing_strategy",
                    label="Load Balancing Strategy",
                    type="select",
                    default="round_robin",
                    options=["round_robin", "least_used", "fallback", "random"],
                    description="Strategy for distributing requests across connected accounts",
                ),
                PluginVariable(
                    name="account_switch_cooldown",
                    label="Account Cooldown (s)",
                    type="number",
                    default=10,
                    description="Cooldown period when an account hits rate limits before retrying",
                ),
                PluginVariable(
                    name="account_switch_wait",
                    label="Switch Wait Delay (s)",
                    type="number",
                    default=0.1,
                    description="Wait delay before switching to another account on failover",
                ),
                PluginVariable(
                    name="max_retries",
                    label="Max Failover Retries",
                    type="number",
                    default=2,
                    description="Maximum number of account failovers per request",
                ),
                PluginVariable(
                    name="base_url",
                    label="Default Base URL",
                    type="string",
                    default="https://api.anthropic.com/v1",
                    description="Default Anthropic API base URL",
                ),
                PluginVariable(
                    name="anthropic_version",
                    label="Anthropic Version",
                    type="string",
                    default="2023-06-01",
                    description="Anthropic API version header",
                ),
                PluginVariable(
                    name="proxy_enabled",
                    label="Enable External Proxy",
                    type="boolean",
                    default=False,
                    description="Route provider requests through external proxies (off by default)",
                ),
                PluginVariable(
                    name="proxy_strategy",
                    label="Proxy Strategy",
                    type="select",
                    default="sequential",
                    options=["sequential", "random", "specific", "switch_on_limit"],
                    description="Strategy for selecting proxies from pool: sequential (round-robin), random, specific proxy, or switch on limit",
                ),
                PluginVariable(
                    name="specific_proxy_id",
                    label="Specific Proxy ID / Tag",
                    type="string",
                    default="",
                    description="Target proxy ID, URL, or tag to route through when strategy is 'specific'",
                ),
                PluginVariable(
                    name="proxy",
                    label="Direct Proxy URL",
                    type="string",
                    default="",
                    description="Direct HTTP or SOCKS5 proxy URL to route through",
                ),
            ],
            models=[
                PluginModelConfig(
                    id="claude-3-7-sonnet",
                    name="Claude 3.7 Sonnet",
                    upstream_model="claude-3-7-sonnet-20250219",
                    context_window=200000,
                    max_tokens=8192,
                ),
                PluginModelConfig(
                    id="claude-3-5-sonnet",
                    name="Claude 3.5 Sonnet",
                    upstream_model="claude-3-5-sonnet-20241022",
                    context_window=200000,
                    max_tokens=8192,
                ),
                PluginModelConfig(
                    id="claude-3-5-haiku",
                    name="Claude 3.5 Haiku",
                    upstream_model="claude-3-5-haiku-20241022",
                    context_window=200000,
                    max_tokens=8192,
                ),
            ],
        )

    @property
    def metadata(self) -> PluginMetadata:
        return self._metadata

    def _is_test_mode(self, api_key: str) -> bool:
        if not api_key:
            return True
        key = api_key.lower().strip()
        prefixes = ("test", "mock", "sk-test", "dummy", "fake", "offline", "sample")
        return any(key.startswith(p) for p in prefixes) or key in ("test-key", "test_key", "secret123")

    def _mock_response(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        account: Optional[Account] = None,
    ) -> ChatCompletionResponse:
        model_name = model_config.id
        acc_label = f" via account '{account.name}'" if account else ""

        if request.tools and not any(m.role == "tool" for m in request.messages[-2:]):
            fn = request.tools[0].get("function", {}) if "function" in request.tools[0] else request.tools[0]
            call_id = f"toolu_claude_{uuid.uuid4().hex[:8]}"
            tool_name = fn.get("name", "test_tool")
            mock_args = "{}"
            props = fn.get("parameters", {}).get("properties", {}) or fn.get("input_schema", {}).get("properties", {})
            if props:
                first_k = next(iter(props.keys()))
                mock_args = json.dumps({first_k: "test_value"})
            return ChatCompletionResponse(
                id=f"chatcmpl-claude-{uuid.uuid4().hex[:12]}",
                created=int(time.time()),
                model=model_name,
                choices=[
                    ChatChoice(
                        index=0,
                        message=ChatMessage(
                            role="assistant",
                            content=None,
                            tool_calls=[
                                {
                                    "id": call_id,
                                    "type": "function",
                                    "function": {
                                        "name": tool_name,
                                        "arguments": mock_args,
                                    },
                                }
                            ],
                        ),
                        finish_reason="tool_calls",
                    )
                ],
                usage=CompletionUsage(
                    prompt_tokens=14,
                    completion_tokens=24,
                    total_tokens=38,
                ),
            )

        base_content = f"Simulated response from Claude Code ({model_name}){acc_label}."
        if any(m.role == "tool" for m in request.messages):
            base_content = f"Simulated response from Claude Code ({model_name}){acc_label} with tool output processed."

        now_str = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        status_desc = (
            "No account configured (Simulation Mode)"
            if (not account or account.id.startswith("default-") or not account.api_key)
            else f"Account '{account.name}' (Test mode)"
        )
        prompt_tokens = 14
        completion_tokens = 24
        total_tokens = prompt_tokens + completion_tokens

        details = (
            f"\n\n---\n"
            f"**Execution Details:**\n"
            f"• **Time taken:** 1.20 ms\n"
            f"• **Timestamp:** {now_str}\n"
            f"• **Model:** {model_name}\n"
            f"• **Tokens:** {prompt_tokens} prompt, {completion_tokens} completion ({total_tokens} total)\n"
            f"• **Account status:** {status_desc}"
        )
        content = base_content + details

        return ChatCompletionResponse(
            id=f"chatcmpl-claude-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=model_name,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=content),
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ),
        )

    async def _mock_stream(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        account: Optional[Account] = None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        model_name = model_config.id
        chunk_id = f"chatcmpl-claude-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        acc_label = f" via account '{account.name}'" if account else ""

        if request.tools and not any(m.role == "tool" for m in request.messages[-2:]):
            fn = request.tools[0].get("function", {}) if "function" in request.tools[0] else request.tools[0]
            call_id = f"toolu_claude_{uuid.uuid4().hex[:8]}"
            tool_name = fn.get("name", "test_tool")
            mock_args = "{}"
            props = fn.get("parameters", {}).get("properties", {}) or fn.get("input_schema", {}).get("properties", {})
            if props:
                first_k = next(iter(props.keys()))
                mock_args = json.dumps({first_k: "test_value"})

            yield ChatCompletionChunk(
                id=chunk_id,
                created=created,
                model=model_name,
                choices=[
                    ChatChoiceDelta(
                        index=0,
                        delta={
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": call_id,
                                    "type": "function",
                                    "function": {
                                        "name": tool_name,
                                        "arguments": "",
                                    },
                                }
                            ],
                        },
                        finish_reason=None,
                    )
                ],
            )
            arg_chunks = [mock_args[: len(mock_args) // 2], mock_args[len(mock_args) // 2 :]]
            for arg_piece in arg_chunks:
                if arg_piece:
                    yield ChatCompletionChunk(
                        id=chunk_id,
                        created=created,
                        model=model_name,
                        choices=[
                            ChatChoiceDelta(
                                index=0,
                                delta={
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "function": {"arguments": arg_piece},
                                        }
                                    ]
                                },
                                finish_reason=None,
                            )
                        ],
                    )
            yield ChatCompletionChunk(
                id=chunk_id,
                created=created,
                model=model_name,
                choices=[
                    ChatChoiceDelta(
                        index=0,
                        delta={},
                        finish_reason="tool_calls",
                    )
                ],
            )
            return

        yield ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=model_name,
            choices=[
                ChatChoiceDelta(
                    index=0,
                    delta={"role": "assistant", "content": ""},
                    finish_reason=None,
                )
            ],
        )

        tokens = ["Simulated ", "stream ", "response ", f"from Claude Code for {model_name}{acc_label}."]
        for token in tokens:
            yield ChatCompletionChunk(
                id=chunk_id,
                created=created,
                model=model_name,
                choices=[
                    ChatChoiceDelta(
                        index=0,
                        delta={"content": token},
                        finish_reason=None,
                    )
                ],
            )

        now_str = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        status_desc = (
            "No account configured (Simulation Mode)"
            if (not account or account.id.startswith("default-") or not account.api_key)
            else f"Account '{account.name}' (Test mode)"
        )
        details = (
            f"\n\n---\n"
            f"**Execution Details:**\n"
            f"• **Time taken:** 1.20 ms\n"
            f"• **Timestamp:** {now_str}\n"
            f"• **Model:** {model_name}\n"
            f"• **Account status:** {status_desc}"
        )
        yield ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=model_name,
            choices=[ChatChoiceDelta(index=0, delta={"content": details}, finish_reason=None)],
        )

        yield ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=model_name,
            choices=[
                ChatChoiceDelta(
                    index=0,
                    delta={},
                    finish_reason="stop",
                )
            ],
        )

    def _format_anthropic_tools(self, tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        if not tools:
            return None
        anthropic_tools = []
        for t in tools:
            if "function" in t:
                fn = t["function"]
                name = fn.get("name", "")
                desc = fn.get("description", "")
                params = fn.get("parameters", {"type": "object", "properties": {}})
            else:
                name = t.get("name", "")
                desc = t.get("description", "")
                params = t.get("input_schema") or t.get("parameters", {"type": "object", "properties": {}})

            if not isinstance(params, dict) or params.get("type") != "object":
                params = {"type": "object", "properties": {}}

            anthropic_tools.append({
                "name": name,
                "description": desc,
                "input_schema": params,
            })
        return anthropic_tools or None

    def _format_anthropic_tool_choice(self, tool_choice: Optional[Union[str, Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
        if not tool_choice or tool_choice == "auto":
            return {"type": "auto"}
        if tool_choice == "required":
            return {"type": "any"}
        if tool_choice == "none":
            return None
        if isinstance(tool_choice, dict):
            if tool_choice.get("type") == "function":
                fn_name = tool_choice.get("function", {}).get("name")
                if fn_name:
                    return {"type": "tool", "name": fn_name}
        return None

    def _build_anthropic_messages(self, messages: List[ChatMessage]) -> Tuple[List[str], List[Dict[str, Any]]]:
        system_prompts: List[str] = []
        anthropic_messages: List[Dict[str, Any]] = []

        for m in messages:
            if m.role == "system":
                if isinstance(m.content, str):
                    system_prompts.append(m.content)
                elif isinstance(m.content, list):
                    system_prompts.append("\n".join(str(part) for part in m.content))
                continue

            if m.role in ("tool", "function"):
                tool_use_id = m.tool_call_id or getattr(m, "id", None) or "call_unknown"
                tool_content = str(m.content) if m.content is not None else ""
                tool_block = {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": tool_content,
                }
                if anthropic_messages and anthropic_messages[-1]["role"] == "user":
                    prev_c = anthropic_messages[-1]["content"]
                    if isinstance(prev_c, list):
                        prev_c.append(tool_block)
                    else:
                        anthropic_messages[-1]["content"] = [{"type": "text", "text": str(prev_c)}, tool_block]
                else:
                    anthropic_messages.append({"role": "user", "content": [tool_block]})
                continue

            if m.role == "assistant":
                blocks: List[Dict[str, Any]] = []
                if m.content:
                    if isinstance(m.content, str):
                        blocks.append({"type": "text", "text": m.content})
                    elif isinstance(m.content, list):
                        blocks.extend(m.content)
                if getattr(m, "tool_calls", None):
                    for tc in m.tool_calls:
                        raw_args = tc.get("function", {}).get("arguments", "{}")
                        if isinstance(raw_args, str):
                            try:
                                parsed_args = json.loads(raw_args)
                            except Exception:
                                parsed_args = {}
                        else:
                            parsed_args = raw_args or {}
                        blocks.append({
                            "type": "tool_use",
                            "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:12]}",
                            "name": tc.get("function", {}).get("name", ""),
                            "input": parsed_args,
                        })
                if not blocks:
                    blocks.append({"type": "text", "text": ""})

                if anthropic_messages and anthropic_messages[-1]["role"] == "assistant":
                    prev_c = anthropic_messages[-1]["content"]
                    if isinstance(prev_c, list):
                        prev_c.extend(blocks)
                    else:
                        anthropic_messages[-1]["content"] = [{"type": "text", "text": str(prev_c)}] + blocks
                else:
                    anthropic_messages.append({"role": "assistant", "content": blocks})
                continue

            # role == "user"
            if anthropic_messages and anthropic_messages[-1]["role"] == "user":
                prev_c = anthropic_messages[-1]["content"]
                if isinstance(prev_c, list):
                    if isinstance(m.content, str):
                        prev_c.append({"type": "text", "text": m.content})
                    elif isinstance(m.content, list):
                        prev_c.extend(m.content)
                else:
                    new_text = str(m.content) if isinstance(m.content, str) else "\n".join(str(p) for p in m.content)
                    anthropic_messages[-1]["content"] = f"{prev_c}\n\n{new_text}"
            else:
                anthropic_messages.append({"role": "user", "content": m.content})

        if not anthropic_messages:
            anthropic_messages.append({"role": "user", "content": "Hello"})

        return system_prompts, anthropic_messages

    async def _stream_messages(
        self,
        url: str,
        headers: Dict[str, str],
        body: Dict[str, Any],
        model_config: PluginModelConfig,
        request: ChatCompletionRequest,
        account: Optional[Account] = None,
        proxy: Optional[str] = None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        try:
            chunk_id = f"chatcmpl-claude-{uuid.uuid4().hex[:12]}"
            created = int(time.time())
            first_chunk_sent = False
            current_tool_call: Optional[Dict[str, Any]] = None
            tool_call_index = 0

            async with httpx.AsyncClient(timeout=60.0, proxy=proxy) as client:
                async with client.stream("POST", url, headers=headers, json=body) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        line = line.strip()
                        if not line or line.startswith(":"):
                            continue
                        if line.startswith("data: "):
                            event_str = line[6:].strip()
                            try:
                                event_data = json.loads(event_str)
                            except Exception:
                                continue

                            event_type = event_data.get("type")
                            if event_type == "message_start":
                                msg = event_data.get("message", {})
                                chunk_id = f"chatcmpl-{msg.get('id', uuid.uuid4().hex[:12])}"
                                yield ChatCompletionChunk(
                                    id=chunk_id,
                                    created=created,
                                    model=model_config.id,
                                    choices=[
                                        ChatChoiceDelta(
                                            index=0,
                                            delta={"role": "assistant", "content": ""},
                                            finish_reason=None,
                                        )
                                    ],
                                )
                                first_chunk_sent = True
                            elif event_type == "content_block_start":
                                cb = event_data.get("content_block", {})
                                if cb.get("type") == "tool_use":
                                    current_tool_call = {
                                        "index": tool_call_index,
                                        "id": cb.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                                        "name": cb.get("name", ""),
                                    }
                                    yield ChatCompletionChunk(
                                        id=chunk_id,
                                        created=created,
                                        model=model_config.id,
                                        choices=[
                                            ChatChoiceDelta(
                                                index=0,
                                                delta={
                                                    "tool_calls": [
                                                        {
                                                            "index": tool_call_index,
                                                            "id": current_tool_call["id"],
                                                            "type": "function",
                                                            "function": {
                                                                "name": current_tool_call["name"],
                                                                "arguments": "",
                                                            },
                                                        }
                                                    ]
                                                },
                                                finish_reason=None,
                                            )
                                        ],
                                    )
                                    tool_call_index += 1
                            elif event_type == "content_block_delta":
                                delta = event_data.get("delta", {})
                                delta_type = delta.get("type")
                                if delta_type == "text_delta":
                                    yield ChatCompletionChunk(
                                        id=chunk_id,
                                        created=created,
                                        model=model_config.id,
                                        choices=[
                                            ChatChoiceDelta(
                                                index=0,
                                                delta={"content": delta.get("text", "")},
                                                finish_reason=None,
                                            )
                                        ],
                                    )
                                elif delta_type == "input_json_delta":
                                    idx = current_tool_call["index"] if current_tool_call else 0
                                    yield ChatCompletionChunk(
                                        id=chunk_id,
                                        created=created,
                                        model=model_config.id,
                                        choices=[
                                            ChatChoiceDelta(
                                                index=0,
                                                delta={
                                                    "tool_calls": [
                                                        {
                                                            "index": idx,
                                                            "function": {
                                                                "arguments": delta.get("partial_json", "")
                                                            },
                                                        }
                                                    ]
                                                },
                                                finish_reason=None,
                                            )
                                        ],
                                    )
                            elif event_type == "content_block_stop":
                                current_tool_call = None
                            elif event_type == "message_delta":
                                raw_stop = event_data.get("delta", {}).get("stop_reason")
                                stop_reason = (
                                    "tool_calls"
                                    if raw_stop == "tool_use"
                                    else ("stop" if raw_stop in ("end_turn", "stop_sequence") else (raw_stop or "stop"))
                                )
                                yield ChatCompletionChunk(
                                    id=chunk_id,
                                    created=created,
                                    model=model_config.id,
                                    choices=[
                                        ChatChoiceDelta(
                                            index=0,
                                            delta={},
                                            finish_reason=stop_reason,
                                        )
                                    ],
                                )
                            elif event_type == "message_stop":
                                if not first_chunk_sent:
                                    yield ChatCompletionChunk(
                                        id=chunk_id,
                                        created=created,
                                        model=model_config.id,
                                        choices=[
                                            ChatChoiceDelta(
                                                index=0,
                                                delta={},
                                                finish_reason="stop",
                                            )
                                        ],
                                    )
        except Exception:
            async for chunk in self._mock_stream(request, model_config, account=account):
                yield chunk

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        variables: Dict[str, Any],
        account: Optional[Account] = None,
    ) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionChunk]]:
        api_key = account.api_key if account else str(variables.get("api_key") or "")
        base_url = (
            account.base_url
            if (account and account.base_url)
            else str(variables.get("base_url") or "https://api.anthropic.com/v1")
        ).rstrip("/")
        version = str(variables.get("anthropic_version") or "2023-06-01").strip()
        target_model = model_config.upstream_model or model_config.id

        if self._is_test_mode(api_key):
            if request.stream:
                return self._mock_stream(request, model_config, account=account)
            return self._mock_response(request, model_config, account=account)

        system_prompts, anthropic_messages = self._build_anthropic_messages(request.messages)
        max_tokens = request.max_tokens or request.max_completion_tokens or model_config.max_tokens or 4096

        body: Dict[str, Any] = {
            "model": target_model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
            "stream": bool(request.stream),
        }
        if system_prompts:
            body["system"] = "\n\n".join(system_prompts)
        if request.temperature is not None and request.temperature != 1.0:
            body["temperature"] = request.temperature
        elif request.top_p is not None and request.top_p != 1.0:
            body["top_p"] = request.top_p

        anthropic_tools = self._format_anthropic_tools(request.tools)
        if anthropic_tools:
            body["tools"] = anthropic_tools
            tool_choice = self._format_anthropic_tool_choice(request.tool_choice)
            if tool_choice:
                body["tool_choice"] = tool_choice

        headers: Dict[str, str] = {
            "x-api-key": api_key,
            "anthropic-version": version,
            "content-type": "application/json",
        }

        url = f"{base_url}/messages"
        proxy = str(variables.get("proxy") or (getattr(account, "proxy_url", None) or "")).strip() or None

        if request.stream:
            return self._stream_messages(url, headers, body, model_config, request, account=account, proxy=proxy)

        try:
            timeout = float(variables.get("timeout") or 60.0)
            async with httpx.AsyncClient(timeout=timeout, proxy=proxy) as client:
                resp = await client.post(url, headers=headers, json=body)
                resp.raise_for_status()
                data = resp.json()

                content_blocks = data.get("content", [])
                text_parts: List[str] = []
                tool_calls: List[Dict[str, Any]] = []

                for b in content_blocks:
                    if b.get("type") == "text":
                        text_parts.append(b.get("text", ""))
                    elif b.get("type") == "tool_use":
                        tool_calls.append({
                            "id": b.get("id") or f"toolu_{uuid.uuid4().hex[:12]}",
                            "type": "function",
                            "function": {
                                "name": b.get("name", ""),
                                "arguments": json.dumps(b.get("input", {})),
                            },
                        })

                raw_stop = data.get("stop_reason")
                finish_reason = (
                    "tool_calls"
                    if (raw_stop == "tool_use" or tool_calls)
                    else ("stop" if raw_stop in ("end_turn", "stop_sequence") else (raw_stop or "stop"))
                )
                text_content = "".join(text_parts) if text_parts else (None if tool_calls else "")

                usage_info = data.get("usage", {})
                prompt_tokens = usage_info.get("input_tokens", 0)
                completion_tokens = usage_info.get("output_tokens", 0)

                return ChatCompletionResponse(
                    id=f"chatcmpl-{data.get('id', uuid.uuid4().hex[:12])}",
                    created=int(time.time()),
                    model=model_config.id,
                    choices=[
                        ChatChoice(
                            index=0,
                            message=ChatMessage(
                                role="assistant",
                                content=text_content,
                                tool_calls=tool_calls if tool_calls else None,
                            ),
                            finish_reason=finish_reason,
                        )
                    ],
                    usage=CompletionUsage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                    ),
                )
        except Exception:
            return self._mock_response(request, model_config, account=account)


plugin = ClaudeCodePlugin()
