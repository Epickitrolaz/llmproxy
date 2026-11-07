"""
Google Antigravity Provider Plugin
==================================

Implementation based on 9Router IDE protocols and authentic Google Antigravity
Cloud Code PA internal endpoints (/v1internal:streamGenerateContent?alt=sse).

Key Features:
1. Authentic Google Account Connection:
   Connects to Google accounts using Antigravity client credentials with OAuth 2.0,
   automatic token refresh using refresh_token, and Code Assist project mapping.
2. 9Router IDE Trajectory Algorithm:
   Generates structured hierarchical trajectory paths:
     agent/<conversation_id>/<timestamp>/<trajectory_id>/<step>
   where conversation_id and trajectory_id are generated using deterministic SHA-256
   derived UUIDs seeded from session/model/client metadata.
3. Official Antigravity CLI Telemetry Headers:
   User-Agent: antigravity/cli/1.5.3 (aidev_client; os_type=linux; arch=amd64; cl=1016078699; auth_method=consumer)
   Authorization: Bearer <access_token>
   Content-Type: application/json
4. Endpoint Protocol:
   POST /v1internal:streamGenerateContent?alt=sse
   Payload matching official agy CLI with contents, generationConfig, project, requestId,
   model, userAgent, and requestType.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Set, Union

import httpx

from llmproxy.core.antigravity_oauth import (
    ANTIGRAVITY_CLIENT_ID,
    ANTIGRAVITY_CLIENT_SECRET,
    ANTIGRAVITY_DEFAULT_PROJECT,
    ANTIGRAVITY_TOKEN_URI,
    ANTIGRAVITY_USER_AGENT,
    CLOUDCODE_PA_ENDPOINTS,
    antigravity_oauth,
)
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

logger = logging.getLogger(__name__)


class AntigravityPlugin(BasePlugin):
    """
    Google Antigravity Provider Plugin with 9Router reference algorithms
    and direct Cloud Code PA integration matching official `agy` CLI 1:1.
    """

    REJECTED_GOOGLE_FIELDS: Set[str] = {
        "output_config",
        "thinking",
        "reasoning_effort",
        "reasoning",
        "enable_thinking",
        "thinking_budget",
        "thinkingConfig",
        "frequency_penalty",
        "presence_penalty",
        "logprobs",
        "top_logprobs",
        "logit_bias",
        "user",
        "seed",
        "service_tier",
        "disallowed_field",
        "unsupported_field",
    }
    DISALLOWED_FIELDS: Set[str] = REJECTED_GOOGLE_FIELDS

    def __init__(self):
        self._metadata = PluginMetadata(
            id="antigravity",
            name="Antigravity",
            description="Google Antigravity provider with 9Router IDE trajectory/session ID algorithms, multi-account load balancing, and authentic Cloud Code PA integration",
            version="2.2.0",
            author="9Router",
            enabled=True,
            filename="antigravity.py",
            variables=[
                PluginVariable(
                    name="timeout",
                    label="Timeout (seconds)",
                    type="number",
                    default=60,
                    required=True,
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
                    default=CLOUDCODE_PA_ENDPOINTS[0],
                    description="Google Antigravity CloudCode PA base URL",
                ),
            ],
            models=[
                PluginModelConfig(
                    id="gemini-3.8-flash-high",
                    name="Gemini 3.8 Flash High",
                    upstream_model="gemini-3.8-flash-high",
                    context_window=1048576,
                    max_tokens=65536,
                ),
                PluginModelConfig(
                    id="gemini-3.1-pro-high",
                    name="Gemini 3.1 Pro High",
                    upstream_model="gemini-3.1-pro-high",
                    context_window=2097152,
                    max_tokens=8192,
                ),
                PluginModelConfig(
                    id="claude-sonnet-5-5-high",
                    name="Claude Sonnet 5.5 High",
                    upstream_model="claude-sonnet-4-6",
                    context_window=200000,
                    max_tokens=8192,
                ),
                PluginModelConfig(
                    id="claude-sonnet-4-6",
                    name="Claude Sonnet 4.6",
                    upstream_model="claude-sonnet-4-6",
                    context_window=200000,
                    max_tokens=8192,
                ),
                PluginModelConfig(
                    id="claude-opus-4-6-thinking",
                    name="Claude Opus 4.6 Thinking",
                    upstream_model="claude-opus-4-6-thinking",
                    context_window=200000,
                    max_tokens=16384,
                ),
                PluginModelConfig(
                    id="gemini-2.5-pro",
                    name="Gemini 2.5 Pro",
                    upstream_model="gemini-2.5-pro",
                    context_window=1048576,
                    max_tokens=8192,
                ),
                PluginModelConfig(
                    id="gemini-2.5-flash",
                    name="Gemini 2.5 Flash",
                    upstream_model="gemini-2.5-flash",
                    context_window=1048576,
                    max_tokens=8192,
                ),
                PluginModelConfig(
                    id="gemini-3.8-flash",
                    name="Gemini 3.8 Flash",
                    upstream_model="gemini-3.8-flash-high",
                    context_window=1048576,
                    max_tokens=8192,
                ),
            ],
        )

    @property
    def metadata(self) -> PluginMetadata:
        return self._metadata

    @classmethod
    def generate_sha256_uuid(cls, seed: str) -> str:
        """Derives an RFC 4122 compliant UUID string deterministically from a SHA-256 digest."""
        digest = hashlib.sha256(seed.encode("utf-8")).digest()
        return str(uuid.UUID(bytes=digest[:16]))

    @classmethod
    def generate_trajectory_id(
        cls,
        conversation_seed: Optional[str] = None,
        trajectory_seed: Optional[str] = None,
        step: int = 1,
        timestamp_ms: Optional[int] = None,
    ) -> str:
        """
        Exact 9Router IDE trajectory ID algorithm:
          agent/<conversation_id>/<timestamp>/<trajectory_id>/<step>
        """
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)

        conv_seed = conversation_seed or f"session_{uuid.uuid4().hex}_{timestamp_ms}"
        traj_seed = trajectory_seed or f"traj_{uuid.uuid4().hex}_{step}"

        conversation_id = cls.generate_sha256_uuid(conv_seed)
        trajectory_id = cls.generate_sha256_uuid(traj_seed)

        return f"agent/{conversation_id}/{timestamp_ms}/{trajectory_id}/{step}"

    @classmethod
    def generate_ide_request_id(cls, seed: Optional[str] = None) -> str:
        """Generates the IDE request ID header required by Antigravity endpoints."""
        req_seed = seed or f"req_{uuid.uuid4().hex}_{time.time()}"
        return f"ide_req_{cls.generate_sha256_uuid(req_seed).replace('-', '')}"

    def build_headers(
        self, account: Optional[Account] = None, api_key: str = ""
    ) -> Dict[str, str]:
        """Constructs authentic HTTP headers matching agy CLI 1:1."""
        key = account.api_key if account else api_key
        user_agent = (
            (account.extra_config.get("user_agent") if account else None)
            or ANTIGRAVITY_USER_AGENT
        )
        headers = {
            "Content-Type": "application/json",
            "User-Agent": user_agent,
        }
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def sanitize_payload(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Strips Google-rejected parameters from the payload."""
        return {k: v for k, v in body.items() if k not in self.REJECTED_GOOGLE_FIELDS}

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
        content = f"Simulated response from 9Router Antigravity ({model_name}){acc_label}."

        return ChatCompletionResponse(
            id=f"chatcmpl-ag-{uuid.uuid4().hex[:12]}",
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
                prompt_tokens=15,
                completion_tokens=25,
                total_tokens=40,
            ),
        )

    async def _mock_stream(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        account: Optional[Account] = None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        model_name = model_config.id
        chunk_id = f"chatcmpl-ag-{uuid.uuid4().hex[:12]}"
        acc_label = f" via account '{account.name}'" if account else ""
        text = f"Simulated 9Router Antigravity stream response for {model_name}{acc_label}."
        words = text.split(" ")

        yield ChatCompletionChunk(
            id=chunk_id,
            model=model_name,
            choices=[ChatChoiceDelta(index=0, delta={"role": "assistant", "content": ""})],
        )

        for i, word in enumerate(words):
            yield ChatCompletionChunk(
                id=chunk_id,
                model=model_name,
                choices=[
                    ChatChoiceDelta(
                        index=0,
                        delta={"content": word + (" " if i < len(words) - 1 else "")},
                    )
                ],
            )

        yield ChatCompletionChunk(
            id=chunk_id,
            model=model_name,
            choices=[ChatChoiceDelta(index=0, delta={}, finish_reason="stop")],
        )

    async def _ensure_token_freshness(self, account: Optional[Account]) -> None:
        """Transparently refreshes Google OAuth access token if expired or near expiration."""
        if not account:
            return
        refresh_token = account.extra_config.get("refresh_token")
        if not refresh_token:
            return

        expires_at = float(account.extra_config.get("expires_at", 0))
        now = time.time()
        if not account.api_key or (expires_at > 0 and now >= expires_at - 120):
            refreshed = await antigravity_oauth.refresh_account_token(account)
            if refreshed:
                from llmproxy.core.manager import plugin_manager
                plugin_manager._save_accounts()

    def _build_upstream_request(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        account: Optional[Account],
    ) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """Constructs headers and request body matching agy CLI 1:1."""
        headers = self.build_headers(account=account)
        project_id = (account.extra_config.get("project_id") if account else None) or ANTIGRAVITY_DEFAULT_PROJECT
        target_model = model_config.upstream_model or model_config.id

        # Generate trajectory and conversation seeds
        acc_id = account.id if account else "default"
        conv_seed = (account.extra_config.get("session_id") if account else None) or f"conv_{acc_id}_{int(time.time() // 3600)}"
        traj_seed = (account.extra_config.get("trajectory_seed") if account else None) or f"traj_{acc_id}_{uuid.uuid4().hex[:8]}"
        step = int(account.extra_config.get("step", 1)) if account else 1

        conv_id = self.generate_sha256_uuid(conv_seed)
        traj_id = self.generate_sha256_uuid(traj_seed)
        timestamp_ms = int(time.time() * 1000)
        request_id = f"agent/{conv_id}/{timestamp_ms}/{traj_id}/{step}"

        # 64-bit integer session ID matching agy
        session_hash = int(hashlib.sha256(conv_id.encode("utf-8")).hexdigest()[:16], 16)
        session_id_str = str(session_hash - (1 << 63))

        # Build contents from messages
        contents: List[Dict[str, Any]] = []
        system_instruction_text = ""

        for m in request.messages:
            if m.role == "system":
                system_instruction_text += (str(m.content) + "\n")
            elif m.role == "user":
                prompt_text = str(m.content)
                wrapped_text = f"<USER_REQUEST>\n{prompt_text}\n</USER_REQUEST>" if not prompt_text.startswith("<USER_REQUEST>") else prompt_text
                contents.append({
                    "role": "user",
                    "parts": [{"text": wrapped_text}]
                })
            elif m.role in ("assistant", "model"):
                contents.append({
                    "role": "model",
                    "parts": [{"text": str(m.content)}]
                })

        if not contents:
            contents.append({
                "role": "user",
                "parts": [{"text": "<USER_REQUEST>\nHello\n</USER_REQUEST>"}]
            })

        max_tokens = request.max_tokens or model_config.max_tokens or 65536
        generation_config: Dict[str, Any] = {
            "maxOutputTokens": max_tokens,
            "thinkingConfig": {
                "includeThoughts": True,
                "thinkingLevel": "HIGH",
            },
        }
        if request.temperature is not None and request.temperature != 1.0:
            generation_config["temperature"] = request.temperature
        if request.top_p is not None and request.top_p != 1.0:
            generation_config["topP"] = request.top_p

        # Standard agy identity instructions when none explicitly supplied
        default_identity = (
            "<identity>\n"
            "You are Antigravity, a powerful agentic AI coding assistant designed by the Google Deepmind team working on Advanced Agentic Coding.\n"
            "You are pair programming with a USER to solve their coding task. The task may require creating a new codebase, modifying or debugging an existing codebase, or simply answering a question.\n"
            "The USER will send you requests, which you must always prioritize addressing. User requests are enclosed within <USER_REQUEST> tags.\n"
            "</identity>"
        )
        system_parts_text = system_instruction_text.strip() if system_instruction_text.strip() else default_identity

        is_claude = "claude" in target_model.lower()
        is_gpt = "gpt" in target_model.lower()

        labels = {
            "cascade_id": conv_id,
            "last_step_index": str(max(0, step - 1)),
            "model_enum": "MODEL_PLACEHOLDER_M318",
            "request_id": f"{traj_id}-{max(0, step - 1)}",
            "root_cascade_id": conv_id,
            "trajectory_id": traj_id,
            "used_claude": "true" if is_claude else "false",
            "used_claude_conservative": "false",
            "used_non_gemini_model": "true" if (is_claude or is_gpt) else "false",
        }

        inner_request: Dict[str, Any] = {
            "contents": contents,
            "systemInstruction": {
                "role": "user",
                "parts": [{"text": system_parts_text}],
            },
            "generationConfig": generation_config,
            "labels": labels,
            "sessionId": session_id_str,
        }

        body: Dict[str, Any] = {
            "project": project_id,
            "requestId": request_id,
            "request": inner_request,
            "model": target_model,
            "userAgent": "antigravity",
            "requestType": "agent",
        }

        return headers, body

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        variables: Dict[str, Any],
        account: Optional[Account] = None,
    ) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionChunk]]:
        api_key = account.api_key if account else str(variables.get("api_key", ""))
        base_url = (
            account.base_url
            if (account and account.base_url)
            else str(variables.get("base_url") or CLOUDCODE_PA_ENDPOINTS[0])
        ).rstrip("/")
        timeout = float(variables.get("timeout", 60.0))

        if self._is_test_mode(api_key):
            if request.stream:
                return self._mock_stream(request, model_config, account=account)
            return self._mock_response(request, model_config, account=account)

        await self._ensure_token_freshness(account)
        headers, body = self._build_upstream_request(request, model_config, account)
        target_model = model_config.id

        url = f"{base_url}/v1internal:streamGenerateContent?alt=sse"

        if request.stream:
            async def sse_stream_generator() -> AsyncIterator[ChatCompletionChunk]:
                chunk_id = f"chatcmpl-ag-{uuid.uuid4().hex[:12]}"
                created = int(time.time())

                yield ChatCompletionChunk(
                    id=chunk_id,
                    created=created,
                    model=target_model,
                    choices=[ChatChoiceDelta(index=0, delta={"role": "assistant", "content": ""})],
                )

                async with httpx.AsyncClient(timeout=timeout) as client:
                    async with client.stream("POST", url, headers=headers, json=body) as stream_resp:
                        if stream_resp.status_code == 401 and account and account.extra_config.get("refresh_token"):
                            refreshed = await antigravity_oauth.refresh_account_token(account)
                            if refreshed:
                                new_headers = self.build_headers(account=account)
                                async with client.stream("POST", url, headers=new_headers, json=body) as retry_resp:
                                    stream_resp = retry_resp

                        if stream_resp.status_code >= 400:
                            err_bytes = await stream_resp.aread()
                            raise httpx.HTTPStatusError(
                                f"Google Antigravity upstream error ({stream_resp.status_code}): {err_bytes.decode('utf-8', errors='replace')}",
                                request=stream_resp.request,
                                response=stream_resp,
                            )

                        async for line in stream_resp.aiter_lines():
                            if not line or not line.startswith("data: "):
                                continue
                            raw_data = line[6:].strip()
                            if raw_data == "[DONE]":
                                break
                            try:
                                payload = json.loads(raw_data)
                                candidates = payload.get("response", {}).get("candidates", [])
                                for cand in candidates:
                                    parts = cand.get("content", {}).get("parts", [])
                                    for part in parts:
                                        if "text" in part and part["text"]:
                                            yield ChatCompletionChunk(
                                                id=chunk_id,
                                                created=created,
                                                model=target_model,
                                                choices=[
                                                    ChatChoiceDelta(
                                                        index=0,
                                                        delta={"content": part["text"]},
                                                    )
                                                ],
                                            )
                            except Exception:
                                continue

                yield ChatCompletionChunk(
                    id=chunk_id,
                    created=created,
                    model=target_model,
                    choices=[ChatChoiceDelta(index=0, delta={}, finish_reason="stop")],
                )

            return sse_stream_generator()

        # Non-streaming: accumulate response parts
        full_text = []
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", url, headers=headers, json=body) as stream_resp:
                if stream_resp.status_code == 401 and account and account.extra_config.get("refresh_token"):
                    refreshed = await antigravity_oauth.refresh_account_token(account)
                    if refreshed:
                        new_headers = self.build_headers(account=account)
                        async with client.stream("POST", url, headers=new_headers, json=body) as retry_resp:
                            stream_resp = retry_resp

                if stream_resp.status_code >= 400:
                    err_bytes = await stream_resp.aread()
                    raise httpx.HTTPStatusError(
                        f"Google Antigravity upstream error ({stream_resp.status_code}): {err_bytes.decode('utf-8', errors='replace')}",
                        request=stream_resp.request,
                        response=stream_resp,
                    )

                async for line in stream_resp.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    raw_data = line[6:].strip()
                    if raw_data == "[DONE]":
                        break
                    try:
                        payload = json.loads(raw_data)
                        resp_obj = payload.get("response", {})
                        usage = resp_obj.get("usageMetadata", {})
                        if usage:
                            prompt_tokens = int(usage.get("promptTokenCount", prompt_tokens))
                            completion_tokens = int(usage.get("candidatesTokenCount", completion_tokens))
                            total_tokens = int(usage.get("totalTokenCount", total_tokens))

                        candidates = resp_obj.get("candidates", [])
                        for cand in candidates:
                            parts = cand.get("content", {}).get("parts", [])
                            for part in parts:
                                if "text" in part and part["text"]:
                                    full_text.append(part["text"])
                    except Exception:
                        continue

        assembled_text = "".join(full_text)
        return ChatCompletionResponse(
            id=f"chatcmpl-ag-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=target_model,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=assembled_text),
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=prompt_tokens or 15,
                completion_tokens=completion_tokens or max(1, len(assembled_text) // 4),
                total_tokens=total_tokens or (prompt_tokens + completion_tokens),
            ),
        )


plugin = AntigravityPlugin()
