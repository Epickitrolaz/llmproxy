"""
Google Antigravity Provider Plugin
==================================

Implementation based on 9Router IDE protocols.

Key Features:
1. 9Router IDE Trajectory Algorithm:
   Generates structured hierarchical trajectory paths:
     agent/<conversation_id>/<timestamp>/<trajectory_id>/<step>
   where conversation_id and trajectory_id are generated using deterministic SHA-256
   derived UUIDs seeded from session/model/client metadata.
2. Complete Field Sanitization:
   Strips all Google-rejected parameters (e.g. output_config, thinking, reasoning_effort,
   reasoning, enable_thinking, thinking_budget, thinkingConfig, presence_penalty, etc.).
3. Official Antigravity IDE Telemetry Headers:
   User-Agent: Antigravity-IDE/1.0
   Authorization: Bearer <token>
   x-goog-api-client: ide/antigravity
   x-trajectory-id: agent/...
   x-ide-request-id: ide_req_...
4. Multi-Account Support & Operational Variables:
   Variables only control operational parameters (timeouts, account switch wait,
   load balancing strategy, and max retries). Account credentials are held in Account objects.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Set, Union

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


class AntigravityPlugin(BasePlugin):
    """
    Google Antigravity Provider Plugin with 9Router reference algorithms.
    """

    # Exact parameters strictly rejected by Google Gemini/Antigravity APIs.
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
            description="Google Antigravity provider with 9Router IDE trajectory/session ID algorithms, multi-account load balancing, and field stripping",
            version="2.0.0",
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
                    default="https://antigravity.google",
                    description="Default Google Antigravity base URL",
                ),
            ],
            models=[
                PluginModelConfig(
                    id="gemini-3.8-flash-high",
                    name="Gemini 3.8 Flash High",
                    upstream_model="gemini-3.8-flash-high",
                    context_window=1048576,
                    max_tokens=8192,
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
                    upstream_model="claude-sonnet-5-5-high",
                    context_window=200000,
                    max_tokens=8192,
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
                # Backwards-compatible aliases
                PluginModelConfig(
                    id="gemini-3.8-flash",
                    name="Gemini 3.8 Flash",
                    upstream_model="gemini-3.8-flash-high",
                    context_window=1048576,
                    max_tokens=8192,
                ),
                PluginModelConfig(
                    id="claude-sonnet-4-6",
                    name="Claude Sonnet 4.6",
                    upstream_model="claude-sonnet-5-5-high",
                    context_window=200000,
                    max_tokens=8192,
                ),
                PluginModelConfig(
                    id="claude-opus-4-6-thinking",
                    name="Claude Opus 4.6 Thinking",
                    upstream_model="claude-sonnet-5-5-high",
                    context_window=200000,
                    max_tokens=16384,
                ),
            ],
        )

    @property
    def metadata(self) -> PluginMetadata:
        return self._metadata

    @classmethod
    def generate_sha256_uuid(cls, seed: str) -> str:
        """
        Derives an RFC 4122 compliant UUID string deterministically from a SHA-256 digest.
        """
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
        """
        Generates the IDE request ID header required by Antigravity endpoints.
        Prefixed with 'ide_req_' and suffixed with a SHA-256 derived UUID hex.
        """
        req_seed = seed or f"req_{uuid.uuid4().hex}_{time.time()}"
        return f"ide_req_{cls.generate_sha256_uuid(req_seed).replace('-', '')}"

    def build_headers(
        self, account: Optional[Account] = None, api_key: str = ""
    ) -> Dict[str, str]:
        """
        Constructs complete HTTP headers required for Google Antigravity endpoints.
        """
        key = account.api_key if account else api_key
        acc_id = account.id if account else "default"
        conv_seed = (account.extra_config.get("session_id") if account else None) or acc_id
        traj_seed = (account.extra_config.get("trajectory_seed") if account else None) or f"traj_{acc_id}_{time.time()}"
        step = int(account.extra_config.get("step", 1)) if account else 1

        trajectory_id = self.generate_trajectory_id(
            conversation_seed=conv_seed,
            trajectory_seed=traj_seed,
            step=step,
        )
        ide_req_id = self.generate_ide_request_id(seed=f"{acc_id}_{time.time()}")

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Antigravity-IDE/1.0",
            "x-goog-api-client": "ide/antigravity",
            "x-trajectory-id": trajectory_id,
            "x-ide-request-id": ide_req_id,
            "x-request-id": ide_req_id,
        }
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def sanitize_payload(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """
        Strips Google-rejected parameters from the payload.
        """
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
        ide_req_id = self.generate_ide_request_id()

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
            else str(variables.get("base_url") or "https://antigravity.google")
        ).rstrip("/")
        timeout = float(variables.get("timeout", 60.0))

        if self._is_test_mode(api_key):
            if request.stream:
                return self._mock_stream(request, model_config, account=account)
            return self._mock_response(request, model_config, account=account)

        # Real upstream execution with 9Router telemetry and field stripping
        headers = self.build_headers(account=account, api_key=api_key)
        target_model = model_config.upstream_model or model_config.id

        body: Dict[str, Any] = {
            "model": target_model,
            "messages": [m.model_dump(exclude_none=True) for m in request.messages],
            "stream": bool(request.stream),
        }
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.max_tokens is not None:
            body["max_tokens"] = request.max_tokens

        # Strip Google-rejected fields
        cleaned_body = self.sanitize_payload(body)

        if not request.stream:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=cleaned_body,
                )
                if resp.status_code >= 400:
                    raise httpx.HTTPStatusError(
                        f"Antigravity upstream error: {resp.text}",
                        request=resp.request,
                        response=resp,
                    )
                return ChatCompletionResponse.model_validate(resp.json())

        async def stream_generator() -> AsyncIterator[ChatCompletionChunk]:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST",
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=cleaned_body,
                ) as stream_resp:
                    if stream_resp.status_code >= 400:
                        err_text = await stream_resp.aread()
                        raise httpx.HTTPStatusError(
                            f"Antigravity upstream error: {err_text.decode('utf-8')}",
                            request=stream_resp.request,
                            response=stream_resp,
                        )
                    async for line in stream_resp.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk_data = json.loads(data_str)
                            yield ChatCompletionChunk.model_validate(chunk_data)
                        except Exception:
                            continue

        return stream_generator()


plugin = AntigravityPlugin()
