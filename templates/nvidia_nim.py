"""NVIDIA NIM (build.nvidia.com) provider plugin for LLMProxy.

NVIDIA hosts an OpenAI-compatible Chat Completions API at
``https://integrate.api.nvidia.com/v1`` in front of 100+ open-weight models.
This plugin talks to that endpoint directly.

What the upstream exposes, and what this adapter does with it:

* **Usage** - ``usage.prompt_tokens`` / ``completion_tokens`` /
  ``total_tokens`` are returned on non-streaming calls and mapped into
  ``CompletionUsage``. ``prompt_tokens_details.cached_tokens`` and
  ``completion_tokens_details.reasoning_tokens`` are forwarded when present.
* **Streaming usage** - ``stream_options.include_usage`` is requested on every
  streamed call. Some NIM deployments reject that field, so a single retry
  without it is attempted on HTTP 400; the request still completes.
* **Reasoning** - OpenAI ``gpt-oss`` endpoints on NIM accept
  ``reasoning_effort`` (low/medium/high), which is advertised through
  ``reasoning_levels``. Other reasoning models on NIM toggle thinking through
  ``chat_template_kwargs`` (for example ``{"thinking": true}``); that is
  forwarded from ``extra_body`` rather than invented as a level.
  ``reasoning_content`` from the upstream is preserved for OpenAI clients.
* **Caching** - NIM has no prompt-cache API (no ``cache_control`` /
  ``cachedContent`` equivalent), so no cache hints are sent. Cache counters are
  still mapped if a future endpoint reports them.
* **Vision** - image parts are normalised with ``llmproxy.core.multimodal`` and
  remote HTTP(S) images are materialised into data URLs, which every NIM vision
  endpoint accepts.

Errors are raised with the upstream status code and a body excerpt so the
router's failover and cooldown logic keep working. No simulation happens for a
real account that failed.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import httpx

from llmproxy.core.interface import BasePlugin
from llmproxy.core.multimodal import (
    infer_vision_support,
    materialize_message_images,
    messages_have_images,
    normalize_message_content,
)
from llmproxy.schemas import (
    Account,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionUsage,
    PluginMetadata,
    PluginModelConfig,
    PluginVariable,
)

NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"

# ``openai/gpt-oss-*`` on NIM accepts these levels through ``reasoning_effort``.
GPT_OSS_REASONING_LEVELS = ("low", "medium", "high")

# Request fields a client may pass through ``extra_body``. NIM inherits much of
# the vLLM sampling surface plus ``chat_template_kwargs``/``nvext``.
PASSTHROUGH_KEYS = {
    "top_k",
    "min_p",
    "repetition_penalty",
    "seed",
    "chat_template_kwargs",
    "guided_json",
    "guided_regex",
    "guided_choice",
    "nvext",
    "min_tokens",
    "ignore_eos",
    "stop_token_ids",
    "logprobs",
    "top_logprobs",
}

EFFORT_ALIASES = {
    "minimal": "low",
    "xhigh": "high",
    "max": "high",
    "off": "none",
    "disabled": "none",
    "false": "none",
    "0": "none",
}


class NvidiaNimPlugin(BasePlugin):
    """NVIDIA NIM provider with a curated default catalog and failover support."""

    def __init__(self) -> None:
        self._metadata = PluginMetadata(
            id="nvidia_nim",
            name="Nvidia NIM",
            description=(
                "NVIDIA-hosted OpenAI-compatible inference for Nemotron, DeepSeek, "
                "gpt-oss, GLM, Kimi and other open-weight models"
            ),
            version="1.0.0",
            author="Community",
            enabled=True,
            filename="nvidia_nim.py",
            supports_usage_meter=False,
            variables=[
                PluginVariable(
                    name="base_url",
                    label="Base URL",
                    type="string",
                    default=NIM_BASE_URL,
                    description="NVIDIA NIM API base URL (no trailing slash)",
                ),
                PluginVariable(
                    name="timeout",
                    label="Timeout (seconds)",
                    type="number",
                    default=120,
                    description="Request timeout in seconds",
                ),
                PluginVariable(
                    name="load_balancing_strategy",
                    label="Load Balancing Strategy",
                    type="select",
                    default="fallback",
                    options=["fallback", "round_robin", "least_used", "random"],
                    description="Strategy for distributing requests across connected accounts",
                ),
                PluginVariable(
                    name="account_switch_cooldown",
                    label="Account Cooldown (s)",
                    type="number",
                    default=60,
                    description=(
                        "Cooldown after a 429. The free tier is capped at 40 RPM "
                        "per model, so one minute is a sensible default"
                    ),
                ),
                PluginVariable(
                    name="account_switch_wait",
                    label="Switch Wait Delay (s)",
                    type="number",
                    default=0.1,
                    description="Wait delay before switching accounts on failover",
                ),
                PluginVariable(
                    name="max_retries",
                    label="Max Failover Retries",
                    type="number",
                    default=2,
                    description="Maximum number of account failovers per request",
                ),
                PluginVariable(
                    name="reasoning_effort",
                    label="Default Reasoning Effort",
                    type="select",
                    default="default",
                    options=["default", "low", "medium", "high"],
                    description=(
                        "Applied only to models that advertise reasoning levels. "
                        "A request-level reasoning_effort always wins."
                    ),
                ),
                PluginVariable(
                    name="extra_body",
                    label="Extra Body (JSON)",
                    type="string",
                    default="{}",
                    description=(
                        'Merged into every request, e.g. {"chat_template_kwargs": '
                        '{"thinking": true}} to enable thinking on DeepSeek/GLM'
                    ),
                ),
                PluginVariable(
                    name="vision_max_image_mb",
                    label="Max Image Size (MB)",
                    type="number",
                    default=20,
                    description="Remote images larger than this are replaced by a text placeholder",
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
                    description="Strategy for selecting proxies from the pool",
                ),
                PluginVariable(
                    name="specific_proxy_id",
                    label="Specific Proxy ID / Tag",
                    type="string",
                    default="",
                    description="Target proxy ID, URL, or tag when the strategy is 'specific'",
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
                    id="nim/nvidia/nemotron-3-ultra-550b-a55b",
                    name="Nemotron 3 Ultra 550B",
                    upstream_model="nvidia/nemotron-3-ultra-550b-a55b",
                    context_window=1_000_000,
                    max_tokens=65_536,
                ),
                PluginModelConfig(
                    id="nim/nvidia/nemotron-3.5-lightning",
                    name="Nemotron 3.5 Lightning",
                    upstream_model="nvidia/nemotron-3.5-lightning",
                    context_window=1_000_000,
                    max_tokens=32_768,
                ),
                PluginModelConfig(
                    id="nim/nvidia/nemotron-3.5-super-vl-120b-a12b",
                    name="Nemotron 3.5 Super VL 120B",
                    upstream_model="nvidia/nemotron-3.5-super-vl-120b-a12b",
                    context_window=1_000_000,
                    max_tokens=65_536,
                    supports_vision=True,
                ),
                PluginModelConfig(
                    id="nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
                    name="Nemotron 3 Nano Omni 30B",
                    upstream_model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
                    context_window=262_144,
                    max_tokens=32_768,
                    supports_vision=True,
                ),
                PluginModelConfig(
                    id="nim/openai/gpt-oss-120b",
                    name="gpt-oss 120B",
                    upstream_model="openai/gpt-oss-120b",
                    context_window=131_072,
                    max_tokens=32_768,
                    reasoning_levels=list(GPT_OSS_REASONING_LEVELS),
                ),
                PluginModelConfig(
                    id="nim/openai/gpt-oss-20b",
                    name="gpt-oss 20B",
                    upstream_model="openai/gpt-oss-20b",
                    context_window=131_072,
                    max_tokens=32_768,
                    reasoning_levels=list(GPT_OSS_REASONING_LEVELS),
                ),
                PluginModelConfig(
                    id="nim/deepseek-ai/deepseek-v4.1-flash",
                    name="DeepSeek V4.1 Flash",
                    upstream_model="deepseek-ai/deepseek-v4.1-flash",
                    context_window=1_000_000,
                    max_tokens=32_768,
                ),
                PluginModelConfig(
                    id="nim/z-ai/glm-5.3-flash",
                    name="GLM 5.3 Flash",
                    upstream_model="z-ai/glm-5.3-flash",
                    context_window=1_000_000,
                    max_tokens=32_768,
                ),
                PluginModelConfig(
                    id="nim/moonshotai/kimi-k2.6",
                    name="Kimi K2.6",
                    upstream_model="moonshotai/kimi-k2.6",
                    context_window=262_144,
                    max_tokens=32_768,
                ),
                PluginModelConfig(
                    id="nim/meta/llama-3.2-90b-vision-instruct",
                    name="Llama 3.2 90B Vision",
                    upstream_model="meta/llama-3.2-90b-vision-instruct",
                    context_window=131_072,
                    max_tokens=4_096,
                    supports_vision=True,
                ),
            ],
        )

        for model in self._metadata.models:
            if not model.supports_vision:
                model.supports_vision = infer_vision_support(model.id, model.name)

    @property
    def metadata(self) -> PluginMetadata:
        return self._metadata

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _is_test_mode(api_key: str, account: Optional[Account]) -> bool:
        key = (api_key or "").strip().lower()
        if not key:
            if account is not None and not str(account.id).startswith("default-"):
                raise ValueError(
                    f"Account '{account.name}' has no API key configured for NVIDIA NIM."
                )
            return True
        prefixes = ("test", "mock", "sk-test", "dummy", "fake", "offline", "sample")
        return any(key.startswith(prefix) for prefix in prefixes) or key in (
            "test-key",
            "test_key",
            "secret123",
        )

    @staticmethod
    def _usage_from_payload(usage: Any) -> Optional[CompletionUsage]:
        """Map a NIM ``usage`` object onto ``CompletionUsage``."""
        if not isinstance(usage, dict):
            return None

        def as_int(value: Any) -> int:
            try:
                return max(0, int(value))
            except (TypeError, ValueError):
                return 0

        prompt = as_int(usage.get("prompt_tokens", usage.get("input_tokens", 0)))
        completion = as_int(
            usage.get("completion_tokens", usage.get("output_tokens", 0))
        )
        total = as_int(usage.get("total_tokens", prompt + completion))

        details = usage.get("prompt_tokens_details")
        completion_details = usage.get("completion_tokens_details")

        extra: Dict[str, Any] = {}
        if isinstance(completion_details, dict):
            extra["completion_tokens_details"] = completion_details

        return CompletionUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            prompt_tokens_details=details if isinstance(details, dict) else None,
            cache_read_input_tokens=usage.get("cache_read_input_tokens"),
            cache_creation_input_tokens=usage.get("cache_creation_input_tokens"),
            **extra,
        )

    @staticmethod
    def _upstream_error(response: httpx.Response, body_text: str) -> httpx.HTTPStatusError:
        excerpt = " ".join((body_text or "").split())[:400]
        message = f"NVIDIA NIM {response.status_code}: {excerpt or response.reason_phrase}"
        return httpx.HTTPStatusError(message, request=response.request, response=response)

    @staticmethod
    def _parse_json_object(raw: Any) -> Dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if not isinstance(raw, str) or not raw.strip():
            return {}
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _resolve_effort(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        variables: Dict[str, Any],
    ) -> Optional[str]:
        """Return the ``reasoning_effort`` value to forward, if any."""
        extra = request.extra_body if isinstance(request.extra_body, dict) else {}
        supported = [level for level in (model_config.reasoning_levels or [])]

        raw: Any = request.reasoning_effort
        if raw is None:
            raw = extra.get("reasoning_effort")
        if raw is None:
            reasoning = extra.get("reasoning")
            if isinstance(reasoning, dict):
                raw = reasoning.get("effort")
            elif isinstance(reasoning, str):
                raw = reasoning

        explicit = raw is not None
        if raw is None:
            if not supported:
                return None
            raw = variables.get("reasoning_effort")

        level = str(raw or "").strip().lower()
        if not level or level in ("default", "auto"):
            return None
        level = EFFORT_ALIASES.get(level, level)
        if level == "none":
            # gpt-oss has no "none" level; omitting the field is the correct way
            # to ask for a non-reasoning answer.
            return None
        if supported and level not in supported:
            if not explicit:
                return None
        return level

    def _apply_extra_body(
        self,
        body: Dict[str, Any],
        request: ChatCompletionRequest,
        variables: Dict[str, Any],
    ) -> None:
        merged: Dict[str, Any] = {}
        merged.update(self._parse_json_object(variables.get("extra_body")))
        if isinstance(request.extra_body, dict):
            merged.update(request.extra_body)

        for key, value in merged.items():
            if key in PASSTHROUGH_KEYS:
                body[key] = value

    def _build_body(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        variables: Dict[str, Any],
        messages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build the NIM Chat Completions body (also exercised by tests)."""
        target_model = model_config.upstream_model or model_config.id

        body: Dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "stream": bool(request.stream),
        }

        # Per-model parameter defaults from the dashboard, never overriding the
        # wire-critical keys.
        for key, value in (model_config.parameters or {}).items():
            if key in ("model", "messages", "stream"):
                continue
            body.setdefault(key, value)

        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.top_p is not None:
            body["top_p"] = request.top_p
        if request.stop is not None:
            body["stop"] = request.stop
        if request.tools is not None:
            body["tools"] = request.tools
        if request.tool_choice is not None:
            body["tool_choice"] = request.tool_choice
        if request.parallel_tool_calls is not None:
            body["parallel_tool_calls"] = request.parallel_tool_calls
        if request.response_format is not None:
            body["response_format"] = request.response_format
        if request.user is not None:
            body["user"] = request.user
        if request.n is not None and request.n > 1:
            body["n"] = request.n

        max_tokens = (
            request.max_tokens
            if request.max_tokens is not None
            else request.max_completion_tokens
        )
        if max_tokens is not None:
            body["max_tokens"] = max_tokens

        effort = self._resolve_effort(request, model_config, variables)
        if effort:
            body["reasoning_effort"] = effort

        self._apply_extra_body(body, request, variables)

        if request.stream:
            # Ask for a terminal usage chunk so cache/reasoning counters reach
            # the request log even when the client did not request usage.
            stream_options = dict(request.stream_options or {})
            stream_options["include_usage"] = True
            body["stream_options"] = stream_options

        return body

    # ------------------------------------------------------------- simulation

    def _mock_response(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        account: Optional[Account] = None,
    ) -> ChatCompletionResponse:
        label = f" via account '{account.name}'" if account else ""
        content = (
            f"Simulated response from NVIDIA NIM ({model_config.id}){label}. "
            "Add an nvapi- key in the Accounts tab to reach build.nvidia.com."
        )
        if any(message.role == "tool" for message in request.messages):
            content = (
                f"Simulated response from NVIDIA NIM ({model_config.id}){label} "
                "with tool output processed."
            )

        return ChatCompletionResponse(
            id=f"chatcmpl-nim-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=model_config.id,
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            usage=CompletionUsage(
                prompt_tokens=12, completion_tokens=22, total_tokens=34
            ),
        )

    async def _mock_stream(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        account: Optional[Account] = None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        chunk_id = f"chatcmpl-nim-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        label = f" via account '{account.name}'" if account else ""

        yield ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=model_config.id,
            choices=[{"index": 0, "delta": {"role": "assistant", "content": ""}}],
        )

        for token in ["Simulated ", "stream ", "from ", "NVIDIA NIM", label, "."]:
            yield ChatCompletionChunk(
                id=chunk_id,
                created=created,
                model=model_config.id,
                choices=[{"index": 0, "delta": {"content": token}}],
            )

        yield ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=model_config.id,
            choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}],
            usage=CompletionUsage(
                prompt_tokens=12, completion_tokens=22, total_tokens=34
            ),
        )

    # ----------------------------------------------------------------- stream

    async def _iter_sse(
        self,
        response: httpx.Response,
        model_config: PluginModelConfig,
    ) -> AsyncIterator[ChatCompletionChunk]:
        chunk_id = f"chatcmpl-nim-{uuid.uuid4().hex[:12]}"
        first_content_chunk = True

        async for line in response.aiter_lines():
            line = line.strip()
            if not line or line.startswith(":"):
                continue
            if not line.startswith("data:"):
                continue

            data = line[5:].strip()
            if data == "[DONE]":
                break

            try:
                parsed = json.loads(data)
            except ValueError:
                continue
            if not isinstance(parsed, dict):
                continue

            if parsed.get("error"):
                error = parsed["error"]
                message = (
                    error.get("message") if isinstance(error, dict) else str(error)
                )
                raise RuntimeError(message or "NVIDIA NIM streaming error")

            usage = self._usage_from_payload(parsed.pop("usage", None))

            parsed.setdefault("id", chunk_id)
            parsed.setdefault("created", int(time.time()))
            parsed.setdefault("object", "chat.completion.chunk")
            parsed["model"] = model_config.id
            if not isinstance(parsed.get("choices"), list):
                parsed["choices"] = []

            if first_content_chunk and parsed["choices"]:
                choice = parsed["choices"][0]
                delta = choice.get("delta") if isinstance(choice, dict) else None
                if isinstance(delta, dict) and not delta.get("role"):
                    delta["role"] = "assistant"
                first_content_chunk = False

            try:
                chunk = ChatCompletionChunk.model_validate(parsed)
            except Exception:
                continue
            if usage is not None:
                chunk.usage = usage
            yield chunk

    async def _stream_request(
        self,
        url: str,
        headers: Dict[str, str],
        body: Dict[str, Any],
        model_config: PluginModelConfig,
        account: Optional[Account] = None,
        proxy: Optional[str] = None,
        timeout: float = 120.0,
    ) -> AsyncIterator[ChatCompletionChunk]:
        payload = dict(body)

        async with httpx.AsyncClient(timeout=timeout, proxy=proxy) as client:
            # Some NIM deployments reject ``stream_options``; retry once without
            # it rather than failing the request.
            for attempt in (0, 1):
                request = client.build_request(
                    "POST", url, headers=headers, json=payload
                )
                response = await client.send(request, stream=True)

                if response.status_code >= 400:
                    text = (await response.aread()).decode("utf-8", "replace")
                    await response.aclose()
                    if (
                        attempt == 0
                        and "stream_options" in payload
                        and "stream_option" in text.lower()
                    ):
                        payload.pop("stream_options", None)
                        continue
                    raise self._upstream_error(response, text)

                try:
                    async for chunk in self._iter_sse(response, model_config):
                        yield chunk
                finally:
                    await response.aclose()
                return

    # ------------------------------------------------------------ entry point

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        variables: Dict[str, Any],
        account: Optional[Account] = None,
    ) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionChunk]]:
        api_key = (account.api_key if account else "") or ""
        base_url = (
            account.base_url
            if (account and account.base_url)
            else str(variables.get("base_url") or NIM_BASE_URL)
        ).rstrip("/")

        if self._is_test_mode(api_key, account):
            if request.stream:
                return self._mock_stream(request, model_config, account=account)
            return self._mock_response(request, model_config, account=account)

        proxy = (
            str(variables.get("proxy") or (getattr(account, "proxy_url", None) or "")).strip()
            or None
        )
        timeout = float(variables.get("timeout") or 120.0)

        messages: List[Dict[str, Any]] = []
        for message in request.messages:
            payload = message.model_dump(exclude_none=True)
            normalize_message_content(payload)
            messages.append(payload)

        if messages_have_images(messages):
            messages = await materialize_message_images(
                messages,
                proxy=proxy,
                max_bytes=int(float(variables.get("vision_max_image_mb") or 20))
                * 1024
                * 1024,
            )

        body = self._build_body(request, model_config, variables, messages)

        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Accept": "text/event-stream" if request.stream else "application/json",
        }

        url = f"{base_url}/chat/completions"

        if request.stream:
            return self._stream_request(
                url,
                headers,
                body,
                model_config,
                account=account,
                proxy=proxy,
                timeout=timeout,
            )

        async with httpx.AsyncClient(timeout=timeout, proxy=proxy) as client:
            response = await client.post(url, headers=headers, json=body)
            if response.status_code >= 400:
                raise self._upstream_error(response, response.text)
            data = response.json()

        if not isinstance(data, dict) or not isinstance(data.get("choices"), list):
            raise RuntimeError(
                "NVIDIA NIM returned an unexpected payload without 'choices'"
            )

        usage = self._usage_from_payload(data.pop("usage", None))
        data.setdefault("id", f"chatcmpl-nim-{uuid.uuid4().hex[:12]}")
        data.setdefault("created", int(time.time()))
        data["model"] = model_config.id
        data["usage"] = usage.model_dump(exclude_none=True) if usage else None

        return ChatCompletionResponse.model_validate(data)

plugin = NvidiaNimPlugin()
