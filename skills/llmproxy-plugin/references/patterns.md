# LLMProxy Plugin Patterns

Use these patterns when implementing a new provider. The working skeleton in
`assets/plugin_skeleton.py` combines most of them.

## 1. Generic OpenAI-compatible upstream

This is the most common case. The upstream accepts
`POST {base_url}/chat/completions` with an OpenAI-style body.

```python
async def chat_completion(self, request, model_config, variables, account=None):
    api_key = account.api_key if account else str(variables.get("api_key") or "")
    base_url = (
        account.base_url if account and account.base_url else str(variables["base_url"])
    ).rstrip("/")
    url = base_url if base_url.endswith("/chat/completions") else f"{base_url}/chat/completions"

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model_config.upstream_model or model_config.id,
        "messages": [m.model_dump(exclude_none=True) for m in request.messages],
        "stream": bool(request.stream),
    }
    # Add temperature, max_tokens, tools, tool_choice, response_format, stop, etc.

    proxy = str(variables.get("proxy") or (account.proxy_url if account else "") or "").strip() or None
    if request.stream:
        return self._stream(url, headers, payload, proxy)
    return await self._complete(url, headers, payload, proxy)
```

## 2. Streaming SSE parser

```python
async def _stream(self, url, headers, payload, proxy):
    async with httpx.AsyncClient(timeout=60.0, proxy=proxy) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as response:
            response.raise_for_status()
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
                    yield ChatCompletionChunk.model_validate(json.loads(data))
                except Exception:
                    continue
```

If the upstream sends a different event format, translate it into
`ChatCompletionChunk` objects. Do not yield raw dicts.

## 3. Tool calls

Forward the request fields:

```python
if request.tools is not None:
    payload["tools"] = request.tools
if request.tool_choice is not None:
    payload["tool_choice"] = request.tool_choice
if request.parallel_tool_calls is not None:
    payload["parallel_tool_calls"] = request.parallel_tool_calls
```

Non-streaming:

```python
ChatChoice(
    index=0,
    message=ChatMessage(
        role="assistant",
        content=None,
        tool_calls=[
            {
                "id": "call_...",
                "type": "function",
                "function": {"name": "lookup", "arguments": "{\"id\": 1}"},
            }
        ],
    ),
    finish_reason="tool_calls",
)
```

Streaming:

```python
# First tool chunk: role + tool_calls with id/name and empty arguments.
# Following chunks: delta["tool_calls"] with the same index and argument pieces.
# Final chunk: delta={} and finish_reason="tool_calls".
```

When the conversation already contains a `tool`/`function` message, send the
tool output to the upstream instead of creating another tool call.

## 4. Accounts and load balancing

The router selects the account. The plugin only consumes it.

```python
api_key = account.api_key if account else str(variables.get("api_key") or "")
base_url = account.base_url if account and account.base_url else variables.get("base_url")
extra = account.extra_config if account else {}
```

- Do not select accounts inside the plugin.
- Do not mark accounts as failed yourself; raise an exception and let the router
  call `plugin_manager.mark_account_result`.
- For token refresh, update `account.api_key`/`extra_config` and call
  `plugin_manager._save_accounts()` (existing OAuth templates do this).

## 5. OAuth token refresh

```python
async def _ensure_token_freshness(self, account):
    if not account or not account.extra_config.get("refresh_token"):
        return
    expires_at = float(account.extra_config.get("expires_at", 0) or 0)
    if account.api_key and (expires_at <= 0 or time.time() < expires_at - 120):
        return
    refreshed = await my_oauth.refresh_account_token(account)
    if refreshed:
        from llmproxy.core.manager import plugin_manager
        plugin_manager._save_accounts()
```

Also retry once on HTTP 401 if a refresh token is available.

## 6. External proxy support

The router injects the resolved proxy into `variables["proxy"]` before calling the
plugin. Use it for every upstream request:

```python
proxy = str(variables.get("proxy") or (account.proxy_url if account else "") or "").strip() or None
async with httpx.AsyncClient(timeout=timeout, proxy=proxy) as client:
    ...
```

Declare the standard proxy variables in metadata so the dashboard exposes the
provider-level proxy controls.

## 7. Simulation mode

Keep the provider usable without credentials:

```python
def _is_test_mode(self, api_key: str) -> bool:
    key = (api_key or "").lower().strip()
    return not key or key.startswith(("test", "mock", "dummy", "fake", "offline"))

if self._is_test_mode(api_key):
    if request.stream:
        return self._mock_stream(request, model_config, account)
    return self._mock_response(request, model_config, account)
```

The mock response must be a valid `ChatCompletionResponse`; the mock stream must
yield valid `ChatCompletionChunk` objects.

## 8. Usage extraction

- OpenAI-compatible: `response.usage` is already a `CompletionUsage` after
  `ChatCompletionResponse.model_validate`.
- Streaming: the final upstream chunk often includes `usage`; forward it on the
  final `ChatCompletionChunk`.
- Anthropic: map `usage.input_tokens`/`output_tokens` to
  `prompt_tokens`/`completion_tokens`.
- Gemini: map `usageMetadata.promptTokenCount`/`candidatesTokenCount`.

The router estimates if usage is missing.

## 9. Rate-limit and error propagation

```python
response = await client.post(url, headers=headers, json=payload)
if response.status_code >= 400:
    body = response.text[:500]
    raise RuntimeError(f"MyProvider upstream error ({response.status_code}): {body}")
```

The router detects `429`, `rate limit`, `quota`, and `too many requests` in the
exception message and applies the account cooldown. Do not catch these errors and
return a fake response.

## 10. Multimodal content

`ChatMessage.content` can be a string or a list of parts. For OpenAI-compatible
upstreams, forwarding `message.model_dump(exclude_none=True)` is usually enough.
For providers with different formats, translate explicitly:

```python
for message in request.messages:
    if isinstance(message.content, list):
        for part in message.content:
            if part.get("type") == "text":
                ...
            elif part.get("type") == "image_url":
                ...
```

## 11. Reasoning / model suffixes

If clients send variants such as `my-model:high` or `my-model-high`:

```python
model = request.model
effort = None
for suffix in (":high", "-high", ":medium", "-medium", ":low", "-low", ":none", "-none"):
    if model.lower().endswith(suffix):
        effort = suffix.lstrip(":-")
        model = model[: -len(suffix)]
        break
upstream_model = model_config.upstream_model or model
```

Strip suffixes before sending the model to the upstream. Map the effort to the
provider's native parameter or model variant.

## 12. Responses API / Anthropic / Gemini translation

If the upstream is not OpenAI Chat Completions:

- Build the provider's native payload from `request.messages`, `tools`, and
  sampling fields.
- Translate the provider's streaming events into `ChatCompletionChunk`.
- Translate the final provider response into `ChatCompletionResponse`.
- Preserve tool calls and usage.
- See `templates/codex.py` (Responses API), `templates/antigravity.py` (Google
  Cloud Code PA), and `templates/claude_code.py` (Anthropic Messages).

## 13. Shared HTTP clients

A new `httpx.AsyncClient` per request is acceptable and is what most templates do.
If you introduce a shared client, make it lazy and close it on shutdown. Do not
create clients or perform I/O at import time. The current `main` branch does not
ship a shared HTTP pool module, so prefer the per-request pattern unless you are
also adding the pool infrastructure.

## 14. Timeouts and retries

- Use `variables["timeout"]` with a safe default (30–60s).
- Do not retry indefinitely.
- The router handles account failover; plugin-internal retries should be limited
  to transient network errors or provider-specific token refreshes.
- Use exponential backoff only inside a bounded loop.

## 15. Logging

```python
import logging
logger = logging.getLogger(__name__)
```

Log status codes and short messages, never tokens, cookies, or full response
bodies. Do not log the entire request payload.
