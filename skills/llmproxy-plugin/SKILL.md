---
name: llmproxy-plugin
description: Build, review, or repair LLMProxy provider plugins. Use when adding a new provider, models, variables, account pools, streaming, tool calling, usage reporting, proxy support, or a built-in provider template for this repository.
---

# LLMProxy Provider Plugin Skill

Use this skill when the task is to add or fix a provider plugin for LLMProxy.

LLMProxy is a FastAPI OpenAI-compatible router. Provider plugins are standalone
Python files that are loaded at runtime. A good plugin behaves like a native
provider: it validates configuration, supports streaming and non-streaming
requests, handles accounts, reports usage when possible, and fails cleanly.

## Read these first

1. `llmproxy/core/interface.py` — the `BasePlugin` contract.
2. `llmproxy/schemas.py` — `PluginMetadata`, `PluginVariable`, `PluginModelConfig`,
   `Account`, `ChatCompletionRequest`, `ChatCompletionResponse`, `ChatCompletionChunk`.
3. The closest existing template:
   - `templates/openai_compatible.py` — generic OpenAI-compatible upstream.
   - `templates/mock_provider.py` — offline provider with no upstream.
   - `templates/codex.py` — ChatGPT/Codex OAuth + Responses wire protocol.
   - `templates/antigravity.py` — Google Cloud Code PA streaming protocol.
   - `templates/anthropic_proxy.py` — Anthropic Messages protocol.
   - `templates/openrouter.py` — OpenRouter gateway with free-model defaults.
   - `templates/opencode_free.py` — anonymous upstream with session rotation.
4. `llmproxy/api/router.py` — how the router selects accounts, resolves proxies,
   calls the plugin, streams chunks, records usage, and detects rate limits.

## Choose the target file

- **Built-in template** → `templates/<provider_id>.py`.
  - Appears in the dashboard's "Add from Template" gallery.
  - Tracked by git.
  - Use this when the provider should ship with LLMProxy.
- **Local/runtime plugin** → `plugins/<provider_id>.py`.
  - Loaded from disk on startup and hot-reloaded by the dashboard.
  - Gitignored; do not use it for repository changes.
- Never edit a user's existing `plugins/*.py` to implement a repo change. Add or
  update the template instead, and let the user re-add it from the dashboard.

The plugin file must expose exactly one of:

```python
plugin = MyProviderPlugin()
```

```python
def create_plugin() -> BasePlugin:
    return MyProviderPlugin()
```

or a `BasePlugin` subclass with a no-argument constructor.

## Copy the working skeleton

Start from `skills/llmproxy-plugin/assets/plugin_skeleton.py`. It already has:

- `PluginMetadata` with standard variables and one model.
- Async `chat_completion` with an optional `account`.
- Streaming SSE parsing and non-streaming JSON parsing.
- Tool-call forwarding and mock tool calls.
- Usage forwarding.
- Account credentials, base URL, and proxy resolution.
- Simulation mode for offline tests.

Replace the provider-specific pieces and keep the contract intact.

## Non-negotiable contract

```python
class MyProviderPlugin(BasePlugin):
    @property
    def metadata(self) -> PluginMetadata:
        ...

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        variables: Dict[str, Any],
        account: Optional[Account] = None,
    ) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionChunk]]:
        ...
```

Requirements:

- `chat_completion` is `async def`.
- It returns a `ChatCompletionResponse` when `request.stream` is false.
- It returns an async iterator of `ChatCompletionChunk` when `request.stream` is true.
- It accepts `account: Optional[Account] = None` (or `**kwargs`). The router passes
  the selected account when the provider has an account pool.
- It does not perform network I/O at import time.
- It does not hardcode secrets.
- It uses `model_config.upstream_model or model_config.id` for the upstream model.
- It reads provider settings from `variables` and credentials from `account`.
- It raises exceptions on upstream errors. The router uses the error text/status to
  detect rate limits and mark accounts for cooldown.

## Metadata conventions

```python
PluginMetadata(
    id="my_provider",              # unique, stable, lowercase, no spaces
    name="My Provider",
    description="One sentence shown in the dashboard.",
    version="1.0.0",
    author="your-name",
    enabled=True,
    filename="my_provider.py",
    variables=[...],
    models=[...],
)
```

- `id` is the provider ID used by accounts, logs, and the dashboard.
- `filename` should match the file name.
- Keep `id` stable after release; accounts are keyed by it.

## Variable conventions

Use `PluginVariable` for everything the dashboard can configure.

Common variables:

```python
PluginVariable(name="timeout", label="Timeout (seconds)", type="number", default=60)
PluginVariable(name="base_url", label="Default Base URL", type="string", default="https://api.example.com/v1")
PluginVariable(name="api_key", label="Default API Key", type="secret", default="")
PluginVariable(name="load_balancing_strategy", label="Load Balancing Strategy", type="select",
               default="round_robin", options=["round_robin", "least_used", "fallback", "random"])
PluginVariable(name="account_switch_cooldown", label="Account Cooldown (s)", type="number", default=10)
PluginVariable(name="account_switch_wait", label="Switch Wait Delay (s)", type="number", default=0)
PluginVariable(name="max_retries", label="Max Plugin Retries", type="number", default=2)
```

If the provider supports the router's external proxy pool, also declare:

```python
PluginVariable(name="proxy_enabled", label="Enable External Proxy", type="boolean", default=False)
PluginVariable(name="proxy_strategy", label="Proxy Strategy", type="select",
               default="sequential", options=["sequential", "random", "specific", "switch_on_limit"])
PluginVariable(name="specific_proxy_id", label="Specific Proxy ID / Tag", type="string", default="")
PluginVariable(name="proxy", label="Direct Proxy URL", type="string", default="")
```

Rules:

- Supported types: `string`, `number`, `boolean`, `secret`, `select`.
- `select` variables must have `options`.
- `default` should be usable without editing.
- Never put a real secret in `default`.
- Use `secret` for API keys so the dashboard masks the value.

## Model conventions

```python
PluginModelConfig(
    id="my-model",
    name="My Model",
    upstream_model="actual-upstream-model",
    context_window=128000,
    max_tokens=4096,
    parameters={},  # optional provider-specific defaults
)
```

- `id` is the model name clients send to `/v1/chat/completions`.
- `upstream_model` is what the provider receives.
- Users can add/edit/delete models in the dashboard; do not assume the default list
  is the final list.
- If the provider supports reasoning variants, either expose separate model IDs or
  parse suffixes such as `:high`/`-high` and strip them before the upstream call.
  The router also accepts a generic `<model>:<level>` ID (for example
  `gpt-6-luna:high`) and passes the level to plugins through the per-request
  `reasoning_effort` variable, so honoring that variable is enough for rule targets.

## Non-streaming implementation

Build a normal OpenAI-compatible payload and return a validated response:

```python
async with httpx.AsyncClient(timeout=timeout, proxy=proxy) as client:
    response = await client.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return ChatCompletionResponse.model_validate(response.json())
```

Always call `raise_for_status()` so the router can see 429/401/5xx errors.

## Streaming implementation

Return an async generator. Do not `await` the generator itself.

```python
async def _stream(...):
    async with httpx.AsyncClient(timeout=timeout, proxy=proxy) as client:
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

Minimum chunk sequence:

1. A role chunk: `delta={"role": "assistant", "content": ""}`.
2. Content/tool-call deltas.
3. A final chunk with `delta={}` and `finish_reason="stop"` or `"tool_calls"`.
4. Include `usage` on the final chunk when the upstream provides it.

Never yield plain dictionaries; the router expects `ChatCompletionChunk` objects
or raw strings.

## Tool calls

Forward tools when the upstream supports them:

```python
if request.tools is not None:
    payload["tools"] = request.tools
if request.tool_choice is not None:
    payload["tool_choice"] = request.tool_choice
if request.parallel_tool_calls is not None:
    payload["parallel_tool_calls"] = request.parallel_tool_calls
```

Non-streaming tool calls go in `ChatMessage.tool_calls` with
`finish_reason="tool_calls"`. Streaming tool calls use `delta["tool_calls"]` with an
`index`, `id`, `type`, `function.name`, and incremental `function.arguments`.

Handle tool results in `request.messages` where `role` is `"tool"` or `"function"`.
Do not send a new tool call when the last messages already contain tool output.

## Usage reporting

- Prefer upstream usage when present: `prompt_tokens`, `completion_tokens`,
  `total_tokens`.
- For non-streaming responses, include `usage=CompletionUsage(...)`.
- For streaming responses, include usage on the final chunk if available.
- If the upstream does not report usage, the router estimates completion tokens
  from streamed content and prompt tokens from the request. Do not invent precise
  numbers.

## Accounts and credentials

The router selects an account and passes it as `account`:

```python
api_key = account.api_key if account else str(variables.get("api_key") or "")
base_url = (account.base_url if account and account.base_url else variables.get("base_url"))
```

- `account.api_key` is the credential.
- `account.base_url` overrides the provider default.
- `account.extra_config` holds provider-specific fields (`email`, `account_id`,
  `refresh_token`, `expires_at`, `project_id`, etc.).
- `account.proxy_url` or `account.proxy_id` can route that account through a proxy.
- Never assume `account` is present. The router may use a synthetic default account
  when no accounts are configured.

For OAuth providers, refresh expired tokens before the request and persist the
updated account with `plugin_manager._save_accounts()` (see `templates/codex.py`
and `templates/antigravity.py`).

## Proxy support

The router injects the resolved proxy into `variables["proxy"]`. Read it with:

```python
proxy = str(
    variables.get("proxy")
    or (getattr(account, "proxy_url", None) if account else "")
    or ""
).strip() or None
```

Pass it to `httpx.AsyncClient(proxy=proxy)`. Do not resolve the proxy pool yourself;
the router owns that logic.

## Simulation mode

Every provider should work without credentials for dashboard tests and offline
development. Detect an empty/test key and return a deterministic response:

```python
def _is_test_mode(self, api_key: str) -> bool:
    key = (api_key or "").lower().strip()
    return not key or key.startswith(("test", "mock", "dummy", "fake", "offline"))
```

Simulation responses should:

- Return a valid `ChatCompletionResponse`.
- Return a valid async iterator of `ChatCompletionChunk` for streams.
- Include a short "Execution Details" footer with timestamp, model, tokens, and
  account status.
- Mock a tool call when `request.tools` is present and no tool result is pending.

## Error handling

- Call `response.raise_for_status()` for non-2xx responses.
- Let exceptions propagate; do not swallow rate limits and return fake success.
- Include the upstream status and a short body excerpt in the exception message.
- The router detects `429`, `rate limit`, `quota`, and `too many requests` and puts
  the account into cooldown.
- Do not log API keys, OAuth tokens, or full upstream responses containing secrets.

## Testing

Add a focused test under `tests/`. Good tests:

- Load the template/plugin and assert metadata.
- Mock `httpx` and assert the upstream URL, headers, and payload.
- Assert non-streaming returns `ChatCompletionResponse`.
- Assert streaming yields `ChatCompletionChunk` and ends with a finish reason.
- Assert tool calls are forwarded and tool results are handled.
- Assert simulation mode works with no account.
- Assert rate-limit errors propagate so the router can fail over.

Use `TestClient`, `monkeypatch`, and `httpx.MockTransport` or fake response objects.
Do not call real upstream APIs in tests.

Run:

```bash
python -m py_compile templates/my_provider.py
python skills/llmproxy-plugin/scripts/validate_plugin.py templates/my_provider.py --smoke
pytest -q --ignore=tests/test_codex_and_example_response.py
```

The two live Codex tests in `tests/test_codex_and_example_response.py` require a
real local Codex session/network and fail without one. They are pre-existing.

## Validation checklist

Before handing back a plugin:

- [ ] `python -m py_compile` passes.
- [ ] `validate_plugin.py --smoke` passes.
- [ ] Metadata has a unique `id`, name, and at least one model.
- [ ] `chat_completion` accepts `account` and is async.
- [ ] Non-streaming returns `ChatCompletionResponse`.
- [ ] Streaming returns an async iterator of `ChatCompletionChunk`.
- [ ] First chunk has the assistant role; final chunk has a finish reason.
- [ ] Tools and tool results are handled.
- [ ] Image parts are forwarded or translated with `llmproxy.core.multimodal`,
      and `supports_vision` is set on vision-capable models.
- [ ] Usage is forwarded when available.
- [ ] Simulation mode works without credentials.
- [ ] 429/401/5xx errors propagate.
- [ ] Proxy is passed to `httpx` when configured.
- [ ] No secrets are logged or committed.
- [ ] A focused test exists.
- [ ] `pytest` stable suite passes.

## Common pitfalls

- Returning a coroutine or list instead of an async generator for streams.
- Yielding dicts instead of `ChatCompletionChunk`.
- Forgetting the optional `account` parameter.
- Creating `httpx.AsyncClient` at import time.
- Using the public model ID instead of `model_config.upstream_model`.
- Ignoring `request.messages` content lists (multimodal input).
- Dropping `tools`, `tool_choice`, or `parallel_tool_calls`.
- Swallowing 429s and returning a fake response.
- Logging tokens or full request bodies.
- Editing `plugins/` for a repository change.
- Adding a dependency that is not in `requirements.txt`.
- Assuming an account always exists.
- Not closing streaming responses/clients.

## References

- `references/plugin-contract.md` — field-by-field contract and router behavior.
- `references/patterns.md` — streaming, tools, accounts, OAuth, proxies, usage.
- `references/testing.md` — test patterns and manual smoke tests.
- `assets/plugin_skeleton.py` — copyable working plugin.
- `scripts/validate_plugin.py` — metadata/signature/simulation validator.
