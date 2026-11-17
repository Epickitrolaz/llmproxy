# Provider Plugin Guidelines

These rules apply to every new or changed provider plugin in `templates/` and
`plugins/`. They exist because a plugin that returns text but drops usage,
streaming, tools, or errors is not a working provider.

For the full contract and code examples, read `skills/llmproxy-plugin/SKILL.md`
and its `references/` folder. This page is the short, mandatory checklist.

## 1. Inspect the upstream before writing code

Do not guess the wire protocol. Read the provider's API documentation and, when
possible, capture one real response (dashboard Playground, `curl`, or the
provider's SDK).

Answer these questions first:

- What are the endpoint paths for non-streaming and streaming?
- Which auth header does it use (bearer token, API key header, query key)?
- What are the exact request fields for system prompts, tools, images, and
  reasoning controls?
- **Does the response include token usage?** Find the exact field names.
- Does the streaming response include usage, or must a stream option be enabled?
- Does the provider expose a cache signal or cache token counters?
- Which error status codes mean rate limit, auth failure, and invalid request?

If the upstream exposes reasoning controls or prompt/context caching, implementing
them is mandatory, not optional. See section 3.

## 2. Usage tracking is mandatory when upstream reports it

If the upstream exposes token counts, the plugin must report them. This is not
optional polish.

- Map the provider's usage object into `CompletionUsage` for **both**
  non-streaming and streaming responses.
- `prompt_tokens` must be the full prompt count, including cached tokens.
- Put cache reads in `prompt_tokens_details.cached_tokens` and/or
  `cache_read_input_tokens`.
- Put reasoning tokens in `completion_tokens_details.reasoning_tokens` when the
  provider reports them separately.
- For streaming, include usage on the final chunk. If the provider only returns
  usage when asked, request it (for example OpenAI's
  `stream_options.include_usage`) even when the client did not.
- Add a test that proves the counters reach `RequestLogEntry`, the dashboard log
  table, and the cache hit-rate statistic.
- If the upstream truly has no usage data, say so in the plugin docstring and
  let the router estimate. Never invent precise token numbers.

Example field names from real providers:

| Provider | Prompt | Completion | Cache read | Reasoning |
|---|---|---|---|---|
| OpenAI | `usage.prompt_tokens` | `usage.completion_tokens` | `prompt_tokens_details.cached_tokens` | `completion_tokens_details.reasoning_tokens` |
| Anthropic | `input_tokens` + cache fields | `output_tokens` | `cache_read_input_tokens` | - |
| Gemini | `usageMetadata.promptTokenCount` | `candidatesTokenCount` + `thoughtsTokenCount` | `cachedContentTokenCount` | `thoughtsTokenCount` |

## 3. Reasoning and caching are mandatory when upstream supports them

A provider is not complete if it returns text but ignores reasoning controls or
cache support that the upstream API already exposes. Optional only means
"the upstream has no such feature", never "the plugin author skipped it".

### Reasoning

- Inspect the upstream API for thinking/reasoning controls and their exact
  field names.
- Map the proxy's `reasoning_effort` into the provider's native control
  (`thinkingConfig`, `reasoning_effort`, `thinking.budget_tokens`,
  `thinkingLevel`, etc.) for both streaming and non-streaming requests.
- Advertise every level the upstream actually accepts through
  `PluginModelConfig.reasoning_levels`. Do not invent levels the API rejects.
- Return reasoning output to OpenAI clients as `reasoning_content` when the
  provider exposes it.
- If the upstream reports reasoning token counts, map them to
  `completion_tokens_details.reasoning_tokens`.
- If the upstream has no reasoning controls, leave `reasoning_levels` empty and
  document that in the plugin docstring. Do not fake a reasoning selector.

### Caching

- Inspect the upstream API for prompt/context cache support.
- Forward the provider's native cache hints from message content and
  `extra_body`: `cache_control` breakpoints, `cachedContent`,
  `prompt_cache_key` / `prompt_cache_retention`, affinity/session IDs, etc.
- For streaming, request upstream usage even when the client did not ask, so
  cache counters are always logged.
- Report cache reads and writes through `CompletionUsage` using the OpenAI or
  Anthropic shapes described below.
- If the upstream has only implicit caching, still forward explicit references
  when the client supplies them, and report cache counters from usage.
- If the upstream has no cache support, document that in the plugin docstring.

Both features need focused tests: reasoning control translation and cache-hint
forwarding, plus cache telemetry reaching the dashboard log and hit rate.

## 4. Implement the whole contract

- `chat_completion` is `async` and accepts `account: Optional[Account] = None`.
- Non-streaming returns `ChatCompletionResponse`.
- Streaming returns an async iterator of `ChatCompletionChunk`.
- The first streamed chunk sets `delta.role = "assistant"`; the last chunk has a
  `finish_reason` and, when available, `usage`.
- Never yield plain dicts or a coroutine instead of an async generator.
- Use `model_config.upstream_model or model_config.id` for the upstream model.
- Read credentials from `account.api_key`; never add an `api_key` plugin variable.
- Pass the router-provided `variables["proxy"]` to `httpx.AsyncClient(proxy=...)`.
- Raise on upstream errors. Do not swallow 429/401/5xx and return fake success.
- Do not perform network I/O at import time.

## 5. Translate the features clients actually use

- **System prompts** become the provider's native system field.
- **Tool calls and tool results** map in both directions, and streamed tool calls
  include `index`, `id`, `type`, `function.name`, and incremental arguments.
- **Vision** uses `llmproxy.core.multimodal` so image parts work across OpenAI,
  Anthropic, Gemini, and data-URL spellings. Set `supports_vision=True`.
- **Reasoning** maps `reasoning_effort` to the provider's native controls and
  advertises valid choices through `reasoning_levels`. Do not publish synthetic
  `<model>:<level>` model IDs.
- **Caching** forwards every native hint (`cache_control`, `cachedContent`,
  `prompt_cache_key`, affinity/session IDs). If the provider has implicit
  caching only, still forward explicit references when supplied.

## 6. Avoid silent half-working states

Reject these patterns:

- Streaming works but non-streaming is missing, or vice versa.
- Tools are forwarded to the upstream but tool results are not translated back.
- Reasoning controls or cache hints exist upstream but the plugin silently drops
  them.
- Usage is logged only when the client asks for it.
- A "test mode" that hides real request-building bugs. Test mode must be
  deterministic, but the real path must be covered by mocked tests.
- Model IDs exposed without `upstream_model`.
- Provider-specific credentials stored in plugin variables or source code.
- Catching every exception and returning `"Simulated response"`.

Blocked or filtered content is a valid upstream result. Return it with
`finish_reason="content_filter"` (or an explicit error) rather than pretending it
was a normal answer.

## 7. Simulation mode must never hide real failures

Simulation exists for offline tests, explicit test keys, and the synthetic
default account used when no provider account is configured. It must not turn a
real failure into a fake success.

- Never simulate when a real account was selected but its request failed,
  including rate limits, quota exhaustion, auth failures, or upstream errors.
- Never simulate when a real account exists but is in cooldown, disabled, or
  missing a credential the provider requires.
- Only treat an empty credential as simulation when there is no configured
  account (the synthetic `default-*` account) or the provider intentionally
  allows anonymous access.
- Raise a clear credential/auth error for a real account with a missing
  credential. Do not return a simulated response.
- The dashboard test/playground endpoints must surface cooldown and no-account
  errors instead of falling back to a no-key synthetic account.
- Treat keys starting with `test`, `mock`, `dummy`, `fake`, `offline`, or
  `sample` as explicit simulation requests; never infer simulation from a
  failed real request.

## 8. Tests before handoff

Every plugin needs focused tests under `tests/` that cover:

- Metadata and model IDs (prefix convention from `README.md`).
- Request translation: system, user, assistant, tools, images, reasoning, cache.
- Reasoning control mapping and reasoning output/token counters.
- Cache-hint forwarding and cached-token usage mapping.
- Non-streaming response parsing and usage.
- Streaming chunks, finish reason, and final usage.
- Rate-limit/error propagation.
- Simulation mode with no credentials.
- Real-account credential/cooldown failures never returning simulation output.
- Usage counters reaching the dashboard log and cache statistics.

Run:

```bash
python -m py_compile templates/<provider>.py
python skills/llmproxy-plugin/scripts/validate_plugin.py templates/<provider>.py --smoke
pytest -q --ignore=tests/test_codex_and_example_response.py
```

## 9. Definition of done

- The plugin is in `templates/` (tracked), not `plugins/` (runtime data).
- The model prefix is added to `README.md`, `AGENTS.md`, and
  `tests/test_model_prefixes.py`.
- Usage and cache telemetry are visible in the dashboard for a mocked upstream
  response.
- Reasoning and cache support are implemented whenever the upstream exposes
  them, and explicitly documented as unsupported when it does not.
- Errors keep the upstream status and a short body excerpt so failover and
  cooldown detection work.
- No secrets, tokens, or full request/response bodies are logged or committed.
