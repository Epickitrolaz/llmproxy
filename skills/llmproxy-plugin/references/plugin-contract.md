# LLMProxy Plugin Contract Reference

This document describes the exact interfaces a provider plugin uses. The source
of truth is `llmproxy/core/interface.py` and `llmproxy/schemas.py`.

## BasePlugin

```python
class BasePlugin(ABC):
    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        ...

    @abstractmethod
    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        variables: Dict[str, Any],
        account: Optional[Account] = None,
    ) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionChunk]]:
        ...

    def validate_variables(self, variables: Dict[str, Any]) -> List[str]:
        ...
```

`validate_variables` is optional. The default implementation checks only variables
declared with `required=True`. Override it for format checks (URLs, JSON headers,
model suffixes, etc.).

## Plugin discovery

The manager scans `plugins/*.py` (not `templates/*.py`) and loads each file. It
accepts:

1. `plugin = MyPlugin()`
2. `create_plugin()` returning a `BasePlugin`
3. Any `BasePlugin` subclass defined in the module (instantiated with no arguments)

The loader sets `metadata.filename` and stores the plugin under `metadata.id`.
If two files use the same ID, the last loaded file wins. Keep IDs unique.

The dashboard can hot-reload a plugin file after editing it. A plugin must be
import-safe: no network calls, no file writes, no long blocking work at import.

## PluginMetadata

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | `str` | yes | Stable provider ID. Accounts and logs use it. |
| `name` | `str` | yes | Dashboard display name. |
| `description` | `str` | no | Shown in provider cards. |
| `version` | `str` | no | Defaults to `1.0.0`. |
| `author` | `str` | no | Defaults to `custom`. |
| `enabled` | `bool` | no | Default enabled state. |
| `pinned` | `bool` | no | Dashboard pin state. |
| `filename` | `str \| None` | no | Set by the loader. |
| `variables` | `list[PluginVariable]` | no | Dashboard-editable settings. |
| `models` | `list[PluginModelConfig]` | no | Default models. |

## PluginVariable

| Field | Type | Notes |
|---|---|---|
| `name` | `str` | Key used in `variables` dict. |
| `label` | `str` | Dashboard label. |
| `type` | `str` | `string`, `number`, `boolean`, `secret`, or `select`. |
| `default` | `Any` | Default value. |
| `value` | `Any` | Persisted/current value. |
| `required` | `bool` | Default validation rejects empty values. |
| `description` | `str` | Help text. |
| `options` | `list[str] \| None` | Required for `select`. |

`secret` variables are rendered as password inputs. Do not put real secrets in
`default`; use an empty default and let the user configure it.

## PluginModelConfig

| Field | Type | Notes |
|---|---|---|
| `id` | `str` | Public model ID exposed by `/v1/models`. |
| `name` | `str` | Display name. |
| `upstream_model` | `str \| None` | Actual upstream model ID. |
| `enabled` | `bool` | Whether the model is exposed. |
| `context_window` | `int \| None` | Metadata only. |
| `max_tokens` | `int \| None` | Metadata/default output cap. |
| `parameters` | `dict` | Provider-specific defaults. |

Users can add, edit, disable, or delete models in the dashboard. The plugin must
work with the model config it receives, not only with the defaults in metadata.

## Account

| Field | Type | Notes |
|---|---|---|
| `id` | `str` | Stable account ID. |
| `provider_id` | `str` | Plugin `metadata.id`. |
| `name` | `str` | Display name. |
| `api_key` | `str` | Primary credential. |
| `base_url` | `str \| None` | Account-specific upstream base URL. |
| `priority` | `int` | Lower is preferred for fallback/priority. |
| `weight` | `int` | Reserved for weighted strategies. |
| `enabled` | `bool` | Disabled accounts are not selected. |
| `pinned` | `bool` | Dashboard ordering only. |
| `rate_limited_until` | `float \| None` | Unix timestamp; account is unavailable until then. |
| `error_count` | `int` | Failure counter. |
| `success_count` | `int` | Success counter. |
| `last_used_at` | `float \| None` | Unix timestamp. |
| `proxy_id` | `str \| None` | Proxy pool entry. |
| `proxy_url` | `str \| None` | Direct proxy URL. |
| `extra_config` | `dict` | Provider-specific credentials/metadata. |

`account.is_available()` returns false when disabled or still rate-limited.

Common `extra_config` keys:

- Codex: `account_id`, `plan_type`, `email`, `refresh_token`, `expires_at`.
- Antigravity: `project_id`, `email`, `tier`, `refresh_token`, `expires_at`,
  `client_id`, `client_secret`.
- Custom providers: whatever the plugin documents.

## ChatCompletionRequest

Important fields:

| Field | Type | Notes |
|---|---|---|
| `model` | `str` | Requested public model ID. |
| `messages` | `list[ChatMessage]` | Conversation. |
| `stream` | `bool` | Streaming requested. |
| `temperature`, `top_p` | `float \| None` | Sampling. |
| `max_tokens`, `max_completion_tokens` | `int \| None` | Output cap. |
| `tools`, `tool_choice`, `parallel_tool_calls` | various | Tool calling. |
| `response_format` | `dict \| None` | Structured output. |
| `stop` | `str \| list[str] \| None` | Stop sequences. |
| `reasoning_effort` | `str \| None` | Reasoning hint. |
| `extra_body` | `dict \| None` | Extra provider parameters. |
| `stream_options` | `dict \| None` | `{"include_usage": true}` when requested. |

`ChatCompletionRequest` allows extra fields (`extra="allow"`), so unknown client
parameters are preserved on the model. Read provider-specific options from
`extra_body` or `model_dump()`.

## ChatMessage

| Field | Type | Notes |
|---|---|---|
| `role` | `str` | `system`, `user`, `assistant`, `function`, `tool`. |
| `content` | `str \| list[dict] \| None` | Text or multimodal parts. |
| `name` | `str \| None` | Optional participant name. |
| `tool_calls` | `list[dict] \| None` | Assistant tool calls. |
| `tool_call_id` | `str \| None` | Tool result correlation ID. |
| `function_call` | `dict \| None` | Legacy function call. |

When forwarding to OpenAI-compatible upstreams, `message.model_dump(exclude_none=True)`
is usually correct. For providers with a different wire format, translate explicitly.

## ChatCompletionResponse

```python
ChatCompletionResponse(
    id="chatcmpl-...",
    object="chat.completion",
    created=1234567890,
    model="public-model-id",
    choices=[
        ChatChoice(
            index=0,
            message=ChatMessage(role="assistant", content="..."),
            finish_reason="stop",
        )
    ],
    usage=CompletionUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
)
```

`ChatChoice` fields: `index`, `message`, `finish_reason`, `logprobs`.

## ChatCompletionChunk

```python
ChatCompletionChunk(
    id="chatcmpl-...",
    object="chat.completion.chunk",
    created=1234567890,
    model="public-model-id",
    choices=[
        ChatChoiceDelta(
            index=0,
            delta={"content": "Hello"},
            finish_reason=None,
        )
    ],
    usage=None,
)
```

`ChatChoiceDelta` fields: `index`, `delta`, `finish_reason`, `logprobs`.

`delta` keys commonly used:

- `{"role": "assistant", "content": ""}` — first chunk.
- `{"content": "..."}` — text delta.
- `{"tool_calls": [{"index": 0, "id": "...", "type": "function", "function": {"name": "...", "arguments": "..."}}]}` — tool call delta.
- `{}` with `finish_reason="stop"` — final chunk.

`usage` belongs on the final chunk when the upstream reports it. The router can
also add a terminal usage-only chunk for `stream_options.include_usage`.

## CompletionUsage

| Field | Type | Notes |
|---|---|---|
| `prompt_tokens` | `int` | Input tokens. |
| `completion_tokens` | `int` | Output tokens. |
| `total_tokens` | `int` | Sum. |

The router estimates tokens if the plugin omits usage. Do not fabricate precise
usage when the upstream does not report it.

## Router request flow

`POST /v1/chat/completions`:

1. Authenticate the local API key (if any keys exist).
2. Handle the special `test-model` without a plugin.
3. Resolve the requested model:
   - custom load-balancing rule first,
   - then enabled models across enabled plugins.
4. Validate plugin variables.
5. Select account(s) using the provider's `load_balancing_strategy` and cooldowns.
6. Resolve the external proxy (account-level first, then provider proxy pool).
7. Call `plugin.chat_completion(...)`.
8. For streaming, forward `ChatCompletionChunk` objects as SSE; optionally emit a
   terminal usage chunk.
9. Record the request log, API key usage, and account/proxy success or failure.

The router handles account failover and rate-limit detection. Plugins should raise
exceptions rather than returning fake error responses.

## Template gallery

`llmproxy/dashboard/templates.py` reads every `templates/*.py` file and extracts
`id`, `name`, and `description` with a regex. The dashboard's "Add from Template"
action copies the file into `plugins/` and loads it. This is why built-in providers
belong in `templates/` and runtime copies belong in `plugins/`.

## Data persistence

Runtime JSON lives in `data/`:

- `plugin_configs.json` — enabled/pinned/variables/custom models per plugin.
- `accounts.json` — provider accounts.
- `api_keys.json` — local client keys and usage counters.
- `load_balancing_rules.json` — virtual models.
- `request_logs.json` — last 1000 request logs.
- `proxies.json` — external proxy pool.
- `stats.json` — lifetime request count.

`data/` is gitignored. Never commit runtime credentials or logs.

## HTTP endpoints

- `GET /v1/models`, `GET /v1/models/{id}` — model catalog.
- `POST /v1/chat/completions` — OpenAI-compatible chat.
- `GET /api/plugins`, `/api/plugins/{id}/code`, `/api/plugins/{id}/config` — plugin management.
- `GET/POST/PUT/DELETE /api/accounts` — account management.
- `GET/POST/PUT/DELETE /api/keys` — local API keys.
- `GET/POST/PUT/DELETE /api/load-balancing-rules` — virtual models.
- `GET/POST/PUT/DELETE /api/proxies` — external proxy pool.
- `GET /api/logs`, `GET /api/stats` — telemetry.
- `GET /api/antigravity/status`, `GET /api/codex/status` — live quota summaries.
- `POST /api/test-chat` — dashboard playground.
- `GET /health` — health check.
