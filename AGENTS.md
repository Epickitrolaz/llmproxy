# AGENTS.md

Guidance for AI coding agents working on LLMProxy.

## What this project is

LLMProxy is a FastAPI-based OpenAI-compatible router. It exposes `/v1`, loads
provider plugins from `plugins/` at runtime, and serves a vanilla JavaScript
dashboard from `/`.

Stack:

- Python 3.12+ (the Docker image uses 3.12; local environments may be newer).
- FastAPI + Uvicorn.
- httpx for upstream requests.
- Pydantic v2 for schemas and validation.
- Vanilla HTML/CSS/JS dashboard, no build step.

## Read first

Before making changes, read these in order:

1. `llmproxy/core/interface.py` — the `BasePlugin` contract.
2. `llmproxy/schemas.py` — API and plugin schemas.
3. `llmproxy/core/manager.py` — plugin discovery, configs, accounts, rules,
   proxies, logs, and persistence.
4. `llmproxy/api/router.py` — the `/v1` request flow, account selection, failover,
   streaming, usage, and logging.
5. The closest provider template in `templates/`.
6. `skills/llmproxy-plugin/SKILL.md` when the task involves a provider plugin.
7. `docs/PLUGIN_GUIDELINES.md` before creating or reviewing any provider plugin.

## Quick commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally with hot reload
uvicorn llmproxy.main:app --host 0.0.0.0 --port 8000 --reload

# Run the stable test suite
pytest -q --ignore=tests/test_codex_and_example_response.py

# Full suite (two live Codex tests require a real local Codex session/network)
pytest -q

# Compile-check a Python file
python -m py_compile path/to/file.py

# Syntax-check the dashboard JS (if Node is available)
node --check llmproxy/static/app.js

# Validate a provider plugin
python skills/llmproxy-plugin/scripts/validate_plugin.py templates/my_provider.py --smoke

# Docker
docker compose up -d
docker build -t llmproxy:latest .
```

## Repository map

```text
llmproxy/
  main.py                 FastAPI app, router registration, static mount
  schemas.py              Pydantic schemas for API, plugins, accounts, logs
  api/router.py           OpenAI-compatible /v1 endpoints
  core/interface.py       BasePlugin contract
  core/manager.py         PluginManager: discovery, configs, accounts, rules,
                          proxies, logs, stats, persistence
  core/*_oauth.py         Provider OAuth helpers (Codex, Antigravity)
  dashboard/router.py     /api dashboard endpoints
  dashboard/loader.py     Plugin file load/validate helpers
  dashboard/templates.py  Template gallery loader
  static/                 index.html, app.js, style.css
plugins/                  Runtime provider plugins (gitignored)
templates/                Tracked built-in provider templates
tests/                    Pytest suite
data/                     Runtime JSON persistence (gitignored)
skills/                   Vendor-neutral agent skills
```

## Request flow

`POST /v1/chat/completions`:

1. Authenticate the local API key if any keys are configured.
2. Handle the special `test-model` without a plugin.
3. Resolve the model through `plugin_manager.resolve_model()`:
   - custom load-balancing rules first,
   - then enabled models across enabled plugins.
4. Validate plugin variables.
5. Select account(s) using the provider's load-balancing strategy and cooldowns.
6. Resolve the external proxy (account-level first, then provider proxy pool).
7. Call `plugin.chat_completion(request, model_config, variables, account=account)`.
8. Forward streamed `ChatCompletionChunk` objects as SSE, or return the
   `ChatCompletionResponse` directly.
9. Record the request log, API key usage, and account/proxy success or failure.

The router owns account failover, rate-limit detection, proxy selection, and
request logging. Plugins should raise exceptions instead of returning fake errors.

## Plugin contract in one screen

A provider plugin is a standalone Python file in `templates/` (tracked, built-in)
or `plugins/` (runtime, gitignored). It must expose `plugin = MyPlugin()`,
`create_plugin()`, or a `BasePlugin` subclass.

```python
class MyPlugin(BasePlugin):
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

Rules:

- `chat_completion` is async.
- Non-streaming returns `ChatCompletionResponse`.
- Streaming returns an async iterator of `ChatCompletionChunk`.
- Accept `account` (or `**kwargs`); the router passes the selected account.
- Use `model_config.upstream_model or model_config.id`.
- Read settings from `variables` and credentials from `account`.
- Do not perform network I/O at import time.
- Do not hardcode secrets.
- Do not swallow rate-limit errors.

For the full contract, field tables, and examples, see
`skills/llmproxy-plugin/SKILL.md` and its `references/` folder.

## Model listings and reasoning levels

- `GET /v1/models` and the dashboard model list expose each model once under its
  exposed ID. Do not publish synthetic `<model>:<level>` entries as separate models.
- Built-in templates expose model IDs as `<short-prefix>/<model-name>` and keep
  the provider's native ID in `upstream_model`. The current prefixes are `ap`,
  `ag`, `cx`, `gm`, `mock`, `nous`, `np`, `compat`, `oc`, `ocg`, and `or`.
- Advertise supported reasoning through `reasoning_levels` on
  `PluginModelConfig` and `ModelCard`. Clients select a level with
  `reasoning_effort`; the `:<level>` suffix stays accepted for compatibility.
- The provider plugin maps the level to its native controls or upstream model ID
  (see `templates/antigravity.py`). The proxy handles the ID switching; clients
  never need the internal ID.
- When a client needs a reasoning selector (for example OpenCode variants), build
  it from `reasoning_levels` instead of adding duplicate model IDs.
- Custom load-balancing rules in manual reasoning mode must not advertise
  reasoning levels. In client-reasoning mode, advertise only
  `exposed_reasoning_levels` that every selected target can serve directly or
  through `reasoning_overrides`, and remap the request per selected target.
  Targets without reasoning support can map a level to `__default__`, which
  accepts the request without forwarding a reasoning control.

## Plugin-owned OAuth login

OAuth login is part of the plugin, not the proxy. The dashboard exposes one
generic route set and calls optional `BasePlugin` methods:

- `POST /api/plugins/{plugin_id}/oauth/start`
- `GET /api/plugins/{plugin_id}/oauth/status/{flow_id}`
- `POST /api/plugins/{plugin_id}/oauth/callback`
- `POST /api/plugins/{plugin_id}/oauth/import-local`

Set the metadata flags and implement the methods in the provider template:

```python
PluginMetadata(oauth_login=True, oauth_local_import=True, oauth_hint="...")

async def oauth_start(self) -> Dict[str, Any]: ...
async def oauth_status(self, flow_id: str) -> Dict[str, Any]: ...
async def oauth_callback(self, flow_id: str, callback_url: str) -> Dict[str, Any]: ...
async def oauth_import_local(self) -> Optional[Account]: ...
async def refresh_account(self, account: Account) -> bool: ...
```

`oauth_start` returns a flow descriptor with `flow_id` and `mode`
(`device`, `loopback`, or `callback`), plus `verification_uri_complete` /
`user_code` for device flows or `auth_url` / `redirect_uri` / `state` for
loopback flows. `oauth_status` returns `{"status": "pending"|"complete"|"error"}`;
on success it includes an `Account`. The dashboard saves the account by ID or by
same provider/name, so singleton providers can return a fresh ID.

Keep the provider's PKCE/device flow, callback parsing, local import, token
exchange, and refresh logic inside the plugin template. Do not add
provider-specific OAuth routes or core OAuth modules to the proxy.

## Provider reasoning and caching

Every provider plugin must fully support reasoning controls and prompt/context
caching whenever the upstream API exposes them. "Optional" only means the
upstream has no such feature; a new provider is not complete if it drops them.

Reasoning requirements:

- Map the proxy's `reasoning_effort` into the provider's native controls for
  streaming and non-streaming requests.
- Advertise only levels the upstream accepts through `reasoning_levels`, and
  return reasoning output as `reasoning_content`.
- Map reasoning token counters to
  `completion_tokens_details.reasoning_tokens` when reported.
- If the upstream has no reasoning controls, leave `reasoning_levels` empty and
  document that in the plugin docstring instead of faking a selector.

Caching requirements:

- Forward the provider's native cache hints (`cache_control` breakpoints,
  `cachedContent`, `prompt_cache_key` / `prompt_cache_retention`, or
  session/affinity IDs) from message content and `extra_body`. Never drop them.
- If the upstream has only implicit caching, still forward explicit references
  when supplied and report cache counters from usage.
- For streaming, always request upstream usage (for example
  `stream_options.include_usage`) so cache counters are logged even when the
  client does not ask for usage. The router forwards usage to the client only
  when the client opts in.
- Report cache counters through `CompletionUsage`: OpenAI-style
  `prompt_tokens_details.cached_tokens` or Anthropic-style
  `cache_read_input_tokens` / `cache_creation_input_tokens`. The schema
  normalizes both shapes, and `prompt_tokens` must be the full prompt count
  including cached tokens (OpenAI semantics).

Add focused tests for reasoning translation and cache telemetry. A new provider
is not complete until its cache hits appear in the dashboard log table and cache
hit rate, and its reasoning controls reach the upstream.

## Simulation mode and account failures

Simulation is only for offline tests, explicit test/mock keys, and the synthetic
default account used when no provider account is configured. It must never hide
a real failure.

- Do not simulate when a real account is in cooldown, disabled, missing a
  required credential, rate limited, or the upstream returned an error.
- Raise a clear credential error when a real account has no API key. Providers
  that intentionally allow anonymous access may still work without one.
- Dashboard test/playground endpoints must return the real cooldown/503/auth
  error instead of falling back to a no-key synthetic account.
- Explicit keys starting with `test`, `mock`, `dummy`, `fake`, `offline`, or
  `sample` may trigger simulation; failed real requests may not.

## Where to make changes

| Task | Location |
|---|---|
| `/v1` routing, failover, streaming, usage | `llmproxy/api/router.py` |
| Plugin discovery/config/accounts/rules/proxies | `llmproxy/core/manager.py` |
| API/plugin/account schemas | `llmproxy/schemas.py` |
| Dashboard API endpoints | `llmproxy/dashboard/router.py` |
| Plugin file load/validate helpers | `llmproxy/dashboard/loader.py` |
| Template gallery | `llmproxy/dashboard/templates.py` |
| Dashboard UI | `llmproxy/static/` |
| Built-in provider templates | `templates/` |
| Runtime provider plugins | `plugins/` (gitignored; do not commit) |
| Tests | `tests/` |
| Agent guidance and plugin skill | `AGENTS.md`, `skills/` |

Never edit `plugins/*.py` to implement a repository change. `plugins/` is runtime
user data. Add or update the corresponding `templates/*.py` file and let the user
re-add it from the dashboard.

## Data and persistence

Runtime state lives in `data/` as JSON:

- `plugin_configs.json` — enabled/pinned/variables/custom models per plugin.
- `accounts.json` — provider accounts and credentials.
- `api_keys.json` — local client keys and usage counters.
- `load_balancing_rules.json` — virtual models.
- `request_logs.json` — last 1000 request logs.
- `proxies.json` — external proxy pool.
- `stats.json` — lifetime request count.

Writes use atomic temp-file replacement. Logs and API key usage are debounced.
`data/` and `plugins/` are gitignored. Never commit runtime credentials or logs.

## Coding conventions

- Python: `from __future__ import annotations`, type hints, `async def`, module
  loggers (`logger = logging.getLogger(__name__)`).
- Pydantic v2: use `model_validate`, `model_dump`, `model_copy`.
- Keep plugins self-contained. Import only the contract, schemas, OAuth helpers,
  multimodal helpers (`llmproxy.core.multimodal`), stdlib, and `httpx`.
- Do not add dependencies unless the feature genuinely requires one; update
  `requirements.txt` if you do.
- Prefer small helper functions over large `chat_completion` bodies.
- Keep existing endpoint shapes and error codes stable unless the task explicitly
  changes them.
- Dashboard: vanilla JS, no framework, no build step. Add DOM elements to the
  `el` cache in `app.js`, use the `api()` helper, and escape rendered text with
  `escapeHtml`.
- Use CSS variables from `style.css`; keep the monochrome design.
- Do not log API keys, OAuth tokens, cookies, or full request/response bodies.

## Testing

- Add focused tests under `tests/` using `TestClient` and `monkeypatch`.
- Mock `httpx` or use `httpx.MockTransport`; never call live upstream APIs in tests.
- For streaming, assert SSE chunks and the final finish reason.
- For account failover, assert every eligible account is attempted before an error.
- For plugin changes, run `python skills/llmproxy-plugin/scripts/validate_plugin.py`
  on the template.
- Run `pytest -q --ignore=tests/test_codex_and_example_response.py` before handing
  back.

Known pre-existing failures:

- `tests/test_codex_and_example_response.py::test_feature2_codex_cli_execution_live`
- `tests/test_codex_and_example_response.py::test_connected_codex_account_executes_real_wire_protocol_not_simulation`

Both require a real local Codex session/network. They fail on a clean checkout.

## Adding a provider plugin

Use the `llmproxy-plugin` skill and read `docs/PLUGIN_GUIDELINES.md`. The short
version:

1. Copy `skills/llmproxy-plugin/assets/plugin_skeleton.py` to
   `templates/<provider_id>.py` (built-in) or `plugins/<provider_id>.py` (local).
2. Set a unique `metadata.id`, name, description, version, variables, and models.
3. Implement `chat_completion` with non-streaming and streaming paths.
4. Handle accounts, base URL, credentials, proxy, tools, usage, reasoning, and
   errors.
5. Check the upstream API for token usage metadata. If it reports usage, map it
   into `CompletionUsage` for both streaming and non-streaming responses; do not
   ship a provider before that works.
6. Check the upstream API for reasoning controls and prompt/context caching. If
   they exist, implement them fully (including streaming and dashboard cache
   counters); do not ship a provider that silently drops them.
7. Keep a deterministic simulation mode for tests and offline use.
8. Forward reasoning controls, request streaming usage, and cache hints; report
   reasoning and cache counters.
9. Add focused tests for reasoning translation and cache telemetry.
10. Run the validator and the stable test suite.
11. Smoke test via the dashboard Playground or `curl`.

## Definition of done

- `python -m py_compile` passes for changed Python files.
- `node --check llmproxy/static/app.js` passes for JS changes (if Node is available).
- Focused tests pass; stable suite passes.
- New behavior is documented in README or the relevant skill/reference when
  user-facing.
- New or changed providers implement reasoning controls and cache support
  whenever the upstream exposes them, and surface reasoning/cache counters in
  the dashboard log table and cache hit rate.
- No secrets, tokens, runtime data, or `plugins/` files are committed.
- Existing API shapes and error codes remain compatible unless intentionally changed.
- The dashboard still renders and the affected flows still work.

## Agent behavior

- Prefer editing existing patterns over inventing new abstractions.
- Investigate before deleting or overwriting unfamiliar files.
- Do not refactor unrelated code in a bug-fix PR.
- Check `git status` before and after changes; keep the diff scoped.
- Use `rg`/`grep` to locate patterns, and run tests rather than guessing.
- If a test failure is pre-existing, verify it on `main` and say so in the handoff.
- Do not commit `data/`, `plugins/`, caches, or generated artifacts.

## Pull requests

- Branch from `main`.
- Keep the PR focused on one issue or feature.
- Run the stable test suite and mention any pre-existing failures.
- Link the issue or PR in the handoff.
- Do not merge or force-push unless asked.

## Skills

- `skills/llmproxy-plugin/` — build/review provider plugins.
- `skills/README.md` — how to point external agents at these skills.

When a task matches a skill, load its `SKILL.md` first and follow its checklist.
