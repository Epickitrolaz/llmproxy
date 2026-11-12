<p align="center">
  <img src="assets/icon.svg" width="72" height="72" alt="LLMProxy Icon">
</p>

# LLMProxy

**Minimalist, High-Performance AI Router & Proxy with Dynamic Plugin Architecture.**

LLMProxy routes OpenAI-compatible chat completion requests to various LLM providers (e.g. Antigravity, Codex, OpenRouter, Anthropic Messages, and generic OpenAI-compatible APIs).

The only components hard-coded into the router are the **monochrome web dashboard** and the **hosted OpenAI API endpoint (`/v1`)**. Every provider is a standalone Python file in `plugins/` that can be swapped, edited live, or hot-reloaded directly from the dashboard.

---

## Features

- **Dynamic Plugin Architecture**:
  - Providers are standalone Python files in `plugins/` adhering to `BasePlugin`.
  - Hot-reload without downtime when files are added or modified.
  - Built-in providers:
    - `plugins/antigravity.py`: Antigravity provider (Gemini 3.8 Flash, Claude Sonnet 4.6, Claude Opus 4.6 Thinking).
    - `plugins/codex.py`: OpenAI Codex provider (GPT-6 Luna, GPT-4o, GPT-4o Mini, reasoning effort control).
    - `plugins/anthropic_proxy.py`: Anthropic Messages API provider (Claude Haiku 5.5, Sonnet 5.5, Opus 5.5, Fable 5.2, extended thinking).
    - `plugins/openrouter.py`: OpenRouter gateway with curated free models (Nemotron 3 Ultra, ERNIE 5.5 Flash Preview, Inkling 2 Small, and more).
    - `plugins/openai_compatible.py`: Generic proxy for local Ollama, vLLM, LMStudio, etc.
- **Context Caching Passthrough**:
  - Anthropic `cache_control` breakpoints are preserved on system prompts, messages, and tools.
  - `anthropic_proxy` automatically adds cache breakpoints when an OpenAI-style client sends none (`auto_prompt_cache`, enabled by default).
  - OpenAI-compatible providers forward `prompt_cache_key` and `prompt_cache_retention`.
  - Gemini/Antigravity accepts `cachedContent` references, and cached-token counts are reported through OpenAI-style usage details.
  - Request logs persist cache-read/write counters, and the dashboard shows a cache hit rate.
- **Provider Management & Fullscreen Configuration**:
  - Three concise actions per provider: **Edit**, **Refresh**, and **Delete**.
  - **Pin to Top**: Pin favorite providers to prioritize them across the dashboard and status overview.
  - **Fullscreen Configuration Window**: Fits the entire screen with dedicated tabs for:
    - **Python Code**: Live syntax validation, template insertion, local file swap, and hot-reload.
    - **Accounts & Pool**: Multi-account credentials, failovers, and priority weights.
    - **Load Balancing Strategy**: Round Robin, Least Used, Random, and Fallback with rate-limit cooldown settings.
    - **Variables**: Dynamically declared plugin variables (keys, base URLs, options).
    - **Models**: Add, edit, or remove custom model aliases and parameter mappings.
- **Local API Keys & Per-Key Usage Tracking**:
  - Create, edit, and delete local API keys directly from the **Overview** tab or via `/api/keys`.
  - Detailed per-key usage analytics: request counts, prompt tokens, completion tokens, total tokens, and last-used timestamps.
  - When local keys are created, clients authenticate using `Authorization: Bearer <key>` or `x-api-key: <key>`. (When no keys are configured, requests are allowed without authentication).
- **Custom Load Balancing Rules (Virtual Models)**:
  - Create custom virtual models exposed on the `/v1` endpoint (e.g. `smart-router`, `hybrid-coder`).
  - Select target models from any installed provider. Reasoning variants are folded
    into one target with a level picker instead of appearing as `:none`/`:high` models.
  - Select a routing strategy: Round Robin, Least Used, Random, or Fallback.
  - Ordered target list: in Fallback mode the first entry is the primary and lower
    entries are the next fallbacks; Round Robin follows the same order.
  - Transparently exposed in `GET /v1/models` and routed dynamically during completions.
- **Unified Status & Limits Dashboard**:
  - Multi-provider tabbed view designed to scale cleanly across 50+ providers.
  - Two-row layout of rounded square account cards sorted by provider name (pinned providers first).
  - Clean limit progress bars with an option to hide specific limits that do not apply to a given account.
  - **CLIProxyAPI-compatible usage hub**: T3 Code and OpenCode can read pooled account limits through `/v0/management/auth-files` and `/v0/management/api-call`. The hub URL and management key are shown on the **Overview** tab.
  - **Persistent Request Audit Logs**: All requests, latency metrics, token estimations, and errors are saved to disk (`data/request_logs.json`) and preserved across server restarts.
- **Interactive Chat Playground**:
  - Balanced 1:1 split between model parameters and the chat thread.
  - Test custom rules and provider models with real-time streaming feedback.

---

## Why Claude Code is not bundled

LLMProxy intentionally does not ship a built-in Claude Code provider. Anthropic's
terms prohibit routing Claude Code traffic through third-party proxies, and
accounts detected doing so may be suspended. Antigravity carries a similar
restriction, but its appeals process is considerably more forgiving, and the
Antigravity CLI (`agy`) is not yet as capable as Claude Code. We therefore keep
the Antigravity integration available while leaving Claude Code automation to
users with an approved setup. If you have your own sanctioned credentials, you can
still add a custom provider from the dashboard without modifying the repository.

## Anthropic and reasoning controls

The bundled `templates/anthropic_proxy.py` template talks to Anthropic's native
Messages API and pre-configures **Claude Haiku 5.5**, **Claude Sonnet 5.5**,
**Claude Opus 5.5**, and **Claude Fable 5.2**. Extended thinking is controlled by
the `reasoning_effort` variable (`none`, `low`, `medium`, `high`, `xhigh`, or
`max`). A request-level `reasoning_effort` value takes precedence, and
temperature/top-p are omitted automatically whenever thinking is active.
Claude 4.6 and newer use adaptive thinking with `output_config.effort`; older
models fall back to an explicit thinking budget.

Reasoning is returned to OpenAI-compatible clients as `reasoning_content` in
both streaming deltas and non-streaming messages, and the dashboard Playground
renders it above the answer. The plugin also keeps the signed thinking blocks
from a tool-use turn and replays them when the tool result is sent back, which
Anthropic requires to continue a reasoning chain across tool calls. Requests
may pass the level as `reasoning_effort`, as a `<model>:<level>` suffix, or as
`extra_body.reasoning.effort`.

The plugin also enables Anthropic prompt caching for OpenAI-style clients that
do not send `cache_control` themselves. With `auto_prompt_cache` on (the
default), it places one ephemeral breakpoint at the end of the system prompt and
one at the end of the final message, provided the request is at least Anthropic's
minimum cacheable size (1024 tokens, or 2048 for Haiku). Explicit client
breakpoints and `extra_body.cache_control` are left untouched. Set
`auto_prompt_cache` to `false` if you prefer to control cache writes yourself.

`GET /v1/models` lists each model once under its base ID. Models that support
reasoning also carry a `reasoning_levels` field, and every plugin-backed model
reports its configured limits through `context_window` and `max_tokens`:

```json
{
  "id": "claude-opus-5-5",
  "object": "model",
  "owned_by": "anthropic_proxy",
  "reasoning_levels": ["none", "low", "medium", "high", "xhigh", "max"],
  "context_window": 200000,
  "max_tokens": 64000
}
```

Clients can use those levels to offer a reasoning selector and send the chosen
level as `reasoning_effort` in the request body. The proxy maps it internally;
no separate model IDs are required.

The `:<level>` suffix is also accepted generically for any provider model and is
stripped before routing. For example `gpt-6-luna:high` resolves to `gpt-6-luna`
with the per-request `reasoning_effort` set to `high`. Custom load balancing
rules use this to attach a reasoning level to each target.

---

## Quickstart with Docker

### 1. Run with Docker Compose (Recommended)

```bash
docker compose up -d
```

The container runs with volume mounts for `./plugins` and `./data`, so any plugin changes on the host or in the container persist and hot-reload automatically.

### 2. Run with Docker CLI

Run the pre-built image from GitHub Container Registry (GHCR):

```bash
docker run -d --name llmproxy -p 8000:8000 -v $(pwd)/plugins:/app/plugins -v $(pwd)/data:/app/data ghcr.io/epickitrolaz/llmproxy:latest
```

Or build and run locally:

```bash
docker build -t llmproxy:latest .
docker run -d --name llmproxy -p 8000:8000 -v $(pwd)/plugins:/app/plugins -v $(pwd)/data:/app/data llmproxy:latest
```

### Accessing the Services

- **Web Dashboard**: [http://localhost:8000](http://localhost:8000)
- **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)
- **OpenAI API Base**: `http://localhost:8000/v1`

### Dashboard Security

Every `/api/*` dashboard route requires an admin token. The dashboard asks for
it once and stores it in the browser's local storage.

- Set `LLMPROXY_ADMIN_TOKEN` to a strong secret when the dashboard is reachable
  from anything other than the local machine. This is required for non-loopback
  clients: without it, remote `/api` requests are rejected even if a token file
  exists.
- When the variable is unset, the router generates a token once and writes it to
  `data/admin_token` (mode `0600`). Local requests can use that value.
- Credential fields in `GET /api/accounts` and `GET /api/keys` are masked.
  Admin-authenticated clients can fetch literal values from
  `/api/accounts/{id}/reveal` and `/api/keys/{id}/reveal`.
- CORS is disabled by default. Set `LLMPROXY_CORS_ORIGINS` to a comma-separated
  allowlist when a separate browser origin must call the API. A wildcard origin
  is supported but always disables credentials.

`/v1` client authentication is unchanged and remains independent of the
dashboard admin token.

---

## Local Development (Without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Run server with hot-reload
uvicorn llmproxy.main:app --host 0.0.0.0 --port 8000 --reload

# Run test suite
pytest
```

---

## API Usage Examples

### 1. Local API Key Authentication

> **Important**:
> If you have configured local API keys in the **Overview** tab of the dashboard, you must provide your key in the `Authorization: Bearer <key>` header or `x-api-key` header. If no local keys are created, requests are permitted without authentication.

### Using `curl`

#### List Models (Includes Provider Models & Custom Rules)
```bash
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer sk-proxy-your-key"
```

#### Chat Completion (Non-Streaming)
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-proxy-your-key" \
  -d '{
    "model": "gemini-3.8-flash",
    "messages": [
      {"role": "user", "content": "Hello LLMProxy!"}
    ]
  }'
```

#### Chat Completion via Custom Load Balancing Rule
If you created a custom rule named `smart-router` in the **Load Balancing Rules** tab:
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-proxy-your-key" \
  -d '{
    "model": "smart-router",
    "messages": [
      {"role": "user", "content": "Route to the best available provider."}
    ]
  }'
```

#### Chat Completion (Streaming SSE)
```bash
curl -N -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-proxy-your-key" \
  -d '{
    "model": "gpt-6-luna",
    "messages": [
      {"role": "user", "content": "Explain quantum computing briefly."}
    ],
    "stream": true
  }'
```

### Using Official OpenAI Python SDK

```python
from openai import OpenAI

# If you configured local API keys, pass your key here.
# If no local keys exist, any string can be passed.
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-proxy-your-key"
)

response = client.chat.completions.create(
    model="gemini-3.8-flash-high",
    messages=[{"role": "user", "content": "Hello from OpenAI Python SDK!"}],
    stream=True
)

for chunk in response:
    content = chunk.choices[0].delta.content or ""
    print(content, end="", flush=True)
print()
```

### Using with OpenCode (`opencode.json`)

To route OpenCode requests through LLMProxy, configure a provider in your `opencode.json` (e.g. `~/.config/opencode/opencode.json`). Make sure to specify `"package": "@opencode/ai/providers/openai-compatible"`:

```json
{
  "providers": {
    "llmproxy": {
      "name": "LLMProxy",
      "settings": {
        "baseURL": "http://localhost:8000/v1",
        "apiKey": "sk-proxy-your-key"
      },
      "models": {
        "gemini-3.8-flash-high": {
          "limit": {
            "context": 512000
          }
        }
      },
      "package": "@opencode/ai/providers/openai-compatible"
    }
  }
}
```

Then run OpenCode with:
```bash
opencode run -m llmproxy/gemini-3.8-flash-high "Explain quantum computing briefly."
```

---

## Troubleshooting & Common User Pitfalls

1. **HTTP 401 Unauthorized (`missing_api_key` or `invalid_api_key`)**:
   - Cause: You have created one or more local API keys in LLMProxy, but your client is sending requests without the `Authorization: Bearer <key>` header or with an invalid key.
   - Fix: Check the **Overview** tab in LLMProxy to copy your active key secret, or pass `x-api-key: <key>`. If you delete all keys, the endpoint reverts to open mode.

2. **HTTP 429 Rate Limit Cooldown (`accounts_rate_limited`)**:
   - Cause: All connected accounts for an upstream provider have hit their rate limits and are in cooldown quarantine.
   - Fix: Click **Edit** on the provider card, go to the **Accounts & Pool** tab, and click **Reset** on any account in cooldown, or add additional accounts to the pool.

3. **HTTP 404 Model Not Found**:
   - Cause: The requested model ID does not exist in any active provider or custom load balancing rule.
   - Fix: Check `GET /v1/models` to verify available models. Ensure the corresponding provider plugin is toggled on, or configure a virtual model in the **Load Balancing Rules** tab.

4. **Dropdown Menus & Search**:
   - Every select menu in the dashboard is an accessible searchable dropdown. Type to filter options and press Enter or click to select.
