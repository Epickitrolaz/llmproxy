<p align="center">
  <img src="assets/icon.svg" width="72" height="72" alt="LLMProxy Icon">
</p>

# LLMProxy

**Minimalist, High-Performance AI Router & Proxy with Dynamic Plugin Architecture.**

LLMProxy routes OpenAI-compatible chat completion requests to various LLM providers (e.g. Antigravity, Codex, Claude Code, and generic OpenAI-compatible APIs). 

The only components hard-coded into the router are the **monochrome web dashboard** and the **hosted OpenAI API endpoint (`/v1`)**. Every provider is a standalone Python file in `plugins/` that can be swapped, edited live, or hot-reloaded directly from the dashboard.

---

## Features

- **Dynamic Plugin Architecture**:
  - Providers are standalone Python files in `plugins/` adhering to `BasePlugin`.
  - Hot-reload without downtime when files are added or modified.
  - Built-in providers:
    - `plugins/antigravity.py`: Antigravity provider (Gemini 3.8 Flash, Claude Sonnet 4.6, Claude Opus 4.6 Thinking).
    - `plugins/codex.py`: OpenAI Codex provider (GPT-6 Luna, GPT-4o, GPT-4o Mini, reasoning effort control).
    - `plugins/claude_code.py`: Anthropic Claude Code provider (Claude 3.7 Sonnet, Claude 3.5 Sonnet, Claude 3.5 Haiku).
    - `plugins/openai_compatible.py`: Generic proxy for local Ollama, vLLM, LMStudio, etc.
- **Full Provider Customization**:
  - Add, edit, or remove custom model aliases and parameter mappings.
  - Custom variables per plugin (API keys, base URLs, options).
  - In-browser code editor with AST syntax checking and instant hot-reload.
- **OpenAI-Compatible Local API**:
  - `POST /v1/chat/completions`: Supports standard JSON and real-time streaming Server-Sent Events (`stream: true`).
  - `GET /v1/models` and `GET /v1/models/{model_id}`: Dynamic catalog of all enabled models.
- **Minimalist Black & White Dashboard**:
  - Obsidian/monochrome high-contrast design.
  - Zero external CDN dependencies (works completely offline).
  - Integrated playground for testing chat models with streaming output.
  - Real-time telemetry, latency metrics, and request logs.

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

### Using `curl`

#### List Models
```bash
curl http://localhost:8000/v1/models
```

#### Chat Completion (Non-Streaming)
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-3.8-flash",
    "messages": [
      {"role": "user", "content": "Hello LLMProxy!"}
    ]
  }'
```

#### Chat Completion (Streaming SSE)
```bash
curl -N -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
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

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed-locally"  # Provider API keys are configured in LLMProxy dashboard
)

response = client.chat.completions.create(
    model="claude-3-7-sonnet",
    messages=[{"role": "user", "content": "Hello from OpenAI Python SDK!"}],
    stream=True
)

for chunk in response:
    content = chunk.choices[0].delta.content or ""
    print(content, end="", flush=True)
print()
```
