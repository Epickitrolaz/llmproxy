# Testing LLMProxy Plugins

## Run the suite

```bash
pytest -q --ignore=tests/test_codex_and_example_response.py
```

The two live Codex tests in `tests/test_codex_and_example_response.py` need a real
local Codex session and network access. They fail on a clean checkout; do not treat
them as regressions unless your change touches Codex auth/live execution.

## Load and validate a plugin

```bash
python -m py_compile templates/my_provider.py
python skills/llmproxy-plugin/scripts/validate_plugin.py templates/my_provider.py
python skills/llmproxy-plugin/scripts/validate_plugin.py templates/my_provider.py --smoke
```

`--smoke` only works when the plugin falls back to simulation without credentials.
Do not use it for a plugin that would contact a real upstream.

## Unit-test metadata and simulation

```python
import asyncio
from pathlib import Path

from llmproxy.dashboard.loader import load_plugin_from_file
from llmproxy.schemas import ChatCompletionRequest, ChatMessage

plugin = load_plugin_from_file(Path("templates/my_provider.py").resolve())
assert plugin.metadata.id == "my_provider"

request = ChatCompletionRequest(
    model=plugin.metadata.models[0].id,
    messages=[ChatMessage(role="user", content="ping")],
)
response = asyncio.run(
    plugin.chat_completion(request, plugin.metadata.models[0], {}, account=None)
)
assert response.choices[0].message.content
```

## Mock an OpenAI-compatible upstream

```python
import httpx
import pytest

@pytest.mark.asyncio
async def test_my_provider_non_streaming(monkeypatch):
    captured = {}

    async def mock_post(self, url, headers=None, json=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json

        class Response:
            status_code = 200
            def raise_for_status(self):
                pass
            def json(self):
                return {
                    "id": "chatcmpl-test",
                    "model": "my-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "OK"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
                }

        return Response()

    monkeypatch.setattr("httpx.AsyncClient.post", mock_post)
    # Call the plugin and assert captured["url"], captured["headers"], captured["json"].
```

For more realistic tests, use `httpx.MockTransport` and inject a client, or
monkeypatch `httpx.AsyncClient.post`/`.get` as the existing tests do.

## Test streaming

```python
@pytest.mark.asyncio
async def test_my_provider_streaming(monkeypatch):
    async def fake_stream(*args, **kwargs):
        yield 'data: {"id":"chatcmpl-test","model":"my-model","choices":[{"index":0,"delta":{"content":"OK"}}]}\n\n'
        yield "data: [DONE]\n\n"

    # Monkeypatch the plugin's internal stream helper, or mock httpx.AsyncClient.stream.
    result = await plugin.chat_completion(stream_request, model_config, {}, account=None)
    chunks = [chunk async for chunk in result]
    assert chunks
```

If you mock `httpx.AsyncClient.stream`, provide an async context manager whose
`aiter_lines()` yields the SSE lines.

## Test through the API

```python
from fastapi.testclient import TestClient
from llmproxy.main import app

client = TestClient(app)

response = client.post(
    "/v1/chat/completions",
    json={
        "model": "my-model",
        "messages": [{"role": "user", "content": "ping"}],
    },
)
assert response.status_code == 200
```

Clear local API keys in the test fixture if the global data directory has keys,
otherwise requests return 401.

## Test account failover

Add accounts for the provider and monkeypatch the plugin's `chat_completion` to
fail the first account and succeed the second. Assert the request returns 200 and
both account IDs were attempted. See `tests/test_issue_36_fixes.py` for a full
example.

## Manual smoke test

Start the server:

```bash
uvicorn llmproxy.main:app --host 0.0.0.0 --port 8000 --reload
```

List models:

```bash
curl http://localhost:8000/v1/models
```

Non-streaming:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"my-model","messages":[{"role":"user","content":"hello"}]}'
```

Streaming:

```bash
curl -N -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"my-model","messages":[{"role":"user","content":"hello"}],"stream":true}'
```

If local API keys exist, add `-H "Authorization: Bearer <key>"`.

## Dashboard checks

1. Open `http://localhost:8000`.
2. Providers & Plugins should list the plugin.
3. The Models tab should show the default models.
4. The Accounts & Pool tab should accept credentials.
5. The Playground should return a response and stream tokens.
6. The Status tab should show request logs and account status.

## Common test failures

- `401 missing_api_key` — the test data directory has local API keys. Clear them
  in the fixture or pass a valid key.
- Plugin not found — the test loaded a template path that is not in `plugins/`.
  `tests/conftest.py` copies templates into `plugins/` if missing.
- Network timeout — the plugin did not enter simulation mode. Use a test key such
  as `test-key` or pass `account=None` with an empty `api_key` variable.
- `RuntimeError: no running event loop` — an async generator or coroutine was
  created outside `asyncio.run`/pytest-asyncio.
