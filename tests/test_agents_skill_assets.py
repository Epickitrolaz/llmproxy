"""Tests for the repository agent guide and the LLMProxy plugin skill assets."""

import asyncio
import subprocess
import sys
from pathlib import Path

from llmproxy.dashboard.loader import load_plugin_from_file
from llmproxy.schemas import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
)

ROOT = Path(__file__).resolve().parent.parent
SKILL_DIR = ROOT / "skills" / "llmproxy-plugin"
SKELETON = SKILL_DIR / "assets" / "plugin_skeleton.py"


def test_agents_guide_covers_core_workflows():
    content = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "## Quick commands" in content
    assert "## Repository map" in content
    assert "## Plugin contract in one screen" in content
    assert "## Testing" in content
    assert "## Definition of done" in content
    assert "skills/llmproxy-plugin/SKILL.md" in content


def test_plugin_skill_has_entrypoint_and_references():
    skill = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    assert skill.startswith("---\n")
    assert "name: llmproxy-plugin" in skill
    assert "description:" in skill
    for reference in ("plugin-contract.md", "patterns.md", "testing.md"):
        assert (SKILL_DIR / "references" / reference).exists()
    assert (ROOT / "skills" / "README.md").exists()


def test_plugin_skeleton_loads_and_simulates():
    plugin = load_plugin_from_file(SKELETON.resolve())
    assert plugin.metadata.id == "my_provider"
    assert plugin.metadata.models

    model = plugin.metadata.models[0]
    request = ChatCompletionRequest(
        model=model.id,
        messages=[ChatMessage(role="user", content="ping")],
    )
    response = asyncio.run(plugin.chat_completion(request, model, {}, account=None))
    assert isinstance(response, ChatCompletionResponse)
    assert response.choices
    assert response.choices[0].message.content

    stream_request = ChatCompletionRequest(
        model=model.id,
        messages=[ChatMessage(role="user", content="ping")],
        stream=True,
    )

    async def collect():
        stream = await plugin.chat_completion(stream_request, model, {}, account=None)
        return [chunk async for chunk in stream]

    chunks = asyncio.run(collect())
    assert chunks
    assert all(isinstance(chunk, ChatCompletionChunk) for chunk in chunks)
    assert chunks[-1].choices[0].finish_reason == "stop"


def test_plugin_validator_accepts_skeleton():
    result = subprocess.run(
        [
            sys.executable,
            str(SKILL_DIR / "scripts" / "validate_plugin.py"),
            str(SKELETON),
            "--smoke",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Validation passed" in result.stdout
