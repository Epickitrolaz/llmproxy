"""Reasoning modes for custom load-balancing rules."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from llmproxy.core.manager import RULE_REASONING_OVERRIDE_KEY, plugin_manager
from llmproxy.main import app
from llmproxy.schemas import (
    ChatChoice,
    ChatCompletionResponse,
    ChatMessage,
    LoadBalancingRuleCreate,
)

ADMIN_HEADERS = {"X-Admin-Token": "test-admin-token"}
client = TestClient(app, headers=ADMIN_HEADERS)


@pytest.fixture(autouse=True)
def clean_rules():
    original_rules = dict(plugin_manager.rules)
    plugin_manager.rules.clear()
    plugin_manager._save_rules()
    yield
    plugin_manager.rules = original_rules
    plugin_manager._save_rules()


def test_rule_reasoning_fields_round_trip():
    response = client.post(
        "/api/load-balancing-rules",
        json={
            "model_id": "auto-router",
            "name": "Auto Router",
            "strategy": "fallback",
            "targets": ["cx/gpt-6-luna"],
            "manual_reasoning": False,
            "exposed_reasoning_levels": ["low", "high"],
            "reasoning_overrides": {"cx/gpt-6-luna": {"high": "max"}},
        },
    )
    assert response.status_code == 200
    rule = response.json()
    assert rule["manual_reasoning"] is False
    assert rule["exposed_reasoning_levels"] == ["low", "high"]
    assert rule["reasoning_overrides"] == {"cx/gpt-6-luna": {"high": "max"}}

    listed = client.get("/api/load-balancing-rules").json()
    stored = next(item for item in listed if item["id"] == rule["id"])
    assert stored["manual_reasoning"] is False
    assert stored["exposed_reasoning_levels"] == ["low", "high"]


def test_manual_reasoning_rule_does_not_advertise_levels():
    response = client.post(
        "/api/load-balancing-rules",
        json={
            "model_id": "manual-router",
            "name": "Manual Router",
            "targets": ["cx/gpt-6-luna:high"],
        },
    )
    assert response.status_code == 200
    assert response.json()["manual_reasoning"] is True

    card = client.get("/v1/models/manual-router").json()
    assert card["reasoning_levels"] == []


def test_auto_reasoning_rule_advertises_only_supported_levels():
    response = client.post(
        "/api/load-balancing-rules",
        json={
            "model_id": "mixed-router",
            "name": "Mixed Router",
            "strategy": "fallback",
            "targets": ["gm/gemini-4-argon", "cx/gpt-6-luna"],
            "manual_reasoning": False,
            "exposed_reasoning_levels": ["low", "high", "max"],
        },
    )
    assert response.status_code == 200

    card = client.get("/v1/models/mixed-router").json()
    # gemini-4-argon does not serve "max", so it must not be advertised.
    assert card["reasoning_levels"] == ["low", "high"]


def test_auto_reasoning_rule_can_map_a_level_to_make_it_valid():
    response = client.post(
        "/api/load-balancing-rules",
        json={
            "model_id": "mapped-router",
            "name": "Mapped Router",
            "strategy": "fallback",
            "targets": ["gm/gemini-4-argon", "cx/gpt-6-luna"],
            "manual_reasoning": False,
            "exposed_reasoning_levels": ["high", "max"],
            "reasoning_overrides": {"gm/gemini-4-argon": {"max": "high"}},
        },
    )
    assert response.status_code == 200

    card = client.get("/v1/models/mapped-router").json()
    assert card["reasoning_levels"] == ["high", "max"]


def test_auto_rule_maps_reasoning_effort_for_selected_target():
    plugin_manager.discover_and_load_all()
    rule = plugin_manager.add_load_balancing_rule(
        LoadBalancingRuleCreate(
            model_id="mapped-router",
            name="Mapped Router",
            strategy="fallback",
            targets=["cx/gpt-6-luna"],
            manual_reasoning=False,
            exposed_reasoning_levels=["high"],
            reasoning_overrides={"cx/gpt-6-luna": {"high": "max"}},
        )
    )
    assert rule.manual_reasoning is False

    resolved = plugin_manager.resolve_model("mapped-router", reasoning_effort="high")
    assert resolved is not None
    model_config, _plugin, variables = resolved
    assert model_config.id == "cx/gpt-6-luna"
    assert variables["reasoning_effort"] == "max"
    assert variables[RULE_REASONING_OVERRIDE_KEY] == "max"

    # The generic suffix spelling produces the same mapping.
    suffixed = plugin_manager.resolve_model("mapped-router:high")
    assert suffixed is not None
    assert suffixed[2]["reasoning_effort"] == "max"


def test_router_applies_mapped_level_to_request(monkeypatch):
    plugin_manager.discover_and_load_all()
    plugin = plugin_manager.get_plugin("codex")
    if plugin is None:
        pytest.skip("codex plugin is not loaded")

    captured = {}

    async def fake_chat_completion(request, model_config, variables, account=None):
        captured["reasoning_effort"] = request.reasoning_effort
        return ChatCompletionResponse(
            id="chatcmpl-rule-reasoning",
            model=model_config.id,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="OK"),
                    finish_reason="stop",
                )
            ],
        )

    monkeypatch.setattr(plugin, "chat_completion", fake_chat_completion)
    plugin_manager.add_load_balancing_rule(
        LoadBalancingRuleCreate(
            model_id="mapped-router",
            name="Mapped Router",
            strategy="fallback",
            targets=["cx/gpt-6-luna"],
            manual_reasoning=False,
            exposed_reasoning_levels=["high"],
            reasoning_overrides={"cx/gpt-6-luna": {"high": "max"}},
        )
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "mapped-router",
            "messages": [{"role": "user", "content": "ping"}],
            "reasoning_effort": "high",
        },
    )
    assert response.status_code == 200
    assert captured["reasoning_effort"] == "max"


def test_auto_rule_reads_reasoning_from_extra_body(monkeypatch):
    plugin_manager.discover_and_load_all()
    plugin = plugin_manager.get_plugin("codex")
    if plugin is None:
        pytest.skip("codex plugin is not loaded")

    captured = {}

    async def fake_chat_completion(request, model_config, variables, account=None):
        captured["reasoning_effort"] = request.reasoning_effort
        return ChatCompletionResponse(
            id="chatcmpl-rule-reasoning",
            model=model_config.id,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="OK"),
                    finish_reason="stop",
                )
            ],
        )

    monkeypatch.setattr(plugin, "chat_completion", fake_chat_completion)
    plugin_manager.add_load_balancing_rule(
        LoadBalancingRuleCreate(
            model_id="mapped-router",
            name="Mapped Router",
            strategy="fallback",
            targets=["cx/gpt-6-luna"],
            manual_reasoning=False,
            exposed_reasoning_levels=["high"],
            reasoning_overrides={"cx/gpt-6-luna": {"high": "max"}},
        )
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "mapped-router",
            "messages": [{"role": "user", "content": "ping"}],
            "extra_body": {"reasoning": {"effort": "high"}},
        },
    )
    assert response.status_code == 200
    assert captured["reasoning_effort"] == "max"


def test_rule_reasoning_config_is_normalized():
    plugin_manager.discover_and_load_all()
    rule = plugin_manager.add_load_balancing_rule(
        {
            "model_id": "normalize-router",
            "name": "Normalize Router",
            "strategy": "fallback",
            "targets": ["cx/gpt-6-luna"],
            "manual_reasoning": False,
            "exposed_reasoning_levels": ["HIGH", "high", "bogus"],
            "reasoning_overrides": {
                "cx/gpt-6-luna:high": {"HIGH": "MAX", "nope": "low"}
            },
        }
    )
    assert rule.exposed_reasoning_levels == ["high"]
    assert rule.reasoning_overrides == {"cx/gpt-6-luna": {"high": "max"}}
    assert json.dumps(rule.model_dump())  # remains serializable
