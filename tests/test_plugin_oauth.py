"""Tests for the generic plugin-owned OAuth dashboard flow."""

from __future__ import annotations

from typing import Any, Dict, Optional

import pytest
from fastapi.testclient import TestClient

from llmproxy.core.manager import plugin_manager
from llmproxy.main import app
from llmproxy.schemas import Account, PluginMetadata

ADMIN_HEADERS = {"X-Admin-Token": "test-admin-token"}


class FakeOAuthPlugin:
    metadata = PluginMetadata(
        id="fake_oauth",
        name="Fake OAuth",
        oauth_login=True,
        oauth_local_import=True,
        oauth_hint="Fake provider sign-in",
    )

    async def oauth_start(self) -> Dict[str, Any]:
        return {
            "flow_id": "flow_fake",
            "mode": "device",
            "verification_uri_complete": "https://example.test/device?code=ABCD",
            "user_code": "ABCD",
            "expires_in": 600,
            "interval": 5,
            "supports_callback_paste": False,
        }

    async def oauth_status(self, flow_id: str) -> Dict[str, Any]:
        return {
            "status": "complete",
            "account": Account(
                id="",
                provider_id="fake_oauth",
                name="Fake Account",
                api_key="token",
                extra_config={"refresh_token": "rt"},
            ),
        }

    async def oauth_callback(self, flow_id: str, callback_url: str) -> Dict[str, Any]:
        return {
            "status": "complete",
            "account": Account(
                id="",
                provider_id="fake_oauth",
                name="Fake Callback",
                api_key="token",
                extra_config={},
            ),
        }

    async def oauth_import_local(self) -> Optional[Account]:
        return Account(
            id="",
            provider_id="fake_oauth",
            name="Fake Local",
            api_key="token",
            extra_config={},
        )

    async def refresh_account(self, account: Account) -> bool:
        return True


@pytest.fixture()
def fake_plugin(monkeypatch):
    plugin = FakeOAuthPlugin()
    monkeypatch.setitem(plugin_manager.plugins, "fake_oauth", plugin)
    monkeypatch.setattr(plugin_manager, "get_accounts", lambda provider_id=None: [])

    def add_account(account):
        return account

    monkeypatch.setattr(plugin_manager, "add_account", add_account)
    return plugin


def test_oauth_start_returns_flow(fake_plugin):
    client = TestClient(app, headers=ADMIN_HEADERS)
    response = client.post("/api/plugins/fake_oauth/oauth/start")
    assert response.status_code == 200
    data = response.json()
    assert data["flow_id"] == "flow_fake"
    assert data["user_code"] == "ABCD"
    assert data["mode"] == "device"


def test_oauth_status_saves_account(fake_plugin):
    client = TestClient(app, headers=ADMIN_HEADERS)
    response = client.get("/api/plugins/fake_oauth/oauth/status/flow_fake")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "complete"
    assert data["account"]["provider_id"] == "fake_oauth"
    assert data["account"]["name"] == "Fake Account"


def test_oauth_callback_completes(fake_plugin):
    client = TestClient(app, headers=ADMIN_HEADERS)
    response = client.post(
        "/api/plugins/fake_oauth/oauth/callback",
        json={"flow_id": "flow_fake", "callback_url": "http://127.0.0.1/cb?code=x"},
    )
    assert response.status_code == 200
    assert response.json()["account"]["name"] == "Fake Callback"


def test_oauth_import_local(fake_plugin):
    client = TestClient(app, headers=ADMIN_HEADERS)
    response = client.post("/api/plugins/fake_oauth/oauth/import-local")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["account"]["name"] == "Fake Local"


def test_oauth_endpoints_require_admin(fake_plugin):
    client = TestClient(app)
    assert client.post("/api/plugins/fake_oauth/oauth/start").status_code == 401


def test_non_oauth_plugin_is_rejected(fake_plugin):
    client = TestClient(app, headers=ADMIN_HEADERS)
    response = client.post("/api/plugins/openrouter/oauth/start")
    assert response.status_code == 400


def test_save_oauth_account_updates_same_name(monkeypatch):
    from llmproxy.dashboard.router import _save_oauth_account

    existing = Account(id="acc_existing", provider_id="nous_oauth", name="Nous Free Tier")
    new = Account(
        id="acc_new",
        provider_id="nous_oauth",
        name="Nous Free Tier",
        api_key="",
        extra_config={"refresh_token": "rt_new"},
    )
    captured = {}

    monkeypatch.setattr(plugin_manager, "get_account", lambda account_id: None)
    monkeypatch.setattr(plugin_manager, "get_accounts", lambda provider_id=None: [existing])

    def update(account_id, updates):
        captured["id"] = account_id
        captured["updates"] = updates
        return existing

    monkeypatch.setattr(plugin_manager, "update_account", update)
    result = _save_oauth_account(new)
    assert result.id == "acc_existing"
    assert captured["updates"]["extra_config"]["refresh_token"] == "rt_new"


def test_oauth_templates_declare_login_support():
    from templates.antigravity import AntigravityPlugin
    from templates.codex import CodexPlugin
    from templates.nous_oauth import NousOAuthPlugin

    for plugin in (AntigravityPlugin(), CodexPlugin(), NousOAuthPlugin()):
        assert plugin.metadata.oauth_login is True, plugin.metadata.id
        for method in ("oauth_start", "oauth_status", "oauth_callback", "oauth_import_local", "refresh_account"):
            assert callable(getattr(plugin, method, None)), (plugin.metadata.id, method)
