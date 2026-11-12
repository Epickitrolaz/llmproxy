"""Security regression tests for the dashboard API (issue #50)."""

from __future__ import annotations

from fastapi.testclient import TestClient
from fastapi.middleware.cors import CORSMiddleware

from llmproxy.core.manager import plugin_manager
from llmproxy.core.security import (
    ADMIN_TOKEN_ENV,
    cors_allow_credentials,
    cors_origins,
    get_admin_token,
)
from llmproxy.main import app
from llmproxy.schemas import Account, APIKey

ADMIN_HEADERS = {"X-Admin-Token": "test-admin-token"}

SENSITIVE_PATHS = [
    ("GET", "/api/accounts"),
    ("GET", "/api/keys"),
    ("GET", "/api/management"),
    ("GET", "/api/plugins"),
    ("GET", "/api/stats"),
    ("GET", "/api/logs"),
    ("POST", "/api/plugins"),
    ("POST", "/api/test-chat"),
]


def test_dashboard_api_rejects_unauthenticated_requests():
    client = TestClient(app)

    for method, path in SENSITIVE_PATHS:
        response = client.request(method, path, json={"model": "test-model", "prompt": "hi"})
        assert response.status_code == 401, (method, path, response.text)
        assert response.json()["detail"]["error"] in {
            "admin_auth_required",
            "admin_token_not_configured",
        }


def test_plugin_upload_and_code_routes_require_admin():
    client = TestClient(app)

    response = client.post(
        "/api/plugins",
        json={"filename": "evil.py", "code": "print('owned')"},
    )
    assert response.status_code == 401
    assert not (plugin_manager.plugins_dir / "evil.py").exists()

    response = client.put(
        "/api/plugins/anything/code",
        json={"code": "print('owned')"},
    )
    assert response.status_code == 401


def test_valid_admin_token_unlocks_dashboard():
    client = TestClient(app, headers=ADMIN_HEADERS)

    assert client.get("/api/plugins").status_code == 200
    assert client.get("/api/accounts").status_code == 200
    assert client.get("/api/keys").status_code == 200
    assert client.get("/api/stats").status_code == 200
    assert client.get("/api/auth/status").json()["authenticated"] is True


def test_auth_status_is_public_and_never_returns_the_token():
    client = TestClient(app)

    response = client.get("/api/auth/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["authenticated"] is False
    assert payload["required"] is True
    assert get_admin_token() not in response.text

    authorized = client.get("/api/auth/status", headers=ADMIN_HEADERS).json()
    assert authorized["authenticated"] is True


def test_remote_requests_fail_closed_without_configured_token(monkeypatch, tmp_path):
    monkeypatch.setattr(plugin_manager, "data_dir", tmp_path)
    monkeypatch.delenv(ADMIN_TOKEN_ENV, raising=False)

    client = TestClient(app, client=("203.0.113.9", 54321))
    response = client.get("/api/accounts")
    assert response.status_code == 401
    assert response.json()["detail"]["error"] == "admin_token_not_configured"


def test_account_and_key_lists_redact_credentials(monkeypatch, tmp_path):
    monkeypatch.setattr(plugin_manager, "data_dir", tmp_path)
    monkeypatch.setattr(plugin_manager, "accounts", {})
    monkeypatch.setattr(plugin_manager, "api_keys", {})

    account = Account(
        provider_id="codex",
        name="Primary",
        api_key="sk-account-secret-1234",
        extra_config={
            "refresh_token": "refresh-secret-9999",
            "access_token": "access-secret-8888",
            "email": "user@example.com",
            "plan_type": "pro",
            "tokens": {"id_token": "nested-secret-7777"},
        },
    )
    plugin_manager.accounts[account.id] = account

    key = APIKey(name="Production", key="sk-proxy-secret-5678")
    plugin_manager.api_keys[key.id] = key

    client = TestClient(app, headers=ADMIN_HEADERS)

    accounts = client.get("/api/accounts").json()
    assert len(accounts) == 1
    listed = accounts[0]
    assert listed["api_key"].endswith("1234")
    assert "sk-account-secret" not in listed["api_key"]
    assert listed["extra_config"]["refresh_token"].endswith("9999")
    assert "refresh-secret" not in listed["extra_config"]["refresh_token"]
    assert listed["extra_config"]["access_token"].endswith("8888")
    assert listed["extra_config"]["tokens"]["id_token"].endswith("7777")
    assert listed["extra_config"]["email"] == "user@example.com"
    assert listed["extra_config"]["plan_type"] == "pro"

    single = client.get(f"/api/accounts/{account.id}").json()
    assert "sk-account-secret" not in single["api_key"]

    revealed = client.get(f"/api/accounts/{account.id}/reveal").json()
    assert revealed["api_key"] == "sk-account-secret-1234"
    assert revealed["extra_config"]["refresh_token"] == "refresh-secret-9999"

    keys = client.get("/api/keys").json()
    assert len(keys) == 1
    assert keys[0]["key"].endswith("5678")
    assert "sk-proxy-secret" not in keys[0]["key"]

    revealed_key = client.get(f"/api/keys/{key.id}/reveal").json()
    assert revealed_key["key"] == "sk-proxy-secret-5678"

    # Redaction must not mutate the objects held by the manager.
    assert plugin_manager.accounts[account.id].api_key == "sk-account-secret-1234"
    assert plugin_manager.api_keys[key.id].key == "sk-proxy-secret-5678"


def test_new_api_key_is_shown_once_then_masked(monkeypatch, tmp_path):
    monkeypatch.setattr(plugin_manager, "data_dir", tmp_path)
    monkeypatch.setattr(plugin_manager, "api_keys_file", tmp_path / "api_keys.json")
    monkeypatch.setattr(plugin_manager, "api_keys", {})

    client = TestClient(app, headers=ADMIN_HEADERS)
    created = client.post("/api/keys", json={"name": "CI", "key": "sk-created-secret-4321"})
    assert created.status_code == 200
    assert created.json()["key"] == "sk-created-secret-4321"

    listed = client.get("/api/keys").json()
    assert "sk-created-secret" not in listed[0]["key"]


def test_cors_never_combines_wildcard_with_credentials(monkeypatch):
    monkeypatch.setenv("LLMPROXY_CORS_ORIGINS", "*")
    wildcard = cors_origins()
    assert wildcard == ["*"]
    assert cors_allow_credentials(wildcard) is False

    monkeypatch.setenv(
        "LLMPROXY_CORS_ORIGINS",
        "https://one.example, https://two.example",
    )
    explicit = cors_origins()
    assert explicit == ["https://one.example", "https://two.example"]
    assert cors_allow_credentials(explicit) is True

    # The app created at import time must not contain a wildcard+credentials
    # CORS middleware combination.
    for middleware in app.user_middleware:
        if middleware.cls is CORSMiddleware:
            options = middleware.kwargs
            assert not (
                options.get("allow_origins") == ["*"]
                and options.get("allow_credentials") is True
            )
