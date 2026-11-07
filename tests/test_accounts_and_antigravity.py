"""Tests for Accounts, Multi-Account Load Balancing, and 9Router Antigravity Implementation."""

import time
import pytest
from fastapi.testclient import TestClient
from llmproxy.main import app
from llmproxy.core.manager import plugin_manager
from llmproxy.schemas import Account, AccountCreate, AccountUpdate
from plugins.antigravity import AntigravityPlugin


def test_antigravity_9router_algorithms():
    """Verify 9Router IDE trajectory generation, field stripping, and telemetry headers."""
    plugin = AntigravityPlugin()

    # 1. Deterministic SHA-256 UUID
    uuid_str = plugin.generate_sha256_uuid("test-session-seed-123")
    assert len(uuid_str) == 36
    assert uuid_str.count("-") == 4
    # Deterministic check
    assert uuid_str == plugin.generate_sha256_uuid("test-session-seed-123")

    # 2. 9Router Trajectory ID format: agent/<conversation_id>/<timestamp>/<trajectory_id>/<step>
    traj_id = plugin.generate_trajectory_id(
        conversation_seed="conv-1",
        trajectory_seed="traj-1",
        step=3,
        timestamp_ms=1700000000000,
    )
    assert traj_id.startswith("agent/")
    parts = traj_id.split("/")
    assert len(parts) == 5
    assert parts[0] == "agent"
    assert parts[2] == "1700000000000"
    assert parts[4] == "3"

    # 3. IDE Request ID format
    ide_req_id = plugin.generate_ide_request_id(seed="req-seed")
    assert ide_req_id.startswith("ide_req_")
    assert len(ide_req_id) > 10

    # 4. Strict field sanitization of Google-rejected fields
    dirty_payload = {
        "model": "gemini-3.8-flash-high",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.7,
        # Google-rejected parameters
        "thinking": True,
        "thinkingConfig": {"thinkingBudget": 1000},
        "reasoning_effort": "high",
        "enable_thinking": True,
        "frequency_penalty": 0.5,
        "presence_penalty": 0.5,
        "logprobs": True,
        "user": "test-user",
    }
    cleaned = plugin.sanitize_payload(dirty_payload)
    assert "thinking" not in cleaned
    assert "thinkingConfig" not in cleaned
    assert "reasoning_effort" not in cleaned
    assert "enable_thinking" not in cleaned
    assert "frequency_penalty" not in cleaned
    assert "presence_penalty" not in cleaned
    assert "logprobs" not in cleaned
    assert "user" not in cleaned
    assert cleaned["model"] == "gemini-3.8-flash-high"
    assert cleaned["temperature"] == 0.7

    # 5. Header injection
    account = Account(
        id="acc_test",
        provider_id="antigravity",
        name="9Router Test Account",
        api_key="ag-token-xyz",
        extra_config={"session_id": "session-456", "step": 2},
    )
    headers = plugin.build_headers(account=account)
    assert "antigravity" in headers["User-Agent"].lower()
    assert headers["Authorization"] == "Bearer ag-token-xyz"
    assert headers["Content-Type"] == "application/json"


def test_accounts_crud_and_load_balancing():
    """Verify multi-account management, round-robin, least-used, fallback, and cooldown."""
    plugin_manager.accounts.clear()

    # 1. Add accounts
    acc1 = plugin_manager.add_account(
        AccountCreate(
            provider_id="antigravity",
            name="Antigravity Account 1",
            api_key="key-1",
            priority=1,
            weight=1,
        )
    )
    acc2 = plugin_manager.add_account(
        AccountCreate(
            provider_id="antigravity",
            name="Antigravity Account 2",
            api_key="key-2",
            priority=2,
            weight=1,
        )
    )
    acc3 = plugin_manager.add_account(
        AccountCreate(
            provider_id="antigravity",
            name="Antigravity Account 3",
            api_key="key-3",
            priority=3,
            weight=1,
        )
    )

    all_ag_accounts = plugin_manager.get_accounts("antigravity")
    assert len(all_ag_accounts) == 3

    # 2. Test Round Robin
    plugin_manager.rr_indices["antigravity"] = 0
    pick1 = plugin_manager.select_account("antigravity", strategy="round_robin")
    pick2 = plugin_manager.select_account("antigravity", strategy="round_robin")
    pick3 = plugin_manager.select_account("antigravity", strategy="round_robin")
    pick4 = plugin_manager.select_account("antigravity", strategy="round_robin")

    assert pick1.id == acc1.id
    assert pick2.id == acc2.id
    assert pick3.id == acc3.id
    assert pick4.id == acc1.id  # cycles back to 1

    # 3. Test Least Used
    plugin_manager.mark_account_result(acc1.id, success=True)
    plugin_manager.mark_account_result(acc1.id, success=True)
    plugin_manager.mark_account_result(acc2.id, success=True)
    # acc3 has 0 success_count, so least_used must pick acc3
    least = plugin_manager.select_account("antigravity", strategy="least_used")
    assert least.id == acc3.id

    # 4. Test Fallback / Priority
    # acc1 has priority 1 (highest), acc2 has 2, acc3 has 3
    priority_pick = plugin_manager.select_account("antigravity", strategy="fallback")
    assert priority_pick.id == acc1.id

    # 5. Cooldown / Rate Limit Test
    # Mark acc1 as rate-limited for 60 seconds
    plugin_manager.mark_account_result(acc1.id, success=False, is_rate_limit=True, cooldown_seconds=60)
    assert not acc1.is_available()

    # Now priority pick should fallback to acc2!
    fallback_pick = plugin_manager.select_account("antigravity", strategy="fallback")
    assert fallback_pick.id == acc2.id

    # Reset cooldown on acc1
    plugin_manager.reset_account_cooldown(acc1.id)
    assert acc1.is_available()
    priority_restored = plugin_manager.select_account("antigravity", strategy="fallback")
    assert priority_restored.id == acc1.id

    # Cleanup
    plugin_manager.delete_account(acc1.id)
    plugin_manager.delete_account(acc2.id)
    plugin_manager.delete_account(acc3.id)
    assert len(plugin_manager.get_accounts("antigravity")) == 0


def test_accounts_api_endpoints():
    """Verify REST API /api/accounts CRUD endpoints."""
    with TestClient(app) as client:
        # Create account
        create_res = client.post(
            "/api/accounts",
            json={
                "provider_id": "codex",
                "name": "Primary Codex Account",
                "api_key": "sk-codex-test-key-123",
                "base_url": "https://api.openai.com/v1",
                "priority": 1,
            },
        )
        assert create_res.status_code == 200
        acc_data = create_res.json()
        account_id = acc_data["id"]
        assert acc_data["name"] == "Primary Codex Account"
        assert acc_data["provider_id"] == "codex"

        # List accounts
        list_res = client.get("/api/accounts?provider_id=codex")
        assert list_res.status_code == 200
        assert any(a["id"] == account_id for a in list_res.json())

        # Update account
        upd_res = client.put(
            f"/api/accounts/{account_id}",
            json={"name": "Renamed Codex Account", "priority": 2},
        )
        assert upd_res.status_code == 200
        assert upd_res.json()["name"] == "Renamed Codex Account"
        assert upd_res.json()["priority"] == 2

        # Reset cooldown endpoint
        cooldown_res = client.post(f"/api/accounts/{account_id}/reset-cooldown")
        assert cooldown_res.status_code == 200

        # Delete account
        del_res = client.delete(f"/api/accounts/{account_id}")
        assert del_res.status_code == 200

        # Check not found
        get_res = client.get(f"/api/accounts/{account_id}")
        assert get_res.status_code == 404


def test_antigravity_oauth_start_and_status():
    """Verify Google Antigravity OAuth start, loopback redirect URL, and status endpoints."""
    with TestClient(app) as client:
        start_res = client.post("/api/antigravity/oauth/start")
        assert start_res.status_code == 200
        data = start_res.json()
        assert "flow_id" in data
        assert "auth_url" in data
        assert "accounts.google.com" in data["auth_url"]
        assert "1071006060591" in data["auth_url"]
        assert "127.0.0.1" in data["redirect_uri"]

        # Check status initially is waiting
        status_res = client.get(f"/api/antigravity/oauth/status/{data['flow_id']}")
        assert status_res.status_code == 200
        assert status_res.json()["status"] == "waiting"


def test_antigravity_import_local_and_live_quota():
    """Verify local Antigravity session auto-discovery and live CloudCode PA quota retrieval."""
    with TestClient(app) as client:
        import_res = client.post("/api/antigravity/oauth/import-local")
        # May be 200 if acp_token is present on host
        if import_res.status_code == 200:
            acc_data = import_res.json()["account"]
            assert acc_data["provider_id"] == "antigravity"
            assert "Google" in acc_data["name"]
            acc_id = acc_data["id"]

            # Test quota endpoint
            quota_res = client.get(f"/api/antigravity/quota/{acc_id}")
            assert quota_res.status_code == 200
            quota_data = quota_res.json()
            assert "groups" in quota_data or "buckets" in quota_data


def test_sanitize_ssl_env_prevents_file_not_found_error(monkeypatch):
    """Verify that invalid SSL_CERT_FILE or CA bundle paths are cleared and don't cause [Errno 2] No such file or directory."""
    import os
    import httpx
    from llmproxy.core.antigravity_oauth import sanitize_ssl_env

    # Simulate environment where SSL_CERT_FILE was set to a non-existent path
    monkeypatch.setenv("SSL_CERT_FILE", "/nonexistent/data/ca.crt")
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", "/nonexistent/bundle.crt")
    monkeypatch.setenv("SSL_CERT_DIR", "/nonexistent/ssl/certs")

    # Before sanitization, httpx.AsyncClient() fails with FileNotFoundError: [Errno 2] No such file or directory
    with pytest.raises(FileNotFoundError):
        httpx.AsyncClient()

    # Call sanitize_ssl_env
    sanitize_ssl_env()

    assert "SSL_CERT_FILE" not in os.environ
    assert "REQUESTS_CA_BUNDLE" not in os.environ
    assert "SSL_CERT_DIR" not in os.environ

    # Now httpx.AsyncClient() succeeds without error
    client = httpx.AsyncClient()
    assert client is not None

