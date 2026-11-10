"""
Tests for External Proxy Support across all Providers:
- Off by default across all providers
- Random routing across proxy pool
- Routing to specific proxies only (by ID, URL, name, or tag)
- Sequential / round-robin proxy routing
- Switching proxies when rate limit (429) is reached
- Routing dedicated accounts through a proxy always
- Proxy CRUD endpoints in dashboard API (/api/proxies)
- Support across all built-in provider templates
"""

import asyncio
import time
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import httpx
from fastapi.testclient import TestClient

from llmproxy.main import app
from llmproxy.core.manager import PluginManager, plugin_manager
from llmproxy.schemas import (
    Account,
    AccountCreate,
    ChatMessage,
    ChatCompletionRequest,
    PluginModelConfig,
    Proxy,
    ProxyCreate,
    ProxyUpdate,
)


@pytest.fixture(autouse=True)
def clean_proxy_pool(tmp_path, monkeypatch):
    """Ensure clean proxy pool before each test."""
    temp_data = tmp_path / "data"
    temp_data.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(plugin_manager, "data_dir", temp_data)
    monkeypatch.setattr(plugin_manager, "proxies_file", temp_data / "proxies.json")
    monkeypatch.setattr(plugin_manager, "proxies", {})
    monkeypatch.setattr(plugin_manager, "proxy_rr_indices", {})


def test_proxy_crud_and_dashboard_endpoints():
    """Verify proxy pool CRUD via API endpoints."""
    client = TestClient(app)

    # 1. Initially empty
    res = client.get("/api/proxies")
    assert res.status_code == 200
    assert res.json() == []

    # 2. Create proxy 1
    p1 = client.post(
        "/api/proxies",
        json={"name": "US Residential Proxy", "url": "http://user:pass@10.1.1.1:8080", "tags": ["us", "fast"]},
    )
    assert p1.status_code == 200
    d1 = p1.json()
    assert d1["name"] == "US Residential Proxy"
    assert d1["url"] == "http://user:pass@10.1.1.1:8080"
    assert "us" in d1["tags"]
    p1_id = d1["id"]

    # 3. Create proxy 2
    p2 = client.post(
        "/api/proxies",
        json={"name": "EU SOCKS5 Proxy", "url": "socks5://10.2.2.2:1080", "tags": ["eu"]},
    )
    assert p2.status_code == 200
    p2_id = p2.json()["id"]

    # 4. List proxies & filter by tag
    all_proxies = client.get("/api/proxies").json()
    assert len(all_proxies) == 2

    eu_proxies = client.get("/api/proxies?tag=eu").json()
    assert len(eu_proxies) == 1
    assert eu_proxies[0]["id"] == p2_id

    # 5. Update proxy
    upd = client.put(f"/api/proxies/{p1_id}", json={"enabled": False})
    assert upd.status_code == 200
    assert upd.json()["enabled"] is False

    # 6. Reset cooldown
    reset = client.post(f"/api/proxies/{p1_id}/reset-cooldown")
    assert reset.status_code == 200
    assert reset.json()["status"] == "ok"

    # 7. Delete proxy
    delete_res = client.delete(f"/api/proxies/{p2_id}")
    assert delete_res.status_code == 200
    assert len(client.get("/api/proxies").json()) == 1


def test_proxies_off_by_default_on_all_providers():
    """Verify that external proxy routing is off by default across all provider templates."""
    templates = client_get_templates = TestClient(app).get("/api/templates").json()
    for tpl in templates:
        p_id = tpl["id"]
        plugin = plugin_manager.get_plugin(p_id)
        if plugin:
            var_names = {v.name for v in plugin.metadata.variables}
            assert "proxy_enabled" in var_names, f"Provider '{p_id}' missing proxy_enabled"
            assert "proxy_strategy" in var_names, f"Provider '{p_id}' missing proxy_strategy"
            var_dict = {v.name: v for v in plugin.metadata.variables}
            # Off by default
            assert var_dict["proxy_enabled"].default is False, f"Provider '{p_id}' proxy_enabled should default to False"

    # When proxy_enabled is False, resolve_request_proxy returns (None, None)
    proxy_url, proxy_obj = plugin_manager.resolve_request_proxy(variables={"proxy_enabled": False})
    assert proxy_url is None
    assert proxy_obj is None


def test_sequential_proxy_routing():
    """Verify sequential round-robin routing distributes requests across active proxies."""
    plugin_manager.add_proxy(ProxyCreate(name="P1", url="http://proxy1:8080"))
    plugin_manager.add_proxy(ProxyCreate(name="P2", url="http://proxy2:8080"))
    plugin_manager.add_proxy(ProxyCreate(name="P3", url="http://proxy3:8080"))

    vars_dict = {"proxy_enabled": True, "proxy_strategy": "sequential"}

    # Sequentially cycles P1 -> P2 -> P3 -> P1
    u1, _ = plugin_manager.resolve_request_proxy(vars_dict, provider_id="test_prov")
    u2, _ = plugin_manager.resolve_request_proxy(vars_dict, provider_id="test_prov")
    u3, _ = plugin_manager.resolve_request_proxy(vars_dict, provider_id="test_prov")
    u4, _ = plugin_manager.resolve_request_proxy(vars_dict, provider_id="test_prov")

    assert u1 == "http://proxy1:8080"
    assert u2 == "http://proxy2:8080"
    assert u3 == "http://proxy3:8080"
    assert u4 == "http://proxy1:8080"


def test_random_proxy_routing():
    """Verify random proxy routing selects from available proxies."""
    plugin_manager.add_proxy(ProxyCreate(name="P1", url="http://proxy1:8080"))
    plugin_manager.add_proxy(ProxyCreate(name="P2", url="http://proxy2:8080"))

    vars_dict = {"proxy_enabled": True, "proxy_strategy": "random"}
    urls_seen = set()
    for _ in range(20):
        url, _ = plugin_manager.resolve_request_proxy(vars_dict)
        assert url in ("http://proxy1:8080", "http://proxy2:8080")
        urls_seen.add(url)

    assert len(urls_seen) > 0


def test_routing_to_specific_proxies_only():
    """Verify routing requests strictly to a specific proxy by ID, URL, name, or tag."""
    p_special = plugin_manager.add_proxy(
        ProxyCreate(name="Dedicated Proxy", url="http://special-proxy:8080", tags=["dedicated-eu"])
    )
    plugin_manager.add_proxy(ProxyCreate(name="Regular Proxy", url="http://regular-proxy:8080"))

    # 1. By ID
    vars_by_id = {
        "proxy_enabled": True,
        "proxy_strategy": "specific",
        "specific_proxy_id": p_special.id,
    }
    url1, p1 = plugin_manager.resolve_request_proxy(vars_by_id)
    assert url1 == "http://special-proxy:8080"
    assert p1.id == p_special.id

    # 2. By Tag
    vars_by_tag = {
        "proxy_enabled": True,
        "proxy_strategy": "specific",
        "specific_proxy_id": "dedicated-eu",
    }
    url2, p2 = plugin_manager.resolve_request_proxy(vars_by_tag)
    assert url2 == "http://special-proxy:8080"
    assert p2.id == p_special.id

    # 3. By Name
    vars_by_name = {
        "proxy_enabled": True,
        "proxy_strategy": "specific",
        "specific_proxy_id": "Dedicated Proxy",
    }
    url3, p3 = plugin_manager.resolve_request_proxy(vars_by_name)
    assert url3 == "http://special-proxy:8080"


def test_routing_some_accounts_through_proxy_always():
    """Verify accounts configured with a proxy always route through that proxy regardless of provider setting."""
    p_account = plugin_manager.add_proxy(
        ProxyCreate(name="Account Proxy", url="socks5://account-proxy:1080", tags=["acc-dedicated"])
    )

    # Provider has proxy_enabled = False (off by default)
    vars_off = {"proxy_enabled": False}

    # 1. Account with dedicated proxy_id
    acc1 = Account(
        id="acc_dedicated_1",
        provider_id="codex",
        name="Account With Proxy ID",
        proxy_id=p_account.id,
    )
    url1, p1 = plugin_manager.resolve_request_proxy(vars_off, account=acc1)
    assert url1 == "socks5://account-proxy:1080"
    assert p1.id == p_account.id

    # 2. Account with direct proxy_url
    acc2 = Account(
        id="acc_dedicated_2",
        provider_id="codex",
        name="Account With Direct URL",
        proxy_url="http://direct-account-proxy:9090",
    )
    url2, p2 = plugin_manager.resolve_request_proxy(vars_off, account=acc2)
    assert url2 == "http://direct-account-proxy:9090"

    # 3. Account with extra_config['proxy']
    acc3 = Account(
        id="acc_dedicated_3",
        provider_id="antigravity",
        name="Account With Extra Config",
        extra_config={"proxy": "socks5://extra-proxy:1080"},
    )
    url3, p3 = plugin_manager.resolve_request_proxy(vars_off, account=acc3)
    assert url3 == "socks5://extra-proxy:1080"


@pytest.mark.anyio
async def test_switching_proxies_when_limit_reached():
    """Verify that when rate limit (429) is hit, router switches to next available proxy in pool."""
    p1 = plugin_manager.add_proxy(ProxyCreate(name="Proxy 1", url="http://proxy1:8080"))
    p2 = plugin_manager.add_proxy(ProxyCreate(name="Proxy 2", url="http://proxy2:8080"))

    vars_dict = {
        "proxy_enabled": True,
        "proxy_strategy": "switch_on_limit",
        "max_retries": 3,
        "account_switch_wait": 0.0,
    }

    # Step 1: Initial resolution gets p1
    url1, current_p = plugin_manager.resolve_request_proxy(vars_dict)
    assert url1 == "http://proxy1:8080"
    assert current_p.id == p1.id

    # Step 2: Rate limit hit on p1 -> mark p1 rate limited
    plugin_manager.mark_proxy_result(p1.id, success=False, is_rate_limit=True, cooldown_seconds=300)
    assert not p1.is_available()

    # Step 3: Next resolution automatically switches to p2
    url2, next_p = plugin_manager.resolve_request_proxy(vars_dict, exclude_proxy_ids={p1.id})
    assert url2 == "http://proxy2:8080"
    assert next_p.id == p2.id
    assert next_p.is_available()

    # Step 4: Success on p2 clears error state
    plugin_manager.mark_proxy_result(p2.id, success=True)
    assert p2.success_count == 1
