from __future__ import annotations

import atexit
import importlib.util
import json
import logging
import os
import random
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from llmproxy.core.interface import BasePlugin
from llmproxy.schemas import (
    Account,
    AccountCreate,
    AccountUpdate,
    APIKey,
    APIKeyCreate,
    APIKeyUpdate,
    LoadBalancingRule,
    LoadBalancingRuleCreate,
    LoadBalancingRuleUpdate,
    PluginMetadata,
    PluginModelConfig,
    PluginVariable,
    Proxy,
    ProxyCreate,
    ProxyUpdate,
    RequestLogEntry,
)

logger = logging.getLogger(__name__)


class PluginManager:
    """
    Manages provider plugins loaded dynamically from the plugins/ directory.
    Contract specification for plugin discovery, persistence, model management,
    and routing.
    """

    def __init__(self, plugins_dir: Optional[Path] = None, data_dir: Optional[Path] = None):
        self.plugins_dir = Path(plugins_dir) if plugins_dir else (Path(__file__).resolve().parent.parent.parent / "plugins")
        self.data_dir = Path(data_dir) if data_dir else (Path(__file__).resolve().parent.parent.parent / "data")
        self.plugins_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.data_dir / "plugin_configs.json"
        self.accounts_file = self.data_dir / "accounts.json"
        self.api_keys_file = self.data_dir / "api_keys.json"
        self.rules_file = self.data_dir / "load_balancing_rules.json"
        self.logs_file = self.data_dir / "request_logs.json"
        self.proxies_file = self.data_dir / "proxies.json"

        self.plugins: Dict[str, BasePlugin] = {}
        self.plugin_files: Dict[str, Path] = {}
        self.configs: Dict[str, Dict[str, Any]] = {}
        self.accounts: Dict[str, Account] = {}
        self.api_keys: Dict[str, APIKey] = {}
        self.rules: Dict[str, LoadBalancingRule] = {}
        self.proxies: Dict[str, Proxy] = {}
        self.rr_indices: Dict[str, int] = {}
        self.rule_rr_indices: Dict[str, int] = {}
        self.proxy_rr_indices: Dict[str, int] = {}
        self.logs: List[RequestLogEntry] = []

        # Write optimization state
        self._write_lock = threading.Lock()
        self._logs_dirty = False
        self._logs_timer: Optional[threading.Timer] = None
        self._api_keys_dirty = False
        self._api_keys_timer: Optional[threading.Timer] = None
        self._flush_interval: float = 3.0

        atexit.register(self.flush)

        self._load_configs()
        self._load_accounts()
        self._load_api_keys()
        self._load_rules()
        self._load_logs()
        self._load_proxies()

    def _atomic_write(self, target_file: Path, content: str) -> None:
        """Write content to target_file safely using an atomic temporary file replacement."""
        try:
            target_file.parent.mkdir(parents=True, exist_ok=True)
            temp_fd, temp_path = tempfile.mkstemp(
                dir=str(target_file.parent),
                prefix=f".{target_file.stem}_",
                suffix=".tmp",
            )
            try:
                with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
                    f.write(content)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(temp_path, target_file)
            except Exception:
                if os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except OSError:
                        pass
                raise
        except Exception as e:
            logger.error("Failed to write to %s: %s", target_file, e)

    def _load_configs(self) -> None:
        """Load persisted plugin configurations from disk."""
        if self.config_file.exists():
            try:
                content = self.config_file.read_text(encoding="utf-8")
                self.configs = json.loads(content)
            except Exception as e:
                logger.warning("Failed to load plugin configs from %s: %s", self.config_file, e)
                self.configs = {}
        else:
            self.configs = {}

    def _save_configs(self) -> None:
        """Persist current plugin configurations to disk."""
        self._atomic_write(self.config_file, json.dumps(self.configs, indent=2))

    def _load_accounts(self) -> None:
        """Load connected accounts from disk."""
        if self.accounts_file.exists():
            try:
                data = json.loads(self.accounts_file.read_text(encoding="utf-8"))
                self.accounts = {
                    item["id"]: Account.model_validate(item)
                    for item in data
                }
            except Exception as e:
                logger.warning("Failed to load accounts from %s: %s", self.accounts_file, e)
                self.accounts = {}
        else:
            self.accounts = {}

    def _save_accounts(self) -> None:
        """Persist accounts to disk."""
        serialized = [acc.model_dump() for acc in self.accounts.values()]
        self._atomic_write(self.accounts_file, json.dumps(serialized, indent=2))

    def _load_api_keys(self) -> None:
        """Load API keys from disk."""
        if self.api_keys_file.exists():
            try:
                data = json.loads(self.api_keys_file.read_text(encoding="utf-8"))
                self.api_keys = {
                    item["id"]: APIKey.model_validate(item)
                    for item in data
                }
            except Exception as e:
                logger.warning("Failed to load API keys from %s: %s", self.api_keys_file, e)
                self.api_keys = {}
        else:
            self.api_keys = {}

    def _save_api_keys(self) -> None:
        """Persist API keys to disk (excluding temporary test keys)."""
        with self._write_lock:
            self._api_keys_dirty = False
            if self._api_keys_timer:
                self._api_keys_timer.cancel()
                self._api_keys_timer = None
        serialized = [k.model_dump() for k in self.api_keys.values() if not k.is_temporary]
        self._atomic_write(self.api_keys_file, json.dumps(serialized, indent=2))

    def _schedule_api_keys_save(self) -> None:
        """Debounce API keys persistence."""
        with self._write_lock:
            self._api_keys_dirty = True
            if self._api_keys_timer is not None:
                return
            timer = threading.Timer(self._flush_interval, self._flush_api_keys)
            timer.daemon = True
            self._api_keys_timer = timer
            timer.start()

    def _flush_api_keys(self) -> None:
        """Flush pending API keys writes to disk."""
        with self._write_lock:
            if not self._api_keys_dirty:
                return
            self._api_keys_dirty = False
            if self._api_keys_timer:
                self._api_keys_timer.cancel()
                self._api_keys_timer = None
        serialized = [k.model_dump() for k in self.api_keys.values() if not k.is_temporary]
        self._atomic_write(self.api_keys_file, json.dumps(serialized, indent=2))

    def _load_rules(self) -> None:
        """Load custom load balancing rules from disk."""
        if self.rules_file.exists():
            try:
                data = json.loads(self.rules_file.read_text(encoding="utf-8"))
                self.rules = {
                    item["id"]: LoadBalancingRule.model_validate(item)
                    for item in data
                }
            except Exception as e:
                logger.warning("Failed to load rules from %s: %s", self.rules_file, e)
                self.rules = {}
        else:
            self.rules = {}

    def _save_rules(self) -> None:
        """Persist custom load balancing rules to disk."""
        serialized = [r.model_dump() for r in self.rules.values()]
        self._atomic_write(self.rules_file, json.dumps(serialized, indent=2))

    def _load_logs(self) -> None:
        """Load request logs from disk."""
        if self.logs_file.exists():
            try:
                data = json.loads(self.logs_file.read_text(encoding="utf-8"))
                self.logs = [RequestLogEntry.model_validate(item) for item in data]
            except Exception as e:
                logger.warning("Failed to load request logs from %s: %s", self.logs_file, e)
                self.logs = []
        else:
            self.logs = []

    def _save_logs(self) -> None:
        """Persist request logs to disk immediately."""
        with self._write_lock:
            self._logs_dirty = False
            if self._logs_timer:
                self._logs_timer.cancel()
                self._logs_timer = None
            logs_snapshot = [log.model_dump() for log in self.logs[:1000]]
        self._atomic_write(self.logs_file, json.dumps(logs_snapshot, indent=2))

    def _schedule_logs_save(self) -> None:
        """Debounce request logs persistence to reduce disk write frequency."""
        with self._write_lock:
            self._logs_dirty = True
            if self._logs_timer is not None:
                return
            timer = threading.Timer(self._flush_interval, self._flush_logs)
            timer.daemon = True
            self._logs_timer = timer
            timer.start()

    def _flush_logs(self) -> None:
        """Flush pending log writes to disk."""
        with self._write_lock:
            if not self._logs_dirty:
                return
            self._logs_dirty = False
            if self._logs_timer:
                self._logs_timer.cancel()
                self._logs_timer = None
            logs_snapshot = [log.model_dump() for log in self.logs[:1000]]
        self._atomic_write(self.logs_file, json.dumps(logs_snapshot, indent=2))

    def flush(self) -> None:
        """Flush any pending dirty data (logs, usage counters) to disk immediately."""
        self._flush_logs()
        self._flush_api_keys()

    def _load_proxies(self) -> None:
        """Load external proxies from disk."""
        if self.proxies_file.exists():
            try:
                data = json.loads(self.proxies_file.read_text(encoding="utf-8"))
                self.proxies = {
                    item["id"]: Proxy.model_validate(item)
                    for item in data
                }
            except Exception as e:
                logger.warning("Failed to load proxies from %s: %s", self.proxies_file, e)
                self.proxies = {}
        else:
            self.proxies = {}

    def _save_proxies(self) -> None:
        """Persist external proxies to disk."""
        serialized = [p.model_dump() for p in self.proxies.values()]
        self._atomic_write(self.proxies_file, json.dumps(serialized, indent=2))

    # ==========================================
    # Accounts Management & Load Balancing
    # ==========================================

    def get_accounts(self, provider_id: Optional[str] = None) -> List[Account]:
        """Return all accounts, optionally filtered by provider_id."""
        if provider_id:
            return [acc for acc in self.accounts.values() if acc.provider_id == provider_id]
        return list(self.accounts.values())

    def get_account(self, account_id: str) -> Optional[Account]:
        """Retrieve an account by ID."""
        return self.accounts.get(account_id)

    def add_account(self, account_data: Union[Account, AccountCreate, Dict[str, Any]]) -> Account:
        """Add a new connected account."""
        if isinstance(account_data, Account):
            account = account_data
        elif isinstance(account_data, AccountCreate):
            account = Account(
                id=f"acc_{uuid.uuid4().hex[:8]}",
                provider_id=account_data.provider_id,
                name=account_data.name,
                api_key=account_data.api_key,
                base_url=account_data.base_url,
                priority=account_data.priority,
                weight=account_data.weight,
                enabled=account_data.enabled,
                pinned=account_data.pinned,
                proxy_id=account_data.proxy_id,
                proxy_url=account_data.proxy_url,
                extra_config=account_data.extra_config,
            )
        else:
            acc_dict = dict(account_data)
            if "id" not in acc_dict:
                acc_dict["id"] = f"acc_{uuid.uuid4().hex[:8]}"
            account = Account.model_validate(acc_dict)

        self.accounts[account.id] = account
        self._save_accounts()
        return account

    # ==========================================
    # Local API Key Management & Usage Tracking
    # ==========================================

    def get_api_keys(self) -> List[APIKey]:
        """Return all local API keys."""
        return list(self.api_keys.values())

    def get_api_key(self, key_id: str) -> Optional[APIKey]:
        """Retrieve an API key by ID."""
        return self.api_keys.get(key_id)

    def get_api_key_by_secret(self, secret: str) -> Optional[APIKey]:
        """Find an API key by its key string."""
        for k in self.api_keys.values():
            if k.key == secret:
                return k
        return None

    def add_api_key(self, key_data: Union[APIKey, APIKeyCreate, Dict[str, Any]]) -> APIKey:
        """Create and persist a new local API key."""
        if isinstance(key_data, APIKey):
            key = key_data
        elif isinstance(key_data, APIKeyCreate):
            secret = key_data.key if key_data.key and key_data.key.strip() else f"sk-proxy-{uuid.uuid4().hex}"
            key = APIKey(
                name=key_data.name,
                key=secret,
                enabled=key_data.enabled,
                expires_at=key_data.expires_at,
                max_requests=key_data.max_requests,
                is_temporary=key_data.is_temporary,
            )
        else:
            k_dict = dict(key_data)
            if "key" not in k_dict or not k_dict["key"] or not str(k_dict["key"]).strip():
                k_dict["key"] = f"sk-proxy-{uuid.uuid4().hex}"
            key = APIKey.model_validate(k_dict)

        self.api_keys[key.id] = key
        self._save_api_keys()
        return key

    def get_or_create_temporary_key(self) -> APIKey:
        """Get an active temporary test API key or create a new one valid for 30 min / 10 requests."""
        now = time.time()
        for k in self.api_keys.values():
            if k.is_temporary and k.is_valid(now):
                return k

        temp_secret = f"sk-proxy-test-{uuid.uuid4().hex[:16]}"
        key = APIKey(
            name="Temporary Test Key",
            key=temp_secret,
            enabled=True,
            expires_at=now + 1800,  # 30 minutes
            max_requests=10,
            is_temporary=True,
        )
        self.api_keys[key.id] = key
        self._save_api_keys()
        return key

    def update_api_key(
        self, key_id: str, updates: Union[APIKeyUpdate, Dict[str, Any]]
    ) -> Optional[APIKey]:
        """Update an existing local API key."""
        key = self.api_keys.get(key_id)
        if not key:
            return None

        upd = updates.model_dump(exclude_unset=True) if isinstance(updates, APIKeyUpdate) else updates
        for k, v in upd.items():
            if v is not None and hasattr(key, k):
                setattr(key, k, v)

        self._save_api_keys()
        return key

    def delete_api_key(self, key_id: str) -> bool:
        """Delete an API key."""
        if key_id in self.api_keys:
            del self.api_keys[key_id]
            self._save_api_keys()
            return True
        return False

    def record_api_key_usage(
        self, key_id: str, prompt_tokens: int = 0, completion_tokens: int = 0
    ) -> None:
        """Increment usage counters and update timestamp for an API key."""
        key = self.api_keys.get(key_id)
        if not key:
            return
        key.total_requests += 1
        key.prompt_tokens += prompt_tokens
        key.completion_tokens += completion_tokens
        key.total_tokens += (prompt_tokens + completion_tokens)
        key.last_used_at = time.time()
        if key.max_requests is not None and key.total_requests >= key.max_requests:
            key.enabled = False
            self._save_api_keys()
        else:
            self._schedule_api_keys_save()

    # ==========================================
    # Custom Load Balancing Rules Management
    # ==========================================

    def get_load_balancing_rules(self) -> List[LoadBalancingRule]:
        """Return all custom load balancing rules."""
        return list(self.rules.values())

    def get_load_balancing_rule(self, rule_id: str) -> Optional[LoadBalancingRule]:
        """Get a custom load balancing rule by ID."""
        return self.rules.get(rule_id)

    def add_load_balancing_rule(
        self, rule_data: Union[LoadBalancingRule, LoadBalancingRuleCreate, Dict[str, Any]]
    ) -> LoadBalancingRule:
        """Add and persist a new custom load balancing rule."""
        if isinstance(rule_data, LoadBalancingRule):
            rule = rule_data
        elif isinstance(rule_data, LoadBalancingRuleCreate):
            rule = LoadBalancingRule(
                model_id=rule_data.model_id,
                name=rule_data.name,
                enabled=rule_data.enabled,
                strategy=rule_data.strategy,
                targets=rule_data.targets,
                description=rule_data.description or "",
            )
        else:
            r_dict = dict(rule_data)
            rule = LoadBalancingRule.model_validate(r_dict)

        self.rules[rule.id] = rule
        self._save_rules()
        return rule

    def update_load_balancing_rule(
        self, rule_id: str, updates: Union[LoadBalancingRuleUpdate, Dict[str, Any]]
    ) -> Optional[LoadBalancingRule]:
        """Update properties of a custom load balancing rule."""
        rule = self.rules.get(rule_id)
        if not rule:
            return None

        upd = updates.model_dump(exclude_unset=True) if isinstance(updates, LoadBalancingRuleUpdate) else updates
        for k, v in upd.items():
            if v is not None and hasattr(rule, k):
                setattr(rule, k, v)

        self._save_rules()
        return rule

    def delete_load_balancing_rule(self, rule_id: str) -> bool:
        """Delete a custom load balancing rule."""
        if rule_id in self.rules:
            del self.rules[rule_id]
            self._save_rules()
            return True
        return False

    def select_rule_target(self, rule: LoadBalancingRule) -> Optional[str]:
        """Select a target model ID based on the rule's load balancing strategy."""
        if not rule.targets:
            return None

        valid_targets = [t for t in rule.targets if t and t.strip()]
        if not valid_targets:
            return None

        strat = (rule.strategy or "round_robin").lower().replace("-", "_")

        if strat == "random":
            return random.choice(valid_targets)
        elif strat in ("fallback", "priority"):
            return valid_targets[0]
        elif strat == "least_used":
            counts = {t: 0 for t in valid_targets}
            for log in self.logs:
                if log.model in counts:
                    counts[log.model] += 1
            return min(valid_targets, key=lambda t: counts[t])
        else:
            # Default: round_robin
            idx = self.rule_rr_indices.get(rule.id, 0)
            selected = valid_targets[idx % len(valid_targets)]
            self.rule_rr_indices[rule.id] = (idx + 1) % len(valid_targets)
            return selected

    def update_account(
        self, account_id: str, updates: Union[AccountUpdate, Dict[str, Any]]
    ) -> Optional[Account]:
        """Update account properties."""
        account = self.accounts.get(account_id)
        if not account:
            return None

        upd = updates.model_dump(exclude_unset=True) if isinstance(updates, AccountUpdate) else updates
        for k, v in upd.items():
            if v is not None and hasattr(account, k):
                setattr(account, k, v)

        self._save_accounts()
        return account

    def delete_account(self, account_id: str) -> bool:
        """Delete an account."""
        if account_id in self.accounts:
            del self.accounts[account_id]
            self._save_accounts()
            return True
        return False

    def reset_account_cooldown(self, account_id: str) -> bool:
        """Clear rate limit / cooldown status on an account."""
        account = self.accounts.get(account_id)
        if not account:
            return False
        account.rate_limited_until = None
        self._save_accounts()
        return True

    def select_account(
        self, provider_id: str, strategy: Optional[str] = None
    ) -> Optional[Account]:
        """
        Select an available account based on load balancing strategy:
        - round_robin: cyclically cycles through active, non-rate-limited accounts
        - least_used: picks account with lowest success count
        - random: picks a random active account
        - fallback/priority: picks highest priority (lowest number) available
        """
        all_accounts = [a for a in self.accounts.values() if a.provider_id == provider_id]
        if not all_accounts:
            return None

        available = [a for a in all_accounts if a.is_available()]
        if not available:
            return None

        strat = (strategy or "round_robin").lower().replace("-", "_")

        if strat == "least_used":
            return min(available, key=lambda a: (a.success_count, a.last_used_at or 0))
        elif strat == "random":
            return random.choice(available)
        elif strat in ("fallback", "priority"):
            return sorted(available, key=lambda a: (a.priority, a.last_used_at or 0))[0]
        else:
            # Default: round_robin
            idx = self.rr_indices.get(provider_id, 0)
            selected = available[idx % len(available)]
            self.rr_indices[provider_id] = (idx + 1) % len(available)
            return selected

    def mark_account_result(
        self,
        account_id: str,
        success: bool,
        is_rate_limit: bool = False,
        cooldown_seconds: float = 60.0,
    ) -> None:
        """Record success or error/rate-limit result for an account."""
        account = self.accounts.get(account_id)
        if not account:
            return
        if success:
            account.success_count += 1
            account.last_used_at = time.time()
        else:
            account.error_count += 1
            if is_rate_limit:
                account.rate_limited_until = time.time() + cooldown_seconds
        self._save_accounts()

    # ==========================================
    # External Proxy Management & Routing
    # ==========================================

    def get_proxies(self, tag: Optional[str] = None) -> List[Proxy]:
        """Return all proxies, optionally filtered by tag."""
        if tag:
            return [p for p in self.proxies.values() if tag in p.tags]
        return list(self.proxies.values())

    def get_proxy(self, proxy_id: str) -> Optional[Proxy]:
        """Retrieve a proxy by ID."""
        return self.proxies.get(proxy_id)

    def add_proxy(self, proxy_data: Union[Proxy, ProxyCreate, Dict[str, Any]]) -> Proxy:
        """Add and persist a new external proxy."""
        if isinstance(proxy_data, Proxy):
            proxy = proxy_data
        elif isinstance(proxy_data, ProxyCreate):
            proxy = Proxy(
                id=f"proxy_{uuid.uuid4().hex[:8]}",
                name=proxy_data.name,
                url=proxy_data.url,
                enabled=proxy_data.enabled,
                tags=proxy_data.tags,
            )
        else:
            p_dict = dict(proxy_data)
            if "id" not in p_dict:
                p_dict["id"] = f"proxy_{uuid.uuid4().hex[:8]}"
            proxy = Proxy.model_validate(p_dict)

        self.proxies[proxy.id] = proxy
        self._save_proxies()
        return proxy

    def update_proxy(
        self, proxy_id: str, updates: Union[ProxyUpdate, Dict[str, Any]]
    ) -> Optional[Proxy]:
        """Update properties of a proxy."""
        proxy = self.proxies.get(proxy_id)
        if not proxy:
            return None

        upd = updates.model_dump(exclude_unset=True) if isinstance(updates, ProxyUpdate) else updates
        for k, v in upd.items():
            if v is not None and hasattr(proxy, k):
                setattr(proxy, k, v)

        self._save_proxies()
        return proxy

    def delete_proxy(self, proxy_id: str) -> bool:
        """Delete a proxy."""
        if proxy_id in self.proxies:
            del self.proxies[proxy_id]
            self._save_proxies()
            return True
        return False

    def reset_proxy_cooldown(self, proxy_id: str) -> bool:
        """Clear rate limit / cooldown status on a proxy."""
        proxy = self.proxies.get(proxy_id)
        if not proxy:
            return False
        proxy.rate_limited_until = None
        self._save_proxies()
        return True

    def mark_proxy_result(
        self,
        proxy_id: str,
        success: bool,
        is_rate_limit: bool = False,
        cooldown_seconds: float = 60.0,
    ) -> None:
        """Record success or rate-limit result for a proxy."""
        proxy = self.proxies.get(proxy_id)
        if not proxy:
            return
        if success:
            proxy.success_count += 1
            proxy.last_used_at = time.time()
        else:
            proxy.error_count += 1
            if is_rate_limit:
                proxy.rate_limited_until = time.time() + cooldown_seconds
        self._save_proxies()

    def select_proxy(
        self,
        strategy: str = "sequential",
        specific_proxy: Optional[str] = None,
        provider_id: Optional[str] = None,
        exclude_ids: Optional[Set[str]] = None,
    ) -> Optional[Proxy]:
        """
        Select an active proxy from the pool according to the strategy:
        - specific: routes to a specific proxy by ID, URL, name, or matching tag
        - random: randomly selects an available proxy
        - sequential / round_robin / switch_on_limit: sequentially rotates through available proxies
        """
        exclude = exclude_ids or set()
        candidates = [p for p in self.proxies.values() if p.is_available() and p.id not in exclude]

        # 1. Specific routing
        if strategy == "specific" or specific_proxy:
            target = (specific_proxy or "").strip()
            # Match by ID
            if target in self.proxies:
                p = self.proxies[target]
                if p.is_available() and p.id not in exclude:
                    return p
            # Match by URL, name, or tag
            for p in candidates:
                if p.url == target or p.name == target or target in p.tags:
                    return p
            return None

        if not candidates:
            return None

        # 2. Random routing
        if strategy == "random":
            return random.choice(candidates)

        # 3. Sequential / Round Robin / Switch on limit
        key = provider_id or "global"
        curr_idx = self.proxy_rr_indices.get(key, 0)
        selected = candidates[curr_idx % len(candidates)]
        self.proxy_rr_indices[key] = (curr_idx + 1) % len(candidates)
        return selected

    def resolve_request_proxy(
        self,
        variables: Dict[str, Any],
        account: Optional[Account] = None,
        exclude_proxy_ids: Optional[Set[str]] = None,
        provider_id: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[Proxy]]:
        """
        Resolve the proxy URL and proxy instance for an outgoing request:
        1. Routing some accounts through a proxy always:
           If account specifies proxy_url, proxy_id, or extra_config['proxy'], always route through that proxy.
        2. Provider external proxy routing (off by default):
           If variables['proxy_enabled'] is True:
             Selects from proxy pool using proxy_strategy (random, sequential, specific, switch_on_limit).
           If static variables['proxy'] is provided, use it directly.
        """
        # 1. Account-level dedicated proxy (always route this account through proxy)
        if account:
            acc_proxy = (
                getattr(account, "proxy_url", None)
                or getattr(account, "proxy_id", None)
                or (account.extra_config.get("proxy") if account.extra_config else None)
                or (account.extra_config.get("proxy_url") if account.extra_config else None)
                or (account.extra_config.get("proxy_id") if account.extra_config else None)
            )
            if acc_proxy:
                acc_proxy_str = str(acc_proxy).strip()
                # Check if it's an ID in self.proxies
                if acc_proxy_str in self.proxies:
                    p = self.proxies[acc_proxy_str]
                    if p.is_available():
                        return (p.url, p)
                # Check if it's a tag or name in self.proxies
                for p in self.proxies.values():
                    if p.is_available() and (acc_proxy_str in p.tags or p.name == acc_proxy_str):
                        return (p.url, p)
                # Direct proxy URL
                if "://" in acc_proxy_str:
                    return (acc_proxy_str, None)

        # 2. Provider external proxy configuration (off by default)
        proxy_enabled = bool(variables.get("proxy_enabled", False))
        static_proxy = str(variables.get("proxy") or "").strip()

        if not proxy_enabled:
            # If external proxy routing is off, but a direct static proxy is set on the provider:
            if static_proxy:
                return (static_proxy, None)
            return (None, None)

        # External proxy routing is enabled for this provider
        strategy = str(variables.get("proxy_strategy") or "sequential").lower().strip()
        specific_target = str(variables.get("specific_proxy_id") or "").strip()

        p = self.select_proxy(
            strategy=strategy,
            specific_proxy=specific_target if (strategy == "specific" or specific_target) else None,
            provider_id=provider_id,
            exclude_ids=exclude_proxy_ids,
        )
        if p:
            return (p.url, p)

        if static_proxy:
            return (static_proxy, None)

        return (None, None)

    def discover_and_load_all(self) -> None:
        """Scan plugins_dir and load all valid python plugin files."""
        if not self.plugins_dir.exists():
            return

        for file_path in sorted(self.plugins_dir.glob("*.py")):
            if file_path.name.startswith("_") or file_path.name.startswith("."):
                continue
            try:
                self.load_plugin_file(file_path)
            except Exception as e:
                logger.error("Failed to load plugin from %s: %s", file_path, e)

    def load_plugin_file(self, file_path: Path) -> Optional[BasePlugin]:
        """Load or reload a single python plugin file."""
        if not file_path.exists() or not file_path.is_file():
            return None

        importlib.invalidate_caches()
        module_name = f"llmproxy_plugins.{file_path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, str(file_path))
        if spec is None:
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        code = compile(file_path.read_text(encoding="utf-8"), str(file_path), "exec")
        exec(code, module.__dict__)

        # 1. Check for `plugin` instance
        plugin_instance: Optional[BasePlugin] = None
        if hasattr(module, "plugin") and isinstance(module.plugin, BasePlugin):
            plugin_instance = module.plugin
        # 2. Check for `create_plugin()` factory function
        elif hasattr(module, "create_plugin") and callable(module.create_plugin):
            created = module.create_plugin()
            if isinstance(created, BasePlugin):
                plugin_instance = created
        # 3. Check for any class inheriting BasePlugin defined in this module
        if plugin_instance is None:
            for attr_name in dir(module):
                val = getattr(module, attr_name)
                if (
                    isinstance(val, type)
                    and issubclass(val, BasePlugin)
                    and val is not BasePlugin
                    and getattr(val, "__module__", None) == module.__name__
                ):
                    try:
                        plugin_instance = val()
                        break
                    except Exception:
                        continue

        # Fallback if class was defined without matching __module__
        if plugin_instance is None:
            for attr_name in dir(module):
                val = getattr(module, attr_name)
                if isinstance(val, type) and issubclass(val, BasePlugin) and val is not BasePlugin:
                    try:
                        plugin_instance = val()
                        break
                    except Exception:
                        continue

        if plugin_instance is None:
            logger.warning("No BasePlugin instance or subclass found in %s", file_path)
            return None

        # Set filename in metadata if writable
        try:
            plugin_instance.metadata.filename = file_path.name
        except Exception:
            pass
        setattr(plugin_instance, "_filename", file_path.name)
        plugin_id = plugin_instance.metadata.id

        self.plugins[plugin_id] = plugin_instance
        self.plugin_files[plugin_id] = file_path

        # Initialize config entry if missing
        if plugin_id not in self.configs:
            default_vars = {
                v.name: v.default
                for v in plugin_instance.metadata.variables
                if v.default is not None
            }
            self.configs[plugin_id] = {
                "enabled": plugin_instance.metadata.enabled,
                "variables": default_vars,
                "custom_models": [],
                "edited_models": {},
                "deleted_models": [],
            }
            self._save_configs()
        else:
            cfg = self.configs[plugin_id]
            cfg.setdefault("enabled", plugin_instance.metadata.enabled)
            cfg.setdefault("variables", {})
            cfg.setdefault("custom_models", [])
            cfg.setdefault("edited_models", {})
            cfg.setdefault("deleted_models", [])

        return plugin_instance

    def reload_plugin(self, plugin_id: str) -> Optional[BasePlugin]:
        """Hot-reload a plugin by its ID."""
        file_path = self.plugin_files.get(plugin_id)
        if not file_path or not file_path.exists():
            file_path = self.plugins_dir / f"{plugin_id}.py"

        if file_path.exists():
            try:
                return self.load_plugin_file(file_path)
            except Exception as e:
                logger.error("Failed to reload plugin %s from %s: %s", plugin_id, file_path, e)
                return None
        return None

    def get_plugin(self, plugin_id: str) -> Optional[BasePlugin]:
        """Retrieve a loaded plugin by ID."""
        return self.plugins.get(plugin_id)

    def get_plugin_models(self, plugin_id: str) -> List[PluginModelConfig]:
        """Retrieve configured models for a plugin (defaults - deleted + edited + custom)."""
        plugin = self.plugins.get(plugin_id)
        if not plugin:
            return []

        cfg = self.configs.get(plugin_id, {})
        deleted = set(cfg.get("deleted_models", []))
        edited = cfg.get("edited_models", {})
        custom = cfg.get("custom_models", [])

        result: List[PluginModelConfig] = []
        seen_ids = set()

        for default_m in plugin.metadata.models:
            if default_m.id in deleted:
                continue
            if default_m.id in edited:
                model_obj = PluginModelConfig.model_validate(edited[default_m.id])
            else:
                model_obj = default_m.model_copy()
            result.append(model_obj)
            seen_ids.add(model_obj.id)

        for c in custom:
            model_obj = PluginModelConfig.model_validate(c)
            if model_obj.id not in seen_ids:
                result.append(model_obj)
                seen_ids.add(model_obj.id)

        return result

    def get_plugin_variables(self, plugin_id: str) -> Dict[str, Any]:
        """Return merged dictionary of variable values for a plugin."""
        plugin = self.plugins.get(plugin_id)
        if not plugin:
            return {}

        cfg = self.configs.get(plugin_id, {})
        stored_vars = cfg.get("variables", {})
        merged: Dict[str, Any] = {}

        for var in plugin.metadata.variables:
            if var.name in stored_vars:
                merged[var.name] = stored_vars[var.name]
            elif var.value != "" and var.value is not None:
                merged[var.name] = var.value
            else:
                merged[var.name] = var.default

        for k, v in stored_vars.items():
            if k not in merged:
                merged[k] = v

        return merged

    def list_plugins(self) -> List[PluginMetadata]:
        """List all loaded plugins with their current merged configuration."""
        result: List[PluginMetadata] = []
        for plugin in self.plugins.values():
            cfg = self.configs.get(plugin.metadata.id, {})
            enabled = cfg.get("enabled", plugin.metadata.enabled)
            pinned = cfg.get("pinned", getattr(plugin.metadata, "pinned", False))
            models = self.get_plugin_models(plugin.metadata.id)
            vars_dict = self.get_plugin_variables(plugin.metadata.id)

            variables: List[PluginVariable] = []
            declared_names = set()
            for v in plugin.metadata.variables:
                v_copy = v.model_copy()
                v_copy.value = vars_dict.get(v.name, v.default)
                variables.append(v_copy)
                declared_names.add(v.name)

            for k, v in vars_dict.items():
                if k not in declared_names:
                    variables.append(PluginVariable(name=k, label=k, value=v, default=v))

            file_path = self.plugin_files.get(plugin.metadata.id)
            filename = file_path.name if file_path else plugin.metadata.filename

            meta_copy = plugin.metadata.model_copy(update={
                "enabled": enabled,
                "pinned": pinned,
                "filename": filename,
                "variables": variables,
                "models": models,
            })
            result.append(meta_copy)

        # Sort pinned plugins first, then alphabetically by name
        result.sort(key=lambda p: (not p.pinned, p.name.lower()))
        return result

    def update_plugin_config(
        self,
        plugin_id: str,
        enabled: Optional[bool] = None,
        pinned: Optional[bool] = None,
        variables: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update plugin variables and/or enabled/pinned status."""
        if plugin_id not in self.plugins and plugin_id not in self.configs:
            return False

        cfg = self.configs.setdefault(
            plugin_id,
            {
                "enabled": True,
                "pinned": False,
                "variables": {},
                "custom_models": [],
                "edited_models": {},
                "deleted_models": [],
            },
        )
        if enabled is not None:
            cfg["enabled"] = enabled
        if pinned is not None:
            cfg["pinned"] = pinned
        if variables is not None:
            cfg.setdefault("variables", {}).update(variables)

        self._save_configs()
        return True

    def add_custom_model(self, plugin_id: str, model_config: PluginModelConfig) -> bool:
        """Add a custom model to a plugin, or un-delete / edit a default model with that ID."""
        if plugin_id not in self.plugins:
            return False

        cfg = self.configs.setdefault(
            plugin_id,
            {
                "enabled": True,
                "variables": {},
                "custom_models": [],
                "edited_models": {},
                "deleted_models": [],
            },
        )
        custom_models = cfg.setdefault("custom_models", [])
        deleted = cfg.setdefault("deleted_models", [])
        if model_config.id in deleted:
            deleted.remove(model_config.id)

        default_model_ids = {m.id for m in self.plugins[plugin_id].metadata.models}
        if model_config.id in default_model_ids:
            cfg.setdefault("edited_models", {})[model_config.id] = model_config.model_dump()
            self._save_configs()
            return True

        for i, m in enumerate(custom_models):
            if m.get("id") == model_config.id:
                custom_models[i] = model_config.model_dump()
                break
        else:
            custom_models.append(model_config.model_dump())

        self._save_configs()
        return True

    def update_custom_model(
        self, plugin_id: str, model_id: str, model_config: PluginModelConfig
    ) -> bool:
        """Update an existing model in a plugin."""
        if plugin_id not in self.plugins:
            return False

        cfg = self.configs.setdefault(
            plugin_id,
            {
                "enabled": True,
                "variables": {},
                "custom_models": [],
                "edited_models": {},
                "deleted_models": [],
            },
        )
        custom_models = cfg.setdefault("custom_models", [])
        for i, m in enumerate(custom_models):
            if m.get("id") == model_id:
                custom_models[i] = model_config.model_dump()
                self._save_configs()
                return True

        default_model_ids = {m.id for m in self.plugins[plugin_id].metadata.models}
        if model_id in default_model_ids:
            cfg.setdefault("edited_models", {})[model_id] = model_config.model_dump()
            deleted = cfg.setdefault("deleted_models", [])
            if model_id in deleted:
                deleted.remove(model_id)
            self._save_configs()
            return True

        custom_models.append(model_config.model_dump())
        self._save_configs()
        return True

    def delete_custom_model(self, plugin_id: str, model_id: str) -> bool:
        """Delete/remove a model from a plugin."""
        if plugin_id not in self.plugins:
            return False

        cfg = self.configs.setdefault(
            plugin_id,
            {
                "enabled": True,
                "variables": {},
                "custom_models": [],
                "edited_models": {},
                "deleted_models": [],
            },
        )
        custom_models = cfg.setdefault("custom_models", [])
        cfg["custom_models"] = [m for m in custom_models if m.get("id") != model_id]

        if model_id in cfg.get("edited_models", {}):
            del cfg["edited_models"][model_id]

        default_model_ids = {m.id for m in self.plugins[plugin_id].metadata.models}
        if model_id in default_model_ids:
            deleted = cfg.setdefault("deleted_models", [])
            if model_id not in deleted:
                deleted.append(model_id)

        self._save_configs()
        return True

    def get_all_active_models(self) -> List[Tuple[PluginModelConfig, BasePlugin]]:
        """Return list of (model_config, plugin) for all enabled plugins and models."""
        if not self.plugins:
            self.discover_and_load_all()
        active: List[Tuple[PluginModelConfig, BasePlugin]] = []
        for plugin_id, plugin in self.plugins.items():
            cfg = self.configs.get(plugin_id, {})
            if not cfg.get("enabled", plugin.metadata.enabled):
                continue
            models = self.get_plugin_models(plugin_id)
            for m in models:
                if m.enabled:
                    active.append((m, plugin))
        return active

    def resolve_model(
        self, model_id: str
    ) -> Optional[Tuple[PluginModelConfig, BasePlugin, Dict[str, Any]]]:
        """Resolve a model ID to its plugin, model config, and active variables."""
        # 1. First check if model_id is a custom load balancing rule
        for rule in self.rules.values():
            if rule.enabled and rule.model_id == model_id:
                target_model_id = self.select_rule_target(rule)
                if target_model_id and target_model_id != model_id:
                    res = self.resolve_model(target_model_id)
                    if res is not None:
                        return res

        # 2. Check directly across active plugin models
        for model_config, plugin in self.get_all_active_models():
            if model_config.id == model_id:
                vars_dict = self.get_plugin_variables(plugin.metadata.id)
                return (model_config, plugin, vars_dict)
        return None

    def add_log(self, entry: RequestLogEntry) -> None:
        """Add a request log entry and persist to disk (debounced)."""
        self.logs.insert(0, entry)
        if len(self.logs) > 1000:
            self.logs.pop()
        self._schedule_logs_save()

    def get_logs(self, limit: Optional[int] = None) -> List[RequestLogEntry]:
        """Retrieve recent request logs."""
        if limit is not None:
            return self.logs[:limit]
        return list(self.logs)


# Global singleton instance
plugin_manager = PluginManager()
