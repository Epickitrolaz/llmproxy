"""Helpers for provider usage meters.

Some OpenAI-compatible providers return live rate-limit or quota information in
response headers. Plugins can pass those headers to
:func:`capture_rate_limit_headers` after every upstream call. The dashboard then
reads the normalized snapshot from ``account.extra_config["usage_limits"]``.

Providers that expose a dedicated usage endpoint can instead implement
``BasePlugin.fetch_usage_quota`` and return the same normalized shape directly.
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Mapping, Optional, Tuple

_HEADER_PREFIXES = ("x-ratelimit-", "anthropic-ratelimit-")
_KIND_ALIASES = {
    "limit": "limit",
    "remaining": "remaining",
    "reset": "reset",
    "resetat": "reset",
    "reset_at": "reset",
}

_LABELS = {
    "requests": "Requests",
    "input_tokens": "Input tokens",
    "output_tokens": "Output tokens",
    "tokens": "Tokens",
    "max_parallel_requests": "Parallel requests",
    "concurrent_requests": "Concurrent requests",
}

_ORDER = {
    "requests": 0,
    "max_parallel_requests": 1,
    "concurrent_requests": 1,
    "input_tokens": 2,
    "output_tokens": 3,
    "tokens": 4,
}

# Compound duration values such as "44s", "1m30s", or "2h".
_DURATION_PART_RE = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>[smhd])", re.IGNORECASE)
_UNIT_SECONDS = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _parse_reset_time(value: Any, now: Optional[datetime] = None) -> Optional[str]:
    """Normalize a reset header to an ISO-8601 UTC timestamp."""
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None

    current = now or datetime.now(timezone.utc)

    # Relative durations, e.g. "44s" or "1m30s".
    matches = list(_DURATION_PART_RE.finditer(raw))
    if matches:
        consumed = "".join(match.group(0) for match in matches).replace(" ", "")
        if consumed == raw.replace(" ", ""):
            seconds = sum(
                float(match.group("value")) * _UNIT_SECONDS[match.group("unit").lower()]
                for match in matches
            )
            target = current + timedelta(seconds=seconds)
            return target.isoformat().replace("+00:00", "Z")

    # Plain seconds, e.g. "44".
    numeric = _as_float(raw)
    if numeric is not None:
        target = current + timedelta(seconds=numeric)
        return target.isoformat().replace("+00:00", "Z")

    # ISO-8601 timestamps, with a trailing Z accepted.
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_header_name(name: str) -> Optional[Tuple[str, str, str]]:
    """Return ``(kind, scope, window)`` for a rate-limit header name."""
    lower = str(name).lower()
    prefix = next((item for item in _HEADER_PREFIXES if lower.startswith(item)), None)
    if prefix is None:
        return None

    remainder = lower[len(prefix) :]
    segments = [segment for segment in remainder.split("-") if segment]
    for index, segment in enumerate(segments):
        kind = _KIND_ALIASES.get(segment) or _KIND_ALIASES.get(segment.replace("-", "_"))
        if not kind:
            continue
        before = segments[:index]
        after = segments[index + 1 :]
        if index == 0:
            # Munito: limit-requests, remaining-input-tokens.
            scope = ""
            window = "_".join(after)
        elif index == len(segments) - 1:
            # Anthropic: input-tokens-limit, input-tokens-remaining.
            scope = ""
            window = "_".join(before)
        else:
            # MoonMath: api_key-limit-max_parallel_requests.
            scope = "_".join(before)
            window = "_".join(after)
        if not window:
            return None
        return kind, scope, window
    return None


def _label_for(window: str) -> str:
    return _LABELS.get(window, window.replace("_", " ").replace("-", " ").title())


def rate_limit_groups_from_headers(
    headers: Mapping[str, Any],
    now: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Convert ``*-ratelimit-*`` response headers into dashboard buckets."""
    collected: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for raw_name, raw_value in (headers or {}).items():
        parsed = _parse_header_name(raw_name)
        if parsed is None:
            continue
        kind, scope, window = parsed
        entry = collected.setdefault((scope, window), {"scope": scope, "window": window})
        entry[kind] = raw_value

    buckets: List[Dict[str, Any]] = []
    for entry in collected.values():
        limit = _as_float(entry.get("limit"))
        remaining = _as_float(entry.get("remaining"))
        if limit is None or remaining is None or limit <= 0:
            continue
        used = max(0.0, min(100.0, ((limit - remaining) / limit) * 100.0))
        reset_time = _parse_reset_time(entry.get("reset"), now=now)
        bucket_id = entry["window"]
        if entry["scope"]:
            bucket_id = f"{entry['scope']}_{bucket_id}"
        buckets.append(
            {
                "bucketId": bucket_id,
                "displayName": _label_for(entry["window"]),
                "usedPercent": round(used, 2),
                "remainingFraction": round(max(0.0, min(1.0, remaining / limit)), 4),
                "resetTime": reset_time,
                "limit": limit,
                "remaining": remaining,
            }
        )

    # Prefer the headline token windows over a combined token counter when both
    # are present (Munito returns requests, input, output, and combined tokens).
    has_split_tokens = any(
        bucket["bucketId"].endswith(("input_tokens", "output_tokens")) for bucket in buckets
    )
    if has_split_tokens:
        buckets = [
            bucket
            for bucket in buckets
            if not (
                bucket["bucketId"].endswith("tokens")
                and not bucket["bucketId"].endswith(("input_tokens", "output_tokens"))
            )
        ]

    buckets.sort(
        key=lambda bucket: (
            _ORDER.get(bucket["bucketId"].split("_", 1)[-1], 99),
            bucket["bucketId"],
        )
    )
    if not buckets:
        return []

    for bucket in buckets:
        bucket.pop("limit", None)
        bucket.pop("remaining", None)
    return [{"displayName": "Rate limits", "buckets": buckets}]


def capture_rate_limit_headers(
    account: Any,
    headers: Mapping[str, Any],
) -> bool:
    """Store normalized rate-limit buckets on an account.

    Returns ``True`` when at least one usable window was found.
    """
    groups = rate_limit_groups_from_headers(headers)
    if not groups:
        return False
    extra = account.extra_config if isinstance(account.extra_config, dict) else {}
    extra["usage_limits"] = {
        "checked_at": time.time(),
        "source": "response-headers",
        "groups": groups,
    }
    account.extra_config = extra
    return True


def stored_usage_limits(account: Any) -> Optional[Dict[str, Any]]:
    """Return the last captured usage snapshot for an account, if any."""
    extra = account.extra_config if isinstance(account.extra_config, dict) else {}
    snapshot = extra.get("usage_limits")
    if not isinstance(snapshot, dict):
        return None
    groups = snapshot.get("groups")
    if not isinstance(groups, list) or not groups:
        return None
    return snapshot


__all__ = [
    "capture_rate_limit_headers",
    "rate_limit_groups_from_headers",
    "stored_usage_limits",
]
