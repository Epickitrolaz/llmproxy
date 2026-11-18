#!/usr/bin/env python3
"""Log in to Nous Portal for the LLMProxy ``nous_oauth`` provider.

Runs the same OAuth device-code flow as Hermes Agent, exchanges the refresh
token for an inference JWT, and (optionally) stores the rotated refresh token in
an LLMProxy account. The secret is never printed.

Examples
--------
Local login, print nothing secret, just write the credential file:

    python skills/llmproxy-plugin/scripts/nous_oauth_login.py \
        --json-out ~/.config/llmproxy/nous_oauth.json

Login and create/update the account on a running LLMProxy:

    export LLMPROXY_ADMIN_TOKEN=...
    python skills/llmproxy-plugin/scripts/nous_oauth_login.py \
        --llmproxy-url http://10.0.0.110:8080

Notes
-----
Nous refresh tokens are single-use and rotate. This script persists the rotated
token on every exchange. Do not run it against a session another client is also
refreshing (for example a Hermes Agent install sharing the same token): reuse
detection revokes the session. Run it once per LLMProxy account.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import stat
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

DEFAULT_PORTAL_URL = "https://portal.nousresearch.com"
DEFAULT_CLIENT_ID = "hermes-cli"
DEFAULT_SCOPE = "inference:invoke"
DEVICE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:device_code"
# Portal edge routes may identify the Hermes client; this is not a credential.
PORTAL_USER_AGENT = "hermes-agent/0.23.2 (cli; linux; x86_64)"


def _form_headers() -> Dict[str, str]:
    return {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
        "User-Agent": PORTAL_USER_AGENT,
    }


def _claims(token: str) -> Dict[str, Any]:
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        data = json.loads(base64.urlsafe_b64decode(payload))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _request_device_code(client: httpx.Client, portal: str, client_id: str, scope: str) -> Dict[str, Any]:
    response = client.post(
        f"{portal}/api/oauth/device/code",
        headers=_form_headers(),
        data={"client_id": client_id, "scope": scope},
    )
    response.raise_for_status()
    payload = response.json()
    required = ("device_code", "user_code", "verification_uri_complete", "expires_in", "interval")
    missing = [key for key in required if not payload.get(key)]
    if missing:
        raise RuntimeError(f"Device code response missing fields: {', '.join(missing)}")
    return payload


def _poll_for_token(
    client: httpx.Client,
    portal: str,
    client_id: str,
    device_code: str,
    expires_in: int,
    interval: int,
) -> Dict[str, Any]:
    deadline = time.monotonic() + expires_in
    wait = max(1, interval)
    while time.monotonic() < deadline:
        response = client.post(
            f"{portal}/api/oauth/token",
            headers=_form_headers(),
            data={
                "grant_type": DEVICE_GRANT_TYPE,
                "client_id": client_id,
                "device_code": device_code,
            },
        )
        if response.status_code == 200:
            payload = response.json()
            if not payload.get("access_token"):
                raise RuntimeError("Token response did not include access_token")
            return payload
        try:
            payload = response.json()
        except ValueError:
            response.raise_for_status()
            raise RuntimeError("Token endpoint returned a non-JSON error response")
        error = str(payload.get("error") or "")
        if error == "authorization_pending":
            time.sleep(wait)
            continue
        if error == "slow_down":
            wait = min(wait + 1, 30)
            time.sleep(wait)
            continue
        raise RuntimeError(
            f"Device login failed ({error or response.status_code}): "
            f"{payload.get('error_description') or ''}".strip()
        )
    raise RuntimeError("Device login expired before it was approved. Run the command again.")


def _refresh(client: httpx.Client, portal: str, client_id: str, refresh_token: str) -> Dict[str, Any]:
    response = client.post(
        f"{portal}/api/oauth/token",
        headers={**_form_headers(), "x-nous-refresh-token": refresh_token},
        data={"grant_type": "refresh_token", "client_id": client_id},
    )
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    if response.status_code != 200 or not payload.get("access_token"):
        raise RuntimeError(
            "Token refresh failed "
            f"(HTTP {response.status_code}): {payload.get('error') or response.text[:200]}"
        )
    return payload


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    try:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass


def _store_in_llmproxy(
    llmproxy_url: str,
    admin_token: str,
    provider_id: str,
    account_name: str,
    extra_config: Dict[str, Any],
) -> str:
    base = llmproxy_url.rstrip("/")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {admin_token}"}
    with httpx.Client(timeout=60) as client:
        response = client.get(
            f"{base}/api/accounts", params={"provider_id": provider_id}, headers=headers
        )
        response.raise_for_status()
        accounts = response.json()
        existing = next(
            (account for account in accounts if account.get("name") == account_name),
            accounts[0] if accounts else None,
        )
        if existing:
            response = client.put(
                f"{base}/api/accounts/{existing['id']}",
                headers=headers,
                json={"extra_config": extra_config, "enabled": True},
            )
            response.raise_for_status()
            return f"updated account {existing['id']} ({existing.get('name')})"
        response = client.post(
            f"{base}/api/accounts",
            headers=headers,
            json={
                "provider_id": provider_id,
                "name": account_name,
                "api_key": "",
                "extra_config": extra_config,
                "enabled": True,
            },
        )
        response.raise_for_status()
        created = response.json()
        return f"created account {created.get('id')} ({created.get('name')})"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--portal-url", default=DEFAULT_PORTAL_URL)
    parser.add_argument("--client-id", default=DEFAULT_CLIENT_ID)
    parser.add_argument("--scope", default=DEFAULT_SCOPE)
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout in seconds")
    parser.add_argument("--json-out", type=Path, help="Write credentials to this file (0600)")
    parser.add_argument("--llmproxy-url", help="Base URL of a running LLMProxy, e.g. http://10.0.0.110:8080")
    parser.add_argument(
        "--admin-token",
        default=os.environ.get("LLMPROXY_ADMIN_TOKEN", ""),
        help="LLMProxy admin token (default: LLMPROXY_ADMIN_TOKEN)",
    )
    parser.add_argument("--provider-id", default="nous_oauth")
    parser.add_argument("--account-name", default="Nous Portal (OAuth)")
    parser.add_argument("--open", action="store_true", help="Open the verification URL in a browser")
    args = parser.parse_args(argv)

    portal = args.portal_url.rstrip("/")
    with httpx.Client(timeout=args.timeout) as client:
        device = _request_device_code(client, portal, args.client_id, args.scope)
        print("To continue:")
        print(f"  1. Open: {device['verification_uri_complete']}")
        print(f"  2. If prompted, enter code: {device['user_code']}")
        if args.open:
            import webbrowser

            webbrowser.open(str(device["verification_uri_complete"]))
        print(f"Waiting for approval (every {max(1, int(device['interval']))}s)...", flush=True)

        token = _poll_for_token(
            client,
            portal,
            args.client_id,
            str(device["device_code"]),
            int(device["expires_in"]),
            int(device["interval"]),
        )
        refresh_token = token.get("refresh_token")
        if not isinstance(refresh_token, str) or not refresh_token:
            raise RuntimeError("Device login returned no refresh token")

        # The device response's access token may not be inference-scoped; the
        # refresh exchange is what Hermes uses for the inference JWT.
        refreshed = _refresh(client, portal, args.client_id, refresh_token)
        rotated = refreshed.get("refresh_token") or refresh_token
        access_token = str(refreshed["access_token"])
        claims = _claims(access_token)
        scope = claims.get("scope") or claims.get("scp") or ""
        scopes = set(scope.split()) if isinstance(scope, str) else set(scope or [])
        if args.scope not in scopes:
            raise RuntimeError(
                f"Refreshed token does not carry {args.scope} (scope={scope!r}); "
                "check the account plan or re-run the login"
            )
        exp = claims.get("exp")
        expires_at = (
            datetime.fromtimestamp(float(exp), tz=timezone.utc).isoformat()
            if isinstance(exp, (int, float))
            else None
        )

        extra_config = {
            "refresh_token": rotated,
            "access_token": access_token,
            "access_token_expires_at": float(exp) if isinstance(exp, (int, float)) else time.time() + 900,
            "inference_base_url": refreshed.get("inference_base_url")
            or "https://inference-api.nousresearch.com/v1",
            "client_id": args.client_id,
            "scope": args.scope,
            "needs_relogin": False,
        }

        if args.json_out:
            _save_json(args.json_out, extra_config)
            print(f"Credentials written to {args.json_out} (mode 0600)")

        if args.llmproxy_url:
            if not args.admin_token:
                print(
                    "LLMPROXY_ADMIN_TOKEN is not set; skipping the account update.",
                    file=sys.stderr,
                )
            else:
                result = _store_in_llmproxy(
                    args.llmproxy_url,
                    args.admin_token,
                    args.provider_id,
                    args.account_name,
                    extra_config,
                )
                print(f"LLMProxy: {result}")

    print("Login successful.")
    if expires_at:
        print(f"Inference token valid until {expires_at}.")
    print("The provider refreshes it automatically; the rotated refresh token is persisted each time.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
