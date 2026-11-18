from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from llmproxy.schemas import (
    Account,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    PluginMetadata,
    PluginModelConfig,
    PluginVariable,
)


class BasePlugin(ABC):
    """
    Contract for all LLMProxy provider plugins.
    Each plugin is an independent Python file in `plugins/`.
    Plugins support:
    - Custom variables (API keys, base URLs, options)
    - Custom models (add, edit, remove)
    - Chat completion (streaming and non-streaming)
    """

    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Return the plugin's metadata including declared variables and default models."""
        pass

    @abstractmethod
    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        model_config: PluginModelConfig,
        variables: Dict[str, Any],
        account: Optional[Account] = None,
    ) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionChunk]]:
        """
        Execute a chat completion request.
        If request.stream is True, must return an AsyncIterator[ChatCompletionChunk].
        Otherwise, must return ChatCompletionResponse.
        """
        pass

    def validate_variables(self, variables: Dict[str, Any]) -> List[str]:
        """Validate variables and return a list of validation error strings, if any."""
        errors: List[str] = []
        for var in self.metadata.variables:
            if var.required and not variables.get(var.name):
                errors.append(f"Variable '{var.name}' ({var.label}) is required.")
        return errors

    # ------------------------------------------------------------------ OAuth
    # Optional plugin-owned OAuth login. Providers that expose "Sign in" in the
    # Accounts tab set ``metadata.oauth_login`` and implement these methods.
    # The dashboard only knows this generic contract; provider wire protocol,
    # PKCE/device flow, local import, and token refresh all live in the plugin.
    async def oauth_start(self) -> Dict[str, Any]:
        """Start an OAuth login and return the flow descriptor.

        Shape::

            {
                "flow_id": "...",
                "mode": "loopback" | "device" | "callback",
                "auth_url": "...",                    # loopback/callback
                "redirect_uri": "...",                # loopback
                "state": "...",                       # loopback
                "verification_uri_complete": "...",   # device
                "user_code": "...",                   # device
                "expires_in": 600,                    # optional
                "interval": 5,                         # optional
                "supports_callback_paste": True,      # headless loopback
                "instructions": "...",                # optional UI copy
            }
        """
        raise NotImplementedError(f"{self.metadata.id} does not support OAuth login")

    async def oauth_status(self, flow_id: str) -> Dict[str, Any]:
        """Return ``{"status": "pending"|"complete"|"error", ...}``.

        On success include ``account`` with a fully populated ``Account``.
        """
        raise NotImplementedError(f"{self.metadata.id} does not support OAuth login")

    async def oauth_callback(self, flow_id: str, callback_url: str) -> Dict[str, Any]:
        """Complete a loopback flow from a pasted redirect URL."""
        raise NotImplementedError(f"{self.metadata.id} does not support OAuth callbacks")

    async def oauth_import_local(self) -> Optional[Account]:
        """Import a credential already present on this host, if the provider supports it."""
        return None

    async def refresh_account(self, account: Account) -> bool:
        """Refresh an OAuth account's access token in place. Return True when changed."""
        return False

    async def fetch_usage_quota(
        self,
        account: Optional[Account],
        variables: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Return a live usage/quota snapshot for an account.

        Plugins that expose a usage meter implement this method and return::

            {
                "groups": [
                    {"displayName": "Rate limits", "buckets": [
                        {"bucketId": "requests", "displayName": "Requests",
                         "usedPercent": 12.5, "resetTime": "2026-01-01T00:00:00Z"},
                    ]},
                ],
                "balance": {"value": 12.5, "label": "Available credits"},
                "summary": [{"label": "Requests (30d)", "value": "42"}],
            }

        ``balance`` and ``summary`` are optional. Returning ``None`` means the
        provider has no usage meter for this account.
        """
        return None
