from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from llmproxy.schemas import (
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
