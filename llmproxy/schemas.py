from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field


# ==========================================
# OpenAI-Compatible Schemas
# ==========================================

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "function", "tool"] = "user"
    content: Union[str, List[Dict[str, Any]]] = ""
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None
    extra_body: Optional[Dict[str, Any]] = None


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Optional[str] = "stop"


class ChatChoiceDelta(BaseModel):
    index: int = 0
    delta: Dict[str, Any]
    finish_reason: Optional[str] = None


class CompletionUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatChoice]
    usage: Optional[CompletionUsage] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatChoiceDelta]


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "llmproxy"
    permission: List[Dict[str, Any]] = []


class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelCard]


# ==========================================
# Plugin & Customization Schemas
# ==========================================

class Account(BaseModel):
    id: str = Field(default_factory=lambda: f"acc_{uuid.uuid4().hex[:8]}")
    provider_id: str
    name: str
    api_key: str = ""
    base_url: Optional[str] = None
    priority: int = 1
    weight: int = 1
    enabled: bool = True
    rate_limited_until: Optional[float] = None
    error_count: int = 0
    success_count: int = 0
    last_used_at: Optional[float] = None
    extra_config: Dict[str, Any] = Field(default_factory=dict)

    def is_available(self, now: Optional[float] = None) -> bool:
        if not self.enabled:
            return False
        current_time = now if now is not None else time.time()
        if self.rate_limited_until is not None and self.rate_limited_until > current_time:
            return False
        return True


class AccountCreate(BaseModel):
    provider_id: str
    name: str
    api_key: str = ""
    base_url: Optional[str] = None
    priority: int = 1
    weight: int = 1
    enabled: bool = True
    extra_config: Dict[str, Any] = Field(default_factory=dict)


class AccountUpdate(BaseModel):
    name: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    priority: Optional[int] = None
    weight: Optional[int] = None
    enabled: Optional[bool] = None
    extra_config: Optional[Dict[str, Any]] = None


class PluginVariable(BaseModel):
    name: str
    label: str
    type: Literal["string", "number", "boolean", "secret", "select"] = "string"
    default: Any = ""
    value: Any = ""
    required: bool = False
    description: str = ""
    options: Optional[List[str]] = None


class PluginModelConfig(BaseModel):
    id: str
    name: str
    upstream_model: Optional[str] = None
    enabled: bool = True
    context_window: Optional[int] = 8192
    max_tokens: Optional[int] = 4096
    parameters: Dict[str, Any] = Field(default_factory=dict)


class PluginMetadata(BaseModel):
    id: str
    name: str
    description: str = ""
    version: str = "1.0.0"
    author: str = "custom"
    enabled: bool = True
    filename: Optional[str] = None
    variables: List[PluginVariable] = Field(default_factory=list)
    models: List[PluginModelConfig] = Field(default_factory=list)


class PluginSourceUpdate(BaseModel):
    code: str


class PluginConfigUpdate(BaseModel):
    enabled: Optional[bool] = None
    variables: Optional[Dict[str, Any]] = None


class RequestLogEntry(BaseModel):
    id: str
    timestamp: float = Field(default_factory=time.time)
    model: str
    plugin_id: Optional[str] = None
    status: Literal["success", "error"] = "success"
    status_code: int = 200
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error_message: Optional[str] = None
