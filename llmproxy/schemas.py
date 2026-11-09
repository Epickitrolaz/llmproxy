from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field


# ==========================================
# OpenAI-Compatible Schemas
# ==========================================

class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["system", "user", "assistant", "function", "tool"] = "user"
    content: Optional[Union[str, List[Dict[str, Any]]]] = ""
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    parallel_tool_calls: Optional[bool] = None
    functions: Optional[List[Dict[str, Any]]] = None
    function_call: Optional[Union[str, Dict[str, Any]]] = None
    response_format: Optional[Dict[str, Any]] = None
    reasoning_effort: Optional[str] = None
    extra_body: Optional[Dict[str, Any]] = None


class ChatChoice(BaseModel):
    model_config = ConfigDict(extra="allow")

    index: int = 0
    message: ChatMessage
    finish_reason: Optional[str] = "stop"
    logprobs: Optional[Any] = None


class ChatChoiceDelta(BaseModel):
    model_config = ConfigDict(extra="allow")

    index: int = 0
    delta: Dict[str, Any]
    finish_reason: Optional[str] = None
    logprobs: Optional[Any] = None


class CompletionUsage(BaseModel):
    model_config = ConfigDict(extra="allow")

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatChoice]
    usage: Optional[CompletionUsage] = None
    system_fingerprint: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatChoiceDelta]
    usage: Optional[CompletionUsage] = None


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
    pinned: bool = False
    rate_limited_until: Optional[float] = None
    error_count: int = 0
    success_count: int = 0
    last_used_at: Optional[float] = None
    proxy_id: Optional[str] = None
    proxy_url: Optional[str] = None
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
    pinned: bool = False
    proxy_id: Optional[str] = None
    proxy_url: Optional[str] = None
    extra_config: Dict[str, Any] = Field(default_factory=dict)


class AccountUpdate(BaseModel):
    name: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    priority: Optional[int] = None
    weight: Optional[int] = None
    enabled: Optional[bool] = None
    pinned: Optional[bool] = None
    proxy_id: Optional[str] = None
    proxy_url: Optional[str] = None
    extra_config: Optional[Dict[str, Any]] = None


# ==========================================
# External Proxy Schemas
# ==========================================

class Proxy(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = Field(default_factory=lambda: f"proxy_{uuid.uuid4().hex[:8]}")
    name: str
    url: str  # e.g. "http://user:pass@host:port" or "socks5://host:port"
    enabled: bool = True
    rate_limited_until: Optional[float] = None
    error_count: int = 0
    success_count: int = 0
    last_used_at: Optional[float] = None
    tags: List[str] = Field(default_factory=list)

    def is_available(self, now: Optional[float] = None) -> bool:
        if not self.enabled:
            return False
        current_time = now if now is not None else time.time()
        if self.rate_limited_until is not None and self.rate_limited_until > current_time:
            return False
        return True


class ProxyCreate(BaseModel):
    name: str
    url: str
    enabled: bool = True
    tags: List[str] = Field(default_factory=list)


class ProxyUpdate(BaseModel):
    name: Optional[str] = None
    url: Optional[str] = None
    enabled: Optional[bool] = None
    tags: Optional[List[str]] = None
    rate_limited_until: Optional[float] = None


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
    pinned: bool = False
    filename: Optional[str] = None
    variables: List[PluginVariable] = Field(default_factory=list)
    models: List[PluginModelConfig] = Field(default_factory=list)


class PluginSourceUpdate(BaseModel):
    code: str


class PluginConfigUpdate(BaseModel):
    enabled: Optional[bool] = None
    pinned: Optional[bool] = None
    variables: Optional[Dict[str, Any]] = None


# ==========================================
# Local API Key & Usage Tracking Schemas
# ==========================================

class APIKey(BaseModel):
    id: str = Field(default_factory=lambda: f"key_{uuid.uuid4().hex[:8]}")
    key: str
    name: str
    enabled: bool = True
    created_at: float = Field(default_factory=time.time)
    last_used_at: Optional[float] = None
    total_requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    expires_at: Optional[float] = None
    max_requests: Optional[int] = None
    is_temporary: bool = False

    def is_valid(self, now: Optional[float] = None) -> bool:
        if not self.enabled:
            return False
        current_time = now if now is not None else time.time()
        if self.expires_at is not None and current_time > self.expires_at:
            return False
        if self.max_requests is not None and self.total_requests >= self.max_requests:
            return False
        return True


class APIKeyCreate(BaseModel):
    name: str
    key: Optional[str] = None
    enabled: bool = True
    expires_at: Optional[float] = None
    max_requests: Optional[int] = None
    is_temporary: bool = False


class APIKeyUpdate(BaseModel):
    name: Optional[str] = None
    enabled: Optional[bool] = None
    expires_at: Optional[float] = None
    max_requests: Optional[int] = None
    is_temporary: Optional[bool] = None


# ==========================================
# Custom Load Balancing Rules Schemas
# ==========================================

class LoadBalancingRule(BaseModel):
    id: str = Field(default_factory=lambda: f"rule_{uuid.uuid4().hex[:8]}")
    model_id: str  # Custom model ID exposed to /v1
    name: str
    enabled: bool = True
    strategy: Literal["round_robin", "least_used", "random", "fallback", "priority"] = "round_robin"
    targets: List[str] = Field(default_factory=list)  # Target model IDs
    created_at: float = Field(default_factory=time.time)
    description: Optional[str] = ""


class LoadBalancingRuleCreate(BaseModel):
    model_id: str
    name: str
    enabled: bool = True
    strategy: str = "round_robin"
    targets: List[str] = Field(default_factory=list)
    description: Optional[str] = ""


class LoadBalancingRuleUpdate(BaseModel):
    model_id: Optional[str] = None
    name: Optional[str] = None
    enabled: Optional[bool] = None
    strategy: Optional[str] = None
    targets: Optional[List[str]] = None
    description: Optional[str] = None


class RequestLogEntry(BaseModel):
    id: str
    timestamp: float = Field(default_factory=time.time)
    model: str
    plugin_id: Optional[str] = None
    api_key_id: Optional[str] = None
    status: Literal["success", "error"] = "success"
    status_code: int = 200
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error_message: Optional[str] = None
