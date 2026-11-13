"""Vision/image input support across provider adapters."""

from __future__ import annotations

import asyncio
import base64

import pytest

from llmproxy.core.multimodal import (
    estimate_content_tokens,
    extract_image_url,
    infer_vision_support,
    is_image_part,
    materialize_content_images,
    normalize_content,
    parse_data_url,
    to_anthropic_image_block,
    to_gemini_part,
    to_openai_image_part,
)
from llmproxy.schemas import ChatCompletionRequest, ChatMessage, PluginModelConfig
from templates.anthropic_proxy import AnthropicProxyPlugin
from templates.antigravity import AntigravityPlugin
from templates.codex import CodexPlugin
from templates.openai_compatible import OpenAICompatiblePlugin


PNG_BASE64 = base64.b64encode(b"not-a-real-png").decode("ascii")
PNG_DATA_URL = f"data:image/png;base64,{PNG_BASE64}"
IMAGE_URL = "https://example.com/cat.png"


def _message(content):
    return ChatMessage(role="user", content=content)


def test_parse_data_url_supports_base64_and_url_encoding():
    assert parse_data_url(PNG_DATA_URL) == ("image/png", PNG_BASE64)

    encoded = "data:image/svg+xml,%3Csvg%3E%3C/svg%3E"
    mime_type, data = parse_data_url(encoded)
    assert mime_type == "image/svg+xml"
    assert base64.b64decode(data) == b"<svg></svg>"

    assert parse_data_url(IMAGE_URL) is None


def test_normalize_content_accepts_openai_anthropic_and_gemini_parts():
    parts = [
        {"type": "text", "text": "What is this?"},
        {"type": "image_url", "image_url": {"url": PNG_DATA_URL, "detail": "high"}},
        {"type": "input_image", "image_url": IMAGE_URL},
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": PNG_BASE64},
        },
        {"inlineData": {"mimeType": "image/webp", "data": PNG_BASE64}},
    ]

    normalized = normalize_content(parts)

    assert normalized[0] == {"type": "text", "text": "What is this?"}
    assert normalized[1]["type"] == "image_url"
    assert normalized[1]["image_url"] == {"url": PNG_DATA_URL, "detail": "high"}
    assert normalized[2]["image_url"]["url"] == IMAGE_URL
    assert normalized[3]["image_url"]["url"] == f"data:image/jpeg;base64,{PNG_BASE64}"
    assert normalized[4]["image_url"]["url"] == f"data:image/webp;base64,{PNG_BASE64}"
    assert all(is_image_part(part) for part in normalized[1:])


def test_image_helpers_extract_canonical_urls():
    assert extract_image_url({"type": "image_url", "image_url": {"url": IMAGE_URL}}) == IMAGE_URL
    assert extract_image_url({"type": "input_image", "image_url": IMAGE_URL}) == IMAGE_URL
    assert extract_image_url(
        {"type": "image", "source": {"type": "url", "url": IMAGE_URL}}
    ) == IMAGE_URL
    assert extract_image_url({"fileData": {"fileUri": "gs://bucket/cat.png"}}) == "gs://bucket/cat.png"

    assert to_anthropic_image_block({"type": "image_url", "image_url": {"url": PNG_DATA_URL}}) == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": PNG_BASE64},
    }
    assert to_anthropic_image_block({"type": "image_url", "image_url": {"url": IMAGE_URL}}) == {
        "type": "image",
        "source": {"type": "url", "url": IMAGE_URL},
    }
    assert to_gemini_part({"type": "image_url", "image_url": {"url": PNG_DATA_URL}}) == {
        "inlineData": {"mimeType": "image/png", "data": PNG_BASE64}
    }
    assert to_openai_image_part(
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG_BASE64}, "detail": "low"}
    ) == {"type": "input_image", "image_url": PNG_DATA_URL, "detail": "low"}


def test_token_estimate_does_not_count_base64_payload():
    text_only = estimate_content_tokens([{"type": "text", "text": "hello world"}])
    with_image = estimate_content_tokens(
        [
            {"type": "text", "text": "hello world"},
            {"type": "image_url", "image_url": {"url": PNG_DATA_URL}},
            {"type": "image_url", "image_url": {"url": PNG_DATA_URL, "detail": "low"}},
        ]
    )
    assert text_only < 10
    assert with_image == text_only + 765 + 85


def test_vision_model_hint_covers_common_families():
    assert infer_vision_support("gpt-4o")
    assert infer_vision_support("claude-sonnet-5-5")
    assert infer_vision_support("gemini-3.8-flash")
    assert infer_vision_support("nvidia/nemotron-3.5-super-vl:free")
    assert not infer_vision_support("cohere/north-mini-code")


def test_anthropic_adapter_translates_images_and_rejects_them_in_system_role():
    plugin = AnthropicProxyPlugin()
    model = plugin.metadata.models[0]
    request = ChatCompletionRequest(
        model=model.id,
        messages=[
            ChatMessage(
                role="system",
                content=[
                    {"type": "text", "text": "Be terse."},
                    {"type": "image_url", "image_url": {"url": PNG_DATA_URL}},
                ],
            ),
            _message(
                [
                    {"type": "text", "text": "Describe."},
                    {"type": "image_url", "image_url": {"url": PNG_DATA_URL}},
                    {"type": "image_url", "image_url": {"url": IMAGE_URL}},
                ]
            ),
        ],
    )

    payload = plugin._build_payload(request, model, {})

    # Images are text-only in the system prompt.
    assert payload["system"][0] == {"type": "text", "text": "Be terse."}
    assert all(block.get("type") != "image" for block in payload["system"])

    blocks = payload["messages"][0]["content"]
    assert blocks[0] == {"type": "text", "text": "Describe."}
    assert blocks[1] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": PNG_BASE64},
    }
    assert blocks[2] == {
        "type": "image",
        "source": {"type": "url", "url": IMAGE_URL},
    }


def test_antigravity_adapter_emits_inline_data_for_vision():
    plugin = AntigravityPlugin()
    model = plugin.metadata.models[0]
    request = ChatCompletionRequest(
        model=model.id,
        messages=[
            _message(
                [
                    {"type": "text", "text": "Describe."},
                    {"type": "image_url", "image_url": {"url": PNG_DATA_URL}},
                ]
            )
        ],
    )

    _, body = plugin._build_upstream_request(request, model, account=None)

    parts = body["request"]["contents"][0]["parts"]
    assert parts[0] == {"text": "Describe."}
    assert parts[1] == {"inlineData": {"mimeType": "image/png", "data": PNG_BASE64}}


def test_codex_adapter_emits_input_image_items():
    plugin = CodexPlugin()
    model = plugin.metadata.models[0]
    request = ChatCompletionRequest(
        model=model.id,
        messages=[
            _message(
                [
                    {"type": "text", "text": "Describe."},
                    {"type": "image_url", "image_url": {"url": PNG_DATA_URL, "detail": "high"}},
                    {"type": "input_image", "image_url": IMAGE_URL},
                ]
            )
        ],
    )

    _, _, payload, _ = plugin._build_upstream_request(request, model, None, {})

    content = payload["input"][0]["content"]
    assert content[0] == {"type": "input_text", "text": "Describe."}
    assert content[1] == {
        "type": "input_image",
        "image_url": PNG_DATA_URL,
        "detail": "high",
    }
    assert content[2] == {"type": "input_image", "image_url": IMAGE_URL}


def test_openai_compatible_forwards_image_parts_unchanged(monkeypatch):
    import templates.openai_compatible as module

    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "id": "chatcmpl-vision-1",
                "object": "chat.completion",
                "created": 1,
                "model": "vision-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "A cat."},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            }

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, headers=None, json=None):
            captured["json"] = json
            return FakeResponse()

    monkeypatch.setattr(module.httpx, "AsyncClient", FakeClient)

    plugin = OpenAICompatiblePlugin()
    model = PluginModelConfig(id="vision-model", name="Vision Model", supports_vision=True)
    request = ChatCompletionRequest(
        model="vision-model",
        messages=[
            _message(
                [
                    {"type": "text", "text": "Describe."},
                    {"type": "image_url", "image_url": {"url": PNG_DATA_URL}},
                ]
            )
        ],
    )

    response = asyncio.run(
        plugin.chat_completion(
            request,
            model,
            {"api_key": "sk-live-123", "base_url": "https://upstream.example/v1"},
        )
    )

    assert response.choices[0].message.content == "A cat."
    sent = captured["json"]["messages"][0]["content"]
    assert sent[0] == {"type": "text", "text": "Describe."}
    assert sent[1] == {"type": "image_url", "image_url": {"url": PNG_DATA_URL}}


def test_materialize_remote_images_replaces_urls_and_falls_back(monkeypatch):
    import llmproxy.core.multimodal as module

    async def fake_fetch(url, **_kwargs):
        if url == IMAGE_URL:
            return PNG_DATA_URL
        raise RuntimeError("offline")

    monkeypatch.setattr(module, "fetch_image_as_data_url", fake_fetch)

    content = [
        {"type": "text", "text": "Describe."},
        {"type": "image_url", "image_url": {"url": IMAGE_URL}},
        {"type": "image_url", "image_url": {"url": "https://example.com/missing.png"}},
    ]
    result = asyncio.run(materialize_content_images(content))

    assert result[1]["image_url"]["url"] == PNG_DATA_URL
    assert result[2] == {
        "type": "text",
        "text": "[Image could not be loaded from the supplied URL]",
    }


def test_model_cards_advertise_vision(monkeypatch):
    from llmproxy.api.router import _model_card

    vision_model = PluginModelConfig(id="vision-model", name="Vision", supports_vision=True)
    text_model = PluginModelConfig(id="text-model", name="Text")

    vision_card = _model_card(vision_model, "demo")
    text_card = _model_card(text_model, "demo")

    assert vision_card.supports_vision is True
    assert vision_card.input_modalities == ["text", "image"]
    assert text_card.supports_vision is False
    assert text_card.input_modalities == ["text"]
