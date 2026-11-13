"""Shared multimodal (image input) helpers.

LLMProxy accepts the OpenAI chat-completions shape, where a message ``content``
can be a string or a list of parts. Vision clients send image parts in a few
common spellings:

* OpenAI Chat Completions: ``{"type": "image_url", "image_url": {"url": ...}}``
* OpenAI Responses API: ``{"type": "input_image", "image_url": ...}``
* Anthropic Messages: ``{"type": "image", "source": {"type": "base64"|"url"}}``
* Gemini: ``{"inlineData": {"mimeType": ..., "data": ...}}``

Provider plugins use these helpers to normalise those spellings and translate
them to their native wire format. Keeping the logic in one place means every
provider behaves the same way for data URLs, remote URLs, and cache hints.
"""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import unquote_to_bytes, urlparse

import httpx

logger = logging.getLogger("llmproxy.multimodal")

# Image parts use several type names across APIs. Unknown types are preserved
# so a provider can decide whether it understands them.
IMAGE_PART_TYPES = {
    "image",
    "image_url",
    "input_image",
    "inline_data",
    "inlinedata",
    "file_data",
    "filedata",
}
TEXT_PART_TYPES = {"text", "input_text", "output_text"}

DEFAULT_IMAGE_TOKENS = 765
LOW_DETAIL_IMAGE_TOKENS = 85
DEFAULT_MAX_IMAGE_BYTES = 20 * 1024 * 1024

# Conservative name hints for templates that ship a catalog without explicit
# modality metadata. Explicit ``PluginModelConfig.supports_vision`` always wins.
VISION_MODEL_HINTS = (
    "gpt-4o",
    "gpt-4.1",
    "gpt-5",
    "gpt-6",
    "gpt-reserve",
    "claude",
    "gemini",
    "gemma",
    "grok",
    "qwen",
    "vl",
    "vision",
    "omni",
    "llava",
    "pixtral",
    "ernie",
    "nemotron",
    "gemma",
    "minimax",
    "glm-4",
    "glm-5",
)


def infer_vision_support(model_id: str, model_name: str = "") -> bool:
    """Guess whether a model accepts image input from its catalog ID/name."""
    haystack = f"{model_id} {model_name}".lower()
    return any(hint in haystack for hint in VISION_MODEL_HINTS)


def _part_type(part: Dict[str, Any]) -> str:
    return str(part.get("type") or "").strip().lower()


def is_image_part(part: Any) -> bool:
    """Return True when *part* looks like an image content part."""
    if not isinstance(part, dict):
        return False
    if _part_type(part) in IMAGE_PART_TYPES:
        return True
    # A few clients omit ``type`` but still send ``image_url``.
    return "image_url" in part or "inlineData" in part or "inline_data" in part


def parse_data_url(value: Any) -> Optional[Tuple[str, str]]:
    """Split a ``data:`` URL into ``(mime_type, base64_data)``.

    URL-encoded (non-base64) payloads are encoded to base64 so every provider
    receives the same representation. Returns ``None`` for normal URLs.
    """
    if not isinstance(value, str) or not value.startswith("data:"):
        return None
    header, separator, payload = value[5:].partition(",")
    if not separator:
        return None

    metadata = [item.strip() for item in header.split(";") if item.strip()]
    mime_type = metadata[0] if metadata else "image/png"
    is_base64 = any(item.lower() == "base64" for item in metadata[1:])

    if is_base64:
        data = "".join(payload.split())
    else:
        data = base64.b64encode(unquote_to_bytes(payload)).decode("ascii")
    return mime_type or "image/png", data


def _data_url_from_inline(inline: Any) -> Optional[str]:
    if not isinstance(inline, dict):
        return None
    data = inline.get("data")
    mime_type = inline.get("mimeType") or inline.get("mime_type") or "image/png"
    if isinstance(data, str) and data:
        return f"data:{mime_type};base64,{''.join(data.split())}"
    return None


def extract_image_url(part: Any) -> Optional[str]:
    """Return the image URL/data URL carried by a content part, if any."""
    if not isinstance(part, dict):
        return None

    ptype = _part_type(part)

    # OpenAI Chat Completions / Responses API.
    if ptype in ("image_url", "input_image") or "image_url" in part:
        value = part.get("image_url")
        if isinstance(value, dict):
            value = value.get("url")
        if not value:
            value = part.get("url")
        if isinstance(value, str) and value:
            return value

    # Anthropic image blocks.
    source = part.get("source")
    if isinstance(source, dict):
        source_type = str(source.get("type") or "").lower()
        if source_type == "base64":
            data = source.get("data")
            media_type = source.get("media_type") or source.get("mime_type") or "image/png"
            if isinstance(data, str) and data:
                return f"data:{media_type};base64,{''.join(data.split())}"
        if source_type == "url":
            url = source.get("url")
            if isinstance(url, str) and url:
                return url

    # Gemini inline/file data blocks.
    for key in ("inlineData", "inline_data"):
        inline_url = _data_url_from_inline(part.get(key))
        if inline_url:
            return inline_url

    for key in ("fileData", "file_data"):
        filedata = part.get(key)
        if isinstance(filedata, dict):
            uri = filedata.get("fileUri") or filedata.get("file_uri") or filedata.get("url")
            if isinstance(uri, str) and uri:
                return uri

    # Permissive fallback for blocks such as {"type":"image","url":...}.
    if ptype in IMAGE_PART_TYPES:
        for key in ("url", "image", "data"):
            value = part.get(key)
            if isinstance(value, str) and value:
                if key == "data":
                    return _data_url_from_inline({"data": value})
                return value

    return None


def image_detail(part: Any, default: Optional[str] = None) -> Optional[str]:
    """Return the OpenAI ``detail`` hint from a content part."""
    if not isinstance(part, dict):
        return default
    detail = part.get("detail")
    if detail is None and isinstance(part.get("image_url"), dict):
        detail = part["image_url"].get("detail")
    return detail if isinstance(detail, str) and detail else default


def normalize_image_part(part: Any) -> Optional[Dict[str, Any]]:
    """Normalise any supported image spelling to an OpenAI ``image_url`` part."""
    url = extract_image_url(part)
    if not url:
        return None

    image_url: Dict[str, Any] = {"url": url}
    detail = image_detail(part)
    if detail:
        image_url["detail"] = detail

    normalized: Dict[str, Any] = {"type": "image_url", "image_url": image_url}
    if isinstance(part, dict) and part.get("cache_control"):
        normalized["cache_control"] = part["cache_control"]
    return normalized


def normalize_content(content: Any) -> Any:
    """Rewrite mixed client image spellings into canonical OpenAI parts.

    Strings are left untouched. Unknown part types are preserved so plugins can
    decide whether they support them.
    """
    if not isinstance(content, list):
        return content

    normalized: List[Any] = []
    for part in content:
        if isinstance(part, str):
            normalized.append({"type": "text", "text": part})
            continue
        if not isinstance(part, dict):
            normalized.append(part)
            continue

        if is_image_part(part):
            image_part = normalize_image_part(part)
            if image_part is not None:
                normalized.append(image_part)
            continue

        ptype = _part_type(part)
        if ptype in TEXT_PART_TYPES:
            text_part: Dict[str, Any] = {"type": "text", "text": part.get("text", "")}
            if part.get("cache_control"):
                text_part["cache_control"] = part["cache_control"]
            normalized.append(text_part)
            continue

        normalized.append(dict(part))

    return normalized


def normalize_message_content(message: Any) -> Any:
    """Normalise a message object or dict's ``content`` field in place."""
    if isinstance(message, dict):
        if "content" in message:
            message["content"] = normalize_content(message.get("content"))
        return message
    if hasattr(message, "content"):
        message.content = normalize_content(message.content)
    return message


def content_text(content: Any) -> str:
    """Flatten string or list content to text, ignoring images."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: List[str] = []
        for part in content:
            if isinstance(part, str):
                pieces.append(part)
            elif isinstance(part, dict):
                if _part_type(part) in TEXT_PART_TYPES:
                    pieces.append(str(part.get("text") or ""))
                elif not is_image_part(part):
                    pieces.append(str(part.get("text") or ""))
        return "\n".join(piece for piece in pieces if piece)
    return str(content)


def is_http_url(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def _mime_type_for_url(url: str) -> str:
    parsed = urlparse(url)
    guessed, _ = mimetypes.guess_type(parsed.path)
    return guessed or "image/png"


def to_anthropic_image_block(part: Any) -> Optional[Dict[str, Any]]:
    """Translate an image part to an Anthropic Messages image block."""
    url = extract_image_url(part)
    if not url:
        return None

    data_url = parse_data_url(url)
    if data_url is not None:
        mime_type, data = data_url
        block: Dict[str, Any] = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": mime_type,
                "data": data,
            },
        }
    elif is_http_url(url):
        block = {"type": "image", "source": {"type": "url", "url": url}}
    else:
        # Anthropic only accepts base64 or http(s) URL sources.
        return None

    if isinstance(part, dict) and part.get("cache_control"):
        block["cache_control"] = part["cache_control"]
    return block


def to_gemini_part(part: Any) -> Optional[Dict[str, Any]]:
    """Translate an image part to a Gemini ``parts`` entry.

    Data URLs become ``inlineData``. Remote HTTP(S) URLs become ``fileData``;
    callers that can download should materialise them first so the request uses
    ``inlineData`` (the public Gemini API does not fetch arbitrary HTTP URLs).
    """
    url = extract_image_url(part)
    if not url:
        return None

    data_url = parse_data_url(url)
    if data_url is not None:
        mime_type, data = data_url
        translated: Dict[str, Any] = {
            "inlineData": {"mimeType": mime_type, "data": data}
        }
    else:
        translated = {
            "fileData": {
                "mimeType": _mime_type_for_url(url),
                "fileUri": url,
            }
        }

    if isinstance(part, dict) and part.get("cache_control"):
        translated["cache_control"] = part["cache_control"]
    return translated


def to_openai_image_part(part: Any) -> Optional[Dict[str, Any]]:
    """Translate an image part to an OpenAI Responses ``input_image`` item."""
    url = extract_image_url(part)
    if not url:
        return None

    item: Dict[str, Any] = {"type": "input_image", "image_url": url}
    detail = image_detail(part)
    if detail:
        item["detail"] = detail
    return item


async def fetch_image_as_data_url(
    url: str,
    *,
    timeout: float = 30.0,
    max_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
    proxy: Optional[str] = None,
    client: Optional[httpx.AsyncClient] = None,
) -> str:
    """Download an HTTP(S) image and return it as a ``data:`` URL."""
    if not is_http_url(url):
        return url

    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(
            timeout=timeout,
            proxy=proxy,
            follow_redirects=True,
        )

    try:
        response = await client.get(url)
        response.raise_for_status()
        mime_type = (response.headers.get("content-type") or "").split(";")[0].strip().lower()
        if not mime_type.startswith("image/"):
            raise ValueError(f"unsupported image content type: {mime_type or 'unknown'}")
        if len(response.content) > max_bytes:
            raise ValueError(f"image exceeds {max_bytes} bytes")
        encoded = base64.b64encode(response.content).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"
    finally:
        if owns_client:
            await client.aclose()


async def materialize_content_images(
    content: Any,
    *,
    timeout: float = 30.0,
    max_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
    proxy: Optional[str] = None,
    client: Optional[httpx.AsyncClient] = None,
) -> Any:
    """Replace remote image URLs in *content* with inline data URLs.

    Providers such as Gemini cannot fetch arbitrary URLs. When a download
    fails the image is replaced by a short text placeholder so the request still
    reaches the model instead of failing with an opaque upstream error.
    """
    if not isinstance(content, list):
        return content

    materialized: List[Any] = []
    changed = False
    for part in content:
        if not is_image_part(part):
            materialized.append(part)
            continue

        url = extract_image_url(part)
        if not url or not is_http_url(url):
            materialized.append(part)
            continue

        try:
            data_url = await fetch_image_as_data_url(
                url,
                timeout=timeout,
                max_bytes=max_bytes,
                proxy=proxy,
                client=client,
            )
        except Exception as exc:  # noqa: BLE001 - surface a model-readable placeholder
            host = urlparse(url).netloc or "remote host"
            logger.warning("Could not download image from %s: %s", host, exc)
            materialized.append(
                {"type": "text", "text": "[Image could not be loaded from the supplied URL]"}
            )
            changed = True
            continue

        if data_url == url:
            materialized.append(part)
        else:
            replacement = normalize_image_part(part) or {"type": "image_url", "image_url": {"url": data_url}}
            if isinstance(replacement.get("image_url"), dict):
                replacement["image_url"]["url"] = data_url
            materialized.append(replacement)
            changed = True

    return materialized if changed else content


async def materialize_message_images(
    messages: Sequence[Any],
    *,
    timeout: float = 30.0,
    max_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
    proxy: Optional[str] = None,
    client: Optional[httpx.AsyncClient] = None,
) -> List[Any]:
    """Return message copies with remote image URLs downloaded as data URLs."""
    result: List[Any] = []
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
        if not isinstance(content, list) or not any(is_image_part(part) for part in content):
            result.append(message)
            continue

        new_content = await materialize_content_images(
            content,
            timeout=timeout,
            max_bytes=max_bytes,
            proxy=proxy,
            client=client,
        )
        if new_content is content:
            result.append(message)
        elif isinstance(message, dict):
            copy = dict(message)
            copy["content"] = new_content
            result.append(copy)
        elif hasattr(message, "model_copy"):
            result.append(message.model_copy(update={"content": new_content}))
        else:
            result.append(message)
    return result


def estimate_image_tokens(part: Any) -> int:
    """Estimate image prompt tokens without decoding the pixels.

    OpenAI bills roughly 85 tokens for a low-detail image and 765 for a
    high-detail image. Text-only fallbacks are handled by the caller.
    """
    if image_detail(part) == "low":
        return LOW_DETAIL_IMAGE_TOKENS
    return DEFAULT_IMAGE_TOKENS


def estimate_content_tokens(content: Any) -> int:
    """Estimate prompt tokens for string or multimodal list content."""
    if content is None or content == "":
        return 0
    if isinstance(content, str):
        return max(1, len(content) // 4)
    if isinstance(content, list):
        total = 0
        for part in content:
            if isinstance(part, str):
                total += max(1, len(part) // 4)
            elif isinstance(part, dict):
                if is_image_part(part):
                    total += estimate_image_tokens(part)
                else:
                    text = part.get("text")
                    if isinstance(text, str):
                        total += max(1, len(text) // 4)
                    else:
                        try:
                            total += max(1, len(json.dumps(part)) // 4)
                        except (TypeError, ValueError):
                            total += 1
            else:
                total += 1
        return max(1, total)
    if isinstance(content, dict):
        return max(1, len(json.dumps(content, default=str)) // 4)
    return 1


def message_has_image(message: Any) -> bool:
    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    return isinstance(content, list) and any(is_image_part(part) for part in content)


def messages_have_images(messages: Sequence[Any]) -> bool:
    return any(message_has_image(message) for message in messages)
