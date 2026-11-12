#!/usr/bin/env python3
"""
Validate an LLMProxy provider plugin file.

Usage:
    python skills/llmproxy-plugin/scripts/validate_plugin.py path/to/plugin.py
    python skills/llmproxy-plugin/scripts/validate_plugin.py path/to/plugin.py --smoke

The default check loads the plugin, validates its metadata and method signature,
and never performs network requests. `--smoke` also runs one non-streaming and
one streaming request through the plugin's simulation mode; only use it when the
plugin falls back to simulation without credentials (as the skeleton does).
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llmproxy.core.interface import BasePlugin  # noqa: E402
from llmproxy.dashboard.loader import load_plugin_from_file  # noqa: E402
from llmproxy.schemas import (  # noqa: E402
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
)

ALLOWED_VARIABLE_TYPES = {"string", "number", "boolean", "secret", "select"}


def _validate_plugin(plugin: BasePlugin, path: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    try:
        meta = plugin.metadata
    except Exception as exc:
        return [f"metadata property raised: {exc}"], warnings

    if not meta.id:
        errors.append("metadata.id is required")
    elif not all(char.isalnum() or char in "_-." for char in meta.id):
        warnings.append("metadata.id should contain only letters, digits, '_', '-', and '.'")

    if not meta.name:
        errors.append("metadata.name is required")
    if not meta.models:
        errors.append("metadata.models must contain at least one model")

    model_ids = [model.id for model in meta.models]
    if len(model_ids) != len(set(model_ids)):
        errors.append("metadata.models contains duplicate model IDs")
    for model in meta.models:
        if not model.id:
            errors.append("every model needs an id")
        if not model.upstream_model:
            warnings.append(f"model '{model.id}' has no upstream_model; the router will use the model id")

    variable_names = [variable.name for variable in meta.variables]
    if len(variable_names) != len(set(variable_names)):
        errors.append("metadata.variables contains duplicate variable names")
    for variable in meta.variables:
        if not variable.name:
            errors.append("every variable needs a name")
        if variable.type not in ALLOWED_VARIABLE_TYPES:
            errors.append(
                f"variable '{variable.name}' has unsupported type '{variable.type}' "
                f"(expected one of {sorted(ALLOWED_VARIABLE_TYPES)})"
            )
        if variable.type == "select" and not variable.options:
            errors.append(f"select variable '{variable.name}' must declare options")

    if not inspect.iscoroutinefunction(plugin.chat_completion):
        errors.append("chat_completion must be an async method")
    else:
        signature = inspect.signature(plugin.chat_completion)
        if "account" not in signature.parameters and not any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        ):
            warnings.append(
                "chat_completion does not accept an 'account' parameter; "
                "account pools and credentials will not work"
            )

    if path.suffix != ".py":
        errors.append("plugin file must be a .py file")

    return errors, warnings


async def _run_smoke(plugin: BasePlugin) -> list[str]:
    errors: list[str] = []
    model = plugin.metadata.models[0]

    request = ChatCompletionRequest(
        model=model.id,
        messages=[ChatMessage(role="user", content="ping")],
    )
    try:
        result = await plugin.chat_completion(request, model, {}, account=None)
    except Exception as exc:
        return [f"non-streaming smoke test failed: {exc}"]
    if not isinstance(result, ChatCompletionResponse):
        errors.append(
            "non-streaming smoke test returned "
            f"{type(result).__name__}, expected ChatCompletionResponse"
        )
    elif not result.choices:
        errors.append("non-streaming smoke test returned no choices")

    stream_request = ChatCompletionRequest(
        model=model.id,
        messages=[ChatMessage(role="user", content="ping")],
        stream=True,
    )
    try:
        stream = await plugin.chat_completion(stream_request, model, {}, account=None)
        if not hasattr(stream, "__aiter__"):
            errors.append(
                "streaming smoke test returned "
                f"{type(stream).__name__}, expected an async iterator"
            )
        else:
            first = await stream.__anext__()
            if not isinstance(first, ChatCompletionChunk):
                errors.append(
                    "streaming smoke test yielded "
                    f"{type(first).__name__}, expected ChatCompletionChunk"
                )
            aclose = getattr(stream, "aclose", None)
            if aclose is not None:
                await aclose()
    except Exception as exc:
        errors.append(f"streaming smoke test failed: {exc}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate an LLMProxy provider plugin")
    parser.add_argument("plugin_file", help="Path to the plugin .py file")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run simulation-mode streaming and non-streaming requests",
    )
    args = parser.parse_args()

    path = Path(args.plugin_file).resolve()
    if not path.exists():
        print(f"ERROR: plugin file not found: {path}")
        return 1

    try:
        plugin = load_plugin_from_file(path)
    except Exception as exc:
        print(f"ERROR: failed to load plugin: {exc}")
        return 1

    errors, warnings = _validate_plugin(plugin, path)
    if args.smoke and not errors:
        errors.extend(asyncio.run(_run_smoke(plugin)))

    meta = plugin.metadata
    print(f"Plugin: {meta.name} ({meta.id}) v{meta.version}")
    print(f"Models: {', '.join(model.id for model in meta.models) or '(none)'}")
    print(f"Variables: {', '.join(variable.name for variable in meta.variables) or '(none)'}")

    for warning in warnings:
        print(f"WARN: {warning}")
    for error in errors:
        print(f"ERROR: {error}")

    if errors:
        print(f"Validation failed with {len(errors)} error(s).")
        return 1

    print("Validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
