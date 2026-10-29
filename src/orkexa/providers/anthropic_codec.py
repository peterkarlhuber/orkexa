"""Lossless conversion between Anthropic Messages API dicts and neutral messages.

``encode_message(decode_message(m)) == m`` for every message the Messages API or
``claude-agent`` 0.8.0 produces: fields without a neutral counterpart
(``cache_control``, thinking ``signature``, raw ``citations``, ``caller``, ...) are
kept in ``Part.meta["anthropic"]`` and block types without a neutral counterpart
(server tool calls and results, compaction, MCP blocks, ...) become
:class:`~orkexa.core.types.NativePart`. This module does not import the SDK.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, TypeVar

from pydantic import ValidationError

from ..core.types import (
    Citation,
    CitationsConfig,
    DocumentPart,
    ImagePart,
    MediaSource,
    Message,
    NativePart,
    Part,
    ReasoningPart,
    ResultContentPart,
    SearchResultPart,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    _PartBase,
)
from ..errors import HistoryConversionError

__all__ = [
    "PROVIDER",
    "decode_block",
    "decode_message",
    "decode_messages",
    "encode_message",
    "encode_messages",
    "encode_part",
    "native_citations",
]

PROVIDER = "anthropic"
_P = TypeVar("_P", bound=_PartBase)
_MEDIA_KEYS = frozenset({"type", "media_type", "data", "url", "file_id"})


def _extras(block: dict[str, Any], mapped: Iterable[str]) -> dict[str, Any]:
    """Keys of ``block`` that the neutral part does not reproduce exactly."""
    consumed = {key for key in mapped if block.get(key) is not None}
    return {key: value for key, value in block.items() if key not in consumed}


def _with_meta(part: _P, extras: dict[str, Any]) -> _P:
    return part.with_native(PROVIDER, extras) if extras else part


def native_citations(raw: Any) -> list[Citation] | None:
    """Neutral citations of a text block's ``citations`` list."""
    if not isinstance(raw, list) or not raw:
        return None
    found: list[Citation] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        cited = str(item.get("cited_text") or "")
        if kind == "search_result_location":
            found.append(
                Citation(
                    source=str(item.get("source") or ""),
                    title=item.get("title"),
                    cited_text=cited,
                    search_result_index=item.get("search_result_index"),
                    provider=PROVIDER,
                )
            )
        elif kind == "web_search_result_location":
            url = item.get("url")
            found.append(
                Citation(
                    source=str(url or ""),
                    title=item.get("title"),
                    cited_text=cited,
                    provider=PROVIDER,
                    url=url,
                )
            )
        else:
            title = item.get("document_title")
            found.append(
                Citation(source=str(title or kind or ""), title=title, cited_text=cited, provider=PROVIDER)
            )
    return found or None


def _media(block: dict[str, Any]) -> tuple[MediaSource | None, dict[str, Any]]:
    source = block.get("source")
    if (
        isinstance(source, dict)
        and set(source) <= _MEDIA_KEYS
        and all(v is not None for v in source.values())
    ):
        try:
            return MediaSource.model_validate(source), {}
        except ValidationError:
            pass
    return None, {"source": source}


def decode_block(block: dict[str, Any]) -> Part:
    """One content block dict as a neutral part (unknown types become ``NativePart``)."""
    kind = block.get("type")
    if kind == "text" and isinstance(block.get("text"), str):
        part: Part = TextPart(text=block["text"], citations=native_citations(block.get("citations")))
        return _with_meta(part, _extras(block, ("type", "text")))
    if kind in ("image", "document"):
        source, source_extras = _media(block)
        if source is None:
            return NativePart(provider=PROVIDER, data=dict(block))
        if kind == "image":
            return _with_meta(
                ImagePart(source=source), {**_extras(block, ("type", "source")), **source_extras}
            )
        citations = block.get("citations")
        config = None
        if (
            isinstance(citations, dict)
            and set(citations) == {"enabled"}
            and isinstance(citations["enabled"], bool)
        ):
            config = CitationsConfig(enabled=citations["enabled"])
        mapped = ["type", "source"] + [k for k in ("title", "context") if isinstance(block.get(k), str)]
        if config is not None:
            mapped.append("citations")
        part = DocumentPart(
            source=source,
            title=block.get("title") if isinstance(block.get("title"), str) else None,
            context=block.get("context") if isinstance(block.get("context"), str) else None,
            citations=config,
        )
        return _with_meta(part, _extras(block, mapped))
    if kind == "search_result":
        content = block.get("content")
        if (
            isinstance(block.get("source"), str)
            and isinstance(block.get("title"), str)
            and isinstance(content, list)
            and all(
                isinstance(c, dict) and set(c) == {"type", "text"} and c["type"] == "text" for c in content
            )
        ):
            citations = block.get("citations")
            config = (
                CitationsConfig(enabled=citations["enabled"])
                if isinstance(citations, dict) and set(citations) == {"enabled"}
                else None
            )
            mapped = ["type", "source", "title", "content"] + (["citations"] if config else [])
            part = SearchResultPart(
                source=block["source"],
                title=block["title"],
                content=[TextPart(text=str(c["text"])) for c in content],
                citations=config,
            )
            return _with_meta(part, _extras(block, mapped))
    if kind == "tool_use" and isinstance(block.get("id"), str) and isinstance(block.get("name"), str):
        arguments = block.get("input")
        if isinstance(arguments, dict):
            part = ToolCallPart(id=block["id"], name=block["name"], arguments=dict(arguments))
            return _with_meta(part, _extras(block, ("type", "id", "name", "input")))
        part = ToolCallPart(
            id=block["id"],
            name=block["name"],
            parse_error=f"tool input is not a JSON object: {arguments!r}"[:500],
        )
        return _with_meta(part, _extras(block, ("type", "id", "name")))
    if kind == "tool_result" and isinstance(block.get("tool_use_id"), str):
        return _decode_tool_result(block)
    if kind == "thinking" and isinstance(block.get("thinking"), str):
        return _with_meta(ReasoningPart(text=block["thinking"]), _extras(block, ("type", "thinking")))
    if kind == "redacted_thinking":
        return _with_meta(ReasoningPart(redacted=True), _extras(block, ("type",)))
    return NativePart(provider=PROVIDER, data=dict(block))


def _decode_tool_result(block: dict[str, Any]) -> Part:
    raw = block.get("content")
    mapped = ["type", "tool_use_id"]
    content: str | list[ResultContentPart] | None = None
    if isinstance(raw, str):
        content = raw
        mapped.append("content")
    elif isinstance(raw, list):
        parts = [decode_block(item) if isinstance(item, dict) else None for item in raw]
        if all(isinstance(p, (TextPart, ImagePart, DocumentPart, SearchResultPart)) for p in parts):
            content = [
                p for p in parts if isinstance(p, (TextPart, ImagePart, DocumentPart, SearchResultPart))
            ]
            mapped.append("content")
    if block.get("is_error") is True:
        mapped.append("is_error")
    part = ToolResultPart(
        call_id=block["tool_use_id"], content=content, is_error=block.get("is_error") is True
    )
    return _with_meta(part, _extras(block, mapped))


def encode_part(part: Part) -> dict[str, Any]:
    """A neutral part as an Anthropic content block (Anthropic extras restored)."""
    extras = part.native(PROVIDER)
    block: dict[str, Any]
    if isinstance(part, TextPart):
        block = {"type": "text", "text": part.text}
    elif isinstance(part, (ImagePart, DocumentPart)):
        source = part.source.model_dump(exclude_none=True)
        source.pop("provider", None)
        block = {"type": part.type, "source": source}
        if isinstance(part, DocumentPart):
            if part.title is not None:
                block["title"] = part.title
            if part.context is not None:
                block["context"] = part.context
            if part.citations is not None:
                block["citations"] = part.citations.model_dump()
    elif isinstance(part, SearchResultPart):
        block = {
            "type": "search_result",
            "source": part.source,
            "title": part.title,
            "content": [{"type": "text", "text": item.text} for item in part.content],
        }
        if part.citations is not None:
            block["citations"] = part.citations.model_dump()
    elif isinstance(part, ToolCallPart):
        block = {"type": "tool_use", "id": part.id, "name": part.name, "input": dict(part.arguments)}
    elif isinstance(part, ToolResultPart):
        block = {"type": "tool_result", "tool_use_id": part.call_id}
        if isinstance(part.content, str):
            block["content"] = part.content
        elif part.content is not None:
            block["content"] = [encode_part(item) for item in part.content]
        if part.is_error:
            block["is_error"] = True
    elif isinstance(part, ReasoningPart):
        block = (
            {"type": "redacted_thinking"}
            if part.redacted
            else {"type": "thinking", "thinking": part.text or ""}
        )
    elif part.provider == PROVIDER:
        return dict(part.data)
    else:
        raise HistoryConversionError(
            f"A {part.provider}-only block ({part.data.get('type')!r}) cannot be sent to Anthropic"
        )
    block.update(extras)
    return block


def decode_message(message: dict[str, Any]) -> Message:
    """A Messages API message dict as a neutral message."""
    role = message.get("role")
    content = message.get("content")
    if role not in ("user", "assistant", "system"):
        raise ValueError(f"Invalid message role: {role!r}")
    provider = PROVIDER if role == "assistant" else None
    if isinstance(content, str):
        return Message(role=role, content=content, provider=provider)
    if not isinstance(content, list):
        raise ValueError(f"Invalid message content: {content!r}")
    parts = [decode_block(block) if isinstance(block, dict) else _invalid(block) for block in content]
    return Message(role=role, content=parts, provider=provider)


def _invalid(block: Any) -> Part:
    raise ValueError(f"Invalid content block: {block!r}")


def encode_message(message: Message) -> dict[str, Any]:
    if isinstance(message.content, str):
        return {"role": message.role, "content": message.content}
    return {"role": message.role, "content": [encode_part(part) for part in message.content]}


def decode_messages(messages: Sequence[dict[str, Any]]) -> list[Message]:
    return [decode_message(m) for m in messages]


def encode_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    return [encode_message(m) for m in messages]
