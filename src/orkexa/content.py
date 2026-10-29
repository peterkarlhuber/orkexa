"""Helpers that build content blocks for multimodal prompts.

``Agent.ask()`` accepts a plain string or a list of content blocks. These helpers
return Orkexa's provider-neutral input block dicts (the same layout ``claude-agent``
used, which is also the Anthropic Messages API layout); every provider adapter
translates them into its own format::

    from orkexa import Agent, document, image

    agent.ask([image("chart.png"), "What trend does this chart show?"])
    agent.ask([document("report.pdf", citations=True), "Summarize the risks."])

Strings in such a list become text blocks.

Tools can return blocks too: annotate a ``@Tool`` function with ``-> ToolContent``
and return a list of blocks and strings, e.g. ``[image(png_bytes, media_type="image/png")]``.
"""

from __future__ import annotations

import base64
from collections.abc import Sequence
import mimetypes
import os
from pathlib import Path
from typing import Annotated, Any, Literal

__all__ = [
    "ContentBlock",
    "Prompt",
    "ToolContent",
    "describe_content",
    "document",
    "image",
    "text",
    "to_content",
    "to_tool_content",
    "uploaded_file",
]

ContentBlock = dict[str, Any]
"""One content block dict, e.g. ``{"type": "text", "text": "..."}``."""

Prompt = str | Sequence[ContentBlock | str]
"""What :meth:`Agent.ask <orkexa.Agent.ask>` accepts: a string or a list of blocks and strings."""


class _ToolContentMarker:
    def __repr__(self) -> str:
        return "ToolContent"


_TOOL_CONTENT = _ToolContentMarker()

ToolContent = Annotated[list[ContentBlock | str], _TOOL_CONTENT]
"""Return annotation for tools that answer with content blocks instead of text or JSON.

Such a tool returns a list of blocks and strings (strings become text blocks).
Allowed blocks: ``text``, ``image``, ``document`` and ``search_result``; a result
with search results must consist of search results only (an API rule).
"""

_IMAGE_TYPES = ("image/jpeg", "image/png", "image/gif", "image/webp")
_TOOL_RESULT_TYPES = ("text", "image", "document", "search_result")


def _is_url(source: object) -> bool:
    return isinstance(source, str) and source.startswith(("https://", "http://"))


def _read(source: str | os.PathLike[str] | bytes, media_type: str | None) -> tuple[bytes, str | None]:
    if isinstance(source, bytes):
        return source, media_type
    path = Path(source)
    return path.read_bytes(), media_type or mimetypes.guess_type(path.name)[0]


def text(value: str) -> ContentBlock:
    """A text block."""
    return {"type": "text", "text": value}


def image(source: str | os.PathLike[str] | bytes, *, media_type: str | None = None) -> ContentBlock:
    """An image block from a file path, raw bytes or an ``http(s)`` URL.

    Args:
        source: Path, bytes or URL of a JPEG, PNG, GIF or WebP image.
        media_type: Required for bytes; guessed from the file name otherwise.
    """
    if _is_url(source):
        return {"type": "image", "source": {"type": "url", "url": source}}
    data, kind = _read(source, media_type)
    if kind not in _IMAGE_TYPES:
        raise ValueError(f"Unsupported image media type {kind!r}; expected one of {_IMAGE_TYPES}")
    encoded = base64.standard_b64encode(data).decode("ascii")
    return {"type": "image", "source": {"type": "base64", "media_type": kind, "data": encoded}}


def document(
    source: str | os.PathLike[str] | bytes,
    *,
    media_type: str | None = None,
    title: str | None = None,
    context: str | None = None,
    citations: bool = False,
) -> ContentBlock:
    """A document block from a PDF or plain-text file, raw bytes or an ``http(s)`` PDF URL.

    With ``citations=True`` the answer's text blocks carry ``citations`` that
    point into the document (Anthropic; see ``agent.last_response``). Citations
    cannot be combined with ``output_format``.

    Args:
        source: Path, bytes or URL. Files ending in ``.pdf`` are sent as PDF,
            everything else as UTF-8 text.
        media_type: ``"application/pdf"`` or ``"text/plain"``; required for bytes.
        title: Optional document title shown to the model.
        context: Optional context about the document (not cited).
        citations: Enable citations for this document.
    """
    block: ContentBlock
    if _is_url(source):
        block = {"type": "document", "source": {"type": "url", "url": source}}
    else:
        data, kind = _read(source, media_type)
        if kind == "application/pdf":
            encoded = base64.standard_b64encode(data).decode("ascii")
            block = {
                "type": "document",
                "source": {"type": "base64", "media_type": "application/pdf", "data": encoded},
            }
        elif kind is None or kind.startswith("text/"):
            block = {
                "type": "document",
                "source": {"type": "text", "media_type": "text/plain", "data": data.decode("utf-8")},
            }
        else:
            raise ValueError(f"Unsupported document media type {kind!r}; use a PDF or a text file")
    return _decorate(block, title=title, context=context, citations=citations)


def uploaded_file(
    file_id: str,
    *,
    kind: Literal["document", "image"] = "document",
    title: str | None = None,
    context: str | None = None,
    citations: bool = False,
    provider: str | None = None,
) -> ContentBlock:
    """A block that references a file uploaded to a provider's file API.

    File ids belong to one provider account and are never converted to another
    provider. Without ``provider`` the id is an Anthropic Files API id
    (``client.files.upload``), as in ``claude-agent``.

    Args:
        file_id: The ``id`` returned by the upload.
        kind: ``"document"`` for PDF/text files, ``"image"`` for images.
        title: Optional document title (documents only).
        context: Optional document context (documents only).
        citations: Enable citations (documents only).
        provider: The provider that issued ``file_id``, e.g. ``"openai"``; ``None``
            means ``"anthropic"``.
    """
    source: dict[str, Any] = {"type": "file", "file_id": file_id}
    if provider is not None:
        source["provider"] = provider
    block: ContentBlock = {"type": kind, "source": source}
    if kind == "image":
        if title or context or citations:
            raise ValueError("title, context and citations apply to documents only")
        return block
    return _decorate(block, title=title, context=context, citations=citations)


def _decorate(
    block: ContentBlock, *, title: str | None, context: str | None, citations: bool
) -> ContentBlock:
    if title:
        block["title"] = title
    if context:
        block["context"] = context
    if citations:
        block["citations"] = {"enabled": True}
    return block


def to_content(prompt: Prompt) -> str | list[ContentBlock]:
    """Normalize a prompt for a user message; raises ``ValueError`` if it is empty."""
    if isinstance(prompt, str):
        if not prompt.strip():
            raise ValueError("prompt must be a nonempty string")
        return prompt
    if not isinstance(prompt, Sequence):
        raise ValueError("prompt must be a string or a list of content blocks")
    blocks: list[ContentBlock] = []
    for item in prompt:
        if isinstance(item, str):
            if item.strip():
                blocks.append(text(item))
        elif isinstance(item, dict) and isinstance(item.get("type"), str):
            blocks.append(dict(item))
        else:
            raise ValueError(f"Invalid content block: {item!r}")
    if not blocks:
        raise ValueError("prompt must contain at least one nonempty block")
    return blocks


def to_tool_content(value: object) -> list[ContentBlock]:
    """Validate what a ``-> ToolContent`` tool returned; raises ``ValueError`` if it is invalid."""
    if isinstance(value, (str, dict)) or not isinstance(value, Sequence):
        raise ValueError("a ToolContent tool must return a list of content blocks and strings")
    blocks: list[ContentBlock] = []
    for item in value:
        if isinstance(item, str):
            if item:
                blocks.append(text(item))
        elif isinstance(item, dict) and item.get("type") in _TOOL_RESULT_TYPES:
            blocks.append(dict(item))
        else:
            raise ValueError(f"Invalid tool result block (allowed: {_TOOL_RESULT_TYPES}): {item!r}")
    if not blocks:
        raise ValueError("a ToolContent tool must return at least one nonempty block")
    kinds = {block["type"] for block in blocks}
    if "search_result" in kinds and kinds != {"search_result"}:
        raise ValueError("search_result blocks cannot be mixed with other blocks in one tool result")
    return blocks


def describe_content(content: str | Sequence[ContentBlock]) -> str:
    """A short text version of content blocks (no base64), e.g. for logs and ``after_tool`` hooks."""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        kind = block.get("type")
        source = block.get("source")
        source = source if isinstance(source, dict) else {}
        if kind == "text":
            parts.append(str(block.get("text", "")))
        elif kind in ("image", "document"):
            detail = next(
                (source[key] for key in ("media_type", "file_id", "url", "type") if source.get(key)), ""
            )
            title = f" {block['title']!r}" if block.get("title") else ""
            parts.append(f"[{kind}{title} {detail}]")
        elif kind == "search_result":
            body = " ".join(
                str(item.get("text", "")) for item in block.get("content", []) if isinstance(item, dict)
            )
            parts.append(f"[search_result {block.get('source')!r} {block.get('title')!r}] {body}")
        else:
            parts.append(f"[{kind}]")
    return "\n".join(parts)
