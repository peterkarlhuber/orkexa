"""Prompt parsing, cross-provider history rules and the versioned history file format."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
import json
from typing import Any, Literal
import warnings

from pydantic import BaseModel, TypeAdapter, ValidationError

from ..content import ContentBlock
from ..errors import HistoryConversionError
from .types import (
    DocumentPart,
    ImagePart,
    Message,
    NativePart,
    Part,
    ReasoningPart,
    ResultContentPart,
    SearchResultPart,
    TextPart,
    ToolResultPart,
)

__all__ = [
    "HISTORY_FORMAT",
    "HISTORY_VERSION",
    "BlockParser",
    "HistoryConversionWarning",
    "HistoryPolicy",
    "dump_history",
    "parse_block",
    "parse_history",
    "prepare_history",
    "result_content",
    "to_parts",
    "validate_output",
]

HISTORY_FORMAT = "orkexa.history"
HISTORY_VERSION = 1

HistoryPolicy = Literal["strict", "convert"]
BlockParser = Callable[[ContentBlock], Part]

_PART: TypeAdapter[Part] = TypeAdapter(Part)
_RESULT_PART: TypeAdapter[ResultContentPart] = TypeAdapter(ResultContentPart)
_MESSAGES: TypeAdapter[list[Message]] = TypeAdapter(list[Message])
_INPUT_TYPES = ("text", "image", "document", "search_result")


class HistoryConversionWarning(UserWarning):
    """Parts were dropped because ``history_policy="convert"`` allowed it."""


def parse_block(block: ContentBlock) -> Part:
    """Parse one input block dict (``image(...)``, ``document(...)``, ...) strictly.

    Raises:
        ValueError: unknown block type or fields. Provider-specific fields (e.g.
            Anthropic's ``cache_control``) are only accepted by that provider's agent.
    """
    if not isinstance(block, dict) or block.get("type") not in _INPUT_TYPES:
        raise ValueError(f"Invalid content block (allowed types: {_INPUT_TYPES}): {block!r}")
    try:
        return _PART.validate_python(block)
    except ValidationError as exc:
        raise ValueError(f"Invalid {block['type']} block: {exc}") from None


def to_parts(content: str | Sequence[ContentBlock], parser: BlockParser = parse_block) -> str | list[Part]:
    """Turn normalized prompt content (see :func:`orkexa.content.to_content`) into parts."""
    if isinstance(content, str):
        return content
    return [parser(dict(block)) for block in content]


def result_content(
    content: str | Sequence[ContentBlock], parser: BlockParser = parse_block
) -> str | list[ResultContentPart]:
    """Tool output (text or ``ToolContent`` blocks) as tool-result content."""
    if isinstance(content, str):
        return content
    parts: list[ResultContentPart] = []
    for block in content:
        part = parser(dict(block))
        parts.append(_RESULT_PART.validate_python(part))
    return parts


def _file_issuer(part: Part) -> str | None:
    if isinstance(part, (ImagePart, DocumentPart)) and part.source.type == "file":
        return part.source.provider or "anthropic"
    return None


def _check_files(parts: Iterable[Any], provider: str) -> None:
    for part in parts:
        issuer = _file_issuer(part)
        if issuer is not None and issuer != provider:
            raise HistoryConversionError(
                f"The conversation references a file uploaded to {issuer} "
                f"(file_id={part.source.file_id!r}); file ids cannot be used with {provider}. "
                "Send the file content instead (e.g. image(path) or document(path))."
            )
        if isinstance(part, ToolResultPart) and isinstance(part.content, list):
            _check_files(part.content, provider)


def prepare_history(
    messages: Sequence[Message], provider: str, policy: HistoryPolicy, model: str | None = None
) -> list[Message]:
    """The conversation as ``model`` of ``provider`` may receive it.

    Messages produced by the same provider and model pass unchanged. In messages
    of other providers, reasoning parts and provider-only parts cannot be replayed:
    with ``policy="strict"`` they raise :class:`HistoryConversionError`, with
    ``policy="convert"`` they are dropped (with a :class:`HistoryConversionWarning`).
    Text, media, tool calls and tool results are converted losslessly; foreign meta
    is ignored by the encoders. File ids of another provider always raise.

    Reasoning is also bound to the model that produced it (signatures and
    encrypted reasoning are only valid for that model). When ``model`` is given
    and a message of the same provider names another model, its reasoning parts
    are dropped under both policies with a :class:`HistoryConversionWarning`: no
    visible content is lost, and the target model could not use them. Provider
    parts (server tool calls and results) stay. Messages without a model (e.g. a
    ``claude-agent`` history) are treated as coming from the current model.

    Message boundaries and roles are kept: messages are never merged, and a
    message is only removed when none of its parts can be replayed.
    """
    prepared: list[Message] = []
    dropped: dict[str, int] = {}
    for message in messages:
        parts = message.content if isinstance(message.content, list) else []
        _check_files(parts, provider)
        origin = message.provider
        if origin is None or not parts:
            prepared.append(message)
            continue
        if origin == provider:
            if model is None or message.model is None or message.model == model:
                prepared.append(message)
                continue
            kept = [part for part in parts if not isinstance(part, ReasoningPart)]
            if len(kept) < len(parts):
                kind = f"{origin}:{message.model} reasoning"
                dropped[kind] = dropped.get(kind, 0) + len(parts) - len(kept)
            if kept:
                prepared.append(message.model_copy(update={"content": kept}))
            continue
        kept = []
        for part in parts:
            if isinstance(part, (ReasoningPart, NativePart)):
                kind = "native" if isinstance(part, NativePart) else "reasoning"
                if policy == "strict":
                    raise HistoryConversionError(
                        f"The conversation contains {kind} content produced by {origin} that {provider} "
                        "cannot replay. Pass history_policy='convert' to drop such parts "
                        "(the conversation then continues without them)."
                    )
                dropped[f"{origin} {kind}"] = dropped.get(f"{origin} {kind}", 0) + 1
                continue
            kept.append(part)
        if kept:
            prepared.append(message.model_copy(update={"content": kept}))
        else:
            dropped[f"{origin} empty message"] = dropped.get(f"{origin} empty message", 0) + 1
    if dropped:
        summary = ", ".join(f"{count} {kind} part(s)" for kind, count in dropped.items())
        target = f"{provider}:{model}" if model is not None else provider
        warnings.warn(
            f"dropped {summary} that {target} cannot replay",
            HistoryConversionWarning,
            stacklevel=3,
        )
    return prepared


# ---- history files ---------------------------------------------------------------------


def dump_history(messages: Sequence[Message]) -> dict[str, Any]:
    """The versioned JSON document :meth:`Agent.save_history` writes."""
    return {
        "format": HISTORY_FORMAT,
        "version": HISTORY_VERSION,
        "messages": [m.model_dump(mode="json", exclude_none=True) for m in messages],
    }


def parse_history(
    data: Any, legacy: Callable[[list[dict[str, Any]]], list[Message]], source: str
) -> list[Message]:
    """Messages from an Orkexa history document or a ``claude-agent`` history list.

    Raises:
        ValueError: ``data`` is neither (the error names ``source``).
    """
    if isinstance(data, list):
        if not all(isinstance(m, dict) and m.get("role") in ("user", "assistant", "system") for m in data):
            raise ValueError(f"{source}: not a conversation history")
        return legacy(data)
    if isinstance(data, dict) and data.get("format") == HISTORY_FORMAT:
        version = data.get("version")
        if version != HISTORY_VERSION:
            raise ValueError(
                f"{source}: unsupported history version {version!r} (supported: {HISTORY_VERSION})"
            )
        try:
            return _MESSAGES.validate_python(data.get("messages"))
        except ValidationError as exc:
            raise ValueError(f"{source}: invalid history: {exc}") from None
    raise ValueError(f"{source}: not a conversation history")


def validate_output(
    output_format: type[BaseModel], text: str, *, strict_schema: bool = False
) -> tuple[Any, str | None]:
    """Validate model output ``text`` locally against ``output_format`` -> ``(instance, error)``.

    With ``strict_schema=True`` (the request used :func:`orkexa.core.schema.strict_schema`,
    where optional properties are required but nullable), ``null`` values of optional
    properties are removed first, so they get their Pydantic defaults.
    """
    try:
        if strict_schema:
            from .schema import drop_null_optionals

            data = json.loads(text)
            if isinstance(data, dict):
                data = drop_null_optionals(data, output_format.model_json_schema())
            return output_format.model_validate(data), None
        return output_format.model_validate_json(text), None
    except ValidationError as exc:
        return None, str(exc)
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc}"


def to_json(document: Any) -> str:
    return json.dumps(document, ensure_ascii=False, indent=1)


def search_result_text(part: SearchResultPart) -> str:
    """A search result as attributed text for providers without search-result blocks."""
    body = "\n".join(p.text for p in part.content)
    return f"[Search result | source: {part.source} | title: {part.title}]\n{body}"


def document_text(part: DocumentPart) -> str:
    """A plain-text document as attributed text."""
    title = f" {part.title!r}" if part.title else ""
    context = f"\nContext: {part.context}" if part.context else ""
    return f"[Document{title}]{context}\n{part.source.data or ''}"


def is_text_document(part: Part) -> bool:
    return isinstance(part, DocumentPart) and part.source.type == "text"


def plain_text(parts: Sequence[Any]) -> str:
    """Text, text documents and search results of ``parts`` as one string (other parts noted)."""
    lines: list[str] = []
    for part in parts:
        if isinstance(part, TextPart):
            lines.append(part.text)
        elif isinstance(part, SearchResultPart):
            lines.append(search_result_text(part))
        elif isinstance(part, DocumentPart) and part.source.type == "text":
            lines.append(document_text(part))
        else:
            lines.append(f"[{getattr(part, 'type', 'content')}]")
    return "\n".join(lines)
