"""The provider-neutral data model of the agent runtime.

The runtime, the tools and every subsystem work only with these types. Provider
adapters translate them to and from their SDKs. Native details that a provider needs
again in a later request (thinking signatures, item ids, encrypted reasoning, thought
signatures, cache markers) are kept per provider in ``Part.meta`` and restored when
the conversation is sent to the same provider again.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Self

__all__ = [
    "Citation",
    "CitationsConfig",
    "DocumentPart",
    "Effort",
    "ImagePart",
    "MediaSource",
    "Message",
    "ModelRequest",
    "ModelResponse",
    "NativePart",
    "Part",
    "ReasoningDelta",
    "ReasoningPart",
    "RefusalDetails",
    "ResponseCompleted",
    "ResultContentPart",
    "Role",
    "SearchResultPart",
    "StopReason",
    "StreamEvent",
    "TextDelta",
    "TextPart",
    "TokenUsage",
    "ToolCallDelta",
    "ToolCallPart",
    "ToolCallStarted",
    "ToolChoice",
    "ToolDefinition",
    "ToolResultPart",
    "TurnCompleted",
    "TurnFailed",
    "UsageReported",
]

Effort = Literal["low", "medium", "high", "xhigh", "max"]
"""Generic reasoning effort; adapters map it only to a matching provider concept."""

Role = Literal["user", "assistant", "system"]

StopReason = Literal[
    "end_turn",
    "tool_use",
    "max_tokens",
    "stop_sequence",
    "refusal",
    "pause_turn",
    "context_window_exceeded",
    "other",
]
"""Normalized reason why the model stopped; ``native_stop_reason`` keeps the original."""


# ---- citations ---------------------------------------------------------------------


class Citation(BaseModel):
    """A source reference attached to an answer.

    ``kind`` tells where the reference comes from: ``"api_native"`` citations were
    returned by the provider API (Anthropic search-result/document citations, OpenAI
    URL/file annotations); ``"retrieval"`` sources are Orkexa-verified hits that the
    search tool actually returned in this turn (:func:`orkexa.retrieval.retrieval_sources`).
    Source claims a model writes into its text are neither and are never parsed.

    For Anthropic citations ``cited_text`` is the cited source passage; OpenAI URL
    annotations carry no source excerpt, so it holds the annotated part of the answer.
    """

    source: str
    title: str | None = None
    cited_text: str
    search_result_index: int | None = None
    kind: Literal["api_native", "retrieval"] = "api_native"
    provider: str | None = None
    url: str | None = None


class CitationsConfig(BaseModel):
    """``{"enabled": true}`` on a document or search result."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool


# ---- parts ---------------------------------------------------------------------------


class _PartBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: dict[str, dict[str, Any]] | None = None
    """Provider-specific extras, keyed by provider name; restored only for that provider."""

    def native(self, provider: str) -> dict[str, Any]:
        """The extras stored for ``provider`` (empty if there are none)."""
        return dict((self.meta or {}).get(provider, {}))

    def with_native(self, provider: str, extras: dict[str, Any]) -> Self:
        """A copy whose ``meta[provider]`` is ``extras`` (removed if empty)."""
        meta = {k: v for k, v in (self.meta or {}).items() if k != provider}
        if extras:
            meta[provider] = extras
        return self.model_copy(update={"meta": meta or None})


class MediaSource(BaseModel):
    """Where an image or document comes from."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["base64", "url", "text", "file"]
    media_type: str | None = None
    data: str | None = None
    url: str | None = None
    file_id: str | None = None
    provider: str | None = None
    """Issuer of ``file_id``; file ids are never sent to another provider."""


class TextPart(_PartBase):
    type: Literal["text"] = "text"
    text: str
    citations: list[Citation] | None = None


class ImagePart(_PartBase):
    type: Literal["image"] = "image"
    source: MediaSource


class DocumentPart(_PartBase):
    type: Literal["document"] = "document"
    source: MediaSource
    title: str | None = None
    context: str | None = None
    citations: CitationsConfig | None = None


class SearchResultPart(_PartBase):
    """A retrieved passage with its source (from :class:`orkexa.retrieval.Hit`)."""

    type: Literal["search_result"] = "search_result"
    source: str
    title: str
    content: list[TextPart]
    citations: CitationsConfig | None = None


ResultContentPart = Annotated[
    TextPart | ImagePart | DocumentPart | SearchResultPart, Field(discriminator="type")
]


class ToolCallPart(_PartBase):
    """A tool call the model requested; ``id`` links it to its :class:`ToolResultPart`."""

    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    parse_error: str | None = None
    """Set if the provider returned arguments that are not a JSON object."""


class ToolResultPart(_PartBase):
    type: Literal["tool_result"] = "tool_result"
    call_id: str
    name: str | None = None
    content: str | list[ResultContentPart] | None = None
    is_error: bool = False


class ReasoningPart(_PartBase):
    """Model reasoning (thinking, reasoning summary, thought); ``text`` may be omitted."""

    type: Literal["reasoning"] = "reasoning"
    text: str | None = None
    redacted: bool = False


class NativePart(_PartBase):
    """A provider-only block (server tool calls and results, compaction, ...), kept verbatim."""

    type: Literal["native"] = "native"
    provider: str
    data: dict[str, Any]


Part = Annotated[
    TextPart
    | ImagePart
    | DocumentPart
    | SearchResultPart
    | ToolCallPart
    | ToolResultPart
    | ReasoningPart
    | NativePart,
    Field(discriminator="type"),
]


class Message(BaseModel):
    """One message of a conversation.

    ``provider``/``model`` record which provider produced an assistant message, so a
    conversation can be checked before it is sent to another provider.
    """

    model_config = ConfigDict(extra="forbid")

    role: Role
    content: str | list[Part]
    provider: str | None = None
    model: str | None = None

    def parts(self) -> list[Part]:
        """The content as parts (a string becomes one text part)."""
        if isinstance(self.content, str):
            return [TextPart(text=self.content)]
        return list(self.content)

    @property
    def text(self) -> str:
        """The text parts joined with newlines."""
        return "\n".join(p.text for p in self.parts() if isinstance(p, TextPart))


# ---- requests and responses ------------------------------------------------------------


@dataclass(frozen=True)
class ToolDefinition:
    """A provider-neutral function tool: name, description and JSON input schema."""

    name: str
    description: str
    input_schema: dict[str, Any]
    strict: bool = True


@dataclass(frozen=True)
class ToolChoice:
    """Whether and which tool the model must call.

    ``ToolChoice("auto")`` lets the model decide, ``"none"`` forbids tools,
    ``"required"`` forces some tool call, ``ToolChoice("tool", name="x")`` forces ``x``.
    """

    mode: Literal["auto", "none", "required", "tool"] = "auto"
    name: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("auto", "none", "required", "tool"):
            raise ValueError(f"Invalid tool choice mode: {self.mode!r}")
        if (self.mode == "tool") != (self.name is not None):
            raise ValueError("ToolChoice needs a name exactly when mode='tool'")

    @property
    def forced(self) -> bool:
        return self.mode in ("required", "tool")

    @classmethod
    def parse(cls, value: ToolChoice | str | dict[str, Any]) -> ToolChoice:
        """``"auto"``, ``"none"``, ``"required"``, ``{"name": "x"}`` or a ``ToolChoice``."""
        if isinstance(value, ToolChoice):
            return value
        if isinstance(value, str):
            return cls(value)  # type: ignore[arg-type]
        if isinstance(value, dict) and set(value) == {"name"}:
            return cls("tool", name=str(value["name"]))
        raise ValueError(f"Invalid tool_choice {value!r}; use 'auto', 'none', 'required' or {{'name': ...}}")


@dataclass(frozen=True)
class RefusalDetails:
    """Why the model declined (``category``/``explanation`` as far as the provider tells)."""

    category: str | None = None
    explanation: str | None = None


@dataclass(frozen=True)
class TokenUsage:
    """Token counts of one response, normalized so nothing is counted twice.

    ``input_tokens`` excludes cache reads and cache writes; ``output_tokens`` includes
    reasoning tokens (``thinking_tokens`` is the part of it spent on reasoning).
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_creation_5m_input_tokens: int = 0
    cache_creation_1h_input_tokens: int = 0
    thinking_tokens: int = 0
    web_search_requests: int = 0
    web_fetch_requests: int = 0


@dataclass
class ModelRequest:
    """Everything the runtime asks a provider for in one model request."""

    model: str
    messages: list[Message]
    system: str | None = None
    tools: list[ToolDefinition] = field(default_factory=list)
    native_tools: list[dict[str, Any]] = field(default_factory=list)
    """Provider-native tool definitions (server tools, MCP toolsets, the native memory tool)."""
    tool_choice: ToolChoice | None = None
    output_format: type[BaseModel] | None = None
    max_tokens: int = 16_000
    stream: bool = False
    effort: Effort | None = None
    options: Any = None
    """The agent's typed provider options (e.g. ``AnthropicOptions``), or ``None``."""
    history_policy: Literal["strict", "convert"] = "strict"
    capabilities: Any = None
    """The effective :class:`~orkexa.ModelCapabilities` (provider defaults merged with the
    agent's ``capabilities=`` overrides); adapters base encoding decisions on it."""


@dataclass
class ModelResponse:
    """One normalized model response."""

    message: Message
    stop_reason: StopReason
    provider: str
    model: str
    native_stop_reason: str | None = None
    parsed: Any = None
    """The validated ``output_format`` instance, if structured output was requested."""
    parse_error: str | None = None
    """Why the structured output did not validate (then ``parsed`` is ``None``)."""
    stop_details: Any = None
    usage: TokenUsage | None = None
    request_id: str | None = None
    raw: Any = None
    """The provider's native response object."""

    @property
    def text(self) -> str:
        return self.message.text

    @property
    def tool_calls(self) -> list[ToolCallPart]:
        return [p for p in self.message.parts() if isinstance(p, ToolCallPart)]

    @property
    def citations(self) -> list[Citation]:
        found: list[Citation] = []
        for part in self.message.parts():
            if isinstance(part, TextPart) and part.citations:
                found.extend(part.citations)
        return found


# ---- stream events ------------------------------------------------------------------------


@dataclass(frozen=True)
class TextDelta:
    """A piece of answer text."""

    text: str
    type: ClassVar[str] = "text_delta"


@dataclass(frozen=True)
class ReasoningDelta:
    """A piece of visible reasoning (thinking text or a reasoning summary)."""

    text: str
    type: ClassVar[str] = "reasoning_delta"


@dataclass(frozen=True)
class ToolCallStarted:
    """The model started a tool call."""

    call_id: str
    name: str
    index: int
    type: ClassVar[str] = "tool_call_started"


@dataclass(frozen=True)
class ToolCallDelta:
    """A fragment of a tool call's JSON arguments, as streamed."""

    call_id: str
    index: int
    arguments_delta: str
    type: ClassVar[str] = "tool_call_delta"


@dataclass(frozen=True)
class ResponseCompleted:
    """The final response of one model request (identical to the non-streamed result)."""

    response: ModelResponse
    type: ClassVar[str] = "response_completed"


@dataclass(frozen=True)
class UsageReported:
    """The usage record of one model request was added to the ledger."""

    record: Any
    type: ClassVar[str] = "usage"


@dataclass(frozen=True)
class TurnCompleted:
    """``ask()`` finished; ``output`` is what it returns."""

    output: Any
    type: ClassVar[str] = "turn_completed"


@dataclass(frozen=True)
class TurnFailed:
    """``ask()`` raised ``error``; the conversation history was not updated."""

    error: BaseException
    type: ClassVar[str] = "turn_failed"


StreamEvent = (
    TextDelta
    | ReasoningDelta
    | ToolCallStarted
    | ToolCallDelta
    | ResponseCompleted
    | UsageReported
    | TurnCompleted
    | TurnFailed
)


def text_of(parts: Sequence[Any]) -> str:
    """The text parts of ``parts`` joined with newlines."""
    return "\n".join(p.text for p in parts if isinstance(p, TextPart))
