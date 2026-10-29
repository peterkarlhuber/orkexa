"""A scripted provider without SDK or network, for tests and examples.

``FakeProvider`` proves that the runtime works without any vendor SDK::

    from orkexa import Agent
    from orkexa.providers.fake import FakeModel, text, tool_calls

    model = FakeModel([tool_calls(("add", {"a": 1, "b": 2})), text("3")])
    agent = Agent(model="fake:scripted", client=model, tools=[add])
    assert agent.ask("1 + 2?") == "3"
    model.requests  # every ModelRequest the runtime sent

A script entry is a :class:`~orkexa.core.types.ModelResponse`, a string (a text
answer), an exception (raised by the "API"), or a callable ``(request) -> entry``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from copy import deepcopy
import json
from typing import Any, ClassVar

from ..core.capabilities import ModelCapabilities, Support
from ..core.conversion import validate_output
from ..core.types import (
    Message,
    ModelRequest,
    ModelResponse,
    Part,
    ReasoningDelta,
    ReasoningPart,
    ResponseCompleted,
    StopReason,
    StreamEvent,
    TextDelta,
    TextPart,
    TokenUsage,
    ToolCallDelta,
    ToolCallPart,
    ToolCallStarted,
    ToolResultPart,
)
from ..errors import ConfigurationError
from .base import Provider, ProviderConfig

__all__ = ["FakeModel", "FakeProvider", "response", "text", "tool_calls"]

ScriptEntry = ModelResponse | str | BaseException | Callable[[ModelRequest], Any]


def response(
    parts: Sequence[Part],
    *,
    stop_reason: StopReason = "end_turn",
    usage: TokenUsage | None = None,
    parsed: Any = None,
) -> ModelResponse:
    """A scripted response with the given parts."""
    return ModelResponse(
        message=Message(role="assistant", content=list(parts)),
        stop_reason=stop_reason,
        provider="fake",
        model="",
        native_stop_reason=stop_reason,
        parsed=parsed,
        usage=usage if usage is not None else TokenUsage(input_tokens=10, output_tokens=5),
    )


def text(value: str, *, stop_reason: StopReason = "end_turn", reasoning: str | None = None) -> ModelResponse:
    """A text answer (optionally preceded by visible reasoning)."""
    parts: list[Part] = []
    if reasoning is not None:
        parts.append(ReasoningPart(text=reasoning))
    parts.append(TextPart(text=value))
    return response(parts, stop_reason=stop_reason)


def tool_calls(
    *calls: tuple[str, dict[str, Any]] | tuple[str, dict[str, Any], str], prefix: str = ""
) -> ModelResponse:
    """A response that calls tools: ``tool_calls(("add", {"a": 1}), ("mul", {"a": 2}, "id-7"))``."""
    parts: list[Part] = []
    for position, call in enumerate(calls):
        name, arguments = call[0], call[1]
        call_id = call[2] if len(call) > 2 else f"call_{prefix}{position}"
        parts.append(ToolCallPart(id=call_id, name=name, arguments=dict(arguments)))
    return response(parts, stop_reason="tool_use")


class FakeModel:
    """The scripted "API" behind :class:`FakeProvider` (pass it as ``client=``).

    Args:
        script: Responses in order (see the module docstring).
        capabilities: Capability values (default: everything ``supported``).
        count: What ``count_tokens`` returns.
    """

    def __init__(
        self,
        script: Sequence[ScriptEntry] = (),
        *,
        capabilities: Mapping[str, Support] | None = None,
        count: int = 42,
    ) -> None:
        self.script: list[ScriptEntry] = list(script)
        self.requests: list[ModelRequest] = []
        self.capabilities = ModelCapabilities.all("supported").merged(dict(capabilities or {}))
        self.count = count

    def add(self, *entries: ScriptEntry) -> None:
        self.script.extend(entries)

    def next(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(deepcopy(request))
        if not self.script:
            raise AssertionError("FakeModel: no scripted response left")
        entry: Any = self.script.pop(0)
        if callable(entry) and not isinstance(entry, (ModelResponse, BaseException)):
            entry = entry(request)
        if isinstance(entry, BaseException):
            raise entry
        if isinstance(entry, str):
            entry = text(entry)
        if not isinstance(entry, ModelResponse):
            raise TypeError(f"FakeModel: invalid script entry {entry!r}")
        return entry


def check_tool_pairing(messages: Sequence[Message]) -> None:
    """Every tool result answers a tool call of the directly preceding assistant message."""
    open_calls: set[str] = set()
    for message in messages:
        parts = message.parts()
        results = [p for p in parts if isinstance(p, ToolResultPart)]
        if results:
            if {r.call_id for r in results} != open_calls:
                raise AssertionError(
                    f"tool results {[r.call_id for r in results]} do not match calls {open_calls}"
                )
            open_calls = set()
        elif open_calls and message.role == "user":
            raise AssertionError(f"tool calls {open_calls} were not answered")
        if message.role == "assistant":
            open_calls = {p.id for p in parts if isinstance(p, ToolCallPart)}


class FakeProvider(Provider):
    """Provider ``"fake"``: answers from a :class:`FakeModel` script."""

    name: ClassVar[str] = "fake"

    def __init__(self, config: ProviderConfig | None = None, asynchronous: bool = False) -> None:
        super().__init__(config, asynchronous)
        if self._client is None:
            self._client = FakeModel()

    @property
    def model(self) -> FakeModel:
        client = self.client
        if not isinstance(client, FakeModel):
            raise ConfigurationError("The fake provider needs client=FakeModel([...])")
        return client

    def capabilities(self, model: str) -> ModelCapabilities:
        return self.model.capabilities

    def prepare(self, request: ModelRequest) -> dict[str, Any]:
        check_tool_pairing(request.messages)
        return {
            "model": request.model,
            "messages": len(request.messages),
            "tools": [t.name for t in request.tools],
        }

    def _respond(self, request: ModelRequest) -> ModelResponse:
        result = self.model.next(request)
        message = result.message.model_copy(update={"provider": self.name, "model": request.model})
        parsed, error = result.parsed, result.parse_error
        if request.output_format is not None and parsed is None and result.stop_reason != "refusal":
            parsed, error = validate_output(request.output_format, message.text)
        return ModelResponse(
            message=message,
            stop_reason=result.stop_reason,
            provider=self.name,
            model=request.model,
            native_stop_reason=result.native_stop_reason,
            parsed=parsed,
            parse_error=error,
            stop_details=result.stop_details,
            usage=result.usage,
            request_id=result.request_id,
            raw=result,
        )

    def call(self, native: dict[str, Any], request: ModelRequest) -> ModelResponse:
        return self._respond(request)

    async def acall(self, native: dict[str, Any], request: ModelRequest) -> ModelResponse:
        return self._respond(request)

    def stream(self, native: dict[str, Any], request: ModelRequest) -> Iterator[StreamEvent]:
        final = self._respond(request)
        yield from _events(final)

    async def astream(self, native: dict[str, Any], request: ModelRequest) -> AsyncIterator[StreamEvent]:
        final = self._respond(request)
        for event in _events(final):
            yield event

    def count_tokens(self, request: ModelRequest) -> int:
        return self.model.count

    async def acount_tokens(self, request: ModelRequest) -> int:
        return self.model.count


def _events(final: ModelResponse) -> Iterator[StreamEvent]:
    """The deltas a streaming API would send for ``final``, then the final response."""
    for index, part in enumerate(final.message.parts()):
        if isinstance(part, TextPart):
            middle = max(1, len(part.text) // 2)
            for piece in (part.text[:middle], part.text[middle:]):
                if piece:
                    yield TextDelta(piece)
        elif isinstance(part, ReasoningPart) and part.text:
            yield ReasoningDelta(part.text)
        elif isinstance(part, ToolCallPart):
            yield ToolCallStarted(call_id=part.id, name=part.name, index=index)
            yield ToolCallDelta(call_id=part.id, index=index, arguments_delta=json.dumps(part.arguments))
    yield ResponseCompleted(final)
