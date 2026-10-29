"""The Chat Completions protocol (``/v1/chat/completions``) through the OpenAI SDK.

Used by :class:`~orkexa.providers.compatible.CompatibleProvider` (and its Ollama and
vLLM presets) in the default ``api="chat_completions"`` mode. The protocol has no
field for replaying reasoning, so reasoning a server returns (``reasoning_content``)
is shown in streaming and kept in the history, but not sent back.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence
from copy import deepcopy
import json
from typing import Any, ClassVar

from ..core.capabilities import ModelCapabilities
from ..core.conversion import document_text, search_result_text, validate_output
from ..core.schema import drop_null_optionals, strict_schema
from ..core.types import (
    DocumentPart,
    ImagePart,
    Message,
    ModelRequest,
    ModelResponse,
    NativePart,
    Part,
    ReasoningDelta,
    ReasoningPart,
    RefusalDetails,
    ResponseCompleted,
    SearchResultPart,
    StopReason,
    StreamEvent,
    TextDelta,
    TextPart,
    TokenUsage,
    ToolCallDelta,
    ToolCallPart,
    ToolCallStarted,
    ToolChoice,
    ToolDefinition,
    ToolResultPart,
)
from ..errors import ConfigurationError, HistoryConversionError, UnexpectedResponseError
from .base import Provider, require_module
from .options import CompatibleOptions

__all__ = ["ChatCompletionsProvider"]

_FINISH: dict[str, StopReason] = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "length": "max_tokens",
    "content_filter": "refusal",
}
_RESERVED = frozenset(
    {
        "model",
        "messages",
        "tools",
        "tool_choice",
        "response_format",
        "stream",
        "max_tokens",
        "max_completion_tokens",
    }
)


def chat_tool_choice(choice: ToolChoice | None) -> Any:
    if choice is None or choice.mode == "auto":
        return "auto"
    if choice.mode == "tool":
        return {"type": "function", "function": {"name": choice.name}}
    return choice.mode


def _get(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    value = getattr(obj, name, None)
    if value is None:
        extra = getattr(obj, "model_extra", None) or {}
        value = extra.get(name)
    return value


def _int(obj: Any, name: str) -> int:
    value = _get(obj, name)
    return int(value) if isinstance(value, int) else 0


def chat_usage(usage: Any) -> TokenUsage | None:
    if usage is None:
        return None
    cached = _int(_get(usage, "prompt_tokens_details"), "cached_tokens")
    return TokenUsage(
        input_tokens=max(0, _int(usage, "prompt_tokens") - cached),
        output_tokens=_int(usage, "completion_tokens"),
        cache_read_input_tokens=cached,
        thinking_tokens=_int(_get(usage, "completion_tokens_details"), "reasoning_tokens"),
    )


class ChatCompletionsProvider(Provider):
    """Base adapter for the Chat Completions protocol (subclasses set name and capabilities)."""

    name: ClassVar[str] = "chat_completions"
    options_type: ClassVar[type | None] = CompatibleOptions

    @staticmethod
    def _options(request: ModelRequest) -> CompatibleOptions:
        options = request.options
        return options if isinstance(options, CompatibleOptions) else CompatibleOptions()

    def make_client(self) -> Any:
        sdk = require_module("openai", "openai")
        cls = sdk.AsyncOpenAI if self.asynchronous else sdk.OpenAI
        return cls(**self.config.client_kwargs())

    def capabilities(self, model: str) -> ModelCapabilities:
        return ModelCapabilities()

    def check(self, request: ModelRequest) -> None:
        options = self._options(request)
        reserved = _RESERVED & set(options.request_options)
        if reserved:
            raise ConfigurationError(
                f"Set these via agent arguments, not request_options: {sorted(reserved)}"
            )
        if request.effort is not None and not options.send_reasoning_effort:
            raise ConfigurationError(
                "effort on an OpenAI-compatible endpoint needs CompatibleOptions(send_reasoning_effort=True) "
                "(and capabilities={'reasoning': 'supported'}); not every server understands reasoning_effort"
            )
        if request.native_tools:
            raise ConfigurationError(
                "Chat Completions has no server tools; use api='responses' if the endpoint has them"
            )
        caps = self.capabilities(request.model)
        strict = caps.strict_tool_schemas == "supported"
        for tool in request.tools:
            self.tool(tool, strict=strict)

    # ---- encoding ----------------------------------------------------------------------

    def tool(self, definition: ToolDefinition, *, strict: bool) -> dict[str, Any]:
        function: dict[str, Any] = {"name": definition.name, "description": definition.description}
        if strict and definition.strict:
            function["parameters"] = strict_schema(definition.input_schema, name=definition.name)
            function["strict"] = True
        else:
            function["parameters"] = deepcopy(definition.input_schema)
        return {"type": "function", "function": function}

    def content(self, parts: Sequence[Any]) -> str | list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for part in parts:
            if isinstance(part, TextPart):
                items.append({"type": "text", "text": part.text})
            elif isinstance(part, SearchResultPart):
                items.append({"type": "text", "text": search_result_text(part)})
            elif isinstance(part, DocumentPart) and part.source.type == "text":
                items.append({"type": "text", "text": document_text(part)})
            elif isinstance(part, ImagePart) and part.source.type in ("base64", "url"):
                source = part.source
                url = source.url if source.type == "url" else f"data:{source.media_type};base64,{source.data}"
                items.append({"type": "image_url", "image_url": {"url": url}})
            elif isinstance(part, DocumentPart) and part.source.type == "base64":
                items.append(
                    {
                        "type": "file",
                        "file": {
                            "filename": part.title or "document.pdf",
                            "file_data": f"data:{part.source.media_type};base64,{part.source.data}",
                        },
                    }
                )
            else:
                raise HistoryConversionError(
                    f"{self.name} (Chat Completions) cannot receive a {part.type!r} part "
                    f"({getattr(getattr(part, 'source', None), 'type', '')} source)"
                )
        if all(item["type"] == "text" for item in items):
            return "\n".join(item["text"] for item in items)
        return items

    def tool_message(self, part: ToolResultPart) -> dict[str, Any]:
        if part.content is None:
            text = ""
        elif isinstance(part.content, str):
            text = part.content
        else:
            rendered = self.content(part.content)
            if not isinstance(rendered, str):
                raise HistoryConversionError(
                    f"{self.name} (Chat Completions) accepts only text tool results; the "
                    f"result of {part.name or part.call_id} contains images or files"
                )
            text = rendered
        if part.is_error and not text.startswith("Error"):
            text = f"Error: {text}"
        return {"role": "tool", "tool_call_id": part.call_id, "content": text}

    def messages(self, request: ModelRequest, system: str | None) -> list[dict[str, Any]]:
        encoded: list[dict[str, Any]] = []
        if system:
            encoded.append({"role": "system", "content": system})
        for message in request.messages:
            if message.role == "system":
                encoded.append({"role": "system", "content": message.text})
            elif message.role == "user":
                if isinstance(message.content, str):
                    encoded.append({"role": "user", "content": message.content})
                    continue
                for part in message.content:
                    if isinstance(part, NativePart):
                        raise HistoryConversionError(
                            f"A {part.provider}-only block cannot be sent to {self.name}"
                        )
                results = [p for p in message.content if isinstance(p, ToolResultPart)]
                others = [p for p in message.content if not isinstance(p, ToolResultPart)]
                encoded.extend(self.tool_message(p) for p in results)
                if others:
                    encoded.append({"role": "user", "content": self.content(others)})
            else:
                encoded.append(self._assistant(message))
        return encoded

    def _assistant(self, message: Message) -> dict[str, Any]:
        texts: list[str] = []
        calls: list[dict[str, Any]] = []
        for part in message.parts():
            if isinstance(part, TextPart):
                texts.append(part.text)
            elif isinstance(part, ToolCallPart):
                calls.append(
                    {
                        "id": part.id,
                        "type": "function",
                        "function": {
                            "name": part.name,
                            "arguments": json.dumps(part.arguments, ensure_ascii=False),
                        },
                    }
                )
            elif isinstance(part, ReasoningPart):
                continue  # Chat Completions has no field to replay reasoning
            else:
                raise HistoryConversionError(f"A {part.type!r} part cannot be sent to {self.name}")
        encoded: dict[str, Any] = {"role": "assistant", "content": "\n".join(texts) if texts else None}
        if calls:
            encoded["tool_calls"] = calls
        return encoded

    def prepare(self, request: ModelRequest) -> dict[str, Any]:
        options = self._options(request)
        caps = self.capabilities(request.model)
        system = request.system
        args: dict[str, Any] = {"model": request.model}
        if request.output_format is not None:
            schema = request.output_format.model_json_schema()
            if options.structured_output == "json_object":
                instruction = "Respond only with a JSON object that matches this JSON schema:\n" + json.dumps(
                    schema, ensure_ascii=False
                )
                system = f"{system}\n\n{instruction}" if system else instruction
                args["response_format"] = {"type": "json_object"}
            else:
                strict = caps.strict_tool_schemas == "supported"
                args["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": request.output_format.__name__,
                        "schema": strict_schema(schema, name=request.output_format.__name__)
                        if strict
                        else schema,
                        **({"strict": True} if strict else {}),
                    },
                }
        args["messages"] = self.messages(request, system)
        args[options.max_tokens_param] = request.max_tokens
        if request.tools:
            strict = caps.strict_tool_schemas == "supported"
            args["tools"] = [self.tool(t, strict=strict) for t in request.tools]
            args["tool_choice"] = chat_tool_choice(request.tool_choice)
        if request.effort is not None:
            args["reasoning_effort"] = request.effort
        for key, value in deepcopy(options.request_options).items():
            args[key] = value
        return args

    # ---- responses ---------------------------------------------------------------------

    def _call(self, data: Any, request: ModelRequest) -> ToolCallPart:
        function = _get(data, "function")
        name = str(_get(function, "name") or "")
        call_id = str(_get(data, "id") or "")
        raw = _get(function, "arguments") or "{}"
        try:
            arguments = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError as exc:
            return ToolCallPart(id=call_id, name=name, parse_error=f"invalid JSON arguments: {exc}")
        if not isinstance(arguments, dict):
            return ToolCallPart(id=call_id, name=name, parse_error="arguments are not a JSON object")
        strict = self.capabilities(request.model).strict_tool_schemas == "supported"
        schemas = {t.name: t.input_schema for t in request.tools if t.strict}
        if strict and name in schemas:
            arguments = drop_null_optionals(arguments, schemas[name])
        return ToolCallPart(id=call_id, name=name, arguments=arguments)

    def build(
        self,
        *,
        content: str | None,
        reasoning: str | None,
        refusal: str | None,
        calls: list[ToolCallPart],
        finish: str | None,
        usage: Any,
        model: str,
        request: ModelRequest,
        raw: Any,
        request_id: str | None,
    ) -> ModelResponse:
        parts: list[Part] = []
        if reasoning:
            parts.append(ReasoningPart(text=reasoning))
        if refusal:
            parts.append(TextPart(text=refusal))
        elif content:
            parts.append(TextPart(text=content))
        parts.extend(calls)
        stop_reason: StopReason = _FINISH.get(str(finish), "other") if finish is not None else "end_turn"
        if calls and stop_reason == "end_turn":
            stop_reason = "tool_use"  # several servers report "stop" with tool calls
        stop_details = None
        if refusal:
            stop_reason = "refusal"
            stop_details = RefusalDetails(explanation=refusal)
        elif stop_reason == "refusal":
            stop_details = RefusalDetails(category="content_filter")
        message = Message(role="assistant", content=parts, provider=self.name, model=model)
        parsed, error = None, None
        if request.output_format is not None and stop_reason == "end_turn":
            parsed, error = validate_output(request.output_format, content or "")
        return ModelResponse(
            message=message,
            stop_reason=stop_reason,
            provider=self.name,
            model=model,
            native_stop_reason=finish,
            parsed=parsed,
            parse_error=error,
            stop_details=stop_details,
            usage=chat_usage(usage),
            request_id=request_id,
            raw=raw,
        )

    def normalize(self, response: Any, request: ModelRequest) -> ModelResponse:
        choices = _get(response, "choices") or []
        if not choices:
            raise UnexpectedResponseError(f"{self.name}: the response has no choices")
        choice = choices[0]
        message = _get(choice, "message")
        calls = [self._call(call, request) for call in _get(message, "tool_calls") or []]
        return self.build(
            content=_get(message, "content"),
            reasoning=_get(message, "reasoning_content") or _get(message, "reasoning"),
            refusal=_get(message, "refusal"),
            calls=calls,
            finish=_get(choice, "finish_reason"),
            usage=_get(response, "usage"),
            model=str(_get(response, "model") or request.model),
            request=request,
            raw=response,
            request_id=getattr(response, "_request_id", None) or _get(response, "id"),
        )

    def call(self, native: dict[str, Any], request: ModelRequest) -> ModelResponse:
        return self.normalize(self.client.chat.completions.create(**native), request)

    async def acall(self, native: dict[str, Any], request: ModelRequest) -> ModelResponse:
        return self.normalize(await self.client.chat.completions.create(**native), request)

    def _stream_args(self, native: dict[str, Any]) -> dict[str, Any]:
        args = dict(native)
        args["stream"] = True
        args.setdefault("stream_options", {"include_usage": True})
        return args

    def stream(self, native: dict[str, Any], request: ModelRequest) -> Iterator[StreamEvent]:
        state = _ChatStream()
        stream = self.client.chat.completions.create(**self._stream_args(native))
        try:
            for chunk in stream:
                yield from state.translate(chunk)
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                close()
        yield ResponseCompleted(state.final(self, request))

    async def astream(self, native: dict[str, Any], request: ModelRequest) -> AsyncIterator[StreamEvent]:
        state = _ChatStream()
        stream = await self.client.chat.completions.create(**self._stream_args(native))
        try:
            async for chunk in stream:
                for event in state.translate(chunk):
                    yield event
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                await close()
        yield ResponseCompleted(state.final(self, request))

    def map_error(self, exc: BaseException) -> BaseException | None:
        try:
            sdk = require_module("openai", "openai")
        except ImportError:
            return None
        if isinstance(exc, (sdk.APIConnectionError, sdk.APITimeoutError)):
            return self.provider_error(exc, status_code=None, connection=True)
        if isinstance(exc, sdk.APIStatusError):
            return self.provider_error(
                exc, status_code=exc.status_code, request_id=getattr(exc, "request_id", None)
            )
        return None


class _ChatStream:
    """Accumulates Chat Completions chunks into deltas and one final response."""

    def __init__(self) -> None:
        self.content: list[str] = []
        self.reasoning: list[str] = []
        self.refusal: list[str] = []
        self.calls: dict[int, dict[str, Any]] = {}
        self.finish: str | None = None
        self.usage: Any = None
        self.model: str | None = None
        self.id: str | None = None

    def translate(self, chunk: Any) -> Iterator[StreamEvent]:
        self.model = _get(chunk, "model") or self.model
        self.id = _get(chunk, "id") or self.id
        if _get(chunk, "usage") is not None:
            self.usage = _get(chunk, "usage")
        for choice in _get(chunk, "choices") or []:
            delta = _get(choice, "delta")
            if _get(choice, "finish_reason"):
                self.finish = _get(choice, "finish_reason")
            text = _get(delta, "content")
            if text:
                self.content.append(text)
                yield TextDelta(text)
            thought = _get(delta, "reasoning_content") or _get(delta, "reasoning")
            if thought:
                self.reasoning.append(thought)
                yield ReasoningDelta(thought)
            refusal = _get(delta, "refusal")
            if refusal:
                self.refusal.append(refusal)
            for call in _get(delta, "tool_calls") or []:
                index = int(_get(call, "index") or 0)
                entry = self.calls.setdefault(index, {"id": None, "name": "", "arguments": ""})
                function = _get(call, "function")
                if _get(call, "id"):
                    entry["id"] = _get(call, "id")
                if _get(function, "name"):
                    entry["name"] += _get(function, "name")
                    yield ToolCallStarted(call_id=str(entry["id"] or index), name=entry["name"], index=index)
                if _get(function, "arguments"):
                    entry["arguments"] += _get(function, "arguments")
                    yield ToolCallDelta(
                        call_id=str(entry["id"] or index),
                        index=index,
                        arguments_delta=_get(function, "arguments"),
                    )

    def final(self, provider: ChatCompletionsProvider, request: ModelRequest) -> ModelResponse:
        calls = [
            provider._call(
                {
                    "id": entry["id"] or f"call_{index}",
                    "function": {"name": entry["name"], "arguments": entry["arguments"]},
                },
                request,
            )
            for index, entry in sorted(self.calls.items())
        ]
        return provider.build(
            content="".join(self.content) or None,
            reasoning="".join(self.reasoning) or None,
            refusal="".join(self.refusal) or None,
            calls=calls,
            finish=self.finish,
            usage=self.usage,
            model=self.model or request.model,
            request=request,
            raw=None,
            request_id=self.id,
        )
