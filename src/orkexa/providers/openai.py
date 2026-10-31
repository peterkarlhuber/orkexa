"""OpenAI Responses API adapter (``pip install 'orkexa[openai]'``).

* Tool calling with strict JSON schemas (:func:`orkexa.core.schema.strict_schema`);
  the runtime runs every tool, the SDK never does.
* The conversation is replayed by Orkexa (``store=False`` by default): output items
  (messages, function calls, reasoning items with ``encrypted_content``) are kept
  verbatim in ``Part.meta["openai"]`` and sent back unchanged in the next request,
  as the Responses API requires for multi-round tool use with reasoning models.
* Structured output with a strict JSON schema (``text.format``), validated locally.
* ``effort`` maps to ``reasoning.effort`` (same value names); ``OpenAIOptions``
  holds everything else.

:class:`ResponsesCodec` is shared with OpenAI-compatible endpoints that implement
the Responses API (:mod:`orkexa.providers.compatible`).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from copy import deepcopy
import json
import re
from typing import Any, ClassVar

from ..core.capabilities import ModelCapabilities, Support
from ..core.conversion import document_text, search_result_text, validate_output
from ..core.schema import drop_null_optionals, strict_schema
from ..core.types import (
    Citation,
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
from ..errors import ConfigurationError, HistoryConversionError, ProviderServerError, UnexpectedResponseError
from .base import Provider, ProviderConfig, require_module
from .options import _OPENAI_RESERVED, OpenAIOptions

__all__ = ["OpenAIProvider", "ResponsesCodec", "openai_media", "openai_tool_choice"]

OPENAI_BASE_URL = "https://api.openai.com/v1"

#: Model families by documented capabilities (GPT-6 Astra, Sol, Terra and Luna: reasoning
#: with low to max effort, text and image input); other models stay "unknown" until
#: declared with ``capabilities=``.
_REASONING_MODELS = re.compile(r"^(gpt-6|gpt-5|o\d)")
_CLASSIC_MODELS = re.compile(r"^(gpt-4|gpt-3\.5|chatgpt-4o)")
_MULTIMODAL_MODELS = re.compile(r"^(gpt-6|gpt-5|gpt-4o|gpt-4\.1|o3|o4)")


def openai_tool_choice(choice: ToolChoice | None) -> Any:
    if choice is None or choice.mode == "auto":
        return "auto"
    if choice.mode == "tool":
        return {"type": "function", "name": choice.name}
    return choice.mode


def openai_tool(definition: ToolDefinition) -> dict[str, Any]:
    """A neutral tool as a Responses API function tool (strict schema when ``strict``)."""
    parameters = (
        strict_schema(definition.input_schema, name=definition.name)
        if definition.strict
        else definition.input_schema
    )
    return {
        "type": "function",
        "name": definition.name,
        "description": definition.description,
        "parameters": parameters,
        "strict": definition.strict,
    }


def openai_media(part: ImagePart | DocumentPart, provider: str) -> dict[str, Any]:
    """An image or PDF as a Responses API input content item."""
    source = part.source
    if source.type == "file":
        if (source.provider or "anthropic") != provider:
            raise HistoryConversionError(
                f"File {source.file_id!r} was uploaded to {source.provider or 'anthropic'}; "
                f"{provider} cannot read it. Send the file content instead."
            )
        kind = "input_image" if isinstance(part, ImagePart) else "input_file"
        return {"type": kind, "file_id": source.file_id}
    if isinstance(part, ImagePart):
        url = source.url if source.type == "url" else f"data:{source.media_type};base64,{source.data}"
        return {"type": "input_image", "image_url": url, "detail": "auto"}
    if source.type == "url":
        return {"type": "input_file", "file_url": source.url}
    return {
        "type": "input_file",
        "filename": part.title or "document.pdf",
        "file_data": f"data:{source.media_type or 'application/pdf'};base64,{source.data}",
    }


class ResponsesCodec:
    """Neutral messages <-> Responses API input/output items."""

    def __init__(self, provider: str) -> None:
        self.provider = provider

    # ---- encoding ---------------------------------------------------------------------

    def content(self, parts: Sequence[Any]) -> list[dict[str, Any]]:
        """User-side content parts as ``input_*`` items."""
        items: list[dict[str, Any]] = []
        for part in parts:
            if isinstance(part, TextPart):
                items.append({"type": "input_text", "text": part.text})
            elif isinstance(part, SearchResultPart):
                items.append({"type": "input_text", "text": search_result_text(part)})
            elif isinstance(part, DocumentPart) and part.source.type == "text":
                items.append({"type": "input_text", "text": document_text(part)})
            elif isinstance(part, (ImagePart, DocumentPart)):
                items.append(openai_media(part, self.provider))
            else:
                raise HistoryConversionError(f"{self.provider} cannot receive a {part.type!r} part as input")
        return items

    def tool_output(self, part: ToolResultPart) -> dict[str, Any]:
        output: Any
        if part.content is None:
            output = ""
        elif isinstance(part.content, str):
            output = part.content
        elif all(isinstance(p, TextPart) for p in part.content):
            output = "\n".join(p.text for p in part.content if isinstance(p, TextPart))
        else:
            output = self.content(part.content)
        if part.is_error and isinstance(output, str):
            output = f"Error: {output}" if not output.startswith("Error") else output
        return {"type": "function_call_output", "call_id": part.call_id, "output": output}

    def encode(self, messages: Sequence[Message]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "system":
                items.append({"role": "developer", "content": message.text})
            elif message.role == "user":
                if isinstance(message.content, str):
                    items.append({"role": "user", "content": message.content})
                    continue
                results = [p for p in message.content if isinstance(p, ToolResultPart)]
                others = [p for p in message.content if not isinstance(p, ToolResultPart)]
                items.extend(self.tool_output(p) for p in results)
                native = [p for p in others if isinstance(p, NativePart)]
                if native:
                    raise HistoryConversionError(
                        f"A {native[0].provider}-only block cannot be sent to {self.provider}"
                    )
                if others:
                    items.append({"role": "user", "content": self.content(others)})
            else:
                items.extend(self._assistant(message))
        return items

    def _assistant(self, message: Message) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        pending: list[str] = []

        def flush() -> None:
            if pending:
                items.append({"role": "assistant", "content": "\n".join(pending)})
                pending.clear()

        for part in message.parts():
            extras = part.native(self.provider)
            if "item_ref" in extras:
                continue
            if "item" in extras:
                flush()
                items.append(deepcopy(extras["item"]))
            elif isinstance(part, TextPart):
                pending.append(part.text)
            elif isinstance(part, ToolCallPart):
                flush()
                items.append(
                    {
                        "type": "function_call",
                        "call_id": part.id,
                        "name": part.name,
                        "arguments": json.dumps(part.arguments, ensure_ascii=False),
                    }
                )
            elif isinstance(part, NativePart) and part.provider == self.provider:
                flush()
                items.append(deepcopy(part.data))
            elif isinstance(part, ReasoningPart):
                continue  # reasoning without its native item cannot be replayed
            else:
                raise HistoryConversionError(f"A {part.type!r} part cannot be sent to {self.provider}")
        flush()
        return items

    # ---- decoding -----------------------------------------------------------------------

    def decode(self, output: Sequence[Any], request: ModelRequest) -> tuple[list[Part], bool, str | None]:
        """Output items -> (parts, has function calls, refusal text)."""
        schemas = {t.name: t.input_schema for t in request.tools if t.strict}
        parts: list[Part] = []
        calls = False
        refusal: str | None = None
        for item in output:
            data = _dump(item)
            kind = data.get("type")
            if kind == "message":
                first = True
                for content in data.get("content") or []:
                    if content.get("type") == "output_text":
                        part = TextPart(text=str(content.get("text", "")), citations=self._citations(content))
                    elif content.get("type") == "refusal":
                        refusal = str(content.get("refusal", ""))
                        part = TextPart(text=refusal)
                    else:
                        continue
                    # The first decoded part carries the whole item for replay, the rest refer to it.
                    meta = {"item": data} if first else {"item_ref": data.get("id")}
                    first = False
                    parts.append(part.with_native(self.provider, meta))
            elif kind == "function_call":
                calls = True
                parts.append(self._call(data, schemas).with_native(self.provider, {"item": data}))
            elif kind == "reasoning":
                texts = [str(s.get("text", "")) for s in data.get("summary") or []]
                texts += [
                    str(c.get("text", ""))
                    for c in data.get("content") or []
                    if c.get("type") == "reasoning_text"
                ]
                text = "\n".join(t for t in texts if t) or None
                parts.append(ReasoningPart(text=text).with_native(self.provider, {"item": data}))
            else:
                parts.append(NativePart(provider=self.provider, data=data))
        return parts, calls, refusal

    @staticmethod
    def _call(data: dict[str, Any], schemas: Mapping[str, dict[str, Any]]) -> ToolCallPart:
        name = str(data.get("name", ""))
        raw = data.get("arguments") or "{}"
        try:
            arguments = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError as exc:
            return ToolCallPart(
                id=str(data.get("call_id")), name=name, parse_error=f"invalid JSON arguments: {exc}"
            )
        if not isinstance(arguments, dict):
            return ToolCallPart(
                id=str(data.get("call_id")), name=name, parse_error="arguments are not a JSON object"
            )
        if name in schemas:
            # Strict mode made optional parameters required + nullable; a null means "default".
            arguments = drop_null_optionals(arguments, schemas[name])
        return ToolCallPart(id=str(data.get("call_id")), name=name, arguments=arguments)

    def _citations(self, content: dict[str, Any]) -> list[Citation] | None:
        text = str(content.get("text", ""))
        found: list[Citation] = []
        for note in content.get("annotations") or []:
            kind = note.get("type")
            if kind == "url_citation":
                start, end = note.get("start_index"), note.get("end_index")
                cited = text[start:end] if isinstance(start, int) and isinstance(end, int) else ""
                found.append(
                    Citation(
                        source=str(note.get("url", "")),
                        title=note.get("title"),
                        cited_text=cited,
                        provider=self.provider,
                        url=note.get("url"),
                    )
                )
            elif kind == "file_citation":
                found.append(
                    Citation(
                        source=str(note.get("filename") or note.get("file_id") or ""),
                        title=note.get("filename"),
                        cited_text="",
                        provider=self.provider,
                    )
                )
        return found or None


def _dump(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return deepcopy(item)
    data: dict[str, Any] = item.model_dump(mode="json", exclude_none=True, by_alias=True)
    return data


def _number(obj: Any, name: str) -> int:
    value = getattr(obj, name, None) if not isinstance(obj, dict) else obj.get(name)
    return int(value) if isinstance(value, int) else 0


#: Output items of hosted tools that OpenAI bills per call on top of tokens.
FEE_BEARING_CALLS = ("code_interpreter_call", "file_search_call", "image_generation_call", "web_search_call")


def responses_usage(usage: Any, response: Any = None, requested_tier: str | None = None) -> TokenUsage | None:
    """Normalized usage: cached and cache-write tokens are not counted as plain input.

    With ``response``, the service tier OpenAI reports (else ``requested_tier``) and
    the number of fee-bearing hosted-tool calls are recorded for pricing.
    """
    if usage is None:
        return None
    details = _field(usage, "input_tokens_details")
    output_details = _field(usage, "output_tokens_details")
    cached = _number(details, "cached_tokens")
    written = _number(details, "cache_write_tokens")
    tier = _field(response, "service_tier") or requested_tier
    calls: dict[str, int] = {}
    for item in _field(response, "output") or []:
        kind = _field(item, "type")
        if kind in FEE_BEARING_CALLS:
            calls[kind] = calls.get(kind, 0) + 1
    token_details = {
        "audio_input": _number(details, "audio_tokens"),
        "audio_output": _number(output_details, "audio_tokens"),
    }
    return TokenUsage(
        input_tokens=max(0, _number(usage, "input_tokens") - cached - written),
        output_tokens=_number(usage, "output_tokens"),
        cache_read_input_tokens=cached,
        cache_creation_input_tokens=written,
        cache_creation_5m_input_tokens=written,
        thinking_tokens=_number(output_details, "reasoning_tokens"),
        service_tier=tier if isinstance(tier, str) and tier else None,
        token_details=tuple((kind, count) for kind, count in token_details.items() if count),
        units=tuple(sorted(calls.items())),
    )


def _field(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


class OpenAIProvider(Provider):
    """Provider ``"openai"``: the OpenAI Responses API through the official SDK."""

    name: ClassVar[str] = "openai"
    options_type: ClassVar[type | None] = OpenAIOptions
    default_base_url: ClassVar[str | None] = OPENAI_BASE_URL
    api_key_variable: ClassVar[str | None] = "OPENAI_API_KEY"

    def __init__(self, config: ProviderConfig | None = None, asynchronous: bool = False) -> None:
        super().__init__(config, asynchronous)
        self.codec = ResponsesCodec(self.name)
        self.check_endpoint()

    def check_endpoint(self) -> None:
        config = self.config
        if (
            config.client is None
            and config.base_url not in (None, self.default_base_url)
            and config.api_key is None
        ):
            raise ConfigurationError(
                f"A custom base_url needs an explicit api_key: Orkexa never sends {self.api_key_variable} to "
                "another host. For local servers use provider='compatible' (or 'ollama'/'vllm')."
            )

    def make_client(self) -> Any:
        sdk = require_module("openai", "openai")
        kwargs = self.config.client_kwargs()
        kwargs.setdefault("base_url", self.default_base_url)  # never the OPENAI_BASE_URL environment variable
        cls = sdk.AsyncOpenAI if self.asynchronous else sdk.OpenAI
        return cls(**kwargs)

    # ---- capabilities ------------------------------------------------------------------

    def capabilities(self, model: str) -> ModelCapabilities:
        reasoning: Support = "unknown"
        if _REASONING_MODELS.match(model):
            reasoning = "supported"
        elif _CLASSIC_MODELS.match(model):
            reasoning = "unsupported"
        media: Support = "supported" if _MULTIMODAL_MODELS.match(model) else "unknown"
        return ModelCapabilities.all(
            "supported",
            reasons={
                "native_memory_tool": (
                    "OpenAI has no native memory tool; Orkexa offers memory as a function tool"
                ),
                "batch": "Orkexa does not implement the OpenAI Batch API; use evaluate_many(batch=False)",
                "reasoning": f"{model} is not a reasoning model; remove effort",
            },
            reasoning=reasoning,
            image_input=media,
            document_input=media,
            native_memory_tool="unsupported",
            batch="unsupported",
        )

    def check(self, request: ModelRequest) -> None:
        for tool in request.tools:
            openai_tool(tool)  # raises ToolSchemaError for schemas strict mode cannot express
        options = self._options(request)
        reserved = _OPENAI_RESERVED & set(options.request_options)
        if reserved:
            raise ConfigurationError(
                f"Set these via agent arguments, not request_options: {sorted(reserved)}"
            )
        if options.request_options.get("background"):
            raise ConfigurationError(
                "request_options={'background': True} is not supported: an agent needs each "
                "response to continue its turn and does not poll background responses"
            )

    def server_tool_name(self, definition: Mapping[str, Any]) -> str:
        return str(definition.get("name") or definition.get("server_label") or definition.get("type") or "")

    @staticmethod
    def _options(request: ModelRequest) -> OpenAIOptions:
        options = request.options
        return options if isinstance(options, OpenAIOptions) else OpenAIOptions()

    # ---- requests ----------------------------------------------------------------------

    def prepare(self, request: ModelRequest) -> dict[str, Any]:
        options = self._options(request)
        args: dict[str, Any] = {
            "model": request.model,
            "input": self.codec.encode(request.messages),
            "max_output_tokens": request.max_tokens,
            "store": options.store,
        }
        if request.system is not None:
            args["instructions"] = request.system
        tools = [openai_tool(t) for t in request.tools] + deepcopy(request.native_tools)
        if tools:
            args["tools"] = tools
            args["tool_choice"] = openai_tool_choice(request.tool_choice)
            if options.parallel_tool_calls is not None:
                args["parallel_tool_calls"] = options.parallel_tool_calls
        reasoning: dict[str, Any] = {}
        if request.effort is not None:
            reasoning["effort"] = request.effort
        if options.reasoning_summary is not None:
            reasoning["summary"] = options.reasoning_summary
        if reasoning:
            args["reasoning"] = reasoning
        caps = self.request_capabilities(request)
        if options.include_encrypted_reasoning and not options.store and caps.reasoning == "supported":
            args["include"] = ["reasoning.encrypted_content"]
        if options.service_tier is not None:
            args["service_tier"] = options.service_tier
        if request.output_format is not None:
            # Orkexa's strict schema (not the SDK's parse()): optional fields become
            # nullable and get their defaults back in local validation.
            name = request.output_format.__name__
            args["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": name,
                    "schema": strict_schema(request.output_format.model_json_schema(), name=name),
                    "strict": True,
                }
            }
        for key, value in deepcopy(options.request_options).items():
            if isinstance(value, dict) and isinstance(args.get(key), dict):
                args[key] = {**args[key], **value}
            else:
                args[key] = value
        return args

    def call(self, native: dict[str, Any], request: ModelRequest) -> ModelResponse:
        api = self.client.responses
        return self.normalize(api.create(**native), request)

    async def acall(self, native: dict[str, Any], request: ModelRequest) -> ModelResponse:
        api = self.client.responses
        return self.normalize(await api.create(**native), request)

    def stream(self, native: dict[str, Any], request: ModelRequest) -> Iterator[StreamEvent]:
        state = _ResponsesStream()
        with self.client.responses.stream(**native) as stream:
            for event in stream:
                yield from state.translate(event)
            final = state.unfinished or stream.get_final_response()
        yield ResponseCompleted(self.normalize(final, request))

    async def astream(self, native: dict[str, Any], request: ModelRequest) -> AsyncIterator[StreamEvent]:
        state = _ResponsesStream()
        async with self.client.responses.stream(**native) as stream:
            async for event in stream:
                for translated in state.translate(event):
                    yield translated
            final = state.unfinished or await stream.get_final_response()
        yield ResponseCompleted(self.normalize(final, request))

    def normalize(self, response: Any, request: ModelRequest) -> ModelResponse:
        status = getattr(response, "status", None)
        if status in ("queued", "in_progress"):
            raise UnexpectedResponseError(
                f"{self.name}: the response is still {status!r} (background responses are not polled)"
            )
        if status == "failed":
            error = getattr(response, "error", None)
            raise ProviderServerError(
                self.redact(
                    f"{self.name}: the response failed: {getattr(error, 'code', None)}: "
                    f"{getattr(error, 'message', error)}"
                ),
                provider=self.name,
                retryable=True,
            )
        parts, calls, refusal = self.codec.decode(getattr(response, "output", None) or [], request)
        stop_reason: StopReason = "tool_use" if calls else "end_turn"
        native_stop: str | None = status
        details = getattr(response, "incomplete_details", None)
        stop_details: Any = None
        if status == "incomplete":
            reason = getattr(details, "reason", None)
            native_stop = str(reason)
            if reason == "max_output_tokens":
                stop_reason = "max_tokens"
            elif reason == "content_filter":
                stop_reason = "refusal"
                stop_details = RefusalDetails(category="content_filter")
            else:
                stop_reason = "other"
        elif status not in (None, "completed"):
            stop_reason = "other"  # "cancelled" or a status this version does not know: not a final answer
        if refusal is not None:
            stop_reason = "refusal"
            stop_details = RefusalDetails(explanation=refusal)
        model = str(getattr(response, "model", None) or request.model)
        message = Message(role="assistant", content=parts, provider=self.name, model=model)
        parsed, error = None, None
        if request.output_format is not None and stop_reason == "end_turn":
            parsed, error = validate_output(request.output_format, message.text, strict_schema=True)
        request_id = getattr(response, "_request_id", None)
        return ModelResponse(
            message=message,
            stop_reason=stop_reason,
            provider=self.name,
            model=model,
            native_stop_reason=native_stop,
            parsed=parsed,
            parse_error=error,
            stop_details=stop_details,
            usage=responses_usage(
                getattr(response, "usage", None), response, self._options(request).service_tier
            ),
            request_id=request_id if isinstance(request_id, str) else getattr(response, "id", None),
            raw=response,
        )

    # ---- token counting and errors --------------------------------------------------------

    def _count_args(self, request: ModelRequest) -> dict[str, Any]:
        native = self.prepare(request)
        allowed = {
            "model",
            "input",
            "instructions",
            "tools",
            "tool_choice",
            "reasoning",
            "parallel_tool_calls",
        }
        return {key: value for key, value in native.items() if key in allowed}

    def count_tokens(self, request: ModelRequest) -> int:
        return int(self.client.responses.input_tokens.count(**self._count_args(request)).input_tokens)

    async def acount_tokens(self, request: ModelRequest) -> int:
        result = await self.client.responses.input_tokens.count(**self._count_args(request))
        return int(result.input_tokens)

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


class _ResponsesStream:
    """Translates Responses API stream events into normalized events."""

    def __init__(self) -> None:
        self.calls: dict[str, str] = {}  # item id -> call id
        #: The final response of a stream that ended without ``response.completed``; the
        #: SDK's ``get_final_response()`` raises a bare ``RuntimeError`` for it.
        self.unfinished: Any = None

    def translate(self, event: Any) -> Iterator[StreamEvent]:
        kind = getattr(event, "type", None)
        if kind in ("response.incomplete", "response.failed"):
            self.unfinished = event.response
        elif kind == "response.output_text.delta":
            yield TextDelta(str(event.delta))
        elif kind in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
            if event.delta:
                yield ReasoningDelta(str(event.delta))
        elif kind == "response.output_item.added":
            item = event.item
            if getattr(item, "type", None) == "function_call":
                self.calls[str(item.id)] = str(item.call_id)
                yield ToolCallStarted(
                    call_id=str(item.call_id), name=str(item.name), index=int(event.output_index)
                )
        elif kind == "response.function_call_arguments.delta":
            call_id = self.calls.get(str(event.item_id), str(event.item_id))
            yield ToolCallDelta(
                call_id=call_id, index=int(event.output_index), arguments_delta=str(event.delta)
            )
