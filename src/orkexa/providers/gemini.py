"""Google Gemini adapter through the official ``google-genai`` SDK (``pip install 'orkexa[gemini]'``).

* API family: ``models.generate_content`` / ``generate_content_stream`` of the Gemini
  Developer API (``api_key``, ``GEMINI_API_KEY``/``GOOGLE_API_KEY``). Vertex AI works
  with a preconfigured ``client=genai.Client(vertexai=True, project=..., location=...)``.
* Function calling is manual: ``automatic_function_calling`` is always disabled and
  no Python callables are passed to the SDK, so every call goes through Orkexa's
  approval hooks, limits and budget checks.
* Native parts are kept in ``Part.meta["gemini"]`` and replayed unchanged, including
  ``thought_signature`` (required by Gemini 3 for multi-round function calling).
* ``effort`` maps to ``thinking_level`` (``low``/``medium``/``high``; Gemini 3 and 4 models);
  ``"xhigh"``/``"max"`` have no Gemini equivalent and are rejected. Gemini 2.5 models
  take ``GeminiOptions(thinking_budget=...)`` instead.
* Gemini does not enforce tool schemas: Orkexa validates every call locally and
  returns invalid arguments to the model as a tool error.
"""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from copy import deepcopy
import re
from typing import Any, ClassVar

from ..core.capabilities import ModelCapabilities, Support
from ..core.conversion import document_text, search_result_text, validate_output
from ..core.schema import inline_refs
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
    ToolResultPart,
)
from ..errors import ConfigurationError, HistoryConversionError, UnsupportedCapabilityError
from .base import Provider, ProviderConfig, require_module
from .options import GeminiOptions

__all__ = ["GeminiProvider"]

_PROVIDER = "gemini"
_LEVELS = {"low": "LOW", "medium": "MEDIUM", "high": "HIGH"}
_REFUSALS = frozenset(
    {
        "SAFETY",
        "PROHIBITED_CONTENT",
        "BLOCKLIST",
        "SPII",
        "RECITATION",
        "IMAGE_SAFETY",
        "IMAGE_PROHIBITED_CONTENT",
        "IMAGE_RECITATION",
    }
)
_THINKING_LEVEL = re.compile(r"^(models/)?gemini-[34]([.-]|$)")  # Gemini 3.x and 4 (e.g. gemini-4-argon)
_GEMINI_2 = re.compile(r"^(models/)?gemini-(2|1)\.")
_RESERVED = frozenset(
    {
        "system_instruction",
        "tools",
        "tool_config",
        "automatic_function_calling",
        "response_mime_type",
        "response_json_schema",
        "response_schema",
        "max_output_tokens",
        "thinking_config",
    }
)


def _value(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def _enum_name(value: Any) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "name", value))


def _modality(usage: Any, name: str, modality: str) -> int:
    total = 0
    for entry in _value(usage, name) or []:
        if _enum_name(_value(entry, "modality")) == modality:
            count = _value(entry, "token_count")
            total += count if isinstance(count, int) else 0
    return total


def gemini_usage(usage: Any, grounding: Any = None) -> TokenUsage | None:
    """Normalized usage: cached tokens are not plain input; output includes thoughts.

    Audio and generated-image tokens are reported as special token classes, and
    Google Search grounding (``grounding``: the candidate's ``grounding_metadata``)
    as the units ``google_search_query`` and ``google_search_grounded_prompt``, so
    prices without rates for them do not produce seemingly complete costs.
    """
    if usage is None:
        return None

    def number(name: str) -> int:
        value = _value(usage, name)
        return int(value) if isinstance(value, int) else 0

    cached = number("cached_content_token_count")
    thoughts = number("thoughts_token_count")
    cached_audio = _modality(usage, "cache_tokens_details", "AUDIO")
    details = {
        "audio_input": max(0, _modality(usage, "prompt_tokens_details", "AUDIO") - cached_audio),
        "audio_cache_read": cached_audio,
        "audio_output": _modality(usage, "candidates_tokens_details", "AUDIO"),
        "image_output": _modality(usage, "candidates_tokens_details", "IMAGE"),
    }
    queries = _value(grounding, "web_search_queries") or []
    units = {"google_search_query": len(queries), "google_search_grounded_prompt": 1 if queries else 0}
    return TokenUsage(
        input_tokens=max(0, number("prompt_token_count") - cached) + number("tool_use_prompt_token_count"),
        output_tokens=number("candidates_token_count") + thoughts,
        cache_read_input_tokens=cached,
        thinking_tokens=thoughts,
        token_details=tuple((kind, count) for kind, count in details.items() if count),
        units=tuple((unit, count) for unit, count in sorted(units.items()) if count),
    )


class GeminiProvider(Provider):
    """Provider ``"gemini"``: Google Gemini through ``google-genai``."""

    name: ClassVar[str] = _PROVIDER
    options_type: ClassVar[type | None] = GeminiOptions

    def __init__(self, config: ProviderConfig | None = None, asynchronous: bool = False) -> None:
        super().__init__(config, asynchronous)
        if self.config.base_url is not None and self.config.api_key is None and self.config.client is None:
            raise ConfigurationError(
                "A custom base_url needs an explicit api_key: Orkexa never sends "
                "GEMINI_API_KEY/GOOGLE_API_KEY "
                "to another host"
            )

    def make_client(self) -> Any:
        genai = require_module("google.genai", "gemini")
        kwargs: dict[str, Any] = dict(self.config.extra)
        if self.config.api_key is not None:
            kwargs["api_key"] = self.config.secret()
        http: dict[str, Any] = dict(kwargs.pop("http_options", {}) or {})
        if self.config.base_url is not None:
            http["base_url"] = self.config.base_url
        if self.config.timeout is not None:
            http["timeout"] = int(self.config.timeout * 1000)  # milliseconds
        if http:
            kwargs["http_options"] = http
        return genai.Client(**kwargs)

    @property
    def types(self) -> Any:
        return require_module("google.genai.types", "gemini")

    # ---- capabilities ---------------------------------------------------------------

    def capabilities(self, model: str) -> ModelCapabilities:
        reasoning: Support = "unknown"
        reasons = {
            "forced_tool_choice": "Gemini forces only 'some tool' or a list of tools",
            "system_messages": "Gemini has no system messages inside the conversation; use system_prompt",
            "native_memory_tool": "Gemini has no native memory tool; Orkexa offers memory as a function tool",
            "strict_tool_schemas": "Gemini does not enforce tool schemas; Orkexa validates arguments locally",
            "batch": "Orkexa does not implement Gemini batch mode; use evaluate_many(batch=False)",
        }
        if _THINKING_LEVEL.match(model):
            reasoning = "supported"
        elif _GEMINI_2.match(model):
            reasoning = "unsupported"
            reasons["reasoning"] = (
                f"{model} takes a thinking budget, not an effort level: "
                "use GeminiOptions(thinking_budget=...)"
            )
        return ModelCapabilities.all(
            "supported",
            reasons=reasons,
            reasoning=reasoning,
            system_messages="unsupported",
            native_memory_tool="unsupported",
            strict_tool_schemas="unsupported",
            batch="unsupported",
            caching="supported",
        )

    def check(self, request: ModelRequest) -> None:
        options = self._options(request)
        if request.effort in ("xhigh", "max"):
            raise UnsupportedCapabilityError(
                f"Gemini's highest thinking level is 'high'; effort={request.effort!r} has no equivalent",
                capability="reasoning",
            )
        if request.effort is not None and options.thinking_budget is not None:
            raise ConfigurationError(
                "Use either effort (thinking_level) or GeminiOptions(thinking_budget=...), not both"
            )
        reserved = _RESERVED & set(options.request_options)
        if reserved:
            raise ConfigurationError(
                f"Set these via agent arguments, not request_options: {sorted(reserved)}"
            )
        for tool in request.tools:
            inline_refs(tool.input_schema, name=tool.name)
        if request.output_format is not None:
            inline_refs(request.output_format.model_json_schema(), name=request.output_format.__name__)

    def server_tool_name(self, definition: Mapping[str, Any]) -> str:
        explicit = definition.get("name")
        if explicit:
            return str(explicit)
        keys = [key for key in definition if key != "type"]
        return str(definition.get("type") or (keys[0] if keys else ""))

    @staticmethod
    def _options(request: ModelRequest) -> GeminiOptions:
        options = request.options
        return options if isinstance(options, GeminiOptions) else GeminiOptions()

    # ---- encoding -----------------------------------------------------------------------

    def _media(self, part: ImagePart | DocumentPart) -> Any:
        types = self.types
        source = part.source
        if source.type == "base64":
            return types.Part(
                inline_data=types.Blob(mime_type=source.media_type, data=base64.b64decode(source.data or ""))
            )
        if source.type == "url":
            mime = source.media_type or ("application/pdf" if isinstance(part, DocumentPart) else None)
            return types.Part(file_data=types.FileData(file_uri=source.url, mime_type=mime))
        if source.type == "file" and source.provider == _PROVIDER:
            return types.Part(file_data=types.FileData(file_uri=source.file_id, mime_type=source.media_type))
        raise HistoryConversionError(
            f"File {source.file_id!r} was uploaded to {source.provider or 'anthropic'}; Gemini cannot read it"
        )

    def _content_parts(self, parts: Sequence[Any]) -> list[Any]:
        types = self.types
        encoded: list[Any] = []
        for part in parts:
            if isinstance(part, TextPart):
                encoded.append(types.Part(text=part.text))
            elif isinstance(part, SearchResultPart):
                encoded.append(types.Part(text=search_result_text(part)))
            elif isinstance(part, DocumentPart) and part.source.type == "text":
                encoded.append(types.Part(text=document_text(part)))
            elif isinstance(part, (ImagePart, DocumentPart)):
                encoded.append(self._media(part))
            elif isinstance(part, NativePart) and part.provider == _PROVIDER:
                encoded.append(types.Part.model_validate(part.data))
            else:
                raise HistoryConversionError(f"Gemini cannot receive a {part.type!r} part")
        return encoded

    def _function_response(self, part: ToolResultPart, names: Mapping[str, str], synthetic: set[str]) -> Any:
        types = self.types
        name = part.name or names.get(part.call_id)
        if not name:
            raise HistoryConversionError(f"Gemini needs the tool name of result {part.call_id!r}")
        media: list[Any] = []
        if part.content is None:
            text = ""
        elif isinstance(part.content, str):
            text = part.content
        else:
            texts = [
                p
                for p in part.content
                if not isinstance(p, (ImagePart, DocumentPart)) or p.source.type == "text"
            ]
            text = "\n".join(
                p.text
                if isinstance(p, TextPart)
                else search_result_text(p)
                if isinstance(p, SearchResultPart)
                else document_text(p)
                if isinstance(p, DocumentPart)
                else ""
                for p in texts
            )
            for item in part.content:
                if isinstance(item, (ImagePart, DocumentPart)) and item.source.type == "base64":
                    media.append(
                        types.FunctionResponsePart(
                            inline_data=types.FunctionResponseBlob(
                                mime_type=item.source.media_type,
                                data=base64.b64decode(item.source.data or ""),
                            )
                        )
                    )
                elif isinstance(item, (ImagePart, DocumentPart)) and item.source.type != "text":
                    raise HistoryConversionError(
                        "Gemini tool results accept only inline images and documents"
                    )
        response = {"error": text} if part.is_error else {"result": text}
        return types.Part(
            function_response=types.FunctionResponse(
                id=None if part.call_id in synthetic else part.call_id,
                name=name,
                response=response,
                parts=media or None,
            )
        )

    def contents(self, messages: Sequence[Message], model: str | None = None) -> list[Any]:
        """Gemini ``contents``; thought signatures are only replayed to ``model``."""
        types = self.types
        names: dict[str, str] = {}
        synthetic: set[str] = set()
        contents: list[Any] = []
        for message in messages:
            if message.role == "system":
                raise UnsupportedCapabilityError(
                    "Gemini has no system messages inside the conversation", capability="system_messages"
                )
            if isinstance(message.content, str):
                role = "model" if message.role == "assistant" else "user"
                contents.append(types.Content(role=role, parts=[types.Part(text=message.content)]))
                continue
            if message.role == "user":
                results = [p for p in message.content if isinstance(p, ToolResultPart)]
                others = [p for p in message.content if not isinstance(p, ToolResultPart)]
                parts = [self._function_response(p, names, synthetic) for p in results]
                parts += self._content_parts(others)
                contents.append(types.Content(role="user", parts=parts))
                continue
            parts = []
            own_model = model is None or message.model is None or message.model == model
            for part in message.content:
                extras = part.native(_PROVIDER)
                if not own_model and "thought_signature" in extras.get("part", {}):
                    # A signature is only valid for the model that produced it.
                    extras["part"] = {k: v for k, v in extras["part"].items() if k != "thought_signature"}
                if isinstance(part, ToolCallPart):
                    names[part.id] = part.name
                    if extras.get("synthetic_id"):
                        synthetic.add(part.id)
                if "part" in extras:
                    parts.append(types.Part.model_validate(deepcopy(extras["part"])))
                elif isinstance(part, TextPart):
                    parts.append(types.Part(text=part.text))
                elif isinstance(part, ToolCallPart):
                    parts.append(
                        types.Part(
                            function_call=types.FunctionCall(id=part.id, name=part.name, args=part.arguments)
                        )
                    )
                elif isinstance(part, NativePart) and part.provider == _PROVIDER:
                    parts.append(types.Part.model_validate(deepcopy(part.data)))
                elif isinstance(part, ReasoningPart):
                    continue  # reasoning of another origin cannot be replayed
                else:
                    raise HistoryConversionError(f"A {part.type!r} part cannot be sent to Gemini")
            if parts:
                contents.append(types.Content(role="model", parts=parts))
        return contents

    def generation_config(self, request: ModelRequest) -> dict[str, Any]:
        options = self._options(request)
        config: dict[str, Any] = {
            "max_output_tokens": request.max_tokens,
            "automatic_function_calling": {"disable": True},
        }
        if request.system is not None:
            config["system_instruction"] = request.system
        tools: list[dict[str, Any]] = []
        if request.tools:
            tools.append(
                {
                    "function_declarations": [
                        {
                            "name": t.name,
                            "description": t.description,
                            "parameters_json_schema": inline_refs(t.input_schema, name=t.name),
                        }
                        for t in request.tools
                    ]
                }
            )
        tools += deepcopy(request.native_tools)
        if tools:
            config["tools"] = tools
        if request.tools and request.tool_choice is not None:
            mode = {"auto": "AUTO", "none": "NONE", "required": "ANY", "tool": "ANY"}[
                request.tool_choice.mode
            ]
            calling: dict[str, Any] = {"mode": mode}
            if request.tool_choice.mode == "tool":
                calling["allowed_function_names"] = [request.tool_choice.name]
            config["tool_config"] = {"function_calling_config": calling}
        thinking: dict[str, Any] = {}
        if request.effort is not None:
            thinking["thinking_level"] = _LEVELS[request.effort]
        if options.thinking_budget is not None:
            thinking["thinking_budget"] = options.thinking_budget
        if options.include_thoughts:
            thinking["include_thoughts"] = True
        if thinking:
            config["thinking_config"] = thinking
        if request.output_format is not None:
            config["response_mime_type"] = "application/json"
            config["response_json_schema"] = inline_refs(
                request.output_format.model_json_schema(), name=request.output_format.__name__
            )
        if options.safety_settings is not None:
            config["safety_settings"] = deepcopy(options.safety_settings)
        config.update(deepcopy(options.request_options))
        return config

    def prepare(self, request: ModelRequest) -> dict[str, Any]:
        return {
            "model": request.model,
            "contents": self.contents(request.messages, request.model),
            "config": self.generation_config(request),
        }

    # ---- responses ---------------------------------------------------------------------

    def _parts(self, raw_parts: Sequence[Any]) -> tuple[list[Part], bool]:
        parts: list[Part] = []
        calls = False
        for position, raw in enumerate(raw_parts):
            data = raw if isinstance(raw, dict) else raw.model_dump(mode="json", exclude_none=True)
            meta: dict[str, Any] = {"part": data}
            call = data.get("function_call")
            if call:
                calls = True
                call_id = call.get("id")
                if not call_id:
                    call_id = f"gemini_call_{position}_{call.get('name')}"
                    meta["synthetic_id"] = True
                args = call.get("args") or {}
                part: Part = ToolCallPart(
                    id=str(call_id), name=str(call.get("name", "")), arguments=dict(args)
                )
            elif "text" in data and data.get("thought"):
                part = ReasoningPart(text=data.get("text") or None)
            elif "text" in data:
                part = TextPart(text=str(data["text"]))
            elif data.get("thought_signature") and len(data) == 1:
                part = ReasoningPart(text=None)  # a bare signature part
            else:
                parts.append(NativePart(provider=_PROVIDER, data=data))
                continue
            parts.append(part.with_native(_PROVIDER, meta))
        return parts, calls

    def build(
        self,
        *,
        raw_parts: Sequence[Any],
        finish: str | None,
        usage: Any,
        block_reason: str | None,
        model: str,
        request: ModelRequest,
        raw: Any,
        request_id: str | None,
        grounding: Any = None,
    ) -> ModelResponse:
        parts, calls = self._parts(raw_parts)
        stop_reason: StopReason
        stop_details: Any = None
        if block_reason:
            stop_reason = "refusal"
            stop_details = RefusalDetails(category=block_reason)
        elif finish in (None, "STOP", "FINISH_REASON_UNSPECIFIED"):
            stop_reason = "tool_use" if calls else "end_turn"
        elif finish == "MAX_TOKENS":
            stop_reason = "max_tokens"
        elif finish in _REFUSALS:
            stop_reason = "refusal"
            stop_details = RefusalDetails(category=finish)
        else:
            stop_reason = "other"
        message = Message(role="assistant", content=parts, provider=_PROVIDER, model=model)
        parsed, error = None, None
        if request.output_format is not None and stop_reason == "end_turn":
            parsed, error = validate_output(request.output_format, message.text)
        return ModelResponse(
            message=message,
            stop_reason=stop_reason,
            provider=_PROVIDER,
            model=model,
            native_stop_reason=finish or block_reason,
            parsed=parsed,
            parse_error=error,
            stop_details=stop_details,
            usage=gemini_usage(usage, grounding),
            request_id=request_id,
            raw=raw,
        )

    def normalize(self, response: Any, request: ModelRequest) -> ModelResponse:
        candidates = _value(response, "candidates") or []
        candidate = candidates[0] if candidates else None
        content = _value(candidate, "content")
        feedback = _value(response, "prompt_feedback")
        return self.build(
            raw_parts=_value(content, "parts") or [],
            finish=_enum_name(_value(candidate, "finish_reason")),
            usage=_value(response, "usage_metadata"),
            block_reason=_enum_name(_value(feedback, "block_reason")) if not candidates else None,
            model=str(_value(response, "model_version") or request.model),
            grounding=_value(candidate, "grounding_metadata"),
            request=request,
            raw=response,
            request_id=_value(response, "response_id"),
        )

    def call(self, native: dict[str, Any], request: ModelRequest) -> ModelResponse:
        return self.normalize(self.client.models.generate_content(**native), request)

    async def acall(self, native: dict[str, Any], request: ModelRequest) -> ModelResponse:
        return self.normalize(await self.client.aio.models.generate_content(**native), request)

    def stream(self, native: dict[str, Any], request: ModelRequest) -> Iterator[StreamEvent]:
        state = _GeminiStream()
        for chunk in self.client.models.generate_content_stream(**native):
            yield from state.translate(chunk)
        yield ResponseCompleted(state.final(self, request))

    async def astream(self, native: dict[str, Any], request: ModelRequest) -> AsyncIterator[StreamEvent]:
        state = _GeminiStream()
        async for chunk in await self.client.aio.models.generate_content_stream(**native):
            for event in state.translate(chunk):
                yield event
        yield ResponseCompleted(state.final(self, request))

    # ---- token counting and errors ----------------------------------------------------------

    def _count_args(self, request: ModelRequest) -> dict[str, Any]:
        if getattr(self.client, "vertexai", False):
            config = self.generation_config(request)
            return {
                "model": request.model,
                "contents": self.contents(request.messages, request.model),
                "config": {k: config[k] for k in ("system_instruction", "tools") if k in config},
            }
        # The Developer API counts contents only: the system prompt is counted as text,
        # tool declarations are not included.
        messages = list(request.messages)
        if request.system:
            messages.insert(0, Message(role="user", content=request.system))
        return {"model": request.model, "contents": self.contents(messages, request.model)}

    def count_tokens(self, request: ModelRequest) -> int:
        return int(self.client.models.count_tokens(**self._count_args(request)).total_tokens or 0)

    async def acount_tokens(self, request: ModelRequest) -> int:
        result = await self.client.aio.models.count_tokens(**self._count_args(request))
        return int(result.total_tokens or 0)

    def map_error(self, exc: BaseException) -> BaseException | None:
        try:
            errors = require_module("google.genai.errors", "gemini")
        except ImportError:
            return None
        if isinstance(exc, errors.APIError):
            code = exc.code if isinstance(exc.code, int) else None
            return self.provider_error(exc, status_code=code)
        name = type(exc).__name__
        if type(exc).__module__.startswith(("httpx", "httpcore")) and (
            "Connect" in name or "Timeout" in name
        ):
            return self.provider_error(exc, status_code=None, connection=True)
        return None


class _GeminiStream:
    """Accumulates streamed chunks: text deltas are joined, other parts kept in order."""

    def __init__(self) -> None:
        self.parts: list[dict[str, Any]] = []
        self.finish: str | None = None
        self.usage: Any = None
        self.model: str | None = None
        self.id: str | None = None
        self.block: str | None = None
        self.calls = 0
        self.grounding: Any = None

    def translate(self, chunk: Any) -> Iterator[StreamEvent]:
        self.usage = _value(chunk, "usage_metadata") or self.usage
        self.model = _value(chunk, "model_version") or self.model
        self.id = _value(chunk, "response_id") or self.id
        candidates = _value(chunk, "candidates") or []
        if not candidates:
            feedback = _value(chunk, "prompt_feedback")
            self.block = _enum_name(_value(feedback, "block_reason")) or self.block
            return
        candidate = candidates[0]
        self.finish = _enum_name(_value(candidate, "finish_reason")) or self.finish
        self.grounding = _value(candidate, "grounding_metadata") or self.grounding
        for raw in _value(_value(candidate, "content"), "parts") or []:
            data = raw if isinstance(raw, dict) else raw.model_dump(mode="json", exclude_none=True)
            if data.get("function_call"):
                call = data["function_call"]
                call_id = str(call.get("id") or f"gemini_call_{len(self.parts)}_{call.get('name')}")
                yield ToolCallStarted(call_id=call_id, name=str(call.get("name", "")), index=len(self.parts))
                yield ToolCallDelta(
                    call_id=call_id, index=len(self.parts), arguments_delta=_json(call.get("args") or {})
                )
                self.parts.append(data)
                continue
            text = data.get("text")
            if isinstance(text, str):
                if text:
                    yield ReasoningDelta(text) if data.get("thought") else TextDelta(text)
                previous = self.parts[-1] if self.parts else None
                if (
                    previous is not None
                    and "text" in previous
                    and bool(previous.get("thought")) == bool(data.get("thought"))
                    and not previous.get("thought_signature")
                ):
                    previous["text"] += text
                    if data.get("thought_signature"):
                        previous["thought_signature"] = data["thought_signature"]
                    continue
            elif data.get("thought_signature") and self.parts and set(data) == {"thought_signature"}:
                self.parts[-1].setdefault("thought_signature", data["thought_signature"])
                continue
            self.parts.append(data)

    def final(self, provider: GeminiProvider, request: ModelRequest) -> ModelResponse:
        return provider.build(
            raw_parts=self.parts,
            finish=self.finish,
            usage=self.usage,
            block_reason=self.block if not self.parts else None,
            model=self.model or request.model,
            request=request,
            raw=None,
            request_id=self.id,
            grounding=self.grounding,
        )


def _json(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False)
