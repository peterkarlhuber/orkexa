"""Anthropic Messages API adapter (``pip install 'orkexa[anthropic]'``).

Encodes requests with prompt caching, thinking, effort, server tools, the native
memory tool, fallbacks, betas and structured output through ``messages.parse``, keeps
every native block lossless through :mod:`orkexa.providers.anthropic_codec`, and
supports token counting and Message Batches. Model restrictions (models that reject
forced tool choice, disabled thinking, ...) are capabilities and request checks.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Mapping
from contextlib import suppress
from copy import deepcopy
from dataclasses import replace
import re
import time
from typing import Any, ClassVar

from ..content import ContentBlock
from ..core.capabilities import ModelCapabilities, Support
from ..core.types import (
    ModelRequest,
    ModelResponse,
    Part,
    ReasoningDelta,
    ResponseCompleted,
    StopReason,
    StreamEvent,
    TextDelta,
    TokenUsage,
    ToolCallDelta,
    ToolCallStarted,
    ToolChoice,
    ToolDefinition,
)
from ..errors import ConfigurationError, UnsupportedCapabilityError
from ..pricing import ModelPricing, resolve_pricing
from ..usage import UsageRecord, record_api_usage
from . import anthropic_codec as codec
from .base import Provider, ProviderConfig, require_module
from .options import AnthropicOptions

__all__ = ["AnthropicProvider", "model_family"]

#: Models that reject a forced tool_choice (``any``/``tool``) with HTTP 400.
_NO_FORCED_TOOL_CHOICE = frozenset(
    {"claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1", "claude-mythos-5-1"}
)
#: Models that reject ``thinking: {"type": "disabled"}`` with HTTP 400.
_NO_DISABLED_THINKING = _NO_FORCED_TOOL_CHOICE | {"claude-fable-5", "claude-mythos-5"}
#: Models that accept ``thinking: {"type": "disabled"}`` only at effort ``high`` or below.
_DISABLED_THINKING_UP_TO_HIGH = frozenset({"claude-opus-5", "claude-haiku-5-5"})
#: The only model that accepts ``thinking: {"type": "between_tools"}``.
_BETWEEN_TOOLS_MODEL = "claude-sonnet-5-5"
#: Models known to reject ``role: "system"`` messages inside ``messages``.
_NO_SYSTEM_MESSAGES = frozenset(
    {"claude-sonnet-5", "claude-opus-4-7", "claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5"}
)
_EFFORTS_ABOVE_HIGH = ("xhigh", "max")

_THINKING_UPDATES_BETA = "thinking-display-updates-2026-08-18"
_FALLBACK_BETA_DEFAULT = "server-side-fallback-2026-07-01"
_FALLBACK_BETA_LIST = "server-side-fallback-2026-06-01"

#: Above this max_tokens value requests are streamed automatically (HTTP timeouts).
_AUTO_STREAM_MAX_TOKENS = 16_000

_STOP_REASONS: dict[str, StopReason] = {
    "end_turn": "end_turn",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
    "stop_sequence": "stop_sequence",
    "refusal": "refusal",
    "pause_turn": "pause_turn",
    "model_context_window_exceeded": "context_window_exceeded",
}


def model_family(model: str) -> str:
    """``"anthropic.claude-opus-5-5-20260101"`` -> ``"claude-opus-5-5"``."""
    name = model.removeprefix("anthropic.").split("[", 1)[0]
    return re.sub(r"[-@]\d{8}$", "", name)


def native_tool_choice(choice: ToolChoice) -> dict[str, Any]:
    if choice.mode == "required":
        return {"type": "any"}
    if choice.mode == "tool":
        return {"type": "tool", "name": choice.name}
    return {"type": choice.mode}


def tool_schema(definition: ToolDefinition) -> dict[str, Any]:
    """A neutral tool definition in Messages API format (identical to ``Tool.schema()``)."""
    return {
        "name": definition.name,
        "description": definition.description,
        "input_schema": deepcopy(definition.input_schema),
        "strict": definition.strict,
    }


def check_options(model: str, options: AnthropicOptions, effort: str | None) -> None:
    """The model/option combinations the Messages API rejects.

    Raises:
        UnsupportedCapabilityError: (a ``ValueError``) naming the rejected combination.
    """
    family = model_family(model)
    choice = options.tool_choice or {}
    if choice.get("type") in ("any", "tool") and family in _NO_FORCED_TOOL_CHOICE:
        raise UnsupportedCapabilityError(_forced_reason(model), capability="forced_tool_choice")
    thinking = options.thinking
    if thinking == "disabled" and family in _NO_DISABLED_THINKING:
        raise UnsupportedCapabilityError(
            f"{model} cannot disable thinking; lower effort instead (Sonnet 5.5: thinking='between_tools')",
            capability="reasoning",
        )
    if thinking == "disabled" and family in _DISABLED_THINKING_UP_TO_HIGH and effort in _EFFORTS_ABOVE_HIGH:
        raise UnsupportedCapabilityError(
            f"{model} disables thinking only at effort 'high' or below", capability="reasoning"
        )
    if thinking == "between_tools":
        if family != _BETWEEN_TOOLS_MODEL and resolve_pricing(family) is not None:
            raise UnsupportedCapabilityError(
                f"thinking='between_tools' is accepted only by {_BETWEEN_TOOLS_MODEL}", capability="reasoning"
            )
        if effort in _EFFORTS_ABOVE_HIGH:
            raise UnsupportedCapabilityError(
                "thinking='between_tools' is accepted only at effort 'high' or below", capability="reasoning"
            )
    if options.thinking_display is not None and thinking in ("disabled", "between_tools"):
        raise UnsupportedCapabilityError(
            f"thinking_display cannot be combined with thinking={thinking!r}", capability="reasoning"
        )


def _forced_reason(model: str) -> str:
    return (
        f"{model} rejects a forced tool_choice; use {{'type': 'auto'}} with a prompt "
        "instruction, or output_format for structured results"
    )


def _usage(usage: Any) -> TokenUsage | None:
    if usage is None:
        return None

    def number(obj: Any, name: str) -> int:
        value = getattr(obj, name, None) if obj is not None else None
        return int(value) if isinstance(value, int) else 0

    creation = getattr(usage, "cache_creation", None)
    tools = getattr(usage, "server_tool_use", None)
    return TokenUsage(
        input_tokens=number(usage, "input_tokens"),
        output_tokens=number(usage, "output_tokens"),
        cache_read_input_tokens=number(usage, "cache_read_input_tokens"),
        cache_creation_input_tokens=number(usage, "cache_creation_input_tokens"),
        cache_creation_5m_input_tokens=number(creation, "ephemeral_5m_input_tokens"),
        cache_creation_1h_input_tokens=number(creation, "ephemeral_1h_input_tokens"),
        thinking_tokens=number(getattr(usage, "output_tokens_details", None), "thinking_tokens"),
        web_search_requests=number(tools, "web_search_requests"),
        web_fetch_requests=number(tools, "web_fetch_requests"),
    )


class AnthropicProvider(Provider):
    """Provider ``"anthropic"``: the Anthropic Messages API through the official SDK."""

    name: ClassVar[str] = "anthropic"
    options_type: ClassVar[type | None] = AnthropicOptions

    def __init__(self, config: ProviderConfig | None = None, asynchronous: bool = False) -> None:
        super().__init__(config, asynchronous)
        if self.config.base_url is not None and self.config.api_key is None and self.config.client is None:
            raise ConfigurationError(
                "A custom base_url needs an explicit api_key: Orkexa never sends the "
                "ANTHROPIC_API_KEY environment variable to another host"
            )

    def make_client(self) -> Any:
        sdk = require_module("anthropic", "anthropic")
        cls = sdk.AsyncAnthropic if self.asynchronous else sdk.Anthropic
        return cls(**self.config.client_kwargs())

    # ---- capabilities and checks ------------------------------------------------------

    def capabilities(self, model: str) -> ModelCapabilities:
        family = model_family(model)
        reasons: dict[str, str] = {}
        overrides: dict[str, Support] = {}
        if family in _NO_FORCED_TOOL_CHOICE:
            overrides["forced_tool_choice"] = "unsupported"
            reasons["forced_tool_choice"] = _forced_reason(model)
        if family in _NO_SYSTEM_MESSAGES:
            overrides["system_messages"] = "unsupported"
            reasons["system_messages"] = f"{model} does not accept system messages inside the conversation"
        return ModelCapabilities.all("supported", reasons=reasons, **overrides)

    def check(self, request: ModelRequest) -> None:
        check_options(request.model, self._options(request), request.effort)

    @staticmethod
    def _options(request: ModelRequest) -> AnthropicOptions:
        options = request.options
        return options if isinstance(options, AnthropicOptions) else AnthropicOptions()

    def parse_block(self, block: ContentBlock) -> Part:
        return codec.decode_block(dict(block))

    def auto_stream(self, max_tokens: int) -> bool:
        return max_tokens > _AUTO_STREAM_MAX_TOKENS

    def lenient_tool_input(self, request: ModelRequest, capabilities: ModelCapabilities) -> bool:
        # Unbuffered tool input can arrive cut off or malformed; Claude retries.
        return self._options(request).eager_input_streaming

    def has_pricing(self, model: str) -> bool:
        return resolve_pricing(model) is not None

    # ---- request encoding ------------------------------------------------------------

    def betas(self, options: AnthropicOptions) -> list[str]:
        betas = list(options.request_options.get("betas", []))
        if options.thinking_display == "updates" and _THINKING_UPDATES_BETA not in betas:
            betas.append(_THINKING_UPDATES_BETA)
        if options.fallbacks is not None:
            beta = _FALLBACK_BETA_DEFAULT if options.fallbacks == "default" else _FALLBACK_BETA_LIST
            if beta not in betas:
                betas.append(beta)
        return betas

    def tool_definitions(self, request: ModelRequest) -> list[dict[str, Any]]:
        options = self._options(request)
        local = [tool_schema(t) for t in request.tools]
        if options.eager_input_streaming and request.stream:
            for definition in local:
                definition["eager_input_streaming"] = True
        return local + deepcopy(request.native_tools)

    def prepare(self, request: ModelRequest) -> dict[str, Any]:
        options = self._options(request)
        args: dict[str, Any] = {"model": request.model, "max_tokens": request.max_tokens}
        if request.system is not None:
            args["system"] = request.system
        args["messages"] = codec.encode_messages(request.messages)
        if request.effort is not None:
            args["output_config"] = {"effort": request.effort}
        if options.thinking is not None or options.thinking_display is not None:
            thinking: dict[str, str] = {"type": options.thinking or "adaptive"}
            if options.thinking_display is not None:
                thinking["display"] = options.thinking_display
            args["thinking"] = thinking
        if options.cache_control:
            cache_settings: dict[str, str] = {"type": "ephemeral"}
            if options.cache_ttl == "1h":
                cache_settings["ttl"] = "1h"
            # messages.create() accepts cache_control directly, but messages.parse()
            # does not (anthropic 2.3); parse() sends the same field via extra_body.
            if request.output_format is None:
                args["cache_control"] = cache_settings
            else:
                args["extra_body"] = {"cache_control": cache_settings}
        tools = self.tool_definitions(request)
        if tools:
            args["tools"] = tools
            args["tool_choice"] = deepcopy(options.tool_choice) or native_tool_choice(
                request.tool_choice or ToolChoice()
            )
        if request.output_format is not None:
            args["output_format"] = request.output_format  # SDK helper, not API output_config.format
        if options.fallbacks is not None:
            args["fallbacks"] = deepcopy(options.fallbacks)
        for key, value in deepcopy(options.request_options).items():
            if key == "betas":
                continue
            if isinstance(value, dict) and isinstance(args.get(key), dict):
                args[key] = {**args[key], **value}  # e.g. output_config.task_budget next to effort
            else:
                args[key] = value
        betas = self.betas(options)
        if betas:
            args["betas"] = betas
        return args

    def _api(self, native: Mapping[str, Any]) -> Any:
        """``client.messages``, or ``client.beta.messages`` when a beta is in use."""
        return self.client.beta.messages if native.get("betas") else self.client.messages

    # ---- responses ---------------------------------------------------------------------

    def normalize(self, response: Any, request: ModelRequest) -> ModelResponse:
        """A native message as a :class:`ModelResponse` (blocks kept lossless)."""
        blocks = [block.model_dump(exclude_none=True) for block in response.content]
        model = getattr(response, "model", None) or request.model
        native_stop = getattr(response, "stop_reason", None)
        request_id = getattr(response, "_request_id", None)
        return ModelResponse(
            message=codec.decode_message({"role": "assistant", "content": blocks}).model_copy(
                update={"model": model}
            ),
            stop_reason=_STOP_REASONS.get(str(native_stop), "other"),
            provider=self.name,
            model=model,
            native_stop_reason=native_stop,
            parsed=getattr(response, "parsed_output", None) if request.output_format is not None else None,
            stop_details=getattr(response, "stop_details", None),
            usage=_usage(getattr(response, "usage", None)),
            request_id=request_id if isinstance(request_id, str) else None,
            raw=response,
        )

    def call(self, native: dict[str, Any], request: ModelRequest) -> ModelResponse:
        api = self._api(native)
        response = api.create(**native) if request.output_format is None else api.parse(**native)
        return self.normalize(response, request)

    async def acall(self, native: dict[str, Any], request: ModelRequest) -> ModelResponse:
        api = self._api(native)
        if request.output_format is None:
            response = await api.create(**native)
        else:
            response = await api.parse(**native)
        return self.normalize(response, request)

    def stream(self, native: dict[str, Any], request: ModelRequest) -> Iterator[StreamEvent]:
        state = _StreamState()
        with self._api(native).stream(**native) as stream:
            for event in stream:
                yield from state.translate(event)
            final = _with_request_id(stream, stream.get_final_message())
        yield ResponseCompleted(self.normalize(final, request))

    async def astream(self, native: dict[str, Any], request: ModelRequest) -> AsyncIterator[StreamEvent]:
        state = _StreamState()
        async with self._api(native).stream(**native) as stream:
            async for event in stream:
                for translated in state.translate(event):
                    yield translated
            final = _with_request_id(stream, await stream.get_final_message())
        yield ResponseCompleted(self.normalize(final, request))

    # ---- token counting, usage, errors, batches -----------------------------------------

    def _count_args(self, request: ModelRequest) -> dict[str, Any]:
        args = self.prepare(request)
        allowed = {"model", "messages", "system", "tools", "tool_choice", "thinking", "output_config"}
        if "betas" in args:
            allowed |= {"betas", "context_management", "mcp_servers"}
        counted = {key: value for key, value in args.items() if key in allowed}
        if request.output_format is not None and "betas" in counted:
            # The beta count_tokens() has no output_format helper; send the API field.
            sdk = require_module("anthropic", "anthropic")
            schema = {"type": "json_schema", "schema": sdk.transform_schema(request.output_format)}
            counted["output_config"] = {**counted.get("output_config", {}), "format": schema}
        elif request.output_format is not None:
            counted["output_format"] = request.output_format
        return counted

    def count_tokens(self, request: ModelRequest) -> int:
        args = self._count_args(request)
        return int(self._api(args).count_tokens(**args).input_tokens)

    async def acount_tokens(self, request: ModelRequest) -> int:
        args = self._count_args(request)
        return int((await self._api(args).count_tokens(**args)).input_tokens)

    def usage_record(
        self,
        response: ModelResponse,
        *,
        agent: str,
        pricing: ModelPricing | None,
        turn_id: str | None,
        model: str,
        batch: bool = False,
    ) -> UsageRecord | None:
        return record_api_usage(
            response.raw,
            agent_name=agent,
            default_model=model,
            custom_pricing=pricing,
            batch=batch,
            provider=self.name,
            turn_id=turn_id,
        )

    def map_error(self, exc: BaseException) -> BaseException | None:
        try:
            sdk = require_module("anthropic", "anthropic")
        except ImportError:
            return None
        if isinstance(exc, (sdk.APIConnectionError, sdk.APITimeoutError)):
            return self.provider_error(exc, status_code=None, connection=True)
        if isinstance(exc, sdk.APIStatusError):
            return self.provider_error(exc, status_code=exc.status_code, request_id=exc.request_id)
        return None

    def batch_params(self, request: ModelRequest) -> tuple[dict[str, Any], list[str]]:
        """One batch item's ``params`` (built like :meth:`prepare`) and the betas it needs.

        A batch item is one request that cannot run local tools, so they are rejected
        rather than sent; SDK-call options without a batch-item counterpart too.
        """
        if request.tools:
            raise UnsupportedCapabilityError(
                f"Message Batches send single requests, local tools cannot run there: "
                f"{sorted(t.name for t in request.tools)}",
                capability="batch",
            )
        args = self.prepare(_without_output_format(request))
        betas = list(args.pop("betas", []))
        unsupported = sorted(key for key in ("extra_headers", "extra_query", "timeout") if key in args)
        if unsupported:
            raise ConfigurationError(f"request_options {unsupported} cannot be applied to batch items")
        extra_body = args.pop("extra_body", {})
        params = {**args, **extra_body}  # a batch item is the request body itself
        if request.output_format is not None:
            sdk = require_module("anthropic", "anthropic")
            schema = {"type": "json_schema", "schema": sdk.transform_schema(request.output_format)}
            params["output_config"] = {**params.get("output_config", {}), "format": schema}
        return params, betas

    def run_batch(
        self,
        requests: Mapping[str, ModelRequest],
        *,
        poll_interval: float,
        timeout: float | None,
    ) -> tuple[str, dict[str, ModelResponse | str]]:
        """Message Batches: one request per item, 50 % price, polled until it ended."""
        items: list[dict[str, Any]] = []
        betas: list[str] = []
        for custom_id, request in requests.items():
            params, needed = self.batch_params(request)
            betas += [beta for beta in needed if beta not in betas]
            items.append({"custom_id": custom_id, "params": params})

        if betas:
            batches = self.client.beta.messages.batches
            created = batches.create(requests=items, betas=betas)
        else:
            batches = self.client.messages.batches
            created = batches.create(requests=items)
        deadline = None if timeout is None else time.monotonic() + timeout
        status = created
        while status.processing_status != "ended":
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"Message batch {created.id} has not ended after {timeout} s")
            time.sleep(poll_interval)
            status = batches.retrieve(created.id)
        results: dict[str, ModelResponse | str] = {}
        for entry in batches.results(created.id):
            result = entry.result
            if result.type != "succeeded":
                results[entry.custom_id] = str(result.type)
            else:
                request = requests[entry.custom_id]
                results[entry.custom_id] = self.normalize(result.message, _without_output_format(request))
        return str(created.id), results


def _without_output_format(request: ModelRequest) -> ModelRequest:
    return replace(request, output_format=None)


def _with_request_id(stream: Any, response: Any) -> Any:
    """Give a streamed final message the ``_request_id`` a non-streamed one has."""
    if getattr(response, "_request_id", None) is None:
        headers = getattr(getattr(stream, "response", None), "headers", None)
        request_id = headers.get("request-id") if headers is not None else None
        if request_id:
            with suppress(AttributeError, TypeError, ValueError):
                response._request_id = request_id
    return response


class _StreamState:
    """Translates the SDK's stream helper events into normalized events."""

    def __init__(self) -> None:
        self.call_id: str | None = None
        self.index = 0

    def translate(self, event: Any) -> Iterator[StreamEvent]:
        kind = getattr(event, "type", None)
        if kind == "text":
            yield TextDelta(str(event.text))
        elif kind == "thinking":
            if event.thinking:
                yield ReasoningDelta(str(event.thinking))
        elif kind == "content_block_start":
            block = getattr(event, "content_block", None)
            self.index = int(getattr(event, "index", 0) or 0)
            if block is not None and getattr(block, "type", None) == "tool_use":
                self.call_id = str(block.id)
                yield ToolCallStarted(call_id=self.call_id, name=str(block.name), index=self.index)
            else:
                self.call_id = None
        elif kind == "input_json" and self.call_id is not None:
            yield ToolCallDelta(
                call_id=self.call_id, index=self.index, arguments_delta=str(event.partial_json)
            )
