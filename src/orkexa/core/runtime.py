"""The one agent runtime shared by every agent class and every provider.

:class:`AgentCore` holds configuration, the committed conversation, tools and usage,
and contains all business rules of a turn in :meth:`AgentCore._turn`: a sans-I/O
generator that yields *effects* and receives their results:

* :class:`CallModel` - send a :class:`~orkexa.core.types.ModelRequest`, return the
  :class:`~orkexa.core.types.ModelResponse`;
* :class:`RunTools` - run tool calls, return their :class:`ToolResultPart` s;
* :class:`Emit` - pass a runtime event to the hooks.

:class:`~orkexa.Agent` and :class:`~orkexa.AsyncAgent` are small drivers that perform
these effects with blocking or ``async`` I/O. Providers only translate requests and
responses; there are no provider-specific branches here.
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
import json
import os
import threading
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Literal, cast
import uuid

from pydantic import BaseModel
from typing_extensions import TypeVar

from ..content import Prompt, describe_content, to_content
from ..errors import (
    ConcurrentTurnError,
    ConfigurationError,
    IncompleteResponseError,
    MaxToolRoundsError,
    MemoryToolError,
    ProviderError,
    RefusalError,
    StructuredOutputError,
    ToolDefinitionError,
    ToolExecutionError,
    ToolInputError,
    ToolResultError,
    UnexpectedResponseError,
)
from ..files import FileStore
from ..memory import MEMORY_TOOL_NAME, MemoryBackend
from ..pricing import ModelPricing
from ..retrieval import Index
from ..tools import Tool, ToolOutput
from ..usage import UsageRecord, UsageSummary, UsageTracker, current_workflow, summarize_usage
from .capabilities import ModelCapabilities, Support
from .conversion import (
    HistoryPolicy,
    dump_history,
    parse_history,
    prepare_history,
    result_content,
    to_json,
    to_parts,
)
from .types import (
    DocumentPart,
    Effort,
    ImagePart,
    Message,
    ModelRequest,
    ModelResponse,
    ReasoningDelta,
    ResponseCompleted,
    StreamEvent,
    TextDelta,
    ToolCallPart,
    ToolChoice,
    ToolResultPart,
    TurnCompleted,
    TurnFailed,
    UsageReported,
)

if TYPE_CHECKING:
    from ..providers.base import Provider

__all__ = ["AgentCore", "CallModel", "Emit", "OutputT", "RefusalMode", "RunTools", "ToolCall"]

OutputT = TypeVar("OutputT", default=str)
RefusalMode = Literal["raise", "return"]

_FINAL = ("end_turn", "stop_sequence", "refusal")
_INCOMPLETE = ("max_tokens", "context_window_exceeded")


@dataclass(frozen=True)
class ToolCall:
    """A tool call the model requested, as passed to ``before_tool``/``after_tool`` hooks."""

    id: str
    name: str
    input: dict[str, Any]
    agent: str


@dataclass(frozen=True)
class CallModel:
    """Effect: send ``request`` to the provider."""

    request: ModelRequest


@dataclass(frozen=True)
class RunTools:
    """Effect: run ``calls`` (all calls of one response) and return their results in order."""

    calls: tuple[ToolCallPart, ...]
    request: ModelRequest


@dataclass(frozen=True)
class Emit:
    """Effect: pass ``event`` to ``on_event``."""

    event: StreamEvent


Effect = CallModel | RunTools | Emit


@dataclass(frozen=True)
class _Step:
    kind: Literal["tools", "continue", "done"]
    tool_calls: tuple[ToolCallPart, ...] = ()
    result: Any = None


class AgentCore(Generic[OutputT]):
    """Configuration, conversation, tools, usage and the turn state machine (no I/O)."""

    _native_hooks: ClassVar[bool] = False
    """``on_request``/``on_response``/``last_response`` use native SDK objects (``ClaudeAgent``)."""
    _map_errors: ClassVar[bool] = True
    """Translate SDK exceptions into :class:`~orkexa.errors.ProviderError` (not in ``ClaudeAgent``)."""
    _asynchronous: ClassVar[bool] = False
    _async_name: ClassVar[str] = "AsyncAgent"

    adapter: Provider

    def __init__(
        self,
        *,
        model: str,
        provider: str | None = None,
        system_prompt: str | None = "You are a helpful assistant.",
        messages: Sequence[Message | Mapping[str, Any]] | None = None,
        tools: Sequence[Tool] | None = None,
        server_tools: Sequence[dict[str, Any]] | None = None,
        memory: MemoryBackend | None = None,
        files: FileStore | None = None,
        index: Index | None = None,
        output_format: type[OutputT] | None = None,
        effort: Effort | None = None,
        tool_choice: ToolChoice | str | dict[str, Any] = "auto",
        max_tokens: int = 16_000,
        max_tool_rounds: int = 8,
        max_parallel_tools: int = 1,
        raise_tool_errors: bool = True,
        tool_timeout: float | None = None,
        stream: bool | None = None,
        on_refusal: RefusalMode = "raise",
        on_text: Callable[[str], Any] | None = None,
        on_thinking: Callable[[str], Any] | None = None,
        on_event: Callable[[StreamEvent], Any] | None = None,
        on_request: Callable[[Any], Any] | None = None,
        on_response: Callable[[Any], Any] | None = None,
        before_tool: Callable[[ToolCall], Any] | None = None,
        after_tool: Callable[[ToolCall, str, bool], Any] | None = None,
        options: Any = None,
        client: Any = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        name: str | None = None,
        usage_tracker: UsageTracker | None = None,
        pricing: ModelPricing | None = None,
        capabilities: Mapping[str, Support] | None = None,
        history_policy: HistoryPolicy = "strict",
        _adapter: Provider | None = None,
    ) -> None:
        from ..providers.base import ProviderConfig
        from ..providers.registry import create_provider, resolve_model

        if output_format is not None and (
            not isinstance(output_format, type) or not issubclass(output_format, BaseModel)
        ):
            raise TypeError("output_format must be None or a Pydantic BaseModel subclass")
        if max_tokens < 1 or max_tool_rounds < 0 or max_parallel_tools < 1:
            raise ValueError(
                "max_tokens and max_parallel_tools must be positive and max_tool_rounds nonnegative"
            )
        if tool_timeout is not None and tool_timeout <= 0:
            raise ValueError("tool_timeout must be positive")
        if on_refusal not in ("raise", "return"):
            raise ValueError("on_refusal must be 'raise' or 'return'")
        if history_policy not in ("strict", "convert"):
            raise ValueError("history_policy must be 'strict' or 'convert'")
        if memory is not None and not isinstance(memory, MemoryBackend):
            raise TypeError("memory must be a MemoryBackend, e.g. EphemeralMemory() or FileMemory(path)")
        if files is not None and not isinstance(files, FileStore):
            raise TypeError("files must be a FileStore, e.g. LocalFileStore(path)")
        if index is not None and not isinstance(index, Index):
            raise TypeError("index must be an Index, e.g. SQLiteIndex(path)")

        provider_name, model_id = resolve_model(model, provider)
        if _adapter is None:
            config = ProviderConfig(
                api_key=api_key,  # type: ignore[arg-type]
                base_url=base_url,
                timeout=timeout,
                max_retries=max_retries,
                client=client,
            )
            _adapter = create_provider(provider_name, config, asynchronous=self._asynchronous)
        self.adapter = _adapter
        self.adapter.check_options(options)
        self.provider = provider_name
        self.model = model_id
        self.name = name or model_id
        self.options = options
        self.capability_overrides: dict[str, Support] = dict(capabilities or {})
        ModelCapabilities().merged(self.capability_overrides)  # validate names and values
        self.history_policy: HistoryPolicy = history_policy
        self.usage_tracker = usage_tracker if usage_tracker is not None else UsageTracker()
        self._usage_records: list[UsageRecord] = []
        self._last_turn_records: list[UsageRecord] = []
        self.pricing = pricing
        self.system_prompt = system_prompt
        self._init_messages(messages)
        self.output_format = output_format
        self.effort = effort
        self.tool_choice = ToolChoice.parse(tool_choice)
        self.max_tokens = max_tokens
        self.max_tool_rounds = max_tool_rounds
        self.max_parallel_tools = max_parallel_tools
        self.raise_tool_errors = raise_tool_errors
        self.tool_timeout = tool_timeout
        self.stream = stream
        self.on_refusal: RefusalMode = on_refusal
        self.on_text = on_text
        self.on_thinking = on_thinking
        self.on_event = on_event
        self.on_request = on_request
        self.on_response = on_response
        self.before_tool = before_tool
        self.after_tool = after_tool
        self.tools: dict[str, Tool] = {}
        self.server_tools: list[dict[str, Any]] = []
        self.memory = memory
        self.files = files
        self.index = index
        self.last_response: Any | None = None
        self._turn_effort: Effort | None = None
        self._turn_max_tokens: int | None = None
        self._turn_id: str | None = None
        self._turn_lock = threading.Lock()
        for tool in tools or []:
            self.add_tool(tool)
        if files is not None:
            self.add_tool(files.tool())
        if index is not None:
            self.add_tool(index.tool())
        for server_tool in server_tools or []:
            self.add_server_tool(server_tool)
        self._check_configuration()

    def _init_messages(self, messages: Sequence[Message | Mapping[str, Any]] | None) -> None:
        self.messages: list[Any] = [
            m.model_copy(deep=True) if isinstance(m, Message) else Message.model_validate(dict(m))
            for m in messages or []
        ]

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.name!r}, model={self.model!r}, "
            f"tools={self._tool_names()!r}, "
            f"messages={len(self.messages)})"
        )

    # ---- client and capabilities ---------------------------------------------------

    @property
    def client(self) -> Any:
        """The provider's SDK client (created on first use)."""
        return self.adapter.client

    @client.setter
    def client(self, value: Any) -> None:
        self.adapter.client = value

    def _model_id(self) -> str:
        from ..providers.registry import split_model

        prefix, model_id = split_model(self.model)
        if prefix is not None and prefix != self.provider:
            raise ConfigurationError(
                f"{self.name}: model {self.model!r} belongs to provider {prefix!r}, but this agent "
                f"uses {self.provider!r}; create a new agent to switch providers"
            )
        return model_id

    @property
    def capabilities(self) -> ModelCapabilities:
        """What the current model supports here, including ``capabilities=`` overrides."""
        return self.adapter.capabilities(self._model_id()).merged(self.capability_overrides)

    # ---- usage -----------------------------------------------------------------------

    @property
    def usage(self) -> UsageSummary:
        """Usage of this agent only, not its delegated child agents."""
        return summarize_usage(self._usage_records)

    @property
    def last_usage(self) -> UsageRecord | None:
        """Usage of the most recent individual API request."""
        return self._usage_records[-1] if self._usage_records else None

    @property
    def last_turn_usage(self) -> UsageSummary:
        """Usage of the most recent ask(), including all its internal tool rounds."""
        return summarize_usage(self._last_turn_records)

    @property
    def usage_records(self) -> tuple[UsageRecord, ...]:
        return tuple(self._usage_records)

    def reset_usage(self) -> None:
        """Clear local counters; the shared usage_tracker ledger remains intact."""
        self._usage_records.clear()
        self._last_turn_records.clear()

    def _add_usage_record(self, record: UsageRecord) -> None:
        label = current_workflow()
        if record.workflow is None and label is not None:
            record = replace(record, workflow=label)
        self._usage_records.append(record)
        self._last_turn_records.append(record)
        self.usage_tracker.add(record)

    # ---- tools -----------------------------------------------------------------------

    def _check_tool(self, tool: Tool) -> None:
        if not isinstance(tool, Tool) or tool.function is None or tool.name is None:
            raise TypeError("tools must contain initialized @Tool-decorated functions")
        if tool.is_async and not self._asynchronous:
            raise ToolDefinitionError(f"{tool.name}: async tools require {self._async_name}")

    def _server_tool_name(self, definition: dict[str, Any]) -> str:
        return self.adapter.server_tool_name(definition)

    def _tool_names(self) -> list[str]:
        names = [*self.tools, *(self._server_tool_name(t) for t in self.server_tools)]
        return [*names, MEMORY_TOOL_NAME] if self.memory is not None else names

    def add_tool(self, tool: Tool) -> None:
        """Add a local tool; it is offered from the next request on."""
        self._check_tool(tool)
        assert tool.name is not None
        if tool.name in self._tool_names():
            raise ValueError(f"Duplicate tool name: {tool.name}")
        self.tools[tool.name] = tool

    def add_server_tool(self, definition: dict[str, Any]) -> None:
        """Add a provider-native server tool (a dict in the provider's own format).

        Examples for Anthropic: ``{"type": "web_fetch_20260209", "name": "web_fetch"}``
        or an MCP connector toolset ``{"type": "mcp_toolset", "mcp_server_name": "crm"}``.
        Requires the ``server_tools`` capability; the provider runs these tools itself.
        """
        if not isinstance(definition, dict) or not definition.get("type"):
            raise TypeError("A server tool is a dict with a 'type'")
        name = self._server_tool_name(definition)
        if not name:
            raise TypeError("A server tool needs a 'name' (or 'mcp_server_name')")
        if name in self._tool_names():
            raise ValueError(f"Duplicate tool name: {name}")
        self.server_tools.append(deepcopy(definition))

    def remove_tool(self, name: str) -> Tool | dict[str, Any]:
        """Remove a local or server tool by name and return it; gone from the next request on."""
        if name in self.tools:
            return self.tools.pop(name)
        for position, definition in enumerate(self.server_tools):
            if self._server_tool_name(definition) == name:
                return self.server_tools.pop(position)
        raise KeyError(f"Unknown tool: {name}")

    # ---- conversation -------------------------------------------------------------------

    def _conversation(self) -> list[Message]:
        """A working copy of the committed conversation."""
        return [m.model_copy(deep=True) for m in self.messages]

    def _commit(self, history: list[Message]) -> None:
        self.messages = history

    def conversation(self) -> list[Message]:
        """The committed conversation as neutral messages (a copy), for every agent class."""
        return self._conversation()

    def reset(self) -> None:
        """Forget the conversation history (usage counters are kept)."""
        self.messages.clear()
        self.last_response = None

    def history(self) -> list[Any]:
        """A deep copy of the committed conversation history."""
        return deepcopy(self.messages)

    def save_history(self, path: str | os.PathLike[str]) -> None:
        """Write the conversation as a versioned Orkexa history file (JSON)."""
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(to_json(dump_history(self._conversation())))

    def load_history(self, path: str | os.PathLike[str]) -> None:
        """Replace the conversation with one from :meth:`save_history`.

        Also reads ``claude-agent`` 0.8.0 history files (plain Messages API lists);
        their assistant messages are marked as produced by Anthropic, so continuing
        them with another provider follows ``history_policy``.
        """
        from ..providers.anthropic_codec import decode_messages

        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        self._commit(parse_history(data, decode_messages, str(path)))
        self.last_response = None

    # ---- per-turn settings and requests --------------------------------------------------

    @property
    def _effort(self) -> Effort | None:
        """The effort of the current request: an ``ask(effort=...)`` override or ``effort``."""
        return self._turn_effort if self._turn_effort is not None else self.effort

    @property
    def _max_tokens(self) -> int:
        return self._turn_max_tokens if self._turn_max_tokens is not None else self.max_tokens

    def _provider_options(self) -> Any:
        return self.options

    def _native_tool_choice(self) -> ToolChoice:
        return self.tool_choice

    def _use_stream(self) -> bool:
        if self.stream is not None:
            return self.stream
        return (
            self.on_text is not None
            or self.on_thinking is not None
            or self.on_event is not None
            or self.adapter.auto_stream(self._max_tokens)
        )

    def _build_request(self, history: list[Message], *, check: bool = True) -> ModelRequest:
        model = self._model_id()
        caps = self.adapter.capabilities(model).merged(self.capability_overrides)
        tools = [tool.definition() for tool in self.tools.values()]
        native_tools = deepcopy(self.server_tools)
        if self.memory is not None:
            if caps.supports("native_memory_tool"):
                native_tools.append(self.memory.definition)
            else:
                tools.append(self.memory.tool_definition())
        request = ModelRequest(
            model=model,
            messages=prepare_history(history, self.adapter.name, self.history_policy, model),
            system=self.system_prompt,
            tools=tools,
            native_tools=native_tools,
            tool_choice=self._native_tool_choice() if tools or native_tools else None,
            output_format=cast("type[BaseModel] | None", self.output_format),
            max_tokens=self._max_tokens,
            stream=self._use_stream(),
            effort=self._effort,
            options=self._provider_options(),
            history_policy=self.history_policy,
            capabilities=caps,
        )
        if check:
            self._check_request(request, caps)
        return request

    def _check_request(self, request: ModelRequest, caps: ModelCapabilities) -> None:
        """Every feature the request uses must be ``supported`` (no silent degradation)."""
        where = {"provider": self.adapter.name, "model": request.model}
        if request.tools or request.native_tools:
            caps.require("tools", **where)
        if self.server_tools:
            caps.require("server_tools", **where)
        if request.tool_choice is not None and request.tool_choice.forced:
            caps.require("forced_tool_choice", **where)
        if request.output_format is not None:
            caps.require("structured_output", **where)
        if request.stream:
            caps.require("streaming", **where)
        if request.effort is not None:
            caps.require("reasoning", **where, feature=f"reasoning effort {request.effort!r}")
        for message in request.messages:
            if message.role == "system":
                caps.require("system_messages", **where)
            for part in message.parts():
                self._check_part(part, caps, where)
        self.adapter.check(request)

    @staticmethod
    def _check_part(part: Any, caps: ModelCapabilities, where: dict[str, str]) -> None:
        if isinstance(part, ImagePart):
            caps.require("image_input", **where)
        elif isinstance(part, DocumentPart) and part.source.type != "text":
            caps.require("document_input", **where)
        elif isinstance(part, ToolResultPart) and isinstance(part.content, list):
            for child in part.content:
                AgentCore._check_part(child, caps, where)

    def _check_configuration(self) -> None:
        """Validate options, tools and capabilities once at construction."""
        self._build_request([])

    def _next_history(self, prompt: Prompt | None) -> list[Message]:
        """History plus ``prompt`` for token counting; does not start a turn."""
        history = self._conversation()
        if prompt is not None:
            history.append(
                Message(role="user", content=to_parts(to_content(prompt), self.adapter.parse_block))
            )
        if not history:
            raise ValueError("Nothing to count: the history is empty and no prompt was given")
        return history

    def _count_request(self, prompt: Prompt | None) -> ModelRequest:
        request = self._build_request(self._next_history(prompt))
        request.stream = False
        return request

    def _run_batch(
        self,
        prompts: Mapping[str, str],
        *,
        poll_interval: float,
        timeout: float | None,
    ) -> tuple[str, dict[str, ModelResponse | str]]:
        """Send one-message requests through the provider's batch API; records usage.

        The usage tracker admits the batch as a whole before it is submitted (like
        one request). Every result's usage is recorded before an error of the
        tracker (e.g. an exceeded budget) is raised.

        Returns the batch id and, per custom id, the response or an error description.
        """
        requests = {
            key: self._build_request([Message(role="user", content=prompt)])
            for key, prompt in prompts.items()
        }
        for request in requests.values():
            request.stream = False
        model = self._model_id()
        self.usage_tracker.before_request(
            agent=self.name,
            provider=self.adapter.name,
            model=model,
            priced=self.pricing is not None or self.adapter.has_pricing(model),
        )
        batch_id, results = self.adapter.run_batch(requests, poll_interval=poll_interval, timeout=timeout)
        failure: Exception | None = None
        for result in results.values():
            if isinstance(result, ModelResponse):
                record = self.adapter.usage_record(
                    result, agent=self.name, pricing=self.pricing, turn_id=None, model=model, batch=True
                )
                if record is not None:
                    try:
                        self._add_usage_record(record)
                    except Exception as exc:  # the tracker keeps the record; raise after all are in
                        failure = failure or exc
        if failure is not None:
            raise failure
        return batch_id, results

    # ---- the turn ------------------------------------------------------------------------

    def _enter_turn(self) -> None:
        if not self._turn_lock.acquire(blocking=False):
            raise ConcurrentTurnError(
                f"{self.name}: ask() is already running for this agent; one agent holds one "
                "conversation, use one agent per concurrent session"
            )

    def _exit_turn(self) -> None:
        self._turn_effort = None
        self._turn_max_tokens = None
        self._turn_id = None
        self._turn_lock.release()

    def _start_turn(
        self,
        prompt: Prompt,
        *,
        system: str | None,
        effort: Effort | None,
        max_tokens: int | None,
    ) -> list[Message]:
        content = to_content(prompt)
        if max_tokens is not None and max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if system is not None and not system.strip():
            raise ValueError("system must not be empty")
        history = self._conversation()
        self._last_turn_records = []
        self._turn_effort = effort
        self._turn_max_tokens = max_tokens
        self._turn_id = uuid.uuid4().hex
        history.append(Message(role="user", content=to_parts(content, self.adapter.parse_block)))
        if system is not None:
            # A mid-conversation system message keeps the cached prefix intact.
            history.append(Message(role="system", content=system))
        return history

    def _turn(
        self,
        prompt: Prompt,
        *,
        system: str | None = None,
        effort: Effort | None = None,
        max_tokens: int | None = None,
    ) -> Generator[Effect, Any, OutputT]:
        """One ``ask()``: the complete request/tool loop as a generator of effects."""
        try:
            history = self._start_turn(prompt, system=system, effort=effort, max_tokens=max_tokens)
            for round_number in range(self.max_tool_rounds + 1):
                request = self._build_request(history)
                self.usage_tracker.before_request(
                    agent=self.name,
                    provider=self.adapter.name,
                    model=request.model,
                    priced=self.pricing is not None or self.adapter.has_pricing(request.model),
                )
                response: ModelResponse = yield CallModel(request)
                record = self._record(response)
                if record is not None:
                    yield Emit(UsageReported(record))
                step = self._process_response(response, history, round_number)
                if step.kind == "done":
                    self._commit(history)
                    yield Emit(TurnCompleted(step.result))
                    return cast(OutputT, step.result)
                if step.kind == "tools":
                    results: list[ToolResultPart] = yield RunTools(step.tool_calls, request)
                    # All results of one assistant turn go back in a single user message.
                    history.append(Message(role="user", content=list(results)))
            raise UnexpectedResponseError("Tool loop ended unexpectedly")
        except Exception as exc:
            if self.on_event is not None:
                yield Emit(TurnFailed(exc))
            raise

    def _record(self, response: ModelResponse) -> UsageRecord | None:
        """Record usage first: API costs never disappear from the ledger, even if the turn fails."""
        self.last_response = response.raw if self._native_hooks else response
        record = self.adapter.usage_record(
            response,
            agent=self.name,
            pricing=self.pricing,
            turn_id=self._turn_id,
            model=self._model_id(),
        )
        if record is not None:
            self._add_usage_record(record)
        return record

    def _process_response(self, response: ModelResponse, history: list[Message], round_number: int) -> _Step:
        """Extend ``history`` and decide what the loop does next."""
        native = response.native_stop_reason or response.stop_reason
        if response.stop_reason in _INCOMPLETE:
            raise IncompleteResponseError(
                f"{self.name}: incomplete response "
                f"(stop_reason={native!r}, max_tokens={self._max_tokens}). "
                "The conversation history was not updated."
            )
        # The requested model id (not a dated version the API reports): reasoning is
        # only replayed to the model the conversation continues with.
        history.append(response.message.model_copy(update={"model": self._model_id()}))

        if response.stop_reason in ("tool_use", "pause_turn"):
            if round_number >= self.max_tool_rounds:
                raise MaxToolRoundsError(
                    f"{self.name}: maximum number of tool rounds ({self.max_tool_rounds}) reached"
                )
            if response.stop_reason == "pause_turn":
                # A long server-tool turn was paused; sending it back resumes it.
                return _Step("continue")
            calls = tuple(response.tool_calls)
            if not calls:
                raise UnexpectedResponseError("stop_reason='tool_use' without tool_use blocks")
            return _Step("tools", tool_calls=calls)

        if response.stop_reason not in _FINAL:
            raise UnexpectedResponseError(f"Unhandled stop_reason: {native}")
        if response.stop_reason == "refusal" and self.on_refusal == "raise":
            category = getattr(response.stop_details, "category", None)
            raise RefusalError(
                f"{self.name}: the model declined the request (category={category!r}). "
                "The conversation history was not updated.",
                stop_details=response.stop_details,
            )
        if self.output_format is None:
            return _Step("done", result=response.text)
        result = response.parsed
        if not isinstance(result, self.output_format):
            message = (
                f"Missing or invalid parsed_output for {self.output_format.__name__} (stop_reason={native})"
            )
            if response.stop_reason == "refusal":
                raise RefusalError(message, stop_details=response.stop_details)
            if response.parse_error:
                message += f": {response.parse_error}"
            raise StructuredOutputError(message)
        return _Step("done", result=result)

    # ---- hooks and events ------------------------------------------------------------------

    def _event_hooks(self, event: StreamEvent) -> list[Any]:
        """Call the hooks for ``event``; returns their results (the async driver awaits them)."""
        results: list[Any] = []
        if isinstance(event, TextDelta) and self.on_text is not None:
            results.append(self.on_text(event.text))
        elif isinstance(event, ReasoningDelta) and self.on_thinking is not None and event.text:
            results.append(self.on_thinking(event.text))
        if self.on_event is not None:
            results.append(self.on_event(event))
        return results

    def _request_hook_arg(self, request: ModelRequest, native: dict[str, Any]) -> Any:
        return native if self._native_hooks else request

    def _response_hook_arg(self, response: ModelResponse) -> Any:
        return response.raw if self._native_hooks else response

    def _provider_failure(self, exc: Exception) -> Exception:
        """The exception to raise for a failed provider call."""
        if not self._map_errors:
            return exc
        mapped = self.adapter.map_error(exc)
        failure = exc if mapped is None else cast(Exception, mapped)
        if isinstance(failure, ProviderError):
            # Adapters build some errors from native response text themselves.
            failure.args = tuple(self.adapter.redact(a) if isinstance(a, str) else a for a in failure.args)
        return failure

    @staticmethod
    def _final_response(event: StreamEvent | None) -> ModelResponse:
        if not isinstance(event, ResponseCompleted):
            raise UnexpectedResponseError("The stream ended without a final response")
        return event.response

    # ---- tool calls -----------------------------------------------------------------------

    def _tool_call(self, part: ToolCallPart) -> ToolCall:
        return ToolCall(id=part.id, name=part.name, input=deepcopy(part.arguments), agent=self.name)

    @staticmethod
    def _denial(reason: Any) -> str | None:
        """Interpret a ``before_tool`` result: ``None``/``True`` allow, a string or ``False`` deny."""
        if reason is None or reason is True:
            return None
        if reason is False:
            return "Tool call denied."
        return f"Tool call denied: {reason}"

    def _is_memory_call(self, call: ToolCall) -> bool:
        return call.name == MEMORY_TOOL_NAME and self.memory is not None

    def _run_memory(self, payload: Any) -> tuple[str, bool]:
        """Run a memory command; expected failures go back to the model as an error result."""
        assert self.memory is not None
        try:
            return self.memory.execute(payload if isinstance(payload, dict) else {}), False
        except MemoryToolError as exc:
            return str(exc), True
        except Exception as exc:
            raise ToolExecutionError(f"memory: {type(exc).__name__}: {exc}") from exc

    def _lookup_tool(self, name: str) -> Tool:
        tool = self.tools.get(name)
        if tool is None:
            raise ToolExecutionError(f"Unknown tool called: {name}")
        return tool

    @staticmethod
    def _check_arguments(part: ToolCallPart) -> None:
        if part.parse_error is not None:
            raise ToolInputError(f"{part.name}: invalid tool inputs: {part.parse_error}")

    def _tool_failure(self, exc: Exception, request: ModelRequest) -> tuple[str, bool]:
        """What goes back to the model for a failed tool call, or re-raise per ``raise_tool_errors``."""
        if isinstance(exc, ToolResultError):
            return str(exc), True
        if isinstance(exc, ToolInputError):
            # Inputs that the provider does not enforce (or streams unbuffered) can be
            # malformed; the model gets the error and retries.
            if self.raise_tool_errors and not self.adapter.lenient_tool_input(
                request, self._caps_for(request)
            ):
                raise exc
            return str(exc), True
        if isinstance(exc, ToolExecutionError):
            if self.raise_tool_errors:
                raise exc
            return str(exc), True
        raise exc

    def _caps_for(self, request: ModelRequest) -> ModelCapabilities:
        return self.adapter.capabilities(request.model).merged(self.capability_overrides)

    def _timeout_error(self, call: ToolCall) -> ToolExecutionError:
        return ToolExecutionError(f"{call.name}: timed out after {self.tool_timeout} s")

    @staticmethod
    def _hook_content(content: ToolOutput) -> str:
        """What ``after_tool`` receives: the text, or a short description of blocks."""
        return describe_content(content)

    def _tool_result(self, call: ToolCall, content: ToolOutput, is_error: bool) -> ToolResultPart:
        return ToolResultPart(
            call_id=call.id,
            name=call.name,
            content=result_content(content, self.adapter.parse_block),
            is_error=is_error,
        )
