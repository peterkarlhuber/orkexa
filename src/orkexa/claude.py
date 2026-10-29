"""``ClaudeAgent`` and ``AsyncClaudeAgent``: the ``claude-agent`` 0.8.0 API on the Orkexa runtime.

The classes are thin facades: the same constructor arguments and defaults, the
same ``ask()`` semantics, ``messages`` as a list of Messages API dicts, native SDK
objects for ``on_request``/``on_response``/``last_response`` and unchanged SDK
exceptions. Every turn runs through the shared runtime
(:class:`~orkexa.core.runtime.AgentCore`) and the Anthropic adapter.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import json
import os
from typing import Any, Literal

from .agent import Agent
from .async_agent import AsyncAgent
from .core.conversion import HISTORY_FORMAT, parse_history, prepare_history
from .core.runtime import AgentCore, OutputT, RefusalMode, ToolCall
from .core.types import Effort, Message, ToolChoice
from .files import FileStore
from .memory import MemoryBackend
from .pricing import ModelPricing
from .providers import anthropic_codec as codec
from .providers.options import AnthropicOptions
from .retrieval import Index
from .tools import Tool
from .usage import UsageTracker

__all__ = [
    "AsyncClaudeAgent",
    "ClaudeAgent",
    "Effort",
    "Fallbacks",
    "RefusalMode",
    "ThinkingDisplay",
    "ThinkingMode",
    "ToolCall",
]

ThinkingMode = Literal["adaptive", "between_tools", "disabled"]
ThinkingDisplay = Literal["summarized", "omitted", "updates"]
Fallbacks = Literal["default"] | list[dict[str, Any]]


class _ClaudeFacade(AgentCore[OutputT]):
    """The 0.8.0 constructor, attributes and history format, shared by both facades."""

    _native_hooks = True
    _map_errors = False

    def __init__(
        self,
        *,
        model: str = "claude-haiku-5-5",
        system_prompt: str = "You are a helpful assistant.",
        messages: list[dict[str, Any]] | None = None,
        tools: list[Tool] | None = None,
        server_tools: list[dict[str, Any]] | None = None,
        memory: MemoryBackend | None = None,
        files: FileStore | None = None,
        index: Index | None = None,
        output_format: type[OutputT] | None = None,
        effort: Effort | None = "medium",
        thinking: ThinkingMode | None = None,
        thinking_display: ThinkingDisplay | None = None,
        tool_choice: dict[str, Any] | None = None,
        cache_control: bool = True,
        cache_ttl: Literal["5m", "1h"] = "5m",
        max_tokens: int = 16_000,
        max_tool_rounds: int = 8,
        max_parallel_tools: int = 1,
        raise_tool_errors: bool = True,
        stream: bool | None = None,
        eager_input_streaming: bool = False,
        fallbacks: Fallbacks | None = None,
        on_refusal: RefusalMode = "raise",
        request_options: dict[str, Any] | None = None,
        on_text: Callable[[str], Any] | None = None,
        on_thinking: Callable[[str], Any] | None = None,
        on_request: Callable[[dict[str, Any]], Any] | None = None,
        on_response: Callable[[Any], Any] | None = None,
        before_tool: Callable[[ToolCall], Any] | None = None,
        after_tool: Callable[[ToolCall, str, bool], Any] | None = None,
        client: Any | None = None,
        api_key: str | None = None,
        name: str | None = None,
        usage_tracker: UsageTracker | None = None,
        pricing: ModelPricing | None = None,
    ) -> None:
        AnthropicOptions(
            thinking=thinking,
            thinking_display=thinking_display,
            cache_ttl=cache_ttl,
            fallbacks=fallbacks,
            request_options=request_options or {},
            tool_choice=tool_choice,
        )  # validates with the 0.8.0 messages
        self.thinking = thinking
        self.thinking_display = thinking_display
        self.cache_control = cache_control
        self.cache_ttl = cache_ttl
        self.eager_input_streaming = eager_input_streaming
        self.fallbacks = deepcopy(fallbacks)
        self.request_options: dict[str, Any] = deepcopy(request_options) if request_options else {}
        self._legacy_tool_choice: dict[str, Any] = deepcopy(tool_choice) if tool_choice else {"type": "auto"}
        super().__init__(
            model=model,
            provider="anthropic",
            system_prompt=system_prompt,
            messages=messages,
            tools=tools,
            server_tools=server_tools,
            memory=memory,
            files=files,
            index=index,
            output_format=output_format,
            effort=effort,
            max_tokens=max_tokens,
            max_tool_rounds=max_tool_rounds,
            max_parallel_tools=max_parallel_tools,
            raise_tool_errors=raise_tool_errors,
            stream=stream,
            on_refusal=on_refusal,
            on_text=on_text,
            on_thinking=on_thinking,
            on_request=on_request,
            on_response=on_response,
            before_tool=before_tool,
            after_tool=after_tool,
            client=client,
            api_key=api_key,
            name=name or model,
            usage_tracker=usage_tracker,
            pricing=pricing,
        )

    # ---- the 0.8.0 attributes ------------------------------------------------------

    @property  # type: ignore[override]
    def tool_choice(self) -> dict[str, Any]:
        """The native ``tool_choice`` dict (``{"type": "auto"}`` by default)."""
        return self._legacy_tool_choice

    @tool_choice.setter
    def tool_choice(self, value: Any) -> None:
        if isinstance(value, ToolChoice):  # set by the shared constructor
            return
        self._legacy_tool_choice = value

    def _native_tool_choice(self) -> ToolChoice:
        choice = self._legacy_tool_choice
        if choice.get("type") == "any":
            return ToolChoice("required")
        if choice.get("type") == "tool":
            return ToolChoice("tool", name=str(choice.get("name")))
        return ToolChoice("none" if choice.get("type") == "none" else "auto")

    def _provider_options(self) -> AnthropicOptions:
        return AnthropicOptions(
            thinking=self.thinking,
            thinking_display=self.thinking_display,
            cache_control=self.cache_control,
            cache_ttl=self.cache_ttl,
            eager_input_streaming=self.eager_input_streaming,
            fallbacks=deepcopy(self.fallbacks),
            request_options=deepcopy(self.request_options),
            tool_choice=deepcopy(self._legacy_tool_choice),
        )

    def _tool_definitions(self) -> list[dict[str, Any]]:
        """The ``tools`` of the next request (local, server, memory), as sent to the API."""
        from .providers.anthropic import AnthropicProvider

        request = self._build_request([], check=False)
        assert isinstance(self.adapter, AnthropicProvider)
        return self.adapter.tool_definitions(request)

    # ---- the 0.8.0 history format ------------------------------------------------------

    def _init_messages(self, messages: Sequence[Message | Mapping[str, Any]] | None) -> None:
        self.messages = deepcopy(list(messages)) if messages is not None else []

    def _conversation(self) -> list[Message]:
        return codec.decode_messages(self.messages)

    def _commit(self, history: list[Message]) -> None:
        self.messages = codec.encode_messages(history)

    def save_history(self, path: str | os.PathLike[str]) -> None:
        """Write the committed conversation history to a JSON file (Messages API format)."""
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.messages, handle, ensure_ascii=False, indent=1)

    def load_history(self, path: str | os.PathLike[str]) -> None:
        """Replace the conversation history with one written by :meth:`save_history`.

        Thinking blocks are only valid with the model that produced them; load a
        history into an agent with the same model. Orkexa history files of other
        providers are accepted if they convert without loss (otherwise
        :class:`~orkexa.HistoryConversionError`).
        """
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict) and data.get("format") == HISTORY_FORMAT:
            neutral = parse_history(data, codec.decode_messages, str(path))
            self.messages = codec.encode_messages(prepare_history(neutral, "anthropic", "strict"))
        else:
            if not isinstance(data, list) or not all(
                isinstance(m, dict) and m.get("role") in ("user", "assistant", "system") for m in data
            ):
                raise ValueError(f"{path}: not a conversation history")
            self.messages = data
        self.last_response = None


class ClaudeAgent(_ClaudeFacade[OutputT], Agent[OutputT]):
    """A synchronous, stateful Claude agent with strict local tools (``claude-agent`` API).

    With ``output_format=None``, :meth:`ask` returns ``str`` (the agent is a
    ``ClaudeAgent[str]``). With a Pydantic ``BaseModel`` subclass ``M``,
    :meth:`ask` returns an ``M`` instance (the agent is a ``ClaudeAgent[M]``).

    Tool failures raise :class:`~orkexa.ToolExecutionError` by default.
    With ``raise_tool_errors=False`` they are sent back to the model as a
    ``tool_result`` with ``is_error: true`` instead, so it can recover.
    Message history is committed only after a complete turn. A refusal
    (``stop_reason="refusal"``) raises :class:`~orkexa.RefusalError`;
    with ``on_refusal="return"`` the text of the refused turn is returned instead.

    Requests are streamed when ``stream=True``, when ``on_text`` is set, or
    (with ``stream=None``) when ``max_tokens`` exceeds 16,000. ``request_options``
    is merged into every request, e.g. ``{"betas": [...], "inference_geo": "eu"}``;
    any ``betas`` route the request to ``client.beta.messages``.

    Hooks (all optional; they run in the calling thread, or in worker threads
    for parallel tools):

    * ``on_text(text)``: streamed text deltas.
    * ``on_thinking(text)``: streamed thinking deltas; they carry text only with
      ``thinking_display="summarized"`` or ``"updates"``.
    * ``on_request(args)``: the keyword arguments of every API request (do not mutate).
    * ``on_response(response)``: every API response (the SDK object).
    * ``before_tool(call)``: return ``None``/``True`` to run the :class:`ToolCall`,
      or a reason string/``False`` to deny it; a denial goes back to the model as
      an error result.
    * ``after_tool(call, content, is_error)``: the result that goes back to the model.

    ``memory`` adds Anthropic's memory tool; its calls pass through
    ``before_tool``/``after_tool`` like any tool. ``files`` adds the tool ``files``;
    ``index`` adds the tool ``search`` for retrieval with citations.

    This is the same runtime as :class:`~orkexa.Agent` with ``provider="anthropic"``;
    use ``Agent`` for new, provider-independent code.
    """

    _async_name = "AsyncClaudeAgent"


class AsyncClaudeAgent(_ClaudeFacade[OutputT], AsyncAgent[OutputT]):
    """Like :class:`ClaudeAgent`, with ``await agent.ask(...)``.

    Takes the same arguments (``client`` is an ``AsyncAnthropic``). Tools may be
    ``async def`` functions; plain tools run in a worker thread. Up to
    ``max_parallel_tools`` tool calls of one turn run concurrently. Hooks may be
    plain functions or coroutine functions. Memory commands run in a worker thread.

    One agent holds one conversation: a concurrent ``ask()`` of the same agent
    raises :class:`~orkexa.ConcurrentTurnError`. Use one agent per concurrent session.
    """
