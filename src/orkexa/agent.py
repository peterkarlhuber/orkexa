"""The synchronous agent: a blocking I/O driver around :class:`~orkexa.core.runtime.AgentCore`."""

from __future__ import annotations

from collections.abc import Generator, Mapping
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from contextvars import copy_context
from typing import Any

from .content import Prompt
from .core.checkpoint import TurnCheckpoint
from .core.runtime import (
    AgentCore,
    CallModel,
    Effect,
    Emit,
    OutputT,
    RefusalMode,
    RunTools,
    ToolCall,
    WaitForBudget,
    _Waiting,
)
from .core.types import (
    Effort,
    ModelRequest,
    ModelResponse,
    ResponseCompleted,
    StreamEvent,
    ToolCallPart,
    ToolResultPart,
)
from .errors import ToolCallSuspended
from .tools import ToolOutput

_BUDGET_WAIT = 0.25
"""Seconds a request waits for a budget release before it checks again."""

__all__ = ["Agent", "RefusalMode", "ToolCall"]


class Agent(AgentCore[OutputT]):
    """A synchronous, stateful agent with strict local tools, for any provider.

    The provider comes from the model name or ``provider=``::

        Agent(model="anthropic:claude-opus-5-5")
        Agent(model="openai:gpt-5.5")
        Agent(model="gemini:gemini-3-pro")
        Agent(provider="ollama", model="llama3.1:8b")

    A model without registered prefix belongs to the default provider ``anthropic``.
    With ``output_format=None`` :meth:`ask` returns ``str``; with a Pydantic
    ``BaseModel`` subclass ``M`` it returns an ``M`` instance (``Agent[M]``).

    Args:
        model: Model id, optionally ``"provider:model"``.
        provider: Registered provider name; must agree with a model prefix.
        system_prompt: The system instruction (``None``: none).
        messages: An initial conversation (:class:`~orkexa.core.types.Message` list).
        tools: ``@Tool`` functions (sync only; use :class:`AsyncAgent` for async tools).
        server_tools: Provider-native tools in the provider's own format (capability ``server_tools``).
        memory: A :class:`~orkexa.MemoryBackend` (native memory tool on Anthropic,
            a regular function tool elsewhere).
        files: A :class:`~orkexa.FileStore`; adds the ``files`` tool.
        index: An :class:`~orkexa.Index`; adds the ``search`` tool.
        output_format: A Pydantic model for structured output.
        effort: ``"low"`` ... ``"max"``; mapped only where the provider has a matching
            reasoning setting (otherwise :class:`~orkexa.UnsupportedCapabilityError`).
        tool_choice: ``"auto"``, ``"none"``, ``"required"`` or ``{"name": "tool"}``.
        max_tokens: Output token limit per request.
        max_tool_rounds: Maximum tool rounds per :meth:`ask`.
        max_parallel_tools: Tool calls of one response run in up to this many threads.
        raise_tool_errors: Raise tool failures (default) or send them back to the model.
        tool_timeout: Seconds per tool call; a call that takes longer fails with
            :class:`~orkexa.ToolExecutionError` (its thread is abandoned, not killed).
        stream: Stream requests (``None``: when a streaming hook is set).
        on_refusal: ``"raise"`` (:class:`~orkexa.RefusalError`) or ``"return"`` the text.
        on_text: Streamed text deltas.
        on_thinking: Streamed visible reasoning deltas.
        on_event: Every normalized :data:`~orkexa.core.types.StreamEvent`.
        on_request: The :class:`~orkexa.core.types.ModelRequest` of every request.
        on_response: The :class:`~orkexa.core.types.ModelResponse` of every request.
        before_tool: Return ``None``/``True`` to run a :class:`ToolCall`, or a reason
            string/``False`` to deny it (the denial goes back to the model).
        after_tool: ``(call, content, is_error)`` of every result.
        options: Typed provider options (``AnthropicOptions``, ``OpenAIOptions``, ...).
        client: A preconfigured SDK client.
        api_key: API key for this agent's provider only.
        base_url: Endpoint override (a custom URL needs an explicit ``api_key``).
        timeout: Client timeout in seconds.
        max_retries: Client retries.
        name: Agent name for usage records (default: the model id).
        usage_tracker: A shared :class:`~orkexa.UsageTracker` (or a budget tracker).
        pricing: Prices for cost calculation: a :class:`~orkexa.pricing.ModelPrice` or
            legacy :class:`~orkexa.ModelPricing` (an override for this agent), a
            :class:`~orkexa.pricing.PriceResolver` or a
            :class:`~orkexa.pricing.PriceSnapshot`. Default: the tracker's
            ``price_resolver``, else :func:`~orkexa.pricing.default_resolver`.
        capabilities: Explicit capability overrides, e.g. ``{"tools": "supported"}``.
        history_policy: ``"strict"`` (default): continuing a conversation of another
            provider raises if reasoning or native parts would be lost; ``"convert"``
            drops them with a warning.

    The conversation is committed only after a complete turn. Usage of every API
    response is recorded immediately, also when the turn fails later; tools that
    already ran are not undone. One agent holds one conversation: a concurrent
    :meth:`ask` raises :class:`~orkexa.ConcurrentTurnError`.
    """

    def ask(
        self,
        prompt: Prompt,
        *,
        system: str | None = None,
        effort: Effort | None = None,
        max_tokens: int | None = None,
    ) -> OutputT:
        """Send a message, complete tool calls, and commit the conversation.

        ``prompt`` is a string or a list of content blocks and strings, e.g.
        ``[image("chart.png"), "Describe the trend."]`` (see :mod:`orkexa.content`).
        ``system`` adds an operator instruction as a ``system`` message after the
        prompt (capability ``system_messages``). ``effort`` and ``max_tokens``
        override the agent's settings for this turn only.
        """
        self._enter_turn()
        try:
            return self._drive(self._turn(prompt, system=system, effort=effort, max_tokens=max_tokens))
        finally:
            self._exit_turn()

    def resume(self, checkpoint: TurnCheckpoint | Mapping[str, Any]) -> OutputT:
        """Continue a turn that stopped at a waiting tool call (:class:`~orkexa.TurnSuspended`).

        The waiting calls go through ``before_tool`` again (``ToolCall.resumed`` is
        ``True``); calls that already ran are not run again; the model gets all results
        of that round and the turn continues like :meth:`ask`. The checkpoint must come
        from this agent with the same prompt, tools and limits
        (:class:`~orkexa.CheckpointError` otherwise). The completed turn's conversation
        (from the checkpoint) becomes the agent's conversation. A call that waits again
        raises :class:`~orkexa.TurnSuspended` with a new checkpoint.
        """
        self._enter_turn()
        try:
            return self._drive(self._resume_turn(checkpoint))
        finally:
            self._exit_turn()

    def count_tokens(self, prompt: Prompt | None = None) -> int:
        """Input tokens of the next request: history, tools, system prompt and ``prompt``."""
        request = self._count_request(prompt)
        try:
            return self.adapter.count_tokens(request)
        except Exception as exc:
            failure = self._provider_failure(exc)
            if failure is exc:
                raise
        raise failure  # outside the handler: no link to the SDK exception

    # ---- effects ------------------------------------------------------------------------

    def _drive(self, gen: Generator[Effect, Any, OutputT]) -> OutputT:
        value: Any = None
        error: Exception | None = None
        try:
            while True:
                try:
                    effect = gen.throw(error) if error is not None else gen.send(value)
                except StopIteration as stop:
                    result: OutputT = stop.value
                    return result
                value, error = None, None
                try:
                    value = self._perform(effect)
                except Exception as exc:
                    error = exc
        finally:
            gen.close()

    def _perform(self, effect: Effect) -> Any:
        if isinstance(effect, CallModel):
            return self._call_model(effect.request)
        if isinstance(effect, RunTools):
            return self._run_tools(effect.calls, effect.request)
        if isinstance(effect, Emit):
            self._event_hooks(effect.event)
            return None
        if isinstance(effect, WaitForBudget):
            effect.tracker.wait_for_release(_BUDGET_WAIT)
            return None
        raise TypeError(f"Unknown effect {effect!r}")

    def _call_model(self, request: ModelRequest) -> ModelResponse:
        native = self.adapter.prepare(request)
        if self.on_request is not None:
            self.on_request(self._request_hook_arg(request, native))
        if request.stream:
            response = self._stream(native, request)
        else:
            failure: Exception | None = None
            try:
                response = self.adapter.call(native, request)
            except Exception as exc:
                failure = self._provider_failure(exc)
                if failure is exc:
                    raise
            if failure is not None:
                raise failure  # outside the handler: no link to the SDK exception
        if self.on_response is not None:
            try:
                self.on_response(self._response_hook_arg(response))
            except BaseException:  # also cancellation: the answered request is billed
                self._record_before_failure(response)
                raise
        return response

    def _stream(self, native: dict[str, Any], request: ModelRequest) -> ModelResponse:
        events = self.adapter.stream(native, request)
        final: StreamEvent | None = None
        try:
            while True:
                failure: Exception | None = None
                try:
                    event = next(events)
                except StopIteration:
                    break
                except Exception as exc:
                    failure = self._provider_failure(exc)
                    if failure is exc:
                        raise
                if failure is not None:
                    raise failure  # outside the handler: no link to the SDK exception
                if isinstance(event, ResponseCompleted):
                    final = event
                try:
                    self._event_hooks(event)
                except BaseException:
                    if isinstance(final, ResponseCompleted):
                        self._record_before_failure(final.response)
                    raise
        finally:
            close = getattr(events, "close", None)
            if close is not None:
                close()
        return self._final_response(final)

    def _run_tools(
        self, calls: tuple[ToolCallPart, ...], request: ModelRequest
    ) -> list[ToolResultPart | _Waiting]:
        workers = min(self.max_parallel_tools, len(calls))
        if workers <= 1:
            return [self._run_tool(call, request) for call in calls]
        # Each call runs in a copy of the caller's context, so context variables
        # (e.g. the orchestrator's delegation depth) reach the worker threads.
        contexts = [copy_context() for _ in calls]
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"{self.name}-tool") as pool:
            # map() keeps the order and re-raises the first failed call's error.
            return list(pool.map(lambda ctx, call: ctx.run(self._run_tool, call, request), contexts, calls))

    def _run_tool(self, part: ToolCallPart, request: ModelRequest) -> ToolResultPart | _Waiting:
        call = self._tool_call(part)
        try:
            denial = self._denial(self.before_tool(call)) if self.before_tool is not None else None
        except ToolCallSuspended as exc:
            return self._waiting(call, exc)
        content: ToolOutput
        if denial is not None:
            content, is_error = denial, True
        else:
            try:
                content, is_error = self._invoke(call, part)
            except Exception as exc:
                content, is_error = self._tool_failure(exc, request)
        if self.after_tool is not None:
            self.after_tool(call, self._hook_content(content), is_error)
        return self._tool_result(call, content, is_error)

    def _invoke(self, call: ToolCall, part: ToolCallPart) -> tuple[ToolOutput, bool]:
        self._check_arguments(part)
        if self._is_memory_call(call):
            return self._run_memory(part.arguments)
        tool = self._lookup_tool(call.name)
        if self.tool_timeout is None:
            return tool.execute(part.arguments), False
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"{self.name}-{call.name}")
        try:
            future = pool.submit(copy_context().run, tool.execute, part.arguments)
            try:
                return future.result(timeout=self.tool_timeout), False
            except FutureTimeout:
                raise self._timeout_error(call) from None
        finally:
            pool.shutdown(wait=False)
