"""The asyncio agent: an ``async`` I/O driver around :class:`~orkexa.core.runtime.AgentCore`."""

from __future__ import annotations

import asyncio
from collections.abc import Generator
import inspect
from typing import Any

from .content import Prompt
from .core.runtime import AgentCore, CallModel, Effect, Emit, OutputT, RunTools, ToolCall
from .core.types import (
    Effort,
    ModelRequest,
    ModelResponse,
    ResponseCompleted,
    StreamEvent,
    ToolCallPart,
    ToolResultPart,
)
from .tools import ToolOutput

__all__ = ["AsyncAgent"]


async def _resolve(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


class AsyncAgent(AgentCore[OutputT]):
    """Like :class:`~orkexa.Agent`, with ``await agent.ask(...)``.

    Takes the same arguments (``client`` is the provider's async SDK client). Tools
    may be ``async def`` functions; plain tools run in a worker thread. Up to
    ``max_parallel_tools`` tool calls of one response run concurrently. Hooks may be
    plain functions or coroutine functions. Memory commands run in a worker thread.

    Cancelling ``ask()`` leaves the committed conversation unchanged; tools that run
    in worker threads finish in the background. ``tool_timeout`` cancels ``async``
    tools and abandons sync tools still running in their thread.
    """

    _asynchronous = True

    async def ask(
        self,
        prompt: Prompt,
        *,
        system: str | None = None,
        effort: Effort | None = None,
        max_tokens: int | None = None,
    ) -> OutputT:
        """Send a message, complete tool calls, and commit the conversation.

        Takes the same arguments as :meth:`Agent.ask <orkexa.Agent.ask>`.
        """
        self._enter_turn()
        try:
            return await self._drive(self._turn(prompt, system=system, effort=effort, max_tokens=max_tokens))
        finally:
            self._exit_turn()

    async def count_tokens(self, prompt: Prompt | None = None) -> int:
        """Input tokens of the next request: history, tools, system prompt and ``prompt``."""
        request = self._count_request(prompt)
        try:
            return await self.adapter.acount_tokens(request)
        except Exception as exc:
            failure = self._provider_failure(exc)
            if failure is exc:
                raise
            raise failure from exc

    # ---- effects ------------------------------------------------------------------------

    async def _drive(self, gen: Generator[Effect, Any, OutputT]) -> OutputT:
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
                    value = await self._perform(effect)
                except Exception as exc:
                    error = exc
        finally:
            gen.close()

    async def _perform(self, effect: Effect) -> Any:
        if isinstance(effect, CallModel):
            return await self._call_model(effect.request)
        if isinstance(effect, RunTools):
            return await self._run_tools(effect.calls, effect.request)
        if isinstance(effect, Emit):
            await self._dispatch(effect.event)
            return None
        raise TypeError(f"Unknown effect {effect!r}")

    async def _dispatch(self, event: StreamEvent) -> None:
        for result in self._event_hooks(event):
            await _resolve(result)

    async def _call_model(self, request: ModelRequest) -> ModelResponse:
        native = self.adapter.prepare(request)
        if self.on_request is not None:
            await _resolve(self.on_request(self._request_hook_arg(request, native)))
        if request.stream:
            response = await self._stream(native, request)
        else:
            try:
                response = await self.adapter.acall(native, request)
            except Exception as exc:
                failure = self._provider_failure(exc)
                if failure is exc:
                    raise
                raise failure from exc
        if self.on_response is not None:
            try:
                await _resolve(self.on_response(self._response_hook_arg(response)))
            except BaseException:  # also cancellation: the answered request is billed
                self._record_before_failure(response)
                raise
        return response

    async def _stream(self, native: dict[str, Any], request: ModelRequest) -> ModelResponse:
        events = self.adapter.astream(native, request)
        final: StreamEvent | None = None
        try:
            while True:
                try:
                    event = await events.__anext__()
                except StopAsyncIteration:
                    break
                except Exception as exc:
                    failure = self._provider_failure(exc)
                    if failure is exc:
                        raise
                    raise failure from exc
                if isinstance(event, ResponseCompleted):
                    final = event
                try:
                    await self._dispatch(event)
                except BaseException:
                    if isinstance(final, ResponseCompleted):
                        self._record_before_failure(final.response)
                    raise
        finally:
            close = getattr(events, "aclose", None)
            if close is not None:
                await close()
        return self._final_response(final)

    async def _run_tools(
        self, calls: tuple[ToolCallPart, ...], request: ModelRequest
    ) -> list[ToolResultPart]:
        slots = asyncio.Semaphore(self.max_parallel_tools)
        return list(await asyncio.gather(*(self._run_tool(call, request, slots) for call in calls)))

    async def _run_tool(
        self, part: ToolCallPart, request: ModelRequest, slots: asyncio.Semaphore
    ) -> ToolResultPart:
        call = self._tool_call(part)
        async with slots:
            denial = None
            if self.before_tool is not None:
                denial = self._denial(await _resolve(self.before_tool(call)))
            content: ToolOutput
            if denial is not None:
                content, is_error = denial, True
            else:
                try:
                    content, is_error = await self._invoke(call, part)
                except Exception as exc:
                    content, is_error = self._tool_failure(exc, request)
            if self.after_tool is not None:
                await _resolve(self.after_tool(call, self._hook_content(content), is_error))
        return self._tool_result(call, content, is_error)

    async def _invoke(self, call: ToolCall, part: ToolCallPart) -> tuple[ToolOutput, bool]:
        self._check_arguments(part)
        if self._is_memory_call(call):
            return await asyncio.to_thread(self._run_memory, part.arguments)
        tool = self._lookup_tool(call.name)
        if self.tool_timeout is None:
            return await tool.aexecute(part.arguments), False
        try:
            return await asyncio.wait_for(tool.aexecute(part.arguments), self.tool_timeout), False
        except asyncio.TimeoutError:
            raise self._timeout_error(call) from None
