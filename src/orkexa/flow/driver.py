"""The reference driver: runs a flow with asyncio and one handler per node type.

It shows how a driver uses the kernel and is the baseline the conformance suite checks
other drivers against. A production driver (Nodestro's runtime) adds persistence,
streaming and permissions around the same kernel calls.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
import time
from typing import Any

from . import kernel
from .model import Flow, FlowEvent, FlowState, Invocation, Node, Outcome, Result, Step, Stop
from .validate import FlowError

__all__ = ["FlowRun", "Handler", "TaskContext", "TaskResult", "run_flow", "run_flow_sync"]


@dataclass(frozen=True)
class TaskContext:
    """What a handler gets for one invocation.

    Attributes:
        flow: The flow being run.
        node: The node to execute.
        invocation: The invocation (id, iteration path, attempt, arrivals).
        inputs: The outputs that arrive over active connections, by source node id. A loop
            head's later iteration also gets the loop source's previous output this way.
        outputs: Every output so far, by invocation id (read-only).
    """

    flow: Flow
    node: Node
    invocation: Invocation
    inputs: Mapping[str, Any]
    outputs: Mapping[str, Any]


@dataclass(frozen=True)
class TaskResult:
    """What a handler returns; ``None`` from a handler means "succeeded, no output".

    ``ports`` must be set for a branching node that succeeded.
    """

    output: Any = None
    ports: tuple[str, ...] | None = None
    status: str = "succeeded"
    error: str | None = None
    stop: Stop | None = None


Handler = Callable[[TaskContext], Awaitable["TaskResult | Result | None"]]


@dataclass
class FlowRun:
    """The result of :func:`run_flow`: store ``state`` and ``outputs`` to continue later."""

    state: FlowState
    outputs: dict[str, Any] = field(default_factory=dict)
    events: list[FlowEvent] = field(default_factory=list)

    @property
    def outcome(self) -> Outcome:
        return kernel.outcome(self.state)


def _to_result(value: TaskResult | Result | None) -> tuple[Result, Any]:
    if value is None:
        return Result(status="succeeded"), None
    if isinstance(value, Result):
        return value, None
    return (
        Result.model_validate(
            {"status": value.status, "ports": value.ports, "error": value.error, "stop": value.stop}
        ),
        value.output,
    )


async def run_flow(
    flow: Flow,
    handlers: Mapping[str, Handler],
    *,
    state: FlowState | None = None,
    outputs: Mapping[str, Any] | None = None,
    resume: bool | str = False,
    clock: Callable[[], float] = time.monotonic,
    on_event: Callable[[FlowEvent], None] | None = None,
) -> FlowRun:
    """Run ``flow`` until it is finished, suspended or stopped.

    Args:
        flow: The flow to run.
        handlers: One coroutine function per node type.
        state: A stored state to continue (``None`` starts the flow).
        outputs: The stored outputs that belong to ``state``.
        resume: ``True`` dispatches every suspended invocation again, an invocation id only that one.
        clock: The time source for ``max_duration_s`` (seconds; injectable for tests).
        on_event: Called with every kernel event in order.

    A handler that raises makes its invocation fail with ``flow.handler_error``; a node type
    without handler fails with ``flow.no_handler``. When the flow is stopped, running
    handlers are cancelled and reported as ``cancelled``.

    Raises:
        FlowError: ``flow.interrupted`` when ``state`` has running invocations: they were
            dispatched by a driver that stopped, so whether they took effect is unknown.
    """
    run = FlowRun(state=state or FlowState(), outputs=dict(outputs or {}))
    nodes = {node.id: node for node in flow.nodes}

    def take(step: Step) -> list[Invocation]:
        run.state = step.state
        run.events.extend(step.events)
        if on_event is not None:
            for event in step.events:
                on_event(event)
        return [action.invocation for action in step.actions]

    if state is None:
        ready = take(kernel.start(flow, now=clock()))
    else:
        interrupted = [item.id for item in state.invocations if item.status == "running"]
        if interrupted:
            raise FlowError(
                "flow.interrupted", f"Invocations {interrupted} were running when the driver stopped"
            )
        if resume:
            ready = take(kernel.resume(flow, state, None if resume is True else resume, now=clock()))
        else:
            ready = take(kernel.advance(flow, state, now=clock()))

    async def execute(item: Invocation) -> tuple[Result, Any]:
        node = nodes[item.node]
        handler = handlers.get(node.type)
        if handler is None:
            return Result(status="failed", error="flow.no_handler"), None
        inputs = {
            run.state.invocation(arrival.source).node: run.outputs.get(arrival.source)
            for arrival in item.arrivals
        }
        context = TaskContext(flow=flow, node=node, invocation=item, inputs=inputs, outputs=dict(run.outputs))
        try:
            return _to_result(await handler(context))
        except asyncio.CancelledError:
            raise
        except Exception:  # recorded on the invocation; the flow continues with other branches
            return Result(status="failed", error="flow.handler_error"), None

    tasks: dict[asyncio.Task[tuple[Result, Any]], str] = {}
    while True:
        for item in ready:
            tasks[asyncio.ensure_future(execute(item))] = item.id
        ready = []
        if not tasks:
            break
        timeout = None
        limit = flow.limits.max_duration_s
        if limit is not None and run.state.started_at is not None and run.state.stop is None:
            timeout = max(0.0, run.state.started_at + limit - clock())
        done, _ = await asyncio.wait(tasks, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        if not done:
            ready = take(kernel.advance(flow, run.state, now=clock()))
        for task in done:
            invocation_id = tasks.pop(task)
            if task.cancelled():
                stop = run.state.stop
                result, output = (
                    Result(status="cancelled", error=stop.code if stop else "flow.cancelled"),
                    None,
                )
            else:
                result, output = task.result()
            if output is not None:
                run.outputs[invocation_id] = output
            ready.extend(take(kernel.complete(flow, run.state, invocation_id, result, now=clock())))
        if run.state.stop is not None:
            for task in tasks:
                task.cancel()
    return run


def run_flow_sync(flow: Flow, handlers: Mapping[str, Handler], **options: Any) -> FlowRun:
    """:func:`run_flow` for synchronous code (starts its own event loop)."""
    return asyncio.run(run_flow(flow, handlers, **options))
