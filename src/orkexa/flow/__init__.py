"""Flow kernel: one deterministic semantics for workflows of agents, people and tools.

``orkexa.flow`` decides *what runs next* in a flow; it never runs anything itself. The
kernel is a set of pure functions (ADR 0009): given the specification, the stored state
and one input (a start, a result, a cancel, a resume, the current time) it returns the
new state, the invocations to execute and the events that happened. No clock, network,
file system or randomness is involved, so a stored state continues identically in every
driver and process.

* :class:`Flow` with :class:`Node` and :class:`Edge`: a directed graph; branching nodes
  open ports; a :class:`Loop` edge repeats a region up to a mandatory iteration limit.
* :class:`FlowState`: every :class:`Invocation` (one execution of one node in one
  iteration), loop counters and the stop reason; plain JSON.
* :func:`start`, :func:`complete`, :func:`advance`, :func:`cancel`, :func:`resume` and
  :func:`outcome`: the kernel.
* :func:`run_flow`: the asyncio reference driver with one handler per node type.
* :mod:`orkexa.flow.conformance`: scripted cases every driver must pass.

Quick start::

    from orkexa.flow import Edge, Flow, Node, TaskResult, run_flow_sync

    flow = Flow(
        nodes=(Node(id="greet", type="say"), Node(id="shout", type="say")),
        edges=(Edge(id="e1", source="greet", target="shout"),),
    )

    async def say(context):
        return TaskResult(output=f"{context.node.id}: {dict(context.inputs)}")

    run = run_flow_sync(flow, {"say": say})
    print(run.outcome, run.outputs)
"""

from __future__ import annotations

from .driver import FlowRun, Handler, TaskContext, TaskResult, run_flow, run_flow_sync
from .kernel import advance, cancel, complete, outcome, resume, start
from .model import (
    CONTRACT_VERSION,
    EVENT_SCHEMA,
    Arrival,
    Edge,
    Flow,
    FlowEvent,
    FlowIssue,
    FlowState,
    Invocation,
    InvocationStatus,
    Join,
    Limits,
    Loop,
    Node,
    Outcome,
    Result,
    RunNode,
    Step,
    Stop,
)
from .validate import FlowError, validate

__all__ = [
    "CONTRACT_VERSION",
    "EVENT_SCHEMA",
    "Arrival",
    "Edge",
    "Flow",
    "FlowError",
    "FlowEvent",
    "FlowIssue",
    "FlowRun",
    "FlowState",
    "Handler",
    "Invocation",
    "InvocationStatus",
    "Join",
    "Limits",
    "Loop",
    "Node",
    "Outcome",
    "Result",
    "RunNode",
    "Step",
    "Stop",
    "TaskContext",
    "TaskResult",
    "advance",
    "cancel",
    "complete",
    "outcome",
    "resume",
    "run_flow",
    "run_flow_sync",
    "start",
    "validate",
]
