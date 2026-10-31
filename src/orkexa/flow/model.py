"""The data model of ``orkexa.flow``: specifications, state, results, actions and events.

Everything here is an immutable Pydantic model that serializes to JSON, so a driver
can store a :class:`FlowState` next to its own run record and continue it later, in
another process, with another driver instance.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "CONTRACT_VERSION",
    "EVENT_SCHEMA",
    "Arrival",
    "Edge",
    "Flow",
    "FlowEvent",
    "FlowIssue",
    "FlowState",
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
]

#: Version of the contract between the kernel and its drivers (specification, state,
#: results and events). The major number changes when a stored state or a driver
#: written for an older version would be interpreted differently.
CONTRACT_VERSION = "1.0"
#: ``schema`` of every :class:`FlowEvent`.
EVENT_SCHEMA = "orkexa.flow-event/1"

#: ``ready``: created, waits to be dispatched. ``running``: dispatched to the driver.
#: ``suspended``: waits for something outside the flow (a person, a child flow).
#: The others are final.
InvocationStatus = Literal[
    "ready", "running", "suspended", "succeeded", "failed", "skipped", "blocked", "cancelled"
]
#: ``all``: a node waits until every predecessor is settled (the default).
#: ``any``: a node starts as soon as one incoming connection is active.
Join = Literal["all", "any"]

_FINAL = frozenset({"succeeded", "failed", "skipped", "blocked", "cancelled"})


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# ---- limits ----------------------------------------------------------------------------


class Limits(_Frozen):
    """Limits of one scope (a flow, or a child flow inside its parent's limits).

    ``None`` means "no limit of this kind". :meth:`within` narrows a child's limits to its
    parent's: a child can never get more than its parent has left.

    Attributes:
        max_invocations: Maximum number of dispatched invocations (node executions).
        max_parallel: Maximum number of invocations running at the same time.
        max_iterations: Upper bound for every loop's ``max_iterations``.
        max_duration_s: Maximum wall time of the flow in seconds (needs ``now`` from the driver).
        max_depth: Maximum nesting depth of child flows (enforced by the driver).
        budget_usd: Cost budget of the scope (enforced by the driver's usage tracking).
    """

    max_invocations: int | None = Field(default=None, ge=1)
    max_parallel: int | None = Field(default=None, ge=1)
    max_iterations: int | None = Field(default=None, ge=1)
    max_duration_s: float | None = Field(default=None, gt=0)
    max_depth: int | None = Field(default=None, ge=0)
    budget_usd: Decimal | None = Field(default=None, ge=0)

    def within(self, parent: Limits) -> Limits:
        """These limits, never above ``parent``'s (the smaller value of each kind)."""

        def low(mine: Any, theirs: Any) -> Any:
            if mine is None:
                return theirs
            if theirs is None:
                return mine
            return min(mine, theirs)

        return Limits(
            max_invocations=low(self.max_invocations, parent.max_invocations),
            max_parallel=low(self.max_parallel, parent.max_parallel),
            max_iterations=low(self.max_iterations, parent.max_iterations),
            max_duration_s=low(self.max_duration_s, parent.max_duration_s),
            max_depth=low(self.max_depth, parent.max_depth),
            budget_usd=low(self.budget_usd, parent.budget_usd),
        )


# ---- specification ---------------------------------------------------------------------


class Loop(_Frozen):
    """Makes an edge a loop edge: taking it starts the next iteration of its loop.

    The loop's body is every node on a path from the edge's target to its source. A loop
    edge must leave a branching node through a port (its condition), and the body runs at
    most ``max_iterations`` times in total.

    Attributes:
        max_iterations: How often the body may run, the first run included (2 or more).
        on_exhausted: When the edge is taken after the last allowed iteration: ``fail``
            stops the flow as failed (``flow.loop_exhausted``); ``continue`` ignores the edge,
            so the flow goes on along the branching node's other open ports.
    """

    max_iterations: int = Field(ge=2, le=1000)
    on_exhausted: Literal["fail", "continue"] = "fail"


class Node(_Frozen):
    """One step of a flow.

    The kernel knows nothing about what a node does: ``type`` and ``data`` are for the
    driver. It only needs to know whether the node branches.

    Attributes:
        id: Unique id within the flow.
        type: What the driver executes (``"agent"``, ``"approval"``, ...).
        ports: The output ports of a branching node; ``None`` for a node with one output.
        join: When the node may start (see :data:`Join`).
        data: Free data for the driver; never read by the kernel.
    """

    id: str = Field(min_length=1, max_length=200)
    type: str = "task"
    ports: tuple[str, ...] | None = None
    join: Join = "all"
    data: Any = None


class Edge(_Frozen):
    """A connection from ``source`` to ``target``.

    Attributes:
        id: Unique id within the flow.
        source: The node the edge leaves.
        target: The node the edge enters.
        port: The port of a branching source this edge belongs to.
        loop: Set for a loop edge (see :class:`Loop`).
    """

    id: str = Field(min_length=1, max_length=200)
    source: str
    target: str
    port: str | None = None
    loop: Loop | None = None


class Flow(_Frozen):
    """A flow specification: nodes, edges and the limits of its scope."""

    nodes: tuple[Node, ...] = ()
    edges: tuple[Edge, ...] = ()
    limits: Limits = Field(default_factory=Limits)
    contract: str = CONTRACT_VERSION


class FlowIssue(_Frozen):
    """A problem found by :func:`orkexa.flow.validate`; ``code`` is stable, ``message`` English."""

    code: str
    message: str
    node: str | None = None
    edge: str | None = None


# ---- state -----------------------------------------------------------------------------


class Arrival(_Frozen):
    """An active incoming connection of an invocation: the edge and the invocation it came from."""

    edge: str
    source: str


class Invocation(_Frozen):
    """One execution of one node.

    Attributes:
        id: Unique and stable within the flow's state (``node/path/attempt``).
        node: The node's id.
        path: The iteration of every loop around the node, outermost first (``()`` outside loops).
        attempt: 1 for the first try; higher for retries of the same iteration.
        status: See :data:`InvocationStatus`.
        ports: The opened ports of a branching node once it succeeded.
        arrivals: The active incoming connections it was started with.
        error: An error code when it did not succeed.
        cause: The invocation that caused it (the loop source for a loop head's new iteration).
        dispatches: How often it was handed to the driver (a resumed invocation runs again).
    """

    id: str
    node: str
    path: tuple[int, ...] = ()
    attempt: int = 1
    status: InvocationStatus = "ready"
    ports: tuple[str, ...] | None = None
    arrivals: tuple[Arrival, ...] = ()
    error: str | None = None
    cause: str | None = None
    dispatches: int = 0

    @property
    def final(self) -> bool:
        return self.status in _FINAL


class Stop(_Frozen):
    """Why a flow stopped before it ran out of work.

    ``outcome`` is ``cancelled`` (somebody stopped it) or ``failed`` (a limit or a loop
    ran out). ``code`` is stable; ``params`` carry details for messages.
    """

    code: str
    outcome: Literal["cancelled", "failed"] = "cancelled"
    params: dict[str, Any] = Field(default_factory=dict)


class FlowState(_Frozen):
    """Everything the kernel needs to continue a flow; store it as JSON.

    Attributes:
        contract: The contract version the state was written with.
        invocations: Every invocation in the order it was created.
        iterations: The current iteration per loop and enclosing iteration (``"<edge>@<path>"``).
        looped: Invocations whose loop edge was already followed.
        exhausted: Loops (keys as in ``iterations``) that ran out of iterations.
        stop: Set when the flow was stopped.
        dispatched: Number of invocations handed to the driver so far.
        started_at: The ``now`` the flow started at (driver clock), when given.
    """

    contract: str = CONTRACT_VERSION
    invocations: tuple[Invocation, ...] = ()
    iterations: dict[str, int] = Field(default_factory=dict)
    looped: tuple[str, ...] = ()
    exhausted: tuple[str, ...] = ()
    stop: Stop | None = None
    dispatched: int = 0
    started_at: float | None = None

    @field_validator("contract")
    @classmethod
    def _known_major(cls, value: str) -> str:
        if value.split(".")[0] != CONTRACT_VERSION.split(".")[0]:
            raise ValueError(f"flow state contract {value} is not supported (expected {CONTRACT_VERSION})")
        return value

    def invocation(self, invocation_id: str) -> Invocation:
        for item in self.invocations:
            if item.id == invocation_id:
                return item
        raise KeyError(invocation_id)

    def latest(self, node: str) -> Invocation | None:
        """The most recently created invocation of ``node``, if any."""
        for item in reversed(self.invocations):
            if item.node == node:
                return item
        return None

    def of(self, node: str) -> tuple[Invocation, ...]:
        """Every invocation of ``node`` in creation order."""
        return tuple(item for item in self.invocations if item.node == node)


# ---- results, actions, events ----------------------------------------------------------


class Result(_Frozen):
    """What a driver reports when an invocation ends or pauses.

    Attributes:
        status: ``succeeded``, ``failed``, ``cancelled`` or ``suspended`` (waits for
            something outside the flow; :func:`orkexa.flow.resume` dispatches it again).
        ports: The opened ports of a branching node (required when it succeeded).
        error: An error code for ``failed`` and ``cancelled``.
        stop: Stop the whole flow (for example: a rejection that ends the run).
    """

    status: Literal["succeeded", "failed", "cancelled", "suspended"]
    ports: tuple[str, ...] | None = None
    error: str | None = None
    stop: Stop | None = None


class RunNode(_Frozen):
    """Action: execute this invocation and report its :class:`Result`."""

    kind: Literal["run_node"] = "run_node"
    invocation: Invocation


class FlowEvent(_Frozen):
    """An event of the kernel.

    Kinds: ``flow_started``, ``invocation_started`` (dispatched), ``invocation_settled``
    (final or suspended), ``invocation_resumed``, ``loop_iteration``, ``limit_reached`` and
    ``flow_stopped``. Drivers add their own events (model output, tool calls) with the same
    envelope and their own kinds.
    """

    schema_: str = Field(default=EVENT_SCHEMA, alias="schema")
    kind: str
    invocation: str | None = None
    node: str | None = None
    path: tuple[int, ...] = ()
    attempt: int | None = None
    cause: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)


class Step(_Frozen):
    """The result of one kernel call: the new state, what to execute and what happened."""

    state: FlowState
    actions: tuple[RunNode, ...] = ()
    events: tuple[FlowEvent, ...] = ()


#: The state of a flow as a whole (see :func:`orkexa.flow.outcome`).
Outcome = Literal["running", "suspended", "succeeded", "failed", "cancelled"]
