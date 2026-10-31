"""The flow kernel: pure functions from a specification, a stored state and an input to a new state.

The kernel performs no I/O. It reads no clock (time comes in as ``now``), starts no task
and calls no model. A driver calls :func:`start`, executes the :class:`RunNode` actions
it gets back, reports each result with :func:`complete` and stores the returned
:class:`FlowState` wherever it likes. Equal inputs always give equal outputs, so a
stored state continues the same way in every driver and every process.
"""

from __future__ import annotations

from typing import Any, Literal

from .model import (
    Arrival,
    Edge,
    Flow,
    FlowEvent,
    FlowState,
    Invocation,
    InvocationStatus,
    Outcome,
    Result,
    RunNode,
    Step,
    Stop,
)
from .validate import FlowError, _Structure, structure

__all__ = ["advance", "cancel", "complete", "outcome", "resume", "start"]

_NOT_OK = frozenset({"failed", "blocked", "cancelled"})
_Kind = Literal["final", "paused", "pending"]


def start(flow: Flow, *, now: float | None = None) -> Step:
    """Start ``flow``: validate it and dispatch every node that can run.

    Raises:
        FlowError: ``flow.invalid`` with every issue of an invalid flow.
    """
    work = _Work(flow, FlowState(started_at=now))
    work.event("flow_started", data={"contract": flow.contract})
    return work.run(now)


def advance(flow: Flow, state: FlowState, *, now: float | None = None) -> Step:
    """Resolve and dispatch what can run now; checks the time limit when ``now`` is given."""
    return _Work(flow, state).run(now)


def complete(
    flow: Flow, state: FlowState, invocation_id: str, result: Result, *, now: float | None = None
) -> Step:
    """Record the result of a dispatched invocation and continue the flow.

    Raises:
        FlowError: ``flow.unknown_invocation``, ``flow.not_running`` (it was not dispatched
            or is already settled) or ``flow.result_ports`` (ports do not fit the node).
    """
    work = _Work(flow, state)
    work.record(invocation_id, result)
    return work.run(now)


def cancel(
    flow: Flow,
    state: FlowState,
    *,
    code: str = "flow.cancelled",
    params: dict[str, Any] | None = None,
    now: float | None = None,
) -> Step:
    """Stop the flow as cancelled.

    Waiting and suspended invocations are cancelled at once and nothing new is dispatched.
    Running invocations stay ``running`` until the driver reports them: stopping them is
    the driver's job. Cancelling a stopped flow changes nothing.
    """
    work = _Work(flow, state)
    work.halt(Stop(code=code, outcome="cancelled", params=params or {}))
    return work.run(now)


def resume(
    flow: Flow, state: FlowState, invocation_id: str | None = None, *, now: float | None = None
) -> Step:
    """Dispatch a suspended invocation again (all suspended ones when ``invocation_id`` is ``None``).

    Raises:
        FlowError: ``flow.unknown_invocation`` or ``flow.not_suspended``.
    """
    work = _Work(flow, state)
    work.wake(invocation_id)
    return work.run(now)


def outcome(state: FlowState) -> Outcome:
    """The state of the flow as a whole.

    ``running`` while an invocation waits for dispatch or runs; then the stop's outcome
    if it was stopped; ``suspended`` while an invocation waits for something outside the
    flow; ``failed`` when any invocation failed, was blocked or cancelled; else ``succeeded``.
    """
    statuses = {item.status for item in state.invocations}
    if statuses & {"ready", "running"}:
        return "running"
    if state.stop is not None:
        return state.stop.outcome
    if "suspended" in statuses:
        return "suspended"
    if statuses & _NOT_OK:
        return "failed"
    return "succeeded"


def _key(loop: str, outer: tuple[int, ...]) -> str:
    return f"{loop}@{'.'.join(str(index) for index in outer)}"


def _invocation_id(node: str, path: tuple[int, ...], attempt: int) -> str:
    return f"{node}/{'.'.join(str(index) for index in path)}/{attempt}"


class _Work:
    """A mutable working copy of a state for one kernel call."""

    def __init__(self, flow: Flow, state: FlowState) -> None:
        self.flow = flow
        self.s: _Structure = structure(flow)
        self.invocations = list(state.invocations)
        self.iterations = dict(state.iterations)
        self.looped = list(state.looped)
        self.exhausted = list(state.exhausted)
        self.stop = state.stop
        self.dispatched = state.dispatched
        self.started_at = state.started_at
        self.contract = state.contract
        self.actions: list[RunNode] = []
        self.events: list[FlowEvent] = []
        self.latest: dict[tuple[str, tuple[int, ...]], int] = {}
        self.by_id: dict[str, int] = {}
        for position, item in enumerate(self.invocations):
            if item.node not in self.s.nodes:
                raise FlowError(
                    "flow.state_mismatch", f"The state has an invocation of the unknown node {item.node!r}"
                )
            self.latest[(item.node, item.path)] = position
            self.by_id[item.id] = position

    # ---- helpers ---------------------------------------------------------------------

    def event(self, kind: str, item: Invocation | None = None, data: dict[str, Any] | None = None) -> None:
        self.events.append(
            FlowEvent(
                kind=kind,
                invocation=item.id if item else None,
                node=item.node if item else None,
                path=item.path if item else (),
                attempt=item.attempt if item else None,
                cause=item.cause if item else None,
                data=data or {},
            )
        )

    def path(self, node: str) -> tuple[int, ...]:
        """The current iteration path of ``node``."""
        path: list[int] = []
        for loop in self.s.chains[node]:
            path.append(self.iterations.get(_key(loop, tuple(path)), 0))
        return tuple(path)

    def get(self, node: str, path: tuple[int, ...]) -> Invocation | None:
        position = self.latest.get((node, path))
        return None if position is None else self.invocations[position]

    def find(self, invocation_id: str) -> int:
        try:
            return self.by_id[invocation_id]
        except KeyError:
            raise FlowError("flow.unknown_invocation", f"Unknown invocation {invocation_id!r}") from None

    def add(self, item: Invocation) -> Invocation:
        self.invocations.append(item)
        self.latest[(item.node, item.path)] = len(self.invocations) - 1
        self.by_id[item.id] = len(self.invocations) - 1
        if item.final:
            self.event("invocation_settled", item, {"status": item.status, "error": item.error})
        return item

    def put(self, position: int, item: Invocation) -> Invocation:
        self.invocations[position] = item
        return item

    def settle(self, position: int, status: InvocationStatus, error: str | None) -> None:
        item = self.put(
            position, self.invocations[position].model_copy(update={"status": status, "error": error})
        )
        self.event("invocation_settled", item, {"status": status, "error": error})

    def active(self, edge: Edge, source: Invocation) -> bool:
        if source.status != "succeeded":
            return False
        return self.s.nodes[edge.source].ports is None or edge.port in (source.ports or ())

    # ---- inputs ----------------------------------------------------------------------

    def record(self, invocation_id: str, result: Result) -> None:
        position = self.find(invocation_id)
        item = self.invocations[position]
        if item.status != "running":
            raise FlowError("flow.not_running", f"Invocation {invocation_id!r} is {item.status}, not running")
        node = self.s.nodes[item.node]
        ports: tuple[str, ...] | None = None
        if result.status == "succeeded":
            if node.ports is None:
                if result.ports:
                    raise FlowError(
                        "flow.result_ports", f"Node {node.id!r} does not branch, but ports were reported"
                    )
            else:
                if result.ports is None or any(port not in node.ports for port in result.ports):
                    raise FlowError(
                        "flow.result_ports",
                        f"Node {node.id!r} must report a subset of its ports {list(node.ports)}",
                    )
                ports = tuple(dict.fromkeys(result.ports))
        item = self.put(
            position, item.model_copy(update={"status": result.status, "ports": ports, "error": result.error})
        )
        self.event("invocation_settled", item, {"status": item.status, "error": item.error, "ports": ports})
        if result.stop is not None:
            self.halt(result.stop)
        if item.status == "succeeded":
            self.follow_loop(item)

    def follow_loop(self, item: Invocation) -> None:
        loop_id = next((key for key, info in self.s.loops.items() if info.source == item.node), None)
        if loop_id is None or self.stop is not None:
            return
        info = self.s.loops[loop_id]
        assert info.edge.loop is not None
        if info.edge.port not in (item.ports or ()):
            return
        outer, index = item.path[:-1], item.path[-1]
        key = _key(loop_id, outer)
        limit = info.edge.loop.max_iterations
        if index + 1 < limit:
            self.iterations[key] = index + 1
            self.looped.append(item.id)
            self.event(
                "loop_iteration", item, {"loop": loop_id, "iteration": index + 1, "max_iterations": limit}
            )
            return
        self.event("limit_reached", item, {"limit": "loop", "loop": loop_id, "max_iterations": limit})
        if info.edge.loop.on_exhausted == "fail":
            self.halt(
                Stop(
                    code="flow.loop_exhausted",
                    outcome="failed",
                    params={"loop": loop_id, "max_iterations": limit},
                )
            )
        else:
            self.exhausted.append(key)

    def halt(self, stop: Stop) -> None:
        if self.stop is not None:
            return
        self.stop = stop
        self.event("flow_stopped", data={"code": stop.code, "outcome": stop.outcome, "params": stop.params})
        for position, item in enumerate(self.invocations):
            if item.status in ("ready", "suspended"):
                self.settle(position, "cancelled", stop.code)

    def wake(self, invocation_id: str | None) -> None:
        if invocation_id is not None:
            position = self.find(invocation_id)
            if self.invocations[position].status != "suspended":
                raise FlowError("flow.not_suspended", f"Invocation {invocation_id!r} is not suspended")
            positions = [position]
        else:
            positions = [index for index, item in enumerate(self.invocations) if item.status == "suspended"]
        for position in positions:
            item = self.put(position, self.invocations[position].model_copy(update={"status": "ready"}))
            self.event("invocation_resumed", item)

    # ---- the step --------------------------------------------------------------------

    def run(self, now: float | None) -> Step:
        limit = self.flow.limits.max_duration_s
        if (
            limit is not None
            and now is not None
            and self.started_at is not None
            and now - self.started_at >= limit
        ):
            if self.stop is None:
                self.event("limit_reached", data={"limit": "max_duration_s", "max_duration_s": limit})
            self.halt(Stop(code="flow.time_limit", outcome="failed", params={"max_duration_s": limit}))
        self.resolve()
        if self.dispatch():
            self.resolve()
        busy = any(item.status in ("ready", "running", "suspended") for item in self.invocations)
        if not busy and self.unresolved():  # pragma: no cover - the resolution rules leave no node behind
            self.halt(Stop(code="flow.stalled", outcome="failed"))
            self.resolve()
        state = FlowState(
            contract=self.contract,
            invocations=tuple(self.invocations),
            iterations=self.iterations,
            looped=tuple(self.looped),
            exhausted=tuple(self.exhausted),
            stop=self.stop,
            dispatched=self.dispatched,
            started_at=self.started_at,
        )
        return Step(state=state, actions=tuple(self.actions), events=tuple(self.events))

    def unresolved(self) -> list[str]:
        return [name for name in self.s.order if self.get(name, self.path(name)) is None]

    def resolve(self) -> None:
        """Create the invocation of every node whose predecessors decide what happens to it."""
        kinds: dict[str, tuple[_Kind, Invocation | None]] = {}
        for name in self.s.order:
            path = self.path(name)
            item = self.get(name, path)
            if item is None:
                item = self.decide(name, path, kinds)
            if item is None:
                continue  # decide() stored the kind
            if item.final:
                kinds[name] = ("final", item)
            elif item.status == "suspended":
                kinds[name] = ("paused", item)
            else:
                kinds[name] = ("pending", item)

    def region(self, loop: str, kinds: dict[str, tuple[_Kind, Invocation | None]]) -> _Kind:
        found = {kinds[member][0] for member in self.s.loops[loop].body}
        if "pending" in found:
            return "pending"
        return "paused" if "paused" in found else "final"

    def decide(
        self, name: str, path: tuple[int, ...], kinds: dict[str, tuple[_Kind, Invocation | None]]
    ) -> Invocation | None:
        node = self.s.nodes[name]
        edges = self.s.incoming[name]
        loop_id = self.s.head_of.get(name)
        cause: str | None = None
        extra: list[Arrival] = []
        if loop_id is not None and path[-1] > 0:
            previous = self.get(self.s.loops[loop_id].source, (*path[:-1], path[-1] - 1))
            assert previous is not None
            cause = previous.id
            extra.append(Arrival(edge=loop_id, source=previous.id))
        seen: list[tuple[Edge, _Kind, Invocation | None]] = []
        for edge in edges:
            kind, source = kinds[edge.source]
            left = self.s.exits.get(edge.id)
            if left is not None and kind == "final":
                kind = self.region(left, kinds)
                if kind != "final":
                    source = None
            seen.append((edge, kind, source))
        active = [
            Arrival(edge=edge.id, source=source.id)
            for edge, kind, source in seen
            if kind == "final" and source is not None and self.active(edge, source)
        ]
        pending = any(kind == "pending" for _, kind, _ in seen)
        ident = _invocation_id(name, path, 1)
        if self.stop is not None:
            if pending:
                kinds[name] = ("pending", None)
                return None
            return self.add(
                Invocation(id=ident, node=name, path=path, status="cancelled", error=self.stop.code)
            )
        if cause is not None:
            return self.add(
                Invocation(id=ident, node=name, path=path, arrivals=(*active, *extra), cause=cause)
            )
        if node.join == "any" and active:
            return self.add(Invocation(id=ident, node=name, path=path, arrivals=tuple(active)))
        if pending:
            kinds[name] = ("pending", None)
            return None
        statuses = {source.status for _, kind, source in seen if kind == "final" and source is not None}
        if statuses & _NOT_OK:
            return self.add(
                Invocation(id=ident, node=name, path=path, status="blocked", error="flow.blocked")
            )
        if any(kind == "paused" for _, kind, _ in seen):
            kinds[name] = ("paused", None)
            return None
        if edges and not active:
            return self.add(Invocation(id=ident, node=name, path=path, status="skipped"))
        return self.add(Invocation(id=ident, node=name, path=path, arrivals=tuple(active)))

    def dispatch(self) -> bool:
        """Hand ready invocations to the driver within the limits; True when the flow stopped here."""
        limits = self.flow.limits
        running = sum(1 for item in self.invocations if item.status == "running")
        for position, item in enumerate(self.invocations):
            if item.status != "ready":  # a stop leaves nothing ready (see halt)
                continue
            if limits.max_parallel is not None and running >= limits.max_parallel:
                break
            first = item.dispatches == 0
            if first and limits.max_invocations is not None and self.dispatched >= limits.max_invocations:
                self.event(
                    "limit_reached",
                    data={"limit": "max_invocations", "max_invocations": limits.max_invocations},
                )
                self.halt(
                    Stop(
                        code="flow.invocation_limit",
                        outcome="failed",
                        params={"max_invocations": limits.max_invocations},
                    )
                )
                return True
            item = self.put(
                position, item.model_copy(update={"status": "running", "dispatches": item.dispatches + 1})
            )
            if first:
                self.dispatched += 1
            running += 1
            self.actions.append(RunNode(invocation=item))
            self.event("invocation_started", item, {"type": self.s.nodes[item.node].type})
        return False
