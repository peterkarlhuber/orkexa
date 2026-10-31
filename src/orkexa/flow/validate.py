"""Static checks of a flow specification and the loop structure the kernel works with."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from ..errors import OrkexaError
from .model import CONTRACT_VERSION, Edge, Flow, FlowIssue, Node

__all__ = ["FlowError", "validate"]


class FlowError(OrkexaError, ValueError):
    """A flow specification, state or driver call the kernel cannot accept.

    ``code`` is stable (``flow.*``); ``issues`` lists every problem of an invalid flow.
    """

    def __init__(self, code: str, message: str, issues: Iterable[FlowIssue] = ()) -> None:
        super().__init__(message)
        self.code = code
        self.issues = tuple(issues)


@dataclass(frozen=True)
class _LoopInfo:
    edge: Edge
    head: str
    source: str
    body: frozenset[str]


@dataclass
class _Structure:
    """What the kernel derives from a valid flow once (internal)."""

    nodes: dict[str, Node]
    order: list[str]
    incoming: dict[str, list[Edge]]
    outgoing: dict[str, list[Edge]]
    loops: dict[str, _LoopInfo]  # by loop edge id
    head_of: dict[str, str] = field(default_factory=dict)  # head node -> loop edge id
    chains: dict[str, tuple[str, ...]] = field(
        default_factory=dict
    )  # node -> loops around it, outermost first
    exits: dict[str, str] = field(default_factory=dict)  # edge id -> outermost loop the edge leaves


def validate(flow: Flow) -> list[FlowIssue]:
    """Every problem of ``flow``; an empty list means the kernel can run it.

    Codes: ``flow.contract``, ``flow.duplicate_id``, ``flow.edge_dangling``, ``flow.self_loop``,
    ``flow.port_unknown``, ``flow.port_missing``, ``flow.cycle``, ``flow.loop_unguarded``,
    ``flow.loop_target``, ``flow.loop_limit``, ``flow.loop_overlap``, ``flow.loop_head_shared``,
    ``flow.loop_source_shared``.
    """
    issues: list[FlowIssue] = []
    if flow.contract.split(".")[0] != CONTRACT_VERSION.split(".")[0]:
        issues.append(
            FlowIssue(
                code="flow.contract",
                message=f"Flow contract {flow.contract} is not supported (expected {CONTRACT_VERSION})",
            )
        )
    nodes: dict[str, Node] = {}
    for node in flow.nodes:
        if node.id in nodes:
            issues.append(
                FlowIssue(
                    code="flow.duplicate_id", message=f"Node id {node.id!r} is used twice", node=node.id
                )
            )
        nodes[node.id] = node
    edge_ids: set[str] = set()
    usable: list[Edge] = []
    for edge in flow.edges:
        if edge.id in edge_ids:
            issues.append(
                FlowIssue(
                    code="flow.duplicate_id", message=f"Edge id {edge.id!r} is used twice", edge=edge.id
                )
            )
            continue
        edge_ids.add(edge.id)
        missing = [name for name in (edge.source, edge.target) if name not in nodes]
        if missing:
            issues.append(
                FlowIssue(
                    code="flow.edge_dangling",
                    message=f"Edge {edge.id!r} refers to the unknown node {missing[0]!r}",
                    edge=edge.id,
                )
            )
            continue
        if edge.source == edge.target and edge.loop is None:
            issues.append(
                FlowIssue(
                    code="flow.self_loop", message=f"Edge {edge.id!r} connects a node to itself", edge=edge.id
                )
            )
            continue
        ports = nodes[edge.source].ports
        if ports is None and edge.port is not None:
            issues.append(
                FlowIssue(
                    code="flow.port_unknown",
                    message=f"Edge {edge.id!r} uses port {edge.port!r}, but {edge.source!r} does not branch",
                    edge=edge.id,
                )
            )
            continue
        if ports is not None and edge.port is None:
            issues.append(
                FlowIssue(
                    code="flow.port_missing",
                    message=f"Edge {edge.id!r} leaves the branching node {edge.source!r} without a port",
                    edge=edge.id,
                )
            )
            continue
        if ports is not None and edge.port not in ports:
            issues.append(
                FlowIssue(
                    code="flow.port_unknown",
                    message=f"Node {edge.source!r} has no port {edge.port!r}",
                    edge=edge.id,
                )
            )
            continue
        usable.append(edge)
    forward = [edge for edge in usable if edge.loop is None]
    order = _topological(list(nodes), forward)
    if order is None:
        issues.append(
            FlowIssue(code="flow.cycle", message="The connections form a cycle without a loop edge")
        )
        return issues
    issues.extend(_loop_issues(flow, nodes, forward, [edge for edge in usable if edge.loop is not None]))
    return issues


def _topological(names: list[str], edges: list[Edge]) -> list[str] | None:
    """Kahn's algorithm, stable in specification order; ``None`` on a cycle."""
    indegree = dict.fromkeys(names, 0)
    successors: dict[str, list[str]] = {name: [] for name in names}
    for edge in edges:
        indegree[edge.target] += 1
        successors[edge.source].append(edge.target)
    ready = [name for name in names if indegree[name] == 0]
    order: list[str] = []
    while ready:
        name = ready.pop(0)
        order.append(name)
        for target in successors[name]:
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    return order if len(order) == len(names) else None


def _reach(start: str, adjacency: dict[str, list[str]]) -> set[str]:
    seen = {start}
    todo = [start]
    while todo:
        for nxt in adjacency.get(todo.pop(), ()):
            if nxt not in seen:
                seen.add(nxt)
                todo.append(nxt)
    return seen


def _bodies(nodes: dict[str, Node], forward: list[Edge], loops: list[Edge]) -> dict[str, frozenset[str]]:
    down: dict[str, list[str]] = {name: [] for name in nodes}
    up: dict[str, list[str]] = {name: [] for name in nodes}
    for edge in forward:
        down[edge.source].append(edge.target)
        up[edge.target].append(edge.source)
    return {edge.id: frozenset(_reach(edge.target, down) & _reach(edge.source, up)) for edge in loops}


def _loop_issues(
    flow: Flow, nodes: dict[str, Node], forward: list[Edge], loops: list[Edge]
) -> list[FlowIssue]:
    issues: list[FlowIssue] = []
    bodies = _bodies(nodes, forward, loops)
    heads: dict[str, str] = {}
    sources: dict[str, str] = {}
    valid: list[Edge] = []
    for edge in loops:
        assert edge.loop is not None
        if nodes[edge.source].ports is None:
            issues.append(
                FlowIssue(
                    code="flow.loop_unguarded",
                    message=f"Loop edge {edge.id!r} must leave a branching node through a port",
                    edge=edge.id,
                )
            )
            continue
        if edge.source not in bodies[edge.id]:
            issues.append(
                FlowIssue(
                    code="flow.loop_target",
                    message=f"Loop edge {edge.id!r} must lead back to {edge.source!r} or a node before it",
                    edge=edge.id,
                )
            )
            continue
        cap = flow.limits.max_iterations
        if cap is not None and edge.loop.max_iterations > cap:
            issues.append(
                FlowIssue(
                    code="flow.loop_limit",
                    message=(
                        f"Loop edge {edge.id!r} allows {edge.loop.max_iterations} iterations, "
                        f"the flow at most {cap}"
                    ),
                    edge=edge.id,
                )
            )
        if edge.target in heads:
            issues.append(
                FlowIssue(
                    code="flow.loop_head_shared",
                    message=f"Node {edge.target!r} starts more than one loop",
                    edge=edge.id,
                    node=edge.target,
                )
            )
            continue
        if edge.source in sources:
            issues.append(
                FlowIssue(
                    code="flow.loop_source_shared",
                    message=f"Node {edge.source!r} closes more than one loop",
                    edge=edge.id,
                    node=edge.source,
                )
            )
            continue
        heads[edge.target] = edge.id
        sources[edge.source] = edge.id
        valid.append(edge)
    for index, first in enumerate(valid):
        for second in valid[index + 1 :]:
            one, two = bodies[first.id], bodies[second.id]
            if one & two and not (one < two or two < one):
                issues.append(
                    FlowIssue(
                        code="flow.loop_overlap",
                        message=f"The loops of {first.id!r} and {second.id!r} overlap without nesting",
                        edge=second.id,
                    )
                )
    return issues


def structure(flow: Flow) -> _Structure:
    """The derived structure of a flow; raises :class:`FlowError` when it is invalid."""
    issues = validate(flow)
    if issues:
        raise FlowError("flow.invalid", f"The flow is invalid: {issues[0].message}", issues)
    nodes = {node.id: node for node in flow.nodes}
    forward = [edge for edge in flow.edges if edge.loop is None]
    loop_edges = [edge for edge in flow.edges if edge.loop is not None]
    order = _topological(list(nodes), forward)
    assert order is not None
    incoming: dict[str, list[Edge]] = {name: [] for name in nodes}
    outgoing: dict[str, list[Edge]] = {name: [] for name in nodes}
    for edge in forward:
        incoming[edge.target].append(edge)
        outgoing[edge.source].append(edge)
    bodies = _bodies(nodes, forward, loop_edges)
    loops = {
        edge.id: _LoopInfo(edge=edge, head=edge.target, source=edge.source, body=bodies[edge.id])
        for edge in loop_edges
    }
    result = _Structure(nodes=nodes, order=order, incoming=incoming, outgoing=outgoing, loops=loops)
    result.head_of = {info.head: key for key, info in loops.items()}
    for name in nodes:
        around = [info for info in loops.values() if name in info.body]
        around.sort(key=lambda info: (-len(info.body), info.edge.id))
        result.chains[name] = tuple(info.edge.id for info in around)
    # A node after a loop waits until the loop is over: it depends on every node of the
    # loop's body, so the kernel resolves the whole body before it (one pass suffices).
    virtual = list(forward)
    for edge in forward:
        source, target = result.chains[edge.source], result.chains[edge.target]
        shared = 0
        while shared < min(len(source), len(target)) and source[shared] == target[shared]:
            shared += 1
        if shared < len(source):
            left = source[shared]
            result.exits[edge.id] = left
            virtual.extend(
                Edge(id=f"{edge.id}~{member}", source=member, target=edge.target)
                for member in sorted(loops[left].body)
                if member != edge.source
            )
    ordered = _topological(list(nodes), virtual)
    if ordered is None:  # pragma: no cover - excluded by the loop checks above
        raise FlowError("flow.invalid", "The loop structure of the flow is inconsistent")
    result.order = ordered
    return result
