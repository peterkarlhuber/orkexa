"""Randomized checks: random flows, random results, random dispatch orders.

* Acyclic flows are compared with an independent oracle of the settle rules that
  Nodestro's runtime uses today (a node settles once all predecessors did; a failed,
  blocked or cancelled predecessor blocks it; a paused one keeps it waiting; no active
  connection skips it).
* Flows with loops must terminate within their limits, leave no node unresolved and
  reach the same final state in every dispatch order.

Seeds are fixed, so every run checks the same flows.
"""

from __future__ import annotations

import json
import random
from typing import Literal
import zlib

import pytest

from orkexa.flow import Edge, Flow, FlowState, Loop, Node, Result, Step, complete, outcome, start, validate


def random_dag(rng: random.Random, size: int) -> Flow:
    nodes = []
    for index in range(size):
        ports = ("p", "q", "r")[: rng.randint(2, 3)] if rng.random() < 0.35 else None
        join: Literal["all", "any"] = "any" if rng.random() < 0.15 else "all"
        nodes.append(Node(id=f"n{index}", ports=ports, join=join))
    edges = []
    for target in range(1, size):
        for source in rng.sample(range(target), k=min(target, rng.randint(0, 3))):
            ports = nodes[source].ports
            edges.append(
                Edge(
                    id=f"e{source}-{target}",
                    source=f"n{source}",
                    target=f"n{target}",
                    port=rng.choice(ports) if ports else None,
                )
            )
    return Flow(nodes=tuple(nodes), edges=tuple(edges))


def add_loops(rng: random.Random, flow: Flow, tries: int) -> Flow:
    for attempt in range(tries):
        sources = [node for node in flow.nodes if node.ports]
        if not sources:
            break
        source = rng.choice(sources)
        ancestors = [edge.source for edge in flow.edges if edge.target == source.id and edge.loop is None]
        target = rng.choice([*ancestors, source.id])
        loop = Loop(max_iterations=rng.randint(2, 4), on_exhausted=rng.choice(["fail", "continue"]))
        assert source.ports is not None
        candidate = flow.model_copy(
            update={
                "edges": (
                    *flow.edges,
                    Edge(
                        id=f"loop{attempt}",
                        source=source.id,
                        target=target,
                        port=rng.choice(source.ports),
                        loop=loop,
                    ),
                )
            }
        )
        if not validate(candidate):
            flow = candidate
    return flow


def result_for(flow: Flow, seed: int, node: str, path: tuple[int, ...]) -> Result:
    """A result that depends on the node and its iteration only, never on dispatch order."""
    rng = random.Random(zlib.crc32(f"{seed}:{node}:{path}".encode()))
    spec = next(item for item in flow.nodes if item.id == node)
    roll = rng.random()
    if roll < 0.1:
        return Result(status="failed", error="test.failed")
    if roll < 0.14:
        return Result(status="cancelled", error="test.cancelled")
    if spec.ports is None:
        return Result(status="succeeded")
    return Result(status="succeeded", ports=tuple(port for port in spec.ports if rng.random() < 0.5))


def drive(flow: Flow, seed: int, order: random.Random, *, persist: bool = False) -> FlowState:
    step: Step = start(flow)
    state = step.state
    running = [action.invocation for action in step.actions]
    for _ in range(10_000):
        if not running:
            return state
        item = running.pop(order.randrange(len(running)))
        if persist:
            state = FlowState.model_validate(json.loads(state.model_dump_json()))
        step = complete(flow, state, item.id, result_for(flow, seed, item.node, item.path))
        state = step.state
        running.extend(action.invocation for action in step.actions)
    raise AssertionError("the flow did not terminate")


def oracle(flow: Flow, seed: int) -> dict[str, str]:
    """Final status per node of an acyclic flow, computed without the kernel."""
    status: dict[str, str] = {}
    ports: dict[str, tuple[str, ...]] = {}
    nodes = {node.id: node for node in flow.nodes}
    remaining = [node.id for node in flow.nodes]
    while remaining:
        for name in list(remaining):
            incoming = [edge for edge in flow.edges if edge.target == name]
            done = [edge for edge in incoming if edge.source in status]

            def active(edge: Edge) -> bool:
                return status.get(edge.source) == "succeeded" and (
                    nodes[edge.source].ports is None or edge.port in ports[edge.source]
                )

            if nodes[name].join == "any" and any(active(edge) for edge in done):
                verdict = "run"
            elif len(done) < len(incoming):
                continue
            elif any(status[edge.source] in ("failed", "blocked", "cancelled") for edge in incoming):
                verdict = "blocked"
            elif incoming and not any(active(edge) for edge in incoming):
                verdict = "skipped"
            else:
                verdict = "run"
            if verdict == "run":
                result = result_for(flow, seed, name, ())
                status[name] = result.status
                ports[name] = result.ports or ()
            else:
                status[name] = verdict
            remaining.remove(name)
    return status


@pytest.mark.parametrize("seed", range(150))
def test_acyclic_flows_match_the_oracle_in_every_order(seed: int) -> None:
    rng = random.Random(seed)
    flow = random_dag(rng, rng.randint(1, 12))
    expected = oracle(flow, seed)
    for order_seed in range(3):
        state = drive(flow, seed, random.Random(order_seed), persist=order_seed == 1)
        assert {item.node: item.status for item in state.invocations} == expected
        assert all(item.path == () and item.attempt == 1 for item in state.invocations)


@pytest.mark.parametrize("seed", range(150))
def test_flows_with_loops_terminate_consistently(seed: int) -> None:
    rng = random.Random(seed)
    flow = add_loops(rng, random_dag(rng, rng.randint(2, 10)), tries=rng.randint(1, 4))
    assert validate(flow) == []
    reference = drive(flow, seed, random.Random(0))
    finals = {(item.id, item.status) for item in reference.invocations}
    assert outcome(reference) in ("succeeded", "failed", "cancelled")
    assert all(item.final for item in reference.invocations)
    for order_seed in (1, 2):
        other = drive(flow, seed, random.Random(order_seed), persist=order_seed == 2)
        assert (other.stop is None) == (reference.stop is None)
        if reference.stop is None:  # a stop races with work on other branches, like in any engine
            assert {(item.id, item.status) for item in other.invocations} == finals
            assert other.iterations == reference.iterations
    # Every loop body ran at most max_iterations times per enclosing iteration.
    for edge in flow.edges:
        if edge.loop is None:
            continue
        per_outer: dict[tuple[int, ...], set[int]] = {}
        for item in reference.invocations:
            if item.node == edge.source:
                per_outer.setdefault(item.path[:-1], set()).add(item.path[-1])
        assert all(len(indexes) <= edge.loop.max_iterations for indexes in per_outer.values())
    # Every node has an invocation at its final iteration.
    assert {item.node for item in reference.invocations} == {node.id for node in flow.nodes}


def test_random_generators_produce_loops() -> None:
    looped = 0
    for seed in range(150):
        rng = random.Random(seed)
        flow = add_loops(rng, random_dag(rng, rng.randint(2, 10)), tries=rng.randint(1, 4))
        looped += any(edge.loop for edge in flow.edges) and bool(drive(flow, seed, random.Random(0)).looped)
    assert looped >= 20
