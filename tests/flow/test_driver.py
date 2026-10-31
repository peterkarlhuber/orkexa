"""The asyncio reference driver."""

from __future__ import annotations

import asyncio
import itertools

import pytest

from orkexa.flow import (
    Edge,
    Flow,
    FlowError,
    FlowEvent,
    Limits,
    Loop,
    Node,
    Result,
    Stop,
    TaskContext,
    TaskResult,
    run_flow_sync,
    start,
)
from orkexa.flow.conformance import CASES, check, scripted


def chain(*names: str, node_type: str = "task", **limits: object) -> Flow:
    return Flow(
        nodes=tuple(Node(id=name, type=node_type) for name in names),
        edges=tuple(Edge(id=f"{a}-{b}", source=a, target=b) for a, b in itertools.pairwise(names)),
        limits=Limits(**limits),  # type: ignore[arg-type]
    )


def test_outputs_flow_along_active_connections() -> None:
    async def task(context: TaskContext) -> TaskResult:
        return TaskResult(output=[context.node.id, dict(context.inputs)])

    events: list[FlowEvent] = []
    run = run_flow_sync(chain("a", "b"), {"task": task}, on_event=events.append)
    assert run.outcome == "succeeded"
    assert run.outputs == {"a//1": ["a", {}], "b//1": ["b", {"a": ["a", {}]}]}
    assert events == run.events
    assert events[0].kind == "flow_started"


def test_conformance_cases_through_the_driver() -> None:
    for case in CASES:
        result = scripted(case)
        counts: dict[str, int] = {}

        async def handler(context: TaskContext, result=result, counts=counts) -> Result:  # type: ignore[no-untyped-def]
            counts[context.node.id] = counts.get(context.node.id, 0) + 1
            return result(context.node.id, counts[context.node.id])  # type: ignore[no-any-return]

        handlers = {node.type: handler for node in case.flow.nodes}
        run = run_flow_sync(case.flow, handlers)
        resumes = case.resumes
        while resumes and run.outcome == "suspended":
            resumes -= 1
            run = run_flow_sync(case.flow, handlers, state=run.state, outputs=run.outputs, resume=True)
        assert check(case, run.state) == [], case.name


def test_loop_head_sees_the_previous_review() -> None:
    flow = Flow(
        nodes=(Node(id="draft", type="write"), Node(id="review", type="review", ports=("pass", "again"))),
        edges=(
            Edge(id="dr", source="draft", target="review"),
            Edge(id="back", source="review", target="draft", port="again", loop=Loop(max_iterations=3)),
        ),
    )
    seen: list[dict[str, object]] = []

    async def write(context: TaskContext) -> TaskResult:
        seen.append(dict(context.inputs))
        return TaskResult(output=f"draft {context.invocation.path[0]}")

    async def review(context: TaskContext) -> TaskResult:
        good = context.invocation.path[0] == 1
        return TaskResult(output="fine" if good else "more detail", ports=("pass",) if good else ("again",))

    run = run_flow_sync(flow, {"write": write, "review": review})
    assert run.outcome == "succeeded"
    assert seen == [{}, {"review": "more detail"}]


def test_handler_errors_and_missing_handlers_fail_the_invocation() -> None:
    async def boom(context: TaskContext) -> None:
        raise RuntimeError("no")

    flow = Flow(nodes=(Node(id="a", type="boom"), Node(id="b", type="unknown")))
    run = run_flow_sync(flow, {"boom": boom})
    assert {item.id: item.error for item in run.state.invocations} == {
        "a//1": "flow.handler_error",
        "b//1": "flow.no_handler",
    }
    assert run.outcome == "failed"


def test_a_stop_cancels_running_handlers() -> None:
    started = asyncio.Event()

    async def slow(context: TaskContext) -> None:
        started.set()
        await asyncio.sleep(60)

    async def stopper(context: TaskContext) -> TaskResult:
        await started.wait()
        return TaskResult(stop=Stop(code="run.approval_rejected"))

    flow = Flow(nodes=(Node(id="slow", type="slow"), Node(id="stop", type="stopper")))
    run = run_flow_sync(flow, {"slow": slow, "stopper": stopper})
    assert run.outcome == "cancelled"
    assert run.state.invocation("slow//1").status == "cancelled"
    assert run.state.invocation("slow//1").error == "run.approval_rejected"


def test_time_limit_cancels_with_an_injected_clock() -> None:
    now = [0.0]

    async def slow(context: TaskContext) -> None:
        now[0] = 100.0
        await asyncio.sleep(60)

    run = run_flow_sync(
        chain("a", "b", node_type="slow", max_duration_s=0.05), {"slow": slow}, clock=lambda: now[0]
    )
    assert run.outcome == "failed"
    assert run.state.stop is not None and run.state.stop.code == "flow.time_limit"
    assert {item.id: item.status for item in run.state.invocations} == {
        "a//1": "cancelled",
        "b//1": "cancelled",
    }


def test_resume_one_invocation_and_interrupted_states() -> None:
    decided: set[str] = set()

    async def ask(context: TaskContext) -> Result:
        return Result(status="succeeded") if context.node.id in decided else Result(status="suspended")

    flow = Flow(nodes=(Node(id="x", type="ask"), Node(id="y", type="ask")))
    run = run_flow_sync(flow, {"ask": ask})
    assert run.outcome == "suspended"
    decided.add("x")
    run = run_flow_sync(flow, {"ask": ask}, state=run.state, resume="x//1")
    assert {item.id: item.status for item in run.state.invocations} == {
        "x//1": "succeeded",
        "y//1": "suspended",
    }
    again = run_flow_sync(flow, {"ask": ask}, state=run.state)
    assert again.state == run.state

    running = start(chain("a")).state
    with pytest.raises(FlowError) as caught:
        run_flow_sync(chain("a"), {}, state=running)
    assert caught.value.code == "flow.interrupted"


def test_a_handler_may_return_nothing() -> None:
    async def quiet(context: TaskContext) -> None:
        return None

    run = run_flow_sync(chain("a", node_type="quiet"), {"quiet": quiet})
    assert run.outcome == "succeeded"
    assert run.outputs == {}
