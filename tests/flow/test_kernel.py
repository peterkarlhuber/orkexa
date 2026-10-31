"""The flow kernel: conformance cases, state handling, limits and driver errors."""

from __future__ import annotations

from decimal import Decimal
import itertools
import json

import pytest

from orkexa.flow import (
    CONTRACT_VERSION,
    EVENT_SCHEMA,
    Edge,
    Flow,
    FlowError,
    FlowState,
    Limits,
    Loop,
    Node,
    Result,
    Step,
    Stop,
    advance,
    cancel,
    complete,
    outcome,
    resume,
    start,
)
from orkexa.flow.conformance import CASES, ConformanceCase, check, run_case

OK = Result(status="succeeded")


def chain(*names: str, **limits: object) -> Flow:
    return Flow(
        nodes=tuple(Node(id=name) for name in names),
        edges=tuple(Edge(id=f"{a}-{b}", source=a, target=b) for a, b in itertools.pairwise(names)),
        limits=Limits(**limits),  # type: ignore[arg-type]
    )


def statuses(step: Step) -> dict[str, str]:
    return {item.id: item.status for item in step.state.invocations}


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_conformance(case: ConformanceCase) -> None:
    assert check(case, run_case(case)) == []


def test_case_names_are_unique() -> None:
    assert len({case.name for case in CASES}) == len(CASES)


def test_check_reports_differences() -> None:
    case = CASES[0]
    wrong = ConformanceCase(
        name="x", flow=case.flow, statuses={"a//1": "failed"}, outcome="failed", stop="nope"
    )
    problems = check(wrong, run_case(case))
    assert problems == [
        "a//1: expected failed, got succeeded",
        "b//1: unexpected invocation (succeeded)",
        "c//1: unexpected invocation (succeeded)",
        "outcome: expected failed, got succeeded",
        "stop: expected nope, got None",
    ]


def test_start_dispatches_sources_and_emits_events() -> None:
    step = start(Flow(nodes=(Node(id="a"), Node(id="b"))), now=10.0)
    assert [action.invocation.id for action in step.actions] == ["a//1", "b//1"]
    assert step.state.started_at == 10.0
    assert step.state.dispatched == 2
    assert [event.kind for event in step.events] == [
        "flow_started",
        "invocation_started",
        "invocation_started",
    ]
    event = step.events[1].model_dump(by_alias=True)
    assert event["schema"] == EVENT_SCHEMA
    assert event["node"] == "a"
    assert step.state.contract == CONTRACT_VERSION
    assert outcome(step.state) == "running"


def test_empty_flow_succeeds_at_once() -> None:
    assert outcome(start(Flow()).state) == "succeeded"


def test_state_round_trips_through_json_and_continues_identically() -> None:
    flow = CASES[[case.name for case in CASES].index("nested-loops")].flow
    first = start(flow)
    stored = FlowState.model_validate(json.loads(first.state.model_dump_json()))
    assert stored == first.state
    item = first.actions[0].invocation
    assert complete(flow, stored, item.id, OK) == complete(flow, first.state, item.id, OK)


def test_unknown_major_contract_is_refused() -> None:
    with pytest.raises(ValueError, match="not supported"):
        FlowState(contract="9.0")


def test_state_of_another_flow_is_refused() -> None:
    state = start(chain("a")).state
    with pytest.raises(FlowError) as caught:
        advance(chain("x"), state)
    assert caught.value.code == "flow.state_mismatch"


def test_complete_errors() -> None:
    flow = Flow(nodes=(Node(id="a"), Node(id="b", ports=("p", "q"))))
    step = start(flow)
    with pytest.raises(FlowError) as caught:
        complete(flow, step.state, "nope", OK)
    assert caught.value.code == "flow.unknown_invocation"
    with pytest.raises(FlowError) as caught:
        complete(flow, step.state, "a//1", Result(status="succeeded", ports=("p",)))
    assert caught.value.code == "flow.result_ports"
    for ports in (None, ("z",)):
        with pytest.raises(FlowError) as caught:
            complete(flow, step.state, "b//1", Result(status="succeeded", ports=ports))
        assert caught.value.code == "flow.result_ports"
    done = complete(flow, step.state, "a//1", OK)
    with pytest.raises(FlowError) as caught:
        complete(flow, done.state, "a//1", OK)
    assert caught.value.code == "flow.not_running"


def test_ports_are_deduplicated_and_failed_branch_needs_no_ports() -> None:
    flow = Flow(nodes=(Node(id="b", ports=("p", "q")),))
    step = complete(flow, start(flow).state, "b//1", Result(status="succeeded", ports=("q", "q")))
    assert step.state.invocation("b//1").ports == ("q",)
    failed = complete(flow, start(flow).state, "b//1", Result(status="failed", error="x"))
    assert failed.state.invocation("b//1").error == "x"


def test_cancel_stops_waiting_work_but_leaves_running_to_the_driver() -> None:
    flow = Flow(
        nodes=(Node(id="a"), Node(id="ask"), Node(id="b"), Node(id="c")),
        edges=(Edge(id="1", source="a", target="b"), Edge(id="2", source="ask", target="c")),
    )
    step = start(flow)
    step = complete(flow, step.state, "ask//1", Result(status="suspended"))
    assert outcome(step.state) == "running"
    step = cancel(flow, step.state, code="run.cancelled", params={"by": "test"})
    assert statuses(step) == {"a//1": "running", "ask//1": "cancelled", "c//1": "cancelled"}
    assert step.state.stop == Stop(code="run.cancelled", params={"by": "test"})
    assert outcome(step.state) == "running"
    step = complete(flow, step.state, "a//1", Result(status="cancelled", error="run.cancelled"))
    assert statuses(step) == {
        "a//1": "cancelled",
        "ask//1": "cancelled",
        "b//1": "cancelled",
        "c//1": "cancelled",
    }
    assert outcome(step.state) == "cancelled"
    again = cancel(flow, step.state, code="other")
    assert again.state.stop == step.state.stop
    assert again.events == ()


def test_resume_one_or_all() -> None:
    flow = Flow(nodes=(Node(id="x"), Node(id="y")))
    step = start(flow)
    for name in ("x//1", "y//1"):
        step = complete(flow, step.state, name, Result(status="suspended"))
    assert outcome(step.state) == "suspended"
    with pytest.raises(FlowError) as caught:
        resume(flow, step.state, "nope")
    assert caught.value.code == "flow.unknown_invocation"
    one = resume(flow, step.state, "x//1")
    assert [action.invocation.id for action in one.actions] == ["x//1"]
    assert one.actions[0].invocation.dispatches == 2
    assert one.state.dispatched == 2  # a second dispatch of the same invocation does not count again
    with pytest.raises(FlowError) as caught:
        resume(flow, one.state, "x//1")
    assert caught.value.code == "flow.not_suspended"
    both = resume(flow, step.state)
    assert [event.kind for event in both.events][:2] == ["invocation_resumed", "invocation_resumed"]
    assert len(both.actions) == 2


def test_blocked_wins_over_a_suspended_predecessor() -> None:
    flow = Flow(
        nodes=(Node(id="ask"), Node(id="bad"), Node(id="j")),
        edges=(Edge(id="1", source="ask", target="j"), Edge(id="2", source="bad", target="j")),
    )
    step = start(flow)
    step = complete(flow, step.state, "ask//1", Result(status="suspended"))
    step = complete(flow, step.state, "bad//1", Result(status="failed"))
    assert statuses(step)["j//1"] == "blocked"
    assert outcome(step.state) == "suspended"


def test_waiting_behind_a_suspended_node_creates_nothing() -> None:
    flow = chain("ask", "b", "c")
    step = complete(flow, start(flow).state, "ask//1", Result(status="suspended"))
    assert statuses(step) == {"ask//1": "suspended"}
    step = resume(flow, step.state)
    step = complete(flow, step.state, "ask//1", OK)
    assert [action.invocation.id for action in step.actions] == ["b//1"]


def test_max_parallel_throttles_dispatch() -> None:
    flow = Flow(nodes=(Node(id="a"), Node(id="b"), Node(id="c")), limits=Limits(max_parallel=2))
    step = start(flow)
    assert [action.invocation.id for action in step.actions] == ["a//1", "b//1"]
    assert statuses(step)["c//1"] == "ready"
    step = complete(flow, step.state, "b//1", OK)
    assert [action.invocation.id for action in step.actions] == ["c//1"]


def test_time_limit() -> None:
    flow = chain("a", "b", max_duration_s=5)
    step = start(flow, now=100.0)
    assert advance(flow, step.state, now=104.0).state.stop is None
    assert advance(flow, step.state).state.stop is None  # no clock, no check
    late = complete(flow, step.state, "a//1", OK, now=105.0)
    assert late.state.stop is not None and late.state.stop.code == "flow.time_limit"
    assert statuses(late) == {"a//1": "succeeded", "b//1": "cancelled"}
    assert outcome(late.state) == "failed"
    kinds = [event.kind for event in late.events]
    assert kinds.index("limit_reached") < kinds.index("flow_stopped")


def test_loop_events_and_counters() -> None:
    case = next(case for case in CASES if case.name == "loop-until-pass")
    state = run_case(case)
    assert state.iterations == {"retry@": 2}
    assert state.looped == ("review/0/1", "review/1/1")
    redo = state.invocation("draft/1/1")
    assert redo.cause == "review/0/1"
    assert [arrival.edge for arrival in redo.arrivals] == ["retry"]


def test_loop_exhausted_continue_is_recorded() -> None:
    case = next(case for case in CASES if case.name == "loop-exhausted-continues")
    assert run_case(case).exhausted == ("retry@",)


def test_nested_loop_counters_are_scoped_to_the_outer_iteration() -> None:
    case = next(case for case in CASES if case.name == "nested-loops")
    assert run_case(case).iterations == {"fix@0": 1, "replan@": 1}


def test_loop_back_after_a_stop_is_ignored() -> None:
    flow = Flow(
        nodes=(Node(id="w"), Node(id="r", ports=("ok", "again")), Node(id="other")),
        edges=(
            Edge(id="wr", source="w", target="r"),
            Edge(id="back", source="r", target="w", port="again", loop=Loop(max_iterations=3)),
        ),
    )
    step = start(flow)
    step = complete(flow, step.state, "w/0/1", OK)
    step = complete(
        flow, step.state, "other//1", Result(status="failed", stop=Stop(code="x", outcome="failed"))
    )
    step = complete(flow, step.state, "r/0/1", Result(status="succeeded", ports=("again",)))
    assert step.state.iterations == {}
    assert outcome(step.state) == "failed"


def test_loop_body_exit_waits_for_the_whole_region() -> None:
    # "side" leaves the body from the head; it must see the last iteration only.
    flow = Flow(
        nodes=(Node(id="h"), Node(id="r", ports=("ok", "again")), Node(id="side")),
        edges=(
            Edge(id="hr", source="h", target="r"),
            Edge(id="hs", source="h", target="side"),
            Edge(id="back", source="r", target="h", port="again", loop=Loop(max_iterations=3)),
        ),
    )
    step = start(flow)
    step = complete(flow, step.state, "h/0/1", OK)
    assert [action.invocation.id for action in step.actions] == ["r/0/1"]
    step = complete(flow, step.state, "r/0/1", Result(status="succeeded", ports=("again",)))
    step = complete(flow, step.state, "h/1/1", OK)
    step = complete(flow, step.state, "r/1/1", Result(status="succeeded", ports=("ok",)))
    side = step.state.invocation("side//1")
    assert [arrival.source for arrival in side.arrivals] == ["h/1/1"]
    assert outcome(complete(flow, step.state, "side//1", OK).state) == "succeeded"


def test_limits_within_a_parent() -> None:
    child = Limits(max_invocations=50, max_parallel=None, budget_usd=Decimal("2"))
    parent = Limits(max_invocations=10, max_parallel=4, max_depth=2)
    assert child.within(parent) == Limits(
        max_invocations=10, max_parallel=4, max_depth=2, budget_usd=Decimal("2")
    )


def test_state_lookups() -> None:
    case = next(case for case in CASES if case.name == "loop-until-pass")
    state = run_case(case)
    assert [item.id for item in state.of("draft")] == ["draft/0/1", "draft/1/1", "draft/2/1"]
    latest = state.latest("review")
    assert latest is not None and latest.id == "review/2/1"
    assert state.latest("ghost") is None
    with pytest.raises(KeyError):
        state.invocation("ghost//1")
