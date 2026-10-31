"""Conformance cases: scripted flows with the statuses every driver must arrive at.

A case says what each node reports (per dispatch, in order) and what the flow must end
with. :func:`run_case` drives the kernel directly; another driver (Nodestro's runtime)
runs the same cases through its own execution path and checks its result with
:func:`check`, so both provably share one flow semantics.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise

from . import kernel
from .model import Edge, Flow, FlowState, Limits, Loop, Node, Outcome, Result, Stop

__all__ = ["CASES", "ConformanceCase", "check", "run_case"]


@dataclass(frozen=True)
class ConformanceCase:
    """One scripted flow and its expected end.

    Attributes:
        name: Unique, stable name.
        flow: The flow.
        script: Per node id, the results of its dispatches in order; the last one repeats.
            Nodes without a script succeed (a branching node opens its first port).
        statuses: Expected final status per invocation id; invocations not listed must not exist.
        outcome: Expected :func:`orkexa.flow.outcome`.
        stop: Expected stop code, if the flow must be stopped.
        resumes: How often every suspended invocation is resumed after the flow suspends.
    """

    name: str
    flow: Flow
    statuses: Mapping[str, str]
    outcome: Outcome
    script: Mapping[str, Sequence[Result]] = field(default_factory=dict)
    stop: str | None = None
    resumes: int = 0


def scripted(case: ConformanceCase) -> Callable[[str, int], Result]:
    """The scripted result of ``node``'s ``count``-th dispatch (1-based)."""
    nodes = {node.id: node for node in case.flow.nodes}

    def result(node: str, count: int) -> Result:
        steps = case.script.get(node)
        if steps:
            return steps[min(count, len(steps)) - 1]
        ports = nodes[node].ports
        return Result(status="succeeded", ports=(ports[0],) if ports else None)

    return result


def run_case(case: ConformanceCase) -> FlowState:
    """Drive the kernel through ``case`` (dispatch order: first come, first served)."""
    result = scripted(case)
    counts: dict[str, int] = {}
    step = kernel.start(case.flow)
    state = step.state
    queue = [action.invocation for action in step.actions]
    resumes = case.resumes
    while True:
        while queue:
            item = queue.pop(0)
            counts[item.node] = counts.get(item.node, 0) + 1
            step = kernel.complete(case.flow, state, item.id, result(item.node, counts[item.node]))
            state = step.state
            queue.extend(action.invocation for action in step.actions)
        if resumes and kernel.outcome(state) == "suspended":
            resumes -= 1
            step = kernel.resume(case.flow, state)
            state = step.state
            queue.extend(action.invocation for action in step.actions)
            continue
        return state


def check(case: ConformanceCase, state: FlowState) -> list[str]:
    """Every difference between ``state`` and what ``case`` expects (empty when it conforms)."""
    problems: list[str] = []
    actual = {item.id: item.status for item in state.invocations}
    for invocation, status in case.statuses.items():
        if actual.get(invocation) != status:
            problems.append(f"{invocation}: expected {status}, got {actual.get(invocation, 'nothing')}")
    for invocation in sorted(set(actual) - set(case.statuses)):
        problems.append(f"{invocation}: unexpected invocation ({actual[invocation]})")
    if kernel.outcome(state) != case.outcome:
        problems.append(f"outcome: expected {case.outcome}, got {kernel.outcome(state)}")
    code = state.stop.code if state.stop else None
    if code != case.stop:
        problems.append(f"stop: expected {case.stop}, got {code}")
    return problems


def _chain(*names: str) -> tuple[Edge, ...]:
    return tuple(Edge(id=f"{a}-{b}", source=a, target=b) for a, b in pairwise(names))


_OK = Result(status="succeeded")
_FAIL = Result(status="failed", error="test.failed")
_WAIT = Result(status="suspended")
_DECIDE = (Node(id="d", type="condition", ports=("yes", "no")),)
_REVIEW = Node(id="review", type="condition", ports=("pass", "again"))


#: The standard cases. Invocation ids are ``node/path/attempt``.
CASES: tuple[ConformanceCase, ...] = (
    ConformanceCase(
        name="linear",
        flow=Flow(nodes=(Node(id="a"), Node(id="b"), Node(id="c")), edges=_chain("a", "b", "c")),
        statuses={"a//1": "succeeded", "b//1": "succeeded", "c//1": "succeeded"},
        outcome="succeeded",
    ),
    ConformanceCase(
        name="parallel-join",
        flow=Flow(
            nodes=(Node(id="s"), Node(id="x"), Node(id="y"), Node(id="j")),
            edges=(*_chain("s", "x", "j"), *_chain("s", "y", "j")),
        ),
        statuses={"s//1": "succeeded", "x//1": "succeeded", "y//1": "succeeded", "j//1": "succeeded"},
        outcome="succeeded",
    ),
    ConformanceCase(
        name="branch-skips-other-side",
        flow=Flow(
            nodes=(Node(id="s"), *_DECIDE, Node(id="y"), Node(id="n"), Node(id="after")),
            edges=(
                *_chain("s", "d"),
                Edge(id="d-y", source="d", target="y", port="yes"),
                Edge(id="d-n", source="d", target="n", port="no"),
                *_chain("n", "after"),
            ),
        ),
        script={"d": [Result(status="succeeded", ports=("yes",))]},
        statuses={
            "s//1": "succeeded",
            "d//1": "succeeded",
            "y//1": "succeeded",
            "n//1": "skipped",
            "after//1": "skipped",
        },
        outcome="succeeded",
    ),
    ConformanceCase(
        name="failure-blocks-successors",
        flow=Flow(
            nodes=(Node(id="s"), Node(id="x"), Node(id="y"), Node(id="j")),
            edges=(*_chain("s", "x", "j"), *_chain("s", "y", "j")),
        ),
        script={"x": [_FAIL]},
        statuses={"s//1": "succeeded", "x//1": "failed", "y//1": "succeeded", "j//1": "blocked"},
        outcome="failed",
    ),
    ConformanceCase(
        name="join-any-starts-early",
        flow=Flow(
            nodes=(Node(id="s"), *_DECIDE, Node(id="y"), Node(id="n"), Node(id="j", join="any")),
            edges=(
                *_chain("s", "d"),
                Edge(id="d-y", source="d", target="y", port="yes"),
                Edge(id="d-n", source="d", target="n", port="no"),
                *_chain("y", "j"),
                *_chain("n", "j"),
            ),
        ),
        statuses={
            "s//1": "succeeded",
            "d//1": "succeeded",
            "y//1": "succeeded",
            "n//1": "skipped",
            "j//1": "succeeded",
        },
        outcome="succeeded",
    ),
    ConformanceCase(
        name="suspend-then-resume",
        flow=Flow(
            nodes=(Node(id="a"), Node(id="ask", type="approval"), Node(id="b")), edges=_chain("a", "ask", "b")
        ),
        script={"ask": [_WAIT, _OK]},
        resumes=1,
        statuses={"a//1": "succeeded", "ask//1": "succeeded", "b//1": "succeeded"},
        outcome="succeeded",
    ),
    ConformanceCase(
        name="suspended-stays-suspended",
        flow=Flow(
            nodes=(Node(id="a"), Node(id="ask", type="approval"), Node(id="b"), Node(id="free")),
            edges=(*_chain("a", "ask", "b"), *_chain("a", "free")),
        ),
        script={"ask": [_WAIT]},
        statuses={"a//1": "succeeded", "ask//1": "suspended", "free//1": "succeeded"},
        outcome="suspended",
    ),
    ConformanceCase(
        name="stop-from-result",
        flow=Flow(
            nodes=(
                Node(id="a"),
                Node(id="ask", type="approval", ports=("approved", "rejected")),
                Node(id="b"),
            ),
            edges=(*_chain("a", "ask"), Edge(id="ask-b", source="ask", target="b", port="approved")),
        ),
        script={
            "ask": [
                Result(
                    status="succeeded",
                    ports=("rejected",),
                    stop=Stop(code="run.approval_rejected", outcome="cancelled"),
                )
            ]
        },
        statuses={"a//1": "succeeded", "ask//1": "succeeded", "b//1": "cancelled"},
        outcome="cancelled",
        stop="run.approval_rejected",
    ),
    ConformanceCase(
        name="loop-until-pass",
        flow=Flow(
            nodes=(Node(id="draft"), _REVIEW, Node(id="publish")),
            edges=(
                *_chain("draft", "review"),
                Edge(id="retry", source="review", target="draft", port="again", loop=Loop(max_iterations=5)),
                Edge(id="review-publish", source="review", target="publish", port="pass"),
            ),
        ),
        script={
            "review": [
                Result(status="succeeded", ports=("again",)),
                Result(status="succeeded", ports=("again",)),
                Result(status="succeeded", ports=("pass",)),
            ]
        },
        statuses={
            "draft/0/1": "succeeded",
            "review/0/1": "succeeded",
            "draft/1/1": "succeeded",
            "review/1/1": "succeeded",
            "draft/2/1": "succeeded",
            "review/2/1": "succeeded",
            "publish//1": "succeeded",
        },
        outcome="succeeded",
    ),
    ConformanceCase(
        name="loop-exhausted-fails",
        flow=Flow(
            nodes=(Node(id="draft"), _REVIEW, Node(id="publish")),
            edges=(
                *_chain("draft", "review"),
                Edge(id="retry", source="review", target="draft", port="again", loop=Loop(max_iterations=2)),
                Edge(id="review-publish", source="review", target="publish", port="pass"),
            ),
        ),
        script={"review": [Result(status="succeeded", ports=("again",))]},
        statuses={
            "draft/0/1": "succeeded",
            "review/0/1": "succeeded",
            "draft/1/1": "succeeded",
            "review/1/1": "succeeded",
            "publish//1": "cancelled",
        },
        outcome="failed",
        stop="flow.loop_exhausted",
    ),
    ConformanceCase(
        name="loop-exhausted-continues",
        flow=Flow(
            nodes=(
                Node(id="draft"),
                Node(id="review", type="condition", ports=("done", "again")),
                Node(id="publish"),
            ),
            edges=(
                *_chain("draft", "review"),
                Edge(
                    id="retry",
                    source="review",
                    target="draft",
                    port="again",
                    loop=Loop(max_iterations=2, on_exhausted="continue"),
                ),
                Edge(id="review-publish", source="review", target="publish", port="done"),
            ),
        ),
        script={"review": [Result(status="succeeded", ports=("again", "done"))]},
        statuses={
            "draft/0/1": "succeeded",
            "review/0/1": "succeeded",
            "draft/1/1": "succeeded",
            "review/1/1": "succeeded",
            "publish//1": "succeeded",
        },
        outcome="succeeded",
    ),
    ConformanceCase(
        name="nested-loops",
        flow=Flow(
            nodes=(
                Node(id="plan"),
                Node(id="write"),
                Node(id="check", type="condition", ports=("ok", "fix")),
                Node(id="judge", type="condition", ports=("ok", "replan")),
                Node(id="done"),
            ),
            edges=(
                *_chain("plan", "write", "check"),
                Edge(id="fix", source="check", target="write", port="fix", loop=Loop(max_iterations=3)),
                Edge(id="check-judge", source="check", target="judge", port="ok"),
                Edge(id="replan", source="judge", target="plan", port="replan", loop=Loop(max_iterations=2)),
                Edge(id="judge-done", source="judge", target="done", port="ok"),
            ),
        ),
        script={
            "check": [
                Result(status="succeeded", ports=("fix",)),
                Result(status="succeeded", ports=("ok",)),
                Result(status="succeeded", ports=("ok",)),
            ],
            "judge": [
                Result(status="succeeded", ports=("replan",)),
                Result(status="succeeded", ports=("ok",)),
            ],
        },
        statuses={
            "plan/0/1": "succeeded",
            "write/0.0/1": "succeeded",
            "check/0.0/1": "succeeded",
            "write/0.1/1": "succeeded",
            "check/0.1/1": "succeeded",
            "judge/0/1": "succeeded",
            "plan/1/1": "succeeded",
            "write/1.0/1": "succeeded",
            "check/1.0/1": "succeeded",
            "judge/1/1": "succeeded",
            "done//1": "succeeded",
        },
        outcome="succeeded",
    ),
    ConformanceCase(
        name="invocation-limit",
        flow=Flow(
            nodes=(Node(id="a"), Node(id="b"), Node(id="c")),
            edges=_chain("a", "b", "c"),
            limits=Limits(max_invocations=2),
        ),
        statuses={"a//1": "succeeded", "b//1": "succeeded", "c//1": "cancelled"},
        outcome="failed",
        stop="flow.invocation_limit",
    ),
)
