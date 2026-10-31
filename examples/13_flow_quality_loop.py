"""13. A flow with a quality loop, a human approval and a stored state (offline).

``orkexa.flow`` decides what runs next; handlers do the work. Here a writer agent
drafts, a reviewer agent sends the draft back until it passes (at most three rounds,
the loop's mandatory limit), and a person approves before publishing. The flow
suspends at the approval, its state is stored as JSON, and a later run continues it.
The agents use the scripted fake provider, so nothing here calls a model API.
"""

from __future__ import annotations

from orkexa import AsyncAgent
from orkexa.flow import (
    Edge,
    Flow,
    FlowState,
    Limits,
    Loop,
    Node,
    Result,
    TaskContext,
    TaskResult,
    run_flow_sync,
)
from orkexa.providers.fake import FakeModel, text

FLOW = Flow(
    nodes=(
        Node(id="draft", type="writer"),
        Node(id="review", type="reviewer", ports=("pass", "again")),
        Node(id="approve", type="human", ports=("approved", "rejected")),
        Node(id="publish", type="publish"),
    ),
    edges=(
        Edge(id="draft-review", source="draft", target="review"),
        Edge(id="retry", source="review", target="draft", port="again", loop=Loop(max_iterations=3)),
        Edge(id="review-approve", source="review", target="approve", port="pass"),
        Edge(id="approve-publish", source="approve", target="publish", port="approved"),
    ),
    limits=Limits(max_invocations=20),
)

writer = AsyncAgent(
    model="fake:writer",
    client=FakeModel(
        [text("Orkexa runs agents."), text("Orkexa runs agents on four providers with one API.")]
    ),
)
reviewer = AsyncAgent(
    model="fake:reviewer",
    client=FakeModel([text("AGAIN: name the providers"), text("PASS")]),
)
decisions: dict[str, str] = {}  # invocation id -> "approved" or "rejected", e.g. from a web form


async def write(context: TaskContext) -> TaskResult:
    feedback = context.inputs.get("review")  # the reviewer's last verdict, in later rounds
    prompt = "Write one sentence about Orkexa." + (f" Feedback: {feedback}" if feedback else "")
    return TaskResult(output=str(await writer.ask(prompt)))


async def review(context: TaskContext) -> TaskResult:
    verdict = str(await reviewer.ask(f"Review: {context.inputs['draft']}"))
    print(f"round {context.invocation.path[0] + 1}: {verdict}")
    if verdict.startswith("PASS"):
        return TaskResult(output=context.inputs["draft"], ports=("pass",))  # hand the draft on
    return TaskResult(output=verdict, ports=("again",))  # the writer gets the verdict as feedback


async def human(context: TaskContext) -> Result | TaskResult:
    decision = decisions.get(context.invocation.id)
    if decision is None:
        return Result(status="suspended")  # wait for a person; the flow can be stored now
    return TaskResult(output=context.inputs["review"], ports=(decision,))


async def publish(context: TaskContext) -> TaskResult:
    return TaskResult(output=f"published: {context.inputs['approve']}")


HANDLERS = {"writer": write, "reviewer": review, "human": human, "publish": publish}


def main() -> None:
    run = run_flow_sync(FLOW, HANDLERS)
    stored = run.state.model_dump_json()  # keep it in a database next to the outputs
    waiting = [item.id for item in run.state.invocations if item.status == "suspended"]
    print(f"{run.outcome}: waiting for {waiting}, state {len(stored)} bytes")

    decisions[waiting[0]] = "approved"
    later = run_flow_sync(
        FLOW, HANDLERS, state=FlowState.model_validate_json(stored), outputs=run.outputs, resume=True
    )
    drafts = [item.id for item in later.state.invocations if item.node == "draft"]
    print(f"{later.outcome}: drafts {drafts}, loop counters {later.state.iterations}")
    print(later.outputs["publish//1"])


if __name__ == "__main__":
    main()
