# Flows: workflows with branches, loops and approvals

`orkexa.flow` describes a workflow as a graph and decides what runs next. You supply
one handler per node type (an agent call, a person's decision, a tool); the kernel
handles waiting, branching, joining, repetition, limits and resumption. Everything it
knows is in a JSON-serializable `FlowState`, so a flow can stop in one process and
continue in another. The design is recorded in
[ADR 0009](../architecture/decisions/0009-flow-kernel.md).

## A first flow

```python
from orkexa.flow import Edge, Flow, Node, TaskContext, TaskResult, run_flow_sync

flow = Flow(
    nodes=(
        Node(id="research", type="agent"),
        Node(id="summary", type="agent"),
    ),
    edges=(Edge(id="e1", source="research", target="summary"),),
)


async def agent(context: TaskContext) -> TaskResult:
    # context.inputs holds the outputs of active predecessors, by node id
    return TaskResult(output=f"{context.node.id} done with {dict(context.inputs)}")


run = run_flow_sync(flow, {"agent": agent})
print(run.outcome)  # succeeded
print(run.outputs)  # {"research//1": ..., "summary//1": ...}
```

Outputs are stored per **invocation**: one execution of one node, identified as
`node/path/attempt`. Outside loops the path is empty (`summary//1`).

## Branches and joins

A branching node declares `ports`; each outgoing edge belongs to one port, and the
handler reports which ports it opened:

```python
Node(id="triage", type="classifier", ports=("urgent", "normal"))
Edge(id="u", source="triage", target="escalate", port="urgent")
Edge(id="n", source="triage", target="queue", port="normal")


async def classifier(context):
    return TaskResult(output="P1", ports=("urgent",))
```

The rules (the same everywhere the kernel runs):

| A node... | when... |
|---|---|
| waits | a predecessor has not settled yet (`join="all"`, the default) |
| is **blocked** | a predecessor failed, was blocked or cancelled |
| keeps waiting | a predecessor is suspended (for example an approval) |
| is **skipped** | none of its incoming connections is active |
| runs | otherwise; with `join="any"` as soon as one connection is active |

A connection is active when its source succeeded and, for a branching source, opened
the edge's port.

## Loops with a mandatory limit

A loop edge leads back from a branching node to itself or an earlier node. It needs a
`Loop` with `max_iterations` (2 to 1000, the first run included); there is no default
and no unbounded loop.

```python
from orkexa.flow import Loop

edges = (
    Edge(id="draft-review", source="draft", target="review"),
    Edge(id="retry", source="review", target="draft", port="again", loop=Loop(max_iterations=3)),
    Edge(id="done", source="review", target="publish", port="pass"),
)
```

* The **body** is every node on a path from the loop edge's target to its source.
  Iterations get their own invocations (`draft/0/1`, `draft/1/1`, ...). In later
  iterations the loop head receives the previous output of the loop source in
  `context.inputs` (here the reviewer's feedback).
* A node **after** the loop waits until the loop is over and sees its last iteration.
* When the limit is reached, `on_exhausted="fail"` (default) stops the flow with
  `flow.loop_exhausted`; `on_exhausted="continue"` ignores the loop edge, so the
  node's other open ports decide.
* Loops can nest (a revision loop inside a planning loop). They must not overlap, and
  a node starts and closes at most one loop. A cycle without a loop edge is invalid.

Check a flow before saving it with `validate(flow)`; every issue has a stable `code`
(`flow.cycle`, `flow.loop_unguarded`, `flow.loop_overlap`, ...).

## Approvals, suspension and stored state

A handler that has to wait for something outside the flow returns
`Result(status="suspended")`. The flow ends the run as `suspended`; store the state and
the outputs, and continue later with `resume=True` (or the id of one invocation):

```python
run = run_flow_sync(flow, handlers)
saved_state, saved_outputs = run.state.model_dump_json(), run.outputs

# later, in another process, after the person decided:
later = run_flow_sync(
    flow,
    handlers,
    state=FlowState.model_validate_json(saved_state),
    outputs=saved_outputs,
    resume=True,
)
```

A result can also stop the whole flow, for example a rejected approval:
`TaskResult(stop=Stop(code="approval_rejected"))`. Running handlers are then cancelled
and the flow ends as `cancelled`.

## Limits

`Flow(limits=Limits(...))` bounds a flow:

| Limit | Effect |
|---|---|
| `max_invocations` | stops the flow (`flow.invocation_limit`) before one more invocation would start |
| `max_parallel` | at most this many invocations run at the same time; the rest wait |
| `max_iterations` | upper bound for every loop's `max_iterations` (checked by `validate`) |
| `max_duration_s` | stops the flow (`flow.time_limit`); measured with the driver's clock |
| `max_depth`, `budget_usd` | for drivers that run child flows or track costs |

`Limits.within(parent)` narrows the limits of a child flow to what its parent allows.

## Writing your own driver

`run_flow` is a reference driver. Applications that need persistence, streaming or
permissions call the kernel directly:

```python
from orkexa.flow import complete, start

step = start(flow, now=clock())
store(step.state)
for action in step.actions:  # RunNode(invocation=...)
    execute_later(action.invocation)

# when an invocation finishes:
step = complete(flow, load_state(), invocation_id, Result(status="succeeded"), now=clock())
```

Every call returns the new state, the next invocations and the events (`flow_started`,
`invocation_started`, `invocation_settled`, `invocation_resumed`, `loop_iteration`,
`limit_reached`, `flow_stopped`; schema `orkexa.flow-event/1`). A driver must not
re-run an invocation it finds `running` in a stored state without deciding what
happened to it first; the reference driver raises `flow.interrupted`.

`orkexa.flow.conformance.CASES` lists scripted flows with the statuses every driver
must reach; run them through your driver and compare with `check(case, state)`.

## Example

[`examples/13_flow_quality_loop.py`](https://github.com/peterkarlhuber/orkexa/blob/master/examples/13_flow_quality_loop.py)
combines a writer and a reviewer agent in a loop, a human approval, a stored state and
its continuation, offline.
