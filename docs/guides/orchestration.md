# Orchestration: choosing pool members at run time

`orkexa.orchestration` lets a model decide which member of a fixed pool works next, and
records every decision ([ADR 0010](../architecture/decisions/0010-orchestration-protocol.md)).

```python
from orkexa import Agent
from orkexa.orchestration import (
    DecisionEnvelope,
    Member,
    OrchestrationLimits,
    Pool,
    run_orchestration,
)

pool = Pool(
    members=(
        Member(id="research", name="Research", description="Finds sources", capabilities=("research",)),
        Member(id="write", name="Writer", contract=("title", "text")),
    )
)


def decide(prompt: str) -> DecisionEnvelope:
    # A new agent per decision: the prompt already holds everything that happened so far.
    chooser = Agent(
        model="anthropic:claude-opus-5-5",
        output_format=DecisionEnvelope,
        system_prompt="You orchestrate a team. Delegate, then finish.",
    )
    return chooser.ask(prompt)


def delegate(
    member: Member, task: str
) -> str: ...  # run the team or agent with its own budget and approvals; return its result


state = run_orchestration(
    "Write a short article about otters",
    pool,
    decide=decide,
    delegate=delegate,
    limits=OrchestrationLimits(max_decisions=6, max_per_member=2),
)
print(state.outcome, state.answer)
for step in state.steps:
    print(step.index, step.decision, step.verdict.code)
```

* **Decisions:** `Delegate(member, task, reason)` or `Finish(answer, reason)`. Nothing
  else: members are selected, never created.
* **Refusals** (unknown member, repeated task, member limit, empty task) are recorded,
  count against `max_decisions` and are explained to the model in the next prompt.
* **Failures** of a member are recorded as the step's `error`; the model can react. A
  `BudgetExceededError` ends the orchestration.
* **Record and replay:** `state.model_dump_json()` is the complete record;
  `replay(state, pool, limits)` checks it again without model calls.

## Teams, parallel work, parameters and pauses (protocol version 2)

[ADR 0011](../architecture/decisions/0011-dynamic-teams-and-paused-delegations.md) gives
the orchestrator more freedom, always inside the pool and the limits:

```python
from orkexa.orchestration import (
    Delegation,
    DelegationSuspended,
    OrchestrationLimits,
    run_orchestration,
)

limits = OrchestrationLimits(staffing=True, max_team=3, max_parallel=2)


def execute(item: Delegation) -> str:
    # item.values holds parameter values the member declared (Member.parameters).
    # item.ref is set when this assignment waited before: continue that execution.
    if needs_approval(item):
        raise DelegationSuspended(ref=start_or_continue(item))
    return run_member(item)


state = run_orchestration("Write the report", pool, decide=decide, execute=execute, limits=limits)
if state.paused:
    save(state)  # later: run_orchestration(..., state=load()) continues the waiting members
```

* **`Staff(add, remove)`** assembles the team with `staffing=True`; only team members
  can get work, the team never exceeds `max_team` and never leaves the pool.
* **`Parallel(assignments)`** starts 2 to `max_parallel` different members at once
  (concurrently in `arun_orchestration`). Every assignment gets its own `Outcome`.
* **Parameters:** a `Member` lists `Parameter`s; the orchestrator sets them in `values`
  and `validate` refuses unknown keys and wrong values.
* **Pauses:** raising `DelegationSuspended` keeps the decision as `state.pending`. Calling
  the driver again with that state runs only the waiting members, with the `ref` they
  waited with, and asks the model nothing until they ended. `DelegationEnded("rejected")`,
  `"expired"` or `"cancelled"` reports a member that ended without a result.
