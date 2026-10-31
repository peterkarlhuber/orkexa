# ADR 0010 – Orchestration protocol: select from a pool, validate every decision, record everything

Status: accepted

## Context

Adaptive Orchestration (Nodestro, decisions AO-1 to AO-13) adds an *orchestrator*: a
model that decides at run time which team or agent works next. `orkexa.dynamic.Orchestrator`
already delegates, but it is a different kind of tool: it may create new agents and
build new tool code while it runs, it has its own approval hooks, and what it decided is
only visible in its conversation. For a workflow product this is not acceptable:

* no new executable tool code may appear while a workflow runs;
* creating agents is reserved for administrators (AO-5); selecting is the default;
* permissions come from one shared mechanism of the application (tool approvals), not
  from the orchestrator;
* every decision must be explainable afterwards and checkable without calling a model
  again (monitor, replay);
* limits must be hard, also against a model that misbehaves.

## Decision

* **A protocol, not an agent.** `orkexa.orchestration` defines two decisions:
  `Delegate(member, task, reason)` and `Finish(answer, reason)`. There is no
  "create" decision: an answer that asks for one is refused as invalid. Members come from
  a fixed `Pool` of `Member`s (id, name, description, kind `team` or `agent`,
  capabilities, contract field names) that the application builds from what exists.
* **Pure validation.** `validate(decision, pool, state, limits)` returns a `Verdict`
  with a stable code (`orchestration.unknown_member`, `member_limit`, `repeated`,
  `decision_limit`, `empty_task`, `task_too_long`, `empty_answer`, `answer_too_long`,
  `finished`). It reads nothing but its arguments.
* **Hard limits.** `OrchestrationLimits`: `max_decisions` (1 to 50, refused and invalid
  answers included), `max_per_member`, `max_answer_chars`. At the decision limit the
  orchestration stops with `orchestration.decision_limit`; the model is told in the
  last prompt that it must finish. Money is limited by the budget tracker (ADR 0008,
  AO-12 reservations), not by the protocol; a `BudgetExceededError` from a member ends
  the orchestration.
* **Complete record.** `OrchestrationState` holds the task, every `Step` (decision,
  verdict, the member's result or error, the raw text of an invalid answer) and the
  outcome. It is plain JSON with a `version`; an application stores it after every step
  (`on_step`) and can continue it after a restart.
* **Replay.** `replay(record, pool, limits)` validates a recorded orchestration again
  without any model call, also under another pool or other limits ("what would a smaller
  pool have refused?").
* **Reference driver, no execution logic.** `run_orchestration` / `arun_orchestration`
  ask a `decide` callable for the next decision (any provider, any structured-output
  method; `DecisionEnvelope` is the schema) and a `delegate` callable to run a member.
  The driver never runs a member itself and has no permission logic, so a member runs
  with exactly the rules it has elsewhere.

## Consequences

* Nodestro's runtime can add an orchestrator node whose members are existing teams
  (sub-workflows with output contracts) and whose decisions are shown in a monitor, with
  the shared tool approvals applying inside every member run.
* `orkexa.dynamic.Orchestrator` stays for library users who want generated agents and
  tools; it is not used by the runtime for workflows.
* The model sees members only through their id, name, description, capabilities and
  contract. Whatever a member returns is data in the next prompt; prompt injection in a
  member's result can at most steer the next choice *within* the pool and the limits.
