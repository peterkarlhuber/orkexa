# ADR 0011 – Dynamic teams, parallel delegation, parameters and paused delegations

Status: accepted

## Context

ADR 0010 lets a model choose, one decision at a time, which member of a fixed pool
works next. Adaptive Orchestration M6b (Nodestro) asks for more, without giving up any
rule of ADR 0010:

* the orchestrator assembles a **temporary team** while it runs, changes it when
  intermediate results call for it, and may call the same member again;
* independent parts run **in parallel**, within a hard limit;
* the orchestrator may set a member's **parameters**, but only within what the member
  declares (decision AO-5, level 2); creating members from approved components
  (level 3) stays outside the protocol;
* a member that **waits for a person** (an approval node, a tool approval) must pause
  the orchestration instead of failing it, possibly several levels deep, and continue
  the same execution after the decision without asking the model again;
* the parent must tell running, waiting, succeeded, failed, rejected, cancelled and
  expired apart.

## Decision

Protocol version 2 (`PROTOCOL_VERSION = 2`; version 1 records stay valid and replay
unchanged, `READABLE_VERSIONS = (1, 2)`).

* **Decisions.** `Delegate` gains `values`; new are `Parallel(assignments, reason)`
  (2 to `max_parallel` different members, each an `Assignment(member, task, values)`)
  and `Staff(add, remove, reason)`. `Finish` is unchanged. Still no "create" decision.
* **Team.** With `OrchestrationLimits.staffing` the orchestrator delegates only to the
  members it added with `Staff`; the team at any point is derived from the recorded
  steps (`OrchestrationState.team()`), so it is visible and replayable. The pool stays
  the hard boundary: `Staff` can only add pool members, at most `max_team` at a time.
  Without `staffing` the whole pool is the team (ADR 0010 behavior).
* **Parameters.** `Member.parameters` declares `Parameter(name, type, required,
  description, choices)` with type `text`, `number`, `boolean` or `choice`. `validate`
  refuses unknown keys (`orchestration.unknown_parameter`) and wrong kinds or values
  (`orchestration.invalid_value`). The application checks the values again against the
  member's own input rules; the protocol never widens what a member accepts.
* **Outcomes.** Every accepted delegation records one `Outcome(member, status, result,
  error, ref)` per assignment; `status` is `succeeded`, `failed`, `rejected`,
  `cancelled`, `expired` or `waiting`. `ref` is the application's reference of the
  member's execution (e.g. a child run id).
* **Pauses.** When an assignment waits, `record()` keeps the decision as
  `OrchestrationState.pending` (a `Pending(index, decision, outcomes)`) instead of a
  step; `validate` refuses every new decision with `orchestration.paused`. Continuing
  the orchestration runs only the waiting assignments again, each with the `ref` it
  waited with, so the application continues that execution and never starts a new
  one; `complete_pending()` records the step once nothing waits any more. The model is
  not asked again for a pending decision. A member that left the pool in the meantime
  fails with `orchestration.unknown_member` instead of running.
* **Driver.** `run_orchestration()`/`arun_orchestration()` accept `execute`, which
  receives a `Delegation(member, task, values, step, slot, ref)` and raises
  `DelegationSuspended(ref)` to pause or `DelegationEnded(status, message)` for a
  known end without result (`delegate(member, task)` of version 1 still works). The
  async driver runs parallel assignments at the same time; a budget error (or a
  `stop_on` exception) of one cancels the others and ends the orchestration. A paused
  state is returned (and passed to `on_step`) so the application can save it.
* **New refusal codes:** `paused`, `not_in_team`, `unknown_parameter`,
  `invalid_value`, `parallel_disabled`, `parallel_limit`, `parallel_duplicate`,
  `staffing_disabled`, `empty_staffing`, `already_in_team`, `team_limit`.

## Consequences

* An orchestration can be stopped at any approval below it and continued after a
  restart from its saved state alone; the decision record shows who was in the team
  at every step and why.
* `replay()` re-validates version 2 records including a pending decision, so a smaller
  pool or stricter limits can be checked against a recorded run.
* Permissions, approvals, budgets and sandboxing stay with the application (ADR 0010):
  the protocol cannot make a member do anything its own rules forbid.
* Level 3 (assembling new members from approved components) is not part of the
  protocol; an application that adds it must do so outside a running orchestration.
