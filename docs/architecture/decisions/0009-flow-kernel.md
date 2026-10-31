# ADR 0009 – Flow kernel: one deterministic, I/O-free semantics for workflows

Status: accepted

## Context

Orkexa runs single agents and lets agents delegate (`orkexa.dynamic.Orchestrator`), but
it had no notion of a *workflow*: a graph of agents, people and tools with branches,
joins, approvals and repetition. Applications built on Orkexa (Nodestro's runtime)
therefore wrote their own scheduler. That scheduler only knew acyclic graphs, kept one
result per node, and its rules (when a node waits, runs, is skipped or blocked; how a
paused run continues) lived in one application. Quality loops ("revise until the
reviewer passes it"), repeated invocations, iteration limits and nested orchestration
need the same rules in more places, and two schedulers would drift apart.

## Decision

* **A kernel, not an engine.** `orkexa.flow` is a set of pure functions: `start`,
  `complete`, `advance`, `cancel`, `resume` and `outcome`. Each takes the flow
  specification, the stored `FlowState` and one explicit input and returns a `Step`:
  the new state, the invocations to execute (`RunNode`) and the events. The kernel
  reads no clock (time arrives as `now`), touches no network or file system and uses
  no randomness. Equal inputs give equal outputs in every process and driver.
* **Drivers execute.** A driver runs the dispatched invocations, reports every
  `Result`, and stores the state wherever it likes. Orkexa ships an asyncio reference
  driver (`run_flow`) and a conformance suite (`orkexa.flow.conformance`); Nodestro's
  runtime becomes a second driver of the same kernel. There is exactly one
  authoritative flow semantics, and `orkexa.flow` imports nothing from any application.
* **Activation-based model.** Nodes are executed as *invocations*, identified as
  `node/path/attempt`, where `path` lists the iteration of every loop around the node.
  A node may run many times, every run keeps its own result, and the history is the
  state itself.
* **Settle rules (contract 1.0)**, identical to the scheduler they replace: a node with
  `join="all"` waits until every predecessor is settled; a failed, blocked or cancelled
  predecessor blocks it; a suspended predecessor keeps it waiting; if no incoming
  connection is active it is skipped; otherwise it is ready. A connection is active when
  its source succeeded and, for a branching source, opened the edge's port.
  `join="any"` starts a node as soon as one connection is active.
* **Loops are explicit edges with a mandatory limit.** A loop edge (`Edge.loop`) must
  leave a branching node through a port and lead back to it or a node before it. Its
  body (every node on a path from target to source) runs at most `max_iterations`
  times (2–1000, no default). Bodies may nest but not overlap; a node starts and closes
  at most one loop. When the limit is reached, `on_exhausted="fail"` stops the flow
  (`flow.loop_exhausted`) and `"continue"` lets the other open ports decide. A node
  after a loop waits until the whole loop is over and sees its last iteration. Cycles
  without a loop edge stay invalid (`flow.cycle`).
* **Limits are part of the contract.** `Limits` caps invocations, parallel invocations,
  loop iterations and wall time (`flow.invocation_limit`, `flow.time_limit`, both
  failures); `max_depth` and `budget_usd` are enforced by drivers. `Limits.within()`
  narrows a child scope to its parent's.
* **Stops and resumption.** A `Result` may stop the flow (for example a rejected
  approval); `cancel()` stops it from outside. Waiting work is cancelled at once;
  running invocations stay running until the driver reports them. A `suspended`
  invocation waits for something outside the flow and is dispatched again by
  `resume()`. A driver that finds `running` invocations in a stored state must not
  re-run them silently (the reference driver raises `flow.interrupted`).
* **Versioned contract.** `CONTRACT_VERSION` (`1.0`) is stored in every state and flow;
  a state with another major version is refused. Kernel events carry
  `schema="orkexa.flow-event/1"`. Changing the meaning of a stored state needs a new
  major version and a migration.
* **Conditions stay outside for now.** The kernel learns which ports a branching node
  opened from its `Result`; how a driver evaluates a condition (rules, a model, a
  person) is the driver's business. A shared condition evaluator can follow without a
  contract change.

## Consequences

* Workflows gain loops, repeated invocations, iteration and parallelism limits, and a
  serializable history without a second scheduler.
* Applications migrate by turning their scheduler into a driver; golden traces and
  differential tests against the old scheduler prove equal behavior before it is
  removed.
* The kernel is small and fully covered: unit tests, the conformance cases, an oracle
  comparison on random acyclic flows and order-independence checks on random flows with
  loops (`tests/flow`).
* Not in contract 1.0: retries as new attempts, state-machine regions, time-outs per
  invocation and budget reservations. They are additive and get minor versions.
