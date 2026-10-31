"""Orchestration protocol: an AI model chooses which pool member works next, within limits.

An orchestrator is a model that solves a task by **delegating** to members of a fixed
**pool** (teams or agents that already exist) and finally **finishing** with an answer.
This package defines the protocol (ADR 0010) and nothing else: what a decision is, how it
is validated, what is recorded, and how a recorded orchestration is replayed.

* :class:`Member` and :class:`Pool`: who may be chosen. Members are selected, never
  created: the protocol has no "create" decision, so no new agent and no new tool code can
  appear while an orchestration runs.
* :class:`Delegate` and :class:`Finish`: the two decisions (:data:`Decision`).
* :class:`OrchestrationLimits`: maximum decisions, delegations per member, and the
  answer length.
* :func:`validate`: the pure check of one decision against pool, limits and history.
* :class:`OrchestrationState` with :class:`Step`: the complete record, plain JSON; it is
  enough to show every decision with its reason and verdict, and to :func:`replay` it.
* :func:`run_orchestration` / :func:`arun_orchestration`: a reference driver that asks a
  ``decide`` callable for the next decision and a ``delegate`` callable to run a member.

The driver never runs a member itself and has no permission logic: whoever implements
``delegate`` runs the member with its own rules (budgets, tool approvals). A rejected
decision is recorded, counts against ``max_decisions`` and is explained to the model in
the next prompt.
"""

from __future__ import annotations

from .driver import arun_orchestration, decision_prompt, run_orchestration
from .protocol import (
    PROTOCOL_VERSION,
    Decision,
    DecisionEnvelope,
    Delegate,
    Finish,
    Member,
    OrchestrationLimits,
    OrchestrationState,
    Pool,
    Step,
    Verdict,
    parse_decision,
    record,
    replay,
    validate,
)

__all__ = [
    "PROTOCOL_VERSION",
    "Decision",
    "DecisionEnvelope",
    "Delegate",
    "Finish",
    "Member",
    "OrchestrationLimits",
    "OrchestrationState",
    "Pool",
    "Step",
    "Verdict",
    "arun_orchestration",
    "decision_prompt",
    "parse_decision",
    "record",
    "replay",
    "run_orchestration",
    "validate",
]
