"""Reference driver: ask for a decision, validate it, delegate, record; until finish or a limit."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
import json
from typing import Any

from ..dynamic.errors import BudgetExceededError
from .protocol import (
    Delegate,
    Finish,
    Member,
    OrchestrationLimits,
    OrchestrationState,
    Pool,
    Verdict,
    parse_decision,
    record,
    validate,
)

__all__ = ["arun_orchestration", "decision_prompt", "run_orchestration"]

#: Errors of ``delegate`` that end the orchestration instead of being shown to the model.
FATAL: tuple[type[BaseException], ...] = (BudgetExceededError,)

#: How much of a member's result the next prompt repeats (the record keeps all of it).
RESULT_PREVIEW_CHARS = 4000

Decide = Callable[[str], Any]
Run = Callable[[Member, str], str]
ADecide = Callable[[str], Awaitable[Any]]
ARun = Callable[[Member, str], Awaitable[str]]


def _member_line(member: Member) -> str:
    parts = [f"- id: {member.id} | {member.kind}: {member.name}"]
    if member.description:
        parts.append(f"  does: {member.description}")
    if member.capabilities:
        parts.append(f"  capabilities: {', '.join(member.capabilities)}")
    if member.contract:
        parts.append(f"  result fields: {', '.join(member.contract)}")
    return "\n".join(parts)


def decision_prompt(state: OrchestrationState, pool: Pool, limits: OrchestrationLimits) -> str:
    """The prompt for the next decision: task, pool, everything so far and what is left."""
    lines = [
        "Task:",
        state.task,
        "",
        "Pool (you can only delegate to these members, by id):",
        *(_member_line(member) for member in pool.members),
        "",
    ]
    if state.steps:
        lines.append("So far:")
        for step in state.steps:
            decision = step.decision
            if decision is None:
                lines.append(f"{step.index + 1}. Invalid answer, refused: {step.verdict.message}")
            elif not step.verdict.ok:
                lines.append(f"{step.index + 1}. Refused ({step.verdict.code}): {step.verdict.message}")
            elif isinstance(decision, Delegate):
                outcome = (
                    f"failed: {step.error}"
                    if step.error is not None
                    else f"result:\n{(step.result or '')[:RESULT_PREVIEW_CHARS]}"
                )
                lines.append(
                    f"{step.index + 1}. Delegated to {decision.member}: {decision.task}\n   {outcome}"
                )
        lines.append("")
    left = limits.max_decisions - len(state.steps)
    lines.append(
        f"Decisions left: {left}. Answer with one decision: delegate to a member with a complete, "
        "self-contained task, or finish with the final answer."
        + (" This is the last decision: finish now." if left == 1 else "")
    )
    return "\n".join(lines)


def _decision_of(answer: Any) -> tuple[Delegate | Finish | None, Verdict | None, str | None]:
    try:
        return parse_decision(answer), None, None
    except ValueError as exc:
        raw = answer if isinstance(answer, str) else json.dumps(answer, default=str)[:2000]
        return None, Verdict(code="orchestration.invalid_decision", message=str(exc)), raw


def run_orchestration(
    task: str,
    pool: Pool,
    *,
    decide: Decide,
    delegate: Run,
    limits: OrchestrationLimits | None = None,
    state: OrchestrationState | None = None,
    on_step: Callable[[OrchestrationState], None] | None = None,
) -> OrchestrationState:
    """Run an orchestration to its end.

    Args:
        task: The task of the orchestrator.
        pool: The members it may choose.
        decide: Gets the prompt (:func:`decision_prompt`) and returns a decision, an
            envelope, a dict or JSON text.
        delegate: Runs ``member`` with ``task`` and returns its result as text. An exception
            is recorded as the step's error (the orchestration continues); budget and
            cancellation errors should be raised by ``decide`` or re-raised by the caller.
        limits: Hard limits (default :class:`OrchestrationLimits`).
        state: A recorded state to continue (e.g. after a restart).
        on_step: Called with the new state after every step (to save or show it).
    """
    limits = limits or OrchestrationLimits()
    state = state or OrchestrationState(task=task)
    while state.outcome is None:
        decision, refused, raw = _decision_of(decide(decision_prompt(state, pool, limits)))
        verdict = refused or validate(decision, pool, state, limits)  # type: ignore[arg-type]
        result = error = None
        if verdict.ok and isinstance(decision, Delegate):
            member = pool.get(decision.member)
            assert member is not None
            try:
                result = delegate(member, decision.task)
            except FATAL:
                raise
            except Exception as exc:  # the member failed; the orchestrator may react to it
                error = f"{type(exc).__name__}: {exc}"
        state = record(state, decision, verdict, result=result, error=error, raw=raw, limits=limits)
        if on_step is not None:
            on_step(state)
    return state


async def arun_orchestration(
    task: str,
    pool: Pool,
    *,
    decide: ADecide,
    delegate: ARun,
    limits: OrchestrationLimits | None = None,
    state: OrchestrationState | None = None,
    on_step: Callable[[OrchestrationState], None] | None = None,
) -> OrchestrationState:
    """The asyncio variant of :func:`run_orchestration` (same arguments, awaitable callables)."""
    limits = limits or OrchestrationLimits()
    state = state or OrchestrationState(task=task)
    while state.outcome is None:
        decision, refused, raw = _decision_of(await decide(decision_prompt(state, pool, limits)))
        verdict = refused or validate(decision, pool, state, limits)  # type: ignore[arg-type]
        result = error = None
        if verdict.ok and isinstance(decision, Delegate):
            member = pool.get(decision.member)
            assert member is not None
            try:
                result = await delegate(member, decision.task)
            except FATAL:
                raise
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        state = record(state, decision, verdict, result=result, error=error, raw=raw, limits=limits)
        if on_step is not None:
            on_step(state)
    return state
