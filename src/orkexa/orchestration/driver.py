"""Reference driver: ask for a decision, validate it, delegate, record; until finish, a limit or a pause."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
import json
from typing import Any

from ..dynamic.errors import BudgetExceededError
from .protocol import (
    AnyDecision,
    Delegate,
    Member,
    OrchestrationLimits,
    OrchestrationState,
    Outcome,
    Parallel,
    Pool,
    Staff,
    Verdict,
    complete_pending,
    parse_decision,
    record,
    validate,
)

__all__ = [
    "Delegation",
    "DelegationEnded",
    "DelegationSuspended",
    "arun_orchestration",
    "decision_prompt",
    "run_orchestration",
]

#: Errors of ``delegate`` that end the orchestration instead of being shown to the model.
FATAL: tuple[type[BaseException], ...] = (BudgetExceededError,)

#: How much of a member's result the next prompt repeats (the record keeps all of it).
RESULT_PREVIEW_CHARS = 4000


@dataclass(frozen=True)
class Delegation:
    """One assignment the application runs (see ``execute`` of :func:`run_orchestration`).

    Attributes:
        member: The pool member.
        task: Its task.
        values: Values for the member's parameters (validated against :attr:`Member.parameters`; the
            application checks them again against the member's own input rules).
        step: The index of the decision this assignment belongs to.
        slot: The position of the assignment within the decision (0 for :class:`Delegate`).
        ref: When the assignment waited before: the reference it raised with :class:`DelegationSuspended`.
            The application continues that execution and must not start a new one.
    """

    member: Member
    task: str
    values: dict[str, Any] = field(default_factory=dict)
    step: int = 0
    slot: int = 0
    ref: str | None = None


class DelegationSuspended(Exception):
    """Raised by ``execute``: the member waits for something outside (e.g. a person's approval).

    The orchestration pauses (:attr:`OrchestrationState.pending`) and continues later with ``ref``.
    """

    def __init__(self, ref: str, message: str = "") -> None:
        super().__init__(message or f"waiting ({ref})")
        self.ref = ref


class DelegationEnded(Exception):
    """Raised by ``execute``: the member ended without a result for a known reason.

    ``status`` is ``failed``, ``rejected`` (a person rejected an approval), ``cancelled`` or ``expired``
    (an approval was not given in time). The orchestration continues and shows the status to the model.
    """

    def __init__(self, status: str, message: str = "", ref: str | None = None) -> None:
        if status not in ("failed", "rejected", "cancelled", "expired"):
            raise ValueError(f"unknown delegation status {status!r}")
        super().__init__(message or status)
        self.status = status
        self.message = message or status
        self.ref = ref


Decide = Callable[[str], Any]
Run = Callable[[Member, str], str]
Execute = Callable[[Delegation], str]
ADecide = Callable[[str], Awaitable[Any]]
ARun = Callable[[Member, str], Awaitable[str]]
AExecute = Callable[[Delegation], Awaitable[str]]


def _member_line(member: Member) -> str:
    parts = [f"- id: {member.id} | {member.kind}: {member.name}"]
    if member.description:
        parts.append(f"  does: {member.description}")
    if member.capabilities:
        parts.append(f"  capabilities: {', '.join(member.capabilities)}")
    if member.contract:
        parts.append(f"  result fields: {', '.join(member.contract)}")
    for parameter in member.parameters:
        kind = parameter.type if not parameter.choices else f"one of {', '.join(parameter.choices)}"
        need = "required" if parameter.required else "optional"
        about = f": {parameter.description}" if parameter.description else ""
        parts.append(f"  parameter {parameter.name} ({kind}, {need}){about}")
    return "\n".join(parts)


def _outcome_text(outcome: Outcome) -> str:
    if outcome.status == "succeeded":
        return f"result:\n{(outcome.result or '')[:RESULT_PREVIEW_CHARS]}"
    return f"{outcome.status}: {outcome.error or outcome.status}"


def _assignment_text(member: str, task: str, values: dict[str, Any]) -> str:
    shown = f" (values: {json.dumps(values, ensure_ascii=False, default=str)})" if values else ""
    return f"{member}{shown}: {task}"


def decision_prompt(state: OrchestrationState, pool: Pool, limits: OrchestrationLimits) -> str:
    """The prompt for the next decision: task, pool, team, everything so far and what is left."""
    lines = [
        "Task:",
        state.task,
        "",
        "Pool (you can only delegate to these members, by id):",
        *(_member_line(member) for member in pool.members),
        "",
    ]
    if limits.staffing:
        team = state.team(pool, limits)
        lines += [
            f"Team now: {', '.join(team) if team else 'nobody yet'} (at most {limits.max_team} members; "
            "add or remove pool members with staff, delegate only to team members).",
            "",
        ]
    if state.steps:
        lines.append("So far:")
        for step in state.steps:
            decision = step.decision
            number = step.index + 1
            if decision is None:
                lines.append(f"{number}. Invalid answer, refused: {step.verdict.message}")
            elif not step.verdict.ok:
                lines.append(f"{number}. Refused ({step.verdict.code}): {step.verdict.message}")
            elif isinstance(decision, Staff):
                change = [f"added {', '.join(decision.add)}"] if decision.add else []
                change += [f"removed {', '.join(decision.remove)}"] if decision.remove else []
                lines.append(f"{number}. Team changed: {'; '.join(change)}")
            elif isinstance(decision, Delegate):
                outcome = (
                    _outcome_text(step.outcomes[0])
                    if step.outcomes
                    else (
                        f"failed: {step.error}"
                        if step.error is not None
                        else f"result:\n{(step.result or '')[:RESULT_PREVIEW_CHARS]}"
                    )
                )
                shown = _assignment_text(decision.member, decision.task, decision.values)
                lines.append(f"{number}. Delegated to {shown}\n   {outcome}")
            elif isinstance(decision, Parallel):
                lines.append(f"{number}. Delegated in parallel:")
                for item, ended in zip(decision.parts(), step.outcomes, strict=False):
                    lines.append(f"   - {_assignment_text(item.member, item.task, item.values)}")
                    lines.append(f"     {_outcome_text(ended)}")
        lines.append("")
    left = limits.max_decisions - len(state.steps)
    choices = ["delegate to a member with a complete, self-contained task"]
    if limits.max_parallel > 1:
        choices.append(f"delegate in parallel to 2 to {limits.max_parallel} different members")
    if limits.staffing:
        choices.append("change the team with staff")
    lines.append(
        f"Decisions left: {left}. Answer with one decision: {', '.join(choices)}, "
        "or finish with the final answer." + (" This is the last decision: finish now." if left == 1 else "")
    )
    return "\n".join(lines)


def _decision_of(answer: Any) -> tuple[AnyDecision | None, Verdict | None, str | None]:
    try:
        return parse_decision(answer), None, None
    except ValueError as exc:
        raw = answer if isinstance(answer, str) else json.dumps(answer, default=str)[:2000]
        return None, Verdict(code="orchestration.invalid_decision", message=str(exc)), raw


def _adapt(delegate: Any, execute: Any) -> Any:
    if (delegate is None) == (execute is None):
        raise TypeError("pass exactly one of delegate and execute")
    if execute is not None:
        return execute
    return lambda item: delegate(item.member, item.task)


def _assignments(decision: AnyDecision, pool: Pool, step: int, refs: list[str | None]) -> list[Delegation]:
    assert isinstance(decision, (Delegate, Parallel))
    found = []
    for slot, item in enumerate(decision.parts()):
        member = pool.get(item.member)
        assert member is not None
        found.append(Delegation(member, item.task, dict(item.values), step, slot, refs[slot]))
    return found


def _ended(item: Delegation, exc: BaseException) -> Outcome:
    if isinstance(exc, DelegationSuspended):
        return Outcome(member=item.member.id, status="waiting", ref=exc.ref)
    if isinstance(exc, DelegationEnded):
        return Outcome(member=item.member.id, status=exc.status, error=exc.message, ref=exc.ref or item.ref)  # type: ignore[arg-type]
    return Outcome(member=item.member.id, status="failed", error=f"{type(exc).__name__}: {exc}", ref=item.ref)


def _run_one(execute: Execute, item: Delegation, stop_on: tuple[type[BaseException], ...]) -> Outcome:
    try:
        result = execute(item)
    except (*FATAL, *stop_on):
        raise
    except Exception as exc:  # the member failed or waits; the orchestrator may react to it
        return _ended(item, exc)
    return Outcome(member=item.member.id, status="succeeded", result=result, ref=item.ref)


async def _arun_one(execute: AExecute, item: Delegation, stop_on: tuple[type[BaseException], ...]) -> Outcome:
    try:
        result = await execute(item)
    except (*FATAL, *stop_on):
        raise
    except Exception as exc:
        return _ended(item, exc)
    return Outcome(member=item.member.id, status="succeeded", result=result, ref=item.ref)


async def _agather(
    execute: AExecute, items: list[Delegation], stop_on: tuple[type[BaseException], ...]
) -> list[Outcome]:
    """All assignments at the same time; a fatal error cancels the others and is raised."""
    if len(items) == 1:
        return [await _arun_one(execute, items[0], stop_on)]
    tasks = [asyncio.ensure_future(_arun_one(execute, item, stop_on)) for item in items]
    try:
        return list(await asyncio.gather(*tasks))
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


def _resume_items(
    state: OrchestrationState, pool: Pool
) -> tuple[list[int], list[Delegation], dict[int, Outcome]]:
    """The waiting assignments of the pending delegation, with the references they continue.

    A waiting member that is no longer in ``pool`` fails (``orchestration.unknown_member``)
    instead of running.
    """
    pending = state.pending
    assert pending is not None
    slots, items, gone = [], [], {}
    assignments = pending.decision.parts()  # type: ignore[union-attr]
    for slot in pending.waiting():
        item, outcome = assignments[slot], pending.outcomes[slot]
        member = pool.get(item.member)
        if member is None:
            gone[slot] = Outcome(
                member=item.member, status="failed", error="orchestration.unknown_member", ref=outcome.ref
            )
            continue
        slots.append(slot)
        items.append(Delegation(member, item.task, dict(item.values), pending.index, slot, outcome.ref))
    return slots, items, gone


def _merge(
    state: OrchestrationState, slots: list[int], outcomes: list[Outcome], gone: dict[int, Outcome]
) -> tuple[Outcome, ...]:
    assert state.pending is not None
    merged = list(state.pending.outcomes)
    for slot, outcome in zip(slots, outcomes, strict=True):
        merged[slot] = outcome
    for slot, outcome in gone.items():
        merged[slot] = outcome
    return tuple(merged)


def run_orchestration(
    task: str,
    pool: Pool,
    *,
    decide: Decide,
    delegate: Run | None = None,
    execute: Execute | None = None,
    limits: OrchestrationLimits | None = None,
    state: OrchestrationState | None = None,
    on_step: Callable[[OrchestrationState], None] | None = None,
    stop_on: tuple[type[BaseException], ...] = (),
) -> OrchestrationState:
    """Run an orchestration until it ends or pauses.

    Args:
        task: The task of the orchestrator.
        pool: The members it may choose.
        decide: Gets the prompt (:func:`decision_prompt`) and returns a decision, an
            envelope, a dict or JSON text.
        delegate: Runs ``member`` with ``task`` and returns its result as text (protocol version 1
            callers; parameters and pauses need ``execute``).
        execute: Runs one :class:`Delegation` and returns its result as text; raises
            :class:`DelegationSuspended` when the member waits and :class:`DelegationEnded` for a known
            end without result. Any other exception is recorded as ``failed`` (the orchestration
            continues). Pass exactly one of ``delegate`` and ``execute``. The assignments of a
            :class:`Parallel` decision run one after the other here; :func:`arun_orchestration` runs
            them at the same time.
        limits: Hard limits (default :class:`OrchestrationLimits`).
        state: A recorded state to continue (after a restart, or a paused one: its waiting
            assignments continue first, with the reference they waited with; nothing is asked again).
        on_step: Called with the new state after every step and on a pause (to save or show it).
        stop_on: Further exceptions of the member that end the orchestration (e.g. the
            application's cancellation), in addition to budget errors.

    Returns:
        The state: ended (``outcome`` set) or paused (``paused``; continue it with the same call later).
    """
    run = _adapt(delegate, execute)
    limits = limits or OrchestrationLimits()
    state = state or OrchestrationState(task=task)
    while state.outcome is None:
        if state.pending is not None:
            slots, items, gone = _resume_items(state, pool)
            outcomes = [_run_one(run, item, stop_on) for item in items]
            state = complete_pending(state, _merge(state, slots, outcomes, gone), limits=limits)
        else:
            decision, refused, raw = _decision_of(decide(decision_prompt(state, pool, limits)))
            verdict = refused or validate(decision, pool, state, limits)  # type: ignore[arg-type]
            found: tuple[Outcome, ...] = ()
            if verdict.ok and isinstance(decision, (Delegate, Parallel)):
                items = _assignments(decision, pool, len(state.steps), [None] * len(decision.parts()))
                found = tuple(_run_one(run, item, stop_on) for item in items)
            state = record(state, decision, verdict, raw=raw, outcomes=found, limits=limits)
        if on_step is not None:
            on_step(state)
        if state.paused:
            break
    return state


async def arun_orchestration(
    task: str,
    pool: Pool,
    *,
    decide: ADecide,
    delegate: ARun | None = None,
    execute: AExecute | None = None,
    limits: OrchestrationLimits | None = None,
    state: OrchestrationState | None = None,
    on_step: Callable[[OrchestrationState], None] | None = None,
    stop_on: tuple[type[BaseException], ...] = (),
) -> OrchestrationState:
    """The asyncio variant of :func:`run_orchestration`; parallel assignments run at the same time."""
    run = _adapt(delegate, execute)
    limits = limits or OrchestrationLimits()
    state = state or OrchestrationState(task=task)
    while state.outcome is None:
        if state.pending is not None:
            slots, items, gone = _resume_items(state, pool)
            outcomes = await _agather(run, items, stop_on) if items else []
            state = complete_pending(state, _merge(state, slots, outcomes, gone), limits=limits)
        else:
            decision, refused, raw = _decision_of(await decide(decision_prompt(state, pool, limits)))
            verdict = refused or validate(decision, pool, state, limits)  # type: ignore[arg-type]
            found: tuple[Outcome, ...] = ()
            if verdict.ok and isinstance(decision, (Delegate, Parallel)):
                items = _assignments(decision, pool, len(state.steps), [None] * len(decision.parts()))
                found = tuple(await _agather(run, items, stop_on))
            state = record(state, decision, verdict, raw=raw, outcomes=found, limits=limits)
        if on_step is not None:
            on_step(state)
        if state.paused:
            break
    return state
