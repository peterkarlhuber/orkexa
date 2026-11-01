"""Decisions, their validation and the record of an orchestration (pure, no I/O)."""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

__all__ = [
    "PROTOCOL_VERSION",
    "READABLE_VERSIONS",
    "AnyDecision",
    "Assignment",
    "Decision",
    "DecisionEnvelope",
    "Delegate",
    "Finish",
    "Member",
    "OrchestrationLimits",
    "OrchestrationState",
    "Outcome",
    "OutcomeStatus",
    "Parallel",
    "Parameter",
    "Pending",
    "Pool",
    "Staff",
    "Step",
    "Verdict",
    "complete_pending",
    "parse_decision",
    "record",
    "replay",
    "validate",
]

PROTOCOL_VERSION = 2
"""Version of the recorded format. Version 2 adds parallel delegation, a team that changes during the
orchestration, parameters, delegation outcomes and pending (paused) delegations; version 1 records are
valid version 2 records and replay unchanged."""

READABLE_VERSIONS = (1, 2)

MAX_TASK_CHARS = 20_000
MAX_VALUE_CHARS = 4000
MAX_PARALLEL = 10
MAX_TEAM = 20

OutcomeStatus = Literal["succeeded", "failed", "rejected", "cancelled", "expired", "waiting"]
"""How one delegation ended; ``waiting`` while the member waits for something outside (an approval)."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Parameter(_Frozen):
    """A value a member accepts besides its task (AO-5 level 2: configure within allowed parameters).

    Attributes:
        name: The key the orchestrator uses in ``values``.
        type: ``text``, ``number``, ``boolean`` or ``choice``.
        required: Whether the member needs it.
        description: What it means, shown to the model.
        choices: The allowed values of a ``choice``.
    """

    name: str = Field(min_length=1, max_length=100)
    type: Literal["text", "number", "boolean", "choice"] = "text"
    required: bool = False
    description: str = Field(default="", max_length=500)
    choices: tuple[str, ...] = ()


class Member(_Frozen):
    """One pool member: a team (workflow) or an agent that already exists.

    Attributes:
        id: Stable identifier the orchestrator names in a :class:`Delegate`.
        name: Display name, shown to the model.
        description: What the member does, shown to the model.
        kind: ``team`` (a workflow) or ``agent``.
        capabilities: Short tags of what the member can do.
        contract: Field names the member's result promises (empty: free text).
        parameters: Values the orchestrator may set when it delegates (empty: only the task).
    """

    id: str = Field(min_length=1, max_length=200)
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)
    kind: Literal["team", "agent"] = "team"
    capabilities: tuple[str, ...] = ()
    contract: tuple[str, ...] = ()
    parameters: tuple[Parameter, ...] = ()


class Pool(_Frozen):
    """The members an orchestrator may choose from (at least one, unique ids)."""

    members: tuple[Member, ...] = Field(min_length=1, max_length=50)

    def model_post_init(self, __context: Any) -> None:
        ids = [member.id for member in self.members]
        if len(set(ids)) != len(ids):
            raise ValueError("pool members need unique ids")

    def get(self, member_id: str) -> Member | None:
        return next((member for member in self.members if member.id == member_id), None)


class OrchestrationLimits(_Frozen):
    """Hard limits of one orchestration.

    Attributes:
        max_decisions: Decisions the model may make, rejected ones included (1 to 50).
        max_per_member: Delegations to one member (``None``: only ``max_decisions`` limits).
        max_answer_chars: Longest final answer.
        max_parallel: Members one :class:`Parallel` decision may start at once (1: no parallel delegation).
        staffing: The orchestrator assembles its team: it delegates only to members it added with
            :class:`Staff` and may remove them again. ``False``: every pool member is in the team.
        max_team: Largest team at any time (with ``staffing``).
    """

    max_decisions: int = Field(default=8, ge=1, le=50)
    max_per_member: int | None = Field(default=None, ge=1, le=50)
    max_answer_chars: int = Field(default=50_000, ge=1, le=1_000_000)
    max_parallel: int = Field(default=1, ge=1, le=MAX_PARALLEL)
    staffing: bool = False
    max_team: int = Field(default=5, ge=1, le=MAX_TEAM)


class Assignment(_Frozen):
    """One member's part of a :class:`Parallel` decision."""

    member: str = Field(description="The id of the pool member.")
    task: str = Field(description="A complete task for the member; it does not see this conversation.")
    values: dict[str, Any] = Field(default_factory=dict, description="Values for the member's parameters.")


class Delegate(_Frozen):
    """Give one member a complete, self-contained task."""

    action: Literal["delegate"] = "delegate"
    member: str = Field(description="The id of the pool member that should work next.")
    task: str = Field(description="A complete task for the member; it does not see this conversation.")
    values: dict[str, Any] = Field(
        default_factory=dict, description="Values for the member's parameters (only the ones it lists)."
    )
    reason: str = Field(default="", description="Why this member, in one or two sentences.")

    def parts(self) -> tuple[Assignment, ...]:
        return (Assignment(member=self.member, task=self.task, values=self.values),)


class Parallel(_Frozen):
    """Give several members a task each; they work at the same time."""

    action: Literal["parallel"] = "parallel"
    assignments: tuple[Assignment, ...] = Field(
        min_length=2, max_length=MAX_PARALLEL, description="Two or more members with a task each."
    )
    reason: str = Field(default="", description="Why these members, in one or two sentences.")

    def parts(self) -> tuple[Assignment, ...]:
        return self.assignments


class Staff(_Frozen):
    """Change the team: add members of the pool, remove members that are no longer needed."""

    action: Literal["staff"] = "staff"
    add: tuple[str, ...] = Field(default=(), description="Pool member ids that join the team.")
    remove: tuple[str, ...] = Field(default=(), description="Team member ids that leave the team.")
    reason: str = Field(default="", description="Why the team changes, in one or two sentences.")


class Finish(_Frozen):
    """End the orchestration with the final answer."""

    action: Literal["finish"] = "finish"
    answer: str = Field(description="The final answer, combining the members' results.")
    reason: str = Field(default="", description="Why the task is done, in one or two sentences.")


Decision = Annotated[Delegate | Parallel | Staff | Finish, Field(discriminator="action")]
"""One decision of the orchestrator."""

AnyDecision = Delegate | Parallel | Staff | Finish

_DECISION: TypeAdapter[AnyDecision] = TypeAdapter(Decision)


class DecisionEnvelope(BaseModel):
    """The structured output a model gives for one decision."""

    decision: Decision


class Verdict(_Frozen):
    """The result of :func:`validate`: ``code`` is ``None`` when the decision is accepted."""

    code: str | None = None
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.code is None


class Outcome(_Frozen):
    """How one delegated assignment ended (or that it still waits).

    Attributes:
        member: The member's id.
        status: :data:`OutcomeStatus`.
        result: The member's result as text (``succeeded``).
        error: Why it did not succeed, as a code or message.
        ref: The application's reference of the member's execution (e.g. a child run id); a ``waiting``
            delegation continues this execution, it is never started again.
    """

    member: str
    status: OutcomeStatus
    result: str | None = None
    error: str | None = None
    ref: str | None = None


class Step(_Frozen):
    """One recorded decision with its verdict and, for accepted delegations, their outcomes."""

    index: int
    decision: AnyDecision | None = Field(discriminator="action")
    verdict: Verdict
    #: What the member returned (an accepted :class:`Delegate`), as text.
    result: str | None = None
    #: Why the member failed (an accepted :class:`Delegate`), as a code or message.
    error: str | None = None
    #: The raw text when the model's answer was no valid decision.
    raw: str | None = None
    #: One outcome per assignment of an accepted :class:`Delegate` or :class:`Parallel` (version 2).
    outcomes: tuple[Outcome, ...] = ()


class Pending(_Frozen):
    """An accepted delegation whose members have not all ended: some wait (e.g. for a person's approval).

    The decision is not asked again: continuing the orchestration continues exactly these assignments.
    """

    index: int
    decision: AnyDecision = Field(discriminator="action")
    outcomes: tuple[Outcome, ...]

    def waiting(self) -> list[int]:
        return [index for index, outcome in enumerate(self.outcomes) if outcome.status == "waiting"]


class OrchestrationState(_Frozen):
    """The complete, JSON-serialisable record of one orchestration."""

    version: int = PROTOCOL_VERSION
    task: str
    steps: tuple[Step, ...] = ()
    #: ``finished`` with an answer, or ``stopped`` with a ``stop_code``; ``None`` while running or paused.
    outcome: Literal["finished", "stopped"] | None = None
    answer: str | None = None
    stop_code: str | None = None
    #: A delegation that waits (the orchestration is paused); ``None`` otherwise.
    pending: Pending | None = None

    def delegations(self, member_id: str | None = None) -> list[Step]:
        """Accepted steps that delegated (to ``member_id``, when given), oldest first."""
        return [
            step
            for step in self.steps
            if step.verdict.ok
            and isinstance(step.decision, (Delegate, Parallel))
            and (member_id is None or any(item.member == member_id for item in step.decision.parts()))
        ]

    def team(self, pool: Pool, limits: OrchestrationLimits) -> tuple[str, ...]:
        """The members the orchestrator may delegate to now: the pool, or (staffing) its current team."""
        if not limits.staffing:
            return tuple(member.id for member in pool.members)
        team: list[str] = []
        for step in self.steps:
            if step.verdict.ok and isinstance(step.decision, Staff):
                team = [member for member in team if member not in step.decision.remove]
                team += [member for member in step.decision.add if member not in team]
        return tuple(team)

    @property
    def paused(self) -> bool:
        return self.pending is not None and self.outcome is None


def parse_decision(value: Any) -> AnyDecision:
    """A decision from a model answer: a decision, an envelope, a dict or JSON text.

    Raises:
        ValueError: The value is no valid decision.
    """
    if isinstance(value, (Delegate, Parallel, Staff, Finish)):
        return value
    if isinstance(value, DecisionEnvelope):
        return value.decision
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError as exc:
            raise ValueError("the answer is no JSON decision") from exc
    if isinstance(value, dict) and "decision" in value and "action" not in value:
        value = value["decision"]
    try:
        return _DECISION.validate_python(value)
    except ValidationError as exc:
        raise ValueError(f"the answer is no valid decision: {exc.errors()[0]['msg']}") from exc


def _reject(code: str, message: str) -> Verdict:
    return Verdict(code=code, message=message)


def _value_problem(member: Member, values: dict[str, Any]) -> Verdict | None:
    """Values only for the member's parameters, of the right kind; the application checks them again."""
    known = {parameter.name: parameter for parameter in member.parameters}
    for key, value in values.items():
        parameter = known.get(key)
        if parameter is None:
            allowed = ", ".join(known) or "none"
            return _reject(
                "orchestration.unknown_parameter",
                f"{member.name} has no parameter {key!r} (parameters: {allowed})",
            )
        if parameter.type == "number" and (isinstance(value, bool) or not isinstance(value, (int, float))):
            return _reject("orchestration.invalid_value", f"{key} of {member.name} must be a number")
        if parameter.type == "boolean" and not isinstance(value, bool):
            return _reject("orchestration.invalid_value", f"{key} of {member.name} must be true or false")
        if parameter.type in ("text", "choice") and not isinstance(value, str):
            return _reject("orchestration.invalid_value", f"{key} of {member.name} must be text")
        if parameter.type == "choice" and parameter.choices and value not in parameter.choices:
            return _reject(
                "orchestration.invalid_value",
                f"{key} of {member.name} must be one of {', '.join(parameter.choices)}",
            )
        if isinstance(value, str) and len(value) > MAX_VALUE_CHARS:
            return _reject(
                "orchestration.invalid_value", f"{key} of {member.name} exceeds {MAX_VALUE_CHARS} characters"
            )
    return None


def _assignment_problem(
    item: Assignment,
    pool: Pool,
    state: OrchestrationState,
    limits: OrchestrationLimits,
    team: tuple[str, ...],
) -> Verdict | None:
    member = pool.get(item.member)
    if member is None:
        known = ", ".join(member.id for member in pool.members)
        return _reject(
            "orchestration.unknown_member", f"{item.member!r} is not in the pool (choose one of {known})"
        )
    if item.member not in team:
        shown = ", ".join(team) or "nobody yet"
        return _reject(
            "orchestration.not_in_team",
            f"{member.name} is not in the team (team: {shown}); add it with staff first",
        )
    if not item.task.strip():
        return _reject("orchestration.empty_task", "The task is empty")
    if len(item.task) > MAX_TASK_CHARS:
        return _reject("orchestration.task_too_long", f"The task exceeds {MAX_TASK_CHARS} characters")
    done = state.delegations(member.id)
    if limits.max_per_member is not None and len(done) >= limits.max_per_member:
        return _reject(
            "orchestration.member_limit", f"{member.name} may work at most {limits.max_per_member} times"
        )
    return _value_problem(member, item.values)


def validate(
    decision: AnyDecision, pool: Pool, state: OrchestrationState, limits: OrchestrationLimits
) -> Verdict:
    """Whether ``decision`` may be carried out next (pure; never raises for a bad decision).

    Codes: ``orchestration.finished``, ``orchestration.paused``, ``orchestration.decision_limit``,
    ``orchestration.unknown_member``, ``orchestration.not_in_team``, ``orchestration.member_limit``,
    ``orchestration.empty_task``, ``orchestration.task_too_long``, ``orchestration.repeated``,
    ``orchestration.unknown_parameter``, ``orchestration.invalid_value``,
    ``orchestration.parallel_disabled``, ``orchestration.parallel_limit``,
    ``orchestration.parallel_duplicate``, ``orchestration.staffing_disabled``,
    ``orchestration.empty_staffing``, ``orchestration.already_in_team``, ``orchestration.team_limit``,
    ``orchestration.answer_too_long``, ``orchestration.empty_answer``.
    """
    if state.outcome is not None:
        return _reject("orchestration.finished", "The orchestration has already ended")
    if state.pending is not None:
        return _reject("orchestration.paused", "A delegation is still waiting; it continues first")
    if len(state.steps) >= limits.max_decisions:
        return _reject("orchestration.decision_limit", f"No decision left (limit {limits.max_decisions})")
    if isinstance(decision, Finish):
        if not decision.answer.strip():
            return _reject("orchestration.empty_answer", "The final answer is empty")
        if len(decision.answer) > limits.max_answer_chars:
            return _reject(
                "orchestration.answer_too_long", f"The answer exceeds {limits.max_answer_chars} characters"
            )
        return Verdict()
    team = state.team(pool, limits)
    if isinstance(decision, Staff):
        if not limits.staffing:
            return _reject("orchestration.staffing_disabled", "This orchestrator works with its whole pool")
        if not decision.add and not decision.remove:
            return _reject("orchestration.empty_staffing", "Add or remove at least one member")
        for member_id in decision.add:
            member = pool.get(member_id)
            if member is None:
                known = ", ".join(item.id for item in pool.members)
                return _reject(
                    "orchestration.unknown_member",
                    f"{member_id!r} is not in the pool (choose one of {known})",
                )
            if member_id in team and member_id not in decision.remove:
                return _reject("orchestration.already_in_team", f"{member.name} is already in the team")
        for member_id in decision.remove:
            if member_id not in team:
                return _reject("orchestration.not_in_team", f"{member_id!r} is not in the team")
        after = [member for member in team if member not in decision.remove]
        after += [member for member in decision.add if member not in after]
        if len(after) > limits.max_team:
            return _reject("orchestration.team_limit", f"The team may have at most {limits.max_team} members")
        return Verdict()
    if isinstance(decision, Parallel):
        if limits.max_parallel < 2:
            return _reject(
                "orchestration.parallel_disabled", "This orchestrator delegates to one member at a time"
            )
        if len(decision.parts()) > limits.max_parallel:
            return _reject(
                "orchestration.parallel_limit",
                f"At most {limits.max_parallel} members may work at the same time",
            )
        members = [item.member for item in decision.parts()]
        if len(set(members)) != len(members):
            return _reject(
                "orchestration.parallel_duplicate", "Give every member one task in a parallel decision"
            )
        for item in decision.parts():
            problem = _assignment_problem(item, pool, state, limits, team)
            if problem is not None:
                return problem
        return Verdict()
    problem = _assignment_problem(decision.parts()[0], pool, state, limits, team)
    if problem is not None:
        return problem
    previous = state.delegations()
    if previous:
        last = previous[-1].decision
        if (
            isinstance(last, Delegate)
            and last.member == decision.member
            and last.task.strip() == decision.task.strip()
            and last.values == decision.values
        ):
            member = pool.get(decision.member)
            assert member is not None
            return _reject(
                "orchestration.repeated",
                f"{member.name} already got exactly this task; change the task or finish",
            )
    return Verdict()


def record(
    state: OrchestrationState,
    decision: AnyDecision | None,
    verdict: Verdict,
    *,
    result: str | None = None,
    error: str | None = None,
    raw: str | None = None,
    outcomes: tuple[Outcome, ...] = (),
    limits: OrchestrationLimits,
) -> OrchestrationState:
    """``state`` with one more step; a finish ends it, the decision limit stops it.

    ``outcomes`` with a ``waiting`` member pause the orchestration instead: the step is kept as
    :attr:`OrchestrationState.pending` and recorded only when it is completed (:func:`complete_pending`).
    A recorded step never has a ``waiting`` outcome.
    """
    if verdict.ok and any(outcome.status == "waiting" for outcome in outcomes):
        assert decision is not None
        pending = Pending(index=len(state.steps), decision=decision, outcomes=outcomes)
        return state.model_copy(update={"pending": pending})
    if outcomes and isinstance(decision, Delegate) and result is None and error is None:
        only = outcomes[0]
        result, error = only.result, (None if only.status == "succeeded" else only.error or only.status)
    step = Step(
        index=len(state.steps),
        decision=decision,
        verdict=verdict,
        result=result,
        error=error,
        raw=raw,
        outcomes=outcomes,
    )
    update: dict[str, Any] = {"steps": (*state.steps, step), "pending": None}
    if verdict.ok and isinstance(decision, Finish):
        update |= {"outcome": "finished", "answer": decision.answer}
    elif len(state.steps) + 1 >= limits.max_decisions:
        update |= {"outcome": "stopped", "stop_code": "orchestration.decision_limit"}
    return state.model_copy(update=update)


def complete_pending(
    state: OrchestrationState, outcomes: tuple[Outcome, ...], *, limits: OrchestrationLimits
) -> OrchestrationState:
    """``state`` with the pending delegation's ``outcomes`` updated (one per assignment, in order).

    While any of them still waits, the orchestration stays paused; otherwise the delegation is
    recorded as a step (with the decision, verdict and index it was accepted with).

    Raises:
        ValueError: Nothing is pending, or ``outcomes`` does not match the pending assignments.
    """
    pending = state.pending
    if pending is None:
        raise ValueError("no delegation is pending")
    if len(outcomes) != len(pending.outcomes) or any(
        new.member != old.member for new, old in zip(outcomes, pending.outcomes, strict=True)
    ):
        raise ValueError("the outcomes do not match the pending assignments")
    for old, new in zip(pending.outcomes, outcomes, strict=True):
        if old.status != "waiting" and new != old:
            raise ValueError("an ended assignment cannot change")
    if any(outcome.status == "waiting" for outcome in outcomes):
        return state.model_copy(update={"pending": pending.model_copy(update={"outcomes": outcomes})})
    unpaused = state.model_copy(update={"pending": None})
    return record(unpaused, pending.decision, Verdict(), outcomes=outcomes, limits=limits)


def replay(recorded: OrchestrationState, pool: Pool, limits: OrchestrationLimits) -> list[Verdict]:
    """Validate a recorded orchestration again, step by step, without any model call.

    Returns the verdict of every recorded step under ``pool`` and ``limits`` (which may
    differ from the original ones, e.g. to see what a smaller pool would have refused).
    A pending delegation at the end is validated too (its verdict comes last).

    Raises:
        ValueError: The record has a protocol version this library cannot read.
    """
    if recorded.version not in READABLE_VERSIONS:
        raise ValueError(
            f"protocol version {recorded.version} cannot be replayed (expected one of {READABLE_VERSIONS})"
        )
    state = OrchestrationState(task=recorded.task)
    verdicts: list[Verdict] = []
    for step in recorded.steps:
        verdict = step.verdict if step.decision is None else validate(step.decision, pool, state, limits)
        verdicts.append(verdict)
        state = record(
            state,
            step.decision,
            verdict,
            result=step.result,
            error=step.error,
            raw=step.raw,
            outcomes=step.outcomes,
            limits=limits,
        )
        if state.outcome is not None:
            return verdicts
    if recorded.pending is not None:
        verdicts.append(validate(recorded.pending.decision, pool, state, limits))
    return verdicts
