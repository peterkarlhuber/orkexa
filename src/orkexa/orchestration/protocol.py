"""Decisions, their validation and the record of an orchestration (pure, no I/O)."""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

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
    "parse_decision",
    "record",
    "replay",
    "validate",
]

PROTOCOL_VERSION = 1
"""Version of the recorded format; a replay refuses records of another version."""

MAX_TASK_CHARS = 20_000


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Member(_Frozen):
    """One pool member: a team (workflow) or an agent that already exists.

    Attributes:
        id: Stable identifier the orchestrator names in a :class:`Delegate`.
        name: Display name, shown to the model.
        description: What the member does, shown to the model.
        kind: ``team`` (a workflow) or ``agent``.
        capabilities: Short tags of what the member can do.
        contract: Field names the member's result promises (empty: free text).
    """

    id: str = Field(min_length=1, max_length=200)
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)
    kind: Literal["team", "agent"] = "team"
    capabilities: tuple[str, ...] = ()
    contract: tuple[str, ...] = ()


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
    """

    max_decisions: int = Field(default=8, ge=1, le=50)
    max_per_member: int | None = Field(default=None, ge=1, le=50)
    max_answer_chars: int = Field(default=50_000, ge=1, le=1_000_000)


class Delegate(_Frozen):
    """Give one member a complete, self-contained task."""

    action: Literal["delegate"] = "delegate"
    member: str = Field(description="The id of the pool member that should work next.")
    task: str = Field(description="A complete task for the member; it does not see this conversation.")
    reason: str = Field(default="", description="Why this member, in one or two sentences.")


class Finish(_Frozen):
    """End the orchestration with the final answer."""

    action: Literal["finish"] = "finish"
    answer: str = Field(description="The final answer, combining the members' results.")
    reason: str = Field(default="", description="Why the task is done, in one or two sentences.")


Decision = Annotated[Delegate | Finish, Field(discriminator="action")]
"""One decision of the orchestrator."""

_DECISION: TypeAdapter[Delegate | Finish] = TypeAdapter(Decision)


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


class Step(_Frozen):
    """One recorded decision with its verdict and, for an accepted delegation, the outcome."""

    index: int
    decision: Decision | None
    verdict: Verdict
    #: What the member returned (accepted delegations), as text.
    result: str | None = None
    #: Why the member failed (accepted delegations), as a code or message.
    error: str | None = None
    #: The raw text when the model's answer was no valid decision.
    raw: str | None = None


class OrchestrationState(_Frozen):
    """The complete, JSON-serialisable record of one orchestration."""

    version: int = PROTOCOL_VERSION
    task: str
    steps: tuple[Step, ...] = ()
    #: ``finished`` with an answer, or ``stopped`` with a ``stop_code``; ``None`` while running.
    outcome: Literal["finished", "stopped"] | None = None
    answer: str | None = None
    stop_code: str | None = None

    def delegations(self, member_id: str | None = None) -> list[Step]:
        return [
            step
            for step in self.steps
            if step.verdict.ok
            and isinstance(step.decision, Delegate)
            and (member_id is None or step.decision.member == member_id)
        ]


def parse_decision(value: Any) -> Delegate | Finish:
    """A decision from a model answer: a decision, an envelope, a dict or JSON text.

    Raises:
        ValueError: The value is no valid decision.
    """
    if isinstance(value, (Delegate, Finish)):
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


def validate(
    decision: Delegate | Finish, pool: Pool, state: OrchestrationState, limits: OrchestrationLimits
) -> Verdict:
    """Whether ``decision`` may be carried out next (pure; never raises for a bad decision).

    Codes: ``orchestration.finished``, ``orchestration.decision_limit``,
    ``orchestration.unknown_member``, ``orchestration.member_limit``,
    ``orchestration.empty_task``, ``orchestration.task_too_long``, ``orchestration.repeated``,
    ``orchestration.answer_too_long``, ``orchestration.empty_answer``.
    """
    if state.outcome is not None:
        return _reject("orchestration.finished", "The orchestration has already ended")
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
    member = pool.get(decision.member)
    if member is None:
        known = ", ".join(item.id for item in pool.members)
        return _reject(
            "orchestration.unknown_member", f"{decision.member!r} is not in the pool (choose one of {known})"
        )
    if not decision.task.strip():
        return _reject("orchestration.empty_task", "The task is empty")
    if len(decision.task) > MAX_TASK_CHARS:
        return _reject("orchestration.task_too_long", f"The task exceeds {MAX_TASK_CHARS} characters")
    done = state.delegations(member.id)
    if limits.max_per_member is not None and len(done) >= limits.max_per_member:
        return _reject(
            "orchestration.member_limit", f"{member.name} may work at most {limits.max_per_member} times"
        )
    previous = state.delegations()
    if previous:
        last = previous[-1].decision
        assert isinstance(last, Delegate)
        if last.member == decision.member and last.task.strip() == decision.task.strip():
            return _reject(
                "orchestration.repeated",
                f"{member.name} already got exactly this task; change the task or finish",
            )
    return Verdict()


def record(
    state: OrchestrationState,
    decision: Delegate | Finish | None,
    verdict: Verdict,
    *,
    result: str | None = None,
    error: str | None = None,
    raw: str | None = None,
    limits: OrchestrationLimits,
) -> OrchestrationState:
    """``state`` with one more step; a finish ends it, the decision limit stops it."""
    step = Step(
        index=len(state.steps), decision=decision, verdict=verdict, result=result, error=error, raw=raw
    )
    update: dict[str, Any] = {"steps": (*state.steps, step)}
    if verdict.ok and isinstance(decision, Finish):
        update |= {"outcome": "finished", "answer": decision.answer}
    elif len(state.steps) + 1 >= limits.max_decisions:
        update |= {"outcome": "stopped", "stop_code": "orchestration.decision_limit"}
    return state.model_copy(update=update)


def replay(recorded: OrchestrationState, pool: Pool, limits: OrchestrationLimits) -> list[Verdict]:
    """Validate a recorded orchestration again, step by step, without any model call.

    Returns the verdict of every recorded step under ``pool`` and ``limits`` (which may
    differ from the original ones, e.g. to see what a smaller pool would have refused).

    Raises:
        ValueError: The record has another protocol version.
    """
    if recorded.version != PROTOCOL_VERSION:
        raise ValueError(
            f"protocol version {recorded.version} cannot be replayed (expected {PROTOCOL_VERSION})"
        )
    state = OrchestrationState(task=recorded.task)
    verdicts: list[Verdict] = []
    for step in recorded.steps:
        verdict = step.verdict if step.decision is None else validate(step.decision, pool, state, limits)
        verdicts.append(verdict)
        state = record(
            state, step.decision, verdict, result=step.result, error=step.error, raw=step.raw, limits=limits
        )
        if state.outcome is not None:
            break
    return verdicts
