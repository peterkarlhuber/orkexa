"""Orchestration protocol (ADR 0010): decisions, validation, record, replay and the reference driver."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from pydantic import ValidationError
import pytest

from orkexa.dynamic import BudgetExceededError
from orkexa.orchestration import (
    DecisionEnvelope,
    Delegate,
    Finish,
    Member,
    OrchestrationLimits,
    OrchestrationState,
    Pool,
    arun_orchestration,
    decision_prompt,
    parse_decision,
    replay,
    run_orchestration,
    validate,
)

POOL = Pool(
    members=(
        Member(id="research", name="Research", description="Finds facts", capabilities=("research",)),
        Member(id="write", name="Writer", contract=("text", "title")),
    )
)
LIMITS = OrchestrationLimits(max_decisions=5)


def scripted(*answers: Any) -> tuple[Any, list[str]]:
    prompts: list[str] = []
    queue = list(answers)

    def decide(prompt: str) -> Any:
        prompts.append(prompt)
        return queue.pop(0)

    return decide, prompts


def test_a_delegation_then_a_finish() -> None:
    decide, prompts = scripted(
        Delegate(member="research", task="Find three facts about otters", reason="needs facts"),
        {"action": "finish", "answer": "Otters hold hands.", "reason": "done"},
    )
    calls: list[tuple[str, str]] = []

    def delegate(member: Member, task: str) -> str:
        calls.append((member.id, task))
        return "1. They hold hands."

    state = run_orchestration("Write about otters", POOL, decide=decide, delegate=delegate, limits=LIMITS)
    assert calls == [("research", "Find three facts about otters")]
    assert state.outcome == "finished" and state.answer == "Otters hold hands."
    assert [step.verdict.ok for step in state.steps] == [True, True]
    assert state.steps[0].result == "1. They hold hands."
    assert (
        "They hold hands" in prompts[1]
        and "id: write" in prompts[0]
        and "result fields: text, title" in prompts[0]
    )


def test_refused_decisions_are_recorded_and_explained() -> None:
    decide, prompts = scripted(
        {"action": "delegate", "member": "nobody", "task": "x"},
        "not json at all",
        {"action": "create", "spec": "a new agent"},  # creating is no decision of this protocol
        Finish(answer="ok"),
    )
    state = run_orchestration("t", POOL, decide=decide, delegate=lambda member, task: "", limits=LIMITS)
    codes = [step.verdict.code for step in state.steps]
    assert codes == [
        "orchestration.unknown_member",
        "orchestration.invalid_decision",
        "orchestration.invalid_decision",
        None,
    ]
    assert state.steps[1].raw == "not json at all"
    assert "not in the pool" in prompts[1] and "Invalid answer" in prompts[2]


def test_the_decision_limit_stops_and_the_last_prompt_says_so() -> None:
    decide, prompts = scripted(*[Delegate(member="research", task=f"step {i}") for i in range(3)])
    state = run_orchestration(
        "t", POOL, decide=decide, delegate=lambda m, t: "r", limits=OrchestrationLimits(max_decisions=3)
    )
    assert state.outcome == "stopped" and state.stop_code == "orchestration.decision_limit"
    assert len(state.steps) == 3 and "This is the last decision" in prompts[-1]


def test_member_limits_and_repeats_are_refused() -> None:
    limits = OrchestrationLimits(max_decisions=6, max_per_member=1)
    state = OrchestrationState(task="t")
    first = Delegate(member="research", task="a")
    assert validate(first, POOL, state, limits).ok
    from orkexa.orchestration import record

    state = record(state, first, validate(first, POOL, state, limits), result="r", limits=limits)
    assert (
        validate(Delegate(member="research", task="b"), POOL, state, limits).code
        == "orchestration.member_limit"
    )
    loose = OrchestrationLimits(max_decisions=6)
    assert (
        validate(Delegate(member="research", task=" a "), POOL, state, loose).code == "orchestration.repeated"
    )
    assert (
        validate(Delegate(member="research", task=""), POOL, state, loose).code == "orchestration.empty_task"
    )
    assert validate(Finish(answer="  "), POOL, state, loose).code == "orchestration.empty_answer"
    assert validate(Finish(answer="x" * 11), POOL, state, OrchestrationLimits(max_answer_chars=10)).code == (
        "orchestration.answer_too_long"
    )


def test_a_failing_member_is_shown_to_the_model_but_budget_errors_end_it() -> None:
    decide, prompts = scripted(Delegate(member="write", task="draft"), Finish(answer="gave up"))

    def broken(member: Member, task: str) -> str:
        raise RuntimeError("tool denied")

    state = run_orchestration("t", POOL, decide=decide, delegate=broken, limits=LIMITS)
    assert state.steps[0].error == "RuntimeError: tool denied" and "failed: RuntimeError" in prompts[1]

    def expensive(member: Member, task: str) -> str:
        raise BudgetExceededError("Budget used up")

    decide, _ = scripted(Delegate(member="write", task="draft"))
    with pytest.raises(BudgetExceededError):
        run_orchestration("t", POOL, decide=decide, delegate=expensive, limits=LIMITS)


def test_the_record_is_json_and_replays_without_model_calls() -> None:
    decide, _ = scripted(
        Delegate(member="research", task="facts"), Delegate(member="write", task="text"), Finish(answer="A")
    )
    saved: list[OrchestrationState] = []
    state = run_orchestration(
        "t", POOL, decide=decide, delegate=lambda m, t: f"{m.id} done", limits=LIMITS, on_step=saved.append
    )
    assert len(saved) == 3
    restored = OrchestrationState.model_validate(json.loads(state.model_dump_json()))
    assert restored == state
    assert [verdict.ok for verdict in replay(restored, POOL, LIMITS)] == [True, True, True]
    # The same record under a smaller pool: the writer would have been refused.
    small = Pool(members=(POOL.members[0],))
    assert [verdict.code for verdict in replay(restored, small, LIMITS)] == [
        None,
        "orchestration.unknown_member",
        None,
    ]
    with pytest.raises(ValueError, match="version"):
        replay(restored.model_copy(update={"version": 99}), POOL, LIMITS)


def test_a_recorded_state_continues() -> None:
    decide, _ = scripted(Delegate(member="research", task="facts"))
    partial: list[OrchestrationState] = []

    def stop_after_one(state: OrchestrationState) -> None:
        partial.append(state)
        raise KeyboardInterrupt  # the process ends here

    with pytest.raises(KeyboardInterrupt):
        run_orchestration(
            "t", POOL, decide=decide, delegate=lambda m, t: "r", limits=LIMITS, on_step=stop_after_one
        )
    decide, _ = scripted(Finish(answer="done"))
    state = run_orchestration(
        "t", POOL, decide=decide, delegate=lambda m, t: "r", limits=LIMITS, state=partial[0]
    )
    assert state.outcome == "finished" and len(state.steps) == 2


def test_parse_decision_accepts_envelopes_and_text() -> None:
    envelope = DecisionEnvelope(decision=Finish(answer="x"))
    assert parse_decision(envelope) == Finish(answer="x")
    assert parse_decision('{"decision": {"action": "delegate", "member": "m", "task": "t"}}') == Delegate(
        member="m", task="t"
    )
    with pytest.raises(ValueError):
        parse_decision({"action": "delegate"})


def test_pools_need_unique_members() -> None:
    with pytest.raises(ValidationError):
        Pool(members=(Member(id="a", name="A"), Member(id="a", name="B")))
    with pytest.raises(ValidationError):
        Pool(members=())


def test_the_async_driver() -> None:
    answers = [Delegate(member="research", task="facts"), Finish(answer="done")]

    async def decide(prompt: str) -> Any:
        return answers.pop(0)

    async def delegate(member: Member, task: str) -> str:
        await asyncio.sleep(0)
        return "facts"

    state = asyncio.run(arun_orchestration("t", POOL, decide=decide, delegate=delegate, limits=LIMITS))
    assert state.outcome == "finished" and state.steps[0].result == "facts"


def test_the_first_prompt_lists_the_pool_and_the_limit() -> None:
    prompt = decision_prompt(OrchestrationState(task="Solve it"), POOL, LIMITS)
    assert (
        prompt.startswith("Task:\nSolve it")
        and "Decisions left: 5" in prompt
        and "capabilities: research" in prompt
    )
