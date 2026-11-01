"""Orchestration protocol version 2 (ADR 0011): dynamic teams, parallel delegation, parameters and pauses."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import pytest

from orkexa.dynamic import BudgetExceededError
from orkexa.orchestration import (
    Delegate,
    Delegation,
    DelegationEnded,
    DelegationSuspended,
    Finish,
    Member,
    OrchestrationLimits,
    OrchestrationState,
    Outcome,
    Parallel,
    Parameter,
    Pool,
    Staff,
    arun_orchestration,
    complete_pending,
    decision_prompt,
    parse_decision,
    replay,
    run_orchestration,
    validate,
)

POOL = Pool(
    members=(
        Member(id="research", name="Research", capabilities=("research",)),
        Member(
            id="write",
            name="Writer",
            parameters=(
                Parameter(name="tone", type="choice", choices=("formal", "casual")),
                Parameter(name="words", type="number"),
                Parameter(name="draft", type="boolean"),
            ),
        ),
        Member(id="review", name="Review"),
        Member(id="legal", name="Legal"),
    )
)


def scripted(*answers: Any) -> tuple[Any, list[str]]:
    prompts: list[str] = []
    queue = list(answers)

    def decide(prompt: str) -> Any:
        prompts.append(prompt)
        return queue.pop(0)

    return decide, prompts


def done(item: Delegation) -> str:
    return f"{item.member.id}:{item.task}"


# Dynamic teams ---------------------------------------------------------------------------------------------


def test_staffing_assembles_extends_and_shrinks_the_team() -> None:
    limits = OrchestrationLimits(max_decisions=10, staffing=True, max_team=2)
    decide, prompts = scripted(
        Delegate(member="research", task="facts"),  # not in the team yet
        Staff(add=("research",), reason="facts first"),
        Delegate(member="research", task="facts"),
        Staff(add=("write", "review")),  # three members: too many
        Staff(add=("write",), remove=("research",), reason="facts are in"),
        Delegate(member="write", task="text", values={"tone": "formal"}),
        Finish(answer="text"),
    )
    state = run_orchestration("t", POOL, decide=decide, execute=done, limits=limits)
    codes = [step.verdict.code for step in state.steps]
    assert codes == ["orchestration.not_in_team", None, None, "orchestration.team_limit", None, None, None]
    assert state.team(POOL, limits) == ("write",)
    assert "Team now: nobody yet" in prompts[0] and "Team now: research" in prompts[2]
    assert "Team changed: added write; removed research" in prompts[5]
    assert state.outcome == "finished"


def test_staffing_refusals() -> None:
    off = OrchestrationLimits()
    on = OrchestrationLimits(staffing=True)
    empty = OrchestrationState(task="t")
    assert validate(Staff(add=("research",)), POOL, empty, off).code == "orchestration.staffing_disabled"
    assert validate(Staff(), POOL, empty, on).code == "orchestration.empty_staffing"
    assert validate(Staff(add=("ghost",)), POOL, empty, on).code == "orchestration.unknown_member"
    assert validate(Staff(remove=("write",)), POOL, empty, on).code == "orchestration.not_in_team"
    staffed = run_orchestration(
        "t", POOL, decide=scripted(Staff(add=("write",)), Finish(answer="x"))[0], execute=done, limits=on
    )
    replayed = staffed.model_copy(update={"steps": staffed.steps[:1], "outcome": None, "answer": None})
    assert validate(Staff(add=("write",)), POOL, replayed, on).code == "orchestration.already_in_team"


def test_without_staffing_the_whole_pool_is_the_team() -> None:
    assert OrchestrationState(task="t").team(POOL, OrchestrationLimits()) == (
        "research",
        "write",
        "review",
        "legal",
    )


# Parameters (AO-5 level 2) ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "code"),
    [
        ({"tone": "formal", "words": 200, "draft": True}, None),
        ({"length": 3}, "orchestration.unknown_parameter"),
        ({"tone": "angry"}, "orchestration.invalid_value"),
        ({"words": "many"}, "orchestration.invalid_value"),
        ({"words": True}, "orchestration.invalid_value"),
        ({"draft": "yes"}, "orchestration.invalid_value"),
        ({"tone": "x" * 5000}, "orchestration.invalid_value"),
    ],
)
def test_values_are_checked_against_the_members_parameters(values: dict[str, Any], code: str | None) -> None:
    decision = Delegate(member="write", task="text", values=values)
    assert validate(decision, POOL, OrchestrationState(task="t"), OrchestrationLimits()).code == code


def test_values_reach_the_member_and_the_prompt_lists_parameters() -> None:
    seen: list[Delegation] = []

    def execute(item: Delegation) -> str:
        seen.append(item)
        return "ok"

    decide, prompts = scripted(
        Delegate(member="write", task="text", values={"tone": "casual"}), Finish(answer="x")
    )
    run_orchestration("t", POOL, decide=decide, execute=execute)
    assert seen[0].values == {"tone": "casual"} and seen[0].step == 0 and seen[0].ref is None
    assert "parameter tone (one of formal, casual, optional)" in prompts[0]
    assert '(values: {"tone": "casual"})' in prompts[1]


def test_repeats_compare_values() -> None:
    decide, _ = scripted(
        Delegate(member="write", task="text", values={"tone": "formal"}),
        Delegate(member="write", task="text", values={"tone": "formal"}),
        Delegate(member="write", task="text", values={"tone": "casual"}),
        Finish(answer="x"),
    )
    state = run_orchestration("t", POOL, decide=decide, execute=done)
    assert [step.verdict.code for step in state.steps] == [None, "orchestration.repeated", None, None]


# Parallel delegation ---------------------------------------------------------------------------------------


def test_parallel_refusals() -> None:
    state = OrchestrationState(task="t")
    two = Parallel(
        assignments=(
            Delegate(member="research", task="a").parts()[0],
            *Delegate(member="review", task="b").parts(),
        )
    )
    assert validate(two, POOL, state, OrchestrationLimits()).code == "orchestration.parallel_disabled"
    limits = OrchestrationLimits(max_parallel=2)
    assert validate(two, POOL, state, limits).ok
    three = parse_decision(
        {
            "action": "parallel",
            "assignments": [
                {"member": "research", "task": "a"},
                {"member": "review", "task": "b"},
                {"member": "legal", "task": "c"},
            ],
        }
    )
    assert validate(three, POOL, state, limits).code == "orchestration.parallel_limit"
    same = parse_decision(
        {
            "action": "parallel",
            "assignments": [{"member": "research", "task": "a"}, {"member": "research", "task": "b"}],
        }
    )
    assert validate(same, POOL, state, limits).code == "orchestration.parallel_duplicate"
    ghost = parse_decision(
        {
            "action": "parallel",
            "assignments": [{"member": "research", "task": "a"}, {"member": "ghost", "task": "b"}],
        }
    )
    assert validate(ghost, POOL, state, limits).code == "orchestration.unknown_member"
    with pytest.raises(ValueError):
        parse_decision({"action": "parallel", "assignments": [{"member": "research", "task": "a"}]})


def test_parallel_members_run_at_the_same_time_and_each_outcome_is_recorded() -> None:
    limits = OrchestrationLimits(max_parallel=3)
    answers = [
        parse_decision(
            {
                "action": "parallel",
                "reason": "independent parts",
                "assignments": [
                    {"member": "research", "task": "a"},
                    {"member": "review", "task": "b"},
                    {"member": "legal", "task": "c"},
                ],
            }
        ),
        Finish(answer="x"),
    ]
    prompts: list[str] = []

    async def decide(prompt: str) -> Any:
        prompts.append(prompt)
        return answers.pop(0)

    running = {"now": 0, "max": 0}

    async def execute(item: Delegation) -> str:
        running["now"] += 1
        running["max"] = max(running["max"], running["now"])
        await asyncio.sleep(0.01)
        running["now"] -= 1
        if item.member.id == "legal":
            raise DelegationEnded("rejected", "approval rejected")
        if item.member.id == "review":
            raise RuntimeError("broken")
        return "facts"

    state = asyncio.run(arun_orchestration("t", POOL, decide=decide, execute=execute, limits=limits))
    assert running["max"] == 3
    outcomes = state.steps[0].outcomes
    assert [(item.member, item.status) for item in outcomes] == [
        ("research", "succeeded"),
        ("review", "failed"),
        ("legal", "rejected"),
    ]
    assert outcomes[1].error == "RuntimeError: broken"
    assert "Delegated in parallel:" in prompts[1] and "rejected: approval rejected" in prompts[1]
    assert state.delegations("legal") and state.outcome == "finished"


def test_a_budget_error_in_one_parallel_member_cancels_the_others() -> None:
    limits = OrchestrationLimits(max_parallel=2)
    cancelled: list[str] = []

    async def decide(prompt: str) -> Any:
        return parse_decision(
            {
                "action": "parallel",
                "assignments": [{"member": "research", "task": "a"}, {"member": "review", "task": "b"}],
            }
        )

    async def execute(item: Delegation) -> str:
        if item.member.id == "research":
            raise BudgetExceededError("budget")
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            cancelled.append(item.member.id)
            raise
        return "late"

    started = time.monotonic()
    with pytest.raises(BudgetExceededError):
        asyncio.run(arun_orchestration("t", POOL, decide=decide, execute=execute, limits=limits))
    assert cancelled == ["review"] and time.monotonic() - started < 2


def test_member_limits_count_parallel_assignments() -> None:
    limits = OrchestrationLimits(max_parallel=2, max_per_member=1)
    decide, _ = scripted(
        parse_decision(
            {
                "action": "parallel",
                "assignments": [{"member": "research", "task": "a"}, {"member": "review", "task": "b"}],
            }
        ),
        Delegate(member="research", task="again"),
        Finish(answer="x"),
    )
    state = run_orchestration("t", POOL, decide=decide, execute=done, limits=limits)
    assert [step.verdict.code for step in state.steps] == [None, "orchestration.member_limit", None]


# Pauses: a member waits for a person's approval -------------------------------------------------------------


def test_a_waiting_member_pauses_and_continues_without_a_new_decision() -> None:
    calls: list[Delegation] = []
    approved = {"yes": False}

    def execute(item: Delegation) -> str:
        calls.append(item)
        if not approved["yes"]:
            raise DelegationSuspended("child-1")
        return "approved result"

    decide, prompts = scripted(Delegate(member="write", task="text", reason="writer"))
    saved: list[OrchestrationState] = []
    paused = run_orchestration("t", POOL, decide=decide, execute=execute, on_step=saved.append)
    assert paused.paused and paused.outcome is None and not paused.steps
    assert paused.pending is not None and paused.pending.outcomes[0] == Outcome(
        member="write", status="waiting", ref="child-1"
    )
    assert saved[-1] == paused and len(prompts) == 1
    assert validate(Finish(answer="x"), POOL, paused, OrchestrationLimits()).code == "orchestration.paused"

    restored = OrchestrationState.model_validate(json.loads(paused.model_dump_json()))
    approved["yes"] = True
    decide, prompts = scripted(Finish(answer="final"))
    state = run_orchestration("t", POOL, decide=decide, execute=execute, state=restored)
    assert calls[-1].ref == "child-1" and calls[-1].step == 0  # the same execution continues
    assert len(prompts) == 1  # only the finish was asked; the delegation was not asked again
    assert state.steps[0].result == "approved result" and state.steps[0].outcomes[0].ref == "child-1"
    assert state.outcome == "finished" and state.pending is None


def test_still_waiting_stays_paused_and_rejection_or_expiry_is_shown() -> None:
    limits = OrchestrationLimits(max_parallel=2)
    answer = parse_decision(
        {
            "action": "parallel",
            "assignments": [{"member": "research", "task": "a"}, {"member": "legal", "task": "b"}],
        }
    )
    rounds = iter(["wait", "wait", "expire"])
    calls: list[tuple[str, str | None]] = []

    def execute(item: Delegation) -> str:
        calls.append((item.member.id, item.ref))
        if item.member.id == "research":
            return "facts"
        step = next(rounds)
        if step == "wait":
            raise DelegationSuspended("legal-run")
        raise DelegationEnded("expired", "approval expired", ref="legal-run")

    state = run_orchestration("t", POOL, decide=scripted(answer)[0], execute=execute, limits=limits)
    assert state.paused and [o.status for o in state.pending.outcomes] == ["succeeded", "waiting"]  # type: ignore[union-attr]
    state = run_orchestration("t", POOL, decide=scripted()[0], execute=execute, limits=limits, state=state)
    assert state.paused  # still waiting; nothing was asked (an empty script would fail)
    decide, prompts = scripted(Finish(answer="without legal"))
    state = run_orchestration("t", POOL, decide=decide, execute=execute, limits=limits, state=state)
    assert calls == [("research", None), ("legal", None), ("legal", "legal-run"), ("legal", "legal-run")]
    assert [o.status for o in state.steps[0].outcomes] == ["succeeded", "expired"]
    assert "expired: approval expired" in prompts[0] and state.outcome == "finished"


def test_a_rejected_single_delegation_is_an_error_the_model_sees() -> None:
    def execute(item: Delegation) -> str:
        raise DelegationEnded("rejected", "Peter rejected the payment")

    decide, prompts = scripted(Delegate(member="legal", task="pay"), Finish(answer="not paid"))
    state = run_orchestration("t", POOL, decide=decide, execute=execute)
    assert state.steps[0].error == "Peter rejected the payment"
    assert (
        state.steps[0].outcomes[0].status == "rejected"
        and "rejected: Peter rejected the payment" in prompts[1]
    )


def test_a_waiting_member_that_left_the_pool_fails_instead_of_running() -> None:
    def execute(item: Delegation) -> str:
        raise DelegationSuspended("x")

    paused = run_orchestration(
        "t", POOL, decide=scripted(Delegate(member="legal", task="b"))[0], execute=execute
    )
    smaller = Pool(members=POOL.members[:2])
    decide, _ = scripted(Finish(answer="x"))
    state = run_orchestration("t", smaller, decide=decide, execute=done, state=paused)
    assert state.steps[0].outcomes[0].status == "failed"
    assert state.steps[0].outcomes[0].error == "orchestration.unknown_member"


def test_complete_pending_guards() -> None:
    limits = OrchestrationLimits()
    with pytest.raises(ValueError, match="pending"):
        complete_pending(OrchestrationState(task="t"), (), limits=limits)

    def execute(item: Delegation) -> str:
        raise DelegationSuspended("r")

    paused = run_orchestration(
        "t", POOL, decide=scripted(Delegate(member="legal", task="b"))[0], execute=execute
    )
    with pytest.raises(ValueError, match="match"):
        complete_pending(paused, (Outcome(member="write", status="succeeded"),), limits=limits)
    with pytest.raises(ValueError, match="match"):
        complete_pending(paused, (), limits=limits)


def test_an_async_pause_and_resume() -> None:
    state_box: dict[str, Any] = {"approved": False}

    async def decide(prompt: str) -> Any:
        return Finish(answer="done") if "Delegated" in prompt else Delegate(member="write", task="t")

    async def execute(item: Delegation) -> str:
        if not state_box["approved"]:
            raise DelegationSuspended("c")
        return "r"

    paused = asyncio.run(arun_orchestration("t", POOL, decide=decide, execute=execute))
    assert paused.paused
    state_box["approved"] = True
    state = asyncio.run(arun_orchestration("t", POOL, decide=decide, execute=execute, state=paused))
    assert state.outcome == "finished" and state.steps[0].result == "r"


# Invalid AI decisions, replay and compatibility ------------------------------------------------------------


def test_invalid_answers_never_run_a_member() -> None:
    calls: list[Delegation] = []
    decide, _ = scripted(
        "not json",
        {"action": "create", "name": "new agent", "code": "import os"},
        {"action": "staff", "add": ["research"]},  # staffing is off
        Finish(answer="x"),
    )
    state = run_orchestration("t", POOL, decide=decide, execute=lambda item: calls.append(item) or "r")  # type: ignore[func-returns-value]
    assert not calls
    assert [step.verdict.code for step in state.steps] == [
        "orchestration.invalid_decision",
        "orchestration.invalid_decision",
        "orchestration.staffing_disabled",
        None,
    ]


def test_a_version_2_record_replays_with_its_team_and_pending_delegation() -> None:
    limits = OrchestrationLimits(staffing=True, max_parallel=2, max_decisions=10)
    rounds = {"n": 0}

    def execute(item: Delegation) -> str:
        if item.member.id == "legal":
            raise DelegationSuspended("legal-1")
        return "r"

    decide, _ = scripted(
        Staff(add=("research", "legal")),
        parse_decision(
            {
                "action": "parallel",
                "assignments": [{"member": "research", "task": "a"}, {"member": "legal", "task": "b"}],
            }
        ),
    )
    paused = run_orchestration("t", POOL, decide=decide, execute=execute, limits=limits)
    rounds["n"] += 1
    restored = OrchestrationState.model_validate(json.loads(paused.model_dump_json()))
    assert restored == paused
    assert [v.code for v in replay(restored, POOL, limits)] == [None, None]
    # Under limits without parallel delegation the pending decision would have been refused.
    assert [v.code for v in replay(restored, POOL, limits.model_copy(update={"max_parallel": 1}))] == [
        None,
        "orchestration.parallel_disabled",
    ]


def test_a_version_1_record_still_reads_and_replays() -> None:
    recorded = {
        "version": 1,
        "task": "t",
        "steps": [
            {
                "index": 0,
                "decision": {"action": "delegate", "member": "research", "task": "facts", "reason": "r"},
                "verdict": {"code": None, "message": ""},
                "result": "facts",
                "error": None,
                "raw": None,
            },
            {
                "index": 1,
                "decision": {"action": "finish", "answer": "A", "reason": ""},
                "verdict": {"code": None, "message": ""},
                "result": None,
                "error": None,
                "raw": None,
            },
        ],
        "outcome": "finished",
        "answer": "A",
        "stop_code": None,
    }
    state = OrchestrationState.model_validate(recorded)
    assert state.version == 1 and state.steps[0].outcomes == ()
    assert [v.ok for v in replay(state, POOL, OrchestrationLimits())] == [True, True]
    assert "Delegated to research: facts\n   result:\nfacts" in decision_prompt(
        state.model_copy(update={"outcome": None}), POOL, OrchestrationLimits()
    )


def test_exactly_one_runner() -> None:
    with pytest.raises(TypeError):
        run_orchestration("t", POOL, decide=scripted()[0])
    with pytest.raises(TypeError):
        run_orchestration("t", POOL, decide=scripted()[0], delegate=lambda m, t: "", execute=done)
    with pytest.raises(ValueError):
        DelegationEnded("waiting")
