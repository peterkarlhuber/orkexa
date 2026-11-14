"""Skill recommendations of the orchestrator (NAL-B4): optional, only the member's skills, never a right."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError
import pytest

from orkexa.orchestration import (
    Delegate,
    Delegation,
    Finish,
    Member,
    OrchestrationState,
    Parallel,
    Pool,
    decision_prompt,
    parse_decision,
    replay,
    run_orchestration,
)
from orkexa.orchestration.protocol import MAX_SKILL_HINTS, OrchestrationLimits

POOL = Pool(
    members=(
        Member(id="research", name="Research", skills=("research-method", "source-check")),
        Member(id="write", name="Writer"),
    )
)


def scripted(*answers: Any) -> tuple[Any, list[str]]:
    prompts: list[str] = []
    queue = list(answers)

    def decide(prompt: str) -> Any:
        prompts.append(prompt)
        return queue.pop(0)

    return decide, prompts


def test_the_pool_lists_the_skills_of_a_member() -> None:
    prompt = decision_prompt(OrchestrationState(task="t"), POOL, OrchestrationLimits())
    assert "skills: research-method, source-check" in prompt
    assert prompt.count("skills:") == 1


def test_a_delegation_carries_the_skills_the_member_has_and_names_the_others() -> None:
    seen: list[Delegation] = []

    def execute(item: Delegation) -> str:
        seen.append(item)
        return "ok"

    decide, _ = scripted(
        Delegate(member="research", task="find", skills=("source-check", "legal-review", "source-check", "")),
        Finish(answer="done"),
    )
    state = run_orchestration("t", POOL, decide=decide, execute=execute)
    assert state.outcome is not None
    assert seen[0].skills == ("source-check",)
    assert seen[0].ignored_skills == ("legal-review",)


def test_a_parallel_decision_hints_per_assignment_and_nothing_is_required() -> None:
    seen: list[Delegation] = []

    def execute(item: Delegation) -> str:
        seen.append(item)
        return "ok"

    decision = {
        "action": "parallel",
        "assignments": [
            {"member": "research", "task": "a", "skills": ["research-method"]},
            {"member": "write", "task": "b", "skills": ["research-method"]},
        ],
    }
    decide, _ = scripted(decision, {"action": "finish", "answer": "x"})
    run_orchestration("t", POOL, decide=decide, execute=execute, limits=OrchestrationLimits(max_parallel=2))
    by_member = {item.member.id: item for item in seen}
    assert by_member["research"].skills == ("research-method",)
    assert by_member["write"].skills == () and by_member["write"].ignored_skills == ("research-method",)
    # Without the field the decision is as before.
    assert parse_decision({"action": "delegate", "member": "write", "task": "x"}).skills == ()  # type: ignore[union-attr]


def test_at_most_eight_hints_and_the_record_replays() -> None:
    with pytest.raises(ValidationError):
        Delegate(member="research", task="x", skills=tuple(f"s{i}" for i in range(MAX_SKILL_HINTS + 1)))
    hint = Delegate(member="research", task="x", skills=("research-method",))
    decide, _ = scripted(hint, Finish(answer="y"))
    state = run_orchestration("t", POOL, decide=decide, execute=lambda item: "ok")
    again = OrchestrationState.model_validate(json.loads(state.model_dump_json()))
    assert again.steps[0].decision.parts()[0].skills == ("research-method",)  # type: ignore[union-attr]
    assert all(verdict.ok for verdict in replay(again, POOL, OrchestrationLimits()))
    assert "skills" in Parallel.model_json_schema()["$defs"]["Assignment"]["properties"]
