"""Request-stage checkpoints: a usage tracker suspends the next model request; resume sends it."""

from __future__ import annotations

import asyncio
from decimal import Decimal
import json
from typing import Any

from pydantic import ValidationError
import pytest

from orkexa import (
    Agent,
    AsyncAgent,
    CheckpointError,
    ModelPricing,
    RequestSuspended,
    Reservation,
    Tool,
    ToolCall,
    ToolCallSuspended,
    TurnCheckpoint,
    TurnSuspended,
    UsageTracker,
)
from orkexa.core import runtime as core_runtime
from orkexa.core.types import TextPart, ToolResultPart, TurnFailed
from orkexa.dynamic import BudgetExceededError, BudgetTracker
from orkexa.providers.fake import FakeModel, response, text, tool_calls

from .helpers import add, async_add, multiply

CHEAP = ModelPricing.of("1", "1", "0", "0", "0")  # a fake request costs $0.000015


@pytest.fixture
def bound_of_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every request can cost at most $1, so limits are easy to read."""
    monkeypatch.setattr(core_runtime, "request_cost_bound", lambda request, price: Decimal(1))


class PausingBudget(BudgetTracker):
    """Suspends a request that does not fit the budget instead of failing the turn."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.refusals: list[BudgetExceededError] = []

    def try_reserve(
        self, *, agent: str, provider: str, model: str, amount: Decimal | None
    ) -> Reservation | None:
        try:
            return super().try_reserve(agent=agent, provider=provider, model=model, amount=amount)
        except BudgetExceededError as exc:
            self.refusals.append(exc)
            raise RequestSuspended(f"needs ${exc.requested_usd}", ref=f"budget-{len(self.refusals)}") from exc


class Scripted(UsageTracker):
    """Suspends the admissions whose numbers (from 0) are in ``pause`` at ``where``."""

    def __init__(self, pause: set[int], where: str = "try_reserve") -> None:
        super().__init__()
        self.pause = pause
        self.where = where
        self.admissions = 0
        self.reserved = 0
        self.released = 0

    def _maybe(self, where: str) -> None:
        if where == self.where and self.admissions in self.pause:
            self.admissions += 1
            raise RequestSuspended(f"paused at {where}", ref=f"r{self.admissions}")

    def before_request(self, *, agent: str, provider: str, model: str, priced: bool) -> None:
        self._maybe("before_request")

    def try_reserve(
        self, *, agent: str, provider: str, model: str, amount: Decimal | None
    ) -> Reservation | None:
        self._maybe("try_reserve")
        self.admissions += 1
        self.reserved += 1
        return super().try_reserve(agent=agent, provider=provider, model=model, amount=amount)

    def release(self, reservation: Reservation) -> None:
        if not reservation.released:
            self.released += 1
        super().release(reservation)


def _agent(script: list[Any], tracker: UsageTracker, **kwargs: Any) -> tuple[Agent[str], FakeModel]:
    model = FakeModel(script)
    kwargs.setdefault("tools", [add, multiply])
    agent: Agent[str] = Agent(model="fake:m", client=model, pricing=CHEAP, usage_tracker=tracker, **kwargs)
    return agent, model


def _suspend(agent: Agent[Any], prompt: str = "compute", **kwargs: Any) -> TurnCheckpoint:
    with pytest.raises(TurnSuspended) as caught:
        agent.ask(prompt, **kwargs)
    assert agent.messages == []  # nothing committed
    checkpoint = caught.value.checkpoint
    assert isinstance(checkpoint, TurnCheckpoint) and checkpoint.stage == "request"
    return checkpoint


# ---- suspending and resuming -------------------------------------------------------------------


def test_the_first_request_suspends_and_resume_sends_it_once(bound_of_one: None) -> None:
    budget = PausingBudget("0.5", on_oversized="reject")  # a $1 request does not fit
    agent, model = _agent([text("done")], budget)
    checkpoint = _suspend(agent)
    assert checkpoint.round == 0 and checkpoint.reason == "needs $1" and checkpoint.ref == "budget-1"
    assert (checkpoint.calls, checkpoint.results, checkpoint.waiting) == ([], [], [])
    assert [message["role"] for message in checkpoint.history] == ["user"]
    assert model.requests == [] and budget.records == ()
    assert budget.reserved_usd == 0 and budget._in_flight == 0
    refusal = budget.refusals[0]
    assert refusal.reason == "request_too_large" and refusal.tracker is budget

    saved = json.loads(json.dumps(checkpoint.model_dump(mode="json")))
    assert budget.set_limit(5) == Decimal("0.5")  # a person raises the budget
    assert agent.resume(saved) == "done"
    assert len(model.requests) == 1 and len(budget.records) == 1
    assert [message.role for message in agent.messages] == ["user", "assistant"]
    assert budget.records[0].turn_id == checkpoint.turn_id
    assert budget.reserved_usd == 0 and budget._in_flight == 0


def test_a_later_round_suspends_after_tool_results(bound_of_one: None) -> None:
    runs: list[int] = []

    @Tool
    def count(n: int) -> int:
        """Count a call.

        Args:
            n: Any number.

        Returns:
            The number.
        """
        runs.append(n)
        return n

    # The first $1 request fits, the second one does not once the first was billed.
    budget = PausingBudget("1.00001", on_oversized="reject")
    agent, model = _agent([tool_calls(("count", {"n": 7}))], budget, tools=[count])
    checkpoint = _suspend(agent)
    assert budget.spent_usd > Decimal("0.00001")
    assert checkpoint.round == 1 and runs == [7]
    assert [message["role"] for message in checkpoint.history] == ["user", "assistant", "user"]
    assert len(model.requests) == 1 and len(budget.records) == 1
    assert budget.reserved_usd == 0 and budget._in_flight == 0

    budget.set_limit(10)
    model.add(text("7 counted"))
    assert agent.resume(checkpoint) == "7 counted"
    assert runs == [7]  # the tool did not run again
    assert len(model.requests) == 2 and len(budget.records) == 2  # no request sent twice
    sent = model.requests[1].messages
    results = [part for part in sent[-1].parts() if isinstance(part, ToolResultPart)]
    assert [(part.name, part.content) for part in results] == [("count", "7")]
    assert [message.role for message in agent.messages] == ["user", "assistant", "user", "assistant"]
    assert {record.turn_id for record in budget.records} == {checkpoint.turn_id}


def test_another_agent_instance_resumes_the_checkpoint() -> None:
    tracker = Scripted({1})
    agent, model = _agent([tool_calls(("add", {"a": 1, "b": 2}))], tracker)
    checkpoint = _suspend(agent)
    data = checkpoint.model_dump(mode="json")
    agent2, model2 = _agent([text("3")], Scripted(set()))
    assert agent2.resume(data) == "3"
    assert len(model.requests) == 1 and len(model2.requests) == 1
    last = model2.requests[0].messages[-1].parts()[0]
    assert isinstance(last, ToolResultPart) and last.content == "3"


def test_suspending_again_gives_a_new_checkpoint_then_succeeds() -> None:
    tracker = Scripted({0, 1, 2})
    agent, model = _agent([text("finally")], tracker)
    first = _suspend(agent)
    with pytest.raises(TurnSuspended) as again:
        agent.resume(first)
    second = again.value.checkpoint
    assert second.stage == "request" and second.round == first.round
    assert second.turn_id == first.turn_id and second.ref == "r2" and first.ref == "r1"
    assert second.history == first.history
    with pytest.raises(TurnSuspended) as third:
        agent.resume(second)
    assert agent.resume(third.value.checkpoint) == "finally"
    assert len(model.requests) == 1 and len(tracker.records) == 1
    assert tracker.reserved == tracker.released == 1
    assert agent.messages[-1].text == "finally"


def test_suspending_in_before_request_reserves_nothing() -> None:
    tracker = Scripted({0}, where="before_request")
    agent, model = _agent([text("ok")], tracker)
    checkpoint = _suspend(agent)
    assert checkpoint.reason == "paused at before_request"
    assert tracker.reserved == 0 and model.requests == []
    assert agent.resume(checkpoint) == "ok"
    assert tracker.reserved == tracker.released == 1


def test_the_conversation_may_end_with_a_system_or_assistant_message() -> None:
    tracker = Scripted({0})
    agent, _ = _agent([text("short")], tracker)
    checkpoint = _suspend(agent, system="Be brief.")
    assert [message["role"] for message in checkpoint.history] == ["user", "system"]
    assert agent.resume(checkpoint) == "short"

    paused = response([TextPart(text="searching")], stop_reason="pause_turn")
    tracker2 = Scripted({1})
    agent2, model2 = _agent([paused], tracker2)
    checkpoint2 = _suspend(agent2)
    assert checkpoint2.round == 1
    assert [message["role"] for message in checkpoint2.history] == ["user", "assistant"]
    model2.add(text("found"))
    assert agent2.resume(checkpoint2) == "found"
    assert len(model2.requests) == 2


def test_a_tool_stage_suspension_can_follow_a_request_stage_one() -> None:
    def gate(call: ToolCall) -> Any:
        if call.name == "multiply" and not call.resumed:
            raise ToolCallSuspended("approval", ref="a1")
        return None

    tracker = Scripted({0})
    agent, model = _agent([tool_calls(("multiply", {"a": 3, "b": 4})), text("12")], tracker, before_tool=gate)
    request_stage = _suspend(agent)
    with pytest.raises(TurnSuspended) as caught:
        agent.resume(request_stage)
    tools_stage = caught.value.checkpoint
    assert tools_stage.stage == "tools" and tools_stage.waiting[0].ref == "a1"
    assert agent.resume(tools_stage) == "12"
    assert len(model.requests) == 2


def test_a_refused_parent_reservation_leaks_nothing(bound_of_one: None) -> None:
    parent = BudgetTracker("0.5", on_oversized="reject")
    child = PausingBudget(100, parent=parent, prefix="team")
    agent, model = _agent([text("x")], child)
    checkpoint = _suspend(agent)
    assert child.refusals[0].tracker is parent
    assert checkpoint.reason == "needs $1"
    assert child.reserved_usd == parent.reserved_usd == 0
    assert child._in_flight == parent._in_flight == 0
    assert child.unbounded_requests == child.oversized_requests == 0
    parent.set_limit(2)
    assert agent.resume(checkpoint) == "x"
    assert len(parent.records) == len(child.records) == 1 and len(model.requests) == 1


def test_suspension_is_no_turn_failure_and_names_the_reason() -> None:
    events: list[Any] = []
    agent, _ = _agent([text("x")], Scripted({0}), on_event=events.append)
    with pytest.raises(TurnSuspended, match="paused at try_reserve"):
        agent.ask("x")
    assert not any(isinstance(event, TurnFailed) for event in events)


def test_the_async_agent_suspends_and_resumes() -> None:
    async def run() -> None:
        tracker = Scripted({1})
        model = FakeModel([tool_calls(("async_add", {"a": 2, "b": 2}))])
        agent: AsyncAgent[str] = AsyncAgent(
            model="fake:m", client=model, tools=[async_add], usage_tracker=tracker
        )
        with pytest.raises(TurnSuspended) as caught:
            await agent.ask("x")
        checkpoint = caught.value.checkpoint
        assert checkpoint.stage == "request" and checkpoint.round == 1
        assert agent.messages == []
        model.add(text("4"))
        assert await agent.resume(checkpoint) == "4"
        assert len(model.requests) == 2 and len(tracker.records) == 2
        assert tracker.reserved == tracker.released == 2

    asyncio.run(run())


# ---- the checkpoint ----------------------------------------------------------------------------


def _request_checkpoint() -> tuple[TurnCheckpoint, Agent[str]]:
    agent, _ = _agent([], Scripted({0}))
    return _suspend(agent), agent


def test_the_checkpoint_round_trips_through_json() -> None:
    checkpoint, _ = _request_checkpoint()
    assert TurnCheckpoint.model_validate_json(checkpoint.model_dump_json()) == checkpoint
    data = json.loads(json.dumps(checkpoint.model_dump(mode="json")))
    assert data["stage"] == "request" and data["ref"] == "r1" and data["version"] == 1
    assert TurnCheckpoint.model_validate(data) == checkpoint


def test_an_old_tool_stage_checkpoint_without_stage_still_resumes() -> None:
    def gate(call: ToolCall) -> Any:
        if not call.resumed:
            raise ToolCallSuspended("approval", ref="x")
        return None

    agent, _ = _agent([tool_calls(("multiply", {"a": 3, "b": 4}))], UsageTracker(), before_tool=gate)
    with pytest.raises(TurnSuspended) as caught:
        agent.ask("x")
    data = caught.value.checkpoint.model_dump(mode="json")
    for key in ("stage", "reason", "ref"):
        data.pop(key)  # what a checkpoint written by an earlier Orkexa looks like
    assert TurnCheckpoint.model_validate(data).stage == "tools"
    agent2, model2 = _agent([text("12")], UsageTracker(), before_tool=gate)
    assert agent2.resume(data) == "12"
    assert len(model2.requests) == 1


def test_invalid_stage_combinations_are_rejected() -> None:
    checkpoint, agent = _request_checkpoint()
    data = checkpoint.model_dump(mode="json")
    call = {"id": "c1", "name": "add", "arguments": {"a": 1, "b": 2}}
    waiting = {"call_id": "c1", "name": "add"}
    result = {"call_id": "c1", "name": "add", "content": "3"}
    for broken in (
        {**data, "calls": [call]},
        {**data, "waiting": [waiting]},
        {**data, "results": [result]},
        {**data, "stage": "tools"},  # tools need calls and waiting calls
        {**data, "stage": "tools", "calls": [call]},
        {**data, "stage": "tools", "calls": [call], "waiting": [waiting], "ref": "r1"},
        {**data, "stage": "elsewhere"},
    ):
        with pytest.raises(ValidationError):
            TurnCheckpoint.model_validate(broken)
        with pytest.raises(ValueError):
            agent.resume(broken)
    with pytest.raises(CheckpointError, match="empty"):
        agent.resume({**data, "history": []})
    with pytest.raises(CheckpointError, match="max_tool_rounds"):
        agent.resume({**data, "round": agent.max_tool_rounds + 1})


def test_a_changed_agent_cannot_resume_a_request_checkpoint() -> None:
    checkpoint, _ = _request_checkpoint()
    prompt, _ = _agent([], UsageTracker(), system_prompt="Other")
    with pytest.raises(CheckpointError, match="changed"):
        prompt.resume(checkpoint)
    fewer, _ = _agent([], UsageTracker(), tools=[add])
    with pytest.raises(CheckpointError, match="changed"):
        fewer.resume(checkpoint)
    named, _ = _agent([], UsageTracker(), name="someone")
    with pytest.raises(CheckpointError, match="another agent"):
        named.resume(checkpoint)
    tampered = {**checkpoint.model_dump(mode="json"), "fingerprint": "0" * 64}
    same, _ = _agent([], UsageTracker())
    with pytest.raises(CheckpointError, match="changed"):
        same.resume(tampered)
