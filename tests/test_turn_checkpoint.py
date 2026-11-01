"""Turn checkpoints: a tool call that waits suspends the turn; resume continues without a new model call."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from orkexa import (
    Agent,
    AsyncAgent,
    CheckpointError,
    ToolCall,
    ToolCallSuspended,
    TurnCheckpoint,
    TurnSuspended,
    UsageTracker,
)
from orkexa.core.types import ToolResultPart, TurnFailed
from orkexa.providers.fake import FakeModel, text, tool_calls

from .helpers import add, async_add, multiply


class Gate:
    """A ``before_tool`` hook that asks for ``multiply`` and decides it on resume."""

    def __init__(self, decision: Any = None) -> None:
        self.decision = decision
        self.seen: list[ToolCall] = []

    def __call__(self, call: ToolCall) -> Any:
        self.seen.append(call)
        if call.name != "multiply":
            return None
        if not call.resumed:
            raise ToolCallSuspended("approval needed", ref=f"approval-{call.id}")
        return self.decision


def make(script: list[Any], gate: Gate, **kwargs: Any) -> tuple[Agent[Any], FakeModel]:
    model = FakeModel(script)
    return Agent(model="fake:m", client=model, tools=[add, multiply], before_tool=gate, **kwargs), model


def suspend(gate: Gate, **kwargs: Any) -> TurnCheckpoint:
    agent, model = make(
        [tool_calls(("add", {"a": 1, "b": 2}), ("multiply", {"a": 3, "b": 4}))], gate, **kwargs
    )
    with pytest.raises(TurnSuspended) as caught:
        agent.ask("compute")
    assert agent.messages == []  # nothing committed
    assert len(model.requests) == 1
    return caught.value.checkpoint  # type: ignore[no-any-return]


def test_a_waiting_call_suspends_and_resume_continues_without_asking_the_model_again() -> None:
    gate = Gate()
    checkpoint = suspend(gate)
    assert [(item.name, item.ref, item.reason) for item in checkpoint.waiting] == [
        ("multiply", "approval-call_1", "approval needed")
    ]
    assert [result.content for result in checkpoint.results] == ["3"]  # add already ran
    saved = json.loads(json.dumps(checkpoint.model_dump(mode="json")))

    # A new process: a new agent with the same configuration.
    gate2 = Gate(decision=None)
    agent, model = make([text("3 and 12")], gate2)
    assert agent.resume(saved) == "3 and 12"
    assert [(call.name, call.resumed) for call in gate2.seen] == [("multiply", True)]  # add did not run again
    assert len(model.requests) == 1  # only the request with both results
    results = [part for part in model.requests[0].messages[-1].parts() if isinstance(part, ToolResultPart)]
    assert [(part.name, part.content) for part in results] == [("add", "3"), ("multiply", "12")]
    assert [message.role for message in agent.messages] == ["user", "assistant", "user", "assistant"]


def test_a_denial_on_resume_goes_back_to_the_model() -> None:
    checkpoint = suspend(Gate())
    agent, model = make([text("not multiplied")], Gate(decision="rejected by Peter"))
    assert agent.resume(checkpoint) == "not multiplied"
    last = model.requests[0].messages[-1].parts()[1]
    assert isinstance(last, ToolResultPart) and last.is_error
    assert last.content == "Tool call denied: rejected by Peter"


def test_waiting_again_gives_a_new_checkpoint() -> None:
    checkpoint = suspend(Gate())

    def still_waiting(call: ToolCall) -> Any:
        raise ToolCallSuspended("still waiting", ref="r2")

    model = FakeModel([])
    agent = Agent(model="fake:m", client=model, tools=[add, multiply], before_tool=still_waiting)
    with pytest.raises(TurnSuspended) as caught:
        agent.resume(checkpoint)
    again = caught.value.checkpoint
    assert again.waiting[0].ref == "r2" and [r.content for r in again.results] == ["3"]
    assert again.turn_id == checkpoint.turn_id and not model.requests


def test_a_changed_agent_cannot_resume() -> None:
    checkpoint = suspend(Gate())
    other = Agent(model="fake:m", client=FakeModel([]), tools=[add], before_tool=Gate())
    with pytest.raises(CheckpointError, match="changed"):
        other.resume(checkpoint)
    prompt = Agent(
        model="fake:m", client=FakeModel([]), tools=[add, multiply], before_tool=Gate(), system_prompt="Other"
    )
    with pytest.raises(CheckpointError, match="changed"):
        prompt.resume(checkpoint)
    named = Agent(model="fake:m", client=FakeModel([]), tools=[add, multiply], name="someone")
    with pytest.raises(CheckpointError, match="another agent"):
        named.resume(checkpoint)


def test_broken_checkpoints_are_refused() -> None:
    checkpoint = suspend(Gate())
    agent, _ = make([], Gate())
    data = checkpoint.model_dump(mode="json")
    with pytest.raises(CheckpointError, match="version"):
        agent.resume({**data, "version": 2})
    with pytest.raises(CheckpointError, match="not a turn checkpoint"):
        agent.resume({**data, "format": "other"})
    with pytest.raises(CheckpointError, match="do not match"):
        agent.resume({**data, "waiting": [{"call_id": "call_9", "name": "multiply"}]})
    with pytest.raises(CheckpointError, match="do not match"):
        agent.resume({**data, "results": []})


def test_usage_is_recorded_once_and_the_turn_id_is_kept() -> None:
    tracker = UsageTracker()
    checkpoint = suspend(Gate(), usage_tracker=tracker)
    assert tracker.usage.requests == 1
    agent, _ = make([text("done")], Gate(), usage_tracker=tracker)
    agent.resume(checkpoint)
    assert tracker.usage.requests == 2
    assert {record.turn_id for record in tracker.records} == {checkpoint.turn_id}


def test_parallel_tools_suspend_without_running_a_call_twice() -> None:
    runs: list[str] = []

    def gate(call: ToolCall) -> Any:
        runs.append(f"{call.name}:{call.resumed}")
        if call.name == "multiply" and not call.resumed:
            raise ToolCallSuspended(ref="x")
        return None

    agent, _ = make(
        [tool_calls(("add", {"a": 1, "b": 2}), ("multiply", {"a": 3, "b": 4}), ("add", {"a": 5, "b": 5}))],
        gate,  # type: ignore[arg-type]
        max_parallel_tools=3,
    )
    with pytest.raises(TurnSuspended) as caught:
        agent.ask("x")
    checkpoint = caught.value.checkpoint
    assert sorted(r.content for r in checkpoint.results) == ["10", "3"]  # type: ignore[type-var]
    agent2, model = make([text("ok")], gate, max_parallel_tools=3)  # type: ignore[arg-type]
    agent2.resume(checkpoint)
    assert sorted(runs) == ["add:False", "add:False", "multiply:False", "multiply:True"]
    contents = [p.content for p in model.requests[0].messages[-1].parts() if isinstance(p, ToolResultPart)]
    assert contents == ["3", "12", "10"]  # in the order the model asked


def test_suspension_is_no_turn_failure() -> None:
    events: list[Any] = []
    model = FakeModel([tool_calls(("multiply", {"a": 1, "b": 2}))])
    agent = Agent(
        model="fake:m", client=model, tools=[add, multiply], before_tool=Gate(), on_event=events.append
    )
    with pytest.raises(TurnSuspended):
        agent.ask("x")
    assert not any(isinstance(event, TurnFailed) for event in events)


def test_the_async_agent_suspends_and_resumes() -> None:
    async def run() -> None:
        async def gate(call: ToolCall) -> Any:
            if not call.resumed:
                raise ToolCallSuspended(ref="a")
            return None

        model = FakeModel([tool_calls(("async_add", {"a": 2, "b": 2}))])
        agent: AsyncAgent[str] = AsyncAgent(model="fake:m", client=model, tools=[async_add], before_tool=gate)
        with pytest.raises(TurnSuspended) as caught:
            await agent.ask("x")
        model2 = FakeModel([text("4")])
        agent2: AsyncAgent[str] = AsyncAgent(
            model="fake:m", client=model2, tools=[async_add], before_tool=gate
        )
        assert await agent2.resume(caught.value.checkpoint) == "4"
        assert len(model2.requests) == 1

    asyncio.run(run())
