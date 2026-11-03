"""ORK2-04: one model response cannot fan out into an unbounded number of tool calls."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from orkexa import Agent, AsyncAgent, MaxToolCallsError, MaxToolRoundsError, Tool
from orkexa.core.runtime import DEFAULT_MAX_TOOL_CALLS_PER_ROUND
from orkexa.providers.fake import FakeModel, text, tool_calls


def _setup(count: int, ran: list[int]) -> tuple[Any, FakeModel]:
    @Tool
    def note(x: int) -> str:
        """Record a number.

        Args:
            x: The number.

        Returns:
            ok.
        """
        ran.append(x)
        return "ok"

    model = FakeModel([tool_calls(*(("note", {"x": i}) for i in range(count))), text("done")])
    return note, model


def test_the_default_limit_refuses_a_fan_out_before_any_call_runs() -> None:
    assert DEFAULT_MAX_TOOL_CALLS_PER_ROUND == 64
    ran: list[int] = []
    tool, model = _setup(500, ran)
    agent = Agent(model="fake:m", client=model, tools=[tool], max_tool_rounds=1)
    with pytest.raises(MaxToolCallsError, match="500 tool calls") as caught:
        agent.ask("go")
    assert isinstance(caught.value, MaxToolRoundsError)  # handled like the round limit
    assert ran == []
    assert agent.messages == []


def test_up_to_the_limit_every_call_runs() -> None:
    ran: list[int] = []
    tool, model = _setup(64, ran)
    assert Agent(model="fake:m", client=model, tools=[tool]).ask("go") == "done"
    assert len(ran) == 64


def test_the_limit_is_configurable_and_can_be_switched_off() -> None:
    ran: list[int] = []
    tool, model = _setup(3, ran)
    with pytest.raises(MaxToolCallsError, match="max_tool_calls_per_round is 2"):
        Agent(model="fake:m", client=model, tools=[tool], max_tool_calls_per_round=2).ask("go")
    assert ran == []
    tool, model = _setup(200, ran)
    unlimited = Agent(model="fake:m", client=model, tools=[tool], max_tool_calls_per_round=None)
    assert unlimited.ask("go") == "done"
    assert len(ran) == 200
    for bad in (0, -1):
        with pytest.raises(ValueError, match="max_tool_calls_per_round"):
            Agent(model="fake:m", client=FakeModel([]), max_tool_calls_per_round=bad)


def test_async_agent_has_the_same_limit() -> None:
    ran: list[int] = []
    tool, model = _setup(65, ran)
    agent = AsyncAgent(model="fake:m", client=model, tools=[tool], max_parallel_tools=8)
    with pytest.raises(MaxToolCallsError):
        asyncio.run(agent.ask("go"))
    assert ran == []
