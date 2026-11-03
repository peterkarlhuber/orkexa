"""ORK2-09: a turn no longer deep-copies the whole history, and the committed history stays untouched.

Every ``ask()`` used to deep-copy every committed message (77 % of the per-turn time with long
histories). The turn now extends a new list holding the committed messages; these tests pin that
neither the runtime nor any provider adapter changes a committed message, and that a failed turn
leaves the committed conversation exactly as it was.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from orkexa import Agent, AsyncAgent, ProviderError, ToolExecutionError
from orkexa.core.conversion import dump_history
from orkexa.providers.fake import FakeModel, text, tool_calls

from ..contract.harness import HARNESSES, Harness
from ..helpers import add

pytestmark = pytest.mark.filterwarnings("ignore::orkexa.HistoryConversionWarning")


@pytest.fixture(params=HARNESSES, ids=[h.name for h in HARNESSES])
def harness(request: pytest.FixtureRequest) -> Harness:
    cls: type[Harness] = request.param
    return cls()


def _snapshot(messages: list[Any]) -> Any:
    return dump_history(messages)


def test_a_turn_does_not_copy_the_committed_messages() -> None:
    agent = Agent(model="fake:m", client=FakeModel([text("one"), text("two")]))
    agent.ask("first")
    committed = list(agent.messages)
    agent.ask("second")
    assert all(new is old for new, old in zip(agent.messages, committed, strict=False))
    assert len(agent.messages) == 4
    # What callers get is still a copy.
    assert all(a is not b for a, b in zip(agent.conversation(), agent.messages, strict=True))
    assert agent.history()[0] is not agent.messages[0]


@pytest.mark.parametrize("stream", [False, True])
def test_adapters_never_change_committed_messages(harness: Harness, stream: bool) -> None:
    agent = harness.agent(
        [
            harness.tool_calls([("c1", "add", {"a": 1, "b": 2})]),
            harness.text("three"),
            harness.status_error(429),  # turn 2 fails at its first request
            harness.tool_calls([("c2", "add", {"a": 2, "b": 2})]),
            harness.status_error(500),  # turn 3 fails after a tool round
            harness.text("ok"),
        ],
        tools=[add],
        stream=stream,
    )
    assert agent.ask("one plus two") == "three"
    before, objects = _snapshot(agent.messages), list(agent.messages)
    for _ in range(2):
        with pytest.raises(ProviderError):
            agent.ask("again")
        assert agent.messages == objects and _snapshot(agent.messages) == before
    assert agent.ask("last") == "ok"
    assert _snapshot(agent.messages[: len(objects)]) == before  # sent again, never changed


def test_a_failed_tool_round_leaves_the_history_untouched() -> None:
    model = FakeModel([text("hi"), tool_calls(("add", {"a": 1, "b": "x"}))])
    agent = Agent(model="fake:m", client=model, tools=[add])
    agent.ask("hello")
    before = _snapshot(agent.messages)
    with pytest.raises(ToolExecutionError):
        agent.ask("add")
    assert _snapshot(agent.messages) == before and len(agent.messages) == 2


def test_async_turns_share_the_same_guarantee() -> None:
    async def main() -> None:
        agent = AsyncAgent(model="fake:m", client=FakeModel([text("one"), RuntimeError("boom")]))
        await agent.ask("first")
        before, objects = _snapshot(agent.messages), list(agent.messages)
        with pytest.raises(RuntimeError):
            await agent.ask("second")
        assert agent.messages == objects and _snapshot(agent.messages) == before

    asyncio.run(main())
