"""Failures around a turn: hooks that raise, history changes during a turn, partial revokes."""

from __future__ import annotations

import asyncio
from pathlib import Path
import threading
from typing import Any

import pytest

from orkexa import Agent, AsyncAgent, ClaudeAgent, ConcurrentTurnError, ModelPricing, UsageTracker
from orkexa.dynamic import AgentRegistry, AgentSpec, BudgetExceededError, BudgetTracker, ToolRegistry
from orkexa.providers.fake import FakeModel, text

from .helpers import add

PRICE = ModelPricing.of("1000000", "1000000", "0", "0", "0")


def _fail(*_: Any) -> None:
    raise RuntimeError("hook failed")


def _run(agent: Agent[Any] | AsyncAgent[Any], prompt: str) -> Any:
    if isinstance(agent, AsyncAgent):
        return asyncio.run(agent.ask(prompt))
    return agent.ask(prompt)


# ---- usage of answered requests when a hook raises ----------------------------------------


@pytest.mark.parametrize("cls", [Agent, AsyncAgent])
@pytest.mark.parametrize("stream", [False, True])
def test_failing_on_response_still_records_usage_once(cls: type[Any], stream: bool) -> None:
    tracker = UsageTracker()
    fake = FakeModel([text("answered")])
    agent = cls(model="fake:m", client=fake, usage_tracker=tracker, on_response=_fail, stream=stream)
    with pytest.raises(RuntimeError, match="hook failed"):
        _run(agent, "hi")
    assert len(fake.requests) == 1
    assert tracker.usage.requests == agent.usage.requests == 1
    assert agent.messages == []  # the failed turn is not committed


@pytest.mark.parametrize("cls", [Agent, AsyncAgent])
def test_failing_event_hook_on_the_final_stream_event_records_usage(cls: type[Any]) -> None:
    from orkexa.core.types import ResponseCompleted

    def on_event(event: Any) -> None:
        if isinstance(event, ResponseCompleted):
            raise RuntimeError("hook failed")

    tracker = UsageTracker()
    agent = cls(
        model="fake:m", client=FakeModel([text("answered")]), usage_tracker=tracker, on_event=on_event
    )
    with pytest.raises(RuntimeError, match="hook failed"):
        _run(agent, "hi")
    assert tracker.usage.requests == 1


def test_hook_failure_counts_against_the_budget() -> None:
    budget = BudgetTracker("15")  # exactly one request of the fake at these prices
    agent = Agent(model="fake:m", client=FakeModel([text("a")]), pricing=PRICE, usage_tracker=budget)
    agent.on_response = _fail
    with pytest.raises(RuntimeError, match="hook failed"):  # the hook's error, not the tracker's
        agent.ask("hi")
    agent.on_response = None
    with pytest.raises(BudgetExceededError, match="not sent"):
        agent.ask("again")


def test_failing_after_tool_keeps_the_usage_of_the_tool_round() -> None:
    from orkexa.providers.fake import tool_calls

    tracker = UsageTracker()
    agent = Agent(
        model="fake:m",
        client=FakeModel([tool_calls(("add", {"a": 1, "b": 2}))]),
        tools=[add],
        usage_tracker=tracker,
        after_tool=_fail,
    )
    with pytest.raises(RuntimeError, match="hook failed"):
        agent.ask("1+2")
    assert tracker.usage.requests == 1 and agent.messages == []


# ---- history changes while a turn runs ----------------------------------------------------


def _blocked_agent(cls: type[Any] = Agent) -> tuple[Any, threading.Event, threading.Event]:
    started, resume = threading.Event(), threading.Event()

    def reply(request: object) -> object:
        started.set()
        assert resume.wait(5)
        return text("second")

    return cls(model="fake:m", client=FakeModel([text("first"), reply])), started, resume


@pytest.mark.parametrize("change", ["reset", "load_history"])
def test_history_cannot_be_replaced_while_ask_runs(tmp_path: Path, change: str) -> None:
    agent, started, resume = _blocked_agent()
    agent.ask("one")
    path = tmp_path / "h.json"
    agent.save_history(path)
    worker = threading.Thread(target=agent.ask, args=("two",))
    worker.start()
    try:
        assert started.wait(5)
        with pytest.raises(ConcurrentTurnError, match="while ask"):
            agent.reset() if change == "reset" else agent.load_history(path)
    finally:
        resume.set()
        worker.join(5)
    assert [m.text for m in agent.messages if m.role == "user"] == ["one", "two"]
    agent.reset()  # after the turn it works
    assert agent.messages == []


def test_claude_agent_load_history_is_refused_during_a_turn(tmp_path: Path) -> None:
    agent = ClaudeAgent(client=object())
    path = tmp_path / "h.json"
    agent.save_history(path)
    agent._turn_lock.acquire()
    try:
        with pytest.raises(ConcurrentTurnError):
            agent.load_history(path)
        with pytest.raises(ConcurrentTurnError):
            agent.reset()
    finally:
        agent._turn_lock.release()
    agent.load_history(path)


def test_async_reset_from_another_task_is_refused() -> None:
    async def scenario() -> None:
        started, resume = asyncio.Event(), asyncio.Event()

        async def on_request(request: object) -> None:
            started.set()
            await resume.wait()

        agent = AsyncAgent(model="fake:m", client=FakeModel([text("answer")]), on_request=on_request)
        task = asyncio.create_task(agent.ask("one"))
        await started.wait()
        with pytest.raises(ConcurrentTurnError):
            agent.reset()
        resume.set()
        assert await task == "answer"
        assert [m.text for m in agent.messages] == ["one", "answer"]

    asyncio.run(asyncio.wait_for(scenario(), 5))


# ---- revoking several versions -------------------------------------------------------------


def _fail_second_save(registry: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    original = registry._save
    calls = {"n": 0}

    def save(record: Any) -> None:
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("disk full")
        original(record)

    monkeypatch.setattr(registry, "_save", save)


def test_failed_agent_revoke_keeps_the_active_version_and_can_be_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = AgentSpec(name="Alpha", description="d", system_prompt="s", model="fake:m", effort="low")
    registry = AgentRegistry(tmp_path)
    registry.register(spec)
    registry.register(spec)
    _fail_second_save(registry, monkeypatch)
    with pytest.raises(OSError):
        registry.revoke("Alpha")
    for view in (registry, AgentRegistry(tmp_path)):
        assert view.get("Alpha").version == 2  # unchanged: the active version is revoked last
    monkeypatch.undo()
    registry.revoke("Alpha")
    assert "Alpha" not in registry and "Alpha" not in AgentRegistry(tmp_path)


def test_failed_tool_revoke_keeps_the_active_version_and_can_be_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = ToolRegistry(tmp_path)
    for version in (1, 2):
        registry.add_generated(
            name="tool",
            function_name="run",
            description="d",
            code="def run(): return 1",
            tests="",
            input_schema={},
            output_schema=None,
        )
        registry.approve("tool", version)
    _fail_second_save(registry, monkeypatch)
    with pytest.raises(OSError):
        registry.revoke("tool")
    for view in (registry, ToolRegistry(tmp_path)):
        assert view.record("tool").version == 2
    monkeypatch.undo()
    registry.revoke("tool")
    assert "tool" not in registry and "tool" not in ToolRegistry(tmp_path)


def test_cancelling_an_async_turn_in_on_response_records_usage() -> None:
    async def scenario() -> None:
        entered = asyncio.Event()

        async def on_response(response: object) -> None:
            entered.set()
            await asyncio.sleep(10)

        tracker = UsageTracker()
        agent = AsyncAgent(
            model="fake:m", client=FakeModel([text("a")]), usage_tracker=tracker, on_response=on_response
        )
        task = asyncio.create_task(agent.ask("hi"))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert tracker.usage.requests == 1 and agent.messages == []

    asyncio.run(asyncio.wait_for(scenario(), 5))
