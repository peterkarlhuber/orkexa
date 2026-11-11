"""Limits of a turn that the agent can handle instead of failing (RT-067 follow-up).

``on_reasoning_overflow="answer"``: a reasoning model that spent the whole output limit on
reasoning (Qwen 3.5 on Ollama thinks without a bound: the 9b model wrote 5,791 tokens for one
step) gets the same request once more without reasoning. ``on_max_tool_rounds="answer"``: the
last request of a turn asks for the answer instead of offering more tool calls, so a model
that keeps calling tools answers from its results; the number of requests stays the same.
"""

from __future__ import annotations

from typing import Any

import pytest

from orkexa import Agent, CompatibleOptions, IncompleteResponseError, LimitHandled, MaxToolRoundsError
from orkexa.core.runtime import FINAL_ROUND_PROMPT, FORMAT_PROMPT
from orkexa.core.types import ToolChoice

from .helpers import add
from .test_chat_messages import Downtime, FakeOllama, _completion, _tool_round


def _ollama(client: Any, **kwargs: Any) -> Agent[Any]:
    return Agent(
        provider="ollama",
        model="qwen3.5:4b",
        client=client,
        tools=[add],
        capabilities={"tools": "supported", "reasoning": "supported"},
        options=CompatibleOptions(send_reasoning_effort=True),
        effort="medium",
        max_tokens=100,
        stream=False,  # the scripted client answers whole completions
        **kwargs,
    )


def _overflow() -> Any:
    return _completion({"content": "", "reasoning": "thinking " * 50}, "length")


# ---- reasoning overflow ----------------------------------------------------------------


def test_reasoning_overflow_raises_by_default_with_details() -> None:
    client = FakeOllama([_overflow()])
    agent = _ollama(client)
    with pytest.raises(IncompleteResponseError) as info:
        agent.ask("Analyze")
    error = info.value
    assert error.stop_reason == "max_tokens" and error.max_tokens == 100
    assert error.reasoning is True and error.answered is False
    assert len(client.script.calls) == 1


def test_reasoning_overflow_is_answered_without_reasoning_once() -> None:
    events: list[Any] = []
    client = FakeOllama([_overflow(), _completion({"content": "It is 3."}, "stop")])
    agent = _ollama(client, on_reasoning_overflow="answer", on_event=events.append)
    assert agent.ask("Analyze") == "It is 3."
    first, second = client.script.calls
    assert first["reasoning_effort"] == "medium"
    assert second["reasoning_effort"] == "none"
    assert second["messages"] == first["messages"]  # the overflowed answer is not in the history
    assert [e for e in events if isinstance(e, LimitHandled)] == [LimitHandled("reasoning", 0)]


def test_the_rest_of_the_turn_runs_without_reasoning_and_the_next_turn_with() -> None:
    client = FakeOllama(
        [
            _overflow(),
            _tool_round(1),
            _completion({"content": "done"}, "stop"),
            _completion({"content": "next"}, "stop"),
        ]
    )
    agent = _ollama(client, on_reasoning_overflow="answer")
    assert agent.ask("Analyze") == "done"
    assert [call["reasoning_effort"] for call in client.script.calls] == ["medium", "none", "none"]
    assert agent.ask("And now?") == "next"
    assert client.script.calls[-1]["reasoning_effort"] == "medium"


def test_a_second_overflow_in_the_same_turn_raises() -> None:
    client = FakeOllama([_overflow(), _completion({"content": "x" * 10}, "length")])
    agent = _ollama(client, on_reasoning_overflow="answer")
    with pytest.raises(IncompleteResponseError) as info:
        agent.ask("Analyze")
    assert info.value.reasoning is False and info.value.answered is True
    assert len(client.script.calls) == 2


def test_an_answer_cut_off_without_reasoning_is_not_retried() -> None:
    client = FakeOllama([_completion({"content": "long answer"}, "length")])
    agent = _ollama(client, on_reasoning_overflow="answer")
    with pytest.raises(IncompleteResponseError):
        agent.ask("Analyze")
    assert len(client.script.calls) == 1


def test_providers_without_a_reasoning_switch_keep_raising() -> None:
    client = FakeOllama([_overflow()])
    agent = Agent(
        provider="compatible",
        model="m",
        base_url="http://localhost:9/v1",
        client=client,
        capabilities={"reasoning": "supported"},
        options=CompatibleOptions(send_reasoning_effort=True),
        effort="medium",
        max_tokens=100,
        on_reasoning_overflow="answer",
    )
    with pytest.raises(IncompleteResponseError):
        agent.ask("Analyze")
    assert len(client.script.calls) == 1


# ---- tool rounds -----------------------------------------------------------------------


def test_tool_rounds_still_raise_by_default() -> None:
    client = FakeOllama([_tool_round(n) for n in range(1, 4)])
    agent = _ollama(client, max_tool_rounds=2)
    with pytest.raises(MaxToolRoundsError):
        agent.ask("Analyze")
    assert len(client.script.calls) == 3


def test_the_last_round_asks_for_the_answer_without_tools_on_ollama() -> None:
    events: list[Any] = []
    client = FakeOllama([_tool_round(1), _tool_round(2), _completion({"content": "It is 3."}, "stop")])
    agent = _ollama(client, max_tool_rounds=2, on_max_tool_rounds="answer", on_event=events.append)
    assert agent.ask("Analyze") == "It is 3."
    calls = client.script.calls
    assert len(calls) == 3  # the same number of requests as with "raise"
    assert all("tools" in call for call in calls[:2])
    assert "tools" not in calls[2] and "tool_choice" not in calls[2]  # Ollama ignores tool_choice
    assert calls[2]["messages"][-1] == {"role": "user", "content": FINAL_ROUND_PROMPT}
    assert [e for e in events if isinstance(e, LimitHandled)] == [LimitHandled("tool_rounds", 2)]
    # The history keeps the note, so a later turn sees why the answer came without more calls.
    assert any(m.text == FINAL_ROUND_PROMPT for m in agent.messages if m.role == "user")


def test_a_tool_call_in_the_last_round_still_raises() -> None:
    client = FakeOllama([_tool_round(1), _tool_round(2)])
    agent = _ollama(client, max_tool_rounds=1, on_max_tool_rounds="answer")
    with pytest.raises(MaxToolRoundsError):
        agent.ask("Analyze")


def test_the_last_round_gives_the_structured_answer() -> None:
    client = FakeOllama(
        [
            _tool_round(1),
            _completion({"content": '{"analysis": "ok", "downtime_share_percent": 4.62}'}, "stop"),
        ]
    )
    agent = _ollama(client, max_tool_rounds=1, on_max_tool_rounds="answer", output_format=Downtime)
    assert agent.ask("Analyze") == Downtime(analysis="ok", downtime_share_percent=4.62)
    last = client.script.calls[-1]
    assert "tools" not in last and last["response_format"]["type"] == "json_schema"
    assert last["messages"][-1]["content"] == FINAL_ROUND_PROMPT
    assert FORMAT_PROMPT not in [m.get("content") for m in last["messages"]]


def test_apis_that_honor_tool_choice_none_keep_the_tools_declared() -> None:
    agent = _ollama(FakeOllama([]), max_tool_rounds=1, on_max_tool_rounds="answer")
    agent.adapter.tool_free_answer = "tool_choice_none"  # type: ignore[misc]
    history = [*agent._working_conversation()]
    from orkexa.core.types import Message

    history.append(Message(role="user", content=FINAL_ROUND_PROMPT))
    request = agent._build_request(history, check=False)
    assert request.tools and request.tool_choice == ToolChoice("none")


def test_providers_without_a_tool_free_answer_keep_raising() -> None:
    client = FakeOllama([_tool_round(1), _tool_round(2)])
    agent = _ollama(client, max_tool_rounds=1, on_max_tool_rounds="answer")
    agent.adapter.tool_free_answer = None  # type: ignore[misc]
    with pytest.raises(MaxToolRoundsError):
        agent.ask("Analyze")
    assert all("tools" in call for call in client.script.calls)


def test_invalid_modes_are_refused() -> None:
    with pytest.raises(ValueError, match="on_max_tool_rounds"):
        _ollama(FakeOllama([]), on_max_tool_rounds="more")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="on_reasoning_overflow"):
        _ollama(FakeOllama([]), on_reasoning_overflow="more")  # type: ignore[arg-type]
