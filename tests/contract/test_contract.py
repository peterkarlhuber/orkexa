"""One set of scenarios, run against every provider adapter with scripted SDK objects.

The harnesses (:mod:`tests.contract.harness`) return real SDK response types and
record the native request arguments; no network access and no API key is used.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from pydantic import BaseModel
import pytest

from orkexa import (
    Agent,
    HistoryConversionError,
    IncompleteResponseError,
    RateLimitError,
    RefusalError,
    UsageTracker,
)
from orkexa.core.conversion import dump_history

from ..helpers import add, async_add, lookup
from .harness import EXPECTED_USAGE, HARNESSES, SECRET, AnthropicHarness, Harness, OpenAIHarness

pytestmark = pytest.mark.filterwarnings("ignore::orkexa.HistoryConversionWarning")


@pytest.fixture(params=HARNESSES, ids=[h.name for h in HARNESSES])
def harness(request: pytest.FixtureRequest) -> Harness:
    cls: type[Harness] = request.param
    return cls()


class City(BaseModel):
    name: str
    population: int


def test_text_answer(harness: Harness) -> None:
    agent = harness.agent([harness.text("Hello!")])
    assert agent.ask("Hi") == "Hello!"
    assert harness.user_texts(harness.script.calls[0]) == ["Hi"]
    assert [m.role for m in agent.messages] == ["user", "assistant"]
    assert agent.messages[-1].text == "Hello!"


def test_two_tool_calls_in_one_response(harness: Harness) -> None:
    agent = harness.agent(
        [
            harness.tool_calls([("c1", "add", {"a": 1, "b": 2}), ("c2", "lookup", {"key": "missing"})]),
            harness.text("done"),
        ],
        tools=[add, lookup],
        raise_tool_errors=False,
    )
    assert agent.ask("go") == "done"
    second = harness.script.calls[1]
    results = harness.tool_results(second)
    assert [(r[0], r[2]) for r in results] == [("c1", False), ("c2", True)]
    assert results[0][1] == "3"
    assert harness.native_replayed(second)


def test_streaming_history_equals_non_streaming(harness: Harness) -> None:
    def script() -> list[Any]:
        return [harness.tool_calls([("c1", "add", {"a": 2, "b": 3})]), harness.text("five")]

    plain = harness.agent(script(), tools=[add], stream=False)
    assert plain.ask("sum") == "five"
    deltas: list[str] = []
    streamed = harness.agent(script(), tools=[add], on_text=deltas.append)
    assert streamed.ask("sum") == "five"
    assert "".join(deltas).endswith("five")
    assert _portable(plain.messages) == _portable(streamed.messages)


def _portable(messages: list[Any]) -> Any:
    """The history without provider metadata (a stream may report ids differently)."""

    def strip(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: strip(v) for k, v in value.items() if k != "meta"}
        if isinstance(value, list):
            return [strip(v) for v in value]
        return value

    return strip(dump_history(messages)["messages"])


def test_async_agent_with_async_tool(harness: Harness) -> None:
    agent = harness.async_agent(
        [harness.tool_calls([("c1", "async_add", {"a": 4, "b": 5})]), harness.text("nine")],
        tools=[async_add],
    )
    assert asyncio.run(agent.ask("add")) == "nine"
    assert harness.tool_results(harness.script.calls[1])[0][:2] == ("c1", "9")


def test_async_streaming(harness: Harness) -> None:
    deltas: list[str] = []
    agent = harness.async_agent([harness.text("streamed")], on_text=deltas.append)
    assert asyncio.run(agent.ask("hi")) == "streamed"
    assert "".join(deltas) == "streamed"


def test_structured_output(harness: Harness) -> None:
    agent = harness.agent([harness.text('{"name": "Vienna", "population": 2000000}')], output_format=City)
    city = agent.ask("Capital of Austria?")
    assert city == City(name="Vienna", population=2000000)


def test_refusal_raises_and_return_mode(harness: Harness) -> None:
    agent = harness.agent([harness.refusal()])
    with pytest.raises(RefusalError):
        agent.ask("bad")
    assert agent.messages == []
    agent = harness.agent([harness.refusal()], on_refusal="return")
    assert isinstance(agent.ask("bad"), str)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_truncation_raises_and_records_usage(harness: Harness, stream: bool, asynchronous: bool) -> None:
    tracker = UsageTracker()
    build = harness.async_agent if asynchronous else harness.agent
    agent = build([harness.truncated("partial")], usage_tracker=tracker, stream=stream)
    with pytest.raises(IncompleteResponseError):
        asyncio.run(agent.ask("long")) if asynchronous else agent.ask("long")
    assert len(tracker.records) == 1
    assert agent.messages == []


def test_usage_normalization(harness: Harness) -> None:
    tracker = UsageTracker()
    agent = harness.agent([harness.text("ok")], usage_tracker=tracker)
    agent.ask("hi")
    (record,) = tracker.records
    assert record.provider == agent.adapter.name
    expected = dict(EXPECTED_USAGE)
    if not harness.reports_thinking_tokens:
        expected["thinking_tokens"] = 0  # included in output_tokens, not reported separately
    for field, value in expected.items():
        assert getattr(record, field) == value, field


def test_rate_limit_is_mapped_and_redacted(harness: Harness) -> None:
    agent = harness.agent([harness.status_error(429)])
    with pytest.raises(RateLimitError) as info:
        agent.ask("hi")
    assert info.value.retryable
    assert SECRET not in str(info.value)
    assert agent.messages == []


def test_save_load_and_continue(harness: Harness, tmp_path: Path) -> None:
    agent = harness.agent(
        [harness.tool_calls([("c1", "add", {"a": 1, "b": 1})]), harness.text("two")], tools=[add]
    )
    agent.ask("one plus one")
    path = tmp_path / "history.json"
    agent.save_history(path)
    resumed = harness.agent([harness.text("again")], tools=[add])
    resumed.load_history(path)
    assert resumed.ask("and now?") == "again"
    call = harness.script.calls[0]
    assert harness.user_texts(call)[0] == "one plus one"
    assert harness.user_texts(call)[-1] == "and now?"
    assert harness.native_replayed(call)


@pytest.mark.parametrize("target", [h for h in HARNESSES if h is not OpenAIHarness], ids=lambda h: h.name)
def test_cross_provider_continuation(target: type[Harness], tmp_path: Path) -> None:
    source = OpenAIHarness()
    agent = source.agent(
        [source.tool_calls([("c1", "add", {"a": 1, "b": 1})]), source.text("two")], tools=[add]
    )
    agent.ask("one plus one")
    path = tmp_path / "history.json"
    agent.save_history(path)

    other = target()
    strict = other.agent([other.text("unused")], tools=[add])
    if strict.adapter.name != "openai":
        strict.load_history(path)
        with pytest.raises(HistoryConversionError):
            strict.ask("continue")

    converted = other.agent([other.text("continued")], tools=[add], history_policy="convert")
    converted.load_history(path)
    assert converted.ask("continue") == "continued"
    call = other.script.calls[0]
    assert other.user_texts(call)[0] == "one plus one"
    results = other.tool_results(_tool_round(other, call))
    assert results == [] or results[0][:2] == ("c1", "2")


def _tool_round(harness: Harness, call: dict[str, Any]) -> dict[str, Any]:
    """The request trimmed to end at the tool results of the first round."""
    if "messages" in call:
        messages = call["messages"]
        for index, message in enumerate(messages):
            if message["role"] in ("tool",) or (
                message["role"] == "user"
                and isinstance(message["content"], list)
                and any(
                    isinstance(b, dict) and (b.get("type") == "tool_result" or "toolResult" in b)
                    for b in message["content"]
                )
            ):
                end = index
                while end + 1 < len(messages) and messages[end + 1]["role"] == "tool":
                    end += 1
                return {**call, "messages": messages[: end + 1]}
    if "contents" in call:
        for index, content in enumerate(call["contents"]):
            if any(p.function_response is not None for p in content.parts or []):
                return {**call, "contents": call["contents"][: index + 1]}
    if "input" in call:
        items = call["input"]
        for index, item in enumerate(items):
            if item.get("type") == "function_call_output":
                return {**call, "input": items[: index + 1]}
    return call


def test_agent_class_is_shared() -> None:
    """Every provider runs on the same Agent class; adapters are the only difference."""
    agents = [h().agent([]) for h in HARNESSES]
    assert {type(a) for a in agents} == {Agent}
    assert len({a.adapter.name for a in agents}) == len(agents)


# ---- multi-step agent runs (Ollama 400 "invalid message content type: <nil>") ----------------


@pytest.mark.parametrize("stream", [False, True])
def test_consecutive_tool_rounds_with_reasoning_then_a_reasoning_only_answer(
    harness: Harness, stream: bool
) -> None:
    """Three tool rounds, each with reasoning and no text, an answer with only reasoning, then
    the next turn: every request is one the provider's server accepts."""
    rounds = [harness.tool_calls([(f"c{n}", "add", {"a": n, "b": 1})]) for n in range(1, 4)]
    agent = harness.agent(
        [*rounds, harness.thinking_only(), harness.text("ok")],
        tools=[add],
        **({"stream": True} if stream else {}),
    )
    assert agent.ask("add three times") == ""
    assert agent.ask("and now?") == "ok"
    calls = harness.script.calls
    assert len(calls) == 5
    for call in calls:
        harness.check_wire(call)
    for number, call in enumerate(calls[1:4], start=1):
        assert harness.tool_results(call)[-1][:2] == (f"c{number}", str(number + 1))
    assert harness.native_replayed(calls[3])
    assert [m.role for m in agent.messages][-3:] == ["assistant", "user", "assistant"]


@pytest.mark.parametrize("stream", [False, True])
def test_structured_answer_after_tool_rounds_and_a_reasoning_only_tool_phase_end(
    harness: Harness, stream: bool
) -> None:
    """The flow of an agent with tools and an output format. Where the provider takes tools and
    the format in separate requests (Ollama), the tool phase may end with reasoning only; the
    request for the structured answer then repeats that turn in a form the server accepts."""
    if stream and isinstance(harness, AnthropicHarness):
        pytest.skip("the scripted Anthropic stream has no parsed output (covered by tests/anthropic)")
    agent = harness.agent([], tools=[add], output_format=City, **({"stream": True} if stream else {}))
    separate = not agent.adapter.structured_output_with_tools
    answer = harness.text('{"name": "Graz", "population": 3}')
    rounds = [
        harness.tool_calls([("c1", "add", {"a": 1, "b": 1})]),
        harness.tool_calls([("c2", "add", {"a": 1, "b": 2})]),
    ]
    harness.script.responses = [*rounds, *([harness.thinking_only()] if separate else []), answer]
    assert agent.ask("How many?") == City(name="Graz", population=3)
    for call in harness.script.calls:
        harness.check_wire(call)
    assert len(harness.script.calls) == (4 if separate else 3)
