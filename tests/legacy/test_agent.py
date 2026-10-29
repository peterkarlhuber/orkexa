from __future__ import annotations

from typing import Any

from pydantic import BaseModel
import pytest

from claude_agent import (
    ClaudeAgent,
    IncompleteResponseError,
    MaxToolRoundsError,
    RefusalError,
    Tool,
    ToolExecutionError,
    UsageTracker,
)

from .conftest import FakeClient, ParsedResponse, make_message


@Tool
def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: First summand.
        b: Second summand.

    Returns:
        The sum.
    """
    return a + b


def _tool_use(name: str, payload: dict[str, Any], block_id: str = "toolu_1") -> Any:
    return make_message(
        [{"type": "tool_use", "id": block_id, "name": name, "input": payload}],
        stop_reason="tool_use",
    )


def _text(text: str, **kwargs: Any) -> Any:
    return make_message([{"type": "text", "text": text}], **kwargs)


def test_plain_text_turn_and_request_shape() -> None:
    client = FakeClient([_text("Hallo!")])
    agent = ClaudeAgent(model="claude-opus-5-5", client=client, effort="high", thinking="adaptive")  # type: ignore[arg-type]

    assert agent.ask("Hi") == "Hallo!"

    call = client.messages.create_calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["output_config"] == {"effort": "high"}
    assert call["thinking"] == {"type": "adaptive"}
    assert call["cache_control"] == {"type": "ephemeral"}
    assert "tools" not in call
    assert agent.history() == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": [{"type": "text", "text": "Hallo!"}]},
    ]


def test_tool_round_trip() -> None:
    client = FakeClient([_tool_use("add", {"a": 2, "b": 3}), _text("5")])
    tracker = UsageTracker()
    agent = ClaudeAgent(client=client, tools=[add], usage_tracker=tracker, cache_ttl="1h")  # type: ignore[arg-type]

    assert agent.ask("2+3?") == "5"

    first, second = client.messages.create_calls
    assert first["tools"] == [add.schema()]
    assert first["tool_choice"] == {"type": "auto"}
    assert first["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert second["messages"][-1] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "5"}],
    }
    assert len(agent.messages) == 4
    assert agent.usage.requests == 2
    assert agent.last_turn_usage.requests == 2
    assert tracker.usage.requests == 2


def test_tool_errors_raise_by_default_and_history_is_untouched() -> None:
    client = FakeClient([_tool_use("add", {"a": "x", "b": 3})])
    agent = ClaudeAgent(client=client, tools=[add])  # type: ignore[arg-type]
    with pytest.raises(ToolExecutionError):
        agent.ask("?")
    assert agent.messages == []


def test_tool_errors_can_be_reported_to_the_model() -> None:
    client = FakeClient([_tool_use("nope", {}), _text("Sorry")])
    agent = ClaudeAgent(client=client, tools=[add], raise_tool_errors=False)  # type: ignore[arg-type]

    assert agent.ask("?") == "Sorry"

    result = client.messages.create_calls[1]["messages"][-1]["content"][0]
    assert result["is_error"] is True
    assert "Unknown tool called: nope" in result["content"]


def test_max_tool_rounds() -> None:
    client = FakeClient([_tool_use("add", {"a": 1, "b": 1})] * 2)
    agent = ClaudeAgent(client=client, tools=[add], max_tool_rounds=1)  # type: ignore[arg-type]
    with pytest.raises(MaxToolRoundsError):
        agent.ask("loop")
    assert agent.messages == []


def test_incomplete_response() -> None:
    client = FakeClient([_text("cut", stop_reason="max_tokens")])
    agent = ClaudeAgent(client=client)  # type: ignore[arg-type]
    with pytest.raises(IncompleteResponseError):
        agent.ask("long")
    assert agent.messages == []
    assert agent.usage.requests == 1  # Usage is billed even for failed turns.


class Answer(BaseModel):
    value: int


def test_structured_output_uses_parse() -> None:
    client = FakeClient([ParsedResponse(_text('{"value": 42}'), Answer(value=42))])
    agent = ClaudeAgent(client=client, output_format=Answer)  # type: ignore[arg-type]

    answer = agent.ask("Answer?")

    assert answer == Answer(value=42)
    call = client.messages.parse_calls[0]
    assert call["output_format"] is Answer
    assert call["extra_body"] == {"cache_control": {"type": "ephemeral"}}
    assert "cache_control" not in call


def test_structured_output_refusal() -> None:
    client = FakeClient([ParsedResponse(_text("No.", stop_reason="refusal"), None)])
    agent = ClaudeAgent(client=client, output_format=Answer)  # type: ignore[arg-type]
    with pytest.raises(RefusalError):
        agent.ask("Answer?")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"output_format": dict},
        {"max_tokens": 0},
        {"max_tool_rounds": -1},
        {"cache_ttl": "2h"},
        {"tool_choice": {"type": "sometimes"}},
    ],
)
def test_invalid_configuration(kwargs: dict[str, Any]) -> None:
    with pytest.raises((TypeError, ValueError)):
        ClaudeAgent(client=FakeClient([]), **kwargs)  # type: ignore[arg-type]


def test_duplicate_tools_and_empty_prompt() -> None:
    agent = ClaudeAgent(client=FakeClient([]), tools=[add])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Duplicate"):
        agent.add_tool(add)
    with pytest.raises(ValueError, match="nonempty"):
        agent.ask("   ")


def test_reset_keeps_usage() -> None:
    client = FakeClient([_text("a")])
    agent = ClaudeAgent(client=client)  # type: ignore[arg-type]
    agent.ask("x")
    agent.reset()
    assert agent.messages == [] and agent.last_response is None
    assert agent.usage.requests == 1
    agent.reset_usage()
    assert agent.usage.requests == 0


def test_remove_tool() -> None:
    agent = ClaudeAgent(tools=[add], client=FakeClient([]))  # type: ignore[arg-type]
    assert agent.remove_tool("add") is add
    assert agent.tools == {}
    with pytest.raises(KeyError, match="Unknown tool: add"):
        agent.remove_tool("add")
