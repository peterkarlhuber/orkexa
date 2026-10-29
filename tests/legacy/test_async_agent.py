from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel
import pytest

from claude_agent import AsyncClaudeAgent, Tool, ToolCall, ToolExecutionError

from .conftest import AsyncFakeClient, ParsedResponse, make_message


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


def _text(text: str) -> Any:
    return make_message([{"type": "text", "text": text}])


def _tool_uses(*calls: tuple[str, dict[str, Any]]) -> Any:
    return make_message(
        [
            {"type": "tool_use", "id": f"toolu_{i}", "name": name, "input": payload}
            for i, (name, payload) in enumerate(calls)
        ],
        stop_reason="tool_use",
    )


def _agent(client: AsyncFakeClient, **kwargs: Any) -> AsyncClaudeAgent[Any]:
    return AsyncClaudeAgent(client=client, **kwargs)  # type: ignore[arg-type]


def test_async_ask_with_sync_tool_commits_history() -> None:
    client = AsyncFakeClient([_tool_uses(("add", {"a": 2, "b": 3})), _text("5")])
    agent = _agent(client, tools=[add])
    assert asyncio.run(agent.ask("2+3?")) == "5"
    assert agent.messages[2]["content"][0]["content"] == "5"
    assert agent.usage.requests == 2


def test_async_tools_run_concurrently_and_hooks_may_be_async() -> None:
    both_started = asyncio.Event()
    started: list[str] = []

    @Tool
    async def wait(label: str) -> str:
        """Wait until another call runs at the same time.

        Args:
            label: Echoed back.

        Returns:
            The label.
        """
        started.append(label)
        if len(started) == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=5)
        return label

    seen: list[str] = []

    async def before_tool(call: ToolCall) -> None:
        seen.append(call.input["label"])

    client = AsyncFakeClient([_tool_uses(("wait", {"label": "a"}), ("wait", {"label": "b"})), _text("ok")])
    agent = _agent(client, tools=[wait], max_parallel_tools=2, before_tool=before_tool)
    assert asyncio.run(agent.ask("Go")) == "ok"
    assert [r["content"] for r in agent.messages[2]["content"]] == ["a", "b"]
    assert sorted(seen) == ["a", "b"]


def test_async_streaming_and_structured_output() -> None:
    class Answer(BaseModel):
        value: int

    chunks: list[str] = []

    async def on_text(text: str) -> None:
        chunks.append(text)

    client = AsyncFakeClient([ParsedResponse(_text('{"value": 1}'), Answer(value=1))])
    agent = AsyncClaudeAgent(client=client, output_format=Answer, on_text=on_text)  # type: ignore[arg-type]
    assert asyncio.run(agent.ask("One?")) == Answer(value=1)
    assert chunks == ['{"value": 1}']
    assert client.messages.stream_calls


def test_async_parse_without_streaming() -> None:
    class Answer(BaseModel):
        value: int

    client = AsyncFakeClient([ParsedResponse(_text('{"value": 2}'), Answer(value=2))])
    agent = AsyncClaudeAgent(client=client, output_format=Answer)  # type: ignore[arg-type]
    assert asyncio.run(agent.ask("Two?")) == Answer(value=2)
    assert client.messages.parse_calls


def test_async_tool_errors_follow_raise_tool_errors() -> None:
    client = AsyncFakeClient([_tool_uses(("add", {"a": "x", "b": 1}))])
    with pytest.raises(ToolExecutionError):
        asyncio.run(_agent(client, tools=[add]).ask("Sum"))

    client = AsyncFakeClient([_tool_uses(("add", {"a": "x", "b": 1})), _text("sorry")])
    agent = _agent(client, tools=[add], raise_tool_errors=False)
    assert asyncio.run(agent.ask("Sum")) == "sorry"
    assert agent.messages[2]["content"][0]["is_error"] is True


def test_async_count_tokens() -> None:
    client = AsyncFakeClient([])
    assert asyncio.run(_agent(client).count_tokens("Hi")) == 1234
