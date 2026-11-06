"""Optional live tests against a real Ollama server with a local Qwen model (never part of a normal run).

Runs only with ``pytest -m live`` and ``ORKEXA_OLLAMA_LIVE_URL`` set (e.g. ``http://127.0.0.1:11434/v1``);
``ORKEXA_OLLAMA_LIVE_MODEL`` names a thinking model with tool support (default ``qwen3.5:0.8b``). On a CPU
these take minutes. They check what the server accepts in multi-step agent runs (regression for HTTP 400
``invalid message content type: <nil>``): a history with an assistant turn that only thought, tool rounds with
reasoning sent back, and the tool phase followed by the structured answer, streaming and not.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel
import pytest

from orkexa import Agent, CompatibleOptions, Tool
from orkexa.core.conversion import dump_history, to_json
from orkexa.core.runtime import FORMAT_PROMPT
from orkexa.core.types import Message, ReasoningPart, TextPart, ToolCallPart, ToolResultPart

pytestmark = pytest.mark.live

URL = os.environ.get("ORKEXA_OLLAMA_LIVE_URL")
MODEL = os.environ.get("ORKEXA_OLLAMA_LIVE_MODEL", "qwen3.5:0.8b")
skip = pytest.mark.skipif(not URL, reason="needs ORKEXA_OLLAMA_LIVE_URL")


@Tool
def downtime_total(minutes: list[int]) -> int:
    """Add up downtime minutes.

    Args:
        minutes: The downtime of each event in minutes.

    Returns:
        The total in minutes.
    """
    return sum(minutes)


@Tool
def share_percent(part: float, whole: float) -> float:
    """The share of ``part`` in ``whole`` in percent.

    Args:
        part: The part.
        whole: The whole.

    Returns:
        The share in percent.
    """
    return round(part / whole * 100, 4)


def _load(agent: Agent, path: Path, messages: list[Message]) -> None:
    path.write_text(to_json(dump_history(messages)), encoding="utf-8")
    agent.load_history(path)


class Share(BaseModel):
    total_minutes: int
    share_percent: float


def _agent(**kwargs: object) -> Agent:
    return Agent(
        provider="ollama",
        model=MODEL,
        base_url=URL,
        capabilities={"tools": "supported", "reasoning": "supported", "structured_output": "supported"},
        options=CompatibleOptions(send_reasoning_effort=True),
        max_tokens=4000,
        **kwargs,  # type: ignore[arg-type]
    )


@skip
@pytest.mark.parametrize("stream", [False, True])
def test_a_turn_that_only_thought_is_accepted_in_the_history(stream: bool, tmp_path: Path) -> None:
    """The structure of the failed run: tool rounds with reasoning, a tool phase that ends with reasoning
    only, then the request for the structured answer."""
    agent = _agent(output_format=Share, stream=stream, tools=[downtime_total, share_percent], effort="low")
    _load(
        agent,
        tmp_path / "history.json",
        [
            Message(
                role="user", content="Downtime 45, 120 and 30 minutes of 14400 planned. Total and share?"
            ),
            Message(
                role="assistant",
                content=[
                    ReasoningPart(text="Add first."),
                    ToolCallPart(id="call_1", name="downtime_total", arguments={"minutes": [45, 120, 30]}),
                ],
                provider="ollama",
                model=MODEL,
            ),
            Message(
                role="user", content=[ToolResultPart(call_id="call_1", name="downtime_total", content="195")]
            ),
            Message(
                role="assistant",
                content=[
                    ReasoningPart(text="Now the share."),
                    ToolCallPart(id="call_2", name="share_percent", arguments={"part": 195, "whole": 14400}),
                ],
                provider="ollama",
                model=MODEL,
            ),
            Message(
                role="user",
                content=[ToolResultPart(call_id="call_2", name="share_percent", content="1.3542")],
            ),
            Message(
                role="assistant",
                content=[ReasoningPart(text="195 minutes, 1.35 %.")],
                provider="ollama",
                model=MODEL,
            ),
        ],
    )
    result = agent.ask(FORMAT_PROMPT)
    assert isinstance(result, Share)
    assert result.total_minutes == 195


@skip
@pytest.mark.parametrize("stream", [False, True])
def test_tools_then_structured_answer(stream: bool) -> None:
    agent = _agent(
        output_format=Share, stream=stream, tools=[downtime_total, share_percent], max_tool_rounds=6
    )
    result = agent.ask(
        "Downtime events: 45, 120, 30, 25 minutes; planned production time 14400 minutes. "
        "Compute the total with downtime_total and the share with share_percent."
    )
    assert isinstance(result, Share)
    assert result.total_minutes == 220
    tool_turns = [
        m
        for m in agent.conversation()
        if m.role == "assistant" and any(isinstance(p, ToolCallPart) for p in m.parts())
    ]
    assert tool_turns  # the tool phase ran


@skip
def test_a_text_answer_after_an_empty_turn(tmp_path: Path) -> None:
    agent = _agent(effort="low")
    _load(
        agent,
        tmp_path / "history.json",
        [
            Message(role="user", content="Say hello."),
            Message(role="assistant", content=[], provider="ollama", model=MODEL),
            Message(role="user", content="Please answer now."),
            Message(role="assistant", content=[TextPart(text="Hello.")], provider="ollama", model=MODEL),
        ],
    )
    assert isinstance(agent.ask("And goodbye?"), str)
