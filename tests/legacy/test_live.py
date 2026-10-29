"""A minimal smoke test against the real Claude API.

Runs only when ``ANTHROPIC_API_KEY`` is set: three small Claude Haiku 5.5 requests
at effort ``low`` (well under one cent). Skip it with ``pytest -m "not live"``.
"""

from __future__ import annotations

from decimal import Decimal
import os

from pydantic import BaseModel
import pytest

from claude_agent import ClaudeAgent, Tool

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="needs ANTHROPIC_API_KEY"),
]

MODEL = "claude-haiku-5-5"


@Tool
def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: The first number.
        b: The second number.

    Returns:
        The sum.
    """
    return a + b


class Capital(BaseModel):
    country: str
    city: str


def test_text_tool_and_structured_output() -> None:
    chat = ClaudeAgent(model=MODEL, effort="low", max_tokens=1024, system_prompt="Answer in one word.")
    assert "paris" in chat.ask("What is the capital of France?").lower()

    calculator = ClaudeAgent(model=MODEL, effort="low", max_tokens=1024, tools=[add])
    assert "4233" in calculator.ask("Use the add tool to compute 1234 + 2999.").replace(",", "")
    assert any(
        block.get("type") == "tool_use"
        for message in calculator.messages
        for block in message["content"]
        if isinstance(block, dict)
    )

    extractor = ClaudeAgent(model=MODEL, effort="low", max_tokens=1024, output_format=Capital)
    capital = extractor.ask("Italy")
    assert capital.city.lower() in ("rome", "roma")

    costs = [agent.usage.estimated_cost_usd for agent in (chat, calculator, extractor)]
    assert all(cost is not None for cost in costs)
    assert sum(cost for cost in costs if cost is not None) < Decimal("0.01")
