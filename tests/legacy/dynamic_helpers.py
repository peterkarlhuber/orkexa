"""Shared fixtures for the claude_agent.dynamic tests."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from claude_agent.dynamic import ToolProposal

from .conftest import ParsedResponse, make_message

MARGIN_CODE = '''\
def calculate_operating_margin(revenue: float, operating_income: float) -> float:
    """Calculate the operating margin.

    Args:
        revenue: Total revenue, must be positive.
        operating_income: Operating income in the same currency.

    Returns:
        Operating margin as a fraction of revenue.
    """
    if revenue <= 0:
        raise ValueError("revenue must be positive")
    return operating_income / revenue
'''

MARGIN_TESTS = """\
from tool import calculate_operating_margin


def test_basic():
    assert calculate_operating_margin(200.0, 50.0) == 0.25


def test_invalid_revenue():
    try:
        calculate_operating_margin(0.0, 1.0)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
"""

MARGIN = ToolProposal(
    function_name="calculate_operating_margin",
    description="Operating margin from revenue and operating income.",
    code=MARGIN_CODE,
    tests=MARGIN_TESTS,
)


def parsed(model: BaseModel) -> Any:
    """A scripted messages.parse() response carrying ``model``."""
    return ParsedResponse(make_message([{"type": "text", "text": model.model_dump_json()}]), model)


def tool_use(name: str, payload: dict[str, Any], block_id: str = "toolu_1") -> Any:
    return make_message(
        [{"type": "tool_use", "id": block_id, "name": name, "input": payload}],
        stop_reason="tool_use",
    )


def text(value: str, **kwargs: Any) -> Any:
    return make_message([{"type": "text", "text": value}], **kwargs)


def tool_result(call: dict[str, Any]) -> dict[str, Any]:
    """The last tool_result block sent in a recorded request."""
    blocks = call["messages"][-1]["content"]
    result: dict[str, Any] = blocks[-1]
    assert result["type"] == "tool_result"
    return result
