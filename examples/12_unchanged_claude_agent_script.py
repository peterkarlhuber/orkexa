"""Example: an agent with a tool, structured output and cost tracking.

Run with ANTHROPIC_API_KEY set:  python examples/weather_agent.py
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from claude_agent import ClaudeAgent, Tool, UsageTracker


class Location(BaseModel):
    city: str
    country_code: str


@Tool
def get_weather(location: Location, unit: Literal["celsius", "fahrenheit"] = "celsius") -> str:
    """Get the current weather for a location.

    Args:
        location: City and ISO country code.
        unit: Temperature unit.

    Returns:
        A short weather description.
    """
    return f"18 degrees {unit}, light rain in {location.city} ({location.country_code})"


class PackingAdvice(BaseModel):
    umbrella: bool
    jacket: bool
    reason: str


def main() -> None:
    tracker = UsageTracker()
    chat = ClaudeAgent(
        name="chat",
        model="claude-opus-5-5",
        tools=[get_weather],
        raise_tool_errors=False,
        usage_tracker=tracker,
    )
    print(chat.ask("What's the weather like in Vienna right now?"))

    advisor = ClaudeAgent(
        name="advisor",
        model="claude-haiku-5-5",
        output_format=PackingAdvice,
        effort="low",
        usage_tracker=tracker,
    )
    advice = advisor.ask("Light rain, 18 degrees celsius. What should I take with me?")
    print(advice.model_dump())

    for agent, usage in tracker.usage_by_agent().items():
        print(f"{agent}: {usage.total_tokens} tokens, ~${usage.estimated_cost_usd}")


if __name__ == "__main__":
    main()
