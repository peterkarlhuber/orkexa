"""2. One ``@Tool`` used by three native providers.

The tool schema is derived once from the type hints and docstring; each adapter
renders it in its provider's format (Anthropic tool, OpenAI strict function,
Gemini function declaration). Orkexa always executes the tool itself.
"""

from __future__ import annotations

from _setup import available_models

from orkexa import Agent, Tool


@Tool
def convert_currency(amount: float, rate: float) -> float:
    """Convert an amount with an exchange rate.

    Args:
        amount: Amount in the source currency.
        rate: Units of the target currency per unit of the source currency.

    Returns:
        The amount in the target currency, rounded to cents.
    """
    return round(amount * rate, 2)


def main() -> None:
    for model in available_models():
        agent = Agent(model=model, tools=[convert_currency])
        answer = agent.ask("How much is 250 EUR in USD at a rate of 1.08? Use the tool.")
        print(f"{model}: {answer}")


if __name__ == "__main__":
    main()
