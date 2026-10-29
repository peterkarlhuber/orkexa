"""4. A multi-round tool loop, synchronous and asynchronous (with parallel tools)."""

from __future__ import annotations

import asyncio

from _setup import model

from orkexa import Agent, AsyncAgent, Tool, ToolCall

PRICES = {"apple": 0.5, "pear": 0.8, "melon": 3.0}


@Tool
def price(item: str) -> float:
    """Look up the unit price of a fruit.

    Args:
        item: Fruit name, e.g. "apple".

    Returns:
        The unit price in EUR.
    """
    if item not in PRICES:
        raise ValueError(f"unknown item {item!r}; known: {sorted(PRICES)}")
    return PRICES[item]


@Tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers.

    Args:
        a: First factor.
        b: Second factor.

    Returns:
        The product.
    """
    return a * b


@Tool
async def stock(item: str) -> int:
    """Units of a fruit in stock (asynchronous lookup).

    Args:
        item: Fruit name.

    Returns:
        Units in stock.
    """
    await asyncio.sleep(0.1)
    return {"apple": 40, "pear": 12, "melon": 3}.get(item, 0)


def log_call(call: ToolCall) -> None:
    print(f"  -> {call.name}({call.input})")


def sync_demo(name: str) -> None:
    agent = Agent(model=name, tools=[price, multiply], raise_tool_errors=False, before_tool=log_call)
    print(agent.ask("What do 7 apples and 3 pears cost? Look up prices, then multiply."))


async def async_demo(name: str) -> None:
    agent = AsyncAgent(model=name, tools=[stock], max_parallel_tools=3, before_tool=log_call)
    print(await agent.ask("How many apples, pears and melons are in stock?"))


def main() -> None:
    name = model()
    sync_demo(name)
    asyncio.run(async_demo(name))


if __name__ == "__main__":
    main()
