"""Shared tools for the provider-neutral tests."""

from __future__ import annotations

import asyncio
import time

from orkexa import Tool, ToolResultError


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


@Tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers.

    Args:
        a: First factor.
        b: Second factor.

    Returns:
        The product.
    """
    return a * b


@Tool
def lookup(key: str) -> str:
    """Look up a key.

    Args:
        key: The key.

    Returns:
        The value.
    """
    if key != "known":
        raise ToolResultError(f"unknown key {key!r}")
    return "value"


@Tool
def broken(x: int) -> int:
    """Always fails.

    Args:
        x: Anything.

    Returns:
        Never.
    """
    raise RuntimeError("boom")


@Tool
def slow(seconds: float) -> str:
    """Sleep.

    Args:
        seconds: How long.

    Returns:
        done.
    """
    time.sleep(seconds)
    return "done"


@Tool
async def async_add(a: int, b: int) -> int:
    """Add asynchronously.

    Args:
        a: First summand.
        b: Second summand.

    Returns:
        The sum.
    """
    await asyncio.sleep(0)
    return a + b


@Tool
async def async_slow(seconds: float) -> str:
    """Sleep asynchronously.

    Args:
        seconds: How long.

    Returns:
        done.
    """
    await asyncio.sleep(seconds)
    return "done"
