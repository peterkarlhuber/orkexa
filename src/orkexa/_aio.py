"""Small asyncio helpers shared by the async drivers."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import TypeVar

__all__ = ["gather_or_cancel"]

T = TypeVar("T")


async def gather_or_cancel(*awaitables: Awaitable[T]) -> list[T]:
    """Run ``awaitables`` concurrently and return their results in order.

    Unlike :func:`asyncio.gather`, a failure (or the cancellation of the caller) cancels the others and
    waits until they have ended before it is raised: no sibling keeps running, commits a side effect or
    calls a hook after the caller gave up (ORK2-02).
    """
    if len(awaitables) == 1:
        return [await awaitables[0]]
    tasks = [asyncio.ensure_future(item) for item in awaitables]
    try:
        return list(await asyncio.gather(*tasks))
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
