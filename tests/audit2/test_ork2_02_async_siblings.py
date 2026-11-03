"""ORK2-02: in AsyncAgent a failing parallel tool call cancels and awaits its siblings."""

from __future__ import annotations

import asyncio

import pytest

from orkexa import AsyncAgent, Tool, ToolExecutionError
from orkexa.providers.fake import FakeModel, text, tool_calls


@Tool
async def fail(x: int) -> str:
    """Fails at once.

    Args:
        x: Unused.

    Returns:
        Never.
    """
    raise RuntimeError("boom")


def test_a_failing_parallel_call_does_not_leave_siblings_running() -> None:
    finished: list[str] = []
    after: list[str] = []

    @Tool
    async def slow(x: int) -> str:
        """Commits a side effect after a delay.

        Args:
            x: Id.

        Returns:
            ok.
        """
        await asyncio.sleep(0.3)
        finished.append("slow")
        return "ok"

    async def main() -> None:
        model = FakeModel([tool_calls(("slow", {"x": 1}), ("fail", {"x": 0})), text("done")])
        agent = AsyncAgent(
            model="fake:x",
            client=model,
            tools=[fail, slow],
            max_parallel_tools=4,
            after_tool=lambda call, content, is_error: after.append(call.name),
        )
        with pytest.raises(ToolExecutionError):
            await agent.ask("go")
        await asyncio.sleep(0.6)
        assert not [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

    asyncio.run(main())
    assert finished == [], "a sibling tool call kept running and committed after the turn had failed"
    assert after == []


def test_parallel_calls_still_run_concurrently_and_keep_their_order() -> None:
    @Tool
    async def wait(x: int) -> str:
        """Waits.

        Args:
            x: Id.

        Returns:
            The id.
        """
        await asyncio.sleep(0.2)
        return str(x)

    async def main() -> float:
        model = FakeModel([tool_calls(*(("wait", {"x": i}) for i in range(4))), text("done")])
        agent = AsyncAgent(model="fake:x", client=model, tools=[wait], max_parallel_tools=4)
        loop = asyncio.get_running_loop()
        started = loop.time()
        assert await agent.ask("go") == "done"
        results = [part for message in model.requests[-1].messages for part in message.content]
        assert [getattr(p, "content", None) for p in results if type(p).__name__ == "ToolResultPart"] == [
            "0",
            "1",
            "2",
            "3",
        ]
        return loop.time() - started

    assert asyncio.run(main()) < 0.6
