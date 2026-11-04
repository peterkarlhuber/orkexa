"""ORK2-01: a tool call that times out or whose task is cancelled stops its sandbox call.

Before the fix, ``Agent(tool_timeout=...)`` abandoned the worker thread and ``AsyncAgent`` task cancellation
left ``asyncio.to_thread`` running, so the sandbox process (or container) lived on until the sandbox's own
timeout. Now the drivers bind a per-call cancel check (linked to the caller's ``cancel_scope``) and set it.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import threading
import time
import warnings

import pytest

from orkexa import Agent, AsyncAgent
from orkexa._cancel import bind_cancel, cancel_requested, linked_check, reset_cancel
from orkexa.dynamic import SandboxedTool, SubprocessSandbox, cancel_scope
from orkexa.providers.fake import FakeModel, text, tool_calls

pytestmark = pytest.mark.skipif(os.name != "posix", reason="checks the process by its pid")

SCHEMA = {
    "type": "object",
    "properties": {"seconds": {"type": "number"}},
    "required": ["seconds"],
    "additionalProperties": False,
}


def _tool(pid_file: Path) -> tuple[SandboxedTool, SubprocessSandbox]:
    code = (
        "import os, time\n"
        "def slow(seconds: float) -> str:\n"
        '    """Sleep.\n\n    Args:\n        seconds: How long.\n\n    Returns:\n        done.\n    """\n'
        f"    with open({str(pid_file)!r}, 'w') as f:\n"
        "        f.write(str(os.getpid()))\n"
        "    time.sleep(seconds)\n"
        "    return 'done'\n"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sandbox = SubprocessSandbox(acknowledge_unsafe=True, timeout=180)
    tool = SandboxedTool(
        name="slow",
        description="Sleep",
        input_schema=SCHEMA,
        output_schema=None,
        code=code,
        function_name="slow",
        sandbox=sandbox,
    )
    return tool, sandbox


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _timeout_after_start(tool: SandboxedTool, pid_file: Path) -> float:
    """A ``tool_timeout`` that fires only after the sandbox call has started, also on a slow runner.

    The timeout must hit a *running* call (that is what these tests prove). A fixed 1 s raced the start
    of the sandbox process on slow CI machines (macOS Intel): the call was stopped before its code ran,
    so there was no process to check. One warm-up call measures how long a start takes here."""
    begin = time.monotonic()
    tool.execute({"seconds": 0})
    startup = time.monotonic() - begin
    pid_file.unlink()
    return max(1.0, 4 * startup + 1.0)


def _pid(pid_file: Path, within: float = 60.0) -> int:
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        if pid_file.exists() and pid_file.read_text():
            return int(pid_file.read_text())
        time.sleep(0.02)
    raise AssertionError("the sandbox call never started")


def _gone(pid: int, within: float = 3.0) -> bool:
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return False


def test_agent_tool_timeout_stops_the_sandbox_call(tmp_path: Path) -> None:
    pid_file = tmp_path / "pid"
    tool, sandbox = _tool(pid_file)
    timeout = _timeout_after_start(tool, pid_file)
    model = FakeModel([tool_calls(("slow", {"seconds": timeout + 60})), text("ok")])
    agent = Agent(model="fake:x", client=model, tools=[tool], tool_timeout=timeout, raise_tool_errors=False)
    try:
        assert agent.ask("go") == "ok"
        assert _gone(_pid(pid_file)), "the sandbox process outlived the tool timeout"
    finally:
        sandbox.cancel()


def test_async_agent_tool_timeout_stops_the_sandbox_call(tmp_path: Path) -> None:
    pid_file = tmp_path / "pid"
    tool, sandbox = _tool(pid_file)
    timeout = _timeout_after_start(tool, pid_file)
    model = FakeModel([tool_calls(("slow", {"seconds": timeout + 60})), text("ok")])
    agent = AsyncAgent(
        model="fake:x", client=model, tools=[tool], tool_timeout=timeout, raise_tool_errors=False
    )

    async def main() -> bool:
        assert await agent.ask("go") == "ok"
        # Checked inside the loop: asyncio.run() would wait for an abandoned worker thread at its end.
        return await asyncio.to_thread(_gone, _pid(pid_file))

    try:
        assert asyncio.run(main()), "the sandbox process outlived the tool timeout"
    finally:
        sandbox.cancel()


def test_cancelling_async_ask_stops_the_sandbox_call(tmp_path: Path) -> None:
    pid_file = tmp_path / "pid"
    tool, sandbox = _tool(pid_file)
    model = FakeModel([tool_calls(("slow", {"seconds": 20})), text("ok")])
    agent = AsyncAgent(model="fake:x", client=model, tools=[tool])

    async def main() -> bool:
        task = asyncio.ensure_future(agent.ask("go"))
        pid = await asyncio.to_thread(_pid, pid_file)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return await asyncio.to_thread(_gone, pid)

    try:
        assert asyncio.run(main()), "the sandbox process outlived the cancelled ask()"
    finally:
        sandbox.cancel()


def test_the_callers_cancel_scope_still_stops_a_call_with_a_timeout(tmp_path: Path) -> None:
    pid_file = tmp_path / "pid"
    tool, sandbox = _tool(pid_file)
    model = FakeModel([tool_calls(("slow", {"seconds": 20})), text("ok")])
    agent = Agent(model="fake:x", client=model, tools=[tool], tool_timeout=30.0, raise_tool_errors=False)
    stop = threading.Event()
    threading.Thread(target=lambda: (_pid(pid_file), stop.set()), daemon=True).start()
    started = time.monotonic()
    try:
        with cancel_scope(stop.is_set):
            assert agent.ask("go") == "ok"
        assert time.monotonic() - started < 10
        assert "cancelled" in str(model.requests[-1]).lower()
    finally:
        sandbox.cancel()


def test_linked_check() -> None:
    stop = threading.Event()
    assert linked_check(stop)() is False
    stop.set()
    assert linked_check(stop)() is True
    outer = threading.Event()
    token = bind_cancel(outer.is_set)
    try:
        own = threading.Event()
        check = linked_check(own)
        assert not check()
        outer.set()
        assert check() and not own.is_set()
        broken = bind_cancel(lambda: 1 / 0 == 0)
        try:
            assert not cancel_requested(linked_check(threading.Event()))  # a broken outer check never cancels
        finally:
            reset_cancel(broken)
    finally:
        reset_cancel(token)
