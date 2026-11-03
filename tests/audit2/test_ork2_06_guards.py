"""ORK2-06: tests that fail when the process-group kill or the sandbox payload cap is removed.

Both guards survived the whole offline suite under mutation (audit round 2, M07 and M08).
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import threading
import time
from typing import Any

import pytest

from orkexa import ToolExecutionError
from orkexa.dynamic import SandboxCancelled, SandboxedTool, SandboxError, run_bounded
from orkexa.dynamic.sandbox import Sandbox
from orkexa.dynamic.sandboxed_tool import _MAX_PAYLOAD_CHARS

# The leader starts a grandchild in the same session, reports its pid and waits.
_SPAWN = (
    "import subprocess, sys, time\n"
    "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
    "open(sys.argv[1], 'w').write(str(child.pid))\n"
    "time.sleep(60)\n"
)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:  # a zombie (killed, not yet reaped by its parent, here init) counts as gone
        with open(f"/proc/{pid}/stat") as stat:
            return stat.read().split(")")[-1].split()[0] != "Z"
    except OSError:
        return True


def _wait_for_pid(path: Path) -> int:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if path.exists() and path.read_text():
            return int(path.read_text())
        time.sleep(0.02)
    raise AssertionError("the grandchild never started")


def _gone(pid: int) -> bool:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return False


@pytest.mark.skipif(sys.platform != "linux", reason="process groups and /proc")
@pytest.mark.parametrize("how", ["cancel", "timeout"])
def test_a_stopped_run_kills_the_whole_process_group(tmp_path: Path, how: str) -> None:
    pid_file = tmp_path / "grandchild"
    stop = threading.Event()
    grandchild: list[int] = []

    def watch() -> None:
        grandchild.append(_wait_for_pid(pid_file))
        if how == "cancel":
            stop.set()

    watcher = threading.Thread(target=watch)
    watcher.start()
    expected = SandboxCancelled if how == "cancel" else SandboxError
    try:
        with pytest.raises(expected):
            run_bounded(
                [sys.executable, "-c", _SPAWN, str(pid_file)],
                b"",
                timeout=30 if how == "cancel" else 2,
                max_stdout=1024,
                stop=stop,
                new_session=True,
            )
    finally:
        watcher.join(20)
    assert grandchild, "the grandchild never started"
    survived = not _gone(grandchild[0])
    if survived:
        os.kill(grandchild[0], 9)
    assert not survived, "a grandchild survived the stopped run (process group not killed)"


class _Recording(Sandbox):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, Any]] = []

    def _execute(self, workdir: Path, request: str) -> str:  # pragma: no cover - never reached here
        raise AssertionError("the sandbox must not run")

    def call(self, code: str, function_name: str, payload: dict[str, Any]) -> str:
        self.calls.append(payload)
        return "ok"


def _tool(sandbox: Sandbox) -> SandboxedTool:
    schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    }
    return SandboxedTool(
        name="echo",
        description="Echo",
        input_schema=schema,
        output_schema=None,
        code="def echo(text: str) -> str:\n    return text\n",
        function_name="echo",
        sandbox=sandbox,
    )


def test_an_oversized_payload_never_reaches_the_sandbox() -> None:
    assert _MAX_PAYLOAD_CHARS == 100_000
    sandbox = _Recording()
    tool = _tool(sandbox)
    assert tool.execute({"text": "a" * 1000}) == "ok"
    with pytest.raises(ToolExecutionError, match="too large"):
        tool.execute({"text": "a" * _MAX_PAYLOAD_CHARS})
    with pytest.raises(ToolExecutionError, match="too large"):
        tool(text="b" * 200_000)  # a direct Python call takes the same path
    assert len(sandbox.calls) == 1
