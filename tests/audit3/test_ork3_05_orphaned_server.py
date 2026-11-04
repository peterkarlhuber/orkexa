"""ORK3-05: a managed server left running by a crashed process is stopped on the next start.

Internal backend audit round 3 (2026-11-04). No network.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import textwrap

import pytest

from orkexa.local import ManagedOllama

from .test_ork3_04_managed_server import _alive, _fake_ollama, _gone

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")


def _crashed_app(home: Path, binary: Path) -> dict[str, int]:
    """Start a managed server in a child process, then SIGKILL that process (a crash)."""
    code = textwrap.dedent(
        f"""
        import time
        from orkexa.local import ManagedOllama
        m = ManagedOllama({str(home)!r}, binary={str(binary)!r})
        print(m.start(timeout=20), flush=True)
        time.sleep(120)
        """
    )
    app = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    assert app.stdout is not None and app.stdout.readline().startswith("http://")
    app.kill()
    app.wait()
    state = json.loads((home / "ollama.state.json").read_text())
    runner = int((home / "ollama-home" / "runner.pid").read_text())
    assert _alive(state["pid"]) and _alive(runner)
    return {"server": state["pid"], "runner": runner}


def test_next_start_stops_the_orphaned_server_and_its_runners(tmp_path: Path) -> None:
    home, binary = tmp_path / "home", _fake_ollama(tmp_path)
    orphan = _crashed_app(home, binary)
    server = ManagedOllama(home, binary=binary)
    try:
        server.start(timeout=20)
        assert _gone(orphan["server"]) and _gone(orphan["runner"])
        state = json.loads((home / "ollama.state.json").read_text())
        assert state["pid"] == server.status().pid != orphan["server"]
    finally:
        server.stop()


def test_a_server_whose_owner_still_runs_is_left_alone(tmp_path: Path) -> None:
    home, binary = tmp_path / "home", _fake_ollama(tmp_path)
    other = ManagedOllama(home, binary=binary)
    other.start(timeout=20)
    state = json.loads((home / "ollama.state.json").read_text())
    # the same state as another live process of the application would leave (owner = a live pid)
    keeper = subprocess.Popen(["sleep", "60"])
    try:
        state["owner"] = keeper.pid
        (home / "ollama.state.json").write_text(json.dumps(state))
        ManagedOllama(home, binary=binary)._reclaim_orphan()
        assert _alive(state["pid"]) and (home / "ollama.state.json").exists()
    finally:
        keeper.kill()
        keeper.wait()
        other.stop()


@pytest.mark.parametrize("content", ["not json", "{}", "OTHER"])
def test_unrelated_processes_are_never_stopped_and_stale_files_go(tmp_path: Path, content: str) -> None:
    home, binary = tmp_path / "home", _fake_ollama(tmp_path)
    home.mkdir()
    stranger = subprocess.Popen(["sleep", "60"], start_new_session=True)
    try:
        if content == "OTHER":  # a recycled pid: alive, own session, but not `<binary> serve`
            content = json.dumps(
                {"pid": stranger.pid, "binary": str(binary), "owner": 999_999_999, "port": 1}
            )
        (home / "ollama.state.json").write_text(content)
        ManagedOllama(home, binary=binary)._reclaim_orphan()
        assert stranger.poll() is None
        assert not (home / "ollama.state.json").exists()
    finally:
        os.kill(stranger.pid, signal.SIGKILL)
        stranger.wait()
