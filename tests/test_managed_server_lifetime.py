"""The managed Ollama server ends with the process that started it (nodestro-desktop DSK-022). No network.

POSIX: a watcher process stops the server's group when the application ends without stopping it. Windows: a
job object with ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`` ends the server with the application's last handle.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
import signal
import subprocess
import sys
import textwrap
import time

import pytest

from orkexa.local import ManagedOllama, _lifeline

from .audit3.test_ork3_04_managed_server import _alive, _fake_ollama, _gone, _start
from .audit3.test_ork3_05_orphaned_server import _crashed_app

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX: the watcher process")
windows_only = pytest.mark.skipif(sys.platform != "win32", reason="Windows: the job object")


@posix_only
def test_a_killed_application_does_not_leave_its_server_running(tmp_path: Path) -> None:
    home, binary = tmp_path / "home", _fake_ollama(tmp_path)
    left = _crashed_app(home, binary, watcher_too=False)
    # The watcher stops the group: SIGTERM, which the fake server obeys at once.
    assert _gone(left["server"], 20) and _gone(left["runner"], 20)
    assert _gone(left["watcher"], 20)


@posix_only
def test_a_regular_stop_ends_the_watcher_without_a_second_stop(tmp_path: Path) -> None:
    server = ManagedOllama(tmp_path / "home", binary=_fake_ollama(tmp_path))
    _start(server)
    watcher = server._lifeline
    assert watcher is not None and watcher.poll() is None
    assert os.getsid(watcher.pid) == watcher.pid  # its own session: a kill of the app's group misses it
    pid = server.status().pid
    assert pid is not None
    server.stop()
    assert watcher.returncode == 0 and server._lifeline is None
    assert _gone(pid)


@posix_only
def test_a_restart_replaces_the_watcher(tmp_path: Path) -> None:
    server = ManagedOllama(tmp_path / "home", binary=_fake_ollama(tmp_path))
    _start(server)
    first = server._lifeline
    try:
        server.restart()
        assert first is not None and first.returncode == 0
        assert server._lifeline is not None and server._lifeline.poll() is None
    finally:
        server.stop()


@posix_only
def test_the_watcher_never_stops_a_process_that_is_not_the_server(tmp_path: Path) -> None:
    binary = _fake_ollama(tmp_path)
    stranger = subprocess.Popen(["sleep", "60"], start_new_session=True)
    try:
        # A recycled pid: alive and leading its own session, but not `<binary> serve`.
        assert _lifeline.watch(stranger.pid, str(binary), io.BytesIO(b"")) is False
        assert stranger.poll() is None
    finally:
        os.kill(stranger.pid, signal.SIGKILL)
        stranger.wait()


@posix_only
def test_the_watcher_waits_for_the_end_of_its_pipe(tmp_path: Path) -> None:
    binary = _fake_ollama(tmp_path)
    stranger = subprocess.Popen(["sleep", "60"], start_new_session=True)
    watcher = _lifeline.start_watcher(stranger.pid, str(binary))
    try:
        assert watcher is not None
        time.sleep(0.5)
        assert watcher.poll() is None  # still waiting: the pipe is open
        _lifeline.release_watcher(watcher)
        assert watcher.returncode == 0
        assert stranger.poll() is None
    finally:
        os.kill(stranger.pid, signal.SIGKILL)
        stranger.wait()


@posix_only
def test_a_frozen_application_starts_no_watcher(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    server = ManagedOllama(tmp_path / "home", binary=_fake_ollama(tmp_path))
    _start(server)
    try:
        assert server._lifeline is None and server.status().running  # the reclaim stays the only cleanup
    finally:
        server.stop()


# ---- Windows -----------------------------------------------------------------------------------------------


def _exited(pid: int, seconds: float) -> bool:
    """Whether the Windows process ``pid`` ended within ``seconds``."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    synchronize = 0x00100000
    handle = kernel32.OpenProcess(synchronize, False, pid)
    if not handle:
        return True  # no such process any more
    try:
        return int(kernel32.WaitForSingleObject(handle, int(seconds * 1000))) == 0  # WAIT_OBJECT_0
    finally:
        kernel32.CloseHandle(handle)


_JOB_OWNER = textwrap.dedent(
    """
    import subprocess, sys, time
    from orkexa.local import _lifeline
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    job = _lifeline.kill_on_close_job(int(child._handle))
    print(child.pid, job is not None, flush=True)
    if sys.argv[1] == "close":
        _lifeline.close_job(job)
        print(child.wait(30), flush=True)
    time.sleep(120)
    """
)


@windows_only
@pytest.mark.parametrize("ending", ["killed", "close"])
def test_a_job_ends_the_server_with_its_owner(ending: str) -> None:
    owner = subprocess.Popen([sys.executable, "-c", _JOB_OWNER, ending], stdout=subprocess.PIPE, text=True)
    try:
        assert owner.stdout is not None
        pid_text, bound = owner.stdout.readline().split()
        child = int(pid_text)
        assert bound == "True"
        assert not _exited(child, 0.5)  # it runs while its owner holds the job
        if ending == "killed":
            owner.kill()  # TerminateProcess: no cleanup code of the owner runs
            owner.wait(30)
        else:
            assert owner.stdout.readline().strip() != ""  # close_job ended it; wait() returned
        assert _exited(child, 20)
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait(30)


def test_alive_is_false_for_impossible_pids() -> None:
    assert _lifeline.alive(0) is False and _lifeline.alive(-1) is False
    if sys.platform != "win32":
        assert _alive(os.getpid())
