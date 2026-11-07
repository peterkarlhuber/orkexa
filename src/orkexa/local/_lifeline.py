"""Tie the managed Ollama server's lifetime to the process that started it.

:class:`~orkexa.local.server.ManagedOllama` starts ``ollama serve`` in a session (POSIX) or process group
(Windows) of its own, so a signal to the application's group does not reach it and its model runners stay in
its group. Without more, a server outlives an application that is killed (a crash, an OOM kill, a stop
timeout) and keeps its loaded model in memory until the next start reclaims it from the state file.

* **POSIX:** a small watcher process (this file, run as a script by the same interpreter with ``-I``) holds
  the read end of a pipe whose write end only the application holds. When the application stops the server
  or ends in any way, the pipe closes; the watcher then stops the server's process group (SIGTERM, SIGKILL
  after a timeout) if that process is still ``<binary> serve`` leading its own group, and exits. The watcher
  runs in a session of its own too, so killing the application's process group does not take it along.
  A process forked from the application without ``exec`` (``multiprocessing`` with ``fork``) keeps the pipe
  open until it ends.
* **Windows:** the server is put into a job object with ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`` whose only
  handle the application holds; Windows closes it when the application ends, and that ends every process of
  the job (the server and its runners).

This module uses only the standard library and no relative imports, so the watcher runs it as a plain script.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
from typing import Any

#: Seconds between SIGTERM and SIGKILL when a server is stopped by the watcher or the reclaim.
STOP_TIMEOUT = 10.0


def alive(pid: int) -> bool:
    """Whether a process ``pid`` exists (POSIX; on Windows ``False``: nothing is reclaimed there)."""
    if pid <= 0 or platform.system() == "Windows":
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def runs_serve(pid: int, binary: str) -> bool:
    """Whether the command line of ``pid`` ends with ``<binary> serve`` (maybe after an interpreter)."""
    proc = Path(f"/proc/{pid}/cmdline")
    if proc.exists():
        try:
            args = proc.read_bytes().rstrip(b"\0").decode("utf-8", "replace").split("\0")
        except OSError:
            return False
        return args[-2:] == [binary, "serve"]
    try:  # macOS and other POSIX systems without /proc
        out = subprocess.run(
            # -ww: the whole command line; without it BSD ps may cut it to the screen width.
            ["ps", "-ww", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    text = out.stdout.strip()
    return text == f"{binary} serve" or text.endswith(f" {binary} serve")


def is_managed_server(pid: int, binary: str) -> bool:
    """Whether ``pid`` is ``<binary> serve`` leading its own process group, as ManagedOllama starts it."""
    if not alive(pid):
        return False
    try:
        if os.getpgid(pid) != pid:
            return False
    except OSError:
        return False
    return runs_serve(pid, binary)


def stop_group(pid: int, timeout: float = STOP_TIMEOUT) -> None:
    """SIGTERM, then SIGKILL, to the process group of a server that is not our child (polled)."""
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, signal.SIGTERM)
        deadline = time.monotonic() + timeout
        while alive(pid) and time.monotonic() < deadline:
            time.sleep(0.1)
        os.killpg(pid, signal.SIGKILL)


# ---- POSIX: the watcher ----------------------------------------------------------------------------


def start_watcher(pid: int, binary: str) -> subprocess.Popen[bytes] | None:
    """Start the watcher of the server ``pid`` (POSIX); ``None`` when no interpreter can run it.

    A frozen application (``sys.frozen``) has no interpreter of its own to run this file with, and a packaged
    module may have no file on disk: then nothing is started, and the reclaim at the next start stays the
    only cleanup. The caller keeps the returned process; :func:`release_watcher` ends it."""
    script = Path(__file__)
    if getattr(sys, "frozen", False) or not sys.executable or not script.is_file():
        return None
    try:
        return subprocess.Popen(
            # -I: no PYTHON* variables, no user site, and not this folder on sys.path (it needs none).
            [sys.executable, "-I", str(script), str(pid), binary],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
        )
    except OSError:
        return None


def release_watcher(watcher: subprocess.Popen[bytes] | None, timeout: float = 5.0) -> None:
    """Close the watcher's pipe after the server was stopped; the watcher finds no server and exits."""
    if watcher is None:
        return
    with contextlib.suppress(OSError):
        if watcher.stdin is not None:
            watcher.stdin.close()
    try:
        watcher.wait(timeout)
    except subprocess.TimeoutExpired:
        watcher.kill()
        watcher.wait(5)


def watch(pid: int, binary: str, stream: Any) -> bool:
    """Block until ``stream`` ends, then stop the server ``pid`` if it is still the managed server.

    Returns whether it was stopped (``False`` after a regular stop: the server is gone already)."""
    with contextlib.suppress(OSError):
        while stream.read(4096):
            pass
    if is_managed_server(pid, binary):
        stop_group(pid)
        return True
    return False


# ---- Windows: a job object -------------------------------------------------------------------------


def kill_on_close_job(process_handle: int) -> int | None:
    """Put a process into a new job object that ends it when its last handle closes (Windows).

    Returns the job's handle, which the caller keeps open (see :func:`close_job`); ``None`` when the job
    cannot be created or the process cannot be assigned (for example a process in a job that does not allow
    nesting): the server then runs as before, without the binding."""
    import ctypes
    from ctypes import wintypes

    class IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class BasicLimits(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", BasicLimits),
            ("IoInfo", IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    extended_limit_information = 9  # JobObjectExtendedLimitInformation
    kill_on_job_close = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.SetInformationJobObject.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    )
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        return None
    info = ExtendedLimits()
    info.BasicLimitInformation.LimitFlags = kill_on_job_close
    size = ctypes.sizeof(info)
    ok = kernel32.SetInformationJobObject(job, extended_limit_information, ctypes.byref(info), size)
    ok = ok and kernel32.AssignProcessToJobObject(job, wintypes.HANDLE(process_handle))
    if not ok:
        close_job(int(job))
        return None
    return int(job)


def close_job(job: int | None) -> None:
    """Close a job handle from :func:`kill_on_close_job`: every process still in the job ends (Windows)."""
    if job is None:
        return
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle(wintypes.HANDLE(job))


if __name__ == "__main__":  # the watcher: python -I _lifeline.py <pid> <binary>
    watch(int(sys.argv[1]), sys.argv[2], sys.stdin.buffer)
