"""Starts the PDF worker (:mod:`orkexa._pdf_worker`) as a separate, isolated process.

Used by :func:`orkexa.retrieval.pdf_pages_isolated`. Not ``fork`` and not ``multiprocessing``: a dedicated
``subprocess.Popen`` of ``[sys.executable, "-I", "-m", "orkexa._pdf_worker", <pypdf directory>]`` on every
platform, so that

* the child gets an explicitly built minimal environment (an allowlist, never a copy of ``os.environ``): no
  provider keys, database URLs, credentials, proxies, ``HOME`` or ``PATH``;
* it inherits no file descriptors, sockets or database connections (``close_fds=True``, only stdin, stdout and
  stderr as pipes);
* it runs in a fresh empty working directory (mode ``0700``) that is deleted afterwards;
* on POSIX it leads its own process group (``start_new_session=True``), so stopping it reaches every process
  it may have started: after ``timeout`` the group gets ``SIGTERM``, two seconds later ``SIGKILL`` (Windows:
  ``terminate()``, then ``kill()`` of the child);
* its output is bounded: the parent reads at most ``max_chars * 4 + 64 KiB`` bytes;
* its memory is bounded by ``max_memory_mb``: on Linux by ``RLIMIT_AS`` (set by the worker itself), on Windows
  by a job object with a process memory limit (allocations beyond it fail), on macOS, which does not enforce
  ``RLIMIT_AS``, by a watcher thread that reads the child's resident memory (``proc_pidinfo``) every 20 ms
  and kills its process group when it passes the budget.

This is not a general sandbox: the child runs as the same operating-system user, without namespaces, seccomp
or network isolation (see the security notes of :func:`~orkexa.retrieval.pdf_pages_isolated`).
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
from typing import IO, Any

WORKER_MODULE = "orkexa._pdf_worker"
KILL_GRACE_SECONDS = 2.0
OUTPUT_OVERHEAD = 64 * 1024
STDERR_LIMIT = 64 * 1024


@dataclass
class WorkerRun:
    """What a worker run produced: its parsed result, or why there is none."""

    result: dict[str, Any] | None
    returncode: int | None
    timed_out: bool = False
    output_exceeded: bool = False
    pid: int = 0
    memory_exceeded: bool = False


def child_environment() -> dict[str, str]:
    """The complete environment of the child: a fixed allowlist, nothing taken over from ``os.environ``.

    Only Windows needs one inherited value, ``SYSTEMROOT``, to start the interpreter.
    """
    env = {"PYTHONIOENCODING": "utf-8", "PYTHONHASHSEED": "0"}
    if sys.platform == "win32":
        env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", r"C:\Windows")
    else:
        env["LC_ALL"] = "C.UTF-8"
    return env


def _pypdf_directory() -> str:
    spec = importlib.util.find_spec("pypdf")
    if spec is None or spec.origin is None:
        raise ImportError('Indexing PDFs needs pypdf: pip install "orkexa[pdf]"')
    return str(Path(spec.origin).resolve().parent.parent)


def _read_bounded(stream: IO[bytes], limit: int, sink: list[bytes], on_overflow: Any) -> None:
    size = 0
    while True:
        chunk = stream.read(65536)
        if not chunk:
            return
        size += len(chunk)
        if size > limit:
            on_overflow()
            while stream.read(65536):  # drain until the child is gone, keep nothing
                pass
            return
        sink.append(chunk)


def _write_input(stream: IO[bytes], payload: bytes) -> None:
    try:
        stream.write(payload)
    except OSError:
        pass  # the child stopped reading (it failed or was stopped)
    finally:
        with contextlib.suppress(OSError):
            stream.close()


# ---- the memory budget outside Linux -----------------------------------------------------------------------

MEMORY_POLL_SECONDS = 0.02


def _resident_bytes_darwin(pid: int) -> int | None:
    """The resident memory of a macOS process (``proc_pidinfo`` with ``PROC_PIDTASKINFO``), or ``None``."""
    import ctypes
    import ctypes.util

    try:
        libproc = ctypes.CDLL(ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib", use_errno=True)
    except OSError:
        return None
    buffer = (ctypes.c_uint64 * 12)()  # struct proc_taskinfo: virtual size, resident size, ...
    size = libproc.proc_pidinfo(
        ctypes.c_int(pid), ctypes.c_int(4), ctypes.c_uint64(0), buffer, ctypes.c_int(ctypes.sizeof(buffer))
    )
    if size < 16:
        return None
    return int(buffer[1])


def _watch_memory(pid: int, limit: int, done: threading.Event, exceeded: Any) -> None:
    """Kill the child's process group once its resident memory passes ``limit`` bytes (macOS)."""
    while not done.wait(MEMORY_POLL_SECONDS):
        resident = _resident_bytes_darwin(pid)
        if resident is not None and resident > limit:
            exceeded()
            return


class _WindowsJob:
    """A job object that limits the memory of each process in it (Windows)."""

    def __init__(self, limit: int) -> None:
        import ctypes
        from ctypes import wintypes

        class Basic(ctypes.Structure):
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

        class Extended(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", Basic),
                ("IoInfo", ctypes.c_uint64 * 6),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        self._ctypes = ctypes
        self._kernel = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        self._kernel.CreateJobObjectW.restype = wintypes.HANDLE
        self._kernel.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        for name in ("SetInformationJobObject", "QueryInformationJobObject"):
            getattr(self._kernel, name).restype = wintypes.BOOL
        self._kernel.SetInformationJobObject.argtypes = [
            wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD
        ]  # fmt: skip
        self._kernel.QueryInformationJobObject.argtypes = [
            wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD, wintypes.LPVOID
        ]  # fmt: skip
        self._kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self._kernel.AssignProcessToJobObject.restype = wintypes.BOOL
        self._kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self._extended = Extended
        self.limit = limit
        self.handle = self._kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")  # type: ignore[attr-defined]
        info = Extended()
        info.BasicLimitInformation.LimitFlags = 0x100 | 0x2000  # PROCESS_MEMORY, KILL_ON_JOB_CLOSE
        info.ProcessMemoryLimit = limit
        if not self._kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
            self.close()
            raise OSError(ctypes.get_last_error(), "SetInformationJobObject failed")  # type: ignore[attr-defined]

    def assign(self, process_handle: int) -> None:
        if not self._kernel.AssignProcessToJobObject(self.handle, process_handle):
            raise OSError(self._ctypes.get_last_error(), "AssignProcessToJobObject failed")  # type: ignore[attr-defined]

    def peak(self) -> int:
        info = self._extended()
        ok = self._kernel.QueryInformationJobObject(
            self.handle, 9, self._ctypes.byref(info), self._ctypes.sizeof(info), None
        )
        return int(info.PeakProcessMemoryUsed) if ok else 0

    def close(self) -> None:
        if self.handle:
            self._kernel.CloseHandle(self.handle)
            self.handle = None


def run_worker(
    data: bytes,
    *,
    timeout: float,
    max_memory_mb: int,
    max_pages: int,
    max_chars: int,
    worker_module: str = WORKER_MODULE,
    extra_header: dict[str, Any] | None = None,
    module_directory: str | None = None,
) -> WorkerRun:
    """Run ``worker_module`` on ``data`` and return its JSON result (``None`` if it gave none).

    ``worker_module`` and ``extra_header`` exist for the isolation tests only (:mod:`orkexa._pdf_probe`, a
    diagnostic worker started the same way); the public :func:`~orkexa.retrieval.pdf_pages_isolated` always
    uses :mod:`orkexa._pdf_worker` and accepts neither. :func:`orkexa.web.fetch.html_to_text_isolated` starts
    :mod:`orkexa.web._html_worker` the same way, with the directory of trafilatura as ``module_directory``
    (default: the directory of pypdf) and the URL in ``extra_header``.
    """
    header = {
        "max_pages": max_pages,
        "max_chars": max_chars,
        "max_memory_mb": max_memory_mb,
        "cpu_seconds": math.ceil(timeout) + 1,
        **(extra_header or {}),
    }
    payload = json.dumps(header).encode("ascii") + b"\n" + data
    limit = max_chars * 4 + OUTPUT_OVERHEAD
    workdir = tempfile.mkdtemp(prefix="orkexa-pdf-")  # mode 0700, empty
    posix = os.name == "posix"
    memory_limit = max_memory_mb << 20
    job = _WindowsJob(memory_limit) if sys.platform == "win32" else None
    process = subprocess.Popen(
        [sys.executable, "-I", "-m", worker_module, module_directory or _pypdf_directory()],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=child_environment(),
        cwd=workdir,
        close_fds=True,
        start_new_session=posix,
    )
    if job is not None:
        try:
            job.assign(int(process._handle))  # type: ignore[attr-defined]
        except OSError:
            process.kill()
            process.wait()
            job.close()
            raise
    stopped = threading.Lock()
    state = {"output_exceeded": False, "memory_exceeded": False}

    def stop(*, hard: bool = False) -> None:
        with stopped:
            if process.poll() is not None:
                return
            with contextlib.suppress(OSError):
                if posix:
                    os.killpg(process.pid, signal.SIGKILL if hard else signal.SIGTERM)
                elif hard:
                    process.kill()
                else:
                    process.terminate()

    def overflow() -> None:
        state["output_exceeded"] = True
        stop(hard=True)

    def too_much_memory() -> None:
        state["memory_exceeded"] = True
        stop(hard=True)

    stdout: list[bytes] = []
    stderr: list[bytes] = []
    assert process.stdin is not None and process.stdout is not None and process.stderr is not None
    threads = [
        threading.Thread(target=_write_input, args=(process.stdin, payload), daemon=True),
        threading.Thread(target=_read_bounded, args=(process.stdout, limit, stdout, overflow), daemon=True),
        threading.Thread(
            target=_read_bounded, args=(process.stderr, STDERR_LIMIT, stderr, lambda: None), daemon=True
        ),
    ]
    finished = threading.Event()
    if sys.platform == "darwin":
        threads.append(
            threading.Thread(
                target=_watch_memory,
                args=(process.pid, memory_limit, finished, too_much_memory),
                daemon=True,
            )
        )
    timed_out = False
    try:
        for thread in threads:
            thread.start()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            stop()
            try:
                process.wait(timeout=KILL_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                stop(hard=True)
                process.wait()
        finished.set()
        if posix:  # anything the worker started in its group is gone too
            # A group id stays reserved while a member lives: this cannot reach an unrelated group.
            with stopped, contextlib.suppress(OSError):
                os.killpg(process.pid, signal.SIGKILL)
        for thread in threads:
            thread.join(timeout=5)
    except BaseException:
        finished.set()
        stop(hard=True)
        process.wait()
        raise
    finally:
        finished.set()
        if job is not None:
            # A process that failed without a result after using most of its memory ran out of it.
            if process.returncode not in (0, 3) and job.peak() >= memory_limit // 2:
                state["memory_exceeded"] = True
            job.close()
        for stream in (process.stdin, process.stdout, process.stderr):
            with contextlib.suppress(OSError):
                stream.close()
        shutil.rmtree(workdir, ignore_errors=True)
    result: dict[str, Any] | None = None
    if (
        not timed_out
        and not state["output_exceeded"]
        and not state["memory_exceeded"]
        and process.returncode in (0, 3)
    ):
        try:
            parsed = json.loads(b"".join(stdout))
        except ValueError:
            parsed = None
        result = parsed if isinstance(parsed, dict) else None
    return WorkerRun(
        result=result,
        returncode=process.returncode,
        timed_out=timed_out,
        output_exceeded=bool(state["output_exceeded"]),
        pid=process.pid,
        memory_exceeded=bool(state["memory_exceeded"]),
    )
