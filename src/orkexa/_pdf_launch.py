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
* its output is bounded: the parent reads at most ``max_chars * 4 + 64 KiB`` bytes.

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


def run_worker(
    data: bytes,
    *,
    timeout: float,
    max_memory_mb: int,
    max_pages: int,
    max_chars: int,
    worker_module: str = WORKER_MODULE,
    extra_header: dict[str, Any] | None = None,
) -> WorkerRun:
    """Run ``worker_module`` on ``data`` and return its JSON result (``None`` if it gave none).

    ``worker_module`` and ``extra_header`` exist for the isolation tests only (:mod:`orkexa._pdf_probe`, a
    diagnostic worker started the same way); the public :func:`~orkexa.retrieval.pdf_pages_isolated` always
    uses :mod:`orkexa._pdf_worker` and accepts neither.
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
    process = subprocess.Popen(
        [sys.executable, "-I", "-m", worker_module, _pypdf_directory()],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=child_environment(),
        cwd=workdir,
        close_fds=True,
        start_new_session=posix,
    )
    stopped = threading.Lock()
    state = {"output_exceeded": False}

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
        if posix:  # anything the worker started in its group is gone too
            # A group id stays reserved while a member lives: this cannot reach an unrelated group.
            with stopped, contextlib.suppress(OSError):
                os.killpg(process.pid, signal.SIGKILL)
        for thread in threads:
            thread.join(timeout=5)
    except BaseException:
        stop(hard=True)
        process.wait()
        raise
    finally:
        for stream in (process.stdin, process.stdout, process.stderr):
            with contextlib.suppress(OSError):
                stream.close()
        shutil.rmtree(workdir, ignore_errors=True)
    result: dict[str, Any] | None = None
    if not timed_out and not state["output_exceeded"] and process.returncode in (0, 3):
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
    )
