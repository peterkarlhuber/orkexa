"""Diagnostic stand-in for :mod:`orkexa._pdf_worker`, used only by Orkexa's isolation tests.

The tests start it through the same launcher (``orkexa._pdf_launch.run_worker(..., worker_module=...)``) to
see what a PDF worker sees: its environment, open file descriptors, resource limits, working directory and
arguments. It reads the same header, applies the same limits, never parses a PDF and reports only facts about
its own process. With ``"probe": "crash"`` in the header it exits abruptly (``os._exit``); with
``"probe": "echo"`` it answers with one page as long as its input.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
from typing import Any

from orkexa._pdf_worker import apply_limits


def _descriptors() -> list[int]:
    if sys.platform.startswith("linux"):
        return sorted(int(name) for name in os.listdir("/proc/self/fd"))
    found = []
    for descriptor in range(256):
        try:
            os.fstat(descriptor)
        except OSError:
            continue
        found.append(descriptor)
    return found


def _limits() -> dict[str, Any]:
    if sys.platform == "win32":
        return {}
    import resource

    names = ("RLIMIT_AS", "RLIMIT_CPU", "RLIMIT_NOFILE", "RLIMIT_FSIZE", "RLIMIT_CORE")
    return {
        name: list(resource.getrlimit(getattr(resource, name))) for name in names if hasattr(resource, name)
    }


def _write_attempt() -> str:
    try:
        with open("probe.txt", "wb") as handle:  # in the private working directory
            handle.write(b"x" * 4096)
            handle.flush()
    except OSError as exc:
        return f"refused: {type(exc).__name__}"
    finally:
        with contextlib.suppress(OSError):
            os.unlink("probe.txt")
    return "written"


def main() -> int:
    header = json.loads(sys.stdin.buffer.readline())
    if header.get("probe") == "crash":
        os._exit(7)
    apply_limits(header)
    if header.get("probe") == "echo":  # more output than any budget allows
        sys.stdout.write(json.dumps({"ok": True, "pages": ["x" * len(sys.stdin.buffer.read())]}))
        sys.stdout.flush()
        return 0
    descriptors = _descriptors()
    cwd = os.getcwd()
    report = {
        "ok": True,
        "pages": [],
        "environ": dict(os.environ),
        "descriptors": descriptors,
        "limits": _limits(),
        "cwd": cwd,
        "cwd_entries": sorted(os.listdir(cwd)),
        "argv": sys.argv,
        "write": _write_attempt() if sys.platform != "win32" else "skipped",
        "stdin_bytes": len(sys.stdin.buffer.read()),
    }
    sys.stdout.write(json.dumps(report))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
