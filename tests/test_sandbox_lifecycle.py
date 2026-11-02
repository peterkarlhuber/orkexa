"""Cancelling sandbox calls (ORK-001) and the capability model, on every operating system.

A call stops at once when its caller's cancel check fires (``cancel_scope``) or when the sandbox is cancelled
for all callers (``Sandbox.cancel``): the process is gone, the run directory removed, and the error is a
:class:`SandboxCancelled`. Other callers of the same sandbox keep running.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import threading
import time

import pytest

from orkexa import ToolExecutionError
from orkexa.dynamic import (
    DockerSandbox,
    MacOSSandbox,
    Sandbox,
    SandboxCancelled,
    SandboxCapabilities,
    SubprocessSandbox,
    cancel_requested,
    cancel_scope,
    current_cancel,
)

SLEEP = '''def nap(seconds: float) -> str:
    """Sleep.

    Args:
        seconds: How long.

    Returns:
        Done.
    """
    import time

    time.sleep(seconds)
    return "done"
'''


def _call_in_thread(
    sandbox: Sandbox, seconds: float, stop: threading.Event | None = None
) -> dict[str, object]:
    outcome: dict[str, object] = {}

    def run() -> None:
        started = time.monotonic()
        try:
            if stop is not None:
                with cancel_scope(stop.is_set):
                    outcome["result"] = sandbox.call(SLEEP, "nap", {"seconds": seconds})
            else:
                outcome["result"] = sandbox.call(SLEEP, "nap", {"seconds": seconds})
        except Exception as exc:
            outcome["error"] = exc
        outcome["elapsed"] = time.monotonic() - started

    thread = threading.Thread(target=run)
    thread.start()
    outcome["thread"] = thread
    return outcome


def _runs() -> set[str]:
    return {p.name for p in Path(tempfile.gettempdir()).glob("orkexa-sandbox-*")}


def test_cancel_scope_stops_only_the_cancelled_call() -> None:
    sandbox = SubprocessSandbox(timeout=60.0, acknowledge_unsafe=True)
    before = _runs()
    stop = threading.Event()
    cancelled = _call_in_thread(sandbox, 30, stop)
    other = _call_in_thread(sandbox, 1.5)
    time.sleep(1.0)
    stop.set()
    cancelled["thread"].join(15)  # type: ignore[attr-defined]
    other["thread"].join(30)  # type: ignore[attr-defined]
    error = cancelled.get("error")
    assert isinstance(error, ToolExecutionError) and isinstance(error.__cause__, SandboxCancelled), cancelled
    assert float(cancelled["elapsed"]) < 10  # type: ignore[arg-type]
    assert other.get("result") == "done", other
    assert _runs() <= before  # both run directories are removed


def test_sandbox_cancel_stops_every_running_call() -> None:
    sandbox = SubprocessSandbox(timeout=60.0, acknowledge_unsafe=True)
    calls = [_call_in_thread(sandbox, 30) for _ in range(2)]
    time.sleep(1.0)
    sandbox.cancel()
    for call in calls:
        call["thread"].join(15)  # type: ignore[attr-defined]
        assert isinstance(call.get("error"), ToolExecutionError), call
        assert isinstance(call["error"].__cause__, SandboxCancelled)  # type: ignore[union-attr]
    assert sandbox.call(SLEEP, "nap", {"seconds": 0}) == "done"  # the sandbox stays usable


def test_timeout_still_applies() -> None:
    sandbox = SubprocessSandbox(timeout=1.0, acknowledge_unsafe=True)
    with pytest.raises(ToolExecutionError, match="Timed out"):
        sandbox.call(SLEEP, "nap", {"seconds": 30})


def test_cancel_helpers() -> None:
    assert current_cancel() is None and not cancel_requested()
    with cancel_scope(lambda: True):
        assert cancel_requested()
    with cancel_scope(lambda: 1 / 0 == 0):  # a broken check never cancels
        assert not cancel_requested()
    assert current_cancel() is None


def test_capabilities_describe_each_backend() -> None:
    subprocess_caps = SubprocessSandbox(acknowledge_unsafe=True).capabilities()
    assert isinstance(subprocess_caps, SandboxCapabilities)
    assert not subprocess_caps.isolating and subprocess_caps.network == "host"
    assert subprocess_caps.maturity == "stable" and subprocess_caps.cancellable
    docker = DockerSandbox().capabilities()
    assert docker.isolating and docker.isolation == "container" and docker.memory == "kernel"
    macos = MacOSSandbox().capabilities()
    assert macos.isolating and macos.maturity == "experimental" and macos.memory == "watchdog"
    assert macos.as_dict()["isolating"] is True


def test_probe_reports_why_a_backend_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda name: None)
    probe = DockerSandbox.probe()
    assert not probe.available and probe.reason == "missing"
    assert SubprocessSandbox.probe().available
