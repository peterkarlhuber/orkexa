"""The experimental macOS sandbox (Seatbelt).

On macOS every test runs a real confined call. Elsewhere the macOS-only tests are skipped, unless
``ORKEXA_REQUIRE_NATIVE_SANDBOX=1`` (set by the macOS jobs of the ``Platforms`` workflow) makes an
unavailable sandbox a failure. A test passes only if the tool code actually ran and the attempt was refused.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import sys
import threading
import time

import pytest

from orkexa import ToolExecutionError
from orkexa.dynamic import MacOSSandbox, SandboxCancelled, cancel_scope
from orkexa.dynamic._macos_confine import profile

ON_MACOS = sys.platform == "darwin"
REQUIRED = os.environ.get("ORKEXA_REQUIRE_NATIVE_SANDBOX") == "1" and ON_MACOS
macos_only = pytest.mark.skipif(not ON_MACOS, reason="needs macOS")


def tool(body: str, args: str = "context: str") -> str:
    """A tool ``probe(<args>) -> str`` whose body is ``body`` (indented by four spaces)."""
    lines = "\n".join("    " + line if line else "" for line in body.strip("\n").splitlines())
    return (
        f"def probe({args}) -> str:\n"
        '    """Probe.\n\n    Args:\n        context: Input.\n\n    Returns:\n        Text.\n    """\n'
        f"{lines}\n"
    )


ATTEMPT = """
import json
c = json.loads(context)
def attempt(action):
    try:
        return "ALLOWED: " + repr(action())[:80]
    except BaseException as exc:
        return "refused: " + type(exc).__name__
"""


@pytest.fixture(scope="module")
def sandbox() -> MacOSSandbox:
    probe = MacOSSandbox.probe()
    if not probe.available:
        if REQUIRED:
            pytest.fail(f"ORKEXA_REQUIRE_NATIVE_SANDBOX=1 but the macOS sandbox is not available: {probe}")
        pytest.skip(f"macOS sandbox not available: {probe.reason}: {probe.detail}")
    return MacOSSandbox(timeout=20.0, memory_mb=512, tmp_mb=16)


def test_profile_denies_by_default() -> None:
    text = profile(["/usr/lib"], ["/private/tmp/x"])
    assert "(deny default)" in text
    assert "network" not in text and "process-fork" not in text and "process-exec" not in text
    assert '(allow file-read* file-write* (subpath "/private/tmp/x"))' in text


def test_probe_off_macos() -> None:
    if ON_MACOS:
        pytest.skip("this is macOS")
    probe = MacOSSandbox.probe(refresh=True)
    assert not probe.available and probe.reason == "platform"


@macos_only
def test_probe_on_macos() -> None:
    probe = MacOSSandbox.probe(refresh=True)
    print(json.dumps(probe.as_dict()))
    if REQUIRED:
        assert probe.available, probe


@macos_only
def test_a_call_runs_and_writes_only_to_tmp(sandbox: MacOSSandbox) -> None:
    code = tool(
        """
import os
open(os.path.join(os.environ["TMPDIR"], "ok.txt"), "w").write("x")
return "ok"
"""
    )
    assert sandbox.call(code, "probe", {"context": "{}"}) == "ok"


@macos_only
def test_private_files_secrets_network_and_processes_are_refused(
    sandbox: MacOSSandbox, tmp_path: Path
) -> None:
    canary = Path.home() / f".orkexa-canary-{os.getpid()}"
    canary.write_text("secret")
    os.environ["ORKEXA_TEST_SECRET"] = "s3cr3t"
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    context = json.dumps(
        {
            "canary": str(canary),
            "home": str(Path.home()),
            "port": server.getsockname()[1],
            "outside": str(tmp_path),
        }
    )
    code = tool(
        ATTEMPT
        + """
import os, socket, subprocess
results = {
    "home file": attempt(lambda: open(c["canary"]).read()),
    "home listing": attempt(lambda: os.listdir(c["home"])),
    "write outside": attempt(lambda: open(c["outside"] + "/escape.txt", "w").write("x")),
    "localhost": attempt(lambda: socket.create_connection(("127.0.0.1", c["port"]), timeout=2)),
    "internet": attempt(lambda: socket.create_connection(("1.1.1.1", 443), timeout=2)),
    "program": attempt(lambda: subprocess.run(["/bin/echo", "x"], capture_output=True)),
    "fork": attempt(lambda: os.fork()),
    "secret": "seen" if "ORKEXA_TEST_SECRET" in os.environ else "absent",
}
return json.dumps(results)
"""
    )
    try:
        results = json.loads(sandbox.call(code, "probe", {"context": context}))
    finally:
        canary.unlink()
        server.close()
        os.environ.pop("ORKEXA_TEST_SECRET", None)
    print(json.dumps(results, indent=1))
    assert results.pop("secret") == "absent"
    allowed = {name: value for name, value in results.items() if not value.startswith("refused")}
    assert not allowed, allowed
    assert not (tmp_path / "escape.txt").exists()


@macos_only
def test_memory_limit_stops_the_call(sandbox: MacOSSandbox) -> None:
    code = tool("blocks = []\nwhile True:\n    blocks.append(bytearray(64 * 1024 * 1024))\nreturn 'never'")
    with pytest.raises(ToolExecutionError):
        sandbox.call(code, "probe", {"context": "{}"})


@macos_only
def test_timeout_kills_the_call() -> None:
    sandbox = MacOSSandbox(timeout=2.0)
    started = time.monotonic()
    with pytest.raises(ToolExecutionError, match="Timed out"):
        sandbox.call(tool("while True:\n    pass\nreturn 'never'"), "probe", {"context": "{}"})
    assert time.monotonic() - started < 15


@macos_only
def test_cancel_stops_the_call(sandbox: MacOSSandbox) -> None:
    stop = threading.Event()
    threading.Timer(1.0, stop.set).start()
    started = time.monotonic()
    with cancel_scope(stop.is_set), pytest.raises(ToolExecutionError) as caught:
        sandbox.call(tool("import time\ntime.sleep(30)\nreturn 'never'"), "probe", {"context": "{}"})
    assert isinstance(caught.value.__cause__, SandboxCancelled)
    assert time.monotonic() - started < 10


@macos_only
def test_concurrent_calls_cannot_read_each_other(sandbox: MacOSSandbox) -> None:
    writer = tool(
        """
import os, time
path = os.path.join(os.environ["TMPDIR"], "mine.txt")
open(path, "w").write("private")
time.sleep(3)
return path
"""
    )
    paths: list[str] = []
    thread = threading.Thread(target=lambda: paths.append(sandbox.call(writer, "probe", {"context": "{}"})))
    thread.start()
    time.sleep(1.5)
    reader = tool(ATTEMPT + 'import glob\nreturn json.dumps(glob.glob(c["pattern"]))')
    pattern = os.path.join(os.path.realpath(os.environ.get("TMPDIR", "/tmp")), "orkexa-sandbox-*", "tmp", "*")
    seen = json.loads(sandbox.call(reader, "probe", {"context": json.dumps({"pattern": pattern})}))
    thread.join(30)
    assert paths, "the writer did not finish"
    assert not any(p.endswith("mine.txt") for p in seen), seen
