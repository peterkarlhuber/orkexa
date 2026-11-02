"""DockerSandbox against a real Docker daemon and the ``orkexa-sandbox`` image (ORK-006).

Skipped without Docker or the image, unless ``ORKEXA_REQUIRE_DOCKER=1`` (set by the ``docker`` CI job, which
builds ``docker/sandbox.Dockerfile``) makes that a failure.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time

import pytest

from orkexa import ToolExecutionError
from orkexa.dynamic import DockerSandbox, SandboxCancelled, cancel_scope

REQUIRED = os.environ.get("ORKEXA_REQUIRE_DOCKER") == "1"


def tool(body: str) -> str:
    lines = "\n".join("    " + line if line else "" for line in body.strip("\n").splitlines())
    return (
        "def probe(context: str) -> str:\n"
        '    """Probe.\n\n    Args:\n        context: Input.\n\n    Returns:\n        Text.\n    """\n'
        f"{lines}\n"
    )


@pytest.fixture(scope="module")
def sandbox() -> DockerSandbox:
    probe = DockerSandbox.probe()
    if not probe.available:
        if REQUIRED:
            pytest.fail(f"ORKEXA_REQUIRE_DOCKER=1 but Docker is not ready: {probe}")
        pytest.skip(f"Docker not ready: {probe.reason}: {probe.detail}")
    return DockerSandbox(timeout=30.0)


def _containers() -> set[str]:
    out = subprocess.run(
        ["docker", "ps", "-a", "--filter", "name=orkexa-sandbox-", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    return set(out.split())


def test_call_runs_isolated(sandbox: DockerSandbox) -> None:
    code = tool(
        """
import json, os, socket
def attempt(action):
    try:
        action()
        return "ALLOWED"
    except BaseException as exc:
        return "refused: " + type(exc).__name__
return json.dumps({
    "network": attempt(lambda: socket.create_connection(("1.1.1.1", 443), timeout=3)),
    "write root": attempt(lambda: open("/work/escape.txt", "w").write("x")),
    "user": os.getuid(),
    "env": sorted(k for k in os.environ if "KEY" in k or "TOKEN" in k),
})
"""
    )
    results = json.loads(sandbox.call(code, "probe", {"context": ""}))
    assert results["network"].startswith("refused")
    assert results["write root"].startswith("refused")
    assert results["user"] == 65534
    assert results["env"] == []


def test_cancel_removes_the_container(sandbox: DockerSandbox) -> None:
    before = _containers()
    stop = threading.Event()
    threading.Timer(3.0, stop.set).start()
    started = time.monotonic()
    with cancel_scope(stop.is_set), pytest.raises(ToolExecutionError) as caught:
        sandbox.call(tool("import time\ntime.sleep(60)\nreturn 'never'"), "probe", {"context": ""})
    assert isinstance(caught.value.__cause__, SandboxCancelled)
    assert time.monotonic() - started < 20
    assert _containers() <= before


def test_timeout_removes_the_container() -> None:
    probe = DockerSandbox.probe()
    if not probe.available:
        if REQUIRED:
            pytest.fail(f"Docker not ready: {probe}")
        pytest.skip("Docker not ready")
    before = _containers()
    with pytest.raises(ToolExecutionError, match="Timed out"):
        DockerSandbox(timeout=5.0).call(tool("while True:\n    pass\nreturn 'x'"), "probe", {"context": ""})
    assert _containers() <= before
