"""DockerSandbox against a real Docker (or Podman) daemon and the ``orkexa-sandbox`` image (ORK-006, ORK-021).

Marked ``docker``. Skipped without a daemon that passes the preflight and the self-test, unless
``ORKEXA_REQUIRE_DOCKER=1`` (set by the ``docker`` CI jobs, which build ``docker/sandbox.Dockerfile``) makes
that a failure. Besides calls through the sandbox, these tests start containers of exactly the sandbox's
shape (:func:`orkexa.dynamic.docker.run_args`) to show that each limit is enforced by the kernel.
"""

from __future__ import annotations

from collections.abc import Iterator
import contextlib
import json
import os
import subprocess
import threading
import time
import uuid

import pytest

from orkexa import ToolExecutionError
from orkexa.dynamic import DockerSandbox, SandboxCancelled, SandboxError, cancel_scope
from orkexa.dynamic.docker import (
    OWNER_LABEL,
    ContainerLimits,
    list_sandbox_containers,
    reap,
    resolve_image_id,
    run_args,
)

pytestmark = pytest.mark.docker

REQUIRED = os.environ.get("ORKEXA_REQUIRE_DOCKER") == "1"
DOCKER = os.environ.get("ORKEXA_DOCKER", "docker")
IMAGE = os.environ.get("ORKEXA_SANDBOX_IMAGE", "orkexa-sandbox")


def tool(body: str) -> str:
    lines = "\n".join("    " + line if line else "" for line in body.strip("\n").splitlines())
    return (
        "def probe(context: str) -> str:\n"
        '    """Probe.\n\n    Args:\n        context: Input.\n\n    Returns:\n        Text.\n    """\n'
        f"{lines}\n"
    )


def _require() -> None:
    probe = DockerSandbox.probe(DOCKER, IMAGE)
    if not probe.available:
        if REQUIRED:
            pytest.fail(f"ORKEXA_REQUIRE_DOCKER=1 but the Docker sandbox is not ready: {probe.as_dict()}")
        pytest.skip(f"Docker sandbox not ready: {probe.reason}: {probe.detail}")


@pytest.fixture(scope="module")
def sandbox() -> DockerSandbox:
    _require()
    return DockerSandbox(docker=DOCKER, image=IMAGE, timeout=30.0)


@pytest.fixture
def owner() -> Iterator[str]:
    """An owner label of this test only, so its containers can be counted."""
    name = f"orkexa-test-{uuid.uuid4().hex[:10]}"
    yield name
    with contextlib.suppress(SandboxError):
        reap(DOCKER, name, remove_all=True)


def _labelled(owner: str) -> list[str]:
    out = subprocess.run(
        [DOCKER, "ps", "-a", "--filter", f"label={OWNER_LABEL}={owner}", "--format", "{{.ID}}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return out.split()


def _in_shape(
    script: str, limits: ContainerLimits, owner: str, *, detach: bool = False, deadline: float | None = None
) -> subprocess.CompletedProcess[str]:
    """Run ``script`` in a container of exactly the sandbox's shape (no runner, so no rlimits of its own)."""
    args = run_args(
        DOCKER,
        f"orkexa-test-{uuid.uuid4().hex[:12]}",
        resolve_image_id(DOCKER, IMAGE),
        limits,
        owner=owner,
        deadline=deadline,
    )
    if detach:
        args.insert(2, "-d")
    return subprocess.run(
        [*args, "python", "-I", "-B", "-c", script],
        input="",
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _report(completed: subprocess.CompletedProcess[str]) -> dict[str, object]:
    assert completed.returncode == 0, (completed.returncode, completed.stderr)
    result: dict[str, object] = json.loads(completed.stdout.strip().splitlines()[-1])
    return result


#: errno values inside the (Linux) container; the host's ``errno`` module differs on macOS (EAGAIN is 35).
LINUX_EAGAIN, LINUX_EACCES, LINUX_EFBIG, LINUX_EROFS = 11, 13, 27, 30
LIMITS = ContainerLimits(memory_mb=128, cpus=0.5, pids=16, fsize_mb=1, lifetime_s=120)


def test_self_test_passes(sandbox: DockerSandbox) -> None:
    probe = sandbox.self_test()
    assert probe.available, probe.as_dict()
    facts = dict(probe.facts)
    assert (
        facts["self_test"] == "passed" and facts["image_id"].startswith(("sha256:", "")) and facts["engine"]
    )
    print(json.dumps(probe.as_dict(), indent=2))


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
    "write root": attempt(lambda: open("/escape.txt", "w").write("x")),
    "user": os.getuid(),
    "secret": "seen" if "ORKEXA_TEST_SECRET" in os.environ else "absent",
})
"""
    )
    os.environ["ORKEXA_TEST_SECRET"] = "s3cr3t"
    try:
        results = json.loads(sandbox.call(code, "probe", {"context": ""}))
    finally:
        os.environ.pop("ORKEXA_TEST_SECRET", None)
    assert results["network"].startswith("refused")
    assert results["write root"].startswith("refused")
    assert results["user"] == 65534
    assert results["secret"] == "absent"


def test_fork_bomb_hits_the_pids_limit(sandbox: DockerSandbox, owner: str) -> None:
    script = (
        "import json, os, time\n"
        "children, error = [], None\n"
        "for _ in range(500):\n"
        "    try:\n"
        "        pid = os.fork()\n"
        "    except OSError as exc:\n"
        "        error = exc.errno\n"
        "        break\n"
        "    if pid == 0:\n"
        "        time.sleep(30)\n"
        "        os._exit(0)\n"
        "    children.append(pid)\n"
        "for pid in children:\n"
        "    try:\n"
        "        os.kill(pid, 9)\n"
        "    except OSError:\n"  # Podman: a child that already went can answer EPERM
        "        pass\n"
        "print(json.dumps({'forked': len(children), 'errno': error}))\n"
    )
    report = _report(_in_shape(script, LIMITS, owner))
    assert report["errno"] == LINUX_EAGAIN
    assert isinstance(report["forked"], int) and report["forked"] < LIMITS.pids


def test_twice_the_memory_is_oom_killed(sandbox: DockerSandbox, owner: str) -> None:
    script = "data = b'x' * (2 * 128 * 1024 * 1024)\nprint('survived', len(data))\n"
    completed = _in_shape(script, LIMITS, owner)
    assert completed.returncode == 137, (completed.returncode, completed.stdout, completed.stderr)
    assert "survived" not in completed.stdout


def test_network_connect_fails(sandbox: DockerSandbox, owner: str) -> None:
    script = (
        "import json, socket\n"
        "results = {}\n"
        "for name, target in (('ip', ('1.1.1.1', 53)), ('dns', ('example.com', 443))):\n"
        "    try:\n"
        "        socket.create_connection(target, timeout=3).close()\n"
        "        results[name] = 'connected'\n"
        "    except OSError as exc:\n"
        "        results[name] = type(exc).__name__\n"
        "print(json.dumps(results))\n"
    )
    report = _report(_in_shape(script, LIMITS, owner))
    assert report["ip"] != "connected" and report["dns"] != "connected"


def test_write_to_root_fails(sandbox: DockerSandbox, owner: str) -> None:
    script = (
        "import json\n"
        "try:\n"
        "    open('/escape', 'w').close()\n"
        "    result = 'written'\n"
        "except OSError as exc:\n"
        "    result = exc.errno\n"
        "print(json.dumps({'result': result}))\n"
    )
    assert _report(_in_shape(script, LIMITS, owner))["result"] in (LINUX_EROFS, LINUX_EACCES)


def test_fsize_ulimit_limits_a_big_write(sandbox: DockerSandbox, owner: str) -> None:
    script = (
        "import json\n"
        "written = 0\n"
        "try:\n"
        "    with open('/tmp/big', 'wb', buffering=0) as handle:\n"
        "        for _ in range(4):\n"
        "            written += handle.write(b'x' * (1024 * 1024))\n"
        "    result = 'written'\n"
        "except OSError as exc:\n"
        "    result = exc.errno\n"
        "print(json.dumps({'result': result, 'written': written}))\n"
    )
    report = _report(_in_shape(script, LIMITS, owner))
    assert report["result"] == LINUX_EFBIG
    assert isinstance(report["written"], int) and report["written"] <= 1024 * 1024


def test_cancel_leaves_no_container(owner: str) -> None:
    _require()
    sandbox = DockerSandbox(docker=DOCKER, image=IMAGE, timeout=60.0, owner=owner)
    assert sandbox.self_test().available
    threading.Timer(3.0, sandbox.cancel).start()
    started = time.monotonic()
    with pytest.raises(ToolExecutionError) as caught:
        sandbox.call(tool("import time\ntime.sleep(60)\nreturn 'never'"), "probe", {"context": ""})
    assert isinstance(caught.value.__cause__, SandboxCancelled)
    assert time.monotonic() - started < 30
    assert _labelled(owner) == []


def test_cancel_scope_removes_the_container(owner: str) -> None:
    _require()
    sandbox = DockerSandbox(docker=DOCKER, image=IMAGE, timeout=60.0, owner=owner)
    stop = threading.Event()
    threading.Timer(3.0, stop.set).start()
    with cancel_scope(stop.is_set), pytest.raises(ToolExecutionError) as caught:
        sandbox.call(tool("import time\ntime.sleep(60)\nreturn 'never'"), "probe", {"context": ""})
    assert isinstance(caught.value.__cause__, SandboxCancelled)
    assert _labelled(owner) == []


def test_timeout_removes_the_container(owner: str) -> None:
    _require()
    with pytest.raises(ToolExecutionError, match="Timed out"):
        DockerSandbox(docker=DOCKER, image=IMAGE, timeout=5.0, owner=owner).call(
            tool("while True:\n    pass\nreturn 'x'"), "probe", {"context": ""}
        )
    assert _labelled(owner) == []


def test_reaper_removes_an_expired_container(sandbox: DockerSandbox, owner: str) -> None:
    sleeper = "import time\ntime.sleep(300)\n"
    expired = _in_shape(sleeper, LIMITS, owner, detach=True, deadline=time.time() - 5)
    current = _in_shape(sleeper, LIMITS, owner, detach=True, deadline=time.time() + 600)
    assert expired.returncode == 0 and current.returncode == 0, (expired.stderr, current.stderr)
    assert len(_labelled(owner)) == 2
    assert reap(DOCKER, owner) == 1
    remaining = list_sandbox_containers(DOCKER, owner)
    assert [c.id[:12] for c in remaining] == [current.stdout.strip()[:12]]
    assert reap(DOCKER, owner, remove_all=True) == 1
    assert _labelled(owner) == []
