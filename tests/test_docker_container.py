"""The container shape, stdin transport, cancellation and reaper (ORK-020), without a daemon."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import warnings

import pytest

from orkexa import ToolExecutionError
from orkexa.dynamic import DockerSandbox, SandboxCancelled, SandboxError
from orkexa.dynamic.docker import (
    BOOTSTRAP,
    DEADLINE_LABEL,
    OWNER_LABEL,
    RUNNER_MODULES,
    ContainerLimits,
    copy_runner,
    envelope,
    list_sandbox_containers,
    reap,
    run_args,
    runner_files,
    runner_fingerprint,
    stop_container,
)

from .docker_fakes import IMAGE_ID, FakeDocker, fake_docker, needs_posix

__all__ = ["fake_docker"]

CODE = (
    "def square(x: int) -> int:\n"
    '    """Square a number.\n\n    Args:\n        x: The number.\n\n'
    '    Returns:\n        The square.\n    """\n'
    "    return x * x\n"
)
TESTS = "from tool import square\n\ndef test_square():\n    assert square(3) == 9\n"
SLEEP = (
    "import time\n\ndef wait() -> str:\n"
    '    """Wait.\n\n    Returns:\n        Never.\n    """\n'
    "    time.sleep(60)\n    return 'never'\n"
)
LIMITS = ContainerLimits(memory_mb=256, cpus=0.5, pids=32, lifetime_s=90)


def _pairs(args: list[str]) -> list[tuple[str, str]]:
    return [(args[i], args[i + 1]) for i in range(len(args) - 1) if args[i].startswith("-")]


def test_run_args_shape() -> None:
    args = run_args("docker", "sb-1", IMAGE_ID, LIMITS, owner="nodestro", deadline=1_700_000_000.2)
    assert args[:4] == ["docker", "run", "--rm", "-i"] and args[-1] == IMAGE_ID
    pairs = _pairs(args)
    for pair in [
        ("--pull", "never"),
        ("--name", "sb-1"),
        ("--network", "none"),
        ("--cap-drop", "ALL"),
        ("--security-opt", "no-new-privileges"),
        ("--user", "65534:65534"),
        ("--ipc", "private"),
        ("--shm-size", "16m"),
        ("--ulimit", "nofile=1024:1024"),
        ("--ulimit", f"fsize={16 * 1024 * 1024}:{16 * 1024 * 1024}"),
        ("--ulimit", "core=0:0"),
        ("--log-driver", "none"),
        ("--pids-limit", "32"),
        ("--memory", "256m"),
        ("--memory-swap", "256m"),
        ("--cpus", "0.5"),
        ("--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=16m"),
        ("--tmpfs", "/work:rw,noexec,nosuid,nodev,size=16m,uid=65534,gid=65534,mode=0700"),
        ("-w", "/work"),
        ("--label", f"{OWNER_LABEL}=nodestro"),
        ("--label", f"{DEADLINE_LABEL}=1700000001"),
    ]:
        assert pair in pairs, pair
    assert "--read-only" in args
    for forbidden in (
        "-v",
        "--volume",
        "--mount",
        "-e",
        "--env",
        "--env-file",
        "--privileged",
        "-p",
        "--device",
    ):
        assert forbidden not in args
    assert not [a for a in args if a.startswith(("--security-opt=", "seccomp", "apparmor"))]


def test_run_args_deadline_and_labels() -> None:
    before = time.time()
    args = run_args("docker", "sb-1", "img", LIMITS, labels={"org.example.call": "c-1"})
    deadline = int(dict(a.split("=", 1) for a in args if a.startswith(DEADLINE_LABEL))[DEADLINE_LABEL])
    assert before + 90 <= deadline <= time.time() + 91
    assert ("--label", "org.example.call=c-1") in _pairs(args)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"owner": "Bad Owner"}, "owner"),
        ({"name": "-rm"}, "container name"),
        ({"image": "--privileged"}, "image"),
        ({"labels": {OWNER_LABEL: "other"}}, "label"),
        ({"labels": {"x": "a\nb"}}, "label"),
    ],
)
def test_run_args_rejects_bad_values(kwargs: dict[str, object], message: str) -> None:
    options: dict[str, object] = {"name": "sb-1", "image": "img", **kwargs}
    name, image = options.pop("name"), options.pop("image")
    with pytest.raises(ValueError, match=message):
        run_args("docker", str(name), str(image), LIMITS, **options)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"memory_mb": 32},
        {"pids": 0},
        {"cpus": 0},
        {"cpus": float("nan")},
        {"nofile": 4},
        {"fsize_mb": 0},
        {"lifetime_s": 0},
        {"lifetime_s": float("inf")},
    ],
)
def test_container_limits_are_validated(kwargs: dict[str, float]) -> None:
    values: dict[str, float] = {"memory_mb": 256, "cpus": 1.0, "pids": 16, **kwargs}
    with pytest.raises(ValueError):
        ContainerLimits(**values)  # type: ignore[arg-type]


def test_sandbox_validates_owner_and_limits_at_construction() -> None:
    with pytest.raises(ValueError, match="owner"):
        DockerSandbox(owner="UPPER")
    with pytest.raises(ValueError, match="transport"):
        DockerSandbox(transport="ssh")  # type: ignore[arg-type]
    assert DockerSandbox(cpus=2, pids_limit=10).limits() == ContainerLimits(
        memory_mb=512, cpus=2, pids=10, lifetime_s=80.0
    )


def test_runner_modules_are_the_runner_import_closure(tmp_path: Path) -> None:
    """The runner works from exactly the modules that travel on stdin."""
    copy_runner(tmp_path)
    script = (
        "import json, sys, types\n"
        f"root = {str(tmp_path / 'orkexa')!r}\n"
        "package = types.ModuleType('orkexa'); package.__path__ = [root]; sys.modules['orkexa'] = package\n"
        "from orkexa.tools import Tool\n"
        "print(json.dumps(sorted(m.__file__[len(root) + 1:] for n, m in sys.modules.items()"
        " if n.startswith('orkexa.') and getattr(m, '__file__', None))))\n"
    )
    loaded = json.loads(
        subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, check=True).stdout
    )
    expected = sorted(m for m in RUNNER_MODULES if m != "dynamic/_sandbox_runner.py")
    assert [m.replace(os.sep, "/") for m in loaded] == expected
    assert sorted(runner_files()) == sorted(f"orkexa/{m}" for m in RUNNER_MODULES)


def test_runner_fingerprint_is_stable(tmp_path: Path) -> None:
    assert runner_fingerprint() == runner_fingerprint() and len(runner_fingerprint()) == 64
    written = copy_runner(tmp_path)
    assert len(written) == len(RUNNER_MODULES) and all(path.is_file() for path in written)


@pytest.mark.parametrize(
    "change",
    [{"files": {"../escape.py": "x"}}, {"files": {".hidden": "x"}}, {"runner": {"orkexa/../x.py": "x"}}],
)
def test_bootstrap_refuses_bad_paths(tmp_path: Path, change: dict[str, object]) -> None:
    data = json.loads(envelope({"tool.py": "x = 1"}, "{}"))
    data.update(change)
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", BOOTSTRAP],
        input=json.dumps(data).encode(),
        capture_output=True,
        cwd=tmp_path,
        env={"TMPDIR": str(tmp_path)},
        check=False,
    )
    assert completed.returncode != 0 and b"invalid" in completed.stderr
    assert not (tmp_path.parent / "escape.py").exists()


# ---- runs through the fake docker CLI ----------------------------------------------------------------


@needs_posix
def test_stdin_transport_end_to_end(fake_docker: FakeDocker) -> None:
    sandbox = DockerSandbox(docker=str(fake_docker.path), owner="unit-test")
    result = sandbox.validate(CODE, TESTS, "square")
    assert result.ok, result.feedback()
    assert result.name == "square" and result.input_schema is not None
    assert sandbox.call(CODE, "square", {"x": 7}) == "49"
    runs = fake_docker.runs()
    assert len(runs) == 2
    for run in runs:
        assert "-v" not in run and "-e" not in run
        assert ("--label", f"{OWNER_LABEL}=unit-test") in _pairs(run)
        assert run[-5:-1] == ["python", "-I", "-B", "-c"] and run[-1] == BOOTSTRAP


@needs_posix
def test_bind_transport_still_works(fake_docker: FakeDocker, tmp_path: Path) -> None:
    with pytest.warns(DeprecationWarning):
        sandbox = DockerSandbox(docker=str(fake_docker.path), transport="bind")
    assert sandbox.call(CODE, "square", {"x": 5}) == "25"
    run = fake_docker.runs()[-1]
    assert "-v" in run and run[-1].endswith("_sandbox_runner.py")
    assert not [a for a in run if a.startswith("/work:")]  # the tmpfs /work is replaced by the mount


@needs_posix
def test_tool_errors_and_silent_containers(fake_docker: FakeDocker) -> None:
    sandbox = DockerSandbox(docker=str(fake_docker.path))
    with pytest.raises(ToolExecutionError, match="square"):
        sandbox.call(CODE.replace("x * x", "1 / 0"), "square", {"x": 1})
    fake_docker.image("orkexa-sandbox", image_id="sha256:" + "cd" * 32)
    fake_docker.fail("run", 125)
    with pytest.raises(SandboxError, match=r"(?s)code 125.*isolated sandbox"):
        sandbox.validate(CODE, TESTS, "square")


def _running(fake: FakeDocker, owner: str) -> list[str]:
    listed = subprocess.run(
        [str(fake.path), "ps", "-a", "--filter", f"label={OWNER_LABEL}={owner}", "--format", "{{.ID}}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return listed.stdout.split()


@needs_posix
def test_cancel_kills_the_container(fake_docker: FakeDocker) -> None:
    sandbox = DockerSandbox(docker=str(fake_docker.path), owner="cancel-test", timeout=30)
    caught: list[BaseException] = []

    def call() -> None:
        try:
            sandbox.call(SLEEP, "wait", {})
        except BaseException as exc:
            caught.append(exc)

    worker = threading.Thread(target=call)
    started = time.monotonic()
    worker.start()
    deadline = time.monotonic() + 15
    while not _running(fake_docker, "cancel-test") and time.monotonic() < deadline:
        time.sleep(0.05)
    assert _running(fake_docker, "cancel-test")
    sandbox.cancel()
    worker.join(20)
    assert time.monotonic() - started < 20
    assert caught and isinstance(caught[0].__cause__, SandboxCancelled)
    name = next(a for a in fake_docker.runs()[-1] if a.startswith("orkexa-sandbox-"))
    assert ["kill", name] in fake_docker.calls() and ["rm", "-f", name] in fake_docker.calls()
    assert _running(fake_docker, "cancel-test") == []


@needs_posix
def test_timeout_removes_the_container(fake_docker: FakeDocker) -> None:
    sandbox = DockerSandbox(docker=str(fake_docker.path), owner="timeout-test", timeout=2)
    with pytest.raises(ToolExecutionError, match="Timed out"):
        sandbox.call(SLEEP, "wait", {})
    assert _running(fake_docker, "timeout-test") == []


def _container(
    name: str, owner: str | None, deadline: str | None, *, running: bool = True
) -> dict[str, object]:
    labels = {}
    if owner is not None:
        labels[OWNER_LABEL] = owner
    if deadline is not None:
        labels[DEADLINE_LABEL] = deadline
    return {
        "Id": name * 4,
        "Name": name,
        "Rm": False,
        "Pid": None,
        "Config": {"Labels": labels},
        "State": {"Running": running},
    }


@needs_posix
def test_list_and_reap(fake_docker: FakeDocker) -> None:
    now = time.time()
    fake_docker.write(
        "containers.json",
        [
            _container("expired", "orkexa", str(int(now - 10))),
            _container("current", "orkexa", str(int(now + 600)), running=False),
            _container("broken", "orkexa", "soon"),
            _container("infinite", "orkexa", "inf"),
            _container("other", "nodestro", str(int(now - 10))),
            _container("unlabelled", None, None),
        ],
    )
    docker = str(fake_docker.path)
    listed = {c.name: c for c in list_sandbox_containers(docker, "orkexa")}
    assert set(listed) == {"expired", "current", "broken", "infinite"}
    assert listed["broken"].deadline is None and listed["infinite"].deadline is None
    assert listed["expired"].running and not listed["current"].running
    assert reap(docker, "orkexa", now=now) == 3
    assert [c.name for c in list_sandbox_containers(docker, "orkexa")] == ["current"]
    assert DockerSandbox(docker=docker).reap(remove_all=True) == 1
    assert list_sandbox_containers(docker, "orkexa") == []
    assert [c.name for c in list_sandbox_containers(docker, "nodestro")] == ["other"]


@needs_posix
def test_reaper_errors(fake_docker: FakeDocker, tmp_path: Path) -> None:
    docker = str(fake_docker.path)
    with pytest.raises(ValueError, match="owner"):
        reap(docker, "Not Valid")
    fake_docker.fail("ps", 1)
    with pytest.raises(SandboxError, match="ps failed"):
        list_sandbox_containers(docker)
    with pytest.raises(SandboxError, match="failed"):
        reap(str(tmp_path / "no-docker"))
    stop_container(str(tmp_path / "no-docker"), "anything")  # never raises


@needs_posix
def test_reaper_ignores_foreign_and_malformed_inspect_output(tmp_path: Path) -> None:
    fake = tmp_path / "docker"
    fake.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = ps ]; then echo abc; echo def; exit 0; fi\n'
        'if [ "$2" = inspect ]; then echo \'[1, {"Id": "abc", "Config": {"Labels": {"org.orkexa.sandbox": '
        '"other"}}}]\'; exit 0; fi\n'
        "exit 1\n"
    )
    fake.chmod(0o755)
    assert list_sandbox_containers(str(fake)) == []
    fake.write_text('#!/bin/sh\nif [ "$1" = ps ]; then echo abc; exit 0; fi\necho not-json\n')
    with pytest.raises(SandboxError, match="no JSON"):
        list_sandbox_containers(str(fake))


def test_command_keeps_its_signature(tmp_path: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        command = DockerSandbox().command(tmp_path, "sb-1")
    assert command[0] == "docker" and "orkexa-sandbox" in command
