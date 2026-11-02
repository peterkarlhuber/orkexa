"""The in-container self-test, image identity and the command line (ORK-021), against a fake docker CLI."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from orkexa import ToolExecutionError
from orkexa.dynamic import DockerSandbox, SandboxError
from orkexa.dynamic import sandbox as sandbox_module
from orkexa.dynamic.docker import (
    ContainerLimits,
    evaluate_self_test,
    inspect_image,
    pin_base,
    resolve_image_id,
)
from orkexa.dynamic.docker._cli import main

from .docker_fakes import IMAGE_ID, PYTHON_ENV, FakeDocker, fake_docker, needs_posix

__all__ = ["fake_docker"]

pytestmark = needs_posix

MIB = 1024 * 1024
CODE = "def one() -> int:\n    '''One.\n\n    Returns:\n        One.\n    '''\n    return 1\n"
DIGEST = "sha256:" + "12" * 32


def sandbox(fake: FakeDocker, **options: Any) -> DockerSandbox:
    return DockerSandbox(docker=str(fake.path), **options)


def test_self_test_passes_on_a_correct_engine(fake_docker: FakeDocker) -> None:
    probe = sandbox(fake_docker).self_test()
    assert probe.available, probe
    facts = dict(probe.facts)
    assert (
        facts["image_id"] == IMAGE_ID
        and facts["self_test"] == "passed"
        and len(facts["host_fingerprint"]) == 64
    )
    run = fake_docker.self_tests()[0]
    assert IMAGE_ID in run and run[-4:-1] == ["-I", "-B", "-c"]
    assert DockerSandbox.available(str(fake_docker.path))


def test_self_test_is_remembered_per_host_image_and_shape(fake_docker: FakeDocker) -> None:
    assert sandbox(fake_docker).self_test().available
    assert sandbox(fake_docker).self_test().available
    assert len(fake_docker.self_tests()) == 1
    assert sandbox(fake_docker, memory_mb=256).self_test().available  # another shape
    assert len(fake_docker.self_tests()) == 2
    fake_docker.host("rootful-cgroup2", ServerVersion="27.4.0")  # another daemon
    assert sandbox(fake_docker).self_test().available
    assert len(fake_docker.self_tests()) == 3
    fake_docker.image("orkexa-sandbox", image_id="sha256:" + "ef" * 32)  # the tag moved
    assert dict(sandbox(fake_docker).self_test().facts)["image_id"] == "sha256:" + "ef" * 32
    assert len(fake_docker.self_tests()) == 4
    DockerSandbox.clear_self_test_cache()
    assert sandbox(fake_docker).self_test().available
    assert len(fake_docker.self_tests()) == 5


@pytest.mark.parametrize(
    ("changes", "problem"),
    [
        ({"version": 2}, "unknown version"),
        ({"uid": 0}, "uid 0"),
        ({"status": {"Uid": "65534\t0\t65534\t65534"}}, "ids are not all 65534"),
        ({"status": {"CapEff": "00000000a80425fb"}}, "CapEff is 00000000a80425fb"),
        ({"status": {"CapBnd": "zz"}}, "CapBnd"),
        ({"status": {"CapEff": None}}, "CapEff is not reported"),
        ({"status": {"NoNewPrivs": "0"}}, "NoNewPrivs is 0"),
        ({"status": {"Seccomp": "0"}}, "Seccomp is 0"),
        ({"interfaces": ["eth0", "lo"]}, "only lo"),
        ({"write_root": "allowed"}, "root file system is writable"),
        ({"mounts": {"/": ["rw"]}}, "root file system is writable"),
        ({"mounts": {"/work": ["rw"]}}, "/work is mounted without nodev, noexec, nosuid"),
        ({"exec_tmp": "allowed"}, "a program in /tmp"),
        ({"sizes": {"/dev/shm": 64 * MIB}}, "/dev/shm has"),
        ({"sizes": {"/work": None}}, "/work has None"),
        ({"cgroup": {"version": "1"}}, "not on cgroup v2"),
        ({"cgroup": {"memory.max": "max"}}, "memory.max is max"),
        ({"cgroup": {"memory.swap.max": "max"}}, "memory.swap.max is max"),
        ({"cgroup": {"pids.max": "max"}}, "pids.max is max"),
        ({"cgroup": {"cpu.max": "max 100000"}}, "cpu.max is max 100000"),
        ({"cgroup": {"cpu.max": "200000 100000"}}, "expected 1 CPUs"),
        ({"rlimits": {"RLIMIT_NOFILE": [1048576, 1048576]}}, "RLIMIT_NOFILE"),
        ({"rlimits": {"RLIMIT_FSIZE": [-1, -1]}}, "RLIMIT_FSIZE"),
        ({"rlimits": {"RLIMIT_CORE": [-1, -1]}}, "RLIMIT_CORE"),
        ({"env": ["HOME", "HOSTNAME", "PATH", "AWS_SECRET_ACCESS_KEY"]}, "unexpected environment variables"),
        ({"env": None}, "environment is not reported"),
    ],
)
def test_self_test_refuses_what_the_engine_did_not_apply(
    fake_docker: FakeDocker, changes: dict[str, Any], problem: str
) -> None:
    fake_docker.self_test(**changes)
    box = sandbox(fake_docker)
    probe = box.self_test()
    assert not probe.available and probe.reason == "self-test"
    assert problem in probe.detail
    with pytest.raises(ToolExecutionError, match="not available \\(self-test\\)"):
        box.call(CODE, "one", {})
    assert fake_docker.runs() == []  # no tool code ran
    assert not DockerSandbox.available(str(fake_docker.path))


def test_a_failed_self_test_is_not_remembered(fake_docker: FakeDocker) -> None:
    fake_docker.self_test(status={"Seccomp": "0"})
    assert not sandbox(fake_docker).self_test().available
    fake_docker.self_test()
    assert sandbox(fake_docker).self_test().available


def test_rootful_cgroup_v1_is_refused_by_the_self_test(fake_docker: FakeDocker) -> None:
    fake_docker.host("rootful-cgroup1")
    probe = sandbox(fake_docker).self_test()
    assert not probe.available and probe.reason == "self-test" and "cgroup v2" in probe.detail


@pytest.mark.parametrize(
    "name", ["docker-desktop", "colima", "orbstack", "podman-compat", "rootless-cgroup2"]
)
def test_other_engines_pass_when_their_containers_do(fake_docker: FakeDocker, name: str) -> None:
    fake_docker.host(name)
    assert sandbox(fake_docker).self_test().available


def test_secret_like_image_variables_are_refused(fake_docker: FakeDocker) -> None:
    fake_docker.image("orkexa-sandbox", env=[*PYTHON_ENV, "OPENAI_API_KEY=sk-x", "DB_PASSWORD=x"])
    probe = sandbox(fake_docker).self_test()
    assert not probe.available
    assert "look like secrets: DB_PASSWORD, OPENAI_API_KEY" in probe.detail
    assert "sk-x" not in probe.detail  # values are never reported
    fake_docker.image("orkexa-sandbox", env=[*PYTHON_ENV, "PIP_NO_CACHE_DIR=1"])
    assert sandbox(fake_docker).self_test().available  # harmless image variables pass


def test_self_test_without_a_report(fake_docker: FakeDocker, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_docker.write("selftest_exit.json", 3)
    probe = sandbox(fake_docker).self_test()
    assert not probe.available and "exited with code 3 without a report" in probe.detail
    fake_docker.write("selftest_exit.json", 0)
    (fake_docker.directory / "selftest_raw.txt").write_text("not json")
    assert "without a report" in sandbox(fake_docker).self_test().detail
    (fake_docker.directory / "selftest_raw.txt").write_text("[1, 2]")
    assert "without a report" in sandbox(fake_docker).self_test().detail
    (fake_docker.directory / "selftest_raw.txt").unlink()
    fake_docker.write("selftest_sleep.json", 10)
    monkeypatch.setattr(sandbox_module, "_SELF_TEST_TIMEOUT", 0.5)
    probe = sandbox(fake_docker, timeout=0.5).self_test()
    assert not probe.available and "did not run: Timed out" in probe.detail


def test_missing_and_unreadable_images(fake_docker: FakeDocker, tmp_path: Path) -> None:
    probe = sandbox(fake_docker, image="not-built").self_test()
    assert not probe.available and probe.reason == "image" and "not built" in probe.detail
    probe = sandbox(fake_docker).self_test(image="also-not-built")
    assert probe.reason == "image" and "also-not-built" in probe.detail
    broken = tmp_path / "broken-docker"
    broken.write_text(
        f'#!/bin/sh\nif [ "$1" = image ]; then echo nope; exit 0; fi\nexec {fake_docker.path} "$@"\n'
    )
    broken.chmod(0o755)
    probe = DockerSandbox(docker=str(broken)).self_test()
    assert probe.reason == "image" and "no JSON" in probe.detail
    assert not DockerSandbox.probe(docker=str(tmp_path / "none")).available


def test_readiness_is_reused_for_a_while(fake_docker: FakeDocker, monkeypatch: pytest.MonkeyPatch) -> None:
    box = sandbox(fake_docker)
    box.call(CODE, "one", {})
    box.call(CODE, "one", {})
    infos = [call for call in fake_docker.calls() if call[:1] == ["info"]]
    assert len(infos) == 1
    monkeypatch.setattr(sandbox_module, "_READY_SECONDS", 0.0)
    box.call(CODE, "one", {})
    assert len([call for call in fake_docker.calls() if call[:1] == ["info"]]) == 2
    fake_docker.host("seccomp-unconfined")  # the daemon changed: the next call is refused
    with pytest.raises(ToolExecutionError, match="seccomp"):
        box.call(CODE, "one", {})


def test_evaluate_self_test_directly() -> None:
    limits = ContainerLimits(memory_mb=128, cpus=0.25, pids=8)
    assert evaluate_self_test({"version": 1}, limits)  # an empty report fails every check
    assert evaluate_self_test({}, limits) == ["the self-test report has an unknown version"]


# ---- image identity ------------------------------------------------------------------------------------


def test_resolve_image_id(fake_docker: FakeDocker) -> None:
    docker = str(fake_docker.path)
    assert resolve_image_id(docker, "orkexa-sandbox") == IMAGE_ID
    assert resolve_image_id(docker, IMAGE_ID) == IMAGE_ID
    with pytest.raises(SandboxError, match="not available"):
        resolve_image_id(docker, "missing")
    with pytest.raises(ValueError):
        inspect_image(docker, "--all")
    info = inspect_image(docker, "orkexa-sandbox")
    assert info is not None and "GPG_KEY" in info.env_names and "PATH" in info.env_names


def test_inspect_image_edge_cases(tmp_path: Path) -> None:
    fake = tmp_path / "docker"
    for output in ("[]", "[1]", '[{"Id": ""}]', "{}"):
        fake.write_text(f"#!/bin/sh\necho '{output}'\n")
        fake.chmod(0o755)
        assert inspect_image(str(fake), "x") is None


@pytest.mark.parametrize(
    ("ref", "digests"),
    [
        (f"python:3.12-slim@{DIGEST}", [f"python@{DIGEST}"]),
        (f"python@{DIGEST}", [f"docker.io/library/python@{DIGEST}"]),  # Podman's spelling
        (f"docker.io/library/python:3.12@{DIGEST}", [f"python@{DIGEST}"]),
        (f"ghcr.io/acme/base:1@{DIGEST}", [f"ghcr.io/acme/base@{DIGEST}"]),
        (f"localhost:5000/base@{DIGEST}", [f"localhost:5000/base@{DIGEST}"]),
    ],
)
def test_pin_base(fake_docker: FakeDocker, ref: str, digests: list[str]) -> None:
    fake_docker.write("pullable.json", {ref: {"Id": "sha256:" + "99" * 32, "RepoDigests": digests}})
    assert pin_base(str(fake_docker.path), ref) == "sha256:" + "99" * 32
    assert ["pull", ref] in fake_docker.calls()


def test_pin_base_refuses(fake_docker: FakeDocker) -> None:
    docker = str(fake_docker.path)
    for unpinned in ("python:3.12-slim", "python@sha256:short", f"python@{DIGEST} x"):
        with pytest.raises(ValueError, match="not pinned"):
            pin_base(docker, unpinned)
    with pytest.raises(SandboxError, match="Cannot pull"):
        pin_base(docker, f"python@{DIGEST}")
    other = "sha256:" + "34" * 32
    fake_docker.write(
        "pullable.json",
        {
            f"python@{DIGEST}": {"Id": "sha256:" + "99" * 32, "RepoDigests": [f"python@{other}"]},
            f"evil@{DIGEST}": {"Id": "sha256:" + "98" * 32, "RepoDigests": [f"python@{DIGEST}"]},
        },
    )
    with pytest.raises(SandboxError, match="does not carry the digest"):
        pin_base(docker, f"python@{DIGEST}")
    with pytest.raises(SandboxError, match="does not carry the digest"):
        pin_base(docker, f"evil@{DIGEST}")


def test_pin_base_image_gone_after_pull(tmp_path: Path) -> None:
    fake = tmp_path / "docker"
    fake.write_text('#!/bin/sh\nif [ "$1" = pull ]; then exit 0; fi\nexit 1\n')
    fake.chmod(0o755)
    with pytest.raises(SandboxError, match="not available after the pull"):
        pin_base(str(fake), f"python@{DIGEST}")


# ---- command line ---------------------------------------------------------------------------------------


def test_cli_self_test(fake_docker: FakeDocker, capsys: pytest.CaptureFixture[str]) -> None:
    docker = str(fake_docker.path)
    assert main(["--self-test", "--docker", docker, "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["available"] and result["facts"]["self_test"] == "passed"
    assert main(["--docker", docker, "--memory-mb", "256", "--cpus", "0.5", "--pids", "16"]) == 0
    assert "docker: available" in capsys.readouterr().out
    fake_docker.self_test(status={"Seccomp": "0"})
    assert main(["--docker", docker, "--memory-mb", "128"]) == 1
    out = capsys.readouterr().out
    assert "refused" in out and "reason: self-test" in out


def test_cli_preflight_and_reap(
    fake_docker: FakeDocker, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    docker = str(fake_docker.path)
    assert main(["--preflight", "--docker", docker]) == 0
    capsys.readouterr()
    fake_docker.host("windows")
    assert main(["--preflight", "--docker", docker, "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["reason"] == "os-type"
    assert main(["--preflight", "--docker", str(tmp_path / "none")]) == 1
    assert "missing" in capsys.readouterr().out
    assert main(["--reap", "--docker", docker, "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {"removed": 0}
    assert main(["--reap", "--docker", docker, "--remove-all"]) == 0
    assert "removed 0 container(s)" in capsys.readouterr().out
    assert main(["--reap", "--docker", docker, "--owner", "Bad"]) == 1


def test_module_entry_point_runs_without_warnings(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-W",
            "error",
            "-m",
            "orkexa.dynamic.docker",
            "--preflight",
            "--docker",
            str(tmp_path / "x"),
        ],
        capture_output=True,
        text=True,
        check=False,
        env={"PYTHONPATH": str(Path(sandbox_module.__file__).resolve().parents[2])},
    )
    assert completed.returncode == 1, completed.stderr
    assert "reason: missing" in completed.stdout and completed.stderr == ""
