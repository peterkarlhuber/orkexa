"""Docker host inspection and preflight (ORK-019) against recorded ``docker info``/``version`` outputs."""

from __future__ import annotations

from pathlib import Path

import pytest

from orkexa.dynamic import DockerSandbox, SandboxError
from orkexa.dynamic.docker import DockerHost, DockerProbeError, host_fingerprint, inspect_host, preflight

from .docker_fakes import HOSTS, FakeDocker, fake_docker, needs_posix

__all__ = ["fake_docker"]

pytestmark = needs_posix

#: host fixture -> (engine, preflight reason or "" when it passes)
EXPECTED = {
    "rootful-cgroup2": ("docker", ""),
    "rootful-arm64": ("docker", ""),
    "rootful-cgroup1": ("docker", ""),  # the self-test refuses cgroup v1
    "rootless-cgroup2": ("docker", ""),
    "rootless-cgroup2-no-delegation": ("docker", "limits"),
    "rootless-cgroup1": ("docker", "rootless-cgroup"),
    "docker-desktop": ("docker-desktop", ""),
    "docker-desktop-unconfined": ("docker-desktop", "seccomp"),
    "colima": ("colima", ""),
    "orbstack": ("orbstack", ""),
    "seccomp-unconfined": ("docker", "seccomp"),
    "no-seccomp": ("docker", "seccomp"),
    "windows": ("docker", "os-type"),
    "podman-compat": ("podman", ""),
    "podman-native": ("podman", ""),
    "podman-native-no-delegation": ("podman", "limits"),
    "podman-native-cgroup1": ("podman", "rootless-cgroup"),
}


def test_every_fixture_is_covered() -> None:
    assert set(EXPECTED) == set(HOSTS)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_preflight_per_host(name: str, fake_docker: FakeDocker) -> None:
    fake_docker.host(name)
    host = inspect_host(str(fake_docker.path))
    engine, reason = EXPECTED[name]
    assert host.engine == engine
    checked = preflight(host)
    assert checked.available is (reason == "")
    assert checked.reason == reason
    assert dict(checked.facts)["engine"] == engine
    if not checked.available:
        assert checked.detail


def test_host_fields(fake_docker: FakeDocker) -> None:
    fake_docker.host("rootless-cgroup2")
    host = inspect_host(str(fake_docker.path))
    assert host.rootless and host.cgroup_version == "2" and host.cgroup_driver == "systemd"
    assert host.seccomp_profile == "builtin" and not host.apparmor and not host.selinux
    assert host.os_type == "linux" and host.arch == "x86_64" and host.server_version == "27.3.1"
    assert host.memory_limit and host.pids_limit and host.cpu_cfs_quota and host.cpu_cfs_period
    assert host.swap_limit is True
    fake_docker.host("podman-native")
    podman = inspect_host(str(fake_docker.path))
    assert podman.rootless and podman.seccomp_profile == "/usr/share/containers/seccomp.json"
    assert podman.server_version == "4.9.3" and podman.cgroup_version == "2"
    assert ["version", "--format", "{{json .}}"] not in fake_docker.calls()[-1:]
    fake_docker.host("podman-compat")
    assert inspect_host(str(fake_docker.path)).seccomp_profile == "default"


def test_the_engine_name_is_never_trusted(fake_docker: FakeDocker) -> None:
    """A daemon that calls itself Docker Desktop but drops limits is refused like any other."""
    fake_docker.host("docker-desktop", PidsLimit=False)
    checked = preflight(inspect_host(str(fake_docker.path)))
    assert not checked.available and checked.reason == "limits" and "PidsLimit" in checked.detail


def test_limits_refusal_names_every_missing_flag(fake_docker: FakeDocker) -> None:
    fake_docker.host("rootless-cgroup2-no-delegation")
    checked = preflight(inspect_host(str(fake_docker.path)))
    for flag in ("PidsLimit", "CpuCfsQuota", "CpuCfsPeriod"):
        assert flag in checked.detail
    assert "MemoryLimit" not in checked.detail


def test_fingerprint_follows_the_host(fake_docker: FakeDocker) -> None:
    host = inspect_host(str(fake_docker.path))
    assert host_fingerprint(host) == host_fingerprint(inspect_host(str(fake_docker.path)))
    fake_docker.host("rootful-cgroup2", ServerVersion="27.3.2")
    assert host_fingerprint(host) != host_fingerprint(inspect_host(str(fake_docker.path)))
    assert len(host_fingerprint(host)) == 64


def test_facts_are_strings() -> None:
    host = DockerHost(
        engine="unknown",
        os_type="linux",
        arch="x86_64",
        rootless=False,
        cgroup_version=None,
        cgroup_driver=None,
        memory_limit=True,
        pids_limit=True,
        cpu_cfs_quota=True,
        cpu_cfs_period=True,
        seccomp_profile="builtin",
        apparmor=True,
        selinux=False,
        daemon_id="x",
        server_version="1",
    )
    facts = dict(host.as_facts())
    assert facts["cgroup_version"] == "" and facts["apparmor"] == "true" and facts["selinux"] == "false"
    assert preflight(host).available


def test_inspect_errors(fake_docker: FakeDocker, tmp_path: Path) -> None:
    with pytest.raises(DockerProbeError) as missing:
        inspect_host(str(tmp_path / "no-docker"))
    assert missing.value.reason == "missing"
    fake_docker.fail("info", 1)
    with pytest.raises(DockerProbeError) as down:
        inspect_host(str(fake_docker.path))
    assert down.value.reason == "unreachable"
    fake_docker.write("fail.json", {})
    fake_docker.host("rootful-cgroup2", ServerErrors=["Cannot connect to the Docker daemon"])
    with pytest.raises(DockerProbeError, match="Cannot connect") as errors:
        inspect_host(str(fake_docker.path))
    assert errors.value.reason == "unreachable"
    fake_docker.host("rootful-cgroup2", OSType="")
    with pytest.raises(DockerProbeError) as malformed:
        inspect_host(str(fake_docker.path))
    assert malformed.value.reason == "malformed"
    fake_docker.write("host.json", {"info": ["not", "an", "object"], "version": None})
    with pytest.raises(DockerProbeError) as not_object:
        inspect_host(str(fake_docker.path))
    assert not_object.value.reason == "malformed"
    fake_docker.host("rootful-cgroup2")
    fake_docker.fail("version", 1)
    with pytest.raises(DockerProbeError, match="version"):
        inspect_host(str(fake_docker.path))


def test_inspect_timeout(tmp_path: Path) -> None:
    slow = tmp_path / "docker"
    slow.write_text("#!/bin/sh\nsleep 5\n")
    slow.chmod(0o755)
    with pytest.raises(DockerProbeError) as caught:
        inspect_host(str(slow), timeout=0.2)
    assert caught.value.reason == "unreachable"


def test_inspect_passes_the_cli_environment(fake_docker: FakeDocker, tmp_path: Path) -> None:
    import os

    env = {**os.environ, "DOCKER_HOST": "unix:///tmp/elsewhere.sock"}
    assert inspect_host(str(fake_docker.path), env=env).engine == "docker"


@pytest.mark.parametrize("name", ["windows", "seccomp-unconfined", "rootless-cgroup2-no-delegation"])
def test_sandbox_refuses_a_failing_host(name: str, fake_docker: FakeDocker) -> None:
    fake_docker.host(name)
    sandbox = DockerSandbox(docker=str(fake_docker.path))
    probe = sandbox.probe(docker=str(fake_docker.path))
    assert not probe.available and probe.reason == EXPECTED[name][1]
    assert not DockerSandbox.available(str(fake_docker.path))
    with pytest.raises(SandboxError, match=f"not available \\({EXPECTED[name][1]}\\)"):
        sandbox.validate("def f() -> int:\n    return 1\n", "", "f")
    assert fake_docker.runs() == []  # nothing ran


def test_probe_reports_a_missing_image_and_an_unreachable_daemon(fake_docker: FakeDocker) -> None:
    probe = DockerSandbox.probe(docker=str(fake_docker.path), image="not-built")
    assert not probe.available and probe.reason == "image"
    fake_docker.fail("info", 1)
    probe = DockerSandbox.probe(docker=str(fake_docker.path))
    assert not probe.available and probe.reason == "unreachable"
    assert not DockerSandbox.available(str(fake_docker.path))


def test_probe_passes_a_good_host(fake_docker: FakeDocker) -> None:
    probe = DockerSandbox.probe(docker=str(fake_docker.path))
    assert probe.available, probe
    assert dict(probe.facts)["engine"] == "docker"
    assert DockerSandbox.available(str(fake_docker.path))
