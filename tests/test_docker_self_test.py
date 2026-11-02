"""The in-container self-test, image identity and the command line (ORK-021), against a fake docker CLI."""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import struct
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
from orkexa.dynamic.docker._self_test import (
    SECCOMP_PROBES,
    SELF_TEST_SCRIPT,
    check_apparmor,
    check_mount_table,
    check_seccomp_probes,
    parse_mountinfo,
)

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
        ({"version": 1}, "unknown version"),
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
        ({"mountinfo": None}, "mount table is not reported"),
        ({"mountinfo": "garbage"}, "mount table cannot be read"),
        ({"write_sealed": {"/run": "allowed"}}, "can write to /run"),
        ({"write_sealed": None}, "can write to /var/tmp"),
        ({"seccomp_probes": {"move_pages": "allowed"}}, "seccomp profile allows move_pages"),
        ({"seccomp_probes": {"unsupported": "riscv64/64-bit"}}, "architecture riscv64"),
        ({"seccomp_probes": None}, "seccomp probes are not reported"),
        ({"apparmor": "unconfined"}, "not confined by an enforcing profile (unconfined)"),
        ({"apparmor": None}, "(no label)"),
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


def test_a_permissive_daemon_fails_every_probe(fake_docker: FakeDocker) -> None:
    """A daemon whose seccomp filter lets the probes through (a custom profile) is refused by name."""
    fake_docker.self_test(seccomp_probes=dict.fromkeys(SECCOMP_PROBES, "allowed"))
    probe = sandbox(fake_docker).self_test()
    assert not probe.available
    assert "allows add_key, bpf, perf_event_open, userfaultfd, move_pages" in probe.detail


def test_apparmor_is_reported_without_failing_where_the_host_has_none(fake_docker: FakeDocker) -> None:
    fake_docker.host("docker-desktop")  # no AppArmor on the host: the label is reported only
    fake_docker.self_test(apparmor="unconfined")
    probe = sandbox(fake_docker).self_test()
    assert probe.available and dict(probe.facts)["apparmor_label"] == "unconfined"
    fake_docker.self_test(apparmor=None)
    DockerSandbox.clear_self_test_cache()
    assert dict(sandbox(fake_docker).self_test().facts)["apparmor_label"] == "not available"
    fake_docker.host("rootful-cgroup2")  # AppArmor on the host: the label must be an enforcing profile
    fake_docker.self_test()
    probe = sandbox(fake_docker).self_test()
    assert probe.available and dict(probe.facts)["apparmor_label"] == "docker-default (enforce)"


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


# ---- the mount table ----------------------------------------------------------------------------------

MOUNTINFO = Path(__file__).parent / "fixtures" / "docker" / "mountinfo"
SAMPLES = sorted(path.stem for path in MOUNTINFO.glob("*.txt"))


def sample(name: str = "docker-rootful", *, add: str = "", drop: str = "") -> str:
    lines = (MOUNTINFO / f"{name}.txt").read_text().splitlines()
    if drop:
        lines = [line for line in lines if line.split()[4] != drop]
    return "\n".join([*lines, *([add] if add else [])]) + "\n"


def test_the_recorded_engines_have_the_samples() -> None:
    assert SAMPLES == [
        "docker-desktop",
        "docker-rootful",
        "docker-rootless",
        "podman-rootful",
        "podman-rootless",
    ]


@pytest.mark.parametrize("name", SAMPLES)
def test_recorded_mount_tables_pass(name: str) -> None:
    """What Docker (rootful, rootless, Desktop) and Podman (rootful, rootless) mount for run_args()."""
    assert check_mount_table(sample(name)) == []


_EXT4 = "rw,relatime - ext4 /dev/root rw,discard"


@pytest.mark.parametrize(
    ("add", "problem"),
    [
        # a volume from containers.conf / a daemon default: host-backed and writable
        (f"900 612 8:1 /srv/shared /data {_EXT4}", "unexpected writable mount at /data (ext4 /dev/root"),
        # Podman's mounts.conf (RHEL subscription secrets): host-backed, even read-only
        (
            "901 612 8:1 /usr/share/rhel/secrets /run/secrets ro,relatime - ext4 /dev/root rw",
            "unexpected read-only mount at /run/secrets",
        ),
        ("902 612 0:70 / /scratch rw,nosuid,nodev - tmpfs tmpfs rw,size=64k", "writable mount at /scratch"),
        ("903 612 0:71 / /scratch ro,nosuid,nodev - tmpfs tmpfs ro", "read-only mount at /scratch"),
        # the host's /proc (another pid namespace) bound somewhere, also below /proc
        ("904 612 0:22 / /hostproc ro,nosuid - proc proc rw", "mount at /hostproc"),
        ("905 613 0:22 / /proc/driver ro,nosuid - proc proc rw", "mount at /proc/driver"),
        ("906 613 0:55 /sys /proc/sys/kernel rw,nosuid - proc proc rw", "writable mount at /proc/sys/kernel"),
        ("907 616 0:7 / /sys/kernel/security rw,nosuid - securityfs securityfs rw", "/sys/kernel/security"),
        ("908 616 8:1 /home/user /sys/kernel/debug ro,relatime - ext4 /dev/root rw", "/sys/kernel/debug"),
        ("909 614 0:5 /sda /dev/sda rw,nosuid - devtmpfs udev rw", "mount at /dev/sda"),
        ("910 614 0:5 /zero /dev/null rw,nosuid - devtmpfs udev rw", "mount at /dev/null"),
        (f"911 612 8:1 / /etc/hosts {_EXT4}", "mount at /etc/hosts"),  # a whole file system, not a file
        ("912 612 8:1 /usr/bin/tini /sbin/docker-init rw,relatime - ext4 /dev/root rw", "/sbin/docker-init"),
        (f"913 612 8:1 /home/user/project /work {_EXT4}", "/work is ext4, not a tmpfs"),  # under the tmpfs
        (r"914 612 8:1 /srv /my\040data rw,relatime - ext4 /dev/root rw", "mount at /my data"),
        ("915 612 0:72 / /proc/acpi rw,relatime - tmpfs tmpfs rw", "writable mount at /proc/acpi"),
    ],
)
def test_unexpected_mounts_are_refused(add: str, problem: str) -> None:
    problems = check_mount_table(sample(add=add))
    assert any(problem in found for found in problems), problems


@pytest.mark.parametrize(
    ("replace", "by", "problem"),
    [
        (
            "/run rw,nosuid,nodev,noexec,relatime",
            "/run rw,nosuid,nodev,relatime",
            "/run is mounted without noexec",
        ),  # Podman's own --read-only-tmpfs /run, had run_args() not replaced it
        (
            "/sys ro,nosuid,nodev,noexec,relatime - sysfs sysfs ro",
            "/sys rw,nosuid,nodev,noexec,relatime - sysfs sysfs rw",
            "/sys is writable",
        ),
        ("/sys/fs/cgroup ro,", "/sys/fs/cgroup rw,", "/sys/fs/cgroup is writable"),
        ("/proc/sys ro,", "/proc/sys rw,", "writable mount at /proc/sys"),
        ("/dev rw,nosuid - tmpfs", "/dev rw - tmpfs", "/dev is mounted without nosuid"),
        ("/dev/pts rw,nosuid,noexec,relatime - devpts", "/dev/pts rw,nosuid - tmpfs", "mount at /dev/pts"),
        ("/tmp rw,nosuid,nodev,noexec,relatime - tmpfs", "/tmp rw,nosuid - ext4", "/tmp is ext4"),
        ("/var/tmp rw,nosuid,nodev,noexec,relatime - tmpfs", "/var/tmp rw - overlay", "/var/tmp is overlay"),
    ],
)
def test_wrong_engine_mounts_are_refused(replace: str, by: str, problem: str) -> None:
    text = sample()
    assert replace in text
    problems = check_mount_table(text.replace(replace, by))
    assert any(problem in found for found in problems), problems


@pytest.mark.parametrize("point", ["/proc", "/dev", "/sys", "/tmp", "/work", "/dev/shm", "/run", "/var/tmp"])
def test_missing_mounts_are_refused(point: str) -> None:
    assert f"nothing is mounted at {point}" in check_mount_table(sample("podman-rootful", drop=point))


def test_init_binaries_and_masks_are_allowed() -> None:
    init = "920 612 8:1 /usr/bin/docker-init /sbin/docker-init ro,relatime - ext4 /dev/root rw"
    podman_init = "921 612 8:1 /usr/libexec/podman/catatonit /run/podman-init ro,relatime - ext4 /dev/root rw"
    mask = "922 616 0:56 /null /sys/kernel/notes rw,nosuid - tmpfs tmpfs rw,size=65536k"
    ro_view = "923 616 0:7 / /sys/kernel/security ro,nosuid - securityfs securityfs rw"
    text = sample(add=init) + podman_init + "\n" + mask + "\n" + ro_view + "\n"
    assert check_mount_table(text) == []


def test_parse_mountinfo() -> None:
    entries = parse_mountinfo(sample("podman-rootless"))
    root = entries[0]
    assert root.point == "/" and root.read_only and root.fstype == "overlay" and "-" not in root.options
    escaped = parse_mountinfo(r"1 0 8:1 /a\040b /x\011y rw shared:1 master:2 - ext4 /dev/s\134d rw" + "\n")
    assert escaped[0].root == "/a b" and escaped[0].point == "/x\ty" and escaped[0].source == "/dev/s\\d"
    assert not escaped[0].read_only
    for bad in ("1 0 8:1 / / rw", "1 0 8:1 / / rw - ext4"):
        with pytest.raises(ValueError):
            parse_mountinfo(bad)


# ---- seccomp and AppArmor -------------------------------------------------------------------------------


def test_check_seccomp_probes() -> None:
    blocked = dict.fromkeys(SECCOMP_PROBES, "errno:1")
    assert check_seccomp_probes(blocked) == []
    assert check_seccomp_probes({**blocked, "bpf": "errno:38", "add_key": "errno:22"}) == []
    assert "allows perf_event_open" in check_seccomp_probes({**blocked, "perf_event_open": "allowed"})[0]
    assert "not reported: userfaultfd" in check_seccomp_probes({**blocked, "userfaultfd": None})[0]
    assert "failed: ImportError" in check_seccomp_probes({"error": "ImportError: no ctypes"})[0]
    assert check_seccomp_probes([]) == ["the seccomp probes are not reported"]


@pytest.mark.parametrize(
    ("label", "ok"),
    [
        ("docker-default (enforce)", True),
        ("containers-default-0.57.4 (enforce)", True),
        ("orkexa-strict (kill)", True),
        ("docker-default (complain)", False),
        ("docker-default (unconfined)", False),
        ("unconfined", False),
        ("", False),
        (None, False),
    ],
)
def test_check_apparmor(label: str | None, ok: bool) -> None:
    assert (check_apparmor(label, required=True) == []) is ok
    assert check_apparmor(label, required=False) == []


def _probe_code() -> str:
    start, end = SELF_TEST_SCRIPT.index("def probe_syscalls"), SELF_TEST_SCRIPT.index("def write_in")
    return "import os\n" + SELF_TEST_SCRIPT[start:end]


def test_the_self_test_script_compiles() -> None:
    compile(SELF_TEST_SCRIPT, "<self-test>", "exec")
    for name in SECCOMP_PROBES:
        assert f"'{name}':" in SELF_TEST_SCRIPT


@pytest.mark.skipif(
    sys.platform != "linux" or platform.machine() not in ("x86_64", "aarch64") or struct.calcsize("P") != 8,
    reason="the seccomp probes have system call numbers for 64-bit x86 and arm Linux",
)
def test_the_seccomp_probes_run_here() -> None:
    """The probes run in this (unconfined or confined) process and report each call; outside a container the
    kernel usually lets add_key, userfaultfd or move_pages through, which is what they are meant to see."""
    namespace: dict[str, Any] = {}
    exec(_probe_code(), namespace)  # the self-test's own code
    results = namespace["seccomp_probes"]()
    assert set(results) == set(SECCOMP_PROBES), results
    assert all(value == "allowed" or value.startswith("errno:") for value in results.values())


def test_unsupported_architectures_report_themselves(monkeypatch: pytest.MonkeyPatch) -> None:
    namespace: dict[str, Any] = {}
    exec(_probe_code(), namespace)
    monkeypatch.setattr(os, "uname", lambda: os.uname_result(("Linux", "h", "6", "1", "riscv64")))
    assert namespace["seccomp_probes"]() == {"unsupported": "riscv64/64-bit"}


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
