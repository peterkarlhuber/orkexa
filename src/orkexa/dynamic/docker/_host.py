"""What the Docker daemon behind a ``docker`` CLI enforces, and whether that is enough (preflight).

:func:`inspect_host` reads ``docker info`` and ``docker version`` (also the native ``podman info``)
into a :class:`DockerHost`; :func:`preflight` refuses a host that cannot enforce the sandbox's limits. The
engine's name (Docker Desktop, Colima, Podman, ...) is only reported: every decision rests on what the daemon
says it enforces, and the in-container self-test (:meth:`orkexa.dynamic.DockerSandbox.self_test`) then
proves it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
import subprocess
from typing import Any, Literal

from ..capabilities import SandboxProbe

__all__ = ["DockerHost", "DockerProbeError", "Engine", "host_fingerprint", "inspect_host", "preflight"]

#: The engine behind the CLI, as far as it can be told; informational only, never a reason to trust a host.
Engine = Literal["docker", "docker-desktop", "podman", "colima", "orbstack", "unknown"]


class DockerProbeError(Exception):
    """The daemon could not be inspected. ``reason`` is a :class:`SandboxProbe` reason code."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class DockerHost:
    """What one Docker (or Podman) daemon reports about itself.

    ``None`` means the daemon did not say. The limit flags tell whether the daemon can enforce ``--memory``,
    ``--pids-limit`` and ``--cpus``; a daemon that cannot silently runs the container without the limit."""

    engine: Engine
    os_type: str
    arch: str
    rootless: bool
    cgroup_version: str | None
    cgroup_driver: str | None
    memory_limit: bool
    pids_limit: bool
    #: ``None`` when the daemon's ``info`` does not report CPU quota support at all (Podman's
    #: Docker-compatible API omits ``CpuCfsQuota``/``CpuCfsPeriod``). On cgroup v2 the self-test then
    #: decides from ``cpu.max``.
    cpu_cfs_quota: bool | None
    cpu_cfs_period: bool | None
    #: ``builtin``/``default``/a profile path when seccomp filters system calls, ``unconfined`` when it is
    #: off, ``None`` when the daemon does not report seccomp at all.
    seccomp_profile: str | None
    apparmor: bool
    selinux: bool
    daemon_id: str
    server_version: str
    #: Whether swap is accounted (``--memory-swap`` effective); reported only, the self-test checks swap.
    swap_limit: bool | None = None
    operating_system: str = ""

    def as_facts(self) -> tuple[tuple[str, str], ...]:
        """The host as :attr:`SandboxProbe.facts`."""
        return tuple((key, _fact(value)) for key, value in asdict(self).items())


def _fact(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def host_fingerprint(host: DockerHost) -> str:
    """A stable SHA-256 of everything :class:`DockerHost` holds: a changed daemon or setting changes it."""
    return hashlib.sha256(json.dumps(asdict(host), sort_keys=True).encode("utf-8")).hexdigest()


def _run_json(docker: str, args: list[str], env: Mapping[str, str] | None, timeout: float) -> Any:
    try:
        completed = subprocess.run(
            [docker, *args],
            capture_output=True,
            timeout=timeout,
            check=False,
            env=dict(env) if env is not None else None,
        )
    except FileNotFoundError as exc:
        raise DockerProbeError("missing", f"The {docker} command is not installed: {exc}") from exc
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DockerProbeError("unreachable", f"{docker} {args[0]} failed: {exc}") from exc
    text = completed.stdout.decode("utf-8", "replace").strip()
    data: Any = None
    if text:
        try:
            data = json.loads(text)
        except ValueError:
            data = None
    if completed.returncode != 0:  # e.g. the CLI prints an empty info object when the daemon is down
        stderr = completed.stderr.decode("utf-8", "replace").strip()
        errors = data.get("ServerErrors") if isinstance(data, dict) else None
        message = "; ".join(map(str, errors)) if errors else stderr
        raise DockerProbeError(
            "unreachable", f"{docker} {args[0]} exited with {completed.returncode}: {message[-500:]}"
        )
    if not isinstance(data, dict):
        raise DockerProbeError("malformed", f"{docker} {args[0]} did not print a JSON object")
    return data


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _security_options(options: Any) -> dict[str, dict[str, str]]:
    """``["name=seccomp,profile=builtin"]`` -> ``{"seccomp": {"profile": "builtin"}}``."""
    parsed: dict[str, dict[str, str]] = {}
    for option in options if isinstance(options, list) else []:
        fields = dict(part.split("=", 1) for part in str(option).split(",") if "=" in part)
        name = fields.pop("name", "")
        if name:
            parsed[name] = fields
    return parsed


def _engine_of(info: Mapping[str, Any], version: Mapping[str, Any]) -> Engine:
    server = _dict(version.get("Server"))
    components = [c for c in server.get("Components") or [] if isinstance(c, dict)]
    names = {str(c.get("Name", "")) for c in components}
    platform = str(_dict(server.get("Platform")).get("Name", ""))
    operating_system = str(info.get("OperatingSystem", ""))
    name = str(info.get("Name", ""))
    if "Podman Engine" in names:
        return "podman"
    if operating_system == "Docker Desktop" or platform.startswith("Docker Desktop"):
        return "docker-desktop"
    if operating_system == "OrbStack":
        return "orbstack"
    if name == "colima" or name.startswith("colima-"):
        return "colima"
    if "Engine" in names:
        return "docker"
    return "unknown"


def _reported(info: Mapping[str, Any], key: str) -> bool | None:
    """``info[key] is True``, or ``None`` when the daemon leaves the key out (it does not say)."""
    return info.get(key) is True if key in info else None


def _from_docker_info(info: Mapping[str, Any], version: Mapping[str, Any]) -> DockerHost:
    options = _security_options(info.get("SecurityOptions"))
    seccomp = options.get("seccomp")
    server = _dict(version.get("Server"))
    cgroup = info.get("CgroupVersion")
    swap = info.get("SwapLimit")
    engine = _engine_of(info, version)
    cpu_quota, cpu_period = _reported(info, "CpuCfsQuota"), _reported(info, "CpuCfsPeriod")
    if engine == "podman":
        # Podman's Docker-compatible info reports ``false`` for both even where it enforces ``--cpus`` (seen
        # with rootful Podman 4.9.3 on cgroup v2), so its answer counts as unreported: the self-test decides.
        cpu_quota = cpu_period = None
    return DockerHost(
        engine=engine,
        os_type=str(info.get("OSType", "")),
        arch=str(info.get("Architecture", "")),
        rootless="rootless" in options,
        cgroup_version=str(cgroup) if cgroup not in (None, "") else None,
        cgroup_driver=str(info["CgroupDriver"]) if info.get("CgroupDriver") else None,
        memory_limit=info.get("MemoryLimit") is True,
        pids_limit=info.get("PidsLimit") is True,
        cpu_cfs_quota=cpu_quota,
        cpu_cfs_period=cpu_period,
        seccomp_profile=(seccomp.get("profile") or "default") if seccomp is not None else None,
        apparmor="apparmor" in options,
        selinux="selinux" in options,
        daemon_id=str(info.get("ID", "")),
        server_version=str(info.get("ServerVersion") or server.get("Version") or ""),
        swap_limit=swap if isinstance(swap, bool) else None,
        operating_system=str(info.get("OperatingSystem", "")),
    )


def _from_podman_info(info: Mapping[str, Any]) -> DockerHost:
    """The native ``podman info`` (``docker=podman``): limits follow the cgroup controllers it may use."""
    host = _dict(info.get("host"))
    security = _dict(host.get("security"))
    controllers = {str(c) for c in host.get("cgroupControllers") or []}
    version = str(host.get("cgroupVersion", "")).lstrip("v") or None
    if version == "1":  # cgroup v1 lists no controllers; Podman reports what the hierarchy offers
        controllers = {"memory", "pids", "cpu"} if not security.get("rootless") else set()
    seccomp_on = security.get("seccompEnabled") is True
    distribution = _dict(host.get("distribution"))
    return DockerHost(
        engine="podman",
        os_type=str(host.get("os", "")),
        arch=str(host.get("arch", "")),
        rootless=security.get("rootless") is True,
        cgroup_version=version,
        cgroup_driver=str(host["cgroupManager"]) if host.get("cgroupManager") else None,
        memory_limit="memory" in controllers,
        pids_limit="pids" in controllers,
        cpu_cfs_quota="cpu" in controllers,
        cpu_cfs_period="cpu" in controllers,
        seccomp_profile=(str(security.get("seccompProfilePath") or "default") if seccomp_on else None),
        apparmor=security.get("apparmorEnabled") is True,
        selinux=security.get("selinuxEnabled") is True,
        daemon_id=str(host.get("hostname", "")),
        server_version=str(_dict(info.get("version")).get("Version", "")),
        swap_limit=None,
        operating_system=str(distribution.get("distribution", "")),
    )


def inspect_host(
    docker: str = "docker", *, env: Mapping[str, str] | None = None, timeout: float = 10.0
) -> DockerHost:
    """Ask the daemon behind ``docker`` what it is and what it enforces.

    ``env`` is the CLI's environment (default: this process's), e.g. ``DOCKER_HOST``. Raises
    :class:`DockerProbeError` (reason ``missing``, ``unreachable`` or ``malformed``)."""
    info = _run_json(docker, ["info", "--format", "{{json .}}"], env, timeout)
    if isinstance(info.get("host"), dict) and "security" in info["host"]:
        return _from_podman_info(info)
    errors = info.get("ServerErrors")
    if errors:
        raise DockerProbeError(
            "unreachable", f"The daemon is not reachable: {'; '.join(map(str, errors))[:500]}"
        )
    if not info.get("OSType"):
        raise DockerProbeError("malformed", "docker info did not report the daemon's OSType")
    version = _run_json(docker, ["version", "--format", "{{json .}}"], env, timeout)
    return _from_docker_info(info, version)


def preflight(host: DockerHost) -> SandboxProbe:
    """Whether ``host`` can enforce the sandbox at all; refuses with a reason code when it cannot.

    * ``os-type``: the daemon does not run Linux containers (e.g. Windows containers);
    * ``limits``: the daemon cannot enforce memory, process or CPU limits (it would run without them);
    * ``seccomp``: no seccomp filter, or the daemon's default profile is ``unconfined``;
    * ``rootless-cgroup``: a rootless daemon on cgroup v1, which cannot apply any resource limit.

    Passing the preflight is necessary, not sufficient: the self-test then checks a real container.
    """
    facts = host.as_facts()

    def refuse(reason: str, detail: str) -> SandboxProbe:
        return SandboxProbe("docker", False, reason, detail, facts)

    if host.os_type.lower() != "linux":
        return refuse(
            "os-type", f"The daemon runs {host.os_type or 'unknown'} containers; Linux containers are needed"
        )
    # A CPU flag the daemon does not report is left to the self-test, which reads the container's cgroup v2
    # ``cpu.max`` and refuses unless the quota is in place; without cgroup v2 nothing verifies it: refuse.
    deferred = host.cgroup_version == "2"
    missing = [
        flag
        for flag, ok in (
            ("MemoryLimit", host.memory_limit),
            ("PidsLimit", host.pids_limit),
            ("CpuCfsQuota", host.cpu_cfs_quota),
            ("CpuCfsPeriod", host.cpu_cfs_period),
        )
        if not (ok or (ok is None and deferred))
    ]
    if host.rootless and host.cgroup_version == "1":
        return refuse("rootless-cgroup", "A rootless daemon on cgroup v1 cannot enforce resource limits")
    if missing:
        return refuse(
            "limits",
            "The daemon cannot enforce "
            + ", ".join(missing)
            + " (for rootless engines: delegate the cpu, memory "
            "and pids cgroup controllers to the user)",
        )
    if host.seccomp_profile is None:
        return refuse("seccomp", "The daemon reports no seccomp filter")
    if host.seccomp_profile == "unconfined":
        return refuse("seccomp", "The daemon's default seccomp profile is unconfined")
    return SandboxProbe("docker", True, facts=facts)
