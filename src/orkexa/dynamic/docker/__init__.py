"""Docker host checks and the container shape of :class:`~orkexa.dynamic.DockerSandbox`.

* :func:`inspect_host` and :func:`preflight`: what the daemon enforces, and a refusal with a reason code
  when that is not enough (``os-type``, ``limits``, ``seccomp``, ``rootless-cgroup``).
* :class:`ContainerLimits` and :func:`run_args`: the one ``docker run`` shape of every sandbox container.
* :data:`BOOTSTRAP` and :func:`runner_files`: code, payload and runner travel on stdin, nothing is mounted.
* :func:`list_sandbox_containers` and :func:`reap`: labelled containers left behind by a crashed process.

``python -m orkexa.dynamic.docker --self-test`` runs the checks on this computer (see
``docs/development/docker-manual-verification.md``).
"""

from __future__ import annotations

from ._container import (
    BOOTSTRAP,
    DEADLINE_LABEL,
    OWNER_LABEL,
    RUNNER_MODULES,
    ContainerLimits,
    SandboxContainer,
    copy_runner,
    envelope,
    list_sandbox_containers,
    reap,
    run_args,
    runner_files,
    runner_fingerprint,
    stop_container,
)
from ._host import DockerHost, DockerProbeError, Engine, host_fingerprint, inspect_host, preflight

__all__ = [
    "BOOTSTRAP",
    "DEADLINE_LABEL",
    "OWNER_LABEL",
    "RUNNER_MODULES",
    "ContainerLimits",
    "DockerHost",
    "DockerProbeError",
    "Engine",
    "SandboxContainer",
    "copy_runner",
    "envelope",
    "host_fingerprint",
    "inspect_host",
    "list_sandbox_containers",
    "preflight",
    "reap",
    "run_args",
    "runner_files",
    "runner_fingerprint",
    "stop_container",
]
