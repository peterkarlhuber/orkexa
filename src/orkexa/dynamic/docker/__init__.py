"""Docker host checks for :class:`~orkexa.dynamic.DockerSandbox`.

* :func:`inspect_host` and :func:`preflight`: what the daemon enforces, and a refusal with a reason code
  when that is not enough (``os-type``, ``limits``, ``seccomp``, ``rootless-cgroup``).

``python -m orkexa.dynamic.docker --self-test`` runs the checks on this computer (see
``docs/development/docker-manual-verification.md``).
"""

from __future__ import annotations

from ._host import DockerHost, DockerProbeError, Engine, host_fingerprint, inspect_host, preflight

__all__ = ["DockerHost", "DockerProbeError", "Engine", "host_fingerprint", "inspect_host", "preflight"]
