"""Docker host checks and the container shape of :class:`~orkexa.dynamic.DockerSandbox`.

* :func:`inspect_host` and :func:`preflight`: what the daemon enforces, and a refusal with a reason code
  when that is not enough (``os-type``, ``limits``, ``seccomp``, ``rootless-cgroup``).
* :class:`ContainerLimits` and :func:`run_args`: the one ``docker run`` shape of every sandbox container.
* :data:`BOOTSTRAP` and :func:`runner_files`: code, payload and runner travel on stdin, nothing is mounted.
* :func:`list_sandbox_containers` and :func:`reap`: labelled containers left behind by a crashed process.
* :data:`SELF_TEST_SCRIPT` and :func:`evaluate_self_test`: what a real container of that shape enforces
  (:meth:`~orkexa.dynamic.DockerSandbox.self_test`); :func:`resolve_image_id` and :func:`pin_base`: run
  images by id, pull base images only by digest.

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
from ._host import (
    DockerHost,
    DockerProbeError,
    Engine,
    host_fingerprint,
    inspect_host,
    preflight,
    redact_docker_output,
)
from ._image import ImageInfo, inspect_image, pin_base, resolve_image_id
from ._self_test import ALLOWED_ENVIRONMENT, SAFE_IMAGE_VARIABLES, SELF_TEST_SCRIPT, evaluate_self_test

__all__ = [
    "ALLOWED_ENVIRONMENT",
    "BOOTSTRAP",
    "DEADLINE_LABEL",
    "OWNER_LABEL",
    "RUNNER_MODULES",
    "SAFE_IMAGE_VARIABLES",
    "SELF_TEST_SCRIPT",
    "ContainerLimits",
    "DockerHost",
    "DockerProbeError",
    "Engine",
    "ImageInfo",
    "SandboxContainer",
    "copy_runner",
    "envelope",
    "evaluate_self_test",
    "host_fingerprint",
    "inspect_host",
    "inspect_image",
    "list_sandbox_containers",
    "pin_base",
    "preflight",
    "reap",
    "redact_docker_output",
    "resolve_image_id",
    "run_args",
    "runner_files",
    "runner_fingerprint",
    "stop_container",
]
