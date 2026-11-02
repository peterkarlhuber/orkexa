"""The shape of a sandbox container, the stdin transport, and the reaper for left-over containers.

Every sandbox container is started with :func:`run_args`: no network, read-only root file system, no
capabilities, no new privileges, user ``nobody``, private IPC, memory (without swap), CPU, process,
open-file, file-size and core limits, no logging, small ``noexec`` temporary file systems, never pulled, no
bind mount and no environment variable from the host. Code and payload reach it on stdin
(:data:`BOOTSTRAP`), together with the few Orkexa modules the runner needs (:func:`runner_files`). Each
container carries the labels ``org.orkexa.sandbox=<owner>`` and ``org.orkexa.deadline=<epoch seconds>``,
so :func:`reap` can remove the containers a crashed host process left behind.
"""

from __future__ import annotations

from collections.abc import Mapping
import contextlib
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import time
from typing import Any

from ..errors import SandboxError
from ._host import _dict

__all__ = [
    "BOOTSTRAP",
    "DEADLINE_LABEL",
    "OWNER_LABEL",
    "RUNNER_MODULES",
    "ContainerLimits",
    "SandboxContainer",
    "copy_runner",
    "envelope",
    "list_sandbox_containers",
    "reap",
    "run_args",
    "runner_files",
    "runner_fingerprint",
    "stop_container",
]

#: Names the owner of a sandbox container (e.g. ``orkexa``, ``nodestro-worker``).
OWNER_LABEL = "org.orkexa.sandbox"
#: When the container must be gone, in seconds since the epoch; :func:`reap` removes it after that.
DEADLINE_LABEL = "org.orkexa.deadline"
#: The user (and group) every sandbox container runs as.
SANDBOX_USER = "65534:65534"
_OWNER = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,62}$")
_LABEL_KEY = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,127}$")


@dataclass(frozen=True)
class ContainerLimits:
    """The resource limits of one sandbox container.

    Args:
        memory_mb: Memory limit; swap is not allowed (``--memory-swap`` equals ``--memory``).
        cpus: CPU limit (``--cpus``, a CFS quota).
        pids: Process (and thread) limit.
        nofile: Open-file limit (soft and hard).
        fsize_mb: Largest file the container may write.
        tmp_mb: Size of each of the ``/tmp`` and ``/work`` file systems (memory-backed).
        shm_mb: Size of ``/dev/shm``.
        lifetime_s: How long the container may exist; written to its deadline label.
    """

    memory_mb: int
    cpus: float
    pids: int
    nofile: int = 1024
    fsize_mb: int = 16
    tmp_mb: int = 16
    shm_mb: int = 16
    lifetime_s: float = 60.0

    def __post_init__(self) -> None:
        if self.memory_mb < 64 or self.pids < 1 or self.nofile < 16:
            raise ValueError("memory_mb must be >= 64, pids >= 1 and nofile >= 16")
        if not (math.isfinite(self.cpus) and self.cpus > 0) or not math.isfinite(self.lifetime_s):
            raise ValueError("cpus must be positive and lifetime_s finite")
        if min(self.fsize_mb, self.tmp_mb, self.shm_mb) < 1 or self.lifetime_s <= 0:
            raise ValueError("fsize_mb, tmp_mb and shm_mb must be >= 1 and lifetime_s positive")


def _check_owner(owner: str) -> str:
    if not _OWNER.match(owner):
        raise ValueError(f"Invalid sandbox owner {owner!r}: lower-case letters, digits, '.', '_', '-'")
    return owner


def run_args(
    docker: str,
    name: str,
    image: str,
    limits: ContainerLimits,
    *,
    owner: str = "orkexa",
    deadline: float | None = None,
    labels: Mapping[str, str] | None = None,
) -> list[str]:
    """``docker run`` up to and including the image, for one sandbox container (the command follows).

    ``image`` should be an image id (``sha256:...``), so the container runs exactly the image that was
    checked. ``deadline`` defaults to now plus ``limits.lifetime_s``. ``labels`` adds labels of the caller;
    they cannot replace the owner and deadline labels. There is no ``-v`` and no ``-e``: nothing of the host's
    file system or environment reaches the container.
    """
    _check_owner(owner)
    if not re.match(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$", name):
        raise ValueError(f"Invalid container name {name!r}")
    if not image or image.startswith("-"):
        raise ValueError(f"Invalid image {image!r}")
    extra = dict(labels or {})
    for key, value in extra.items():
        if not _LABEL_KEY.match(key) or key in (OWNER_LABEL, DEADLINE_LABEL) or "\n" in value:
            raise ValueError(f"Invalid label {key!r}")
    expires = math.ceil(deadline if deadline is not None else time.time() + limits.lifetime_s)
    fsize = limits.fsize_mb * 1024 * 1024
    tmpfs = f"rw,noexec,nosuid,nodev,size={limits.tmp_mb}m"
    args = [
        docker,
        "run",
        "--rm",
        "-i",
        "--pull",
        "never",
        "--name",
        name,
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--user",
        SANDBOX_USER,
        "--ipc",
        "private",
        "--shm-size",
        f"{limits.shm_mb}m",
        "--ulimit",
        f"nofile={limits.nofile}:{limits.nofile}",
        "--ulimit",
        f"fsize={fsize}:{fsize}",
        "--ulimit",
        "core=0:0",
        "--log-driver",
        "none",
        "--pids-limit",
        str(limits.pids),
        "--memory",
        f"{limits.memory_mb}m",
        "--memory-swap",
        f"{limits.memory_mb}m",
        "--cpus",
        f"{limits.cpus:g}",
        "--tmpfs",
        f"/tmp:{tmpfs}",
        "--tmpfs",
        # mode=1777 rather than uid=/gid=: Podman 4 rejects those tmpfs options. Every process in the
        # container runs as the sandbox user, so no other user exists there to read or replace the files.
        f"/work:{tmpfs},mode=1777",
        "-w",
        "/work",
        "--label",
        f"{OWNER_LABEL}={owner}",
        "--label",
        f"{DEADLINE_LABEL}={expires}",
    ]
    for key, value in sorted(extra.items()):
        args += ["--label", f"{key}={value}"]
    return [*args, image]


# ---- the stdin transport ------------------------------------------------------------------------------

#: The Orkexa modules the sandbox runner imports, relative to the ``orkexa`` package (a test checks it).
RUNNER_MODULES = (
    "content.py",
    "core/__init__.py",
    "core/types.py",
    "dynamic/_sandbox_runner.py",
    "errors.py",
    "tools.py",
)
_PACKAGE_DIR = Path(__file__).resolve().parents[2]  # .../orkexa


def runner_files() -> dict[str, str]:
    """The runner and the modules it imports, as ``{"orkexa/<path>": source}``; sent with every call."""
    return {
        f"orkexa/{module}": (_PACKAGE_DIR / module).read_text(encoding="utf-8") for module in RUNNER_MODULES
    }


def runner_fingerprint() -> str:
    """The SHA-256 of :func:`runner_files`: what an image that carries a copy of the runner must match."""
    digest = hashlib.sha256()
    for path, content in sorted(runner_files().items()):
        digest.update(path.encode("utf-8") + b"\0" + content.encode("utf-8") + b"\0")
    return digest.hexdigest()


def copy_runner(target: Path) -> list[Path]:
    """Write :func:`runner_files` below ``target`` (``target/orkexa/...``), e.g. into a build context."""
    written = []
    for path, content in sorted(runner_files().items()):
        destination = target.joinpath(*path.split("/"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
        written.append(destination)
    return written


#: ``python -I -B -c BOOTSTRAP`` in the container: reads one JSON envelope from stdin
#: (``{"runner": {...}, "files": {...}, "request": "..."}``), writes the runner modules below the temporary
#: directory and the tool files into the working directory (both ``noexec`` file systems; Python only reads
#: them), then runs the runner with the request as its stdin.
BOOTSTRAP = """\
import io, json, os, runpy, sys, tempfile
envelope = json.loads(sys.stdin.read())
root = os.path.join(tempfile.gettempdir(), 'orkexa-runner')
for path, content in envelope['runner'].items():
    parts = path.split('/')
    if parts[0] != 'orkexa' or not path.endswith('.py') or any(not p or p.startswith('.') for p in parts):
        raise SystemExit('invalid runner path')
    target = os.path.join(root, *parts)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, 'w', encoding='utf-8') as handle:
        handle.write(content)
work = os.getcwd()
for name, content in envelope['files'].items():
    if not name or '/' in name or '\\\\' in name or name.startswith('.'):
        raise SystemExit('invalid file name')
    with open(os.path.join(work, name), 'w', encoding='utf-8') as handle:
        handle.write(content)
runner = os.path.join(root, 'orkexa', 'dynamic', '_sandbox_runner.py')
sys.stdin = io.StringIO(envelope['request'])
sys.argv = [runner]
runpy.run_path(runner, run_name='__main__')
"""


def envelope(files: Mapping[str, str], request: str) -> bytes:
    """The stdin of :data:`BOOTSTRAP` for one call."""
    return json.dumps({"runner": runner_files(), "files": dict(files), "request": request}).encode("utf-8")


# ---- left-over containers ----------------------------------------------------------------------------


@dataclass(frozen=True)
class SandboxContainer:
    """A container with the owner label (:func:`list_sandbox_containers`)."""

    id: str
    name: str
    owner: str
    #: The deadline label, or ``None`` when it is missing or not a number.
    deadline: float | None
    running: bool


def _cli(
    docker: str, args: list[str], env: Mapping[str, str] | None, timeout: float
) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            [docker, *args],
            capture_output=True,
            timeout=timeout,
            check=False,
            env=dict(env) if env is not None else None,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SandboxError(f"{docker} {args[0]} failed: {exc}") from exc


def list_sandbox_containers(
    docker: str = "docker",
    owner: str = "orkexa",
    *,
    env: Mapping[str, str] | None = None,
    timeout: float = 30.0,
) -> list[SandboxContainer]:
    """The containers (running or not) labelled ``org.orkexa.sandbox=<owner>``.

    Raises :class:`SandboxError`."""
    listed = _cli(
        docker,
        ["ps", "-a", "--filter", f"label={OWNER_LABEL}={_check_owner(owner)}", "--format", "{{.ID}}"],
        env,
        timeout,
    )
    if listed.returncode != 0:
        raise SandboxError(f"{docker} ps failed: {listed.stderr.decode('utf-8', 'replace')[-500:]}")
    ids = [line.strip() for line in listed.stdout.decode("utf-8", "replace").splitlines() if line.strip()]
    if not ids:
        return []
    inspected = _cli(docker, ["container", "inspect", *ids], env, timeout)
    try:
        data: Any = json.loads(inspected.stdout.decode("utf-8", "replace") or "[]")
    except ValueError as exc:
        raise SandboxError(f"{docker} container inspect printed no JSON") from exc
    containers = []
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict):
            continue
        config = _dict(item.get("Config"))
        labels = _dict(config.get("Labels"))
        if labels.get(OWNER_LABEL) != owner:  # the filter is the daemon's; check again
            continue
        try:
            deadline: float | None = float(labels.get(DEADLINE_LABEL, ""))
        except ValueError:
            deadline = None
        if deadline is not None and not math.isfinite(deadline):
            deadline = None
        state = _dict(item.get("State"))
        containers.append(
            SandboxContainer(
                id=str(item.get("Id", "")),
                name=str(item.get("Name", "")).lstrip("/"),
                owner=owner,
                deadline=deadline,
                running=state.get("Running") is True,
            )
        )
    return containers


def reap(
    docker: str = "docker",
    owner: str = "orkexa",
    *,
    now: float | None = None,
    remove_all: bool = False,
    env: Mapping[str, str] | None = None,
    timeout: float = 30.0,
) -> int:
    """Remove the containers of ``owner`` whose deadline has passed (or that have no valid deadline).

    ``remove_all`` removes every container of ``owner``, e.g. when a host process starts and knows that none
    of its calls are running. Returns how many containers were removed. Raises :class:`SandboxError` when the
    containers cannot be listed."""
    moment = time.time() if now is None else now
    removed = 0
    for container in list_sandbox_containers(docker, owner, env=env, timeout=timeout):
        expired = remove_all or container.deadline is None or container.deadline <= moment
        if expired and _cli(docker, ["rm", "-f", container.id], env, timeout).returncode == 0:
            removed += 1
    return removed


def stop_container(
    docker: str, name: str, *, env: Mapping[str, str] | None = None, timeout: float = 30.0
) -> None:
    """Kill and remove the container ``name`` (killing the docker client does not stop it). Never raises."""
    for args in (["kill", name], ["rm", "-f", name]):
        with contextlib.suppress(SandboxError):
            _cli(docker, args, env, timeout)
