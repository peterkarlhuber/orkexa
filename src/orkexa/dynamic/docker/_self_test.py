"""The in-container self-test: proves on a real container that the daemon applied the sandbox shape.

:data:`SELF_TEST_SCRIPT` runs (standard library only) as ``python -I -B -c`` in a container started with
exactly :func:`~orkexa.dynamic.docker.run_args` and prints one JSON report; :func:`evaluate_self_test` checks
it against the limits on the host. A daemon may accept a flag and not apply it (a missing cgroup controller,
an unconfined seccomp default, a runtime that ignores ``noexec``); the self-test is what catches that.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import re
from typing import Any

from ._container import ContainerLimits
from ._host import _dict

__all__ = ["ALLOWED_ENVIRONMENT", "SAFE_IMAGE_VARIABLES", "SELF_TEST_SCRIPT", "evaluate_self_test"]

#: Variables the engine itself sets in a container (besides the image's own ``ENV``).
ALLOWED_ENVIRONMENT = frozenset({"PATH", "HOME", "HOSTNAME", "TERM", "container"})
#: Image variables whose names look like secrets but are public (the official Python images set them).
SAFE_IMAGE_VARIABLES = frozenset({"GPG_KEY", "PYTHON_SHA256", "PYTHON_GET_PIP_SHA256"})
_SECRET_LIKE = re.compile(r"KEY|TOKEN|SECRET|PASS|CRED|AUTH|COOKIE|SESSION|PRIVATE|CERT", re.IGNORECASE)
_MIB = 1024 * 1024

#: Runs in the container; prints one JSON object. The marker comment names it in logs (and in test fakes).
SELF_TEST_SCRIPT = """\
# ORKEXA_SELF_TEST v1
import json, os, resource, subprocess

def read(path):
    try:
        with open(path) as handle:
            return handle.read().strip()
    except OSError:
        return None

def attempt(action):
    try:
        action()
    except OSError as exc:
        return '%s:%s' % (type(exc).__name__, exc.errno)
    return 'allowed'

def size(path):
    try:
        stat = os.statvfs(path)
    except OSError:
        return None
    return stat.f_blocks * stat.f_frsize

def exec_in(directory):
    path = os.path.join(directory, '.orkexa-self-test')
    def run():
        with open(path, 'w') as handle:
            handle.write('#!/bin/sh\\nexit 0\\n')
        os.chmod(path, 0o755)
        subprocess.run([path], check=False)
    try:
        return attempt(run)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

status = {}
for line in (read('/proc/self/status') or '').splitlines():
    key, _, value = line.partition(':')
    status[key] = value.strip()
devices = (read('/proc/net/dev') or '').splitlines()[2:]
interfaces = sorted(line.split(':')[0].strip() for line in devices if ':' in line)
mounts = {}
for line in (read('/proc/self/mountinfo') or '').splitlines():
    parts = line.split()
    if '-' not in parts:
        continue
    separator = parts.index('-')
    options = set(parts[5].split(',')) | set(parts[separator + 3].split(','))
    mounts[parts[4]] = sorted(options)
lines = [line for line in (read('/proc/self/cgroup') or '').splitlines() if line]
cgroup = {'version': '2' if len(lines) == 1 and lines[0].startswith('0::') else '1'}
if cgroup['version'] == '2':
    relative = lines[0][3:]
    for base in ('/sys/fs/cgroup', '/sys/fs/cgroup' + relative):
        if os.path.exists(os.path.join(base, 'memory.max')):
            for name in ('memory.max', 'memory.swap.max', 'pids.max', 'cpu.max'):
                cgroup[name] = read(os.path.join(base, name))
            break
names = ('RLIMIT_NOFILE', 'RLIMIT_FSIZE', 'RLIMIT_CORE')
fields = ('Uid', 'Gid', 'Groups', 'CapInh', 'CapPrm', 'CapEff', 'CapBnd', 'CapAmb', 'NoNewPrivs', 'Seccomp')
report = {
    'version': 1,
    'status': {key: status.get(key) for key in fields},
    'uid': os.getuid(),
    'gid': os.getgid(),
    'interfaces': interfaces,
    'write_root': attempt(lambda: open('/.orkexa-self-test', 'w').close()),
    'mounts': {point: mounts.get(point) for point in ('/', '/tmp', '/work', '/dev/shm')},
    'sizes': {point: size(point) for point in ('/tmp', '/work', '/dev/shm')},
    'exec_work': exec_in('/work'),
    'exec_tmp': exec_in('/tmp'),
    'cgroup': cgroup,
    'rlimits': {name: list(resource.getrlimit(getattr(resource, name))) for name in names},
    'env': sorted(os.environ),
}
print(json.dumps(report))
"""


def _zero_capabilities(value: Any) -> bool:
    try:
        return int(str(value), 16) == 0
    except ValueError:
        return False


def _ids(value: Any) -> set[str]:
    return set(str(value or "").split())


def evaluate_self_test(
    report: Mapping[str, Any], limits: ContainerLimits, image_variables: Iterable[str] = ()
) -> list[str]:
    """What is wrong with a container, according to its :data:`SELF_TEST_SCRIPT` report (empty: nothing).

    ``image_variables`` are the names of the image's ``ENV``; they may appear in the container, every other
    name except :data:`ALLOWED_ENVIRONMENT` is refused, and so is an image variable whose name looks like a
    secret (``*KEY*``, ``*TOKEN*``, ...) unless it is in :data:`SAFE_IMAGE_VARIABLES`."""
    problems: list[str] = []
    if report.get("version") != 1:
        return ["the self-test report has an unknown version"]
    status = _dict(report.get("status"))
    if report.get("uid") != 65534 or report.get("gid") != 65534:
        problems.append(f"runs as uid {report.get('uid')} gid {report.get('gid')}, not 65534")
    if _ids(status.get("Uid")) != {"65534"} or _ids(status.get("Gid")) != {"65534"}:
        problems.append("real, effective, saved or file-system ids are not all 65534")
    for capability in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb"):
        if (
            capability in status
            and status[capability] is not None
            and not _zero_capabilities(status[capability])
        ):
            problems.append(f"{capability} is {status[capability]}, not 0")
    if status.get("CapEff") is None:
        problems.append("CapEff is not reported")
    if status.get("NoNewPrivs") != "1":
        problems.append(f"NoNewPrivs is {status.get('NoNewPrivs')}, not 1")
    if status.get("Seccomp") != "2":
        problems.append(f"Seccomp is {status.get('Seccomp')}, not 2 (filter)")
    if report.get("interfaces") != ["lo"]:
        problems.append(f"network interfaces {report.get('interfaces')}, expected only lo")
    mounts = _dict(report.get("mounts"))
    if report.get("write_root") == "allowed" or "ro" not in (mounts.get("/") or []):
        problems.append("the root file system is writable")
    for point, key in (("/work", "exec_work"), ("/tmp", "exec_tmp")):
        options = set(mounts.get(point) or [])
        missing = {"noexec", "nosuid", "nodev"} - options
        if missing:
            problems.append(f"{point} is mounted without {', '.join(sorted(missing))}")
        if report.get(key) == "allowed":
            problems.append(f"a program in {point} could be executed")
    sizes = _dict(report.get("sizes"))
    for point, limit in (("/tmp", limits.tmp_mb), ("/work", limits.tmp_mb), ("/dev/shm", limits.shm_mb)):
        found = sizes.get(point)
        if not isinstance(found, int) or found > limit * _MIB:
            problems.append(f"{point} has {found} bytes, more than {limit} MiB")
    problems += _check_cgroup(report.get("cgroup"), limits)
    rlimits = _dict(report.get("rlimits"))
    fsize = limits.fsize_mb * _MIB
    for name, expected in (
        ("RLIMIT_NOFILE", [limits.nofile, limits.nofile]),
        ("RLIMIT_FSIZE", [fsize, fsize]),
        ("RLIMIT_CORE", [0, 0]),
    ):
        if rlimits.get(name) != expected:
            problems.append(f"{name} is {rlimits.get(name)}, expected {expected}")
    problems += _check_environment(report.get("env"), set(image_variables))
    return problems


def _check_cgroup(cgroup: Any, limits: ContainerLimits) -> list[str]:
    if not isinstance(cgroup, dict) or cgroup.get("version") != "2":
        return ["the container is not on cgroup v2, so its limits cannot be verified"]
    problems = []
    if cgroup.get("memory.max") != str(limits.memory_mb * _MIB):
        problems.append(f"memory.max is {cgroup.get('memory.max')}, expected {limits.memory_mb * _MIB}")
    if cgroup.get("memory.swap.max") not in (None, "0"):
        problems.append(f"memory.swap.max is {cgroup.get('memory.swap.max')}, expected 0")
    if cgroup.get("pids.max") != str(limits.pids):
        problems.append(f"pids.max is {cgroup.get('pids.max')}, expected {limits.pids}")
    quota, _, period = str(cgroup.get("cpu.max") or "").partition(" ")
    try:
        cpus = int(quota) / int(period)
    except (ValueError, ZeroDivisionError):
        cpus = None
    if cpus is None or abs(cpus - limits.cpus) > 0.01 * limits.cpus + 1e-6:
        problems.append(f"cpu.max is {cgroup.get('cpu.max')}, expected {limits.cpus:g} CPUs")
    return problems


def _check_environment(names: Any, image_variables: set[str]) -> list[str]:
    if not isinstance(names, list):
        return ["the environment is not reported"]
    problems = []
    unexpected = sorted(set(map(str, names)) - ALLOWED_ENVIRONMENT - image_variables)
    if unexpected:
        problems.append(f"unexpected environment variables: {', '.join(unexpected)}")
    secret_like = sorted(
        name
        for name in set(map(str, names)) & image_variables
        if _SECRET_LIKE.search(name) and name not in SAFE_IMAGE_VARIABLES
    )
    if secret_like:
        problems.append(f"the image sets variables that look like secrets: {', '.join(secret_like)}")
    return problems
