"""The in-container self-test: proves on a real container that the daemon applied the sandbox shape.

:data:`SELF_TEST_SCRIPT` runs (standard library only) as ``python -I -B -c`` in a container started with
exactly :func:`~orkexa.dynamic.docker.run_args` and prints one JSON report; :func:`evaluate_self_test` checks
it against the limits on the host. A daemon may accept a flag and not apply it (a missing cgroup controller,
an unconfined seccomp default, a runtime that ignores ``noexec``), or add what nobody asked for (a volume from
the engine's defaults, e.g. Podman's ``containers.conf``; a permissive custom seccomp profile); the self-test
is what catches that. Besides the process status it reads the whole mount table (:func:`check_mount_table`),
calls system calls that every default seccomp profile blocks (:func:`check_seccomp_probes`) and, where the
host reports AppArmor, the container's AppArmor label (:func:`check_apparmor`).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import re
from typing import Any

from ._container import ContainerLimits
from ._host import _dict

__all__ = [
    "ALLOWED_ENVIRONMENT",
    "SAFE_IMAGE_VARIABLES",
    "SECCOMP_PROBES",
    "SELF_TEST_SCRIPT",
    "MountEntry",
    "check_apparmor",
    "check_mount_table",
    "check_seccomp_probes",
    "evaluate_self_test",
    "parse_mountinfo",
]

#: Variables the engine itself sets in a container (besides the image's own ``ENV``).
ALLOWED_ENVIRONMENT = frozenset({"PATH", "HOME", "HOSTNAME", "TERM", "container"})
#: Image variables whose names look like secrets but are public (the official Python images set them).
SAFE_IMAGE_VARIABLES = frozenset({"GPG_KEY", "PYTHON_SHA256", "PYTHON_GET_PIP_SHA256"})
_SECRET_LIKE = re.compile(r"KEY|TOKEN|SECRET|PASS|CRED|AUTH|COOKIE|SESSION|PRIVATE|CERT", re.IGNORECASE)
_MIB = 1024 * 1024
_VERSION = 2

#: System calls the self-test makes and that must fail: the default seccomp profiles of Docker (Moby) and
#: Podman (containers/common) both block them for a process without capabilities, while the kernel itself
#: lets an unprivileged process make them (with common settings). One that succeeds proves a seccomp profile
#: more permissive than the defaults. Not probed, because Podman's default allows them: ``unshare``,
#: ``keyctl``, ``mount``, ``ptrace``; ``io_uring_setup`` is allowed by Docker before 25.
SECCOMP_PROBES = ("add_key", "bpf", "perf_event_open", "userfaultfd", "move_pages")
#: Runs in the container; prints one JSON object. The marker comment names it in logs (and in test fakes).
SELF_TEST_SCRIPT = """\
# ORKEXA_SELF_TEST v2
import json, os, resource, subprocess

def read(path):
    try:
        with open(path) as handle:
            return handle.read().strip()
    except OSError:
        return None

def environment_names():
    # /proc/self/environ is the environment the container started this process with. os.environ also holds
    # what the interpreter set itself: C locale coercion (PEP 538) adds LC_CTYPE in an image without LANG.
    try:
        with open('/proc/self/environ', 'rb') as handle:
            entries = handle.read().split(b'\\0')
    except OSError:
        return sorted(os.environ)
    return sorted({entry.split(b'=', 1)[0].decode('utf-8', 'replace') for entry in entries if entry})

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
def probe_syscalls():
    import ctypes, struct
    machine = os.uname().machine
    names = ('add_key', 'bpf', 'perf_event_open', 'userfaultfd', 'move_pages')
    numbers = {
        'x86_64': dict(zip(names, (248, 321, 298, 323, 279))),
        'aarch64': dict(zip(names, (217, 280, 241, 282, 239))),
    }.get(machine)
    if numbers is None or struct.calcsize('P') != 8:
        return {'unsupported': '%s/%d-bit' % (machine, struct.calcsize('P') * 8)}
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    word = ctypes.c_long
    # perf_event_attr (PERF_ATTR_SIZE_VER0): a software task clock of this process, user space only
    # (type SOFTWARE, size, config TASK_CLOCK, flags disabled | exclude_kernel | exclude_hv)
    perf = struct.pack('=IIQQQQQIIQ', 1, 64, 1, 0, 0, 0, 1 | 32 | 64, 0, 0, 0)
    perf = ctypes.create_string_buffer(perf, 64)
    # bpf_attr for BPF_MAP_CREATE: an array map with one 4-byte entry
    bpf = ctypes.create_string_buffer(struct.pack('=IIIII', 2, 4, 4, 1, 0), 64)
    calls = {
        'add_key': (ctypes.c_char_p(b'user'), ctypes.c_char_p(b'orkexa-self-test'), ctypes.c_char_p(b'x'),
                    ctypes.c_size_t(1), word(-2)),
        'bpf': (word(0), bpf, word(64)),
        'perf_event_open': (perf, word(0), word(-1), word(-1), word(8)),
        'userfaultfd': (word(0o2000000 | 1),),
        'move_pages': (word(0), word(0), ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p(), word(0)),
    }
    results = {}
    for name, args in calls.items():
        ctypes.set_errno(0)
        value = libc.syscall(word(numbers[name]), *args)
        if value < 0:
            results[name] = 'errno:%d' % ctypes.get_errno()
            continue
        results[name] = 'allowed'
        if name in ('bpf', 'perf_event_open', 'userfaultfd'):
            try:
                os.close(value)
            except OSError:
                pass
    return results

def seccomp_probes():
    try:
        return probe_syscalls()
    except Exception as exc:
        return {'error': '%s: %s' % (type(exc).__name__, exc)}

def write_in(directory):
    path = os.path.join(directory, '.orkexa-self-test')
    try:
        return attempt(lambda: open(path, 'w').close())
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

apparmor = read('/proc/self/attr/apparmor/current')
if apparmor is None:
    apparmor = read('/proc/self/attr/current')
names = ('RLIMIT_NOFILE', 'RLIMIT_FSIZE', 'RLIMIT_CORE')
fields = ('Uid', 'Gid', 'Groups', 'CapInh', 'CapPrm', 'CapEff', 'CapBnd', 'CapAmb', 'NoNewPrivs', 'Seccomp')
report = {
    'version': 2,
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
    'env': environment_names(),
    'mountinfo': read('/proc/self/mountinfo'),
    'write_sealed': {point: write_in(point) for point in ('/run', '/var/tmp')},
    'apparmor': apparmor,
}
report['seccomp_probes'] = seccomp_probes()  # last: a call that gets through may change this process
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
    report: Mapping[str, Any],
    limits: ContainerLimits,
    image_variables: Iterable[str] = (),
    *,
    apparmor: bool = False,
) -> list[str]:
    """What is wrong with a container, according to its :data:`SELF_TEST_SCRIPT` report (empty: nothing).

    ``image_variables`` are the names of the image's ``ENV``; they may appear in the container, every other
    name except :data:`ALLOWED_ENVIRONMENT` is refused, and so is an image variable whose name looks like a
    secret (``*KEY*``, ``*TOKEN*``, ...) unless it is in :data:`SAFE_IMAGE_VARIABLES`. ``apparmor``: the host
    reports AppArmor (:attr:`~orkexa.dynamic.docker.DockerHost.apparmor`), so the container must be confined
    by an enforcing profile; without it the label is only reported."""
    problems: list[str] = []
    if report.get("version") != _VERSION:
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
    mountinfo = report.get("mountinfo")
    if isinstance(mountinfo, str) and mountinfo.strip():
        problems += check_mount_table(mountinfo)
    else:
        problems.append("the mount table is not reported")
    sealed = _dict(report.get("write_sealed"))
    for point in _SEALED:
        if sealed.get(point) == "allowed" or point not in sealed:
            problems.append(f"the sandbox user can write to {point}")
    problems += check_seccomp_probes(report.get("seccomp_probes"))
    problems += check_apparmor(report.get("apparmor"), required=apparmor)
    return problems


# ---- the mount table -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class MountEntry:
    """One line of ``/proc/self/mountinfo`` (``proc(5)``), with the escapes (``\\040``) decoded."""

    device: str
    root: str
    point: str
    options: frozenset[str]
    fstype: str
    source: str
    super_options: frozenset[str]

    @property
    def read_only(self) -> bool:
        """Read-only as a mount (per-mount ``ro``) or as a file system (super block ``ro``)."""
        return "ro" in self.options or "ro" in self.super_options


_ESCAPE = re.compile(r"\\([0-7]{3})")


def _unescape(text: str) -> str:
    return _ESCAPE.sub(lambda match: chr(int(match.group(1), 8)), text)


def parse_mountinfo(text: str) -> list[MountEntry]:
    """The entries of a ``/proc/self/mountinfo`` text; raises :class:`ValueError` on a malformed line."""
    entries = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if "-" not in parts[6:] or len(parts) < 10:
            raise ValueError(f"malformed mountinfo line: {line[:200]!r}")
        separator = parts.index("-", 6)
        if len(parts) < separator + 4:
            raise ValueError(f"malformed mountinfo line: {line[:200]!r}")
        entries.append(
            MountEntry(
                device=parts[2],
                root=_unescape(parts[3]),
                point=_unescape(parts[4]),
                options=frozenset(parts[5].split(",")),
                fstype=parts[separator + 1],
                source=_unescape(parts[separator + 2]),
                super_options=frozenset(parts[separator + 3].split(",")),
            )
        )
    return entries


#: Writable tmpfs file systems of the sandbox (checked in detail by :func:`evaluate_self_test`).
_SCRATCH = frozenset({"/tmp", "/work", "/dev/shm"})
#: Root-owned, empty ``noexec`` tmpfs file systems of :func:`~orkexa.dynamic.docker.run_args`.
_SEALED = ("/run", "/var/tmp")
#: Files the engine bind-mounts from its own state directory (Docker and Podman).
_ENGINE_FILES = frozenset({"/etc/hosts", "/etc/hostname", "/etc/resolv.conf", "/run/.containerenv"})
#: Init binaries an engine adds with ``--init`` (Docker's ``"init": true``, Podman's ``init = true``).
_INIT_BINARIES = frozenset({"/sbin/docker-init", "/run/podman-init"})
#: Device nodes a runtime in a user namespace (rootless) bind-mounts from the host's ``/dev``.
_DEVICES = frozenset({"null", "zero", "full", "tty", "random", "urandom"})
#: The kernel's own file systems a read-only mount under ``/sys`` may show.
_SYS_PSEUDO = frozenset(
    {
        "sysfs",
        "tmpfs",
        "cgroup",
        "cgroup2",
        "securityfs",
        "pstore",
        "bpf",
        "debugfs",
        "tracefs",
        "configfs",
        "fusectl",
        "efivarfs",
        "binfmt_misc",
    }
)
#: What a mount must be at a fixed place: file system types, and whether it must be read-only.
_FIXED: dict[str, tuple[frozenset[str], bool]] = {
    "/proc": (frozenset({"proc"}), False),
    "/dev": (frozenset({"tmpfs"}), False),
    "/dev/pts": (frozenset({"devpts"}), False),
    "/dev/mqueue": (frozenset({"mqueue"}), False),
    "/sys": (frozenset({"sysfs"}), True),
    "/sys/fs/cgroup": (frozenset({"cgroup2", "cgroup", "tmpfs"}), True),
}
_REQUIRED = ("/", "/proc", "/dev", "/sys", "/tmp", "/work", "/dev/shm", *_SEALED)


def _masked(entry: MountEntry) -> bool:
    """A path the runtime hides by binding ``/dev/null`` over it (runc and crun ``maskedPaths``)."""
    return entry.root == "/null" and entry.fstype in ("tmpfs", "devtmpfs")


def _mount_problem(entry: MountEntry, proc_devices: set[str]) -> str | None:
    point = entry.point
    mode = "read-only" if entry.read_only else "writable"
    unexpected = f"unexpected {mode} mount at {point} ({entry.fstype} {entry.source}, root {entry.root})"
    if point == "/":
        return None  # evaluate_self_test checks the root (read-only, not writable)
    if point in _SCRATCH:
        return None if entry.fstype == "tmpfs" else f"{point} is {entry.fstype}, not a tmpfs"
    if point in _SEALED:
        missing = {"noexec", "nosuid", "nodev"} - entry.options
        if entry.fstype != "tmpfs":
            return f"{point} is {entry.fstype}, not a tmpfs"
        return f"{point} is mounted without {', '.join(sorted(missing))}" if missing else None
    if point in _FIXED:
        fstypes, read_only = _FIXED[point]
        if entry.fstype not in fstypes or entry.root != "/":
            return unexpected
        if read_only and not entry.read_only:
            return f"{point} is writable"
        if point == "/dev" and "nosuid" not in entry.options:
            return "/dev is mounted without nosuid"
        return None
    if point in _ENGINE_FILES:
        return None if entry.root != "/" else unexpected  # a single file from the engine, not a file system
    if point in _INIT_BINARIES:
        return None if entry.read_only and entry.root != "/" else unexpected
    if point.startswith("/dev/") and point[5:] in _DEVICES:
        return None if entry.root == point[4:] and entry.fstype in ("devtmpfs", "tmpfs") else unexpected
    if point.startswith(("/proc/", "/sys/")) and _masked(entry):
        return None
    if point.startswith("/proc/"):
        # read-only views of the container's own /proc (readonlyPaths) and empty read-only tmpfs masks
        if entry.read_only and (
            (entry.fstype == "proc" and entry.device in proc_devices) or entry.fstype == "tmpfs"
        ):
            return None
        return unexpected
    if point.startswith("/sys/"):
        return None if entry.read_only and entry.fstype in _SYS_PSEUDO else unexpected
    return unexpected


def check_mount_table(mountinfo: str) -> list[str]:
    """What is wrong with a container's mount table (``/proc/self/mountinfo``; empty: nothing).

    Only what Docker and Podman mount themselves is allowed: the read-only root, ``/proc`` (with read-only
    or masked sub-paths), ``/dev`` with ``/dev/pts``, ``/dev/mqueue`` and the device nodes a rootless runtime
    binds, a read-only ``/sys`` and ``/sys/fs/cgroup``, the sandbox's tmpfs file systems, the engine's
    ``/etc/hosts``, ``/etc/hostname``, ``/etc/resolv.conf`` and ``/run/.containerenv``, and a read-only
    ``--init`` binary. Every other mount, read-only or not, is refused: a volume or bind mount from engine
    defaults (``containers.conf``, ``mounts.conf``) reaches the host's files."""
    try:
        entries = parse_mountinfo(mountinfo)
    except ValueError as exc:
        return [f"the mount table cannot be read: {exc}"]
    proc_devices = {entry.device for entry in entries if entry.point == "/proc" and entry.fstype == "proc"}
    problems = [problem for entry in entries if (problem := _mount_problem(entry, proc_devices))]
    present = {entry.point for entry in entries}
    problems += [f"nothing is mounted at {point}" for point in _REQUIRED if point not in present]
    return problems


# ---- seccomp and AppArmor -------------------------------------------------------------------------------


def check_seccomp_probes(probes: Any) -> list[str]:
    """What the :data:`SECCOMP_PROBES` results of a self-test report show (empty: every call was blocked).

    A call counts as blocked when it failed with any error: the probes prove a permissive profile, not a
    strict one (the kernel may refuse a call on its own). Missing results and architectures without probes
    (only x86-64 and arm64 have them) are refused."""
    if not isinstance(probes, dict):
        return ["the seccomp probes are not reported"]
    if "unsupported" in probes:
        return [f"no seccomp probes for the architecture {probes['unsupported']}"]
    if "error" in probes:
        return [f"the seccomp probes failed: {probes['error']}"]
    problems = []
    allowed = [name for name in SECCOMP_PROBES if probes.get(name) == "allowed"]
    if allowed:
        problems.append(
            f"the seccomp profile allows {', '.join(allowed)}, which the default profiles of Docker and "
            "Podman block: the daemon uses a more permissive custom profile"
        )
    missing = [name for name in SECCOMP_PROBES if not isinstance(probes.get(name), str)]
    if missing:
        problems.append(f"seccomp probes not reported: {', '.join(missing)}")
    return problems


# Profiles the engines load in enforce mode for every container (Docker's ``docker-default``, the
# ``containers-default-<version>`` profile of containers/common that Podman applies).
_CONTAINER_PROFILE = re.compile(r"^(docker-default|containers-default(-[0-9][0-9A-Za-z.\-]*)?)$")
# Profiles some distributions attach to the OCI runtime binary itself, as ``unconfined`` profiles that
# exist only to grant user namespaces (Ubuntu 23.10+ ``crun``, ``runc``).
_RUNTIME_PROFILE = re.compile(r"^(crun|runc|youki)$")


def _engine_stack(text: str) -> bool:
    """A stacked label such as ``containers-default-0.57.4-apparmor1//&crun (mixed)``: the container profile
    (enforcing) stacked with the OCI runtime's unconfined helper profile. A stack is confined by each of its
    members, so the container profile still applies; anything else in the stack is refused."""
    body, _, mode = text.rpartition(" ")
    if mode != "(mixed)" or "//&" not in body:
        return False
    members = [member.strip() for member in body.split("//&")]
    if len(members) < 2 or not _CONTAINER_PROFILE.match(members[0]):
        return False
    return all(_CONTAINER_PROFILE.match(m) or _RUNTIME_PROFILE.match(m) for m in members[1:])


def check_apparmor(label: Any, *, required: bool) -> list[str]:
    """Whether the container's AppArmor label (``/proc/self/attr/apparmor/current``) is an enforcing
    profile, as needed when the host reports AppArmor (``required``); otherwise nothing is refused."""
    if not required:
        return []
    text = str(label or "").strip()
    if text.endswith(("(enforce)", "(kill)")):
        return []
    if _engine_stack(text):
        return []
    return [
        "the host reports AppArmor, but the container is not confined by an enforcing profile "
        f"({text or 'no label'})"
    ]


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
