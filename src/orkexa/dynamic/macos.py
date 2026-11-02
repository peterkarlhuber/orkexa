"""Experimental macOS sandbox: Apple's Seatbelt, applied by the sandbox process to itself.

Each call starts the interpreter as a new process that confines itself before it reads any tool code
(:mod:`orkexa.dynamic._macos_confine`): a deny-by-default Seatbelt profile (``sandbox_init``) that allows
reading only the interpreter, the system libraries and the call's directory, writing only to the call's
private ``tmp``, no network (also no localhost), no new processes or programs and no IPC; resource limits
for CPU time, file size, open files and processes; an empty environment. The host adds a wall-clock
timeout, a memory watchdog (macOS does not enforce ``RLIMIT_AS``: the host polls the process's physical
footprint and kills it above the limit), a limit for the size of ``tmp``, and kills the process group on
timeout or cancel.

**Experimental.** It is implemented and tested on GitHub's macOS runners, but its guarantees have not been
independently reviewed, ``sandbox_init`` is deprecated by Apple, and the memory limit is a watchdog (a short
peak can pass). Applications must not use it for untrusted code until it is reviewed and released; Orkexa
reports it as ``maturity="experimental"``.
"""

from __future__ import annotations

import contextlib
import ctypes
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, ClassVar

from .capabilities import SandboxCapabilities, SandboxProbe
from .errors import SandboxError
from .sandbox import Sandbox, response_limit, run_bounded

__all__ = ["MacOSSandbox"]

_HERE = Path(__file__).resolve().parent
_RUNNER = _HERE / "_sandbox_runner.py"
_CONFINE = _HERE / "_macos_confine.py"
_MARKER = "orkexa-sandbox-error:"
#: Variables every sandbox process gets; nothing is inherited from the host.
BASE_ENV = {
    "LANG": "en_US.UTF-8",
    "LC_ALL": "en_US.UTF-8",
    "PYTHONIOENCODING": "utf-8",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}
#: Most files and folders a call may create in its ``tmp``.
MAX_TMP_ENTRIES = 10_000
#: Seconds between two checks of a call's memory and temporary files.
GUARD_INTERVAL = 0.01

_PING = (
    "def ping(text: str) -> str:\n"
    '    """Echo the text.\n\n    Args:\n        text: Any text.\n\n'
    '    Returns:\n        The same text.\n    """\n'
    "    return text\n"
)

_PROBE: SandboxProbe | None = None
_PROBE_LOCK = threading.Lock()

_RUSAGE_FIELDS = (
    "ri_user_time",
    "ri_system_time",
    "ri_pkg_idle_wkups",
    "ri_interrupt_wkups",
    "ri_pageins",
    "ri_wired_size",
    "ri_resident_size",
    "ri_phys_footprint",
    "ri_proc_start_abstime",
    "ri_proc_exit_abstime",
    "ri_child_user_time",
    "ri_child_system_time",
    "ri_child_pkg_idle_wkups",
    "ri_child_interrupt_wkups",
    "ri_child_pageins",
    "ri_child_elapsed_abstime",
    "ri_diskio_bytesread",
    "ri_diskio_byteswritten",
)


class _RusageInfoV2(ctypes.Structure):
    """``struct rusage_info_v2`` of ``<sys/resource.h>``."""

    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [(name, ctypes.c_uint64) for name in _RUSAGE_FIELDS]


_libsystem: Any = None


def footprint(pid: int) -> int:
    """The physical memory footprint of a process in bytes (``proc_pid_rusage``), or -1 if unknown."""
    global _libsystem
    if sys.platform != "darwin":
        return -1
    if _libsystem is None:
        lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
        lib.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        lib.proc_pid_rusage.restype = ctypes.c_int
        _libsystem = lib
    info = _RusageInfoV2()
    if _libsystem.proc_pid_rusage(pid, 2, ctypes.byref(info)) != 0:  # 2 = RUSAGE_INFO_V2
        return -1
    return int(info.ri_phys_footprint)


class MacOSSandbox(Sandbox):
    """Runs Orkexa's sandbox runner confined by Seatbelt (module docstring). **Experimental.**

    Args:
        timeout: Wall-clock limit per call in seconds.
        memory_mb: Memory limit (physical footprint, enforced by a watchdog).
        tmp_mb: Limit for the total size of the call's ``tmp`` directory (and for one file).
        python: Interpreter, e.g. a virtual environment's; defaults to the current one.
        env: Extra variables (never secrets).
        read: Extra directories the code may read (e.g. a virtual environment's packages).
        protected: Paths that must never become readable; the home directory and the temp root always are.
    """

    isolated: ClassVar[bool] = True

    def __init__(
        self,
        *,
        timeout: float = 15.0,
        memory_mb: int = 1024,
        max_output_chars: int = 65_536,
        tmp_mb: int = 64,
        python: str | None = None,
        env: dict[str, str] | None = None,
        read: tuple[str, ...] = (),
        protected: tuple[str, ...] = (),
    ) -> None:
        super().__init__(timeout=timeout, memory_mb=memory_mb, max_output_chars=max_output_chars)
        self.tmp_mb = tmp_mb
        self.python = python or sys.executable
        self.env = {**(env or {}), **BASE_ENV}
        self.read = tuple(read)
        self.protected = tuple(protected)

    def capabilities(self) -> SandboxCapabilities:
        limits = {"wall_clock", "cpu", "memory", "file_size", "open_files", "processes", "output"}
        limits.add("temp_size")
        return SandboxCapabilities(
            backend="macos-seatbelt",
            isolation="os",
            filesystem="restricted",
            network="none",
            processes="blocked",
            clean_environment=True,
            limits=frozenset(limits),
            memory="watchdog",
            cancellable=True,
            maturity="experimental",
            platforms=("darwin",),
            architectures=("arm64", "x86_64"),
            notes=(
                "Uses sandbox_init (Seatbelt), which Apple has deprecated without a replacement.",
                "Memory and temporary files are checked by a watchdog every 10 ms; a short peak can pass.",
                "File metadata outside the allowed paths (existence, size) stays visible.",
                "System V IPC cannot be denied; shared memory segments a call leaves are removed afterwards.",
                "Not released: the independent review (ORK-014) found issues; use the VM sandbox instead.",
            ),
        )

    @classmethod
    def probe(cls, *, refresh: bool = False) -> SandboxProbe:
        """Checks this computer once (cached): macOS, and a confined self-test call."""
        global _PROBE
        with _PROBE_LOCK:
            if _PROBE is None or refresh:
                _PROBE = _check()
            return _PROBE

    def _execute(self, workdir: Path, request: str) -> str:
        tmp = workdir / "tmp"
        tmp.mkdir(mode=0o700)
        probe = workdir.parent / f"{workdir.name}.probe"
        probe.write_bytes(b"outside")
        listener = _listener()
        status_read, status_write = os.pipe()
        config = {
            "runner": str(_RUNNER),
            "read": [str(_RUNNER.parents[1]), *self.read],
            "write": [str(tmp)],
            "forbidden": [str(Path.home()), tempfile.gettempdir(), *self.protected],
            "probe_file": str(probe),
            "probe_port": listener.getsockname()[1] if listener is not None else 0,
            "status_fd": status_write,
            "limits": {"cpu_seconds": int(self.timeout) + 1, "file_bytes": self.tmp_mb * 1024 * 1024},
        }
        env = {**self.env, "TMPDIR": str(tmp), "HOME": str(tmp)}
        guard = _Guard(
            tmp,
            memory=self.memory_mb * 1024 * 1024,
            memory_mb=self.memory_mb,
            tmp_limit=self.tmp_mb * 1024 * 1024,
            tmp_mb=self.tmp_mb,
        )

        write_open = [True]

        def close_write() -> None:
            if write_open[0]:
                write_open[0] = False
                os.close(status_write)

        def popen(*args: Any, **kwargs: Any) -> Any:
            process = subprocess.Popen(*args, **kwargs)
            close_write()  # only the child keeps the write end
            guard.start(process)
            return process

        status = b""
        try:
            with self._call_scope() as stop:
                code, output = run_bounded(
                    [self.python, "-I", "-B", str(_CONFINE), json.dumps(config)],
                    request.encode("utf-8"),
                    timeout=self.timeout,
                    max_stdout=response_limit(self.max_output_chars),
                    env=env,
                    cwd=workdir,
                    stop=stop,
                    new_session=True,
                    watch=lambda _process: guard.problem,
                    pass_fds=(status_write,),
                    popen=popen,
                )
        except OSError as exc:
            raise SandboxError(f"Cannot start the sandbox interpreter {self.python}: {exc}") from exc
        finally:
            close_write()
            guard.stop()
            with contextlib.suppress(OSError):
                status = _read_all(status_read)
            os.close(status_read)
            probe.unlink(missing_ok=True)
            if listener is not None:
                listener.close()
            guard.remove_ipc()
        if guard.problem:
            raise SandboxError(guard.problem)
        report = status.decode("utf-8", "replace").strip()
        if report != "ok":
            # Only the confinement step writes to the status pipe, before any tool code runs: no "ok" means
            # the code was never run.
            detail = report.split(_MARKER, 1)[1].strip() if _MARKER in report else "no confirmation"
            raise SandboxError(f"The macOS sandbox could not confine the run, nothing was run ({detail})")
        stdout = output.stdout.decode("utf-8", "replace")
        stderr = output.stderr.decode("utf-8", "replace")
        if code < 0 and not stdout.strip():
            raise SandboxError(f"Sandbox process was killed by signal {-code}")
        if not stdout.strip():
            raise SandboxError(f"Sandbox exited with code {code}: {stderr[-1000:]}")
        return stdout


def _read_all(fd: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        chunk = os.read(fd, 4096)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


class _Guard:
    """Polls one call every :data:`GUARD_INTERVAL` seconds and kills its process group on a violation.

    Watches the physical memory footprint, the size and entry count of ``tmp`` (a directory the host cannot
    list counts as a violation, finding MS-4) and records the System V IPC segments the call creates so
    that they can be removed afterwards (Seatbelt cannot deny System V IPC, finding MS-3).
    """

    def __init__(self, tmp: Path, *, memory: int, memory_mb: int, tmp_limit: int, tmp_mb: int) -> None:
        self.tmp = tmp
        self.memory = memory
        self.memory_mb = memory_mb
        self.tmp_limit = tmp_limit
        self.tmp_mb = tmp_mb
        self.problem: str | None = None
        self.pid = 0
        self._done = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, process: Any) -> None:
        self.pid = process.pid
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._done.set()
        if self._thread is not None:
            self._thread.join(5)

    def _check(self) -> str | None:
        if footprint(self.pid) > self.memory:
            return f"The call used more than {self.memory_mb} MB of memory and was stopped"
        size, entries, hidden = _usage(self.tmp)
        if hidden:
            return "The call made part of its temporary files unreadable and was stopped"
        if size > self.tmp_limit:
            return f"The temporary files exceeded the limit of {self.tmp_mb} MB"
        if entries > MAX_TMP_ENTRIES:
            return f"The temporary files exceeded the limit of {MAX_TMP_ENTRIES} files"
        return None

    def _run(self) -> None:
        while not self._done.wait(GUARD_INTERVAL):
            problem = self._check()
            if problem:
                self.problem = problem
                with contextlib.suppress(OSError):
                    os.killpg(self.pid, 9)
                return

    def remove_ipc(self) -> None:
        """Remove the System V shared memory segments the call created (matched by creator pid)."""
        if self.pid:
            _remove_shm_of(self.pid)


def _remove_shm_of(pid: int) -> int:
    """Remove System V shared memory segments created by ``pid``; returns how many were removed."""
    if sys.platform != "darwin":
        return 0
    try:
        listing = subprocess.run(
            ["/usr/bin/ipcs", "-m", "-p"],
            capture_output=True,
            text=True,
            timeout=5,
            env={"PATH": "/usr/bin:/bin"},
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return 0
    removed = 0
    for line in listing.splitlines():
        fields = line.split()
        # "m <id> <key> <mode> <owner> <group> <cpid> <lpid>"
        if len(fields) >= 8 and fields[0] == "m" and fields[6] == str(pid):
            with contextlib.suppress(OSError, subprocess.SubprocessError):
                subprocess.run(
                    ["/usr/bin/ipcrm", "-m", fields[1]],
                    capture_output=True,
                    timeout=5,
                    env={"PATH": "/usr/bin:/bin"},
                )
                removed += 1
    return removed


def _listener() -> socket.socket | None:
    """A localhost TCP listener the confined process must not be able to reach."""
    try:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        return server
    except OSError:
        return None


def _usage(directory: Path) -> tuple[int, int, bool]:
    """Total size (allocated blocks), number of entries below ``directory`` and whether a directory below it
    could not be listed (its contents are then unknown), without following links."""
    total = count = 0
    hidden = False
    stack = [str(directory)]
    while stack:
        path = stack.pop()
        try:
            with os.scandir(path) as entries:
                for entry in entries:
                    count += 1
                    with contextlib.suppress(OSError):
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                        else:
                            total += getattr(entry.stat(follow_symlinks=False), "st_blocks", 0) * 512
        except FileNotFoundError:
            continue  # removed by the call in the meantime
        except OSError:
            hidden = True
    return total, count, hidden


def _check() -> SandboxProbe:
    system = f"macOS {platform.mac_ver()[0]}" if sys.platform == "darwin" else sys.platform
    facts = (("os", system), ("arch", platform.machine().lower()))
    if sys.platform != "darwin":
        detail = f"needs macOS, this is {sys.platform}"
        return SandboxProbe("macos-seatbelt", False, "platform", detail, facts)
    started = time.monotonic()
    try:
        result = MacOSSandbox(timeout=30.0).call(_PING, "ping", {"text": "ok"})
    except Exception as exc:
        text = str(exc)
        reason = "failed"
        for code in ("seatbelt", "verify", "paths", "limits", "root", "platform"):
            if f"({code}:" in text:
                reason = code
        return SandboxProbe("macos-seatbelt", False, reason, text, facts)
    if result != "ok":
        detail = f"unexpected self-test result {result!r}"
        return SandboxProbe("macos-seatbelt", False, "failed", detail, facts)
    elapsed = f"{(time.monotonic() - started) * 1000:.0f} ms"
    return SandboxProbe("macos-seatbelt", True, "", "", (*facts, ("self_test", elapsed)))
