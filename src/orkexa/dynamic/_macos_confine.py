"""Confines the current process with Apple's Seatbelt before it runs tool code (experimental MacOSSandbox).

**Runs inside the sandbox process, before any tool code.** :class:`orkexa.dynamic.MacOSSandbox` starts a
fresh interpreter with this file and a JSON configuration as its only argument::

    python -I -B _macos_confine.py '{"runner": ..., "read": [...], "write": [...], "forbidden": [...]}'

It uses only the standard library. In this order it:

1. checks that it runs on macOS and not as root;
2. sets resource limits (CPU time, file size, open files, no new processes, no core dumps);
3. collects the directories the interpreter loads code from and refuses to start if one of them is, or
   contains, a forbidden path (the home directory, the temp root, paths the host protects);
4. applies a deny-by-default Seatbelt profile with ``sandbox_init``: read access only to the interpreter,
   the system libraries and the call's directory, write access only below its ``tmp``; no network (also
   no localhost and no UNIX sockets), no new processes or programs, no IPC or Mach services, signals only
   to itself;
5. verifies the confinement from the inside (a file outside the call, a connection to localhost, a new
   process and a write outside ``tmp`` must all be refused) and only then runs Orkexa's sandbox runner,
   which reads the request from stdin.

If a step fails, the process prints ``orkexa-sandbox-error: <code>: <detail>`` on stderr and exits with
:data:`EXIT_UNAVAILABLE` before any tool code is read.

``sandbox_init`` is deprecated by Apple without a replacement for this use; Chromium, Codex CLI and
Microsoft's MXC use it on macOS. Metadata of files outside the allowed paths (``stat``, existence) stays
visible, as with the built-in Linux sandbox of Nodestro.
"""

from __future__ import annotations

import ctypes
import json
import os
import runpy
import socket
import sys

EXIT_UNAVAILABLE = 97
MARKER = "orkexa-sandbox-error:"

#: System locations the interpreter and compiled packages may read. None of them holds user data.
SYSTEM_READ = (
    "/usr/lib",
    "/usr/share/zoneinfo",
    "/usr/share/icu",
    "/System/Library",
    "/Library/Apple/System/Library",
    "/private/var/db/timezone",
    "/private/etc/localtime",
    "/dev/urandom",
    "/dev/random",
    "/dev/null",
)


class Unavailable(Exception):
    """The confinement cannot be applied; ``code`` names the reason for the host."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code


def _is_within(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def interpreter_paths(forbidden: list[str]) -> list[str]:
    """The existing directories the interpreter loads code from.

    Raises :class:`Unavailable` if one of them is or contains a forbidden path."""
    candidates = {sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix}
    candidates.add(os.path.dirname(sys.executable))
    candidates.update(p for p in sys.path if p)
    candidates.add(os.path.dirname(os.path.realpath(sys.executable)))
    found: list[str] = []
    for candidate in candidates:
        real = os.path.realpath(candidate)
        if not os.path.isdir(real) or real in found:
            continue
        for path in forbidden:
            if _is_within(path, real):
                raise Unavailable("paths", f"the interpreter path {real} contains the protected path {path}")
        found.append(real)
    found.sort()
    return [p for p in found if not any(p != q and _is_within(p, q) for q in found)]


def _quote(path: str) -> str:
    return json.dumps(path)  # a valid SBPL string literal for any path (quotes and backslashes escaped)


def profile(read: list[str], write: list[str]) -> str:
    """The Seatbelt profile (SBPL): everything is denied unless a rule below allows it."""
    lines = [
        "(version 1)",
        "(deny default)",
        # Path lookups (stat, realpath, the import system's directory checks). Contents stay unreadable.
        "(allow file-read-metadata)",
        "(allow sysctl-read)",
        "(allow signal (target self))",
        "(allow process-info* (target self))",
        '(allow file-write-data (literal "/dev/null"))',
        '(allow file-ioctl (literal "/dev/null"))',
    ]
    for path in read:
        kind = "subpath" if os.path.isdir(path) else "literal"
        lines.append(f"(allow file-read* file-map-executable ({kind} {_quote(path)}))")
    for path in write:
        lines.append(f"(allow file-read* file-write* (subpath {_quote(path)}))")
    return "\n".join(lines) + "\n"


def _set_limits(limits: dict[str, int]) -> None:
    import resource

    def limit(kind: int, value: int) -> None:
        try:
            _soft, hard = resource.getrlimit(kind)
            value = value if hard == resource.RLIM_INFINITY else min(value, hard)
            resource.setrlimit(kind, (value, value))
        except (ValueError, OSError) as exc:
            raise Unavailable("limits", f"cannot set resource limit {kind}: {exc}") from exc

    limit(resource.RLIMIT_CORE, 0)
    limit(resource.RLIMIT_NOFILE, int(limits.get("open_files", 256)))
    limit(resource.RLIMIT_FSIZE, int(limits["file_bytes"]))
    limit(resource.RLIMIT_CPU, int(limits["cpu_seconds"]))
    # Counts every process of the user, so 0 refuses any new one; the profile denies process-fork too.
    limit(resource.RLIMIT_NPROC, 0)


def _apply(text: str) -> None:
    try:
        lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
        init = lib.sandbox_init
        free = lib.sandbox_free_error
    except (OSError, AttributeError) as exc:
        raise Unavailable("seatbelt", f"sandbox_init is not available: {exc}") from exc
    init.argtypes = [ctypes.c_char_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_char_p)]
    init.restype = ctypes.c_int
    free.argtypes = [ctypes.c_char_p]
    error = ctypes.c_char_p()
    if init(text.encode("utf-8"), 0, ctypes.byref(error)) != 0:
        message = error.value.decode("utf-8", "replace") if error.value else "unknown error"
        if error.value:
            free(error)
        raise Unavailable("seatbelt", f"cannot apply the profile: {message}")


def _verify(config: dict[str, object]) -> None:
    """Prove from the inside that the confinement holds, so a run never counts as sandboxed when it is not."""
    try:
        with open(str(config["probe_file"]), "rb"):
            pass
    except PermissionError:
        pass
    else:
        raise Unavailable("verify", "a file outside the sandbox is still readable")
    try:
        with open(os.path.join(os.getcwd(), "escape.txt"), "w", encoding="utf-8"):
            pass
    except PermissionError:
        pass
    else:
        raise Unavailable("verify", "the call's directory is writable outside tmp")
    port = int(config.get("probe_port", 0))  # type: ignore[call-overload]
    if port:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=2).close()
        except PermissionError:
            pass
        except OSError as exc:
            raise Unavailable("verify", f"localhost was not refused by the sandbox: {exc}") from exc
        else:
            raise Unavailable("verify", "a connection to localhost is still possible")
    try:
        child = os.fork()
    except OSError:
        pass
    else:  # pragma: no cover - only reached if the confinement failed
        if child == 0:
            os._exit(0)
        raise Unavailable("verify", "a new process can still be started")


def confine(config: dict[str, object]) -> None:
    """Apply every step of the module docstring to this process; raise :class:`Unavailable` if one fails."""
    if sys.platform != "darwin":
        raise Unavailable("platform", f"needs macOS, this is {sys.platform}")
    if os.geteuid() == 0:
        raise Unavailable("root", "the sandbox does not run as the administrator (root)")
    forbidden = [os.path.realpath(str(p)) for p in config.get("forbidden", [])]
    read = interpreter_paths(forbidden)
    read += [os.path.realpath(str(p)) for p in config.get("read", [])]
    read += [p for p in SYSTEM_READ if os.path.exists(p)]
    read.append(os.path.realpath(os.getcwd()))
    write = [os.path.realpath(str(p)) for p in config.get("write", [])]
    for path in read + write:
        for protected in forbidden:
            if _is_within(protected, path) and path not in write:
                raise Unavailable("paths", f"the sandbox path {path} contains the protected path {protected}")
    _set_limits(config["limits"])
    _apply(profile(read, write))
    _verify(config)


def main(argv: list[str]) -> int:
    try:
        config = json.loads(argv[1])
        confine(config)
    except Unavailable as exc:
        print(f"{MARKER} {exc.code}: {exc}", file=sys.stderr, flush=True)
        return EXIT_UNAVAILABLE
    except Exception as exc:
        print(f"{MARKER} internal: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return EXIT_UNAVAILABLE
    sys.argv = [str(config["runner"])]
    runpy.run_path(str(config["runner"]), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
