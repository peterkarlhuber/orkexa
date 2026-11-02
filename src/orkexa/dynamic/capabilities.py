"""What a sandbox backend actually enforces on the computer it runs on.

A single ``isolated`` flag cannot tell a container from a process with only a timeout, or a kernel-enforced
memory limit from a watchdog that polls. :class:`SandboxCapabilities` describes each guarantee separately, and
:class:`SandboxProbe` says whether the backend can run on this computer right now and, if not, why.

Orkexa only *reports* capabilities. It never decides which backend may run which code: that is the
application's policy (Nodestro, for example, accepts only containers and backends that passed its own
security tests and an independent review). A backend's name or technology is not evidence of isolation; its
probe and its tests are.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

__all__ = [
    "FileSystemAccess",
    "IsolationKind",
    "Maturity",
    "NetworkAccess",
    "ProcessControl",
    "SandboxCapabilities",
    "SandboxProbe",
]

#: The mechanism that separates the code from the host: none (a plain process), a container, or the operating
#: system's own sandbox (Landlock and seccomp, Seatbelt, AppContainer).
IsolationKind = Literal["none", "container", "os"]
#: ``host``: every file the user can read; ``restricted``: only the interpreter, the system libraries and the
#: call's own directory (write only to its ``tmp``); ``private``: an own file system (a container).
FileSystemAccess = Literal["host", "restricted", "private"]
#: ``host``: like any process of the user, including localhost; ``none``: no sockets or no route at all.
NetworkAccess = Literal["host", "none"]
#: ``allowed``: may start programs; ``limited``: may start processes, but only inside a process limit;
#: ``blocked``: cannot start any process or program.
ProcessControl = Literal["allowed", "limited", "blocked"]
#: ``stable``: tested and meant for production use; ``experimental``: implemented, but its guarantees are
#: not yet proven (it must not be used for untrusted code until they are).
Maturity = Literal["stable", "experimental"]


@dataclass(frozen=True)
class SandboxCapabilities:
    """The guarantees of one sandbox backend; see the type aliases of this module for the values.

    ``limits`` names the limits the backend enforces: ``wall_clock``, ``cpu``, ``memory``, ``file_size``,
    ``open_files``, ``processes``, ``output``, ``temp_size``. ``memory`` says how memory is limited:
    ``kernel`` (the kernel refuses more), ``watchdog`` (the host polls and kills, so a short peak can pass) or
    ``none``.
    """

    backend: str
    isolation: IsolationKind
    filesystem: FileSystemAccess
    network: NetworkAccess
    processes: ProcessControl
    #: Whether the code sees an empty environment (no variables, keys or tokens of the host process).
    clean_environment: bool
    limits: frozenset[str]
    memory: Literal["kernel", "watchdog", "none"]
    #: Whether a running call stops at once when its caller cancels it (``orkexa.dynamic.lifecycle``).
    cancellable: bool
    maturity: Maturity
    #: ``sys.platform`` values the backend supports (``linux``, ``darwin``, ``win32``) and CPU architectures
    #: (``platform.machine()`` values, lower case); empty means any.
    platforms: tuple[str, ...] = ()
    architectures: tuple[str, ...] = ()
    #: Known limitations, as short English sentences.
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def isolating(self) -> bool:
        """Whether the backend separates the code from the user's files, the network and other processes.

        This is a technical statement only; whether it is *good enough* for untrusted code is the
        application's decision (it should also require ``maturity == "stable"`` and its own evidence)."""
        return (
            self.isolation != "none"
            and self.filesystem != "host"
            and self.network == "none"
            and self.processes != "allowed"
            and self.clean_environment
        )

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["limits"] = sorted(self.limits)
        data["isolating"] = self.isolating
        return data


@dataclass(frozen=True)
class SandboxProbe:
    """Whether a backend can run here. ``reason`` is a short code when it cannot (``platform``, ``missing``,
    ``unreachable``, ``kernel``, ``verify``, ``failed``, ...), ``detail`` a sentence for people."""

    backend: str
    available: bool
    reason: str = ""
    detail: str = ""
    #: Facts the probe found, e.g. ``{"os": "macOS 15.4", "arch": "arm64"}``.
    facts: tuple[tuple[str, str], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "available": self.available,
            "reason": self.reason,
            "detail": self.detail,
            "facts": dict(self.facts),
        }
