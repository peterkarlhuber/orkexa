"""Removed: the native macOS sandbox (Seatbelt). Docker is the sandbox for untrusted code on every platform.

Orkexa shipped an experimental ``MacOSSandbox`` (ORK-014) that confined a Python process with Apple's
Seatbelt. An independent review found it not releasable (MS-1: sandboxed code could read the environment of
same-user processes through ``KERN_PROCARGS2``; MS-2 to MS-10), and on 2026-11-02 the project decided
"Docker-first":
:class:`~orkexa.dynamic.DockerSandbox` is the default sandbox for untrusted code on all platforms. See
``docs/architecture/decisions/0012-macos-sandbox-removed.md``.

The name stays importable for one deprecation cycle so that code which only *reports* the backend keeps
working. It never runs anything: :meth:`MacOSSandbox.probe` always reports it unavailable (reason
``removed``), its capabilities claim no isolation, and every run raises :class:`~orkexa.dynamic.SandboxError`.
It will be deleted in a later release; use :class:`~orkexa.dynamic.DockerSandbox`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar
import warnings

from .capabilities import SandboxCapabilities, SandboxProbe
from .errors import SandboxError
from .sandbox import Sandbox

__all__ = ["MacOSSandbox"]

#: The backend name the experimental sandbox reported; kept so that status reports stay comparable.
BACKEND = "macos-seatbelt"
#: The probe's reason code and its sentence.
REASON = "removed"
DETAIL = "removed; use DockerSandbox (Docker-first)"


class MacOSSandbox(Sandbox):
    """Deprecated placeholder for the removed macOS Seatbelt sandbox; it never runs code.

    Creating one emits a :class:`DeprecationWarning` (status reports may still ask for its capabilities).
    :meth:`validate` and :meth:`call` fail with :class:`SandboxError` (``call`` wraps it in
    :class:`orkexa.ToolExecutionError`, like every backend) before any file is written. Constructor arguments
    of the former backend are accepted and ignored.
    """

    isolated: ClassVar[bool] = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            f"MacOSSandbox was {DETAIL}; it never runs code and will be deleted in a later release",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__()

    def capabilities(self) -> SandboxCapabilities:
        return SandboxCapabilities(
            backend=BACKEND,
            isolation="none",
            filesystem="host",
            network="host",
            processes="allowed",
            clean_environment=False,
            limits=frozenset(),
            memory="none",
            cancellable=False,
            maturity="experimental",
            platforms=("darwin",),
            notes=(f"MacOSSandbox was {DETAIL}. It never runs code.",),
        )

    @classmethod
    def probe(cls, *, refresh: bool = False) -> SandboxProbe:
        """Always unavailable: the backend was removed."""
        return SandboxProbe(BACKEND, False, REASON, DETAIL)

    def _run(self, files: dict[str, str], request: dict[str, Any]) -> dict[str, Any]:
        raise SandboxError(f"MacOSSandbox was {DETAIL}")

    def _execute(self, workdir: Path, request: str) -> str:
        raise SandboxError(f"MacOSSandbox was {DETAIL}")
