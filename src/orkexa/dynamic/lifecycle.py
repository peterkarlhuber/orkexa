"""Cancelling sandbox calls: a per-call cancel check and the processes a sandbox is running.

A sandbox is often shared by many callers (every run of a workflow, every agent of an orchestrator), so
cancelling one caller must stop only that caller's call. The caller binds a *cancel check* to its thread with
:func:`cancel_scope` (a context variable, so the threads that :mod:`orkexa` starts for tool calls inherit it
when they copy the context); every sandbox call polls it and, once it returns ``True``, kills the call's
process (and removes its container), cleans up and raises :class:`SandboxCancelled`::

    stop = threading.Event()
    with cancel_scope(stop.is_set):
        sandbox.call(code, "tool", payload)   # another thread: stop.set()

:meth:`orkexa.dynamic.Sandbox.cancel` stops every call one sandbox is running, of every caller.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from .._cancel import CancelCheck, bind_cancel, cancel_requested, current_cancel, reset_cancel
from .errors import SandboxError

__all__ = [
    "CancelCheck",
    "SandboxCancelled",
    "bind_cancel",
    "cancel_requested",
    "cancel_scope",
    "current_cancel",
    "reset_cancel",
]


class SandboxCancelled(SandboxError):
    """The call was stopped because its caller cancelled it; its process is gone and its files are removed."""


@contextmanager
def cancel_scope(check: CancelCheck | None) -> Iterator[None]:
    """Sandbox calls made inside the block stop when ``check()`` returns ``True``."""
    token = bind_cancel(check)
    try:
        yield
    finally:
        reset_cancel(token)
