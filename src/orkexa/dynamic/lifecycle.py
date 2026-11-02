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

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token

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

#: Returns ``True`` once the caller wants its running call stopped. It must be cheap: it is polled
#: every 50 ms.
CancelCheck = Callable[[], bool]

_CHECK: ContextVar[CancelCheck | None] = ContextVar("orkexa_sandbox_cancel", default=None)


class SandboxCancelled(SandboxError):
    """The call was stopped because its caller cancelled it; its process is gone and its files are removed."""


def bind_cancel(check: CancelCheck | None) -> Token[CancelCheck | None]:
    """Bind ``check`` to the current context (thread or task); undo with :func:`reset_cancel`."""
    return _CHECK.set(check)


def reset_cancel(token: Token[CancelCheck | None]) -> None:
    _CHECK.reset(token)


@contextmanager
def cancel_scope(check: CancelCheck | None) -> Iterator[None]:
    """Sandbox calls made inside the block stop when ``check()`` returns ``True``."""
    token = bind_cancel(check)
    try:
        yield
    finally:
        reset_cancel(token)


def current_cancel() -> CancelCheck | None:
    """The check bound to the current context, for a thread that watches a call on behalf of its caller."""
    return _CHECK.get()


def cancel_requested(check: CancelCheck | None = None) -> bool:
    """Whether the caller of the current call cancelled it (``False`` without a bound check).

    A check that raises counts as "not cancelled": a broken check must never stop a call by itself."""
    found = check if check is not None else _CHECK.get()
    if found is None:
        return False
    try:
        return bool(found())
    except Exception:
        return False
