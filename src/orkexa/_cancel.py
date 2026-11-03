"""The per-call cancel check of sandbox calls (ORK-001), shared by the agents and :mod:`orkexa.dynamic`.

The public interface is :mod:`orkexa.dynamic.lifecycle` (:func:`~orkexa.dynamic.cancel_scope` and
friends); this module only holds the context variable, so that the agents can bind a check for the tool
calls they start without importing the sandbox backends.
"""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar, Token
import threading

__all__ = ["CancelCheck", "bind_cancel", "cancel_requested", "current_cancel", "linked_check", "reset_cancel"]

#: Returns ``True`` once the caller wants its running call stopped. It must be cheap: it is polled
#: every 50 ms.
CancelCheck = Callable[[], bool]

_CHECK: ContextVar[CancelCheck | None] = ContextVar("orkexa_sandbox_cancel", default=None)


def bind_cancel(check: CancelCheck | None) -> Token[CancelCheck | None]:
    """Bind ``check`` to the current context (thread or task); undo with :func:`reset_cancel`."""
    return _CHECK.set(check)


def reset_cancel(token: Token[CancelCheck | None]) -> None:
    _CHECK.reset(token)


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


def linked_check(stop: threading.Event) -> CancelCheck:
    """A check that fires when ``stop`` is set or when the check bound now (the caller's) fires.

    The agents bind it for one tool call and set ``stop`` when that call times out or its task is
    cancelled, so a sandbox call it started is stopped (its process killed), not abandoned (ORK2-01)."""
    outer = _CHECK.get()
    if outer is None:
        return stop.is_set
    return lambda: stop.is_set() or cancel_requested(outer)
