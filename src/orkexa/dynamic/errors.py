"""Exceptions of the dynamic agent extension.

All of them derive from :class:`orkexa.OrkexaError` via
:class:`DynamicAgentError`, and additionally from the closest builtin exception.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Literal

from ..errors import OrkexaError

if TYPE_CHECKING:
    from .policy import BudgetTracker

__all__ = [
    "ApprovalDeniedError",
    "BudgetExceededError",
    "DynamicAgentError",
    "PolicyViolationError",
    "RecursionLimitError",
    "RegistryError",
    "SandboxError",
    "ToolValidationError",
    "UnpricedModelError",
    "UnpricedUsageWarning",
    "UnreliablePriceError",
    "UnreliablePriceWarning",
    "UnsafeOptInWarning",
]


class DynamicAgentError(OrkexaError):
    """Base class of all errors raised by orkexa.dynamic."""


class PolicyViolationError(DynamicAgentError, ValueError):
    """A proposal or request exceeds what the DynamicPolicy allows."""


class BudgetExceededError(DynamicAgentError, RuntimeError):
    """The shared USD budget of a BudgetTracker has been used up.

    The tracker fills the optional attributes, so an application can react without
    parsing the message (e.g. offer to raise the limit with
    :meth:`BudgetTracker.set_limit <orkexa.dynamic.BudgetTracker.set_limit>`). All of
    them are ``None`` when the error was raised without them.

    Attributes:
        limit_usd: The limit of ``tracker`` at the time of the refusal.
        spent_usd: What ``tracker`` had spent (known costs).
        reserved_usd: What requests in flight held back in ``tracker``.
        requested_usd: The upper bound of the refused request's cost; ``None`` when it
            is unknown or unbounded, or when no request was refused (``"exceeded"``).
        tracker: The tracker whose limit refused the request. With child trackers
            (``parent=``) it is the tracker whose limit actually blocked, which may be
            a parent.
        reason: ``"used_up"`` (nothing left: ``spent >= limit``), ``"request_too_large"``
            (the request's bound does not fit even with nothing in flight,
            ``on_oversized="reject"``) or ``"exceeded"`` (recorded usage went past the
            limit).
    """

    def __init__(
        self,
        *args: object,
        limit_usd: Decimal | None = None,
        spent_usd: Decimal | None = None,
        reserved_usd: Decimal | None = None,
        requested_usd: Decimal | None = None,
        tracker: BudgetTracker | None = None,
        reason: Literal["used_up", "request_too_large", "exceeded"] | None = None,
    ) -> None:
        super().__init__(*args)
        self.limit_usd = limit_usd
        self.spent_usd = spent_usd
        self.reserved_usd = reserved_usd
        self.requested_usd = requested_usd
        self.tracker = tracker
        self.reason = reason


class UnpricedModelError(BudgetExceededError):
    """A request was refused because its model has no known price (``on_unpriced="reject"``)."""


class UnreliablePriceError(UnpricedModelError):
    """A price basis failed the tracker's :class:`~orkexa.pricing.PricePolicy`
    (``on_price_issue="reject"``): stale, of an unaccepted quality, or incomplete."""


class UnpricedUsageWarning(UserWarning):
    """Usage of a model without a known price; its cost is unknown, not zero."""


class UnreliablePriceWarning(UserWarning):
    """A price basis failed the tracker's :class:`~orkexa.pricing.PricePolicy`."""


class UnsafeOptInWarning(UserWarning):
    """An unsafe opt-in such as :class:`~orkexa.dynamic.AutoApprover` is in use."""


class RecursionLimitError(DynamicAgentError, RuntimeError):
    """Agent delegation went deeper than DynamicPolicy.max_depth."""


class SandboxError(DynamicAgentError, RuntimeError):
    """The sandbox could not run the code (timeout, crash, malformed output)."""


class ToolValidationError(DynamicAgentError, ValueError):
    """Generated tool code failed the static check or its tests."""


class ApprovalDeniedError(DynamicAgentError, RuntimeError):
    """The approver rejected a generated tool or agent."""


class RegistryError(DynamicAgentError, LookupError):
    """A tool or agent is unknown, revoked, or its stored code was modified."""
