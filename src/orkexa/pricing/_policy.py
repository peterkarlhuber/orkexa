"""Rules for which price bases are acceptable (used by ``BudgetTracker``)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from ..errors import PricingError
from ._cost import CostBreakdown, PriceReference
from ._model import _QUALITIES, ModelPrice

__all__ = ["PricePolicy"]


@dataclass(frozen=True, init=False)
class PricePolicy:
    """Which prices are trustworthy enough, e.g. to enforce a budget with.

    Attributes:
        max_age: A price whose ``checked_at`` is older than this (or unset) is stale.
        accepted_quality: Accepted :data:`~orkexa.pricing.PriceQuality` values
            (``None``: all).
        allow_incomplete: Whether a cost with unknown components is acceptable.

    The default policy accepts everything; it is a pure check without side effects.
    """

    max_age: timedelta | None
    accepted_quality: frozenset[str] | None
    allow_incomplete: bool

    def __init__(
        self,
        *,
        max_age: timedelta | None = None,
        accepted_quality: Iterable[str] | None = None,
        allow_incomplete: bool = True,
    ) -> None:
        if max_age is not None and (not isinstance(max_age, timedelta) or max_age <= timedelta(0)):
            raise PricingError("max_age must be a positive timedelta")
        qualities = None if accepted_quality is None else frozenset(accepted_quality)
        if qualities is not None and not qualities <= set(_QUALITIES):
            raise PricingError(f"accepted_quality must be a subset of {_QUALITIES}")
        object.__setattr__(self, "max_age", max_age)
        object.__setattr__(self, "accepted_quality", qualities)
        object.__setattr__(self, "allow_incomplete", allow_incomplete)

    def _issues(
        self, label: str, quality: str, checked_at: datetime | None, at: datetime | None
    ) -> list[str]:
        issues = []
        if self.accepted_quality is not None and quality not in self.accepted_quality:
            issues.append(f"price of {label} has quality {quality!r}")
        if self.max_age is not None:
            now = at or datetime.now(timezone.utc)
            if checked_at is None:
                issues.append(f"price of {label} has no check date")
            elif now - checked_at > self.max_age:
                issues.append(f"price of {label} was last checked {checked_at.date().isoformat()} (stale)")
        return issues

    def check_price(self, price: ModelPrice, *, at: datetime | None = None) -> tuple[str, ...]:
        """Why ``price`` is not acceptable (empty: it is)."""
        return tuple(self._issues(f"{price.provider}:{price.model}", price.quality, price.checked_at, at))

    def check_reference(self, reference: PriceReference, *, at: datetime | None = None) -> tuple[str, ...]:
        label = f"{reference.provider}:{reference.model}"
        return tuple(self._issues(label, reference.quality, reference.checked_at, at))

    def check_cost(self, cost: CostBreakdown, *, at: datetime | None = None) -> tuple[str, ...]:
        """Issues of every price a cost was calculated with, plus incompleteness."""
        issues: list[str] = []
        for reference in cost.references:
            issues.extend(self.check_reference(reference, at=at))
        if not self.allow_incomplete and cost.missing and cost.missing != ("price",):
            issues.append(f"cost is incomplete (unknown: {', '.join(cost.missing)})")
        return tuple(dict.fromkeys(issues))
