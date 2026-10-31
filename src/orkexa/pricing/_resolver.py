"""Price resolvers (thread-safe holders of the active snapshot) and pinned pricing bases."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
import threading
from typing import TYPE_CHECKING, Union

from ..errors import PricingError
from ._cost import CostBreakdown, MeteredUsage, PriceReference, calculate_cost
from ._model import ModelPrice
from ._snapshot import PriceLookup, PriceSnapshot

if TYPE_CHECKING:
    from ._legacy import ModelPricing

__all__ = ["PriceResolver", "PricingBasis", "PricingInput"]


class PriceResolver:
    """Resolves prices from one active :class:`PriceSnapshot`; isolated per instance.

    A resolver holds no global state: two resolvers never see each other's prices.
    Its active snapshot can be replaced at any time (:meth:`publish`,
    :meth:`register`); the replacement is atomic, and callers that pinned a
    snapshot (:meth:`basis`, ``snapshot=``) keep calculating with the old one, so a
    request is never priced with a mix of two price versions.

    Previously published snapshots stay available by version (up to
    ``max_history``) for audits and recalculations.

    Args:
        snapshot: The initial snapshot, or prices to build one from.
        version: Version id when ``snapshot`` is a list of prices
            (default: ``"local+<content hash>"``).
        max_history: How many snapshots :meth:`get_snapshot` keeps.
    """

    def __init__(
        self,
        snapshot: PriceSnapshot | Iterable[ModelPrice] | None = None,
        *,
        version: str | None = None,
        max_history: int = 32,
    ) -> None:
        if max_history < 1:
            raise ValueError("max_history must be at least 1")
        self._lock = threading.Lock()
        self._max_history = max_history
        self._history: OrderedDict[str, PriceSnapshot] = OrderedDict()
        if isinstance(snapshot, PriceSnapshot):
            initial = snapshot
        else:
            prices = tuple(snapshot or ())
            draft = PriceSnapshot("draft", prices)
            initial = PriceSnapshot(version or f"local+{draft.content_hash()[:12]}", prices)
        self._snapshot = initial
        self._remember(initial)

    # ---- snapshots -----------------------------------------------------------------------

    @property
    def snapshot(self) -> PriceSnapshot:
        """The active snapshot (read once and keep it to price consistently)."""
        return self._snapshot

    def publish(self, snapshot: PriceSnapshot) -> None:
        """Make ``snapshot`` the active one (atomic). Versions must be unique."""
        if not isinstance(snapshot, PriceSnapshot):
            raise TypeError("publish() needs a PriceSnapshot")
        with self._lock:
            known = self._history.get(snapshot.version)
            if known is not None and known.content_hash() != snapshot.content_hash():
                raise PricingError(
                    f"Snapshot version {snapshot.version!r} was already published with other prices"
                )
            self._snapshot = snapshot
            self._remember(snapshot)

    def register(self, *prices: ModelPrice, version: str | None = None) -> PriceSnapshot:
        """Add or replace prices; publishes and returns the new snapshot."""
        with self._lock:
            snapshot = self._snapshot.with_prices(prices, version=version)
            known = self._history.get(snapshot.version)
            if known is not None and known.content_hash() != snapshot.content_hash():
                raise PricingError(
                    f"Snapshot version {snapshot.version!r} was already published with other prices"
                )
            self._snapshot = snapshot
            self._remember(snapshot)
            return snapshot

    def get_snapshot(self, version: str) -> PriceSnapshot:
        """A previously published snapshot (``KeyError`` if unknown or evicted)."""
        with self._lock:
            return self._history[version]

    @property
    def versions(self) -> tuple[str, ...]:
        """Versions of the retained snapshots, oldest first."""
        with self._lock:
            return tuple(self._history)

    def _remember(self, snapshot: PriceSnapshot) -> None:
        self._history[snapshot.version] = snapshot
        self._history.move_to_end(snapshot.version)
        while len(self._history) > self._max_history:
            self._history.popitem(last=False)

    def _select(self, snapshot: PriceSnapshot | str | None) -> PriceSnapshot:
        if snapshot is None:
            return self.snapshot
        if isinstance(snapshot, PriceSnapshot):
            return snapshot
        return self.get_snapshot(snapshot)

    # ---- resolution and calculation ------------------------------------------------------

    def resolve(
        self,
        provider: str,
        model: str,
        *,
        at: datetime | None = None,
        snapshot: PriceSnapshot | str | None = None,
    ) -> PriceLookup:
        """Look up a price in the active snapshot (or the given snapshot / version)."""
        return self._select(snapshot).lookup(provider, model, at=at)

    def basis(self, *, snapshot: PriceSnapshot | str | None = None) -> PricingBasis:
        """Pin the active (or given) snapshot for consistent pricing of one or more requests."""
        return PricingBasis(self._select(snapshot))

    def calculate(
        self,
        provider: str,
        model: str,
        usage: MeteredUsage,
        *,
        requested_model: str | None = None,
        at: datetime | None = None,
        snapshot: PriceSnapshot | str | None = None,
    ) -> CostBreakdown:
        """Resolve and price ``usage`` against one snapshot."""
        return self.basis(snapshot=snapshot).calculate(
            provider, model, usage, requested_model=requested_model, at=at
        )


@dataclass(frozen=True)
class PricingBasis:
    """One pinned price basis: a snapshot plus an optional agent-level override.

    Everything priced through the same basis uses exactly the same prices,
    whatever happens to the resolver it came from in the meantime.

    Attributes:
        snapshot: The snapshot prices are looked up in (``None``: none, every price
            not given by ``override`` is unknown).
        override: An agent's ``pricing=``: a :class:`ModelPrice` (used when it
            matches the provider and model, otherwise the snapshot is used) or a
            legacy ``ModelPricing`` (applies to every request of the agent).
    """

    snapshot: PriceSnapshot | None = None
    override: ModelPrice | ModelPricing | None = None

    @property
    def version(self) -> str | None:
        return None if self.snapshot is None else self.snapshot.version

    def lookup(
        self,
        provider: str,
        model: str,
        *,
        requested_model: str | None = None,
        at: datetime | None = None,
    ) -> PriceLookup:
        """Resolve the reported ``model``; fall back to ``requested_model`` if it is unknown.

        The fallback covers endpoints that report another id than the configured
        one (deployment names, ``:latest`` tags); it never crosses providers.
        """
        from ._legacy import ModelPricing

        override = self.override
        if isinstance(override, ModelPricing):
            return PriceLookup(
                provider,
                model,
                override.to_model_price(provider, model, source="custom"),
                "override",
                self.version,
            )
        candidates = [model] + ([requested_model] if requested_model and requested_model != model else [])
        first: PriceLookup | None = None
        for index, name in enumerate(candidates):
            if isinstance(override, ModelPrice) and override.provider == provider:
                found = PriceSnapshot("override", (override,)).lookup(provider, name, at=at)
                if found.price is not None:
                    return PriceLookup(provider, model, found.price, "override", self.version)
            if self.snapshot is None:
                result = PriceLookup(
                    provider, name, None, None, None, f"No known pricing for {provider}:{name}"
                )
            else:
                result = self.snapshot.lookup(provider, name, at=at)
            if result.price is not None:
                if index:
                    return PriceLookup(provider, model, result.price, "requested", result.snapshot_version)
                return result
            first = first or result
        assert first is not None
        return first

    def calculate(
        self,
        provider: str,
        model: str,
        usage: MeteredUsage,
        *,
        requested_model: str | None = None,
        at: datetime | None = None,
    ) -> CostBreakdown:
        """Resolve and price ``usage``; the breakdown carries a :class:`PriceReference`."""
        lookup = self.lookup(provider, model, requested_model=requested_model, at=at)
        price = lookup.price
        if price is None:
            return CostBreakdown.unknown("price")
        reference = PriceReference(
            provider=provider,
            model=model,
            price_model=price.model,
            matched_by=lookup.matched_by or "exact",
            snapshot_version=lookup.snapshot_version,
            price_version=price.version,
            source=price.source,
            quality=price.quality,
            checked_at=price.checked_at,
            valid_from=price.valid_from,
            valid_until=price.valid_until,
            currency=price.currency,
        )
        return calculate_cost(price, usage, reference=reference, model=model)


PricingInput = Union["ModelPricing", ModelPrice, PriceResolver, PriceSnapshot, PricingBasis, None]
"""Everything ``pricing=`` accepts (agents, ``record_*_usage``)."""
