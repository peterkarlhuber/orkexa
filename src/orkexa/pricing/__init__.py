"""Orkexa's pricing engine: price data, resolution, versioned snapshots and cost calculation.

The public API (everything in ``__all__``) is stable and needs no knowledge of the
internals:

* **Data model** - :class:`ModelPrice` with :class:`TokenRates`,
  :class:`LongContextTier`, :class:`Tariff`/:class:`TariffOption` and
  :class:`FeeRate`; amounts are :class:`~decimal.Decimal`, token rates per million
  tokens, currency explicit.
* **Snapshots** - :class:`PriceSnapshot`: an immutable, versioned, JSON-serializable
  set of prices; one snapshot is one price basis.
* **Resolution** - :class:`PriceResolver` (isolated, thread-safe, holds the active
  snapshot) and :class:`PricingBasis` (a pinned snapshot for consistent pricing).
* **Calculation** - :func:`calculate_cost` turns :class:`MeteredUsage` into a
  :class:`CostBreakdown` whose unknown parts are listed, never priced as 0.
* **Policy** - :class:`PricePolicy` decides which prices are trustworthy.

The original API (:class:`ModelPricing`, :data:`MODEL_PRICING`,
:func:`register_pricing`, :func:`resolve_pricing`) keeps working and feeds the
process-wide :func:`default_resolver`.

:data:`MODEL_PRICING` holds Anthropic first-party prices (USD per million tokens),
checked against https://platform.claude.com/docs/en/about-claude/pricing on
2026-10-31. Orkexa ships **no** prices for other providers.
"""

from __future__ import annotations

from ..errors import PriceNotFoundError, PricingError
from ._builtin import BUILTIN_SNAPSHOT, BUILTIN_VERSION
from ._cost import CostBreakdown, CostComponent, MeteredUsage, PriceReference, calculate_cost, round_cost
from ._legacy import (
    MODEL_PRICING,
    PROVIDER_PRICING,
    ModelPricing,
    default_resolver,
    register_pricing,
    resolve_pricing,
)
from ._model import (
    NEUTRAL_CONDITIONS,
    TOKENS_PER_RATE_UNIT,
    FeeRate,
    LongContextTier,
    ModelPrice,
    PriceQuality,
    Tariff,
    TariffOption,
    TokenRates,
)
from ._policy import PricePolicy
from ._resolver import PriceResolver, PricingBasis, PricingInput
from ._snapshot import SNAPSHOT_FORMAT, PriceLookup, PriceSnapshot

__all__ = [
    "BUILTIN_SNAPSHOT",
    "BUILTIN_VERSION",
    "MODEL_PRICING",
    "NEUTRAL_CONDITIONS",
    "PROVIDER_PRICING",
    "SNAPSHOT_FORMAT",
    "TOKENS_PER_RATE_UNIT",
    "CostBreakdown",
    "CostComponent",
    "FeeRate",
    "LongContextTier",
    "MeteredUsage",
    "ModelPrice",
    "ModelPricing",
    "PriceLookup",
    "PriceNotFoundError",
    "PricePolicy",
    "PriceQuality",
    "PriceReference",
    "PriceResolver",
    "PriceSnapshot",
    "PricingBasis",
    "PricingError",
    "PricingInput",
    "Tariff",
    "TariffOption",
    "TokenRates",
    "calculate_cost",
    "default_resolver",
    "pricing_basis",
    "register_pricing",
    "resolve_pricing",
    "round_cost",
]


def pricing_basis(pricing: PricingInput = None, *, resolver: PriceResolver | None = None) -> PricingBasis:
    """Pin a price basis for ``pricing`` (what agents accept as ``pricing=``).

    * ``None``: the active snapshot of ``resolver`` (default: :func:`default_resolver`).
    * a :class:`PriceResolver`: its active snapshot.
    * a :class:`PriceSnapshot`: that snapshot.
    * a :class:`ModelPrice` or :class:`ModelPricing`: an override on top of
      ``resolver``'s active snapshot.
    * a :class:`PricingBasis`: returned unchanged.
    """
    if isinstance(pricing, PricingBasis):
        return pricing
    if isinstance(pricing, PriceResolver):
        return pricing.basis()
    if isinstance(pricing, PriceSnapshot):
        return PricingBasis(pricing)
    base = resolver if resolver is not None else default_resolver()
    if pricing is None:
        return base.basis()
    if isinstance(pricing, (ModelPrice, ModelPricing)):
        return PricingBasis(base.snapshot, pricing)
    raise TypeError(
        "pricing must be a ModelPricing, ModelPrice, PriceResolver, PriceSnapshot, "
        f"PricingBasis or None, not {type(pricing).__name__}"
    )
