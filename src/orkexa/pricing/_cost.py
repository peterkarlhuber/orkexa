"""Cost calculation: metered usage in, an itemized, traceable cost breakdown out."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from typing import Any

from ..errors import PricingError
from ._model import (
    NEUTRAL_CONDITIONS,
    TOKENS_PER_RATE_UNIT,
    LongContextTier,
    ModelPrice,
    TokenRates,
    _aware,
    _FrozenDict,
    to_decimal,
)

__all__ = [
    "CostBreakdown",
    "CostComponent",
    "MeteredUsage",
    "PriceReference",
    "calculate_cost",
    "round_cost",
]

_PRECISION = 60  # digits; far beyond any realistic amount, so no intermediate rounding
_ZERO = Decimal("0")


def round_cost(amount: Decimal, places: int = 6, rounding: str = ROUND_HALF_EVEN) -> Decimal:
    """Round an amount for display or invoicing (banker's rounding by default).

    Orkexa itself never rounds: breakdowns keep full precision so sums of many
    small requests stay exact. Round only at the boundary where you present or
    book an amount.
    """
    return amount.quantize(Decimal(1).scaleb(-places), rounding=rounding)


def _pairs(value: Any, what: str, kind: type) -> Mapping[str, Any]:
    items = value.items() if isinstance(value, Mapping) else value
    result: dict[str, Any] = {}
    for key, item in items:
        if not isinstance(key, str) or not key:
            raise PricingError(f"{what} keys must be nonempty strings")
        if kind is int and (not isinstance(item, int) or isinstance(item, bool) or item < 0):
            raise PricingError(f"{what}[{key!r}] must be a nonnegative int")
        if kind is str and not isinstance(item, str):
            raise PricingError(f"{what}[{key!r}] must be a string")
        result[key] = item
    return _FrozenDict(sorted(result.items()))


@dataclass(frozen=True)
class MeteredUsage:
    """Billable quantities of **one** model request, as the provider reported them.

    The categories do not overlap: ``input_tokens`` excludes cache reads and writes,
    ``output_tokens`` includes reasoning (``thinking_tokens`` is informational and
    never charged on its own).

    Attributes:
        input_tokens: Uncached input tokens.
        output_tokens: Output tokens including reasoning tokens.
        cache_read_input_tokens: Cache hits.
        cache_write_input_tokens: All cache writes. If larger than the 5-minute plus
            1-hour breakdown, the rest has an unknown lifetime and is only priced when
            both cache-write rates are equal.
        cache_write_5m_input_tokens: Cache writes with a 5-minute lifetime.
        cache_write_1h_input_tokens: Cache writes with a 1-hour lifetime.
        thinking_tokens: Reasoning share of ``output_tokens`` (not charged separately).
        token_details: Token classes billed at special rates although they are part of
            a category, keyed like :attr:`TokenRates.special` (``"audio_input"``).
        units: Non-token quantities (``{"web_search_request": 3}``).
        conditions: Tariff conditions the provider reported (``{"service_tier": "batch"}``).
        missing: Components the usage itself cannot quantify (e.g. code-execution time).
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    cache_write_5m_input_tokens: int = 0
    cache_write_1h_input_tokens: int = 0
    thinking_tokens: int = 0
    token_details: Mapping[str, int] = field(default_factory=dict)
    units: Mapping[str, int] = field(default_factory=dict)
    conditions: Mapping[str, str] = field(default_factory=dict)
    missing: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_write_input_tokens",
            "cache_write_5m_input_tokens",
            "cache_write_1h_input_tokens",
            "thinking_tokens",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise PricingError(f"{name} must be a nonnegative int, got {value!r}")
        object.__setattr__(self, "token_details", _pairs(self.token_details, "token_details", int))
        object.__setattr__(self, "units", _pairs(self.units, "units", int))
        object.__setattr__(self, "conditions", _pairs(self.conditions, "conditions", str))
        object.__setattr__(self, "missing", tuple(self.missing))
        split = self.cache_write_5m_input_tokens + self.cache_write_1h_input_tokens
        if self.cache_write_input_tokens == 0 and split:
            object.__setattr__(self, "cache_write_input_tokens", split)
        elif split > self.cache_write_input_tokens:
            raise PricingError("cache_write_5m + cache_write_1h exceed cache_write_input_tokens")
        if self.thinking_tokens > self.output_tokens:
            raise PricingError("thinking_tokens are part of output_tokens and cannot exceed them")

    @property
    def prompt_tokens(self) -> int:
        """Total input (uncached + cache reads + cache writes): selects long-context tiers."""
        return self.input_tokens + self.cache_read_input_tokens + self.cache_write_input_tokens

    def condition(self, dimension: str) -> str | None:
        return self.conditions.get(dimension)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            name: getattr(self, name)
            for name in (
                "input_tokens",
                "output_tokens",
                "cache_read_input_tokens",
                "cache_write_input_tokens",
                "cache_write_5m_input_tokens",
                "cache_write_1h_input_tokens",
                "thinking_tokens",
            )
        }
        data["token_details"] = dict(self.token_details)
        data["units"] = dict(self.units)
        data["conditions"] = dict(self.conditions)
        data["missing"] = list(self.missing)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MeteredUsage:
        values = dict(data)
        values["missing"] = tuple(values.get("missing", ()))
        return cls(**values)


@dataclass(frozen=True)
class CostComponent:
    """One line of a cost breakdown: ``quantity`` * ``rate`` / ``per`` = ``amount``."""

    name: str
    """``"input"``, ``"output"``, ``"cache_read"``, ``"cache_write_5m"``, ``"cache_write_1h"``,
    ``"cache_write"`` (lifetime unknown, equal rates), a token class (``"audio_input"``)
    or ``"fee:<unit>"``."""
    quantity: int
    rate: Decimal
    """Effective rate after tariff multipliers."""
    per: int
    amount: Decimal
    model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "quantity": self.quantity,
            "rate": str(self.rate),
            "per": self.per,
            "amount": str(self.amount),
            "model": self.model,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CostComponent:
        return cls(
            name=data["name"],
            quantity=int(data["quantity"]),
            rate=to_decimal(data["rate"], "rate"),
            per=int(data["per"]),
            amount=to_decimal(data["amount"], "amount"),
            model=data.get("model"),
        )


@dataclass(frozen=True)
class PriceReference:
    """Which price a cost was calculated with: enough to reproduce and audit it."""

    provider: str
    model: str
    """The model id the usage was reported for."""
    price_model: str
    """The id of the price entry that matched."""
    matched_by: str
    """``"exact"``, ``"alias"``, ``"dated"``, ``"requested"`` (the configured id, after the
    reported id was unknown) or ``"override"`` (an agent's ``pricing=``)."""
    snapshot_version: str | None
    price_version: str | None
    source: str
    quality: str
    checked_at: datetime | None = None
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    currency: str = "USD"
    conditions: tuple[tuple[str, str], ...] = ()
    """The tariff options applied (``(("inference_geo", "us"), ("service_tier", "batch"))``)."""
    long_context_above: int | None = None
    """Threshold of the long-context tier applied, ``None`` for standard context rates."""

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "provider": self.provider,
            "model": self.model,
            "price_model": self.price_model,
            "matched_by": self.matched_by,
            "snapshot_version": self.snapshot_version,
            "price_version": self.price_version,
            "source": self.source,
            "quality": self.quality,
            "currency": self.currency,
            "conditions": dict(self.conditions),
            "long_context_above": self.long_context_above,
        }
        for name in ("checked_at", "valid_from", "valid_until"):
            value = getattr(self, name)
            data[name] = None if value is None else value.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PriceReference:
        values = dict(data)
        values["conditions"] = tuple(sorted(dict(values.get("conditions") or {}).items()))
        for name in ("checked_at", "valid_from", "valid_until"):
            values[name] = _aware(values.get(name), name)
        return cls(**values)


@dataclass(frozen=True)
class CostBreakdown:
    """The itemized cost of one or more model requests.

    ``total`` is the exact sum of all components **only if nothing is missing**;
    otherwise it is ``None`` and ``known_total`` is the sum of the components that
    could be priced (a lower bound, never a complete cost). Unknown is never 0.
    """

    currency: str
    components: tuple[CostComponent, ...]
    missing: tuple[str, ...]
    """Components that could not be priced, e.g. ``"price"`` (no price at all),
    ``"cache_write_ttl"``, ``"tariff:service_tier=priority"``, ``"fee:web_search_call"``,
    ``"tokens:audio_input"`` or ``"currency"`` (mixed currencies)."""
    references: tuple[PriceReference, ...]
    known_total: Decimal

    @property
    def complete(self) -> bool:
        return not self.missing

    @property
    def total(self) -> Decimal | None:
        """The complete cost, or ``None`` if any component is unknown."""
        return self.known_total if not self.missing else None

    @classmethod
    def empty(cls, currency: str = "USD") -> CostBreakdown:
        return cls(currency, (), (), (), _ZERO)

    @classmethod
    def unknown(cls, reason: str = "price", currency: str = "USD") -> CostBreakdown:
        return cls(currency, (), (reason,), (), _ZERO)

    @classmethod
    def combine(cls, parts: Iterable[CostBreakdown]) -> CostBreakdown:
        """Sum breakdowns (several iterations of a request, or several requests).

        Mixed currencies are never converted: the result then reports
        ``"currency"`` as missing and keeps the first currency.
        """
        parts = list(parts)
        if not parts:
            return cls.empty()
        currency = parts[0].currency
        missing: list[str] = []
        known = _ZERO
        for part in parts:
            if part.currency != currency:
                missing.append("currency")
                continue
            known += part.known_total
            missing.extend(part.missing)
        return cls(
            currency,
            tuple(c for part in parts if part.currency == currency for c in part.components),
            tuple(dict.fromkeys(missing)),
            tuple(r for part in parts for r in part.references),
            known,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "currency": self.currency,
            "total": None if self.total is None else str(self.total),
            "known_total": str(self.known_total),
            "missing": list(self.missing),
            "components": [component.to_dict() for component in self.components],
            "references": [reference.to_dict() for reference in self.references],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CostBreakdown:
        return cls(
            currency=data["currency"],
            components=tuple(CostComponent.from_dict(c) for c in data.get("components", ())),
            missing=tuple(data.get("missing", ())),
            references=tuple(PriceReference.from_dict(r) for r in data.get("references", ())),
            known_total=to_decimal(data["known_total"], "known_total"),
        )


def _tier(tiers: tuple[LongContextTier, ...], prompt_tokens: int) -> LongContextTier | None:
    chosen = None
    for tier in tiers:  # sorted by threshold
        if prompt_tokens > tier.above_input_tokens:
            chosen = tier
    return chosen


def calculate_cost(
    price: ModelPrice | None,
    usage: MeteredUsage,
    *,
    reference: PriceReference | None = None,
    model: str | None = None,
) -> CostBreakdown:
    """Price ``usage`` with ``price``.

    Deterministic and side-effect free. Every quantity that cannot be priced is
    listed in :attr:`CostBreakdown.missing`, so an incomplete cost is never shown
    as a complete one. ``price=None`` (unknown price) yields a breakdown with
    ``missing=("price",)``.

    Rules:

    * Tariffs: each reported condition must be an option of the price's tariff for
      that dimension (or a neutral value such as ``service_tier="default"`` for an
      undeclared dimension); otherwise all token components are unknown.
    * Long context: the tier with the highest threshold below
      :attr:`MeteredUsage.prompt_tokens` prices every token of the request.
    * Multipliers of all applied tariff options multiply every token rate; fees are
      not multiplied.
    * Token classes in ``token_details`` are priced at their special rate and taken
      out of their category.
    """
    if not isinstance(usage, MeteredUsage):
        raise TypeError("usage must be a MeteredUsage")
    if price is None:
        return CostBreakdown.unknown("price")
    model = model or (reference.model if reference else price.model)
    missing: list[str] = list(usage.missing)
    components: list[CostComponent] = []
    applied: list[tuple[str, str]] = []
    multiplier = Decimal("1")
    replacement = None
    reported = dict(usage.conditions)
    tariffs_ok = True
    for tariff in price.tariffs:
        value = reported.get(tariff.dimension, tariff.default)
        option = tariff.option(value)
        neutral = NEUTRAL_CONDITIONS.get(tariff.dimension, frozenset())
        if option is None and value in neutral and tariff.default in neutral:
            option = tariff.option(tariff.default)  # e.g. "default" == "standard"
        if option is None:
            missing.append(f"tariff:{tariff.dimension}={value}")
            tariffs_ok = False
            continue
        applied.append((tariff.dimension, value))
        multiplier *= option.multiplier
        if option.rates is not None:
            replacement = option
    for dimension, value in reported.items():
        if price.tariff(dimension) is None and value not in NEUTRAL_CONDITIONS.get(dimension, ()):
            missing.append(f"tariff:{dimension}={value}")
            tariffs_ok = False

    tier_threshold = None
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        if tariffs_ok:
            if replacement is not None:
                rates: TokenRates = replacement.rates  # type: ignore[assignment]
                tiers = replacement.long_context or ()
            else:
                rates, tiers = price.rates, price.long_context
            tier = _tier(tiers, usage.prompt_tokens)
            if tier is not None:
                rates, tier_threshold = tier.rates, tier.above_input_tokens
            components.extend(_token_components(rates, usage, multiplier, model, missing))
        for unit, quantity in usage.units.items():
            if quantity == 0:
                continue
            fee = price.fee(unit)
            if fee is None:
                missing.append(f"fee:{unit}")
                continue
            components.append(
                CostComponent(
                    f"fee:{unit}", quantity, fee.price, fee.per, quantity * fee.price / fee.per, model
                )
            )
        known = sum((c.amount for c in components), _ZERO)
    if reference is not None:
        reference = replace(
            reference,
            conditions=tuple(sorted(applied)),
            long_context_above=tier_threshold,
            currency=price.currency,
        )
    return CostBreakdown(
        price.currency,
        tuple(components),
        tuple(dict.fromkeys(missing)),
        () if reference is None else (reference,),
        known,
    )


def _token_components(
    rates: TokenRates, usage: MeteredUsage, multiplier: Decimal, model: str | None, missing: list[str]
) -> list[CostComponent]:
    buckets = {
        "input": usage.input_tokens,
        "output": usage.output_tokens,
        "cache_read": usage.cache_read_input_tokens,
    }
    lines: list[tuple[str, int, Decimal | None]] = []
    for kind, quantity in usage.token_details.items():
        if quantity == 0:
            continue
        bucket = next(b for b in ("cache_read", "input", "output") if kind.endswith("_" + b))
        quantity = min(quantity, buckets[bucket])
        buckets[bucket] -= quantity
        lines.append((kind, quantity, rates.special.get(kind)))
    lines[:0] = [
        ("input", buckets["input"], rates.input),
        ("output", buckets["output"], rates.output),
        ("cache_read", buckets["cache_read"], rates.cache_read),
        ("cache_write_5m", usage.cache_write_5m_input_tokens, rates.cache_write_5m),
        ("cache_write_1h", usage.cache_write_1h_input_tokens, rates.cache_write_1h),
    ]
    unspecified = (
        usage.cache_write_input_tokens - usage.cache_write_5m_input_tokens - usage.cache_write_1h_input_tokens
    )
    if unspecified:
        if rates.cache_write_5m is not None and rates.cache_write_5m == rates.cache_write_1h:
            lines.append(("cache_write", unspecified, rates.cache_write_5m))
        else:
            missing.append("cache_write_ttl")
    result = []
    for name, quantity, rate in lines:
        if quantity == 0:
            continue
        if rate is None:
            missing.append(
                name if name in ("cache_read", "cache_write_5m", "cache_write_1h") else f"tokens:{name}"
            )
            continue
        effective = rate * multiplier
        result.append(
            CostComponent(
                name,
                quantity,
                effective,
                TOKENS_PER_RATE_UNIT,
                quantity * effective / TOKENS_PER_RATE_UNIT,
                model,
            )
        )
    return result
