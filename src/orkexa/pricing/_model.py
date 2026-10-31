"""The price data model: rates, long-context tiers, tariffs, fees and model prices.

Every amount is a :class:`~decimal.Decimal`. Token rates are prices per
:data:`TOKENS_PER_RATE_UNIT` (one million) tokens, fee rates are prices per ``per``
units, both in the price's ``currency`` (an ISO 4217 code, ``"USD"`` by default).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, fields, replace
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import re
from typing import Any, Literal, cast, get_args

from ..errors import PricingError

__all__ = [
    "NEUTRAL_CONDITIONS",
    "TOKENS_PER_RATE_UNIT",
    "FeeRate",
    "LongContextTier",
    "ModelPrice",
    "PriceQuality",
    "Tariff",
    "TariffOption",
    "TokenRates",
    "to_decimal",
]

TOKENS_PER_RATE_UNIT = 1_000_000
"""Token rates are prices per this many tokens (``USD / MTok`` like the providers' price lists)."""

PriceQuality = Literal["verified", "contract", "unverified", "estimated"]
"""How trustworthy a price is.

* ``"verified"``: checked against the provider's official price list (see ``checked_at``).
* ``"contract"``: taken from an individual agreement; authoritative for that account.
* ``"unverified"``: entered without verification (the default for prices you register).
* ``"estimated"``: an approximation, e.g. derived from another model's price.
"""

_QUALITIES: tuple[str, ...] = get_args(PriceQuality)

#: Condition values that need no tariff entry: a price without a ``service_tier``
#: tariff still prices usage that reports ``service_tier="default"``. Any other
#: reported value of an undeclared dimension makes the cost unknown.
NEUTRAL_CONDITIONS: Mapping[str, frozenset[str]] = {
    "service_tier": frozenset({"standard", "default"}),
    "speed": frozenset({"standard"}),
    "inference_geo": frozenset({"global"}),
}

_TOKEN_KIND_BUCKETS = ("_cache_read", "_input", "_output")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/@+-]*")
_CURRENCY = re.compile(r"[A-Z]{3}")


def to_decimal(value: Any, name: str = "value") -> Decimal:
    """Convert ``value`` (``Decimal``, ``int`` or a decimal string) to a finite ``Decimal``.

    Floats are rejected because they cannot represent most prices exactly
    (``0.1 + 0.2 != 0.3``); pass ``"0.1"`` instead.
    """
    if isinstance(value, (bool, float)):
        raise PricingError(f"{name} must be a Decimal, int or decimal string, not {type(value).__name__}")
    if isinstance(value, Decimal):
        result = value
    elif isinstance(value, (int, str)):
        try:
            result = Decimal(value)
        except InvalidOperation:
            raise PricingError(f"{name} is not a decimal number: {value!r}") from None
    else:
        raise PricingError(f"{name} must be a Decimal, int or decimal string, not {type(value).__name__}")
    if not result.is_finite():
        raise PricingError(f"{name} must be finite, not {value!r}")
    return result


def _rate(value: Any, name: str, *, optional: bool = False) -> Decimal | None:
    if value is None:
        if optional:
            return None
        raise PricingError(f"{name} is required")
    result = to_decimal(value, name)
    if result < 0:
        raise PricingError(f"{name} must not be negative, got {result}")
    return result


def _check_name(value: Any, what: str) -> str:
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise PricingError(f"Invalid {what}: {value!r} (letters, digits and _.:/@+- only)")
    return value


def _aware(value: Any, name: str) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            raise PricingError(f"{name} is not an ISO 8601 timestamp: {value!r}") from None
    if not isinstance(value, datetime):
        raise PricingError(f"{name} must be a datetime, not {type(value).__name__}")
    if value.tzinfo is None or value.utcoffset() is None:
        raise PricingError(f"{name} must be timezone-aware (e.g. datetime(..., tzinfo=timezone.utc))")
    return value.astimezone(timezone.utc)


def _set(obj: object, name: str, value: Any) -> None:
    object.__setattr__(obj, name, value)


@dataclass(frozen=True)
class TokenRates:
    """Prices per million tokens for the token categories a provider reports.

    Attributes:
        input: Uncached input tokens.
        output: Output tokens, **including** reasoning/thinking tokens (all supported
            providers bill reasoning as output; it is never charged twice).
        cache_read: Cache hits (cached input). ``None``: unknown, so usage with
            cache reads has an unknown cost.
        cache_write_5m: Cache writes with a 5-minute lifetime (or the only cache-write
            rate of providers without lifetimes). ``None``: unknown.
        cache_write_1h: Cache writes with a 1-hour lifetime. ``None``: unknown.
        special: Rates of token classes that providers bill differently although they
            are part of a reported category, keyed ``"<class>_input"``,
            ``"<class>_output"`` or ``"<class>_cache_read"`` (e.g. ``"audio_input"``).
            Usage that reports such a class without a rate has an unknown cost.
    """

    input: Decimal
    output: Decimal
    cache_read: Decimal | None = None
    cache_write_5m: Decimal | None = None
    cache_write_1h: Decimal | None = None
    special: Mapping[str, Decimal] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _set(self, "input", _rate(self.input, "input"))
        _set(self, "output", _rate(self.output, "output"))
        for name in ("cache_read", "cache_write_5m", "cache_write_1h"):
            _set(self, name, _rate(getattr(self, name), name, optional=True))
        special: dict[str, Decimal] = {}
        for kind, value in dict(self.special).items():
            _check_name(kind, "token class")
            if not kind.endswith(_TOKEN_KIND_BUCKETS):
                raise PricingError(f"Token class {kind!r} must end in '_input', '_output' or '_cache_read'")
            special[kind] = cast(Decimal, _rate(value, f"special[{kind!r}]"))
        _set(self, "special", _FrozenDict(sorted(special.items())))

    @classmethod
    def of(
        cls,
        input: str | int | Decimal,
        output: str | int | Decimal,
        cache_read: str | int | Decimal | None = None,
        cache_write_5m: str | int | Decimal | None = None,
        cache_write_1h: str | int | Decimal | None = None,
        **special: str | int | Decimal,
    ) -> TokenRates:
        """Build rates from decimal strings, e.g. ``TokenRates.of("4", "20", "0.20", "5", "8")``."""
        return cls(
            to_decimal(input, "input"),
            to_decimal(output, "output"),
            None if cache_read is None else to_decimal(cache_read, "cache_read"),
            None if cache_write_5m is None else to_decimal(cache_write_5m, "cache_write_5m"),
            None if cache_write_1h is None else to_decimal(cache_write_1h, "cache_write_1h"),
            {kind: to_decimal(value, kind) for kind, value in special.items()},
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"input": str(self.input), "output": str(self.output)}
        for name in ("cache_read", "cache_write_5m", "cache_write_1h"):
            value = getattr(self, name)
            if value is not None:
                data[name] = str(value)
        if self.special:
            data["special"] = {kind: str(value) for kind, value in self.special.items()}
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TokenRates:
        _only(data, {"input", "output", "cache_read", "cache_write_5m", "cache_write_1h", "special"}, "rates")
        return cls(
            input=data.get("input"),  # type: ignore[arg-type]
            output=data.get("output"),  # type: ignore[arg-type]
            cache_read=data.get("cache_read"),
            cache_write_5m=data.get("cache_write_5m"),
            cache_write_1h=data.get("cache_write_1h"),
            special=dict(data.get("special") or {}),
        )


class _FrozenDict(dict):  # type: ignore[type-arg]
    """A read-only, hashable dict (keeps dataclass equality and hashing working)."""

    def _readonly(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("price data is immutable")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = _readonly  # type: ignore[assignment]

    def __hash__(self) -> int:  # type: ignore[override]
        return hash(tuple(self.items()))

    def __copy__(self) -> _FrozenDict:
        return self

    def __deepcopy__(self, memo: Any) -> _FrozenDict:
        return self


@dataclass(frozen=True)
class LongContextTier:
    """Rates for requests whose prompt is longer than ``above_input_tokens``.

    The prompt length is the request's total input: uncached input plus cache reads
    plus cache writes (the definition Anthropic and Google use). The tier with the
    highest threshold below the prompt length prices the **whole** request.
    """

    above_input_tokens: int
    rates: TokenRates

    def __post_init__(self) -> None:
        if not isinstance(self.above_input_tokens, int) or self.above_input_tokens < 0:
            raise PricingError("above_input_tokens must be a nonnegative int")
        if not isinstance(self.rates, TokenRates):
            raise PricingError("LongContextTier.rates must be TokenRates")

    def to_dict(self) -> dict[str, Any]:
        return {"above_input_tokens": self.above_input_tokens, "rates": self.rates.to_dict()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LongContextTier:
        _only(data, {"above_input_tokens", "rates"}, "long_context tier")
        return cls(data.get("above_input_tokens"), TokenRates.from_dict(_mapping(data.get("rates"), "rates")))  # type: ignore[arg-type]


def _tiers(value: Any, what: str) -> tuple[LongContextTier, ...]:
    tiers = tuple(value)
    for tier in tiers:
        if not isinstance(tier, LongContextTier):
            raise PricingError(f"{what} must contain LongContextTier objects")
    thresholds = [tier.above_input_tokens for tier in tiers]
    if len(set(thresholds)) != len(thresholds):
        raise PricingError(f"{what} has duplicate thresholds")
    return tuple(sorted(tiers, key=lambda tier: tier.above_input_tokens))


@dataclass(frozen=True)
class TariffOption:
    """One value of a :class:`Tariff` dimension and how it changes the token rates.

    Attributes:
        value: The value as the provider reports it (``"batch"``, ``"fast"``, ``"us"``).
        multiplier: Factor applied to every token rate (``"0.5"`` for a 50 % batch
            discount, ``"1.1"`` for a 10 % surcharge). Fees are not multiplied.
        rates: Replacement token rates for this option (applied before ``multiplier``),
            for tariffs that are not a plain factor.
        long_context: Long-context tiers of the replacement ``rates``. Required (``()``
            for none) when ``rates`` is set on a price that has long-context tiers, so
            Orkexa never guesses how the two combine.
    """

    value: str
    multiplier: Decimal = Decimal("1")
    rates: TokenRates | None = None
    long_context: tuple[LongContextTier, ...] | None = None

    def __post_init__(self) -> None:
        _check_name(self.value, "tariff value")
        multiplier = to_decimal(self.multiplier, "multiplier")
        if multiplier < 0:
            raise PricingError("multiplier must not be negative")
        _set(self, "multiplier", multiplier)
        if self.rates is not None and not isinstance(self.rates, TokenRates):
            raise PricingError("TariffOption.rates must be TokenRates")
        if self.long_context is not None:
            if self.rates is None:
                raise PricingError("TariffOption.long_context needs replacement rates")
            _set(self, "long_context", _tiers(self.long_context, "TariffOption.long_context"))

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"value": self.value, "multiplier": str(self.multiplier)}
        if self.rates is not None:
            data["rates"] = self.rates.to_dict()
        if self.long_context is not None:
            data["long_context"] = [tier.to_dict() for tier in self.long_context]
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TariffOption:
        _only(data, {"value", "multiplier", "rates", "long_context"}, "tariff option")
        rates = data.get("rates")
        long_context = data.get("long_context")
        return cls(
            value=data.get("value"),  # type: ignore[arg-type]
            multiplier=data.get("multiplier", "1"),
            rates=None if rates is None else TokenRates.from_dict(_mapping(rates, "rates")),
            long_context=None
            if long_context is None
            else tuple(
                LongContextTier.from_dict(_mapping(t, "tier")) for t in _list(long_context, "long_context")
            ),
        )


@dataclass(frozen=True)
class Tariff:
    """A pricing dimension such as ``service_tier``, ``speed`` or ``inference_geo``.

    Usage that reports a value not listed in ``options`` has an unknown cost: Orkexa
    never falls back to a standard rate for unknown tariff conditions. Usage that
    reports no value for the dimension is priced with ``default``.
    """

    dimension: str
    default: str
    options: tuple[TariffOption, ...]

    def __post_init__(self) -> None:
        _check_name(self.dimension, "tariff dimension")
        options = tuple(self.options)
        values = [option.value for option in options if isinstance(option, TariffOption)]
        if len(values) != len(options):
            raise PricingError("Tariff.options must contain TariffOption objects")
        if len(set(values)) != len(values):
            raise PricingError(f"Tariff {self.dimension!r} lists a value twice")
        if self.default not in values:
            raise PricingError(f"Tariff {self.dimension!r}: default {self.default!r} is not an option")
        _set(self, "options", options)

    def option(self, value: str) -> TariffOption | None:
        return next((option for option in self.options if option.value == value), None)

    @classmethod
    def of(cls, dimension: str, default: str = "standard", /, **multipliers: str | int | Decimal) -> Tariff:
        """A tariff of plain multipliers: ``Tariff.of("service_tier", standard="1", batch="0.5")``."""
        if default not in multipliers:
            multipliers = {default: "1", **multipliers}
        return cls(
            dimension,
            default,
            tuple(TariffOption(value, to_decimal(factor, value)) for value, factor in multipliers.items()),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "default": self.default,
            "options": [option.to_dict() for option in self.options],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Tariff:
        _only(data, {"dimension", "default", "options"}, "tariff")
        return cls(
            data.get("dimension"),  # type: ignore[arg-type]
            data.get("default"),  # type: ignore[arg-type]
            tuple(
                TariffOption.from_dict(_mapping(o, "option")) for o in _list(data.get("options"), "options")
            ),
        )


@dataclass(frozen=True)
class FeeRate:
    """A non-token charge: ``price`` per ``per`` units of ``unit``.

    Units are what the usage reports, e.g. ``"web_search_request"`` (Anthropic),
    ``"web_search_call"`` (OpenAI) or ``"google_search_query"`` (Gemini). A price of
    ``0`` declares a unit as free; a reported unit without a fee rate makes the cost
    unknown.
    """

    unit: str
    price: Decimal
    per: int = 1

    def __post_init__(self) -> None:
        _check_name(self.unit, "fee unit")
        _set(self, "price", _rate(self.price, f"fee {self.unit!r}"))
        if not isinstance(self.per, int) or isinstance(self.per, bool) or self.per < 1:
            raise PricingError("FeeRate.per must be a positive int")

    def to_dict(self) -> dict[str, Any]:
        return {"unit": self.unit, "price": str(self.price), "per": self.per}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FeeRate:
        _only(data, {"unit", "price", "per"}, "fee")
        return cls(data.get("unit"), data.get("price"), data.get("per", 1))  # type: ignore[arg-type]


@dataclass(frozen=True)
class ModelPrice:
    """The complete, immutable price of one model of one provider.

    Attributes:
        provider: Registered provider name (``"anthropic"``, ``"openai"``, ``"gemini"``,
            a compatible endpoint's name). Prices never apply across providers.
        model: The model id the price is for.
        rates: Standard token rates (per million tokens).
        currency: ISO 4217 currency code of every amount (default ``"USD"``).
        aliases: Further ids that mean exactly this model (no pattern matching).
        match_dated_versions: Also resolve dated ids of ``model``: ``<model>-YYYYMMDD``,
            ``<model>-YYYY-MM-DD`` and ``<model>@YYYYMMDD``. Off by default.
        long_context: Rates for long prompts, see :class:`LongContextTier`.
        tariffs: Pricing dimensions (service tier, speed, data residency, region, ...).
        fees: Non-token charges (server tools etc.).
        source: Where the price comes from (``"builtin"``, ``"registered"``, a catalog
            or contract name).
        source_url: Link to the price list or agreement.
        version: Version label of this price entry (e.g. the price list's date).
        valid_from: Start of validity (inclusive, timezone-aware), ``None``: always.
        valid_until: End of validity (exclusive, timezone-aware), ``None``: open-ended.
        checked_at: When the price was last checked against its source.
        quality: :data:`PriceQuality` of the price.
        notes: Free text.
    """

    provider: str
    model: str
    rates: TokenRates
    currency: str = "USD"
    aliases: tuple[str, ...] = ()
    match_dated_versions: bool = False
    long_context: tuple[LongContextTier, ...] = ()
    tariffs: tuple[Tariff, ...] = ()
    fees: tuple[FeeRate, ...] = ()
    source: str = "custom"
    source_url: str | None = None
    version: str | None = None
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    checked_at: datetime | None = None
    quality: PriceQuality = "unverified"
    notes: str | None = None

    def __post_init__(self) -> None:
        _check_name(self.provider, "provider")
        _check_name(self.model, "model id")
        if not isinstance(self.rates, TokenRates):
            raise PricingError("ModelPrice.rates must be TokenRates (use TokenRates.of(...))")
        if not isinstance(self.currency, str) or not _CURRENCY.fullmatch(self.currency):
            raise PricingError(f"currency must be an ISO 4217 code such as 'USD', not {self.currency!r}")
        aliases = tuple(_check_name(alias, "alias") for alias in self.aliases)
        if self.model in aliases or len(set(aliases)) != len(aliases):
            raise PricingError(f"{self.provider}:{self.model}: aliases must be unique and differ from the id")
        _set(self, "aliases", aliases)
        _set(self, "match_dated_versions", bool(self.match_dated_versions))
        _set(self, "long_context", _tiers(self.long_context, "long_context"))
        tariffs = tuple(self.tariffs)
        if not all(isinstance(t, Tariff) for t in tariffs):
            raise PricingError("ModelPrice.tariffs must contain Tariff objects")
        dimensions = [tariff.dimension for tariff in tariffs]
        if len(set(dimensions)) != len(dimensions):
            raise PricingError(f"{self.provider}:{self.model}: tariff dimension listed twice")
        replacing = [t.dimension for t in tariffs if any(o.rates is not None for o in t.options)]
        if len(replacing) > 1:
            raise PricingError(
                f"{self.provider}:{self.model}: only one tariff dimension may replace rates ({replacing})"
            )
        if self.long_context:
            for tariff in tariffs:
                for option in tariff.options:
                    if option.rates is not None and option.long_context is None:
                        raise PricingError(
                            f"{self.provider}:{self.model}: tariff {tariff.dimension}={option.value} "
                            "replaces the rates of a price with long-context tiers; set its "
                            "long_context (() for none)"
                        )
        _set(self, "tariffs", tariffs)
        fees = tuple(self.fees)
        if not all(isinstance(fee, FeeRate) for fee in fees):
            raise PricingError("ModelPrice.fees must contain FeeRate objects")
        units = [fee.unit for fee in fees]
        if len(set(units)) != len(units):
            raise PricingError(f"{self.provider}:{self.model}: fee unit listed twice")
        _set(self, "fees", fees)
        if not isinstance(self.source, str) or not self.source:
            raise PricingError("source must be a nonempty string")
        for name in ("valid_from", "valid_until", "checked_at"):
            _set(self, name, _aware(getattr(self, name), name))
        if self.valid_from and self.valid_until and self.valid_until <= self.valid_from:
            raise PricingError(f"{self.provider}:{self.model}: valid_until must be after valid_from")
        if self.quality not in _QUALITIES:
            raise PricingError(f"quality must be one of {_QUALITIES}, not {self.quality!r}")

    # ---- lookups -------------------------------------------------------------------------

    @property
    def names(self) -> tuple[str, ...]:
        """The model id followed by its aliases."""
        return (self.model, *self.aliases)

    def tariff(self, dimension: str) -> Tariff | None:
        return next((tariff for tariff in self.tariffs if tariff.dimension == dimension), None)

    def fee(self, unit: str) -> FeeRate | None:
        return next((fee for fee in self.fees if fee.unit == unit), None)

    def valid_at(self, when: datetime) -> bool:
        """Whether ``when`` (timezone-aware) lies in the validity period."""
        if self.valid_from is not None and when < self.valid_from:
            return False
        return not (self.valid_until is not None and when >= self.valid_until)

    def overlaps(self, other: ModelPrice) -> bool:
        """Whether the validity periods of two prices overlap."""
        start = max(self.valid_from or _MIN, other.valid_from or _MIN)
        end = min(self.valid_until or _MAX, other.valid_until or _MAX)
        return start < end

    def with_changes(self, **changes: Any) -> ModelPrice:
        """A copy with some fields replaced (prices are immutable)."""
        return replace(self, **changes)

    # ---- serialization -------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """A JSON-compatible dict (decimals as strings, timestamps as ISO 8601)."""
        data: dict[str, Any] = {
            "provider": self.provider,
            "model": self.model,
            "currency": self.currency,
            "rates": self.rates.to_dict(),
        }
        if self.aliases:
            data["aliases"] = list(self.aliases)
        if self.match_dated_versions:
            data["match_dated_versions"] = True
        if self.long_context:
            data["long_context"] = [tier.to_dict() for tier in self.long_context]
        if self.tariffs:
            data["tariffs"] = [tariff.to_dict() for tariff in self.tariffs]
        if self.fees:
            data["fees"] = [fee.to_dict() for fee in self.fees]
        data["source"] = self.source
        for name in ("source_url", "version", "notes"):
            value = getattr(self, name)
            if value is not None:
                data[name] = value
        for name in ("valid_from", "valid_until", "checked_at"):
            value = getattr(self, name)
            if value is not None:
                data[name] = value.isoformat()
        data["quality"] = self.quality
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModelPrice:
        """Inverse of :meth:`to_dict`; unknown keys are rejected."""
        known = {f.name for f in fields(cls)}
        _only(data, known, "model price")
        values: dict[str, Any] = dict(data)
        values["rates"] = TokenRates.from_dict(_mapping(data.get("rates"), "rates"))
        values["aliases"] = tuple(_list(data.get("aliases", []), "aliases"))
        values["long_context"] = tuple(
            LongContextTier.from_dict(_mapping(t, "tier"))
            for t in _list(data.get("long_context", []), "long_context")
        )
        values["tariffs"] = tuple(
            Tariff.from_dict(_mapping(t, "tariff")) for t in _list(data.get("tariffs", []), "tariffs")
        )
        values["fees"] = tuple(
            FeeRate.from_dict(_mapping(f, "fee")) for f in _list(data.get("fees", []), "fees")
        )
        for required in ("provider", "model"):
            if required not in values:
                raise PricingError(f"model price is missing {required!r}")
        return cls(**values)


_MIN = datetime.min.replace(tzinfo=timezone.utc)
_MAX = datetime.max.replace(tzinfo=timezone.utc)


def _only(data: Mapping[str, Any], allowed: Iterable[str], what: str) -> None:
    if not isinstance(data, Mapping):
        raise PricingError(f"{what} must be an object")
    unknown = set(data) - set(allowed)
    if unknown:
        raise PricingError(f"Unknown keys in {what}: {sorted(unknown)}")


def _mapping(value: Any, what: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PricingError(f"{what} must be an object")
    return value


def _list(value: Any, what: str) -> list[Any]:
    if not isinstance(value, (list, tuple)):
        raise PricingError(f"{what} must be a list")
    return list(value)
