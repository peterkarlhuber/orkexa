"""The original pricing API (``ModelPricing``, ``MODEL_PRICING``, ``register_pricing``,
``resolve_pricing``) and the process-wide default resolver built on top of it.

The legacy dictionaries stay the source of truth for the default resolver: entries
added, replaced or deleted in :data:`MODEL_PRICING` / :data:`PROVIDER_PRICING` at
runtime are picked up by the next lookup, exactly as before.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import hashlib
import re
import threading
from typing import Any

from ._builtin import BUILTIN_SNAPSHOT, BUILTIN_VERSION
from ._model import FeeRate, ModelPrice, Tariff, TokenRates
from ._resolver import PriceResolver
from ._snapshot import PriceSnapshot

__all__ = [
    "MODEL_PRICING",
    "PROVIDER_PRICING",
    "ModelPricing",
    "default_resolver",
    "register_pricing",
    "resolve_pricing",
]


@dataclass(frozen=True)
class ModelPricing:
    """USD per million tokens (standard, synchronous, first-party API rates).

    The original, flat price type. It is still accepted everywhere; for aliases,
    long-context tiers, tariffs, fees, validity and provenance use
    :class:`~orkexa.pricing.ModelPrice` (see :meth:`to_model_price`).
    """

    input: Decimal
    output: Decimal
    cache_write_5m: Decimal
    cache_write_1h: Decimal
    cache_read: Decimal

    @classmethod
    def of(
        cls,
        input: str,
        output: str,
        cache_write_5m: str,
        cache_write_1h: str,
        cache_read: str,
    ) -> ModelPricing:
        """Build prices from decimal strings, e.g. ``ModelPricing.of("4", "20", ...)``."""
        return cls(
            Decimal(input),
            Decimal(output),
            Decimal(cache_write_5m),
            Decimal(cache_write_1h),
            Decimal(cache_read),
        )

    def to_model_price(
        self, provider: str, model: str, *, source: str = "registered", **changes: Any
    ) -> ModelPrice:
        """The equivalent :class:`~orkexa.pricing.ModelPrice` (how Orkexa prices legacy entries).

        It keeps the legacy semantics: the batch tier costs 50 %, other service
        tiers, speeds and data-residency options are unknown; for ``"anthropic"``
        web search costs $10 per 1,000 searches and dated ids resolve.
        """
        values: dict[str, Any] = {
            "provider": provider,
            "model": model,
            "rates": TokenRates(
                self.input, self.output, self.cache_read, self.cache_write_5m, self.cache_write_1h
            ),
            "match_dated_versions": provider == "anthropic",
            "tariffs": (
                Tariff.of("service_tier", "standard", default="1", batch="0.5"),
                Tariff.of("speed", "standard"),
                Tariff.of("inference_geo", "global"),
            ),
            "fees": _ANTHROPIC_FEES if provider == "anthropic" else (),
            "source": source,
        }
        values.update(changes)
        return ModelPrice(**values)


_ANTHROPIC_FEES = (
    FeeRate("web_search_request", Decimal("10"), per=1000),
    FeeRate("web_fetch_request", Decimal("0")),
)


def _legacy_view(price: ModelPrice) -> ModelPricing:
    rates = price.rates

    def value(rate: Decimal | None) -> Decimal:
        return rate if rate is not None else Decimal("0")

    return ModelPricing(
        rates.input,
        rates.output,
        value(rates.cache_write_5m),
        value(rates.cache_write_1h),
        value(rates.cache_read),
    )


# Every legacy entry that has a richer ModelPrice behind it, keyed by provider and model.
# The rich price is used only while the dictionary still holds the very same object.
_RICH: dict[tuple[str, str], tuple[ModelPricing, ModelPrice]] = {}
for _price in BUILTIN_SNAPSHOT.prices:
    _RICH[(_price.provider, _price.model)] = (_legacy_view(_price), _price)

MODEL_PRICING: dict[str, ModelPricing] = {
    model: legacy for (provider, model), (legacy, _) in _RICH.items() if provider == "anthropic"
}
"""Anthropic prices (USD per million tokens), keyed by model id; editable at runtime."""

# Claude Haiku 5.5 is priced by prompt length: prompts over 100k input tokens
# (including cache reads and writes) pay these higher rates.
_HAIKU_55_LONG_CONTEXT = ModelPricing.of("0.50", "2.50", "0.625", "1", "0.05")
_HAIKU_55_LONG_CONTEXT_THRESHOLD = 100_000


PROVIDER_PRICING: dict[tuple[str, str], ModelPricing] = {}
"""Prices registered for other providers, keyed ``(provider, model)``."""

_lock = threading.Lock()
_generation = 0


def register_pricing(provider: str, model: str, pricing: ModelPricing | ModelPrice) -> None:
    """Register prices of ``provider``'s ``model`` in the process-wide default resolver.

    ``pricing`` is a legacy :class:`ModelPricing` or a full
    :class:`~orkexa.pricing.ModelPrice` (whose ``provider`` and ``model`` must match).
    For OpenAI and Gemini, a :class:`ModelPricing`'s ``cache_write_5m``/``cache_write_1h``
    price cache writes (use the same value if the provider has one rate) and
    ``cache_read`` cached input.

    Applications that need isolated catalogs use their own
    :class:`~orkexa.pricing.PriceResolver` instead.
    """
    global _generation
    if isinstance(pricing, ModelPrice):
        if (pricing.provider, pricing.model) != (provider, model):
            raise ValueError(f"ModelPrice is for {pricing.provider}:{pricing.model}, not {provider}:{model}")
        legacy = _legacy_view(pricing)
        with _lock:
            _RICH[(provider, model)] = (legacy, pricing)
            _generation += 1
    else:
        legacy = pricing
        with _lock:
            _generation += 1
    if provider == "anthropic":
        MODEL_PRICING[model] = legacy
    else:
        PROVIDER_PRICING[(provider, model)] = legacy


def resolve_pricing(
    model: str, total_input_tokens: int = 0, provider: str | None = None
) -> ModelPricing | None:
    """Return the prices of ``model``, or ``None`` if the model is unknown.

    Without ``provider`` (or with ``"anthropic"``) the Anthropic table is used: a dated
    model ID (``<id>-YYYYMMDD``) resolves to its catalog entry and
    ``total_input_tokens`` selects long-context prices where they exist. Entries added
    to :data:`MODEL_PRICING` at runtime are taken into account. Other providers only
    have the prices registered with :func:`register_pricing`.

    This is the legacy lookup; :class:`~orkexa.pricing.PriceResolver` returns full
    :class:`~orkexa.pricing.ModelPrice` objects with provenance.
    """
    if provider is not None and provider != "anthropic":
        return PROVIDER_PRICING.get((provider, model))
    # Longer keys first, so "claude-opus-5-5" is never resolved as "claude-opus-5".
    for key in sorted(MODEL_PRICING, key=len, reverse=True):
        if model == key or re.fullmatch(re.escape(key) + r"-\d{8}", model):
            if key == "claude-haiku-5-5" and total_input_tokens > _HAIKU_55_LONG_CONTEXT_THRESHOLD:
                return _HAIKU_55_LONG_CONTEXT
            return MODEL_PRICING[key]
    return None


class _DefaultResolver(PriceResolver):
    """The process-wide resolver: the built-in catalog plus the legacy dictionaries.

    Its snapshot is rebuilt whenever :data:`MODEL_PRICING` or
    :data:`PROVIDER_PRICING` change; an unchanged state always maps to the same
    snapshot (and version), so concurrent requests share one basis.
    """

    def __init__(self) -> None:
        super().__init__(BUILTIN_SNAPSHOT)
        self._state: tuple[Any, ...] | None = None
        self._sync_lock = threading.Lock()

    @property
    def snapshot(self) -> PriceSnapshot:
        state = (_generation, tuple(MODEL_PRICING.items()), tuple(PROVIDER_PRICING.items()))
        with self._sync_lock:
            if state != self._state:
                self._snapshot = _build_default_snapshot(state[1], state[2])
                self._remember(self._snapshot)
                self._state = state
            return self._snapshot

    def publish(self, snapshot: PriceSnapshot) -> None:
        raise TypeError(
            "The default resolver mirrors MODEL_PRICING/register_pricing(); create a "
            "PriceResolver(snapshot) for your own catalog and pass it as pricing= or "
            "UsageTracker(price_resolver=...)"
        )

    def register(self, *prices: ModelPrice, version: str | None = None) -> PriceSnapshot:
        if version is not None:
            raise TypeError("The default resolver derives its versions from its content")
        for price in prices:
            register_pricing(price.provider, price.model, price)
        return self.snapshot


def _build_default_snapshot(
    anthropic: tuple[tuple[str, ModelPricing], ...], others: tuple[tuple[tuple[str, str], ModelPricing], ...]
) -> PriceSnapshot:
    prices: list[ModelPrice] = []
    entries = [(("anthropic", model), legacy) for model, legacy in anthropic] + list(others)
    pristine = True
    for (provider, model), legacy in entries:
        rich = _RICH.get((provider, model))
        if rich is not None and rich[0] is legacy:
            prices.append(rich[1])
            pristine = pristine and rich[1].source == "builtin"
        else:
            prices.append(legacy.to_model_price(provider, model))
            pristine = False
    builtin_count = sum(1 for price in BUILTIN_SNAPSHOT.prices)
    if pristine and len(prices) == builtin_count:
        return BUILTIN_SNAPSHOT
    draft = PriceSnapshot("draft", tuple(prices))
    digest = hashlib.sha256(draft.content_hash().encode()).hexdigest()[:12]
    return PriceSnapshot(
        f"{BUILTIN_VERSION}+local.{digest}",
        draft.prices,
        source="orkexa",
        description="Built-in catalog with prices registered in this process",
    )


_DEFAULT = _DefaultResolver()


def default_resolver() -> PriceResolver:
    """The process-wide resolver used when an agent and its tracker set no resolver.

    It contains the built-in Anthropic prices and everything registered with
    :func:`register_pricing`.
    """
    return _DEFAULT
