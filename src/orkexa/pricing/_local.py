"""Local inference: no external model API billing.

A request to a model that runs on your own hardware (provider ``ollama``) is not billed
by a model API. Orkexa prices it with an explicit zero *API* price whose source is
``"local"``, so budgets and strict pricing treat it as known instead of as an unpriced
cloud request. That is not a claim that local inference costs nothing: hardware,
energy, storage and administration are real costs Orkexa does not measure. Show such
records as "local, no API billing" (``pricing_source == "local"``), never as a total of 0.

A price you register for the model yourself (e.g. to allocate operating costs) takes
precedence over this default.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from ._model import ModelPrice, TokenRates

__all__ = ["LOCAL_INFERENCE_PROVIDERS", "LOCAL_PRICE_SOURCE", "is_local_inference", "local_price"]

LOCAL_INFERENCE_PROVIDERS = frozenset({"ollama"})
"""Providers whose models run locally, without a billed model API."""

LOCAL_PRICE_SOURCE = "local"

_ZERO = Decimal(0)
_SINCE = datetime(2026, 11, 3, tzinfo=UTC)


def is_local_inference(provider: str | None) -> bool:
    return provider in LOCAL_INFERENCE_PROVIDERS


def local_price(provider: str, model: str) -> ModelPrice:
    """The zero API price of a local model (source ``"local"``)."""
    return ModelPrice(
        provider=provider,
        model=model,
        rates=TokenRates(
            input=_ZERO, output=_ZERO, cache_read=_ZERO, cache_write_5m=_ZERO, cache_write_1h=_ZERO
        ),
        source=LOCAL_PRICE_SOURCE,
        version="local",
        checked_at=_SINCE,
        quality="verified",
        notes="No external model API billing; local operating costs (hardware, energy) are not recorded",
    )
