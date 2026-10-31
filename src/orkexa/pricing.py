"""Model prices used to estimate the USD cost of API requests.

:data:`MODEL_PRICING` holds Anthropic first-party prices (USD per million tokens).
Orkexa ships **no** prices for other providers: prices change often and differ by
account, region and tier. Register them with :func:`register_pricing` or pass
``pricing=`` to an agent; unpriced requests get ``estimated_cost_usd=None``.

Anthropic prices:
Keep this table up to date with https://platform.claude.com/docs/en/about-claude/pricing
(last checked: 2026-10-27).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import re

__all__ = ["MODEL_PRICING", "PROVIDER_PRICING", "ModelPricing", "register_pricing", "resolve_pricing"]


@dataclass(frozen=True)
class ModelPricing:
    """USD per million tokens (standard, synchronous, first-party API rates)."""

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


MODEL_PRICING: dict[str, ModelPricing] = {
    "claude-fable-5-1": ModelPricing.of("10", "50", "12.50", "20", "0.25"),
    "claude-fable-5": ModelPricing.of("10", "50", "12.50", "20", "1"),
    "claude-opus-5-5": ModelPricing.of("4", "20", "5", "8", "0.20"),
    "claude-opus-5": ModelPricing.of("5", "25", "6.25", "10", "0.50"),
    "claude-opus-4-8": ModelPricing.of("5", "25", "6.25", "10", "0.50"),
    "claude-opus-4-7": ModelPricing.of("5", "25", "6.25", "10", "0.50"),
    "claude-opus-4-6": ModelPricing.of("5", "25", "6.25", "10", "0.50"),
    "claude-sonnet-5-5": ModelPricing.of("2", "10", "2.50", "4", "0.20"),
    "claude-sonnet-5": ModelPricing.of("2", "10", "2.50", "4", "0.20"),
    "claude-sonnet-4-6": ModelPricing.of("3", "15", "3.75", "6", "0.30"),
    "claude-haiku-5-5": ModelPricing.of("0.10", "0.50", "0.125", "0.20", "0.01"),
    "claude-haiku-4-5": ModelPricing.of("1", "5", "1.25", "2", "0.10"),
}

# Claude Haiku 5.5 is priced by prompt length: prompts over 100k input tokens
# (including cache reads and writes) pay these higher rates.
_HAIKU_55_LONG_CONTEXT = ModelPricing.of("0.50", "2.50", "0.625", "1", "0.05")
_HAIKU_55_LONG_CONTEXT_THRESHOLD = 100_000


PROVIDER_PRICING: dict[tuple[str, str], ModelPricing] = {}
"""Prices registered for other providers, keyed ``(provider, model)``."""


def register_pricing(provider: str, model: str, pricing: ModelPricing) -> None:
    """Register prices of ``provider``'s ``model`` (e.g. from your contract or price list).

    For OpenAI and Gemini, ``cache_write_5m``/``cache_write_1h`` price cache writes
    (use the same value if the provider has one rate) and ``cache_read`` cached input.
    """
    if provider == "anthropic":
        MODEL_PRICING[model] = pricing
    else:
        PROVIDER_PRICING[(provider, model)] = pricing


def resolve_pricing(
    model: str, total_input_tokens: int = 0, provider: str | None = None
) -> ModelPricing | None:
    """Return the prices of ``model``, or ``None`` if the model is unknown.

    Without ``provider`` (or with ``"anthropic"``) the Anthropic table is used: a dated
    model ID (``<id>-YYYYMMDD``) resolves to its catalog entry and
    ``total_input_tokens`` selects long-context prices where they exist. Entries added
    to :data:`MODEL_PRICING` at runtime are taken into account. Other providers only
    have the prices registered with :func:`register_pricing`.
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
