"""Orkexa's built-in price catalog: Anthropic first-party Claude API prices.

Source: https://platform.claude.com/docs/en/about-claude/pricing (checked 2026-10-31).
Orkexa ships **no** prices for other providers: they change often and differ by
account, region and tier. Register them, or pass your own catalog.
"""

from __future__ import annotations

from datetime import datetime, timezone

from ._model import FeeRate, LongContextTier, ModelPrice, Tariff, TokenRates
from ._snapshot import PriceSnapshot

__all__ = ["BUILTIN_CHECKED_AT", "BUILTIN_SNAPSHOT", "BUILTIN_SOURCE_URL", "BUILTIN_VERSION"]

BUILTIN_VERSION = "orkexa-builtin-2026-10-31"
BUILTIN_SOURCE_URL = "https://platform.claude.com/docs/en/about-claude/pricing"
BUILTIN_CHECKED_AT = datetime(2026, 10, 31, tzinfo=timezone.utc)

# (model id, input, output, 5m cache write, 1h cache write, cache read, fast mode, data residency)
_ANTHROPIC: tuple[tuple[str, str, str, str, str, str, bool, bool], ...] = (
    ("claude-fable-5-1", "10", "50", "12.50", "20", "0.25", False, True),
    ("claude-mythos-5-1", "10", "50", "12.50", "20", "0.25", False, True),
    ("claude-fable-5", "10", "50", "12.50", "20", "1", False, True),
    ("claude-mythos-5", "10", "50", "12.50", "20", "1", False, True),
    ("claude-opus-5-5", "4", "20", "5", "8", "0.20", True, True),
    ("claude-opus-5", "5", "25", "6.25", "10", "0.50", True, True),
    ("claude-opus-4-8", "5", "25", "6.25", "10", "0.50", True, True),
    ("claude-opus-4-7", "5", "25", "6.25", "10", "0.50", False, True),
    ("claude-opus-4-6", "5", "25", "6.25", "10", "0.50", False, True),
    ("claude-opus-4-5", "5", "25", "6.25", "10", "0.50", False, False),
    ("claude-sonnet-5-5", "2", "10", "2.50", "4", "0.10", False, True),
    ("claude-sonnet-5", "2", "10", "2.50", "4", "0.20", False, True),
    ("claude-sonnet-4-6", "3", "15", "3.75", "6", "0.30", False, True),
    ("claude-sonnet-4-5", "3", "15", "3.75", "6", "0.30", False, False),
    ("claude-haiku-5-5", "0.10", "0.50", "0.125", "0.20", "0.01", False, True),
    ("claude-haiku-4-5", "1", "5", "1.25", "2", "0.10", False, False),
)

# Claude Haiku 5.5 is priced by prompt length: prompts over 100k input tokens
# (including cache reads and writes) pay these rates for the whole request.
_LONG_CONTEXT = {
    "claude-haiku-5-5": (LongContextTier(100_000, TokenRates.of("0.50", "2.50", "0.05", "0.625", "1")),),
}

# Web search: $10 per 1,000 searches. Web fetch: no fee beyond tokens. Code
# execution without web search/fetch is billed per container hour, which usage
# does not report, so it is deliberately left unpriced (see the pricing guide).
_FEES = (FeeRate("web_search_request", "10", per=1000), FeeRate("web_fetch_request", "0"))  # type: ignore[arg-type]


def _tariffs(fast: bool, residency: bool) -> tuple[Tariff, ...]:
    # Batch: 50 % on every token category (caching multipliers stack). Priority Tier
    # is contractual and therefore unknown. Fast mode is 2x the standard rates
    # (caching and data-residency multipliers stack on top). US-only inference
    # (inference_geo="us") is 1.1x on every token category for Claude 4.6 and later.
    tariffs = [Tariff.of("service_tier", "standard", standard="1", batch="0.5")]
    tariffs.append(Tariff.of("speed", "standard", fast="2") if fast else Tariff.of("speed", "standard"))
    tariffs.append(
        Tariff.of("inference_geo", "global", us="1.1") if residency else Tariff.of("inference_geo", "global")
    )
    return tuple(tariffs)


def _anthropic_prices() -> tuple[ModelPrice, ...]:
    return tuple(
        ModelPrice(
            provider="anthropic",
            model=model,
            rates=TokenRates.of(input, output, read, write_5m, write_1h),
            match_dated_versions=True,
            long_context=_LONG_CONTEXT.get(model, ()),
            tariffs=_tariffs(fast, residency),
            fees=_FEES,
            source="builtin",
            source_url=BUILTIN_SOURCE_URL,
            version="2026-10-31",
            checked_at=BUILTIN_CHECKED_AT,
            quality="verified",
        )
        for model, input, output, write_5m, write_1h, read, fast, residency in _ANTHROPIC
    )


BUILTIN_SNAPSHOT = PriceSnapshot(
    BUILTIN_VERSION,
    _anthropic_prices(),
    created_at=BUILTIN_CHECKED_AT,
    source="orkexa",
    description="Anthropic first-party Claude API list prices (USD per million tokens)",
)
"""The built-in catalog as an immutable snapshot."""
