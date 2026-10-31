"""The pricing engine: data model, resolution, snapshots, calculation and serialization.

Everything here is offline and deterministic: prices are test data, never fetched.
"""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
import threading

import pytest

from orkexa import MODEL_PRICING, ModelPricing, register_pricing, resolve_pricing
from orkexa.errors import PriceNotFoundError, PricingError
from orkexa.pricing import (
    BUILTIN_SNAPSHOT,
    BUILTIN_VERSION,
    PROVIDER_PRICING,
    SNAPSHOT_FORMAT,
    CostBreakdown,
    FeeRate,
    LongContextTier,
    MeteredUsage,
    ModelPrice,
    PricePolicy,
    PriceResolver,
    PriceSnapshot,
    PricingBasis,
    Tariff,
    TariffOption,
    TokenRates,
    calculate_cost,
    default_resolver,
    pricing_basis,
    round_cost,
)

UTC = timezone.utc
BUILTIN = PriceResolver(BUILTIN_SNAPSHOT)  # independent of what other tests registered
NOW = datetime(2026, 11, 1, tzinfo=UTC)
M = 1_000_000


def _price(provider: str = "openai", model: str = "gpt-5", **changes: object) -> ModelPrice:
    values: dict[str, object] = {
        "provider": provider,
        "model": model,
        "rates": TokenRates.of("1.25", "10", "0.125", "1.25", "1.25"),
        "source": "test-catalog",
        "version": "2026-11",
        "quality": "contract",
        "checked_at": datetime(2026, 10, 30, tzinfo=UTC),
    }
    values.update(changes)
    return ModelPrice(**values)  # type: ignore[arg-type]


@pytest.fixture
def clean_registry() -> Iterator[None]:
    """Restore the process-wide legacy dictionaries after a test."""
    anthropic, others = dict(MODEL_PRICING), dict(PROVIDER_PRICING)
    PROVIDER_PRICING.clear()  # other tests and examples may have registered prices
    try:
        yield
    finally:
        MODEL_PRICING.clear()
        MODEL_PRICING.update(anthropic)
        PROVIDER_PRICING.clear()
        PROVIDER_PRICING.update(others)


# ---- data model ------------------------------------------------------------------------------


def test_amounts_are_decimals_and_floats_are_rejected() -> None:
    rates = TokenRates.of("0.10", 1, Decimal("0.01"))
    assert rates.input == Decimal("0.10") and isinstance(rates.output, Decimal)
    with pytest.raises(PricingError, match="float"):
        TokenRates.of(0.1, "1")  # type: ignore[arg-type]
    with pytest.raises(PricingError, match="negative"):
        TokenRates.of("-1", "1")
    with pytest.raises(PricingError, match="decimal"):
        TokenRates.of("ten", "1")
    with pytest.raises(PricingError, match="_input"):
        TokenRates.of("1", "1", audio="2")


def test_model_price_validation() -> None:
    with pytest.raises(PricingError, match="currency"):
        _price(currency="usd")
    with pytest.raises(PricingError, match="aliases"):
        _price(aliases=("gpt-5",))
    with pytest.raises(PricingError, match="timezone-aware"):
        _price(valid_from=datetime(2026, 1, 1))
    with pytest.raises(PricingError, match="valid_until"):
        _price(valid_from=NOW, valid_until=NOW)
    with pytest.raises(PricingError, match="quality"):
        _price(quality="great")
    with pytest.raises(PricingError, match="not an option"):
        Tariff("service_tier", "standard", (TariffOption("batch", Decimal("0.5")),))
    with pytest.raises(PricingError, match="long_context"):
        _price(
            long_context=(LongContextTier(272_000, TokenRates.of("2.5", "15")),),
            tariffs=(
                Tariff(
                    "service_tier",
                    "default",
                    (TariffOption("default"), TariffOption("flex", rates=TokenRates.of("1", "5"))),
                ),
            ),
        )


def test_prices_are_immutable_and_hashable() -> None:
    price = _price(rates=TokenRates.of("1", "2", audio_input="3"))
    with pytest.raises(AttributeError):
        price.model = "other"  # type: ignore[misc]
    with pytest.raises(TypeError):
        price.rates.special["audio_input"] = Decimal("0")  # type: ignore[index]
    assert hash(price) == hash(_price(rates=TokenRates.of("1", "2", audio_input="3")))
    changed = price.with_changes(version="2027-01")
    assert changed.version == "2027-01" and price.version == "2026-11"


def test_builtin_catalog_matches_the_legacy_table_and_is_verified() -> None:
    assert BUILTIN_SNAPSHOT.version == BUILTIN_VERSION
    for model, legacy in MODEL_PRICING.items():
        price = BUILTIN_SNAPSHOT.get("anthropic", model, at=NOW)
        assert price is not None and price.quality == "verified" and price.source == "builtin"
        assert price.checked_at is not None and price.currency == "USD"
        assert (price.rates.input, price.rates.output, price.rates.cache_read) == (
            legacy.input,
            legacy.output,
            legacy.cache_read,
        )
        assert (price.rates.cache_write_5m, price.rates.cache_write_1h) == (
            legacy.cache_write_5m,
            legacy.cache_write_1h,
        )
    # Spot checks against the official price list (2026-10-31).
    opus = BUILTIN_SNAPSHOT.get("anthropic", "claude-opus-5-5")
    assert opus is not None and opus.rates == TokenRates.of("4", "20", "0.20", "5", "8")
    sonnet = BUILTIN_SNAPSHOT.get("anthropic", "claude-sonnet-5-5")
    assert sonnet is not None and sonnet.rates.cache_read == Decimal("0.10")


# ---- resolution ------------------------------------------------------------------------------


def test_exact_alias_and_dated_resolution() -> None:
    snapshot = PriceSnapshot(
        "v1",
        (
            _price(model="gpt-5", aliases=("gpt-5-latest",), match_dated_versions=True),
            _price(model="gpt-5-mini", rates=TokenRates.of("0.25", "2")),
            _price(provider="gemini", model="gemini-3-pro"),
        ),
    )
    exact = snapshot.lookup("openai", "gpt-5", at=NOW)
    assert exact.matched_by == "exact" and exact.snapshot_version == "v1"
    assert snapshot.lookup("openai", "gpt-5-latest", at=NOW).matched_by == "alias"
    for dated in ("gpt-5-2025-08-07", "gpt-5-20250807", "gpt-5@20250807"):
        found = snapshot.lookup("openai", dated, at=NOW)
        assert found.matched_by == "dated" and found.require().model == "gpt-5"
    # No prefix matches: gpt-5-mini has its own price, gpt-5-nano has none.
    assert snapshot.lookup("openai", "gpt-5-mini", at=NOW).require().rates.input == Decimal("0.25")
    assert not snapshot.lookup("openai", "gpt-5-nano", at=NOW).found
    assert not snapshot.lookup("openai", "gpt-5-pro-2025", at=NOW).found
    # Dated ids only resolve where the price opts in.
    assert not snapshot.lookup("openai", "gpt-5-mini-2025-08-07", at=NOW).found
    # Aliases do not resolve dated variants.
    assert not snapshot.lookup("openai", "gpt-5-latest-2025-08-07", at=NOW).found


def test_builtin_anthropic_resolution_keeps_its_rules() -> None:
    snapshot = BUILTIN_SNAPSHOT
    assert snapshot.lookup("anthropic", "claude-opus-5-20260101", at=NOW).require().model == "claude-opus-5"
    assert (
        snapshot.lookup("anthropic", "claude-opus-5-5-20260101", at=NOW).require().model == "claude-opus-5-5"
    )
    assert not snapshot.lookup("anthropic", "claude-opus-5-5-x", at=NOW).found
    assert not snapshot.lookup("anthropic", "claude-opus", at=NOW).found


def test_prices_never_cross_providers() -> None:
    snapshot = PriceSnapshot("v1", (_price(provider="openai", model="gpt-5"),))
    assert not snapshot.lookup("azure", "gpt-5", at=NOW).found
    assert not snapshot.lookup("openrouter", "gpt-5", at=NOW).found
    assert not BUILTIN_SNAPSHOT.lookup("bedrock", "claude-opus-5-5", at=NOW).found
    basis = PricingBasis(snapshot, _price(provider="gemini", model="gpt-5"))
    assert basis.lookup("openai", "gpt-5", at=NOW).matched_by == "exact"  # gemini override ignored


def test_unknown_prices_are_unknown_never_zero() -> None:
    lookup = BUILTIN_SNAPSHOT.lookup("openai", "gpt-5", at=NOW)
    assert lookup.price is None and "No known pricing" in (lookup.reason or "")
    with pytest.raises(PriceNotFoundError):
        lookup.require()
    cost = PricingBasis(BUILTIN_SNAPSHOT).calculate(
        "openai", "gpt-5", MeteredUsage(input_tokens=1000), at=NOW
    )
    assert cost.total is None and cost.missing == ("price",) and not cost.complete
    assert cost.known_total == 0 and cost.references == ()
    assert calculate_cost(None, MeteredUsage()).total is None


def test_validity_periods_select_the_price_in_force() -> None:
    switch = datetime(2026, 12, 1, tzinfo=UTC)
    old = _price(valid_until=switch, version="2026-11")
    new = _price(valid_from=switch, rates=TokenRates.of("1", "8", "0.1", "1", "1"), version="2026-12")
    snapshot = PriceSnapshot("history", (old, new))
    assert snapshot.lookup("openai", "gpt-5", at=NOW).require().version == "2026-11"
    assert snapshot.lookup("openai", "gpt-5", at=switch).require().version == "2026-12"
    ended = PriceSnapshot("ended", (old,)).lookup("openai", "gpt-5", at=switch)
    assert not ended.found and "valid at" in (ended.reason or "")
    with pytest.raises(PricingError, match="overlapping"):
        PriceSnapshot("bad", (old, _price(valid_from=NOW - timedelta(days=1))))
    with pytest.raises(PricingError, match="overlapping"):
        PriceSnapshot("bad", (_price(model="a", aliases=("b",)), _price(model="b")))


def test_requested_model_fallback_and_overrides() -> None:
    snapshot = PriceSnapshot("v1", (_price(model="my-deployment"),))
    basis = PricingBasis(snapshot)
    found = basis.lookup("openai", "gpt-5-2025-08-07", requested_model="my-deployment", at=NOW)
    assert found.matched_by == "requested" and found.require().model == "my-deployment"
    assert basis.lookup("openai", "my-deployment", at=NOW).matched_by == "exact"
    legacy = PricingBasis(snapshot, ModelPricing.of("7", "0", "0", "0", "0"))
    lookup = legacy.lookup("openai", "anything", at=NOW)
    assert lookup.matched_by == "override" and lookup.require().source == "custom"
    rich = PricingBasis(snapshot, _price(model="gpt-5", rates=TokenRates.of("2", "4")))
    assert rich.lookup("openai", "gpt-5", at=NOW).matched_by == "override"
    assert rich.lookup("openai", "my-deployment", at=NOW).matched_by == "exact"  # falls back to the snapshot


# ---- calculation -----------------------------------------------------------------------------


def test_anthropic_cost_with_cache_ttls_and_server_tools() -> None:
    usage = MeteredUsage(
        input_tokens=M,
        output_tokens=100_000,
        thinking_tokens=40_000,
        cache_read_input_tokens=M,
        cache_write_5m_input_tokens=200_000,
        cache_write_1h_input_tokens=100_000,
        units={"web_search_request": 3, "web_fetch_request": 2},
    )
    cost = BUILTIN.calculate("anthropic", "claude-opus-5-5", usage, at=NOW)
    # 4 (input) + 2 (output incl. thinking) + 0.2 (read) + 1.0 (5m) + 0.8 (1h) + 0.03 (searches)
    assert cost.total == Decimal("8.03") and cost.complete
    names = {c.name: c.amount for c in cost.components}
    assert names["output"] == Decimal("2")  # thinking is part of output, charged once
    assert names["fee:web_search_request"] == Decimal("0.03") and names["fee:web_fetch_request"] == 0
    (reference,) = cost.references
    assert reference.snapshot_version == BUILTIN_VERSION and reference.quality == "verified"
    assert reference.conditions == (
        ("inference_geo", "global"),
        ("service_tier", "standard"),
        ("speed", "standard"),
    )


def test_cache_writes_without_a_lifetime() -> None:
    usage = MeteredUsage(cache_write_input_tokens=M, cache_write_5m_input_tokens=400_000)
    cost = BUILTIN.calculate("anthropic", "claude-opus-5-5", usage, at=NOW)
    assert cost.total is None and cost.missing == ("cache_write_ttl",)
    assert cost.known_total == Decimal("2")  # the 5-minute share is known
    single_rate = _price(rates=TokenRates.of("1", "1", "0.1", "1.25", "1.25"))
    assert calculate_cost(single_rate, usage).total == Decimal("1.25")
    no_cache_prices = _price(rates=TokenRates.of("1", "1"))
    unknown = calculate_cost(
        no_cache_prices, MeteredUsage(cache_read_input_tokens=10, cache_write_1h_input_tokens=5)
    )
    assert set(unknown.missing) == {"cache_read", "cache_write_1h"}


def test_long_context_tier_prices_the_whole_request() -> None:
    resolver = BUILTIN
    short = resolver.calculate("anthropic", "claude-haiku-5-5", MeteredUsage(input_tokens=100_000), at=NOW)
    assert short.total == Decimal("0.01")  # 100k x $0.10/MTok
    assert short.references[0].long_context_above is None
    # Cache reads count towards the prompt length; then every token pays the long-context rate.
    long = resolver.calculate(
        "anthropic",
        "claude-haiku-5-5",
        MeteredUsage(input_tokens=90_000, cache_read_input_tokens=10_001, output_tokens=1_000),
        at=NOW,
    )
    assert long.total == Decimal("0.045") + Decimal("0.00050005") + Decimal("0.0025")
    assert long.references[0].long_context_above == 100_000
    tiers = _price(
        long_context=(
            LongContextTier(500_000, TokenRates.of("4", "30")),
            LongContextTier(200_000, TokenRates.of("2", "20")),
        )
    )
    assert calculate_cost(tiers, MeteredUsage(input_tokens=300_000)).total == Decimal("0.6")
    assert calculate_cost(tiers, MeteredUsage(input_tokens=600_000)).total == Decimal("2.4")


def test_batch_fast_mode_and_data_residency_stack() -> None:
    resolver = BUILTIN

    def cost(model: str, **conditions: str) -> Decimal | None:
        usage = MeteredUsage(input_tokens=M, cache_read_input_tokens=M, conditions=conditions)
        return resolver.calculate("anthropic", model, usage, at=NOW).total

    assert cost("claude-opus-5-5") == Decimal("4.20")
    assert cost("claude-opus-5-5", service_tier="batch") == Decimal("2.10")
    assert cost("claude-opus-5-5", speed="fast") == Decimal("8.40")
    assert cost("claude-opus-5-5", speed="fast", inference_geo="us") == Decimal("9.240")
    assert cost("claude-opus-5-5", service_tier="batch", inference_geo="us") == Decimal("2.310")
    assert cost("claude-opus-4-8", speed="fast") == Decimal("11.0")  # 10 + 1 (2x 0.50)
    # Unknown tariff conditions are never priced as standard.
    assert cost("claude-opus-5-5", service_tier="priority") is None
    assert cost("claude-sonnet-5-5", speed="fast") is None  # no fast mode for this model
    assert cost("claude-haiku-4-5", inference_geo="us") is None  # residency pricing starts with 4.6
    assert cost("claude-opus-5-5", region="eu-west") is None  # undeclared dimension
    haiku_long_us = resolver.calculate(
        "anthropic",
        "claude-haiku-5-5",
        MeteredUsage(input_tokens=200_000, conditions={"inference_geo": "us"}),
        at=NOW,
    )
    assert haiku_long_us.total == Decimal("0.11")  # 200k x $0.50 x 1.1


def test_service_tiers_with_replacement_rates() -> None:
    price = _price(
        tariffs=(
            Tariff(
                "service_tier",
                "default",
                (
                    TariffOption("default"),
                    TariffOption("flex", rates=TokenRates.of("0.625", "5", "0.0625")),
                    TariffOption("priority", rates=TokenRates.of("2.5", "20", "0.25")),
                    TariffOption("batch", Decimal("0.5")),
                ),
            ),
        )
    )
    usage = MeteredUsage(input_tokens=M, output_tokens=M)
    assert calculate_cost(price, usage).total == Decimal("11.25")
    assert calculate_cost(
        price, MeteredUsage(input_tokens=M, conditions={"service_tier": "flex"})
    ).total == Decimal("0.625")
    assert calculate_cost(
        price, MeteredUsage(output_tokens=M, conditions={"service_tier": "priority"})
    ).total == Decimal("20")
    assert calculate_cost(
        price, MeteredUsage(output_tokens=M, conditions={"service_tier": "batch"})
    ).total == Decimal("5")
    # "standard" is a neutral synonym of the "default" option.
    assert calculate_cost(
        price, MeteredUsage(input_tokens=M, conditions={"service_tier": "standard"})
    ).total == Decimal("1.25")
    assert (
        calculate_cost(price, MeteredUsage(input_tokens=M, conditions={"service_tier": "scale"})).total
        is None
    )
    # A price without a service-tier tariff accepts only the neutral values.
    plain = _price()
    assert calculate_cost(
        plain, MeteredUsage(input_tokens=M, conditions={"service_tier": "default"})
    ).complete
    assert calculate_cost(
        plain, MeteredUsage(input_tokens=M, conditions={"service_tier": "flex"})
    ).missing == ("tariff:service_tier=flex",)


def test_fees_and_special_token_classes() -> None:
    price = _price(
        rates=TokenRates.of("0.30", "2.50", "0.03", audio_input="1.00", audio_cache_read="0.10"),
        fees=(
            FeeRate("google_search_query", Decimal("14"), per=1000),
            FeeRate("google_search_grounded_prompt", Decimal("0")),
        ),
    )
    usage = MeteredUsage(
        input_tokens=M,
        cache_read_input_tokens=M,
        token_details={"audio_input": 400_000, "audio_cache_read": 100_000},
        units={"google_search_query": 5, "google_search_grounded_prompt": 1},
    )
    cost = calculate_cost(price, usage)
    amounts = {c.name: c.amount for c in cost.components}
    assert amounts["input"] == Decimal("0.18")  # 600k text tokens
    assert amounts["audio_input"] == Decimal("0.4")
    assert amounts["cache_read"] == Decimal("0.027") and amounts["audio_cache_read"] == Decimal("0.01")
    assert amounts["fee:google_search_query"] == Decimal("0.07")
    assert cost.total == Decimal("0.687")
    missing = calculate_cost(
        _price(),
        MeteredUsage(input_tokens=M, token_details={"audio_input": 10}, units={"web_search_call": 2}),
    )
    assert set(missing.missing) == {"tokens:audio_input", "fee:web_search_call"}
    assert missing.total is None and missing.known_total == Decimal("1.24998750")


def test_incomplete_usage_is_never_a_complete_cost() -> None:
    cost = calculate_cost(_price(), MeteredUsage(input_tokens=M, missing=("code_execution_time",)))
    assert cost.total is None and cost.known_total == Decimal("1.25")
    with pytest.raises(PricingError, match="thinking"):
        MeteredUsage(output_tokens=1, thinking_tokens=2)
    with pytest.raises(PricingError, match="nonnegative"):
        MeteredUsage(input_tokens=-1)
    with pytest.raises(PricingError, match="exceed"):
        MeteredUsage(cache_write_input_tokens=1, cache_write_5m_input_tokens=2)


def test_decimal_precision_and_rounding() -> None:
    price = _price(rates=TokenRates.of("0.01", "0.03"))
    one = calculate_cost(price, MeteredUsage(input_tokens=1, output_tokens=1))
    assert one.total == Decimal("0.00000004")  # exact, no float noise
    total = CostBreakdown.combine([one] * 25_000)
    assert total.total == Decimal("0.001") and len(total.components) == 50_000
    assert round_cost(Decimal("0.0000025")) == Decimal("0.000002")  # banker's rounding
    assert round_cost(Decimal("0.0000035")) == Decimal("0.000004")
    assert round_cost(Decimal("1.005"), 2) == Decimal("1.00")
    huge = calculate_cost(price, MeteredUsage(input_tokens=10**15))
    assert huge.total == Decimal("10000000")


def test_mixed_currencies_are_not_added() -> None:
    eur = calculate_cost(_price(currency="EUR"), MeteredUsage(input_tokens=M))
    usd = calculate_cost(_price(), MeteredUsage(input_tokens=M))
    assert eur.currency == "EUR" and eur.total == Decimal("1.25")
    combined = CostBreakdown.combine([usd, eur])
    assert (
        combined.total is None and "currency" in combined.missing and combined.known_total == Decimal("1.25")
    )


# ---- snapshots, resolvers and serialization -------------------------------------------------


def test_snapshot_serialization_round_trip() -> None:
    price = _price(
        aliases=("gpt-5-latest",),
        match_dated_versions=True,
        long_context=(LongContextTier(272_000, TokenRates.of("2.5", "15", "0.25")),),
        tariffs=(
            Tariff(
                "service_tier",
                "default",
                (
                    TariffOption("default"),
                    TariffOption("flex", rates=TokenRates.of("0.625", "5"), long_context=()),
                    TariffOption("batch", Decimal("0.5")),
                ),
            ),
        ),
        fees=(FeeRate("web_search_call", Decimal("10"), per=1000),),
        valid_from=datetime(2026, 1, 1, tzinfo=UTC),
        valid_until=datetime(2027, 1, 1, tzinfo=UTC),
        source_url="https://example.com/prices",
        notes="Contract 42",
        rates=TokenRates.of("1.25", "10", "0.125", "1.25", "1.25", audio_input="32"),
    )
    snapshot = PriceSnapshot("acme-2026-11", (price,), created_at=NOW, source="acme", description="test")
    text = snapshot.to_json()
    data = json.loads(text)
    assert data["format"] == SNAPSHOT_FORMAT and data["prices"][0]["rates"]["input"] == "1.25"
    loaded = PriceSnapshot.from_json(text)
    assert loaded == snapshot and loaded.content_hash() == snapshot.content_hash()
    assert loaded.prices[0] == price
    assert PriceSnapshot.from_dict(BUILTIN_SNAPSHOT.to_dict()) == BUILTIN_SNAPSHOT
    with pytest.raises(PricingError, match="format"):
        PriceSnapshot.from_dict({**data, "format": "other/9"})
    with pytest.raises(PricingError, match="Unknown keys"):
        PriceSnapshot.from_dict({**data, "prices": [{**data["prices"][0], "discount": "5"}]})
    with pytest.raises(PricingError, match="JSON"):
        PriceSnapshot.from_json("{")
    with pytest.raises(PricingError, match="float"):
        ModelPrice.from_dict({**data["prices"][0], "rates": {"input": 1.25, "output": "1"}})


def test_cost_breakdown_serialization_round_trip() -> None:
    usage = MeteredUsage(input_tokens=1234, output_tokens=56, conditions={"speed": "fast"})
    cost = BUILTIN.calculate("anthropic", "claude-opus-5-5", usage, at=NOW)
    assert CostBreakdown.from_dict(json.loads(json.dumps(cost.to_dict()))) == cost
    assert MeteredUsage.from_dict(usage.to_dict()) == usage


def test_snapshots_are_immutable_and_versioned() -> None:
    v1 = PriceSnapshot("v1", (_price(),))
    v2 = v1.with_prices([_price(rates=TokenRates.of("2", "20"))])
    assert v1.get("openai", "gpt-5", at=NOW).rates.input == Decimal("1.25")  # type: ignore[union-attr]
    assert v2.get("openai", "gpt-5", at=NOW).rates.input == Decimal("2")  # type: ignore[union-attr]
    assert (
        v2.version.startswith("v1+")
        and v2.version == v1.with_prices([_price(rates=TokenRates.of("2", "20"))]).version
    )
    assert len(v1.with_prices([_price(model="gpt-5-mini")], version="v3")) == 2
    with pytest.raises(AttributeError):
        v1.version = "v9"  # type: ignore[misc]


def test_resolver_instances_are_isolated(clean_registry: None) -> None:
    first = PriceResolver([_price()], version="first")
    second = PriceResolver(version="second")
    first.register(_price(model="gpt-5-mini"))
    assert first.resolve("openai", "gpt-5-mini", at=NOW).found
    assert not second.resolve("openai", "gpt-5-mini", at=NOW).found
    assert not default_resolver().resolve("openai", "gpt-5", at=NOW).found
    assert second.versions == ("second",) and len(first.versions) == 2
    assert first.get_snapshot("first").get("openai", "gpt-5-mini") is None
    with pytest.raises(PricingError, match="already published"):
        first.publish(PriceSnapshot("first", (_price(rates=TokenRates.of("9", "9")),)))
    pinned = first.basis(snapshot="first")
    assert pinned.version == "first"


def test_resolver_history_is_bounded() -> None:
    resolver = PriceResolver(version="v0", max_history=2)
    for index in range(1, 4):
        resolver.publish(PriceSnapshot(f"v{index}"))
    assert resolver.versions == ("v2", "v3")
    with pytest.raises(KeyError):
        resolver.get_snapshot("v0")


def test_concurrent_updates_never_mix_price_versions() -> None:
    """Many threads price requests while another publishes new snapshots."""
    resolver = PriceResolver([_price(rates=TokenRates.of("1", "1"))], version="v0")
    stop = threading.Event()

    def publisher() -> None:
        index = 0
        while not stop.is_set() and index < 200:
            index += 1
            resolver.publish(PriceSnapshot(f"v{index}", (_price(rates=TokenRates.of(str(index + 1), "1")),)))

    def price_requests(_: int) -> list[tuple[str | None, Decimal | None]]:
        results = []
        for _ in range(200):
            basis = resolver.basis()  # one pinned basis per request
            first = basis.calculate("openai", "gpt-5", MeteredUsage(input_tokens=M), at=NOW)
            second = basis.calculate("openai", "gpt-5", MeteredUsage(input_tokens=M), at=NOW)
            assert first == second  # both parts of a request use the same prices
            results.append((first.references[0].snapshot_version, first.total))
        return results

    thread = threading.Thread(target=publisher)
    thread.start()
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = [item for chunk in pool.map(price_requests, range(8)) for item in chunk]
    stop.set()
    thread.join()
    for version, total in results:
        assert version is not None
        assert total == Decimal(int(version[1:]) + 1)  # the amount always matches its version


def test_pinned_basis_keeps_its_prices_after_an_update() -> None:
    resolver = PriceResolver([_price()], version="v1")
    basis = resolver.basis()
    resolver.register(_price(rates=TokenRates.of("5", "5")), version="v2")
    assert basis.calculate("openai", "gpt-5", MeteredUsage(input_tokens=M), at=NOW).total == Decimal("1.25")
    assert resolver.calculate("openai", "gpt-5", MeteredUsage(input_tokens=M), at=NOW).total == Decimal("5")
    assert resolver.calculate(
        "openai", "gpt-5", MeteredUsage(input_tokens=M), at=NOW, snapshot="v1"
    ).total == Decimal("1.25")


def test_pricing_basis_accepts_every_pricing_input() -> None:
    resolver = PriceResolver([_price()], version="mine")
    snapshot = PriceSnapshot("snap", (_price(),))
    assert pricing_basis(resolver).version == "mine"
    assert pricing_basis(snapshot).version == "snap"
    assert pricing_basis(None).version == default_resolver().snapshot.version
    assert pricing_basis(None, resolver=resolver).version == "mine"
    assert pricing_basis(_price(), resolver=resolver).override == _price()
    basis = PricingBasis(snapshot)
    assert pricing_basis(basis) is basis
    with pytest.raises(TypeError):
        pricing_basis("gpt-5")  # type: ignore[arg-type]


# ---- the default resolver and the legacy API -------------------------------------------------


def test_default_resolver_follows_the_legacy_dictionaries(clean_registry: None) -> None:
    resolver = default_resolver()
    assert resolver.snapshot is BUILTIN_SNAPSHOT
    register_pricing("openai", "gpt-5", ModelPricing.of("1.25", "10", "1.25", "1.25", "0.125"))
    registered = resolver.resolve("openai", "gpt-5", at=NOW).require()
    assert registered.source == "registered" and registered.quality == "unverified"
    version = resolver.snapshot.version
    assert version.startswith(BUILTIN_VERSION + "+local.")
    assert resolver.snapshot.version == version  # unchanged state, same snapshot
    MODEL_PRICING["my-claude"] = ModelPricing.of("1", "2", "3", "4", "5")  # direct edits still count
    assert resolver.resolve("anthropic", "my-claude-20260101", at=NOW).require().source == "registered"
    register_pricing("gemini", "gemini-3-pro", _price(provider="gemini", model="gemini-3-pro"))
    rich = resolver.resolve("gemini", "gemini-3-pro", at=NOW).require()
    assert rich.source == "test-catalog" and rich.quality == "contract"
    assert resolve_pricing("gemini-3-pro", provider="gemini") == ModelPricing(
        Decimal("1.25"), Decimal("10"), Decimal("1.25"), Decimal("1.25"), Decimal("0.125")
    )
    with pytest.raises(ValueError, match="not"):
        register_pricing("gemini", "other", _price(provider="gemini", model="gemini-3-pro"))
    with pytest.raises(TypeError):
        resolver.publish(PriceSnapshot("x"))
    del MODEL_PRICING["my-claude"]
    del PROVIDER_PRICING[("openai", "gpt-5")]
    del PROVIDER_PRICING[("gemini", "gemini-3-pro")]
    assert resolver.snapshot is BUILTIN_SNAPSHOT  # back to the pristine catalog


def test_legacy_model_pricing_converts_with_legacy_semantics() -> None:
    legacy = ModelPricing.of("1", "2", "1", "1", "0.1")
    price = legacy.to_model_price("openai", "gpt-5")
    assert price.rates == TokenRates.of("1", "2", "0.1", "1", "1")
    assert calculate_cost(
        price, MeteredUsage(input_tokens=M, conditions={"service_tier": "batch"})
    ).total == Decimal("0.5")
    assert calculate_cost(
        price, MeteredUsage(input_tokens=M, conditions={"service_tier": "default"})
    ).total == Decimal("1")
    assert (
        calculate_cost(price, MeteredUsage(input_tokens=M, conditions={"service_tier": "flex"})).total is None
    )
    anthropic = legacy.to_model_price("anthropic", "claude-x")
    assert anthropic.match_dated_versions and anthropic.fee("web_search_request") is not None


# ---- price policy ----------------------------------------------------------------------------


def test_price_policy_flags_stale_and_unaccepted_prices() -> None:
    policy = PricePolicy(max_age=timedelta(days=30), accepted_quality={"verified", "contract"})
    assert policy.check_price(_price(), at=NOW) == ()
    stale = policy.check_price(_price(checked_at=NOW - timedelta(days=31)), at=NOW)
    assert stale and "stale" in stale[0]
    assert "no check date" in policy.check_price(_price(checked_at=None), at=NOW)[0]
    assert "quality" in policy.check_price(_price(quality="estimated"), at=NOW)[0]
    strict = PricePolicy(allow_incomplete=False)
    incomplete = calculate_cost(_price(), MeteredUsage(units={"web_search_call": 1}))
    assert strict.check_cost(incomplete) and not PricePolicy().check_cost(incomplete)
    with pytest.raises(PricingError):
        PricePolicy(accepted_quality={"perfect"})
    with pytest.raises(PricingError):
        PricePolicy(max_age=timedelta(0))


def test_the_documented_snapshot_format_loads() -> None:
    from pathlib import Path

    guide = (Path(__file__).resolve().parents[1] / "docs" / "guides" / "pricing.md").read_text()
    block = guide.split("### Serialization format", 1)[1].split("```json", 1)[1].split("```", 1)[0]
    snapshot = PriceSnapshot.from_json(block)
    price = snapshot.lookup("openai", "gpt-5-prod", at=NOW).require()
    assert price.fee("web_search_call") is not None and price.long_context[0].above_input_tokens == 272_000
