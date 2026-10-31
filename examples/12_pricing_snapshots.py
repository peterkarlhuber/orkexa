"""12. External price catalogs, versioned snapshots and traceable costs (offline).

An application keeps its own prices (a contract, a regional price list) outside
Orkexa, e.g. in a database or a pricing service, and hands them over as a JSON
snapshot. Orkexa resolves prices from it, calculates itemized costs from the usage
a provider reported and records which snapshot version every amount came from.
Nothing here calls a model API.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from orkexa import Agent, UsageTracker
from orkexa.pricing import (
    FeeRate,
    MeteredUsage,
    ModelPrice,
    PriceResolver,
    PriceSnapshot,
    Tariff,
    TariffOption,
    TokenRates,
    round_cost,
)
from orkexa.providers.fake import FakeModel, text

CHECKED = datetime(2026, 10, 30, tzinfo=timezone.utc)


def contract_snapshot(version: str, input_rate: str) -> str:
    """What an external pricing service would deliver (example prices, USD per MTok)."""
    gpt = ModelPrice(
        provider="openai",
        model="gpt-5",
        rates=TokenRates.of(
            input_rate, "10", cache_read="0.125", cache_write_5m="1.25", cache_write_1h="1.25"
        ),
        aliases=("gpt-5-prod",),  # an explicit alias, never a prefix match
        match_dated_versions=True,  # gpt-5-2025-08-07 resolves to gpt-5
        tariffs=(
            Tariff(
                "service_tier",
                "default",
                (
                    TariffOption("default"),
                    TariffOption("flex", Decimal("0.5")),
                    TariffOption("batch", Decimal("0.5")),
                ),
            ),
        ),
        fees=(FeeRate("web_search_call", Decimal("10"), per=1000),),
        source="acme-contract",
        source_url="https://example.com/contracts/acme",
        version=version,
        checked_at=CHECKED,
        quality="contract",
    )
    fake = ModelPrice(
        provider="fake",
        model="demo",
        rates=TokenRates.of("1", "2"),
        source="acme-contract",
        quality="estimated",
    )
    return PriceSnapshot(version, (gpt, fake), created_at=CHECKED, source="acme-pricing-service").to_json()


def main() -> None:
    resolver = PriceResolver(PriceSnapshot.from_json(contract_snapshot("acme-2026-11", "1.25")))

    # 1. Price usage data from anywhere (here: what OpenAI reported for one request).
    usage = MeteredUsage(
        input_tokens=120_000,
        output_tokens=8_000,
        thinking_tokens=3_000,  # part of output_tokens, never charged twice
        cache_read_input_tokens=400_000,
        units={"web_search_call": 2},
        conditions={"service_tier": "flex"},
    )
    cost = resolver.calculate("openai", "gpt-5-2025-08-07", usage)
    for component in cost.components:
        print(
            f"{component.name:>20}: {component.quantity:>8} x {component.rate}"
            f" / {component.per} = {component.amount}"
        )
    reference = cost.references[0]
    total = round_cost(cost.total or Decimal(0))
    print(
        f"total ${total} from snapshot {reference.snapshot_version} ({reference.source}, {reference.quality})"
    )

    # 2. Unknown conditions stay unknown: no standard rate is assumed.
    priority = resolver.calculate(
        "openai", "gpt-5", MeteredUsage(input_tokens=1000, conditions={"service_tier": "priority"})
    )
    print(f"priority tier: total={priority.total} missing={list(priority.missing)}")

    # 3. A pinned basis keeps its prices while the catalog is updated.
    pinned = resolver.basis()
    resolver.publish(PriceSnapshot.from_json(contract_snapshot("acme-2026-12", "1.00")))
    before = pinned.calculate("openai", "gpt-5", MeteredUsage(input_tokens=1_000_000)).total
    after = resolver.calculate("openai", "gpt-5", MeteredUsage(input_tokens=1_000_000)).total
    print(f"pinned acme-2026-11: ${before}, current {resolver.snapshot.version}: ${after}")

    # 4. Agents record the price basis of every request.
    tracker = UsageTracker(price_resolver=resolver)
    agent = Agent(model="fake:demo", client=FakeModel([text("ok")]), usage_tracker=tracker)
    agent.ask("hi")
    record = tracker.records[-1]
    print(
        f"record: cost=${record.estimated_cost_usd} version={record.pricing_version} "
        f"quality={record.pricing_quality} source={record.pricing_source}"
    )


if __name__ == "__main__":
    main()
