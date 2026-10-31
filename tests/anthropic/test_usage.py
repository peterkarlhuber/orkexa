from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from orkexa import MODEL_PRICING, ModelPricing, UsageTracker, resolve_pricing
from orkexa.usage import record_api_usage, summarize_usage


def _response(model: str, **usage: Any) -> dict[str, Any]:
    return {"model": model, "usage": usage}


def test_resolve_pricing_exact_and_dated_ids() -> None:
    assert resolve_pricing("claude-opus-5-5") is MODEL_PRICING["claude-opus-5-5"]
    assert resolve_pricing("claude-opus-5") is MODEL_PRICING["claude-opus-5"]
    assert resolve_pricing("claude-opus-5-20260101") is MODEL_PRICING["claude-opus-5"]
    assert resolve_pricing("claude-opus-5-5-x") is None
    assert resolve_pricing("unknown-model") is None


def test_haiku_55_long_context_pricing() -> None:
    short = resolve_pricing("claude-haiku-5-5", 100_000)
    long = resolve_pricing("claude-haiku-5-5", 100_001)
    assert short is not None and long is not None
    assert short.input == Decimal("0.10")
    assert long.input == Decimal("0.50")


def test_runtime_pricing_entries_are_used() -> None:
    MODEL_PRICING["my-custom-model"] = ModelPricing.of("1", "2", "3", "4", "5")
    try:
        assert resolve_pricing("my-custom-model") is MODEL_PRICING["my-custom-model"]
    finally:
        del MODEL_PRICING["my-custom-model"]


def test_cost_includes_cache_and_web_search() -> None:
    record = record_api_usage(
        _response(
            "claude-opus-5-5",
            input_tokens=1_000_000,
            output_tokens=100_000,
            cache_read_input_tokens=1_000_000,
            cache_creation_input_tokens=300_000,
            cache_creation={"ephemeral_5m_input_tokens": 200_000, "ephemeral_1h_input_tokens": 100_000},
            output_tokens_details={"thinking_tokens": 40_000},
            server_tool_use={"web_search_requests": 3, "web_fetch_requests": 2},
        ),
        agent_name="a",
        default_model="claude-opus-5-5",
    )
    assert record is not None
    # 4 + 2 (output) + 0.2 (read) + 1.0 (5m write) + 0.8 (1h write) + 0.03 (search)
    assert record.estimated_cost_usd == Decimal("8.03")
    assert record.thinking_tokens == 40_000
    assert record.total_input_tokens == 2_300_000
    assert record.total_tokens == 2_400_000
    assert record.pricing_note is None


@pytest.mark.parametrize(
    ("model", "usage", "note"),
    [
        ("mystery", {}, "No known pricing"),
        ("claude-opus-5-5", {"cache_creation_input_tokens": 10}, "TTL breakdown incomplete"),
        # Fast mode and US data residency are priced where the price list has them
        # (see test_pricing.py); elsewhere they stay unknown.
        ("claude-sonnet-5-5", {"speed": "fast"}, "Nonstandard"),
        ("claude-haiku-4-5", {"inference_geo": "us"}, "Nonstandard"),
        ("claude-opus-5-5", {"service_tier": "priority"}, "Nonstandard"),
    ],
)
def test_unpriceable_requests(model: str, usage: dict[str, Any], note: str) -> None:
    record = record_api_usage(
        _response(model, input_tokens=1, output_tokens=1, **usage),
        agent_name="a",
        default_model=model,
    )
    assert record is not None
    assert record.estimated_cost_usd is None
    assert record.pricing_note is not None and note in record.pricing_note


def test_custom_pricing_overrides_catalog() -> None:
    record = record_api_usage(
        _response("mystery", input_tokens=1_000_000, output_tokens=0),
        agent_name="a",
        default_model="mystery",
        custom_pricing=ModelPricing.of("7", "0", "0", "0", "0"),
    )
    assert record is not None and record.estimated_cost_usd == Decimal("7")


def test_missing_usage_returns_none() -> None:
    assert record_api_usage({}, agent_name="a", default_model="m") is None


def test_summary_and_tracker() -> None:
    priced = record_api_usage(
        _response("claude-haiku-5-5", input_tokens=10, output_tokens=5),
        agent_name="parent",
        default_model="claude-haiku-5-5",
    )
    unpriced = record_api_usage(
        _response("mystery", input_tokens=1, output_tokens=1),
        agent_name="child",
        default_model="mystery",
    )
    assert priced is not None and unpriced is not None

    assert summarize_usage([priced]).estimated_cost_usd == priced.estimated_cost_usd
    summary = summarize_usage([priced, unpriced])
    assert summary.requests == 2
    assert summary.input_tokens == 11
    assert summary.estimated_cost_usd is None
    assert summary.unpriced_requests == 1

    tracker = UsageTracker()
    tracker.add(priced)
    tracker.add(unpriced)
    assert tracker.records == (priced, unpriced)
    assert list(tracker.usage_by_agent()) == ["parent", "child"]
    tracker.reset()
    assert tracker.usage.requests == 0
    assert tracker.usage.estimated_cost_usd == Decimal("0")
