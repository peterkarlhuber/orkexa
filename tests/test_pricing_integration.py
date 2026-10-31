"""The pricing engine inside Orkexa: providers, usage records, trackers and budgets.

Every agent talks to a scripted SDK client (``tests/contract/harness.py``) or the
fake provider; no network access, no API key, no paid request.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
from typing import Any
import warnings

import pytest

from orkexa import Agent, ModelPricing, UsageTracker
from orkexa.core.types import TokenUsage
from orkexa.dynamic import (
    BudgetExceededError,
    BudgetTracker,
    UnpricedModelError,
    UnpricedUsageWarning,
    UnreliablePriceError,
    UnreliablePriceWarning,
)
from orkexa.pricing import (
    BUILTIN_VERSION,
    FeeRate,
    MeteredUsage,
    ModelPrice,
    PricePolicy,
    PriceResolver,
    PriceSnapshot,
    TokenRates,
)
from orkexa.providers.chat_completions import chat_usage
from orkexa.providers.fake import FakeModel, text
from orkexa.providers.gemini import gemini_usage
from orkexa.providers.openai import responses_usage
from orkexa.usage import UsageRecord, record_api_usage, record_token_usage, summarize_usage

from .contract.harness import GeminiHarness, OpenAIHarness

UTC = timezone.utc
M = 1_000_000
CHECKED = datetime.now(UTC) - timedelta(days=1)


def _price(provider: str, model: str, **changes: Any) -> ModelPrice:
    values: dict[str, Any] = {
        "provider": provider,
        "model": model,
        "rates": TokenRates.of("1", "2", "0.1", "1", "1"),
        "source": "acme-contract",
        "version": "2026-11",
        "quality": "contract",
        "checked_at": CHECKED,
    }
    values.update(changes)
    return ModelPrice(**values)


# The fake provider reports 10 input and 5 output tokens per request.
FAKE_PRICE = _price("fake", "m", rates=TokenRates.of("1000000", "1000000"))
FAKE_COST = Decimal("15")


def _fake_agent(tracker: UsageTracker, script: list[Any], **kwargs: Any) -> tuple[Agent[str], FakeModel]:
    fake = FakeModel(script)
    return Agent(model="fake:m", client=fake, usage_tracker=tracker, **kwargs), fake


# ---- OpenAI ----------------------------------------------------------------------------------


def test_openai_agent_with_its_own_resolver_records_the_price_basis() -> None:
    oa = OpenAIHarness()
    resolver = PriceResolver([_price("openai", oa.model)], version="acme-2026-11")
    agent = oa.agent([oa.text("a")], pricing=resolver)
    agent.ask("hi")
    record = agent.last_usage
    assert record is not None
    # 80 uncached x $1 + 50 output x $2 + 20 cached x $0.1 per million tokens
    assert record.estimated_cost_usd == Decimal("0.000182") == record.known_cost_usd
    assert record.pricing_version == "acme-2026-11" and record.pricing_source == "acme-contract"
    assert record.pricing_quality == "contract" and record.requested_model == oa.model
    assert record.missing_cost_components == () and record.cost is not None
    assert {c.name for c in record.cost.components} == {"input", "output", "cache_read"}
    (reference,) = record.cost.references
    assert reference.price_version == "2026-11" and reference.matched_by == "exact"


def test_openai_service_tier_is_never_priced_as_standard() -> None:
    oa = OpenAIHarness()
    resolver = PriceResolver([_price("openai", oa.model)], version="v1")
    default = oa._response([oa._message("a")]).model_copy(update={"service_tier": "default"})
    flex = oa._response([oa._message("b")]).model_copy(update={"service_tier": "flex"})
    agent = oa.agent([default, flex], pricing=resolver)
    agent.ask("one")
    assert agent.last_usage is not None and agent.last_usage.estimated_cost_usd == Decimal("0.000182")
    agent.ask("two")
    record = agent.last_usage
    assert record is not None and record.estimated_cost_usd is None
    assert record.missing_cost_components == ("tariff:service_tier=flex",)
    assert "Nonstandard" in (record.pricing_note or "")


def test_openai_requested_service_tier_counts_when_none_is_reported() -> None:
    usage = responses_usage(
        {"input_tokens": 10, "output_tokens": 1}, {"output": []}, requested_tier="priority"
    )
    assert usage is not None and usage.service_tier == "priority"
    reported = responses_usage({"input_tokens": 10}, {"service_tier": "default"}, requested_tier="auto")
    assert reported is not None and reported.service_tier == "default"


def test_openai_hosted_tool_calls_need_a_fee() -> None:
    usage = responses_usage(
        {"input_tokens": 1000, "output_tokens": 100, "input_tokens_details": {"audio_tokens": 40}},
        {"output": [{"type": "web_search_call"}, {"type": "web_search_call"}, {"type": "function_call"}]},
    )
    assert usage is not None
    assert usage.units == (("web_search_call", 2),) and usage.token_details == (("audio_input", 40),)
    unpriced = record_token_usage(
        usage, agent_name="a", model="gpt-5", provider="openai", custom_pricing=_price("openai", "gpt-5")
    )
    assert unpriced.estimated_cost_usd is None
    assert set(unpriced.missing_cost_components) == {"fee:web_search_call", "tokens:audio_input"}
    assert unpriced.known_cost_usd == Decimal("0.001160")  # 960 x $1 + 100 x $2 per million
    priced = _price(
        "openai",
        "gpt-5",
        rates=TokenRates.of("1", "2", "0.1", audio_input="32"),
        fees=(FeeRate("web_search_call", Decimal("10"), per=1000),),
    )
    record = record_token_usage(
        usage, agent_name="a", model="gpt-5", provider="openai", custom_pricing=priced
    )
    assert record.estimated_cost_usd == Decimal("0.00116") + Decimal("0.00128") + Decimal("0.02")


def test_chat_completions_report_tier_and_audio() -> None:
    usage = chat_usage(
        {"prompt_tokens": 100, "completion_tokens": 10, "completion_tokens_details": {"audio_tokens": 4}},
        "flex",
    )
    assert (
        usage is not None and usage.service_tier == "flex" and usage.token_details == (("audio_output", 4),)
    )


# ---- Gemini ----------------------------------------------------------------------------------


def test_gemini_agent_with_a_tracker_resolver() -> None:
    ge = GeminiHarness()
    tracker = UsageTracker(price_resolver=PriceResolver([_price("gemini", ge.model)], version="g1"))
    agent = ge.agent([ge.text("a")], usage_tracker=tracker)
    agent.ask("hi")
    record = tracker.records[-1]
    # 80 input + 50 output (40 candidates + 10 thoughts, charged once) + 20 cached
    assert record.output_tokens == 50 and record.thinking_tokens == 10
    assert record.estimated_cost_usd == Decimal("0.000182") and record.pricing_version == "g1"


def test_gemini_audio_images_and_grounding_are_reported() -> None:
    usage = gemini_usage(
        {
            "prompt_token_count": 1000,
            "cached_content_token_count": 200,
            "candidates_token_count": 300,
            "prompt_tokens_details": [
                {"modality": "TEXT", "token_count": 700},
                {"modality": "AUDIO", "token_count": 300},
            ],
            "cache_tokens_details": [{"modality": "AUDIO", "token_count": 50}],
            "candidates_tokens_details": [{"modality": "IMAGE", "token_count": 250}],
        },
        {"web_search_queries": ["a", "b"]},
    )
    assert usage is not None
    assert dict(usage.token_details) == {"audio_input": 250, "audio_cache_read": 50, "image_output": 250}
    assert dict(usage.units) == {"google_search_query": 2, "google_search_grounded_prompt": 1}
    text_only = _price("gemini", "gemini-3-pro")
    assert (
        record_token_usage(
            usage, agent_name="a", model="gemini-3-pro", provider="gemini", custom_pricing=text_only
        ).estimated_cost_usd
        is None
    )
    full = _price(
        "gemini",
        "gemini-3-pro",
        rates=TokenRates.of("1", "2", "0.1", audio_input="3", audio_cache_read="0.3", image_output="30"),
        fees=(
            FeeRate("google_search_query", Decimal("14"), per=1000),
            FeeRate("google_search_grounded_prompt", Decimal("0")),
        ),
    )
    record = record_token_usage(
        usage, agent_name="a", model="gemini-3-pro", provider="gemini", custom_pricing=full
    )
    # text 550 x 1 + audio 250 x 3 + cached text 150 x 0.1 + cached audio 50 x 0.3
    # + text output 50 x 2 + image output 250 x 30 (per million) + 2 queries x $0.014
    expected = (Decimal(550 + 750 + 15 + 15 + 100 + 7500) / M) + Decimal("0.028")
    assert record.estimated_cost_usd == expected


# ---- Anthropic -------------------------------------------------------------------------------


def _anthropic(model: str = "claude-opus-5-5", **usage: Any) -> dict[str, Any]:
    return {"model": model, "usage": {"input_tokens": M, "output_tokens": 0, **usage}}


def test_anthropic_fast_mode_residency_and_batch_are_priced() -> None:
    fast = record_api_usage(
        _anthropic(speed="fast", inference_geo="us"), agent_name="a", default_model="claude-opus-5-5"
    )
    assert fast is not None and fast.estimated_cost_usd == Decimal("8.8")
    assert dict(fast.pricing_conditions) == {
        "inference_geo": "us",
        "service_tier": "standard",
        "speed": "fast",
    }
    assert (fast.pricing_version or "").startswith(BUILTIN_VERSION) and fast.pricing_source == "builtin"
    batch = record_api_usage(_anthropic(), agent_name="a", default_model="claude-opus-5-5", batch=True)
    assert batch is not None and batch.estimated_cost_usd == Decimal("2")
    reported = record_api_usage(
        _anthropic(service_tier="batch"), agent_name="a", default_model="claude-opus-5-5"
    )
    assert reported is not None and reported.estimated_cost_usd == Decimal("2")


def test_anthropic_code_execution_without_web_tools_is_incomplete() -> None:
    alone = record_api_usage(
        _anthropic(server_tool_use={"code_execution_requests": 2}),
        agent_name="a",
        default_model="claude-opus-5-5",
    )
    assert alone is not None and alone.estimated_cost_usd is None
    assert alone.known_cost_usd == Decimal("4") and alone.missing_cost_components == (
        "fee:code_execution_request",
    )
    assert "Cost incomplete" in (alone.pricing_note or "")
    with_search = record_api_usage(
        _anthropic(server_tool_use={"code_execution_requests": 2, "web_search_requests": 1}),
        agent_name="a",
        default_model="claude-opus-5-5",
    )
    assert with_search is not None and with_search.estimated_cost_usd == Decimal("4.01")


def test_anthropic_iterations_are_priced_per_model() -> None:
    response = {
        "model": "claude-opus-5-5",
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "iterations": [
                {"type": "compaction", "model": "claude-haiku-5-5", "input_tokens": M, "output_tokens": 0},
                {"type": "message", "model": "claude-opus-5-5", "input_tokens": M, "output_tokens": 0},
                {"type": "advisor", "model": "unknown-model", "input_tokens": 10, "output_tokens": 0},
            ],
        },
    }
    record = record_api_usage(response, agent_name="a", default_model="claude-opus-5-5")
    assert record is not None and record.estimated_cost_usd is None
    # Haiku 5.5 at its long-context rate (1M > 100k: $0.50) + Opus 5.5 ($4); the advisor is unknown
    assert record.known_cost_usd == Decimal("4.50")
    assert "unknown-model" in (record.pricing_note or "")
    assert [r.model for r in record.cost.references] == ["claude-haiku-5-5", "claude-opus-5-5"]  # type: ignore[union-attr]


# ---- one pinned basis per request ------------------------------------------------------------


def test_a_price_update_during_a_request_does_not_change_its_bill() -> None:
    resolver = PriceResolver([FAKE_PRICE], version="v1")
    tracker = UsageTracker(price_resolver=resolver)

    def reply(request: object) -> Any:
        resolver.publish(PriceSnapshot("v2", (FAKE_PRICE.with_changes(rates=TokenRates.of("0", "0")),)))
        return text("answer")

    agent, _ = _fake_agent(tracker, [reply, text("second")])
    agent.ask("first")
    agent.ask("second")
    first, second = tracker.records
    assert (first.pricing_version, first.estimated_cost_usd) == ("v1", FAKE_COST)
    assert (second.pricing_version, second.estimated_cost_usd) == ("v2", Decimal("0"))


def test_agent_pricing_wins_over_the_tracker_resolver() -> None:
    tracker = UsageTracker(price_resolver=PriceResolver([FAKE_PRICE], version="tracker"))
    own = FAKE_PRICE.with_changes(rates=TokenRates.of("1", "1"), source="agent-deal")
    agent, _ = _fake_agent(tracker, [text("a")], pricing=own)
    agent.ask("x")
    record = tracker.records[-1]
    assert record.estimated_cost_usd == Decimal("0.000015") and record.pricing_source == "custom"
    assert record.cost is not None and record.cost.references[0].source == "agent-deal"
    with pytest.raises(TypeError):
        Agent(model="fake:m", client=FakeModel([]), pricing="cheap")  # type: ignore[arg-type]


# ---- records and summaries -------------------------------------------------------------------


def _legacy_record(cost: Decimal | None) -> UsageRecord:
    return UsageRecord(
        agent="old",
        model="m",
        input_tokens=1,
        output_tokens=1,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
        cache_creation_5m_input_tokens=0,
        cache_creation_1h_input_tokens=0,
        thinking_tokens=0,
        web_search_requests=0,
        web_fetch_requests=0,
        estimated_cost_usd=cost,
    )


def test_summary_separates_known_and_complete_costs() -> None:
    incomplete = record_api_usage(
        _anthropic(server_tool_use={"code_execution_requests": 1}),
        agent_name="a",
        default_model="claude-opus-5-5",
    )
    complete = record_api_usage(_anthropic(), agent_name="a", default_model="claude-opus-5-5")
    assert incomplete is not None and complete is not None
    summary = summarize_usage([complete, incomplete, _legacy_record(Decimal("1")), _legacy_record(None)])
    assert summary.estimated_cost_usd is None and not summary.cost_complete
    assert summary.unpriced_requests == 2
    assert summary.known_cost_usd == Decimal("9")  # 4 + 4 (known part) + 1 + 0
    assert summary.missing_cost_components == ("fee:code_execution_request", "price")
    assert summarize_usage([complete, _legacy_record(Decimal("1"))]).estimated_cost_usd == Decimal("5")


def test_usage_records_serialize_and_old_records_stay_readable() -> None:
    record = record_api_usage(_anthropic(speed="fast"), agent_name="a", default_model="claude-opus-5-5")
    assert record is not None
    data = json.loads(json.dumps(record.to_dict()))
    assert data["estimated_cost_usd"] == "8" and data["pricing_conditions"]["speed"] == "fast"
    assert UsageRecord.from_dict(data) == record
    old = {
        key: value
        for key, value in _legacy_record(Decimal("1")).to_dict().items()
        if key in UsageRecord.__dataclass_fields__
        and key
        not in {
            "cost",
            "requested_model",
            "pricing_version",
            "pricing_quality",
            "pricing_conditions",
            "known_cost_usd",
            "missing_cost_components",
        }
    }
    loaded = UsageRecord.from_dict(old)
    assert loaded.cost is None and loaded.known_cost == Decimal("1") and loaded.cost_complete
    with pytest.raises(ValueError, match="Unknown"):
        UsageRecord.from_dict({**data, "surcharge": "1"})


def test_token_usage_defaults_keep_old_adapters_working() -> None:
    record = record_token_usage(
        TokenUsage(input_tokens=M, output_tokens=0),
        agent_name="a",
        model="gpt-5",
        provider="openai",
        custom_pricing=ModelPricing.of("1", "2", "1", "1", "0.1"),
        batch=True,
    )
    assert record.estimated_cost_usd == Decimal("0.5") and record.pricing_source == "custom"


# ---- BudgetTracker ---------------------------------------------------------------------------


def test_budget_uses_the_tracker_resolver_before_and_after_requests() -> None:
    budget = BudgetTracker(
        FAKE_COST, on_unpriced="reject", price_resolver=PriceResolver([FAKE_PRICE], version="b1")
    )
    agent, fake = _fake_agent(budget, [text("one"), text("two")])
    agent.ask("first")
    assert budget.spent_usd == FAKE_COST and budget.records[0].pricing_version == "b1"
    with pytest.raises(BudgetExceededError, match="used up"):
        agent.ask("second")
    assert len(fake.requests) == 1
    unpriced = BudgetTracker(100, on_unpriced="reject", price_resolver=PriceResolver(version="empty"))
    agent, fake = _fake_agent(unpriced, [text("never")])
    with pytest.raises(UnpricedModelError):
        agent.ask("x")
    assert fake.requests == []


def test_stale_prices_are_rejected_before_the_request() -> None:
    old = FAKE_PRICE.with_changes(checked_at=datetime(2025, 1, 1, tzinfo=UTC))
    budget = BudgetTracker(
        100,
        price_resolver=PriceResolver([old], version="old"),
        price_policy=PricePolicy(max_age=timedelta(days=90)),
        on_price_issue="reject",
    )
    agent, fake = _fake_agent(budget, [text("never")])
    with pytest.raises(UnreliablePriceError, match="stale"):
        agent.ask("x")
    assert fake.requests == [] and budget.records == ()


def test_unaccepted_quality_warns_once_per_model() -> None:
    estimated = FAKE_PRICE.with_changes(quality="estimated")
    budget = BudgetTracker(
        1000,
        price_resolver=PriceResolver([estimated], version="est"),
        price_policy=PricePolicy(accepted_quality={"verified", "contract"}),
    )
    agent, _ = _fake_agent(budget, [text("a"), text("b")])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        agent.ask("one")
        agent.ask("two")
    issues = [w for w in caught if issubclass(w.category, UnreliablePriceWarning)]
    assert len(issues) == 1 and "estimated" in str(issues[0].message)
    assert budget.spent_usd == 2 * FAKE_COST


@pytest.mark.filterwarnings("ignore::orkexa.dynamic.UnpricedUsageWarning")
def test_price_issue_found_after_the_request_is_raised_after_recording() -> None:
    """The requested model passes the pre-check; the billed price fails the policy."""
    budget = BudgetTracker(
        1000,
        price_policy=PricePolicy(allow_incomplete=False),
        on_price_issue="reject",
        price_resolver=PriceResolver([FAKE_PRICE], version="p"),
    )

    def with_fee(request: object) -> Any:
        response = text("x")
        response.usage = TokenUsage(input_tokens=10, output_tokens=5, units=(("web_search_call", 1),))
        return response

    agent, fake = _fake_agent(budget, [with_fee])
    with pytest.raises(UnreliablePriceError, match="incomplete"):
        agent.ask("x")
    assert len(fake.requests) == 1 and len(budget.records) == 1
    assert budget.spent_usd == FAKE_COST  # the known part counts against the budget


def test_incomplete_costs_count_their_known_part_and_warn() -> None:
    budget = BudgetTracker(1000, price_resolver=PriceResolver([FAKE_PRICE], version="p"))

    def with_fee(request: object) -> Any:
        response = text("x")
        response.usage = TokenUsage(input_tokens=10, output_tokens=5, units=(("web_search_call", 1),))
        return response

    agent, _ = _fake_agent(budget, [with_fee])
    with pytest.warns(UnpricedUsageWarning, match="incomplete"):
        agent.ask("x")
    assert budget.spent_usd == FAKE_COST and len(budget.unpriced_records) == 1
    strict = BudgetTracker(
        1000, on_unpriced="reject", price_resolver=PriceResolver([FAKE_PRICE], version="p")
    )
    agent, fake = _fake_agent(strict, [with_fee])
    with pytest.raises(UnpricedModelError, match="recorded"):
        agent.ask("x")
    assert len(strict.records) == 1 and len(fake.requests) == 1


def test_budget_defaults_are_unchanged() -> None:
    budget = BudgetTracker()
    assert budget.on_unpriced == "ignore" and budget.on_price_issue == "warn"
    assert budget.price_policy == PricePolicy() and budget.price_resolver is None
    assert BudgetTracker(5).on_unpriced == "warn"
    with pytest.raises(ValueError):
        BudgetTracker(on_price_issue="panic")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        UsageTracker(price_resolver="prices")  # type: ignore[arg-type]


def test_custom_trackers_that_override_before_request_still_work() -> None:
    seen: list[bool] = []

    class Legacy(UsageTracker):
        def before_request(self, *, agent: str, provider: str, model: str, priced: bool) -> None:
            seen.append(priced)

    tracker = Legacy(price_resolver=PriceResolver([FAKE_PRICE], version="p"))
    agent, _ = _fake_agent(tracker, [text("a")])
    agent.ask("x")
    unpriced, _ = _fake_agent(Legacy(), [text("b")])
    unpriced.ask("y")
    assert seen == [True, False]


def test_metered_usage_from_external_systems() -> None:
    """An application prices its own usage data without any agent."""
    resolver = PriceResolver(
        PriceSnapshot.from_json(PriceSnapshot("ext-1", (_price("openai", "gpt-5"),)).to_json())
    )
    cost = resolver.calculate(
        "openai", "gpt-5-2025-08-07", MeteredUsage(input_tokens=M), requested_model="gpt-5"
    )
    assert cost.total == Decimal("1") and cost.references[0].matched_by == "requested"
