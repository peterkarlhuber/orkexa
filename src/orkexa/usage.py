"""Token usage records, aggregation and cost estimation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any, TypedDict

from .pricing import ModelPricing, resolve_pricing

if TYPE_CHECKING:
    from .core.types import TokenUsage

__all__ = [
    "UsageRecord",
    "UsageSummary",
    "UsageTracker",
    "record_api_usage",
    "record_token_usage",
    "summarize_usage",
]

_WEB_SEARCH_USD_PER_REQUEST = Decimal("0.01")  # $10 per 1,000 searches
_ONE_MILLION = Decimal("1000000")


@dataclass(frozen=True)
class UsageRecord:
    """Actual reported API tokens and estimated USD cost of *one* model request.

    Counts are normalized across providers so nothing is counted twice:
    ``input_tokens`` excludes cache reads and writes, ``output_tokens`` includes
    reasoning tokens (``thinking_tokens`` is the reasoning share of it).
    ``estimated_cost_usd`` is ``None`` (never 0) when no price is known.

    The fields after ``request_id`` were added by Orkexa: ``provider`` (registered
    provider name), ``turn_id`` (one id per ``ask()``), ``workflow`` (free label set by
    orchestration) and ``pricing_source`` (``"builtin"``, ``"registered"``, ``"custom"``
    or ``None`` when unpriced).
    """

    agent: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int
    cache_creation_5m_input_tokens: int
    cache_creation_1h_input_tokens: int
    thinking_tokens: int
    web_search_requests: int
    web_fetch_requests: int
    estimated_cost_usd: Decimal | None
    pricing_note: str | None = None
    request_id: str | None = None
    provider: str | None = None
    turn_id: str | None = None
    workflow: str | None = None
    pricing_source: str | None = None

    @property
    def total_input_tokens(self) -> int:
        return self.input_tokens + self.cache_read_input_tokens + self.cache_creation_input_tokens

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.output_tokens


@dataclass(frozen=True)
class UsageSummary:
    """Aggregated token counters; cost is None if any request has unknown pricing."""

    requests: int
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int
    cache_creation_5m_input_tokens: int
    cache_creation_1h_input_tokens: int
    thinking_tokens: int
    web_search_requests: int
    web_fetch_requests: int
    estimated_cost_usd: Decimal | None
    unpriced_requests: int

    @property
    def total_input_tokens(self) -> int:
        return self.input_tokens + self.cache_read_input_tokens + self.cache_creation_input_tokens

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.output_tokens


_SUMMED_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "cache_creation_5m_input_tokens",
    "cache_creation_1h_input_tokens",
    "thinking_tokens",
    "web_search_requests",
    "web_fetch_requests",
)


def summarize_usage(records: Iterable[UsageRecord]) -> UsageSummary:
    """Aggregate individual request records into one :class:`UsageSummary`."""
    records = list(records)
    totals = {key: sum(getattr(record, key) for record in records) for key in _SUMMED_FIELDS}
    costs = [record.estimated_cost_usd for record in records]
    unpriced = sum(cost is None for cost in costs)
    total_cost = None if unpriced else sum((c for c in costs if c is not None), Decimal("0"))
    return UsageSummary(
        requests=len(records),
        estimated_cost_usd=total_cost,
        unpriced_requests=unpriced,
        **totals,
    )


class UsageTracker:
    """Shared ledger for several agents, including agents delegated as tools."""

    def __init__(self) -> None:
        self._records: list[UsageRecord] = []

    def add(self, record: UsageRecord) -> None:
        self._records.append(record)

    def before_request(self, *, agent: str, provider: str, model: str, priced: bool) -> None:
        """Called before every model request; budget trackers raise here to stop early."""

    @property
    def records(self) -> tuple[UsageRecord, ...]:
        return tuple(self._records)

    @property
    def usage(self) -> UsageSummary:
        return summarize_usage(self._records)

    def usage_by_agent(self) -> dict[str, UsageSummary]:
        """Per-agent summaries, keyed by agent name in order of first appearance."""
        grouped: dict[str, list[UsageRecord]] = {}
        for record in self._records:
            grouped.setdefault(record.agent, []).append(record)
        return {agent: summarize_usage(records) for agent, records in grouped.items()}

    def usage_by_provider(self) -> dict[str, UsageSummary]:
        """Per-provider summaries (records without a provider are listed under ``"unknown"``)."""
        return self._group(lambda record: record.provider or "unknown")

    def usage_by_model(self) -> dict[str, UsageSummary]:
        """Per-model summaries, keyed ``"provider:model"``."""
        return self._group(lambda record: f"{record.provider or 'unknown'}:{record.model}")

    def _group(self, key: Any) -> dict[str, UsageSummary]:
        grouped: dict[str, list[UsageRecord]] = {}
        for record in self._records:
            grouped.setdefault(key(record), []).append(record)
        return {name: summarize_usage(records) for name, records in grouped.items()}

    def reset(self) -> None:
        self._records.clear()


def _get_field(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _int_field(obj: Any, name: str) -> int:
    value = _get_field(obj, name, 0)
    return int(value) if value is not None else 0


def _token_cost(prices: ModelPricing, counts: dict[str, int]) -> Decimal:
    # Thinking tokens are INCLUDED in output_tokens; do not charge twice.
    return (
        counts["input_tokens"] * prices.input
        + counts["output_tokens"] * prices.output
        + counts["cache_read_input_tokens"] * prices.cache_read
        + counts["cache_creation_5m_input_tokens"] * prices.cache_write_5m
        + counts["cache_creation_1h_input_tokens"] * prices.cache_write_1h
    ) / _ONE_MILLION


def _token_counts(usage: Any) -> dict[str, int]:
    creation = _get_field(usage, "cache_creation")
    return {
        "input_tokens": _int_field(usage, "input_tokens"),
        "output_tokens": _int_field(usage, "output_tokens"),
        "cache_read_input_tokens": _int_field(usage, "cache_read_input_tokens"),
        "cache_creation_input_tokens": _int_field(usage, "cache_creation_input_tokens"),
        "cache_creation_5m_input_tokens": _int_field(creation, "ephemeral_5m_input_tokens"),
        "cache_creation_1h_input_tokens": _int_field(creation, "ephemeral_1h_input_tokens"),
    }


class _Counts(TypedDict):
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int
    cache_creation_5m_input_tokens: int
    cache_creation_1h_input_tokens: int


def _count_fields(counts: dict[str, int]) -> _Counts:
    return _Counts(
        input_tokens=counts["input_tokens"],
        output_tokens=counts["output_tokens"],
        cache_read_input_tokens=counts["cache_read_input_tokens"],
        cache_creation_input_tokens=counts["cache_creation_input_tokens"],
        cache_creation_5m_input_tokens=counts["cache_creation_5m_input_tokens"],
        cache_creation_1h_input_tokens=counts["cache_creation_1h_input_tokens"],
    )


def _total_input(counts: dict[str, int]) -> int:
    return counts["input_tokens"] + counts["cache_read_input_tokens"] + counts["cache_creation_input_tokens"]


def record_api_usage(
    response: Any,
    *,
    agent_name: str,
    default_model: str,
    custom_pricing: ModelPricing | None = None,
    batch: bool = False,
    provider: str | None = "anthropic",
    turn_id: str | None = None,
) -> UsageRecord | None:
    """Build a :class:`UsageRecord` from an Anthropic ``response.usage``.

    Uses the token counts the API actually reported; never reconstructs them
    from text. Returns ``None`` if the response carries no usage block.

    When ``usage.iterations`` contains more than plain sampling iterations
    (server-side compaction, a refusal fallback, an advisor), tokens and cost
    are summed over the iterations, each priced at its own model, because
    compaction tokens are not part of the top-level counters. Batch requests
    (``service_tier == "batch"`` or ``batch=True``) are priced at 50 %.
    """
    usage = _get_field(response, "usage")
    if usage is None:
        return None

    model = _get_field(response, "model", default_model) or default_model
    iterations = list(_get_field(usage, "iterations") or [])
    if not any(_get_field(item, "type") != "message" for item in iterations):
        iterations = []
    entries = [(_token_counts(item), _get_field(item, "model") or model) for item in iterations]
    if entries:
        counts = {key: sum(c[key] for c, _ in entries) for key in entries[0][0]}
    else:
        counts = _token_counts(usage)
        entries = [(counts, model)]
    thinking = _int_field(_get_field(usage, "output_tokens_details"), "thinking_tokens")
    server_tools = _get_field(usage, "server_tool_use")
    searches = _int_field(server_tools, "web_search_requests")
    fetches = _int_field(server_tools, "web_fetch_requests")

    note: str | None = None
    cost: Decimal | None = None
    priced = [
        (entry, custom_pricing or resolve_pricing(entry_model, _total_input(entry)), entry_model)
        for entry, entry_model in entries
    ]
    unknown = sorted({entry_model for _, prices, entry_model in priced if prices is None})
    service_tier = "batch" if batch else _get_field(usage, "service_tier")
    speed = _get_field(usage, "speed")
    inference_geo = _get_field(usage, "inference_geo")
    breakdown_ok = all(
        entry["cache_creation_5m_input_tokens"] + entry["cache_creation_1h_input_tokens"]
        == entry["cache_creation_input_tokens"]
        for entry, _, _ in priced
    )
    if unknown:
        note = f"No known pricing for model {', '.join(map(repr, unknown))}"
    elif not breakdown_ok:
        note = "Cache write TTL breakdown incomplete; cannot price this request reliably"
    elif (
        service_tier not in (None, "standard", "batch")
        or speed not in (None, "standard")
        or inference_geo not in (None, "global")
    ):
        note = "Nonstandard tier/speed/data residency requires adjusted pricing"
    else:
        token_cost = sum(
            (_token_cost(prices, entry) for entry, prices, _ in priced if prices is not None), Decimal("0")
        )
        if service_tier == "batch":
            token_cost /= 2
        # Web search costs $10 / 1,000 searches; web fetch has no extra fee.
        cost = token_cost + searches * _WEB_SEARCH_USD_PER_REQUEST

    request_id = getattr(response, "_request_id", None)
    return UsageRecord(
        agent=agent_name,
        model=model,
        thinking_tokens=thinking,
        web_search_requests=searches,
        web_fetch_requests=fetches,
        estimated_cost_usd=cost,
        pricing_note=note,
        request_id=request_id if isinstance(request_id, str) else None,
        provider=provider,
        turn_id=turn_id,
        pricing_source=None if cost is None else ("custom" if custom_pricing else "builtin"),
        **_count_fields(counts),
    )


def record_token_usage(
    usage: TokenUsage,
    *,
    agent_name: str,
    model: str,
    provider: str,
    custom_pricing: ModelPricing | None = None,
    request_id: str | None = None,
    turn_id: str | None = None,
    batch: bool = False,
) -> UsageRecord:
    """Build a :class:`UsageRecord` from normalized counts (every provider but Anthropic).

    Prices come from ``custom_pricing`` or :func:`orkexa.pricing.register_pricing`;
    Orkexa ships no prices for other providers, so without them the cost is ``None``.
    Cache writes are priced at the 5-minute cache-write rate.
    """
    counts = {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_input_tokens": usage.cache_read_input_tokens,
        "cache_creation_input_tokens": usage.cache_creation_input_tokens,
        "cache_creation_5m_input_tokens": usage.cache_creation_5m_input_tokens
        or (usage.cache_creation_input_tokens - usage.cache_creation_1h_input_tokens),
        "cache_creation_1h_input_tokens": usage.cache_creation_1h_input_tokens,
    }
    prices = custom_pricing or resolve_pricing(model, _total_input(counts), provider=provider)
    cost: Decimal | None = None
    note: str | None = None
    if prices is None:
        note = (
            f"No known pricing for {provider}:{model}; set pricing= or "
            "orkexa.pricing.register_pricing() to estimate costs"
        )
    else:
        cost = _token_cost(prices, counts)
        if batch:
            cost /= 2
    return UsageRecord(
        agent=agent_name,
        model=model,
        thinking_tokens=usage.thinking_tokens,
        web_search_requests=usage.web_search_requests,
        web_fetch_requests=usage.web_fetch_requests,
        estimated_cost_usd=cost,
        pricing_note=note,
        request_id=request_id,
        provider=provider,
        turn_id=turn_id,
        pricing_source=None if prices is None else ("custom" if custom_pricing else "registered"),
        **_count_fields(counts),
    )
