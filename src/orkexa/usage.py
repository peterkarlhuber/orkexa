"""Token usage records, aggregation and cost estimation."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, fields
from datetime import datetime
from decimal import Decimal
import threading
from typing import TYPE_CHECKING, Any, TypedDict

from .pricing import CostBreakdown, MeteredUsage, PriceResolver, PricingInput, pricing_basis

if TYPE_CHECKING:
    from .core.types import TokenUsage
    from .pricing import PriceLookup

__all__ = [
    "Reservation",
    "UsageRecord",
    "UsageSummary",
    "UsageTracker",
    "current_workflow",
    "record_api_usage",
    "record_token_usage",
    "summarize_usage",
    "workflow",
]


_RELEASED = threading.Condition()
"""Notified whenever a budget reservation is released (waiters re-check their trackers)."""


@dataclass(frozen=True)
class UsageRecord:
    """Actual reported API tokens and estimated USD cost of *one* model request.

    Counts are normalized across providers so nothing is counted twice:
    ``input_tokens`` excludes cache reads and writes, ``output_tokens`` includes
    reasoning tokens (``thinking_tokens`` is the reasoning share of it).
    ``estimated_cost_usd`` is ``None`` (never 0) when no price is known.

    The fields after ``request_id`` were added by Orkexa: ``provider`` (registered
    provider name), ``turn_id`` (one id per ``ask()`` of the agent that sent the
    request), ``workflow`` (the label of the enclosing :func:`workflow`, e.g. one
    :meth:`~orkexa.dynamic.Orchestrator.ask` including every delegated agent) and
    ``pricing_source`` (the price's source: ``"builtin"``, ``"registered"``, ``"custom"``
    for an agent's ``pricing=``, a catalog name from your own snapshot, or ``None``
    when no price was found).

    The pricing fields describe the price basis: ``requested_model`` (the id the agent
    was configured with; ``model`` is the id the API reported), ``pricing_version``
    (version of the :class:`~orkexa.pricing.PriceSnapshot` used),
    ``pricing_quality`` (the lowest :data:`~orkexa.pricing.PriceQuality` involved),
    ``pricing_conditions`` (the tariff options applied, e.g. ``service_tier=batch``),
    ``known_cost_usd`` (sum of the components that could be priced; equals
    ``estimated_cost_usd`` when the cost is complete), ``missing_cost_components``
    (what could not be priced) and ``cost`` (the itemized
    :class:`~orkexa.pricing.CostBreakdown`). Records created before these fields
    existed simply have their defaults.
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
    requested_model: str | None = None
    pricing_version: str | None = None
    pricing_quality: str | None = None
    pricing_conditions: tuple[tuple[str, str], ...] = ()
    known_cost_usd: Decimal | None = None
    missing_cost_components: tuple[str, ...] = ()
    cost: CostBreakdown | None = None

    @property
    def total_input_tokens(self) -> int:
        return self.input_tokens + self.cache_read_input_tokens + self.cache_creation_input_tokens

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.output_tokens

    @property
    def cost_complete(self) -> bool:
        """Whether ``estimated_cost_usd`` is a complete cost (not ``None``)."""
        return self.estimated_cost_usd is not None

    @property
    def known_cost(self) -> Decimal:
        """The known part of the cost in USD (``0`` when nothing is known)."""
        if self.estimated_cost_usd is not None:
            return self.estimated_cost_usd
        return self.known_cost_usd if self.known_cost_usd is not None else Decimal("0")

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible form (amounts as decimal strings) for external ledgers."""
        data: dict[str, Any] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, Decimal):
                value = str(value)
            elif isinstance(value, CostBreakdown):
                value = value.to_dict()
            elif item.name == "pricing_conditions":
                value = dict(value)
            elif isinstance(value, tuple):
                value = list(value)
            data[item.name] = value
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> UsageRecord:
        """Inverse of :meth:`to_dict`; keys of newer Orkexa versions are rejected, missing ones default."""
        known = {item.name for item in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown UsageRecord fields: {sorted(unknown)}")
        values = dict(data)
        for name in ("estimated_cost_usd", "known_cost_usd"):
            if values.get(name) is not None:
                values[name] = Decimal(str(values[name]))
        if values.get("cost") is not None:
            values["cost"] = CostBreakdown.from_dict(values["cost"])
        values["pricing_conditions"] = tuple(sorted(dict(values.get("pricing_conditions") or {}).items()))
        values["missing_cost_components"] = tuple(values.get("missing_cost_components") or ())
        return cls(**values)


@dataclass(frozen=True)
class UsageSummary:
    """Aggregated token counters and costs.

    ``estimated_cost_usd`` is the complete total, or ``None`` if any request has an
    unknown or incomplete cost (``unpriced_requests`` counts them). ``known_cost_usd``
    is the sum of everything that could be priced: a lower bound when the total is
    unknown. ``missing_cost_components`` lists what could not be priced.
    """

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
    known_cost_usd: Decimal = Decimal("0")
    missing_cost_components: tuple[str, ...] = ()

    @property
    def total_input_tokens(self) -> int:
        return self.input_tokens + self.cache_read_input_tokens + self.cache_creation_input_tokens

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.output_tokens

    @property
    def cost_complete(self) -> bool:
        """Whether every request's cost is known, so ``estimated_cost_usd`` is the total."""
        return self.estimated_cost_usd is not None


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
    missing: dict[str, None] = {}
    for record in records:
        if record.estimated_cost_usd is None:
            missing.update(dict.fromkeys(record.missing_cost_components or ("price",)))
    return UsageSummary(
        requests=len(records),
        estimated_cost_usd=total_cost,
        unpriced_requests=unpriced,
        known_cost_usd=sum((record.known_cost for record in records), Decimal("0")),
        missing_cost_components=tuple(missing),
        **totals,
    )


_WORKFLOW: ContextVar[str | None] = ContextVar("orkexa_workflow", default=None)


@contextmanager
def workflow(label: str) -> Iterator[str]:
    """Label the usage of every request sent inside the ``with`` block.

    The label follows the context into tool threads and asyncio tasks, so agents
    delegated as tools are included. An inner block replaces the label for its
    duration::

        with workflow("nightly-report"):
            agent.ask(...)
        tracker.usage_by_workflow()["nightly-report"]
    """
    if not isinstance(label, str) or not label:
        raise ValueError("workflow label must be a nonempty string")
    token = _WORKFLOW.set(label)
    try:
        yield label
    finally:
        _WORKFLOW.reset(token)


def current_workflow() -> str | None:
    """The label of the enclosing :func:`workflow` block, if any."""
    return _WORKFLOW.get()


@dataclass(eq=False)
class Reservation:
    """Money held back for one model request until its usage is recorded.

    :meth:`UsageTracker.try_reserve` returns it before the request; the agent releases
    it with :meth:`UsageTracker.release` once the request's usage is recorded (or the
    request failed). ``amount`` is ``None`` when the request's cost has no upper bound.
    """

    tracker: UsageTracker
    amount: Decimal | None
    agent: str
    parent: Reservation | None = None
    released: bool = False

    @property
    def bounded(self) -> bool:
        return self.amount is not None


class UsageTracker:
    """Shared ledger for several agents, including agents delegated as tools.

    Args:
        price_resolver: Prices every request of the agents that use this tracker and
            set no resolver of their own (``pricing=``). ``None``: the process-wide
            :func:`~orkexa.pricing.default_resolver`.
    """

    def __init__(self, *, price_resolver: PriceResolver | None = None) -> None:
        if price_resolver is not None and not isinstance(price_resolver, PriceResolver):
            raise TypeError("price_resolver must be a PriceResolver")
        self._records: list[UsageRecord] = []
        self.price_resolver = price_resolver

    def add(self, record: UsageRecord) -> None:
        self._records.append(record)

    def before_request(self, *, agent: str, provider: str, model: str, priced: bool) -> None:
        """Called before every model request; budget trackers raise here to stop early."""

    def check_request(self, *, agent: str, provider: str, model: str, lookup: PriceLookup) -> None:
        """Called before every model request with the price lookup of the pinned basis.

        The default delegates to :meth:`before_request` (``priced=lookup.found``), so
        trackers that override only :meth:`before_request` keep working.
        """
        self.before_request(agent=agent, provider=provider, model=model, priced=lookup.found)

    def check(self) -> None:
        """Raise if a limit is exceeded; a plain tracker has none (see ``BudgetTracker``)."""

    def try_reserve(
        self, *, agent: str, provider: str, model: str, amount: Decimal | None
    ) -> Reservation | None:
        """Hold back ``amount`` USD for one request; ``None`` means: wait and try again.

        A plain tracker has no limit and admits every request at once. Budget trackers
        raise when the request cannot fit even after the requests in flight finished.
        """
        return Reservation(self, amount, agent)

    def release(self, reservation: Reservation) -> None:
        """Give back what :meth:`try_reserve` held; releasing twice does nothing."""
        reservation.released = True

    def wait_for_release(self, timeout: float) -> None:
        """Block until any reservation is released, or ``timeout`` seconds passed."""
        with _RELEASED:
            _RELEASED.wait(timeout)

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

    def usage_by_turn(self) -> dict[str | None, UsageSummary]:
        """Per-turn summaries, keyed by ``turn_id`` (``None``: requests outside a turn, e.g. batches)."""
        return self._group(lambda record: record.turn_id)

    def usage_by_workflow(self) -> dict[str | None, UsageSummary]:
        """Per-workflow summaries, keyed by label (``None``: requests outside any :func:`workflow`)."""
        return self._group(lambda record: record.workflow)

    def _group(self, key: Any) -> dict[Any, UsageSummary]:
        grouped: dict[Any, list[UsageRecord]] = {}
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


def _metered(
    counts: dict[str, int],
    *,
    conditions: Mapping[str, str],
    units: Mapping[str, int] | None = None,
    token_details: Mapping[str, int] | None = None,
) -> MeteredUsage:
    split = counts["cache_creation_5m_input_tokens"] + counts["cache_creation_1h_input_tokens"]
    return MeteredUsage(
        input_tokens=counts["input_tokens"],
        output_tokens=counts["output_tokens"],
        cache_read_input_tokens=counts["cache_read_input_tokens"],
        cache_write_input_tokens=max(counts["cache_creation_input_tokens"], split),
        cache_write_5m_input_tokens=counts["cache_creation_5m_input_tokens"],
        cache_write_1h_input_tokens=counts["cache_creation_1h_input_tokens"],
        token_details=token_details or {},
        units=units or {},
        conditions=conditions,
    )


_QUALITY_RANK = {"verified": 3, "contract": 3, "unverified": 1, "estimated": 0}


def _note(cost: CostBreakdown, unpriced_models: list[str], unpriced_label: str) -> str | None:
    if cost.complete and cost.currency == "USD":
        return None
    notes: list[str] = []
    other: list[str] = []
    for item in cost.missing:
        if item == "price":
            if unpriced_models:
                notes.append(unpriced_label)
        elif item == "cache_write_ttl":
            notes.append("Cache write TTL breakdown incomplete; cannot price this request reliably")
        elif item.startswith("tariff:"):
            notes.append(f"Nonstandard tier/speed/data residency requires adjusted pricing ({item[7:]})")
        else:
            other.append(item)
    if other:
        notes.append(f"Cost incomplete; unknown: {', '.join(other)}")
    if cost.currency != "USD":
        notes.append(f"Priced in {cost.currency}; no USD amount is computed")
    return "; ".join(dict.fromkeys(notes)) or None


def _priced_record(
    cost: CostBreakdown,
    *,
    basis_version: str | None,
    note: str | None,
    **values: Any,
) -> UsageRecord:
    usd = cost.currency == "USD"
    references = cost.references
    source = None
    if references:
        source = "custom" if any(r.matched_by == "override" for r in references) else references[0].source
    quality = min((r.quality for r in references), key=lambda q: _QUALITY_RANK.get(q, 0), default=None)
    conditions: dict[str, str] = {}
    for reference in references:
        conditions.update(dict(reference.conditions))
    missing = cost.missing if usd else (*cost.missing, "currency")
    return UsageRecord(
        estimated_cost_usd=cost.total if usd else None,
        pricing_note=note,
        pricing_source=source,
        pricing_version=basis_version,
        pricing_quality=quality,
        pricing_conditions=tuple(sorted(conditions.items())),
        known_cost_usd=cost.known_total if usd else None,
        missing_cost_components=tuple(dict.fromkeys(missing)),
        cost=cost,
        **values,
    )


def record_api_usage(
    response: Any,
    *,
    agent_name: str,
    default_model: str,
    custom_pricing: PricingInput = None,
    batch: bool = False,
    provider: str | None = "anthropic",
    turn_id: str | None = None,
    at: datetime | None = None,
) -> UsageRecord | None:
    """Build a :class:`UsageRecord` from an Anthropic ``response.usage``.

    Uses the token counts the API actually reported; never reconstructs them
    from text. Returns ``None`` if the response carries no usage block.

    ``custom_pricing`` is anything an agent accepts as ``pricing=`` (a
    :class:`~orkexa.ModelPricing`, :class:`~orkexa.pricing.ModelPrice`,
    :class:`~orkexa.pricing.PriceResolver`, :class:`~orkexa.pricing.PriceSnapshot` or a
    pinned :class:`~orkexa.pricing.PricingBasis`); ``None`` uses the default resolver.
    ``at`` is the time prices must be valid at (default: now).

    When ``usage.iterations`` contains more than plain sampling iterations
    (server-side compaction, a refusal fallback, an advisor), tokens and cost
    are summed over the iterations, each priced at its own model, because
    compaction tokens are not part of the top-level counters. Batch requests
    (``service_tier == "batch"`` or ``batch=True``) use the price's batch tariff
    (50 % for the built-in prices); ``speed`` and ``inference_geo`` select the
    fast-mode and data-residency tariffs. Server-tool fees (web search) are priced
    with the response's model; code execution without web search or web fetch is
    billed per container hour, which usage does not report, so it makes the cost
    incomplete.
    """
    usage = _get_field(response, "usage")
    if usage is None:
        return None
    provider_name = provider or "anthropic"
    basis = pricing_basis(custom_pricing)

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
    code_runs = _int_field(server_tools, "code_execution_requests")

    conditions: dict[str, str] = {}
    service_tier = "batch" if batch else _get_field(usage, "service_tier")
    for dimension, value in (
        ("service_tier", service_tier),
        ("speed", _get_field(usage, "speed")),
        ("inference_geo", _get_field(usage, "inference_geo")),
    ):
        if isinstance(value, str) and value:
            conditions[dimension] = value
    units = {"web_search_request": searches, "web_fetch_request": fetches}
    if code_runs and not (searches or fetches):
        units["code_execution_request"] = code_runs

    parts = [
        basis.calculate(provider_name, entry_model, _metered(entry, conditions=conditions), at=at)
        for entry, entry_model in entries
    ]
    unpriced = sorted({m for (_, m), part in zip(entries, parts, strict=True) if "price" in part.missing})
    if any(units.values()):
        fee_usage = MeteredUsage(units=units, conditions=conditions)
        parts.append(basis.calculate(provider_name, model, fee_usage, at=at))
    cost = CostBreakdown.combine(parts)
    note = _note(cost, unpriced, f"No known pricing for model {', '.join(map(repr, unpriced))}")

    request_id = getattr(response, "_request_id", None)
    return _priced_record(
        cost,
        basis_version=basis.version,
        note=note,
        agent=agent_name,
        model=model,
        thinking_tokens=thinking,
        web_search_requests=searches,
        web_fetch_requests=fetches,
        request_id=request_id if isinstance(request_id, str) else None,
        provider=provider,
        turn_id=turn_id,
        requested_model=default_model,
        **_count_fields(counts),
    )


def record_token_usage(
    usage: TokenUsage,
    *,
    agent_name: str,
    model: str,
    provider: str,
    custom_pricing: PricingInput = None,
    request_id: str | None = None,
    turn_id: str | None = None,
    batch: bool = False,
    requested_model: str | None = None,
    at: datetime | None = None,
    conditions: Mapping[str, str] | None = None,
) -> UsageRecord:
    """Build a :class:`UsageRecord` from normalized counts (every provider but Anthropic).

    Prices come from ``custom_pricing`` (anything an agent accepts as ``pricing=``)
    or the default resolver (:func:`orkexa.pricing.register_pricing`); Orkexa ships
    no prices for other providers, so without them the cost is ``None``.
    The price is resolved for ``model`` (the id the API reported) first and then for
    ``requested_model`` (the id the agent was configured with, e.g. a deployment
    name). Cache writes without a lifetime are priced at the 5-minute cache-write
    rate. The reported ``service_tier`` (``"batch"`` with ``batch=True``) must be a
    tariff of the price; reported server-tool units and special token classes
    (audio) need a fee or rate, otherwise the cost is incomplete.
    ``conditions`` are further tariff conditions the provider reports (e.g.
    ``{"aws_region": "eu-central-1"}`` for Bedrock); like ``service_tier``, a value
    the price has no tariff option for makes the cost unknown.
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
    basis = pricing_basis(custom_pricing)
    reported: dict[str, str] = {}
    tier = "batch" if batch else getattr(usage, "service_tier", None)
    if isinstance(tier, str) and tier:
        reported["service_tier"] = tier
    reported.update(conditions or {})
    units = dict(getattr(usage, "units", ()) or ())
    if usage.web_search_requests:
        units["web_search_request"] = units.get("web_search_request", 0) + usage.web_search_requests
    if usage.web_fetch_requests:
        units["web_fetch_request"] = units.get("web_fetch_request", 0) + usage.web_fetch_requests
    metered = _metered(
        counts,
        conditions=reported,
        units=units,
        token_details=dict(getattr(usage, "token_details", ()) or ()),
    )
    cost = basis.calculate(provider, model, metered, requested_model=requested_model, at=at)
    unpriced = [model] if "price" in cost.missing else []
    note = _note(
        cost,
        unpriced,
        f"No known pricing for {provider}:{model}; set pricing= or "
        "orkexa.pricing.register_pricing() to estimate costs",
    )
    return _priced_record(
        cost,
        basis_version=basis.version,
        note=note,
        agent=agent_name,
        model=model,
        thinking_tokens=usage.thinking_tokens,
        web_search_requests=usage.web_search_requests,
        web_fetch_requests=usage.web_fetch_requests,
        request_id=request_id,
        provider=provider,
        turn_id=turn_id,
        requested_model=requested_model,
        **_count_fields(counts),
    )
