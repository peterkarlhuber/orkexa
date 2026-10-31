"""Limits for dynamically created tools and agents, and a budget-enforcing tracker."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from decimal import Decimal
import threading
from typing import Literal, get_args
import warnings

from ..core.types import Effort
from ..pricing import PriceLookup, PricePolicy, PriceResolver
from ..usage import _RELEASED, Reservation, UsageRecord, UsageTracker
from .errors import (
    BudgetExceededError,
    UnpricedModelError,
    UnpricedUsageWarning,
    UnreliablePriceError,
    UnreliablePriceWarning,
)

__all__ = [
    "RESERVED_TOOL_NAMES",
    "BudgetTracker",
    "DynamicPolicy",
    "OversizedPolicy",
    "PriceIssuePolicy",
    "UnpricedPolicy",
    "clamp_effort",
]

_EFFORT_ORDER: tuple[Effort, ...] = get_args(Effort)

#: Tools that let an agent create or call other agents or tools. Generated
#: agents only get them when ``DynamicPolicy.allow_agent_delegation`` is set.
RESERVED_TOOL_NAMES = frozenset(
    {"list_agents", "create_agent", "run_agent", "find_tools", "create_tool", "load_tool"}
)


def clamp_effort(effort: Effort, maximum: Effort) -> Effort:
    """Return ``effort``, lowered to ``maximum`` if it is higher."""
    if _EFFORT_ORDER.index(effort) > _EFFORT_ORDER.index(maximum):
        return maximum
    return effort


@dataclass(frozen=True, init=False)
class DynamicPolicy:
    """Everything a model may *not* decide on its own.

    The Agent Architect and the Coding Agent only make proposals; this policy,
    set by the developer, decides what is accepted.

    Attributes:
        allowed_models: Models a generated agent may use. Entries may be
            provider-qualified (``"openai:gpt-5.5"``); a bare name means the
            provider it resolves to (``"claude-opus-5-5"`` is Anthropic).
        max_effort: Highest effort a generated agent may use (higher is lowered).
        max_tokens: Upper bound for a generated agent's ``max_tokens`` (higher is lowered).
        max_tool_rounds: Upper bound for a generated agent's ``max_tool_rounds``.
        allowed_agent_tools: If set, generated agents may only use these registered
            tools, and :meth:`ToolBuilder.as_tools`' ``load_tool`` only loads these.
        allow_agent_delegation: Whether generated agents may receive the reserved
            orchestration tools (``run_agent`` etc.).
        max_depth: Maximum nesting of ``run_agent`` delegations.
        max_agents: Maximum number of active agents in the agent registry.
        max_generated_tools: Maximum number of tools one ToolBuilder may generate.
        budget_usd: Shared USD budget for all agents, or ``None`` for no limit.
    """

    allowed_models: tuple[str, ...] = ("claude-sonnet-5-5", "claude-opus-5-5")
    max_effort: Effort = "high"
    max_tokens: int = 16384
    max_tool_rounds: int = 10
    allowed_agent_tools: frozenset[str] | None = None
    allow_agent_delegation: bool = False
    max_depth: int = 2
    max_agents: int = 10
    max_generated_tools: int = 10
    budget_usd: Decimal | None = None

    def __init__(
        self,
        *,
        allowed_models: Iterable[str] = ("claude-sonnet-5-5", "claude-opus-5-5"),
        max_effort: Effort = "high",
        max_tokens: int = 16384,
        max_tool_rounds: int = 10,
        allowed_agent_tools: Iterable[str] | None = None,
        allow_agent_delegation: bool = False,
        max_depth: int = 2,
        max_agents: int = 10,
        max_generated_tools: int = 10,
        budget_usd: Decimal | float | str | None = None,
    ) -> None:
        models = tuple(allowed_models)
        if not models:
            raise ValueError("allowed_models must not be empty")
        if max_effort not in _EFFORT_ORDER:
            raise ValueError(f"Invalid max_effort: {max_effort!r}")
        if max_tokens < 256 or max_tool_rounds < 0 or max_depth < 0:
            raise ValueError("max_tokens must be >= 256, max_tool_rounds and max_depth >= 0")
        if max_agents < 0 or max_generated_tools < 0:
            raise ValueError("max_agents and max_generated_tools must be >= 0")
        budget = None if budget_usd is None else Decimal(str(budget_usd))
        if budget is not None and budget < 0:
            raise ValueError("budget_usd must be nonnegative")
        set_ = object.__setattr__
        set_(self, "allowed_models", models)
        set_(self, "max_effort", max_effort)
        set_(self, "max_tokens", max_tokens)
        set_(self, "max_tool_rounds", max_tool_rounds)
        set_(
            self,
            "allowed_agent_tools",
            None if allowed_agent_tools is None else frozenset(allowed_agent_tools),
        )
        set_(self, "allow_agent_delegation", allow_agent_delegation)
        set_(self, "max_depth", max_depth)
        set_(self, "max_agents", max_agents)
        set_(self, "max_generated_tools", max_generated_tools)
        set_(self, "budget_usd", budget)


UnpricedPolicy = Literal["ignore", "warn", "reject"]
PriceIssuePolicy = Literal["ignore", "warn", "reject"]
OversizedPolicy = Literal["exclusive", "reject"]


class BudgetTracker(UsageTracker):
    """A :class:`~orkexa.UsageTracker` that enforces a USD limit across providers.

    Every request is recorded first; if the summed cost then exceeds
    ``limit_usd``, :class:`BudgetExceededError` is raised. Because the agent
    records usage right after each API response, this aborts a running tool
    loop. Before each request the tracker refuses to start when the budget is
    already used up (``spent_usd >= limit_usd``, so a zero budget sends nothing).

    **Reservations.** Before every request the agent reserves an upper bound of its
    cost (:func:`orkexa.core.cost_bound.request_cost_bound`: input size and
    ``max_tokens`` times the highest rates) and releases it once the usage is
    recorded. A request is admitted only while ``spent + reserved + bound <= limit``;
    if it does not fit because other requests are in flight, it waits for them. So
    requests that run at the same time, also of agents that share the tracker or of
    child trackers (``parent=``), cannot together spend more than the limit.

    A request whose bound exceeds the rest of the budget even with nothing in flight
    is *oversized*. ``on_oversized`` decides: ``"exclusive"`` (default) admits it
    alone, while no other request runs, so it can overshoot the limit by itself as a
    sequential run always could (counted in :attr:`oversized_requests`; the limit is
    then no hard maximum); ``"reject"`` refuses it with :class:`BudgetExceededError`
    before it is sent, which makes the limit a hard maximum for bounded requests.

    Requests whose cost has **no** upper bound (no price, provider-native tools,
    images, binary documents) are admitted while the budget is not used up, but they
    are counted in :attr:`unbounded_requests` and :attr:`hard_maximum` becomes
    ``False``: the limit is then no hard maximum. A native batch is admitted as one
    request without a reservation, as before, so it can overshoot by all of its
    items. The limit is never a billing limit: use the provider's own spend limits for
    that.

    Requests of a model without a known price (no built-in price, no ``pricing=``,
    no :func:`~orkexa.register_pricing`, nothing in the tracker's
    ``price_resolver``) have an *unknown* cost, which is not the same as zero; a
    request whose cost could only be partly priced (e.g. an unpriced server tool)
    has an *incomplete* cost. ``on_unpriced`` decides deterministically:

    * ``"ignore"``: count only the known part (zero for unknown prices).
    * ``"warn"``: the same, and emit one :class:`UnpricedUsageWarning` per provider
      and model.
    * ``"reject"``: raise :class:`UnpricedModelError` *before* the request is sent
      when no price is known, and *after* a request (it is recorded first) whose
      cost turned out unknown or incomplete.

    The default is ``"ignore"`` without a limit and ``"warn"`` with one.
    :attr:`unpriced_records` lists the requests whose cost is not completely known.

    Args:
        limit_usd: The budget, ``None`` for none.
        parent: A tracker whose limit also applies: every record and reservation is
            passed on to it (records with ``agent`` renamed ``"<prefix>/<agent>"``).
        prefix: See ``parent``.
        on_oversized: ``"exclusive"`` or ``"reject"``, see above.
        on_unpriced: See above.
        price_resolver: Resolver for agents without their own ``pricing=``
            (see :class:`~orkexa.UsageTracker`).
        price_policy: Which prices are acceptable (age, quality, completeness);
            checked before every request (the resolved price) and after it (every
            price used for the bill). Default: accept every price.
        on_price_issue: What a policy violation does: ``"ignore"``, ``"warn"``
            (default; one :class:`UnreliablePriceWarning` per provider, model and
            issue) or ``"reject"`` (:class:`UnreliablePriceError`, before the request
            or after it is recorded).
    """

    def __init__(
        self,
        limit_usd: Decimal | float | str | None = None,
        *,
        on_unpriced: UnpricedPolicy | None = None,
        price_resolver: PriceResolver | None = None,
        price_policy: PricePolicy | None = None,
        on_price_issue: PriceIssuePolicy = "warn",
        parent: BudgetTracker | None = None,
        prefix: str | None = None,
        on_oversized: OversizedPolicy = "exclusive",
    ) -> None:
        super().__init__(price_resolver=price_resolver)
        if parent is not None and not isinstance(parent, BudgetTracker):
            raise TypeError("parent must be a BudgetTracker")
        self.parent = parent
        self.prefix = prefix
        self._reserved = Decimal(0)
        self._in_flight = 0
        if on_oversized not in ("exclusive", "reject"):
            raise ValueError(f"on_oversized must be 'exclusive' or 'reject', not {on_oversized!r}")
        self.on_oversized: OversizedPolicy = on_oversized
        self.unbounded_requests = 0
        """Admitted requests without an upper bound of their cost (while a limit applied)."""
        self.oversized_requests = 0
        """Requests admitted alone although their upper bound exceeded the rest of the budget."""
        self.limit_usd = None if limit_usd is None else Decimal(str(limit_usd))
        if on_unpriced is None:
            on_unpriced = "ignore" if self.limit_usd is None else "warn"
        if on_unpriced not in ("ignore", "warn", "reject"):
            raise ValueError(f"on_unpriced must be 'ignore', 'warn' or 'reject', not {on_unpriced!r}")
        if on_price_issue not in ("ignore", "warn", "reject"):
            raise ValueError(f"on_price_issue must be 'ignore', 'warn' or 'reject', not {on_price_issue!r}")
        if price_policy is not None and not isinstance(price_policy, PricePolicy):
            raise TypeError("price_policy must be a PricePolicy")
        self.on_unpriced: UnpricedPolicy = on_unpriced
        self.price_policy = price_policy if price_policy is not None else PricePolicy()
        self.on_price_issue: PriceIssuePolicy = on_price_issue
        self._warned: set[tuple[str | None, str, str]] = set()
        self._lock = threading.Lock()

    @property
    def spent_usd(self) -> Decimal:
        """Complete costs plus the known part of incomplete ones (a lower bound if any is unknown)."""
        return sum((r.known_cost for r in self.records), Decimal("0"))

    @property
    def remaining_usd(self) -> Decimal | None:
        return None if self.limit_usd is None else self.limit_usd - self.spent_usd

    @property
    def reserved_usd(self) -> Decimal:
        """Money held back for requests in flight."""
        return self._reserved

    @property
    def hard_maximum(self) -> bool:
        """Whether the limit capped every admitted request (no unbounded, oversized or unpriced one)."""
        if self.limit_usd is None or self.unbounded_requests or self.oversized_requests:
            return False
        if self.unpriced_records:
            return False
        return (
            self.parent.hard_maximum
            if self.parent is not None and self.parent.limit_usd is not None
            else True
        )

    @property
    def unpriced_records(self) -> tuple[UsageRecord, ...]:
        """Recorded requests whose cost is unknown or incomplete."""
        return tuple(r for r in self.records if r.estimated_cost_usd is None)

    def check(self) -> None:
        """Raise :class:`BudgetExceededError` if the budget is already used up."""
        if self.limit_usd is not None and self.spent_usd > self.limit_usd:
            raise BudgetExceededError(
                f"Budget exceeded: spent ${self.spent_usd:.4f} of ${self.limit_usd:.4f}"
            )
        if self.parent is not None:
            self.parent.check()

    def _upstream(self, agent: str) -> str:
        return f"{self.prefix}/{agent}" if self.prefix else agent

    def before_request(self, *, agent: str, provider: str, model: str, priced: bool) -> None:
        # A budget that is exactly used up (e.g. limit 0) admits no further request.
        if self.limit_usd is not None and self.spent_usd >= self.limit_usd:
            raise BudgetExceededError(
                f"Budget used up: spent ${self.spent_usd:.4f} of ${self.limit_usd:.4f}; "
                f"request of agent {agent!r} to {provider}:{model} not sent"
            )
        if not priced and self.on_unpriced == "reject":
            raise UnpricedModelError(
                f"No price known for {provider}:{model} (agent {agent!r}); the budget cannot be enforced. "
                "Pass pricing=ModelPricing(...) to the agent, call orkexa.register_pricing(...), "
                "or use BudgetTracker(on_unpriced='warn')."
            )
        if self.parent is not None:
            self.parent.before_request(
                agent=self._upstream(agent), provider=provider, model=model, priced=priced
            )

    def try_reserve(
        self, *, agent: str, provider: str, model: str, amount: Decimal | None
    ) -> Reservation | None:
        """Hold back ``amount`` for one request here and in every parent, atomically per tracker.

        Returns ``None`` when the request has to wait for requests in flight, and raises
        :class:`BudgetExceededError` when it cannot fit even once they finished. On
        ``None`` or an error nothing stays reserved.
        """
        need = Decimal(0) if amount is None else amount
        oversized = False
        with self._lock:
            if self.limit_usd is not None:
                spent = self.spent_usd
                if spent >= self.limit_usd:
                    raise BudgetExceededError(
                        f"Budget used up: spent ${spent:.4f} of ${self.limit_usd:.4f}; "
                        f"request of agent {agent!r} to {provider}:{model} not sent"
                    )
                if spent + self._reserved + need > self.limit_usd:
                    if self._in_flight:
                        return None
                    if self.on_oversized == "reject":
                        raise BudgetExceededError(
                            f"Budget too small for the request: it can cost up to ${need:.4f}, "
                            f"${self.limit_usd - spent:.4f} of ${self.limit_usd:.4f} left; "
                            f"request of agent {agent!r} to {provider}:{model} not sent"
                        )
                    oversized = True
                if amount is None:
                    self.unbounded_requests += 1
                if oversized:
                    self.oversized_requests += 1
            self._reserved += need
            self._in_flight += 1
        reservation = Reservation(self, amount, agent)
        if self.parent is not None:
            try:
                upper = self.parent.try_reserve(
                    agent=self._upstream(agent), provider=provider, model=model, amount=amount
                )
            except BaseException:
                self._undo(reservation, oversized=oversized)
                raise
            if upper is None:
                self._undo(reservation, oversized=oversized)
                return None
            reservation.parent = upper
        return reservation

    def _undo(self, reservation: Reservation, *, oversized: bool) -> None:
        with self._lock:
            if reservation.amount is None and self.limit_usd is not None:
                self.unbounded_requests -= 1
            if oversized:
                self.oversized_requests -= 1
        self.release(reservation)

    def release(self, reservation: Reservation) -> None:
        if reservation.released:
            return
        reservation.released = True
        with self._lock:
            self._reserved -= reservation.amount or Decimal(0)
            self._in_flight -= 1
        if reservation.parent is not None:
            reservation.parent.tracker.release(reservation.parent)
        with _RELEASED:
            _RELEASED.notify_all()

    def check_request(self, *, agent: str, provider: str, model: str, lookup: PriceLookup) -> None:
        """Budget and unpriced checks (:meth:`before_request`), then the price policy."""
        self.before_request(agent=agent, provider=provider, model=model, priced=lookup.found)
        if lookup.price is not None and self.on_price_issue != "ignore":
            issues = self.price_policy.check_price(lookup.price)
            if issues and self.on_price_issue == "reject":
                raise UnreliablePriceError(
                    f"Request of agent {agent!r} to {provider}:{model} not sent: {'; '.join(issues)}"
                )
            self._warn_issues(provider, model, issues)

    def add(self, record: UsageRecord) -> None:
        super().add(record)
        failure: Exception | None = None
        if self.parent is not None:
            # The parent's ledger gets every record, even if a limit is exceeded on the way.
            try:
                self.parent.add(replace(record, agent=self._upstream(record.agent)))
            except Exception as exc:
                failure = exc
        if record.estimated_cost_usd is None:
            if self.on_unpriced == "warn":
                self._warn_once(
                    record.provider,
                    record.model,
                    "unpriced",
                    self._unpriced_message(record),
                    UnpricedUsageWarning,
                )
            elif self.on_unpriced == "reject":
                failure = failure or UnpricedModelError(
                    f"{self._unpriced_message(record)} (the request was recorded)"
                )
        if record.cost is not None and self.on_price_issue != "ignore":
            issues = self.price_policy.check_cost(record.cost)
            if issues and self.on_price_issue == "reject":
                failure = failure or UnreliablePriceError(
                    f"Request of agent {record.agent!r} to {record.provider}:{record.model} "
                    f"was priced with an unacceptable basis (recorded): {'; '.join(issues)}"
                )
            elif issues:
                self._warn_issues(record.provider, record.model, issues)
        self.check()
        if failure is not None:
            raise failure

    @staticmethod
    def _unpriced_message(record: UsageRecord) -> str:
        name = f"{record.provider or 'anthropic'}:{record.model}"
        missing = [item for item in record.missing_cost_components if item != "price"]
        if record.known_cost_usd and missing:
            return (
                f"The cost of {name} is incomplete (unknown: {', '.join(missing)}); only the "
                f"known ${record.known_cost_usd:.6f} counts against the budget"
            )
        if missing and "price" not in record.missing_cost_components:
            return f"The cost of {name} is unknown ({', '.join(missing)}) and not counted against the budget"
        return f"No price known for {name}; its cost is unknown and not counted against the budget"

    def _warn_issues(self, provider: str | None, model: str, issues: tuple[str, ...]) -> None:
        for issue in issues:
            self._warn_once(provider, model, issue, f"Price basis: {issue}", UnreliablePriceWarning)

    def _warn_once(
        self, provider: str | None, model: str, key: str, message: str, category: type[Warning]
    ) -> None:
        with self._lock:
            new = (provider, model, key) not in self._warned
            self._warned.add((provider, model, key))
        if new:
            warnings.warn(message, category, stacklevel=3)
