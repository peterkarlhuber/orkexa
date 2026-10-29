"""Limits for dynamically created tools and agents, and a budget-enforcing tracker."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal
import threading
from typing import Literal, get_args
import warnings

from ..core.types import Effort
from ..usage import UsageRecord, UsageTracker
from .errors import BudgetExceededError, UnpricedModelError, UnpricedUsageWarning

__all__ = ["RESERVED_TOOL_NAMES", "BudgetTracker", "DynamicPolicy", "UnpricedPolicy", "clamp_effort"]

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


class BudgetTracker(UsageTracker):
    """A :class:`~orkexa.UsageTracker` that enforces a USD limit across providers.

    Every request is recorded first; if the summed estimated cost then exceeds
    ``limit_usd``, :class:`BudgetExceededError` is raised. Because the agent
    records usage right after each API response, this aborts a running tool
    loop. Before each request the tracker refuses to start when the budget is
    already used up (``spent_usd >= limit_usd``, so a zero budget sends nothing).

    The limit is a *soft* limit: the cost of a request is only known after it
    ran, so the last admitted request can overshoot it, and agents that share a
    tracker and run concurrently can each overshoot it by one request.

    Requests of a model without a known price (no built-in price, no
    ``pricing=``/:func:`~orkexa.register_pricing`) have an *unknown* cost, which
    is not the same as zero. ``on_unpriced`` decides deterministically:

    * ``"ignore"`` (the ``claude-agent`` 0.8.0 behavior): count them as zero.
    * ``"warn"``: count them as zero and emit one :class:`UnpricedUsageWarning`
      per provider and model.
    * ``"reject"``: raise :class:`UnpricedModelError` *before* the request is
      sent, so no unbudgeted cost is incurred.

    The default is ``"ignore"`` without a limit and ``"warn"`` with one.
    :attr:`unpriced_records` lists the requests whose cost is unknown.
    """

    def __init__(
        self,
        limit_usd: Decimal | float | str | None = None,
        *,
        on_unpriced: UnpricedPolicy | None = None,
    ) -> None:
        super().__init__()
        self.limit_usd = None if limit_usd is None else Decimal(str(limit_usd))
        if on_unpriced is None:
            on_unpriced = "ignore" if self.limit_usd is None else "warn"
        if on_unpriced not in ("ignore", "warn", "reject"):
            raise ValueError(f"on_unpriced must be 'ignore', 'warn' or 'reject', not {on_unpriced!r}")
        self.on_unpriced: UnpricedPolicy = on_unpriced
        self._warned: set[tuple[str | None, str]] = set()
        self._lock = threading.Lock()

    @property
    def spent_usd(self) -> Decimal:
        return sum(
            (r.estimated_cost_usd for r in self.records if r.estimated_cost_usd is not None),
            Decimal("0"),
        )

    @property
    def remaining_usd(self) -> Decimal | None:
        return None if self.limit_usd is None else self.limit_usd - self.spent_usd

    @property
    def unpriced_records(self) -> tuple[UsageRecord, ...]:
        """Recorded requests whose cost is unknown (not counted in :attr:`spent_usd`)."""
        return tuple(r for r in self.records if r.estimated_cost_usd is None)

    def check(self) -> None:
        """Raise :class:`BudgetExceededError` if the budget is already used up."""
        if self.limit_usd is not None and self.spent_usd > self.limit_usd:
            raise BudgetExceededError(
                f"Budget exceeded: spent ${self.spent_usd:.4f} of ${self.limit_usd:.4f}"
            )

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

    def add(self, record: UsageRecord) -> None:
        super().add(record)
        if record.estimated_cost_usd is None and self.on_unpriced == "warn":
            key = (record.provider, record.model)
            with self._lock:
                new = key not in self._warned
                self._warned.add(key)
            if new:
                warnings.warn(
                    f"No price known for {record.provider or 'anthropic'}:{record.model}; "
                    "its cost is unknown and not counted against the budget",
                    UnpricedUsageWarning,
                    stacklevel=2,
                )
        self.check()
