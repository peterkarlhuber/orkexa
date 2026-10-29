"""Limits for dynamically created tools and agents, and a budget-enforcing tracker."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal
from typing import get_args

from ..core.types import Effort
from ..usage import UsageRecord, UsageTracker
from .errors import BudgetExceededError

__all__ = ["RESERVED_TOOL_NAMES", "BudgetTracker", "DynamicPolicy", "clamp_effort"]

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
        allowed_models: Models a generated agent may use.
        max_effort: Highest effort a generated agent may use (higher is lowered).
        max_tokens: Upper bound for a generated agent's ``max_tokens`` (higher is lowered).
        max_tool_rounds: Upper bound for a generated agent's ``max_tool_rounds``.
        allowed_agent_tools: If set, generated agents may only use these registered tools.
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


class BudgetTracker(UsageTracker):
    """A :class:`~claude_agent.UsageTracker` that enforces a USD limit.

    Every request is recorded first; if the summed estimated cost then exceeds
    ``limit_usd``, :class:`BudgetExceededError` is raised. Because the agent
    records usage right after each API response, this aborts a running tool
    loop. A single request can therefore overshoot the limit slightly.
    Requests without a known price count as zero.
    """

    def __init__(self, limit_usd: Decimal | float | str | None = None) -> None:
        super().__init__()
        self.limit_usd = None if limit_usd is None else Decimal(str(limit_usd))

    @property
    def spent_usd(self) -> Decimal:
        return sum(
            (r.estimated_cost_usd for r in self.records if r.estimated_cost_usd is not None),
            Decimal("0"),
        )

    @property
    def remaining_usd(self) -> Decimal | None:
        return None if self.limit_usd is None else self.limit_usd - self.spent_usd

    def check(self) -> None:
        """Raise :class:`BudgetExceededError` if the budget is already used up."""
        if self.limit_usd is not None and self.spent_usd > self.limit_usd:
            raise BudgetExceededError(
                f"Budget exceeded: spent ${self.spent_usd:.4f} of ${self.limit_usd:.4f}"
            )

    def add(self, record: UsageRecord) -> None:
        super().add(record)
        self.check()
