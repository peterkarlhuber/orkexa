"""Exceptions of the dynamic agent extension.

All of them derive from :class:`claude_agent.ClaudeAgentError` via
:class:`DynamicAgentError`, and additionally from the closest builtin exception.
"""

from __future__ import annotations

from ..errors import ClaudeAgentError

__all__ = [
    "ApprovalDeniedError",
    "BudgetExceededError",
    "DynamicAgentError",
    "PolicyViolationError",
    "RecursionLimitError",
    "RegistryError",
    "SandboxError",
    "ToolValidationError",
]


class DynamicAgentError(ClaudeAgentError):
    """Base class of all errors raised by claude_agent.dynamic."""


class PolicyViolationError(DynamicAgentError, ValueError):
    """A proposal or request exceeds what the DynamicPolicy allows."""


class BudgetExceededError(DynamicAgentError, RuntimeError):
    """The shared USD budget of a BudgetTracker has been used up."""


class RecursionLimitError(DynamicAgentError, RuntimeError):
    """Agent delegation went deeper than DynamicPolicy.max_depth."""


class SandboxError(DynamicAgentError, RuntimeError):
    """The sandbox could not run the code (timeout, crash, malformed output)."""


class ToolValidationError(DynamicAgentError, ValueError):
    """Generated tool code failed the static check or its tests."""


class ApprovalDeniedError(DynamicAgentError, RuntimeError):
    """The approver rejected a generated tool or agent."""


class RegistryError(DynamicAgentError, LookupError):
    """A tool or agent is unknown, revoked, or its stored code was modified."""
