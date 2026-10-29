"""Approval of generated tools and agents before they are registered."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from ..evaluation import Criterion, Evaluation, Evaluator
from ..usage import UsageTracker
from .review import AGENT_REVIEW_CRITERIA, TOOL_REVIEW_CRITERIA, review_agent, review_tool

__all__ = [
    "ApprovalRequest",
    "Approver",
    "AutoApprover",
    "CallbackApprover",
    "ConsoleApprover",
    "DenyApprover",
    "LLMApprover",
    "share_usage_tracker",
]


@dataclass(frozen=True)
class ApprovalRequest:
    """What an approver gets to see.

    For a tool, ``details`` contains ``requirement``, ``function_name``,
    ``code``, ``tests`` and ``test_results``; for an agent, ``requirement``
    and ``spec`` (the validated AgentSpec as a dict).
    """

    kind: Literal["tool", "agent"]
    name: str
    summary: str
    details: dict[str, Any] = field(default_factory=dict)


class Approver(ABC):
    """Decides whether a generated tool or agent may be registered."""

    @abstractmethod
    def approve(self, request: ApprovalRequest) -> bool:
        """Return True to approve, False to reject."""


class AutoApprover(Approver):
    """Approves everything that reached it.

    A tool only reaches the approver after it passed the static check and all
    its tests in the sandbox; an agent only after its spec passed the policy.
    Choose this for fully automatic operation inside a single ``ask()``.
    """

    def approve(self, request: ApprovalRequest) -> bool:
        return True


class DenyApprover(Approver):
    """Rejects everything (useful for dry runs and tests)."""

    def approve(self, request: ApprovalRequest) -> bool:
        return False


class CallbackApprover(Approver):
    """Delegates the decision to a function, e.g. a UI or a ticket system."""

    def __init__(self, callback: Callable[[ApprovalRequest], bool]) -> None:
        self.callback = callback

    def approve(self, request: ApprovalRequest) -> bool:
        return bool(self.callback(request))


class ConsoleApprover(Approver):
    """Shows code, tests and results on the console and asks y/N."""

    def __init__(
        self,
        *,
        input_func: Callable[[str], str] = input,
        print_func: Callable[[str], None] = print,
    ) -> None:
        self.input_func = input_func
        self.print_func = print_func

    def approve(self, request: ApprovalRequest) -> bool:
        self.print_func(f"\n=== Approval required: {request.kind} {request.name!r} ===")
        self.print_func(request.summary)
        for key, value in request.details.items():
            self.print_func(f"\n--- {key} ---\n{value}")
        answer = self.input_func(f"Approve {request.kind} {request.name!r}? [y/N] ")
        return answer.strip().lower() in ("y", "yes", "j", "ja")


class LLMApprover(Approver):
    """Puts an LLM review in front of another approver.

    The review can only make the decision stricter: a request that fails it is
    rejected (or handed to ``on_reject``, e.g. a :class:`ConsoleApprover`), and a
    request that passes still needs ``then`` to approve. The verdict is added to
    the request as ``details["evaluation"]``, so a human approver sees it. This
    is a quality check, not a security boundary: sandbox, static check and
    policy stay in force.

    Args:
        then: Decides about requests the review passed. Required on purpose;
            ``AutoApprover()`` gives fully automatic operation.
        evaluator: The judge; defaults to an :class:`~claude_agent.Evaluator`.
        on_reject: Decides about requests the review failed; ``None`` rejects them.
        tool_criteria: Criteria for tools.
        agent_criteria: Criteria for agents.
    """

    def __init__(
        self,
        *,
        then: Approver,
        evaluator: Evaluator | None = None,
        on_reject: Approver | None = None,
        tool_criteria: Iterable[Criterion] = TOOL_REVIEW_CRITERIA,
        agent_criteria: Iterable[Criterion] = AGENT_REVIEW_CRITERIA,
    ) -> None:
        self.then = then
        self.evaluator = evaluator if evaluator is not None else Evaluator()
        self.on_reject = on_reject
        self.tool_criteria = tuple(tool_criteria)
        self.agent_criteria = tuple(agent_criteria)
        self.last_evaluation: Evaluation | None = None

    def approve(self, request: ApprovalRequest) -> bool:
        requirement = str(request.details.get("requirement", request.summary))
        if request.kind == "tool":
            evaluation = review_tool(
                self.evaluator, requirement, request.details, criteria=self.tool_criteria
            )
        else:
            evaluation = review_agent(
                self.evaluator,
                requirement,
                request.details.get("spec", request.details),
                criteria=self.agent_criteria,
            )
        self.last_evaluation = evaluation
        annotated = replace(request, details={**request.details, "evaluation": evaluation.feedback()})
        if evaluation.passed:
            return self.then.approve(annotated)
        if self.on_reject is None:
            return False
        return self.on_reject.approve(annotated)


def share_usage_tracker(approver: Approver | None, tracker: UsageTracker) -> None:
    """Count the cost of the LLM reviews in an approver chain in ``tracker``."""
    while isinstance(approver, LLMApprover):
        approver.evaluator.agent.usage_tracker = tracker
        share_usage_tracker(approver.on_reject, tracker)
        approver = approver.then
