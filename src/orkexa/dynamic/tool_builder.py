"""Pipeline: requirement -> Coding Agent -> checks -> sandbox tests -> review -> approval -> registry."""

from __future__ import annotations

from collections.abc import Iterable
import json
import logging
from typing import Any

from ..agent import Agent
from ..evaluation import Criterion, Evaluation, Evaluator
from ..tools import Tool
from ..usage import UsageTracker
from .approval import ApprovalRequest, Approver, share_usage_tracker
from .coding_agent import CodingAgent, ToolProposal
from .errors import (
    ApprovalDeniedError,
    PolicyViolationError,
    RegistryError,
    SandboxError,
    ToolValidationError,
)
from .policy import RESERVED_TOOL_NAMES, BudgetTracker, DynamicPolicy
from .review import TOOL_REVIEW_CRITERIA, review_tool
from .sandbox import Sandbox, ValidationResult
from .static_check import check_tool_source, check_tool_tests
from .tool_registry import ToolRegistry

__all__ = ["ToolBuilder"]

logger = logging.getLogger(__name__)

_EXPECTED_ERRORS = (ToolValidationError, ApprovalDeniedError, PolicyViolationError, RegistryError)


class ToolBuilder:
    """Turns a natural-language requirement into an approved, sandboxed tool.

    Args:
        coding_agent: Writes the code and tests.
        registry: Receives the tool; its sandbox runs the tests and later calls.
        approver: Decides about registration after all checks passed. There is
            deliberately no default; use :class:`AutoApprover` for fully
            automatic operation.
        policy: Limits, here ``max_generated_tools``.
        max_attempts: Proposals per requirement; failed checks are fed back.
        usage_tracker: If given, attached to the Coding Agent, the reviewer and
            an :class:`LLMApprover` (e.g. a shared :class:`BudgetTracker`).
        reviewer: Optional LLM review that runs after the tests passed, inside
            the attempt loop: a failed review is fed back to the Coding Agent
            like a failed test. It cannot approve anything by itself.
        review_criteria: Criteria for the reviewer.
        extra_imports: Further top-level modules the static check permits, e.g.
            approved third-party libraries; see :func:`check_tool_source`. The
            attribute can be changed between proposals.
        sandbox: Sandbox for the tests, e.g. one whose interpreter has those
            libraries installed. Defaults to the registry's sandbox, which also
            runs the registered tool later.
    """

    def __init__(
        self,
        coding_agent: CodingAgent,
        registry: ToolRegistry,
        *,
        approver: Approver,
        policy: DynamicPolicy | None = None,
        max_attempts: int = 3,
        usage_tracker: UsageTracker | None = None,
        reviewer: Evaluator | None = None,
        review_criteria: tuple[Criterion, ...] = TOOL_REVIEW_CRITERIA,
        extra_imports: Iterable[str] = (),
        sandbox: Sandbox | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.coding_agent = coding_agent
        self.registry = registry
        self.approver = approver
        self.policy = policy if policy is not None else DynamicPolicy()
        self.max_attempts = max_attempts
        self.reviewer = reviewer
        self.review_criteria = review_criteria
        self.extra_imports = frozenset(extra_imports)
        self.test_sandbox = sandbox
        self.generated_count = 0
        self._attached: dict[str, list[Agent[Any]]] = {}
        if usage_tracker is not None:
            coding_agent.agent.usage_tracker = usage_tracker
            if reviewer is not None:
                reviewer.agent.usage_tracker = usage_tracker
            share_usage_tracker(approver, usage_tracker)

    @property
    def sandbox(self) -> Sandbox:
        """The sandbox that runs the tests: ``sandbox=`` if given, else the registry's."""
        return self.test_sandbox if self.test_sandbox is not None else self.registry.sandbox

    def _validate(self, proposal: ToolProposal) -> tuple[list[str], ValidationResult | None]:
        errors = check_tool_source(
            proposal.code, proposal.function_name, extra_imports=self.extra_imports
        ).errors
        errors += check_tool_tests(proposal.tests, extra_imports=self.extra_imports).errors
        if proposal.function_name in RESERVED_TOOL_NAMES:
            errors.append(f"{proposal.function_name!r} is a reserved tool name; choose another")
        elif proposal.function_name in self.registry:
            errors.append(f"A tool named {proposal.function_name!r} already exists; choose another name")
        if errors:
            return errors, None
        try:
            result = self.sandbox.validate(proposal.code, proposal.tests, proposal.function_name)
        except SandboxError as exc:
            return [f"Sandbox: {exc}"], None
        if not result.ok:
            return [result.feedback()], result
        if result.name is None or result.description is None or result.input_schema is None:
            return ["Sandbox did not return a tool schema"], result
        return [], result

    def build(self, requirement: str) -> Tool:
        """Develop, test, approve and register a tool; return it ready to use.

        Raises:
            PolicyViolationError: ``max_generated_tools`` reached.
            ToolValidationError: no proposal passed within ``max_attempts``.
            ApprovalDeniedError: the approver rejected the tool.
            BudgetExceededError: the shared budget is used up.
        """
        if not requirement.strip():
            raise ValueError("requirement must be a nonempty string")
        if self.generated_count >= self.policy.max_generated_tools:
            raise PolicyViolationError(f"Limit of {self.policy.max_generated_tools} generated tools reached")
        tracker = self.coding_agent.agent.usage_tracker
        if isinstance(tracker, BudgetTracker):
            tracker.check()

        feedback: str | None = None
        review: Evaluation | None = None
        for attempt in range(1, self.max_attempts + 1):
            proposal = self.coding_agent.propose(requirement, feedback=feedback)
            errors, result = self._validate(proposal)
            if not errors and result is not None and self.reviewer is not None:
                review = review_tool(
                    self.reviewer,
                    requirement,
                    {
                        "code": proposal.code,
                        "tests": proposal.tests,
                        "test_results": [t.name for t in result.tests],
                    },
                    criteria=self.review_criteria,
                )
                if not review.passed:
                    errors = [f"The tests passed, but the code review failed:\n{review.feedback()}"]
            if not errors and result is not None:
                break
            feedback = "\n".join(errors)
            logger.info("Tool proposal %s, attempt %d failed:\n%s", proposal.function_name, attempt, feedback)
        else:
            raise ToolValidationError(
                f"No valid tool after {self.max_attempts} attempts. Last errors:\n{feedback}"
            )

        assert result.name is not None and result.description is not None and result.input_schema is not None
        record = self.registry.add_generated(
            name=result.name,
            function_name=proposal.function_name,
            description=result.description,
            code=proposal.code,
            tests=proposal.tests,
            input_schema=result.input_schema,
            output_schema=result.output_schema,
            requirement=requirement,
        )
        request = ApprovalRequest(
            kind="tool",
            name=record.name,
            summary=(
                f"{proposal.description}\n"
                f"All {len(result.tests)} tests passed in {type(self.sandbox).__name__}."
            ),
            details={
                "requirement": requirement,
                "function_name": proposal.function_name,
                "code": proposal.code,
                "tests": proposal.tests,
                "test_results": [f"{t.name}: {'ok' if t.ok else t.error}" for t in result.tests],
            },
        )
        if review is not None:
            request = ApprovalRequest(
                kind=request.kind,
                name=request.name,
                summary=request.summary,
                details={**request.details, "review": review.feedback()},
            )
        if not self.approver.approve(request):
            self.registry.reject(record.name, record.version)
            raise ApprovalDeniedError(f"Tool {record.name!r} was not approved")
        self.registry.approve(record.name, record.version)
        self.generated_count += 1
        return self.registry.get_tool(record.name)

    # ---- tools for agents ----------------------------------------------------

    def attach(self, agent: Agent[Any], name: str) -> bool:
        """Add a registered tool to ``agent``; False if it already has it."""
        if name in agent.tools:
            return False
        agent.add_tool(self.registry.get_tool(name))
        self._attached.setdefault(name, []).append(agent)
        return True

    def revoke(self, name: str) -> None:
        """Revoke a tool in the registry and remove it from agents it was attached to."""
        self.registry.revoke(name)
        for agent in self._attached.pop(name, []):
            if name in agent.tools:
                agent.remove_tool(name)

    def as_tools(self, agent: Agent[Any]) -> list[Tool]:
        """Tools that let ``agent`` find, load and create tools for itself.

        Returns ``find_tools``, ``load_tool`` and ``create_tool``. Add them with
        ``agent.add_tool()`` (or pass them as ``tools=``). Expected failures are
        returned to the model as text, so it can continue without the tool.
        """
        builder = self

        @Tool
        def find_tools(query: str) -> str:
            """Search the registry of approved tools before creating a new one.

            Args:
                query: Keywords describing the needed capability.

            Returns:
                JSON list of matching tools with name and description.
            """
            matches = builder.registry.search(query)
            return json.dumps([{"name": r.name, "description": r.description} for r in matches])

        @Tool
        def load_tool(name: str) -> str:
            """Make an approved tool from the registry available to you.

            Args:
                name: Exact tool name as returned by find_tools.

            Returns:
                Status message; on success the tool can be called in the next step.
            """
            try:
                added = builder.attach(agent, name)
            except RegistryError as exc:
                return f"Error: {exc}"
            return f"Tool {name!r} is {'now' if added else 'already'} available. You can call it now."

        @Tool
        def create_tool(requirement: str) -> str:
            """Have a new Python tool developed, tested and registered for you.

            Use this only if find_tools found nothing suitable. The tool must be a
            pure computation (no network, files or external services).

            Args:
                requirement: Detailed specification: purpose, inputs with types and
                    units, output, edge cases and error behaviour.

            Returns:
                Status message with the new tool's name, or the reason it failed.
            """
            try:
                tool = builder.build(requirement)
                assert tool.name is not None
                builder.attach(agent, tool.name)
            except _EXPECTED_ERRORS as exc:
                return f"Error: tool could not be created: {exc}"
            return f"Tool {tool.name!r} was created and registered. You can call it now."

        return [find_tools, load_tool, create_tool]
