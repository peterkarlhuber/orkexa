"""Dynamic Agent Architecture: agents that create tools and agents at runtime.

An optional extension layer on top of :class:`claude_agent.ClaudeAgent`; the
core package never imports it.

* :class:`CodingAgent` writes a Python function and tests (:class:`ToolProposal`).
* :class:`ToolBuilder` checks the code statically, runs its tests in a
  :class:`Sandbox`, asks an :class:`Approver` and stores it in a :class:`ToolRegistry`.
  Generated code never runs in the host process; each call goes through the sandbox
  (:class:`SandboxedTool`).
* :class:`AgentArchitect` designs specialist agents (:class:`AgentSpec`),
  :class:`AgentFactory` validates them against the :class:`DynamicPolicy` and
  instantiates them, :class:`AgentRegistry` keeps them.
* :class:`LLMApprover` and ``ToolBuilder(reviewer=...)`` add an LLM review
  (:class:`claude_agent.Evaluator`) that can reject or trigger revisions, but
  never approve on its own.
* :class:`Orchestrator` delegates with ``list_agents`` / ``create_agent`` / ``run_agent``.

Quick start (one ``ask()`` that creates and then uses a tool)::

    from claude_agent import ClaudeAgent
    from claude_agent.dynamic import AutoApprover, CodingAgent, ToolBuilder, ToolRegistry

    builder = ToolBuilder(CodingAgent(), ToolRegistry(), approver=AutoApprover())
    analyst = ClaudeAgent(model="claude-opus-5-5", system_prompt="You are an investment analyst.")
    for tool in builder.as_tools(analyst):
        analyst.add_tool(tool)
    print(analyst.ask("Revenue is 200m, operating income 50m. What is the operating margin?"))
"""

from __future__ import annotations

from .agent_registry import AgentRecord, AgentRegistry
from .agent_spec import AgentSpec, ToolRequirement
from .approval import (
    ApprovalRequest,
    Approver,
    AutoApprover,
    CallbackApprover,
    ConsoleApprover,
    DenyApprover,
    LLMApprover,
)
from .architect import AgentArchitect
from .coding_agent import CodingAgent, ToolProposal
from .errors import (
    ApprovalDeniedError,
    BudgetExceededError,
    DynamicAgentError,
    PolicyViolationError,
    RecursionLimitError,
    RegistryError,
    SandboxError,
    ToolValidationError,
)
from .factory import AgentFactory
from .orchestrator import Orchestrator
from .output_registry import OutputFormatRegistry
from .policy import RESERVED_TOOL_NAMES, BudgetTracker, DynamicPolicy
from .review import AGENT_REVIEW_CRITERIA, TOOL_REVIEW_CRITERIA, review_agent, review_tool
from .sandbox import DockerSandbox, Sandbox, SandboxTestResult, SubprocessSandbox, ValidationResult
from .sandboxed_tool import SandboxedTool
from .static_check import StaticReport, check_tool_source, check_tool_tests
from .tool_builder import ToolBuilder
from .tool_registry import ToolRecord, ToolRegistry

__all__ = [
    "AGENT_REVIEW_CRITERIA",
    "RESERVED_TOOL_NAMES",
    "TOOL_REVIEW_CRITERIA",
    "AgentArchitect",
    "AgentFactory",
    "AgentRecord",
    "AgentRegistry",
    "AgentSpec",
    "ApprovalDeniedError",
    "ApprovalRequest",
    "Approver",
    "AutoApprover",
    "BudgetExceededError",
    "BudgetTracker",
    "CallbackApprover",
    "CodingAgent",
    "ConsoleApprover",
    "DenyApprover",
    "DockerSandbox",
    "DynamicAgentError",
    "DynamicPolicy",
    "LLMApprover",
    "Orchestrator",
    "OutputFormatRegistry",
    "PolicyViolationError",
    "RecursionLimitError",
    "RegistryError",
    "Sandbox",
    "SandboxError",
    "SandboxTestResult",
    "SandboxedTool",
    "StaticReport",
    "SubprocessSandbox",
    "ToolBuilder",
    "ToolProposal",
    "ToolRecord",
    "ToolRegistry",
    "ToolRequirement",
    "ToolValidationError",
    "ValidationResult",
    "check_tool_source",
    "check_tool_tests",
    "review_agent",
    "review_tool",
]
