"""The Orchestrator: delegates to specialist agents and creates missing ones."""

from __future__ import annotations

from contextvars import ContextVar
import json
from typing import Any

from anthropic import Anthropic
from pydantic import BaseModel

from ..claude import ClaudeAgent
from ..core.types import Effort
from ..tools import Tool
from ..usage import UsageTracker
from .agent_registry import AgentRecord, AgentRegistry
from .approval import ApprovalRequest, Approver, share_usage_tracker
from .architect import AgentArchitect
from .errors import (
    ApprovalDeniedError,
    PolicyViolationError,
    RecursionLimitError,
    RegistryError,
    ToolValidationError,
)
from .factory import AgentFactory
from .output_registry import OutputFormatRegistry
from .policy import BudgetTracker, DynamicPolicy
from .tool_builder import ToolBuilder
from .tool_registry import ToolRegistry

__all__ = ["DEFAULT_ORCHESTRATOR_PROMPT", "Orchestrator"]

_EXPECTED_ERRORS = (
    ApprovalDeniedError,
    PolicyViolationError,
    RecursionLimitError,
    RegistryError,
    ToolValidationError,
)

#: Current delegation depth; 0 in the orchestrator, 1 in a specialist it called.
_depth: ContextVar[int] = ContextVar("claude_agent_dynamic_depth", default=0)


DEFAULT_ORCHESTRATOR_PROMPT = """\
You are an orchestrator that solves tasks by delegating to specialist agents.

1. Call list_agents to see which specialists exist.
2. If a suitable specialist exists, delegate with run_agent. Give it a complete,
   self-contained task: it does not see this conversation.
3. Only if no suitable specialist exists, call create_agent with a precise
   requirement, then delegate to the new agent with run_agent.
4. Combine the specialists' results into your final answer.

Prefer reusing existing agents over creating new ones. If a tool returns an error,
adapt your plan instead of retrying the same call.
"""


class Orchestrator:
    """Coordinates a Dynamic Agent Architecture.

    The orchestrator is a :class:`~claude_agent.ClaudeAgent` with three generic
    tools: ``list_agents``, ``create_agent`` and ``run_agent``. ``create_agent``
    asks the :class:`AgentArchitect` for an :class:`AgentSpec`, has missing tools
    built by the :class:`ToolBuilder`, validates the spec with the
    :class:`AgentFactory`, asks the approver and stores the spec in the
    :class:`AgentRegistry`. ``run_agent`` instantiates a specialist from its
    spec and runs one task with a clean history.

    All agents (orchestrator, architect, coding agent, specialists) share one
    :class:`BudgetTracker` built from ``policy.budget_usd`` unless
    ``usage_tracker`` is given.

    Args:
        architect: Designs new agents.
        approver: Approves new agents (tools are approved by the ToolBuilder's approver).
        tool_builder: Builds missing tools; ``None`` disables tool generation.
        tools: Tool registry; defaults to the tool builder's registry.
        agents: Agent registry; defaults to an in-memory registry.
        output_formats: Output formats the architect may choose from.
        policy: Limits for everything created dynamically.
    """

    def __init__(
        self,
        *,
        architect: AgentArchitect,
        approver: Approver,
        tool_builder: ToolBuilder | None = None,
        tools: ToolRegistry | None = None,
        agents: AgentRegistry | None = None,
        output_formats: OutputFormatRegistry | None = None,
        policy: DynamicPolicy | None = None,
        model: str = "claude-opus-5-5",
        system_prompt: str = DEFAULT_ORCHESTRATOR_PROMPT,
        effort: Effort | None = "high",
        max_tokens: int = 8192,
        max_tool_rounds: int = 12,
        client: Anthropic | None = None,
        api_key: str | None = None,
        usage_tracker: UsageTracker | None = None,
        name: str = "Orchestrator",
    ) -> None:
        if tools is None:
            if tool_builder is None:
                raise ValueError("Pass a ToolRegistry as tools= or a tool_builder")
            tools = tool_builder.registry
        elif tool_builder is not None and tool_builder.registry is not tools:
            raise ValueError("tools and tool_builder.registry must be the same registry")
        self.policy = policy if policy is not None else DynamicPolicy()
        self.usage_tracker = (
            usage_tracker if usage_tracker is not None else BudgetTracker(self.policy.budget_usd)
        )
        self.architect = architect
        self.approver = approver
        self.tool_builder = tool_builder
        self.tool_registry = tools
        self.agents = agents if agents is not None else AgentRegistry()
        self.output_formats = output_formats if output_formats is not None else OutputFormatRegistry()

        architect.agent.usage_tracker = self.usage_tracker
        share_usage_tracker(approver, self.usage_tracker)
        if tool_builder is not None:
            tool_builder.coding_agent.agent.usage_tracker = self.usage_tracker
            if tool_builder.reviewer is not None:
                tool_builder.reviewer.agent.usage_tracker = self.usage_tracker
            share_usage_tracker(tool_builder.approver, self.usage_tracker)

        list_agents, create_agent, run_agent = self._make_tools()
        self.factory = AgentFactory(
            tools,
            self.output_formats,
            self.policy,
            delegation_tools={"list_agents": list_agents, "run_agent": run_agent},
            client=client,
            api_key=api_key,
            usage_tracker=self.usage_tracker,
        )
        self.agent: ClaudeAgent[str] = ClaudeAgent(
            name=name,
            model=model,
            system_prompt=system_prompt,
            tools=[list_agents, create_agent, run_agent],
            effort=effort,
            max_tokens=max_tokens,
            max_tool_rounds=max_tool_rounds,
            raise_tool_errors=False,
            client=client,
            api_key=api_key,
            usage_tracker=self.usage_tracker,
        )

    # ---- Python API ----------------------------------------------------------

    def ask(self, task: str) -> str:
        """Let the orchestrator solve ``task``."""
        self._check_budget()
        return self.agent.ask(task)

    def _check_budget(self) -> None:
        if isinstance(self.usage_tracker, BudgetTracker):
            self.usage_tracker.check()

    def create_agent(self, requirement: str) -> AgentRecord:
        """Design, validate, approve and register a new specialist agent.

        Raises:
            PolicyViolationError: limit reached, or the spec violates the policy.
            ToolValidationError: a missing tool could not be built.
            ApprovalDeniedError: the approver rejected the agent or a tool.
        """
        if not requirement.strip():
            raise ValueError("requirement must be a nonempty string")
        if len(self.agents) >= self.policy.max_agents:
            raise PolicyViolationError(f"Limit of {self.policy.max_agents} agents reached")
        self._check_budget()
        spec = self.architect.design(
            requirement,
            tools=self.tool_registry.catalog(),
            output_formats=self.output_formats.catalog(),
            existing_agents=self.agents.names(),
            policy=self.policy,
        )
        if spec.name in self.agents:
            raise PolicyViolationError(f"An agent named {spec.name!r} already exists")
        if spec.missing_tools:
            if self.tool_builder is None:
                raise PolicyViolationError(
                    f"Agent needs tools that do not exist and tool generation is disabled: "
                    f"{[t.name for t in spec.missing_tools]}"
                )
            tool_names = list(spec.tools)
            for missing in spec.missing_tools:
                tool = self.tool_builder.build(f"Suggested name: {missing.name}\n{missing.requirement}")
                assert tool.name is not None
                tool_names.append(tool.name)
            spec = spec.model_copy(update={"tools": tool_names, "missing_tools": []})
        spec = self.factory.validate(spec)
        self.factory.create(spec)  # make sure it can be instantiated (no API call)
        request = ApprovalRequest(
            kind="agent",
            name=spec.name,
            summary=spec.description,
            details={"requirement": requirement, "spec": spec.model_dump()},
        )
        if not self.approver.approve(request):
            raise ApprovalDeniedError(f"Agent {spec.name!r} was not approved")
        return self.agents.register(spec, requirement)

    def run_agent(self, name: str, task: str) -> str:
        """Run one task on a registered specialist with a clean history.

        Structured results are returned as JSON.

        Raises:
            RecursionLimitError: the delegation depth would exceed ``policy.max_depth``.
            RegistryError: unknown agent.
        """
        depth = _depth.get()
        if depth >= self.policy.max_depth:
            raise RecursionLimitError(f"Maximum delegation depth {self.policy.max_depth} reached")
        self._check_budget()
        agent = self.factory.create(self.agents.get(name).spec)
        token = _depth.set(depth + 1)
        try:
            result: Any = agent.ask(task)
        finally:
            _depth.reset(token)
        if isinstance(result, BaseModel):
            return result.model_dump_json()
        return str(result)

    # ---- tools for the model -------------------------------------------------

    def _make_tools(self) -> tuple[Tool, Tool, Tool]:
        orchestrator = self

        @Tool
        def list_agents(query: str) -> str:
            """List registered specialist agents.

            Args:
                query: Keywords to filter by name or description; an empty string lists all.

            Returns:
                JSON list of agents with name and description.
            """
            matches = orchestrator.agents.search(query, limit=50)
            return json.dumps([{"name": r.name, "description": r.spec.description} for r in matches])

        @Tool
        def create_agent(requirement: str) -> str:
            """Create and register a new specialist agent.

            Use this only if list_agents shows no suitable agent.

            Args:
                requirement: What the specialist must be able to do, which inputs it
                    gets and what results it must return.

            Returns:
                Name and description of the new agent, or the reason it failed.
            """
            try:
                record = orchestrator.create_agent(requirement)
            except _EXPECTED_ERRORS as exc:
                return f"Error: agent could not be created: {exc}"
            return json.dumps(
                {"created": record.name, "description": record.spec.description, "tools": record.spec.tools}
            )

        @Tool
        def run_agent(agent_name: str, task: str) -> str:
            """Execute a task using a registered specialist agent.

            Args:
                agent_name: Exact name of the specialist agent.
                task: Complete, self-contained task for the specialist.

            Returns:
                Result produced by the specialist agent (text or JSON).
            """
            try:
                return orchestrator.run_agent(agent_name, task)
            except _EXPECTED_ERRORS as exc:
                return f"Error: {exc}"

        return list_agents, create_agent, run_agent
