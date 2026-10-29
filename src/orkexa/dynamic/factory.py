"""Turns a validated AgentSpec into a ClaudeAgent; the policy has the last word."""

from __future__ import annotations

import re
from typing import Any

from anthropic import Anthropic

from ..agent import ClaudeAgent
from ..tools import Tool
from ..usage import UsageTracker
from .agent_spec import AgentSpec
from .errors import PolicyViolationError
from .output_registry import OutputFormatRegistry
from .policy import RESERVED_TOOL_NAMES, DynamicPolicy, clamp_effort
from .tool_registry import ToolRegistry

__all__ = ["AgentFactory"]

_AGENT_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}")


class AgentFactory:
    """Validates :class:`AgentSpec` objects and instantiates agents.

    Args:
        tools: Registry of approved tools the spec may reference.
        output_formats: Registered output formats.
        policy: Limits that override whatever the spec asks for.
        delegation_tools: Orchestration tools (``run_agent`` ...) that generated
            agents may receive if ``policy.allow_agent_delegation`` is set.
        client: Anthropic client passed to every created agent.
        api_key: API key passed to every created agent.
        usage_tracker: Shared tracker injected into every created agent.
    """

    def __init__(
        self,
        tools: ToolRegistry,
        output_formats: OutputFormatRegistry | None = None,
        policy: DynamicPolicy | None = None,
        *,
        delegation_tools: dict[str, Tool] | None = None,
        client: Anthropic | None = None,
        api_key: str | None = None,
        usage_tracker: UsageTracker | None = None,
    ) -> None:
        self.tools = tools
        self.output_formats = output_formats if output_formats is not None else OutputFormatRegistry()
        self.policy = policy if policy is not None else DynamicPolicy()
        self.delegation_tools = dict(delegation_tools or {})
        self.client = client
        self.api_key = api_key
        self.usage_tracker = usage_tracker

    def validate(self, spec: AgentSpec) -> AgentSpec:
        """Check a spec against the policy; return a copy with limits applied.

        Raises:
            PolicyViolationError: name, model, tool or output format not allowed,
                or the spec still lists missing tools.
        """
        policy = self.policy
        errors: list[str] = []
        if not _AGENT_NAME.fullmatch(spec.name):
            errors.append(f"invalid agent name {spec.name!r}")
        if not spec.system_prompt.strip():
            errors.append("empty system prompt")
        if spec.model not in policy.allowed_models:
            errors.append(f"model {spec.model!r} not allowed (allowed: {list(policy.allowed_models)})")
        if spec.missing_tools:
            errors.append(f"unresolved missing tools: {[t.name for t in spec.missing_tools]}")
        if len(set(spec.tools)) != len(spec.tools):
            errors.append("duplicate tool names")
        for name in spec.tools:
            if name in RESERVED_TOOL_NAMES:
                if not policy.allow_agent_delegation or name not in self.delegation_tools:
                    errors.append(f"tool {name!r} is reserved for the orchestrator")
            elif name not in self.tools:
                errors.append(f"unknown tool {name!r}")
            elif policy.allowed_agent_tools is not None and name not in policy.allowed_agent_tools:
                errors.append(f"tool {name!r} is not allowed for generated agents")
        if spec.output_format is not None and spec.output_format not in self.output_formats:
            errors.append(f"unknown output format {spec.output_format!r}")
        if errors:
            raise PolicyViolationError(f"AgentSpec {spec.name!r} rejected: " + "; ".join(errors))
        return spec.model_copy(
            update={
                "effort": clamp_effort(spec.effort, policy.max_effort),
                "max_tokens": min(max(spec.max_tokens, 256), policy.max_tokens),
                "max_tool_rounds": min(max(spec.max_tool_rounds, 0), policy.max_tool_rounds),
            }
        )

    def create(self, spec: AgentSpec) -> ClaudeAgent[Any]:
        """Validate ``spec`` and build a fresh agent from it."""
        spec = self.validate(spec)
        tools = [
            self.delegation_tools[name] if name in RESERVED_TOOL_NAMES else self.tools.get_tool(name)
            for name in spec.tools
        ]
        output_format = self.output_formats.get(spec.output_format) if spec.output_format else None
        return ClaudeAgent(
            name=spec.name,
            model=spec.model,
            system_prompt=spec.system_prompt,
            tools=tools,
            output_format=output_format,
            effort=spec.effort,
            max_tokens=spec.max_tokens,
            max_tool_rounds=spec.max_tool_rounds,
            raise_tool_errors=False,
            client=self.client,
            api_key=self.api_key,
            usage_tracker=self.usage_tracker,
        )
