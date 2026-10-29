"""The Agent Architect: designs specialist agents as AgentSpec objects."""

from __future__ import annotations

import json

from anthropic import Anthropic

from ..agent import ClaudeAgent, Effort
from ..usage import UsageTracker
from .agent_spec import AgentSpec
from .policy import DynamicPolicy

__all__ = ["DEFAULT_ARCHITECT_PROMPT", "AgentArchitect"]

DEFAULT_ARCHITECT_PROMPT = """\
You are an expert multi-agent systems architect. Design one specialized AI agent
for the given requirement and return it as an AgentSpec:

- Define a precise purpose in `description`.
- Write a comprehensive, specific `system_prompt`.
- Choose the cheapest allowed model and the lowest effort that will do the job reliably.
- Select only the existing tools the agent really needs (`tools`, exact names).
- If the agent needs a capability that no existing tool provides and that is a pure
  computation, describe it in `missing_tools`; it will be implemented for you.
  Never list missing tools that need network, file or external service access.
- Choose a registered output format if structured results are useful, else null.
- Keep `max_tokens` and `max_tool_rounds` within the limits given.
"""


class AgentArchitect:
    """Wraps a :class:`~claude_agent.ClaudeAgent` that returns :class:`AgentSpec` objects.

    The architect only proposes; the :class:`AgentFactory` enforces the policy.
    The underlying agent is available as ``architect.agent``.
    """

    def __init__(
        self,
        *,
        model: str = "claude-opus-5-5",
        system_prompt: str = DEFAULT_ARCHITECT_PROMPT,
        effort: Effort | None = "high",
        max_tokens: int = 8192,
        client: Anthropic | None = None,
        api_key: str | None = None,
        usage_tracker: UsageTracker | None = None,
        name: str = "Agent Architect",
    ) -> None:
        self.agent: ClaudeAgent[AgentSpec] = ClaudeAgent(
            name=name,
            model=model,
            system_prompt=system_prompt,
            output_format=AgentSpec,
            effort=effort,
            max_tokens=max_tokens,
            max_tool_rounds=0,
            client=client,
            api_key=api_key,
            usage_tracker=usage_tracker,
        )

    def design(
        self,
        requirement: str,
        *,
        tools: list[dict[str, str]],
        output_formats: list[dict[str, str]],
        existing_agents: list[str],
        policy: DynamicPolicy,
    ) -> AgentSpec:
        """Design an agent; the catalogs and limits are included in the prompt."""
        self.agent.reset()
        limits = {
            "allowed_models": list(policy.allowed_models),
            "max_effort": policy.max_effort,
            "max_tokens": policy.max_tokens,
            "max_tool_rounds": policy.max_tool_rounds,
        }
        prompt = (
            f"Requirement:\n{requirement}\n\n"
            f"Existing tools:\n{json.dumps(tools, indent=1)}\n\n"
            f"Registered output formats:\n{json.dumps(output_formats, indent=1)}\n\n"
            f"Names already taken by other agents: {json.dumps(existing_agents)}\n\n"
            f"Limits:\n{json.dumps(limits, indent=1)}"
        )
        return self.agent.ask(prompt)
