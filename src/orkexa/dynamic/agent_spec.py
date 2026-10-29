"""Structured definition of an agent, produced by the Agent Architect."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..agent import Effort

__all__ = ["AgentSpec", "ToolRequirement"]


class ToolRequirement(BaseModel):
    """A tool the new agent needs that does not exist yet."""

    name: str = Field(description="Suggested snake_case name of the missing tool.")
    requirement: str = Field(
        description="Detailed specification: purpose, inputs with types and units, output, edge cases."
    )


class AgentSpec(BaseModel):
    """Machine-readable description of a specialist agent.

    The Agent Architect proposes it; :class:`AgentFactory` validates it against
    the :class:`DynamicPolicy` and turns it into a :class:`~claude_agent.ClaudeAgent`.
    """

    name: str = Field(description="Unique name: letters, digits, '_' or '-', starting with a letter.")
    description: str = Field(description="One or two sentences on what the agent is for.")
    system_prompt: str = Field(description="Complete system prompt of the agent.")
    model: str = Field(description="One of the allowed model IDs.")
    effort: Effort = Field(description="Reasoning effort.")
    tools: list[str] = Field(default_factory=list, description="Names of existing tools the agent needs.")
    missing_tools: list[ToolRequirement] = Field(
        default_factory=list, description="Tools the agent needs that do not exist yet."
    )
    output_format: str | None = Field(
        default=None, description="Name of a registered output format, or null for plain text."
    )
    max_tokens: int = Field(default=4096, description="Output token limit per request.")
    max_tool_rounds: int = Field(default=8, description="Maximum tool rounds per task.")
