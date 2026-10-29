"""The Coding Agent: writes one Python function plus tests for a requirement."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field

from ..agent import Agent
from ..core.types import Effort
from ..usage import UsageTracker
from ._agents import ProviderSettings, build_agent
from .static_check import ALLOWED_IMPORTS, TOOL_MODULE

__all__ = ["DEFAULT_CODING_PROMPT", "CodingAgent", "ToolProposal"]


class ToolProposal(BaseModel):
    """A generated tool: one public function and its tests."""

    function_name: str = Field(description="Name of the single public function in `code`.")
    description: str = Field(description="One or two sentences on what the tool does.")
    code: str = Field(description="Complete Python module source, without Markdown fences.")
    tests: str = Field(description="Complete Python test module source, without Markdown fences.")


DEFAULT_CODING_PROMPT = f"""\
You are an expert Python developer who writes small, reusable, well-tested tools
for AI agents.

For each requirement, return one ToolProposal:

`code` is a complete Python module that
- defines exactly ONE public function (named `function_name`); helper functions and
  classes must start with an underscore, Pydantic models and Enums used as parameter
  types are allowed;
- has complete type annotations for every parameter and the return value (never `Any`,
  never untyped `dict`; use a Pydantic BaseModel for structured data);
- has a Google-style docstring with a description, an `Args:` entry for every parameter
  and a `Returns:` section;
- only imports from: {", ".join(sorted(ALLOWED_IMPORTS - {"__future__"}))};
- performs pure computation: no file, network, process or environment access, no
  `open`, `eval`, `exec`, `getattr`, `type`, no dunder names, no code at module level
  other than imports, constants, classes and functions;
- raises ValueError with a clear message for invalid input.

`tests` is a complete Python module that
- imports the function with `from {TOOL_MODULE} import <function_name>`;
- defines several `test_*` functions without parameters using plain `assert`
  (no pytest, no fixtures; check exceptions with try/except);
- covers normal cases, edge cases and invalid input.

Return the source code only, without Markdown fences. If you receive validation
errors for a previous proposal, return a complete corrected proposal.
"""


class CodingAgent:
    """Wraps an agent that returns :class:`ToolProposal` objects.

    ``model`` may name any provider (``"openai:gpt-5.5"``); ``credentials`` holds
    the :class:`ProviderSettings` per provider. ``client``/``api_key`` are the
    Anthropic credentials of ``claude-agent`` 0.8.0 and apply to Anthropic models only.

    The Coding Agent only writes code; it never runs it. The underlying agent
    is available as ``coding_agent.agent``.
    """

    def __init__(
        self,
        *,
        model: str = "claude-opus-5-5",
        system_prompt: str = DEFAULT_CODING_PROMPT,
        effort: Effort | None = "high",
        max_tokens: int = 8192,
        client: Any | None = None,
        api_key: str | None = None,
        credentials: Mapping[str, ProviderSettings] | None = None,
        usage_tracker: UsageTracker | None = None,
        name: str = "Coding Agent",
    ) -> None:
        self.agent: Agent[ToolProposal] = build_agent(
            model,
            settings=credentials,
            client=client,
            api_key=api_key,
            name=name,
            system_prompt=system_prompt,
            output_format=ToolProposal,
            effort=effort,
            max_tokens=max_tokens,
            max_tool_rounds=0,
            usage_tracker=usage_tracker,
        )

    def propose(self, requirement: str, *, feedback: str | None = None) -> ToolProposal:
        """Write a new proposal, or correct the previous one if ``feedback`` is given."""
        if feedback is None:
            self.agent.reset()
            return self.agent.ask(f"Requirement:\n{requirement}")
        return self.agent.ask(
            "Your previous proposal failed validation:\n"
            f"{feedback}\n\nReturn a complete, corrected ToolProposal for the same requirement."
        )
