"""What a provider, model or endpoint can do: ``supported``, ``unsupported`` or ``unknown``.

``unknown`` is never treated as ``supported``. The runtime calls
:meth:`ModelCapabilities.require` before it uses a feature; for ``unknown`` the error
names the override to use::

    Agent(model="compatible:my-model", base_url=..., capabilities={"tools": "supported"})
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from typing import Literal, get_args

from ..errors import UnknownCapabilityError, UnsupportedCapabilityError

__all__ = ["CAPABILITY_NAMES", "Capability", "ModelCapabilities", "Support"]

Support = Literal["supported", "unsupported", "unknown"]

Capability = Literal[
    "tools",
    "forced_tool_choice",
    "parallel_tool_calls",
    "strict_tool_schemas",
    "structured_output",
    "streaming",
    "image_input",
    "document_input",
    "reasoning",
    "caching",
    "server_tools",
    "native_memory_tool",
    "token_counting",
    "batch",
    "system_messages",
]

CAPABILITY_NAMES: tuple[str, ...] = get_args(Capability)

_DESCRIPTIONS: dict[str, str] = {
    "tools": "tool (function) calling",
    "forced_tool_choice": "forcing a tool call (tool_choice)",
    "parallel_tool_calls": "several tool calls in one response",
    "strict_tool_schemas": "server-side enforcement of tool schemas",
    "structured_output": "native structured output (output_format)",
    "streaming": "streaming",
    "image_input": "image input",
    "document_input": "document (PDF) input",
    "reasoning": "reasoning effort",
    "caching": "prompt caching",
    "server_tools": "server-side tools",
    "native_memory_tool": "the native memory tool",
    "token_counting": "token counting",
    "batch": "batch processing",
    "system_messages": "system messages inside the conversation",
}


@dataclass(frozen=True)
class ModelCapabilities:
    """Feature support of one provider/model/endpoint combination.

    ``reasons`` holds provider-specific explanations for unsupported features, used as
    error messages.
    """

    tools: Support = "unknown"
    forced_tool_choice: Support = "unknown"
    parallel_tool_calls: Support = "unknown"
    strict_tool_schemas: Support = "unknown"
    structured_output: Support = "unknown"
    streaming: Support = "unknown"
    image_input: Support = "unknown"
    document_input: Support = "unknown"
    reasoning: Support = "unknown"
    caching: Support = "unknown"
    server_tools: Support = "unknown"
    native_memory_tool: Support = "unknown"
    token_counting: Support = "unknown"
    batch: Support = "unknown"
    system_messages: Support = "unknown"
    reasons: Mapping[str, str] = field(default_factory=dict, compare=False)

    def support(self, name: str) -> Support:
        if name not in CAPABILITY_NAMES:
            raise KeyError(f"Unknown capability {name!r}; known: {', '.join(CAPABILITY_NAMES)}")
        value: Support = getattr(self, name)
        return value

    def supports(self, name: str) -> bool:
        return self.support(name) == "supported"

    def require(self, name: str, *, provider: str, model: str, feature: str | None = None) -> None:
        """Raise unless ``name`` is ``supported``.

        Raises:
            UnsupportedCapabilityError: the feature is ``unsupported``.
            UnknownCapabilityError: the feature is ``unknown`` (declare it explicitly).
        """
        status = self.support(name)
        if status == "supported":
            return
        what = feature or _DESCRIPTIONS.get(name, name)
        if status == "unsupported":
            reason = self.reasons.get(name) or f"{provider}:{model} does not support {what}"
            raise UnsupportedCapabilityError(reason, capability=name)
        raise UnknownCapabilityError(
            f"It is unknown whether {provider}:{model} supports {what}. If it does, declare it with "
            f"capabilities={{{name!r}: 'supported'}}; Orkexa does not assume support.",
            capability=name,
        )

    def merged(self, overrides: Mapping[str, Support] | None) -> ModelCapabilities:
        """A copy with explicit user overrides applied (validated)."""
        if not overrides:
            return self
        valid = {f.name for f in fields(self)} - {"reasons"}
        changes: dict[str, Support] = {}
        for name, value in overrides.items():
            if name not in valid:
                raise ValueError(f"Unknown capability {name!r}; known: {', '.join(CAPABILITY_NAMES)}")
            if value not in get_args(Support):
                raise ValueError(f"Capability {name!r} must be 'supported', 'unsupported' or 'unknown'")
            changes[name] = value
        reasons = {k: v for k, v in self.reasons.items() if k not in changes}
        return replace(self, reasons=reasons, **changes)

    @classmethod
    def all(cls, value: Support, **overrides: Support) -> ModelCapabilities:
        """Every capability set to ``value``, except ``overrides``."""
        values: dict[str, Support] = dict.fromkeys(CAPABILITY_NAMES, value)
        values.update(overrides)
        return cls(**values)  # type: ignore[arg-type]
