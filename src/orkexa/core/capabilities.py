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

__all__ = ["CAPABILITY_NAMES", "EFFORTS", "Capability", "ModelCapabilities", "ModelLimits", "Support"]

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
    def all(
        cls, value: Support, *, reasons: Mapping[str, str] | None = None, **overrides: Support
    ) -> ModelCapabilities:
        """Every capability set to ``value``, except ``overrides``."""
        values: dict[str, Support] = dict.fromkeys(CAPABILITY_NAMES, value)
        values.update(overrides)
        return cls(**values, reasons=dict(reasons or {}))  # type: ignore[arg-type,unused-ignore]


EFFORTS: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")
"""Every generic effort level, lowest first (``orkexa.core.types.Effort``)."""


@dataclass(frozen=True)
class ModelLimits:
    """Documented token limits and reasoning settings of one model; ``None`` means unknown.

    Unknown values are never guessed: a limit that is not documented for a model (most models
    of OpenAI-compatible endpoints, typed ids of other providers) stays ``None`` and nothing is
    checked against it.

    Attributes:
        context_window: Most tokens one request may hold in total (input and output share it).
        max_output_tokens: Most output tokens per response (``max_tokens``; reasoning counts
            against it). A request asking for more is refused before it is sent.
        efforts: The effort levels the model accepts (``()``: none, the model takes no effort).
        thinking_modes: Values of a provider's own thinking switch the model accepts (Anthropic's
            ``thinking``: ``adaptive``, ``between_tools``, ``disabled``); ``()``: not configurable.
        thinking_budget: Whether the model takes a thinking token budget instead of an effort
            (Gemini 2.5: ``GeminiOptions(thinking_budget=...)``).
        source: ``documented`` (the provider's model documentation as Orkexa knows it),
            ``declared`` (set by whoever configured the endpoint) or ``unknown``.
    """

    context_window: int | None = None
    max_output_tokens: int | None = None
    efforts: tuple[str, ...] | None = None
    thinking_modes: tuple[str, ...] | None = None
    thinking_budget: bool | None = None
    source: Literal["documented", "declared", "unknown"] = "unknown"

    def __post_init__(self) -> None:
        for name in ("context_window", "max_output_tokens"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 1):
                raise ValueError(f"{name} must be a positive integer or None")
        if self.efforts is not None:
            unknown = [effort for effort in self.efforts if effort not in EFFORTS]
            if unknown:
                raise ValueError(f"Unknown effort levels {unknown}; known: {', '.join(EFFORTS)}")

    def declared(self, **values: int | None) -> ModelLimits:
        """A copy with declared ``context_window``/``max_output_tokens`` for values that are unknown here.

        A documented value is never replaced; a declared one only fills a gap (an endpoint's
        model)."""
        changes = {
            name: value for name, value in values.items() if value is not None and getattr(self, name) is None
        }
        unknown = set(values) - {"context_window", "max_output_tokens"}
        if unknown:
            raise ValueError(
                f"Only context_window and max_output_tokens can be declared, not {sorted(unknown)}"
            )
        if not changes:
            return self
        source: Literal["documented", "declared", "unknown"] = (
            "documented" if self.source == "documented" else "declared"
        )
        return replace(
            self,
            source=source,
            context_window=changes.get("context_window", self.context_window),
            max_output_tokens=changes.get("max_output_tokens", self.max_output_tokens),
        )

    def check_max_tokens(self, max_tokens: int, *, provider: str, model: str) -> None:
        """Raise :class:`~orkexa.errors.ConfigurationError` when ``max_tokens`` exceeds the maximum."""
        if self.max_output_tokens is not None and max_tokens > self.max_output_tokens:
            from ..errors import ConfigurationError

            raise ConfigurationError(
                f"max_tokens={max_tokens} exceeds the maximum output of {provider}:{model} "
                f"({self.max_output_tokens} tokens); lower max_tokens"
            )

    def check_effort(self, effort: str, *, provider: str, model: str) -> None:
        """Raise :class:`UnsupportedCapabilityError` for an effort outside the documented levels."""
        if self.efforts is not None and effort not in self.efforts:
            accepted = ", ".join(self.efforts) if self.efforts else "none"
            raise UnsupportedCapabilityError(
                f"{provider}:{model} does not accept effort {effort!r} (accepted: {accepted})",
                capability="reasoning",
            )
