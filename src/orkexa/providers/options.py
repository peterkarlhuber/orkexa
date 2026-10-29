"""Typed, provider-specific options (``Agent(options=...)``).

Generic agent arguments only carry meaning every provider shares. Everything that is
specific to one API lives in these option classes and is only accepted by the
matching provider (passing ``OpenAIOptions`` to a Gemini agent is a
:class:`~orkexa.errors.ConfigurationError`). ``request_options`` are merged into the
native request as a last step; fields the runtime manages itself are rejected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

_ANTHROPIC_TOOL_CHOICES = ("auto", "none", "any", "tool")
_THINKING_DISPLAYS = ("summarized", "omitted", "updates")
#: Request fields the agent manages itself; they cannot be set via request_options.
_ANTHROPIC_RESERVED = frozenset(
    {
        "model",
        "messages",
        "max_tokens",
        "system",
        "tools",
        "tool_choice",
        "thinking",
        "stream",
        "output_format",
        "fallbacks",
    }
)
_OPENAI_RESERVED = frozenset(
    {
        "model",
        "input",
        "instructions",
        "tools",
        "tool_choice",
        "text_format",
        "text",
        "stream",
        "max_output_tokens",
    }
)

__all__ = [
    "AnthropicOptions",
    "CompatibleOptions",
    "GeminiOptions",
    "OpenAIOptions",
    "ProviderOptions",
]


@dataclass
class AnthropicOptions:
    """Anthropic Messages API options (the ``ClaudeAgent`` arguments of the same names).

    Attributes:
        thinking: ``"adaptive"``, ``"between_tools"`` (Sonnet 5.5) or ``"disabled"``.
        thinking_display: ``"summarized"``, ``"omitted"`` or ``"updates"``.
        cache_control: Automatic prompt caching (default on).
        cache_ttl: ``"5m"`` or ``"1h"``.
        eager_input_streaming: Stream tool inputs unbuffered.
        fallbacks: Server-side model fallbacks (``"default"`` or a list).
        request_options: Merged into every request (``betas`` route to the beta API).
        tool_choice: A native ``tool_choice`` dict (e.g. with
            ``disable_parallel_tool_use``); overrides the agent's generic ``tool_choice``.
    """

    thinking: Literal["adaptive", "between_tools", "disabled"] | None = None
    thinking_display: Literal["summarized", "omitted", "updates"] | None = None
    cache_control: bool = True
    cache_ttl: Literal["5m", "1h"] = "5m"
    eager_input_streaming: bool = False
    fallbacks: Literal["default"] | list[dict[str, Any]] | None = None
    request_options: dict[str, Any] = field(default_factory=dict)
    tool_choice: dict[str, Any] | None = None

    provider = "anthropic"

    def __post_init__(self) -> None:
        if self.cache_ttl not in ("5m", "1h"):
            raise ValueError("cache_ttl must be '5m' or '1h'")
        if self.tool_choice is not None and self.tool_choice.get("type") not in _ANTHROPIC_TOOL_CHOICES:
            raise ValueError("Invalid tool_choice")
        if self.thinking_display is not None and self.thinking_display not in _THINKING_DISPLAYS:
            raise ValueError(f"thinking_display must be one of {_THINKING_DISPLAYS}")
        if (
            self.fallbacks is not None
            and self.fallbacks != "default"
            and not isinstance(self.fallbacks, list)
        ):
            raise ValueError("fallbacks must be 'default' or a list like [{'model': '...'}]")
        reserved = _ANTHROPIC_RESERVED & set(self.request_options or {})
        if reserved:
            raise ValueError(f"Set these via constructor arguments, not request_options: {sorted(reserved)}")


@dataclass
class OpenAIOptions:
    """OpenAI Responses API options.

    Attributes:
        reasoning_summary: Request reasoning summaries (``"auto"``, ``"concise"``,
            ``"detailed"``); they are streamed as reasoning deltas.
        store: Whether OpenAI stores responses. Default ``False``: the conversation is
            replayed by Orkexa (including encrypted reasoning items), nothing is kept
            server-side.
        include_encrypted_reasoning: Request ``reasoning.encrypted_content`` so reasoning
            survives across tool rounds with ``store=False`` (default on).
        parallel_tool_calls: Allow several function calls per response (API default if ``None``).
        service_tier: ``"auto"``, ``"default"``, ``"flex"``, ``"priority"``.
        request_options: Merged into every ``responses.create()`` call.
    """

    reasoning_summary: Literal["auto", "concise", "detailed"] | None = None
    store: bool = False
    include_encrypted_reasoning: bool = True
    parallel_tool_calls: bool | None = None
    service_tier: str | None = None
    request_options: dict[str, Any] = field(default_factory=dict)

    provider = "openai"


@dataclass
class GeminiOptions:
    """Google Gemini (``google-genai``) options.

    Attributes:
        include_thoughts: Return thought summaries (streamed as reasoning deltas).
        thinking_budget: Token budget for thinking (Gemini 2.5 models); ``effort``
            maps to ``thinking_level`` instead (Gemini 3 models). Not both.
        safety_settings: Native ``SafetySetting`` dicts.
        request_options: Merged into ``GenerateContentConfig``.
    """

    include_thoughts: bool = False
    thinking_budget: int | None = None
    safety_settings: list[dict[str, Any]] | None = None
    request_options: dict[str, Any] = field(default_factory=dict)

    provider = "gemini"


@dataclass
class CompatibleOptions:
    """Options of OpenAI-compatible endpoints (Ollama, vLLM, gateways, ...).

    Attributes:
        api: The protocol: ``"chat_completions"`` (default, the most widely implemented)
            or ``"responses"`` (only if the endpoint really implements the Responses
            API). There is no automatic switching on errors.
        structured_output: ``"json_schema"`` (native ``response_format`` with a JSON
            schema) or ``"json_object"``: the explicit fallback for endpoints without
            schema support. The schema is then added to the instructions, JSON mode is
            requested and the result is validated locally.
        max_tokens_param: Name of the output token limit (``"max_tokens"`` for most
            servers, ``"max_completion_tokens"`` for newer OpenAI-style servers).
        send_reasoning_effort: Send ``effort`` as ``reasoning_effort`` (only if the
            endpoint supports it; requires ``capabilities={"reasoning": "supported"}``).
        request_options: Merged into every request (e.g. ``{"extra_body": {...}}``).
    """

    api: Literal["chat_completions", "responses"] = "chat_completions"
    structured_output: Literal["json_schema", "json_object"] = "json_schema"
    max_tokens_param: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"
    send_reasoning_effort: bool = False
    request_options: dict[str, Any] = field(default_factory=dict)

    provider = "compatible"


ProviderOptions = AnthropicOptions | OpenAIOptions | GeminiOptions | CompatibleOptions
