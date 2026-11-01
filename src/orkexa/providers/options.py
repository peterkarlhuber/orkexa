"""Typed, provider-specific options (``Agent(options=...)``).

Generic agent arguments only carry meaning every provider shares. Everything that is
specific to one API lives in these option classes and is only accepted by the
matching provider (passing ``OpenAIOptions`` to a Gemini agent is a
:class:`~orkexa.errors.ConfigurationError`). ``request_options`` are merged into the
native request as a last step; fields the runtime manages itself are rejected.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from ..errors import ConfigurationError

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
#: Sampling fields that Anthropic API version 2026-10-21 (sent by anthropic>=2.0) rejects with
#: HTTP 400 and that the SDK no longer accepts as arguments.
_ANTHROPIC_REMOVED = ("temperature", "top_p", "top_k")
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

#: Fields of Bedrock's ``additionalModelRequestFields`` that the runtime manages itself or
#: that change what the request may do (tool definitions, output format, beta features).
_BEDROCK_RESERVED_FIELDS = frozenset(
    {
        "anthropic_version",
        "anthropic_beta",
        "messages",
        "system",
        "tools",
        "tool_choice",
        "max_tokens",
        "max_completion_tokens",
        "stream",
        "output_config",
        "output_format",
        "response_format",
        "model",
        "modelId",
        "inferenceConfig",
        "toolConfig",
        "outputConfig",
        "guardrailConfig",
        "promptVariables",
    }
)
_BEDROCK_RESIDENCY = ("eu_only",)
_BEDROCK_TIERS = ("default", "priority", "flex")
_BEDROCK_LATENCY = ("standard", "optimized")
_GUARDRAIL_KEYS = frozenset({"guardrailIdentifier", "guardrailVersion", "trace"})

__all__ = [
    "AnthropicOptions",
    "BedrockOptions",
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
        removed = [key for key in _ANTHROPIC_REMOVED if key in (self.request_options or {})]
        if removed:
            raise ConfigurationError(
                f"request_options {removed}: Anthropic API version 2026-10-21 (anthropic>=2.0) "
                "removed temperature, top_p and top_k and rejects requests that send them. "
                "Remove them; a client pinned to anthropic-version 2023-06-01 can send them "
                "via request_options={'extra_body': {...}}"
            )


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
            maps to ``thinking_level`` instead (Gemini 3 and 4 models). Not both.
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


def check_bedrock_fields(fields: Mapping[str, Any]) -> None:
    """Reject Bedrock ``additionalModelRequestFields`` the runtime manages or that widen a request."""
    reserved = _BEDROCK_RESERVED_FIELDS & set(fields or {})
    if reserved:
        raise ConfigurationError(
            "These fields are managed by the agent or change what the request may do and "
            f"cannot be set via additional_model_request_fields: {sorted(reserved)}"
        )


@dataclass(frozen=True)
class BedrockOptions:
    """Amazon Bedrock (``bedrock-runtime`` Converse API) options (immutable).

    The options are frozen because they decide the region, credentials and data
    residency of the clients; use :func:`dataclasses.replace` to derive changed options.

    AWS credentials never appear here: they come from the boto3 credential chain
    (environment, shared config/profile, SSO, container and instance roles) of
    ``session`` or of a new ``boto3.Session(profile_name=..., region_name=...)``.

    Attributes:
        region_name: AWS region of the ``bedrock-runtime`` client (default: the
            session's region, e.g. ``AWS_DEFAULT_REGION``).
        profile_name: Named profile of the shared AWS config to build the session from.
        session: A preconfigured ``boto3.Session`` (then ``profile_name`` must be ``None``).
        data_residency: ``"eu_only"`` restricts inference to the EU regions in
            ``allowed_regions`` and verifies every model/inference profile before the
            first paid request (see :mod:`orkexa.providers.bedrock`).
        allowed_regions: Explicit allowlist of regions where inference may run. With
            ``data_residency="eu_only"`` it defaults to the EU regions
            (:data:`orkexa.providers.bedrock.EU_REGIONS`) and may only narrow them;
            without it, setting it enforces the allowlist for any geography.
        control_client: A ``boto3.client("bedrock")`` for residency checks
            (``GetInferenceProfile``/``GetGuardrail``); default: built from the same
            session and region as the runtime client. Needed with an injected ``client=``.
        temperature, top_p, stop_sequences: Converse ``inferenceConfig`` values.
        guardrail: Converse ``guardrailConfig``: ``{"guardrailIdentifier": ...,
            "guardrailVersion": ..., "trace": "enabled"|"disabled"}``.
        service_tier: ``"default"``, ``"priority"`` or ``"flex"`` (Converse ``serviceTier``).
        performance_latency: ``"standard"`` or ``"optimized"`` (Converse ``performanceConfig``).
        prompt_caching: Add cache points after the system prompt, the tool definitions
            and the last message (requires the ``caching`` capability).
        cache_ttl: ``"5m"`` or ``"1h"`` for those cache points.
        additional_model_request_fields: Model-specific fields (e.g. ``{"top_k": 50}``);
            fields the runtime manages are rejected.
        request_metadata: Converse ``requestMetadata`` (key/value tags for invocation logs).
    """

    region_name: str | None = None
    profile_name: str | None = None
    session: Any = field(default=None, repr=False, compare=False)
    data_residency: Literal["eu_only"] | None = None
    allowed_regions: tuple[str, ...] | None = None
    control_client: Any = field(default=None, repr=False, compare=False)
    temperature: float | None = None
    top_p: float | None = None
    stop_sequences: tuple[str, ...] = ()
    guardrail: dict[str, str] | None = None
    service_tier: Literal["default", "priority", "flex"] | None = None
    performance_latency: Literal["standard", "optimized"] | None = None
    prompt_caching: bool = False
    cache_ttl: Literal["5m", "1h"] = "5m"
    additional_model_request_fields: dict[str, Any] = field(default_factory=dict)
    request_metadata: dict[str, str] = field(default_factory=dict)

    provider = "bedrock"

    def __post_init__(self) -> None:
        if self.session is not None and self.profile_name is not None:
            raise ConfigurationError("Pass either session= or profile_name=, not both")
        if self.data_residency is not None and self.data_residency not in _BEDROCK_RESIDENCY:
            raise ConfigurationError(f"data_residency must be one of {_BEDROCK_RESIDENCY} or None")
        if self.allowed_regions is not None:
            regions = tuple(self.allowed_regions)
            if isinstance(self.allowed_regions, str) or not regions:
                raise ConfigurationError("allowed_regions must be a non-empty tuple of region names")
            if not all(isinstance(r, str) and r == r.strip().lower() and r for r in regions):
                raise ConfigurationError(f"allowed_regions must hold lowercase region names, got {regions!r}")
            object.__setattr__(self, "allowed_regions", regions)
        if self.data_residency == "eu_only" and self.allowed_regions is not None:
            from .bedrock_residency import EU_REGIONS

            outside = sorted(set(self.allowed_regions) - EU_REGIONS)
            if outside:
                raise ConfigurationError(
                    f"data_residency='eu_only' allows only EU regions ({', '.join(sorted(EU_REGIONS))}); "
                    f"not allowed: {', '.join(outside)}"
                )
        if self.service_tier is not None and self.service_tier not in _BEDROCK_TIERS:
            raise ConfigurationError(f"service_tier must be one of {_BEDROCK_TIERS}")
        if self.performance_latency is not None and self.performance_latency not in _BEDROCK_LATENCY:
            raise ConfigurationError(f"performance_latency must be one of {_BEDROCK_LATENCY}")
        if self.cache_ttl not in ("5m", "1h"):
            raise ConfigurationError("cache_ttl must be '5m' or '1h'")
        if isinstance(self.stop_sequences, str):
            raise ConfigurationError("stop_sequences must be a tuple of strings")
        object.__setattr__(self, "stop_sequences", tuple(self.stop_sequences))
        if self.guardrail is not None:
            unknown = set(self.guardrail) - _GUARDRAIL_KEYS
            if unknown or not self.guardrail.get("guardrailIdentifier"):
                raise ConfigurationError(
                    "guardrail needs 'guardrailIdentifier' (and 'guardrailVersion'); "
                    f"allowed keys: {sorted(_GUARDRAIL_KEYS)}"
                )
        # Private copies, so later changes to the caller's dicts cannot bypass these checks.
        object.__setattr__(
            self, "additional_model_request_fields", dict(self.additional_model_request_fields)
        )
        object.__setattr__(self, "request_metadata", dict(self.request_metadata))
        if self.guardrail is not None:
            object.__setattr__(self, "guardrail", dict(self.guardrail))
        check_bedrock_fields(self.additional_model_request_fields)
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in self.request_metadata.items()):
            raise ConfigurationError("request_metadata must map strings to strings")


ProviderOptions = AnthropicOptions | OpenAIOptions | GeminiOptions | CompatibleOptions | BedrockOptions
