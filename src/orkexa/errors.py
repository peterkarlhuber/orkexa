"""Exception hierarchy of Orkexa.

Every exception derives from :class:`OrkexaError`. The concrete classes additionally
derive from the closest builtin exception (``TypeError``, ``ValueError``,
``RuntimeError``, ``ImportError``), so generic ``except`` clauses keep working.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .evaluation import Evaluation

__all__ = [
    "AuthenticationError",
    "CheckpointError",
    "ConcurrentTurnError",
    "ConfigurationError",
    "DataResidencyError",
    "EvaluationFailedError",
    "HistoryConversionError",
    "IncompleteResponseError",
    "InvalidRequestError",
    "MaxToolCallsError",
    "MaxToolRoundsError",
    "MemoryToolError",
    "MissingCredentialsError",
    "MissingDependencyError",
    "OrkexaError",
    "PriceNotFoundError",
    "PricingError",
    "ProviderConnectionError",
    "ProviderError",
    "ProviderServerError",
    "RateLimitError",
    "RefusalError",
    "RequestSuspended",
    "StructuredOutputError",
    "ToolCallSuspended",
    "ToolDefinitionError",
    "ToolExecutionError",
    "ToolInputError",
    "ToolResultError",
    "ToolSchemaError",
    "TurnSuspended",
    "UnexpectedResponseError",
    "UnknownCapabilityError",
    "UnsupportedCapabilityError",
]


class OrkexaError(Exception):
    """Base class of all errors raised by Orkexa."""


# ---- configuration -------------------------------------------------------------------


class ConfigurationError(OrkexaError, ValueError):
    """An agent or provider is configured inconsistently (e.g. conflicting provider names)."""


class DataResidencyError(ConfigurationError):
    """A request could break the configured data residency, or residency cannot be verified.

    Raised before anything is sent to the model, e.g. by the Bedrock provider when
    ``BedrockOptions(data_residency="eu_only")`` is set and the region, inference
    profile or endpoint is not on the allowlist or cannot be checked.
    """


class PricingError(OrkexaError, ValueError):
    """Invalid price data, an inconsistent price snapshot or an unreadable serialized snapshot."""


class PriceNotFoundError(PricingError, LookupError):
    """No price is known for a provider and model (raised only by ``require``-style lookups)."""


class MissingDependencyError(OrkexaError, ImportError):
    """An optional dependency (a provider SDK, pypdf, sqlite-vec) is not installed."""


class UnsupportedCapabilityError(OrkexaError, ValueError):
    """The provider or model does not support a requested feature."""

    def __init__(self, message: str, *, capability: str | None = None) -> None:
        super().__init__(message)
        self.capability = capability


class UnknownCapabilityError(UnsupportedCapabilityError):
    """It is unknown whether the endpoint supports a feature; it must be declared explicitly.

    Pass ``capabilities={"<name>": "supported"}`` to the agent if the endpoint supports it.
    """


class HistoryConversionError(OrkexaError, ValueError):
    """A conversation cannot be continued losslessly with another provider."""


# ---- tools ---------------------------------------------------------------------------


class ToolDefinitionError(OrkexaError, TypeError):
    """A tool has an incomplete/unsupported signature, documentation or schema."""


class ToolSchemaError(ToolDefinitionError):
    """A tool schema cannot be expressed in a provider's schema dialect without changing its meaning."""


class ToolExecutionError(OrkexaError, RuntimeError):
    """Tool arguments or return values are invalid, or the handler raised."""


class ToolInputError(ToolExecutionError):
    """The input the model produced for a tool does not match its schema.

    It goes back to the model as an error result when the input can be cut off
    (``eager_input_streaming=True``) or when the provider does not enforce tool
    schemas; otherwise it follows ``raise_tool_errors``.
    """


class ToolResultError(OrkexaError, RuntimeError):
    """A tool failed in an expected way; raise it from a tool to tell the model.

    The agent sends the message back to the model as an error result, also with
    ``raise_tool_errors=True``.
    """


class MemoryToolError(ToolResultError):
    """A memory tool command failed in an expected way (missing file, invalid path, ...).

    The agent sends the message back to the model as an error result.
    """


# ---- responses -----------------------------------------------------------------------


class IncompleteResponseError(OrkexaError, RuntimeError):
    """The model stopped early (output token limit or context window exceeded)."""


class MaxToolRoundsError(OrkexaError, RuntimeError):
    """The model kept calling tools beyond ``max_tool_rounds``."""


class MaxToolCallsError(MaxToolRoundsError):
    """One model response asked for more tool calls than ``max_tool_calls_per_round``.

    None of that response's calls ran. A subclass of :class:`MaxToolRoundsError`, so code
    that handles the round limit handles this limit too."""


class RefusalError(OrkexaError, RuntimeError):
    """The model declined the request.

    Raised for every refusal unless the agent has ``on_refusal="return"``; with
    structured output it is raised whenever no parsed output is available.

    ``stop_details`` is the provider's refusal detail (for Anthropic the native
    ``stop_details`` object, otherwise :class:`orkexa.core.types.RefusalDetails`) or
    ``None``; ``category`` is a shortcut for ``stop_details.category``.
    """

    def __init__(self, message: str, *, stop_details: Any = None) -> None:
        super().__init__(message)
        self.stop_details = stop_details

    @property
    def category(self) -> str | None:
        return getattr(self.stop_details, "category", None)


class UnexpectedResponseError(OrkexaError, RuntimeError):
    """The API response does not match what the agent loop expects."""


class StructuredOutputError(UnexpectedResponseError):
    """The model returned output that does not validate against ``output_format``."""


class ConcurrentTurnError(OrkexaError, RuntimeError):
    """``ask()`` was called while another turn of the same agent is running.

    One agent holds one conversation; use one agent per concurrent session.
    """


class ToolCallSuspended(OrkexaError, RuntimeError):
    """Raised by a ``before_tool`` hook: this call waits for a decision outside the turn.

    The turn stops before the call runs and :class:`TurnSuspended` carries a checkpoint;
    ``resume(checkpoint)`` continues the turn later without calling the model again.
    ``ref`` is the application's reference of what the call waits for (e.g. an approval id).
    """

    def __init__(self, reason: str = "", ref: str | None = None) -> None:
        super().__init__(reason or "waiting for a decision")
        self.reason = reason
        self.ref = ref


class RequestSuspended(OrkexaError, RuntimeError):
    """Raised by a usage tracker before a model request: the request waits for a decision.

    A tracker raises it from ``check_request``, ``before_request`` or ``try_reserve``
    (e.g. because the request does not fit a budget and a person may raise it). The
    request is not sent, recorded or reserved; the turn stops with
    :class:`TurnSuspended`, whose checkpoint has ``stage="request"``, and
    ``resume(checkpoint)`` admits the same request again later. ``ref`` is the
    application's reference of what the request waits for (e.g. an approval id).
    Native batch requests (``ask_batch``) cannot be suspended: the error propagates.
    """

    def __init__(self, reason: str = "", ref: str | None = None) -> None:
        super().__init__(reason or "the request waits for a decision")
        self.reason = reason
        self.ref = ref


class TurnSuspended(OrkexaError, RuntimeError):
    """``ask()`` or ``resume()`` stopped because the turn waits for a decision outside it.

    ``checkpoint`` (a :class:`~orkexa.core.checkpoint.TurnCheckpoint`) holds everything
    to continue. Its ``stage`` says where the turn stopped:

    * ``"tools"``: a tool call waits (:class:`ToolCallSuspended`); the checkpoint holds
      the conversation of the turn, the requested calls, the results of the calls that
      already ran and the waiting ones.
    * ``"request"``: the usage tracker suspended the next model request
      (:class:`RequestSuspended`) before it was sent; the checkpoint holds the
      conversation up to that request and the tracker's ``reason`` and ``ref``.

    The committed conversation is unchanged in both cases.
    """

    def __init__(self, checkpoint: Any) -> None:
        if getattr(checkpoint, "stage", "tools") == "request":
            message = f"the turn waits before a model request: {checkpoint.reason or 'suspended'}"
        else:
            waiting = ", ".join(item.name for item in checkpoint.waiting)
            message = f"the turn waits for: {waiting}"
        super().__init__(message)
        self.checkpoint = checkpoint


class CheckpointError(OrkexaError, ValueError):
    """A turn checkpoint cannot be resumed by this agent (format, agent, model, tools or prompt differ)."""


class EvaluationFailedError(OrkexaError, RuntimeError):
    """An output did not pass its evaluation; the verdict is in ``evaluation``."""

    def __init__(self, message: str, *, evaluation: Evaluation | None = None) -> None:
        super().__init__(message)
        self.evaluation = evaluation


# ---- provider errors -----------------------------------------------------------------


class ProviderError(OrkexaError, RuntimeError):
    """A provider API call failed; raised by generic agents instead of SDK exceptions.

    The message is redacted (no API keys) and names the SDK exception type. The SDK
    exception is not attached (``__cause__`` and ``__context__`` are ``None``), so a
    traceback cannot print its request or headers.

    Attributes:
        provider: Registered provider name, e.g. ``"openai"``.
        status_code: HTTP status, if the API answered.
        request_id: The provider's request id, if known.
        retryable: Whether retrying the same request can succeed (rate limits,
            overload, server errors, timeouts).
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        status_code: int | None = None,
        request_id: str | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.request_id = request_id
        self.retryable = retryable


class AuthenticationError(ProviderError):
    """Missing, invalid or insufficient credentials (HTTP 401/403)."""


class MissingCredentialsError(AuthenticationError):
    """No credentials were found for the provider (no API key passed or in the environment).

    Raised for every provider in place of the SDK's own exception (``TypeError``,
    ``openai.OpenAIError``, ``ValueError``, botocore's ``NoCredentialsError``), before a
    request is sent. A subclass of :class:`AuthenticationError`.

    Attributes:
        variables: The environment variables the provider reads its key from.
    """

    def __init__(self, message: str, *, provider: str, variables: tuple[str, ...] = ()) -> None:
        super().__init__(message, provider=provider)
        self.variables = variables


class RateLimitError(ProviderError):
    """The provider rate-limited the request (HTTP 429)."""


class InvalidRequestError(ProviderError):
    """The provider rejected the request as invalid (HTTP 400/404/409/413/422)."""


class ProviderConnectionError(ProviderError):
    """The provider could not be reached or the request timed out."""


class ProviderServerError(ProviderError):
    """The provider failed or is overloaded (HTTP 5xx, 529)."""
