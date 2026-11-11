"""The provider protocol: what an adapter between the runtime and one API implements.

An adapter translates :class:`~orkexa.core.types.ModelRequest` into native SDK
arguments and native responses back into :class:`~orkexa.core.types.ModelResponse`.
It never runs tools and never contains a tool loop; the runtime
(:mod:`orkexa.core.runtime`) does.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
import os
import re
from typing import Any, ClassVar, Literal

from pydantic import SecretStr

from ..content import ContentBlock
from ..core.capabilities import ModelCapabilities, ModelLimits
from ..core.conversion import parse_block
from ..core.reasoning import ReasoningStream
from ..core.types import ModelRequest, ModelResponse, Part, StreamEvent
from ..errors import (
    AuthenticationError,
    ConfigurationError,
    InvalidRequestError,
    InvalidRequestReason,
    MissingCredentialsError,
    MissingDependencyError,
    ProviderConnectionError,
    ProviderError,
    ProviderServerError,
    RateLimitError,
    UnsupportedCapabilityError,
)
from ..pricing import PricingInput, resolve_pricing
from ..usage import UsageRecord, record_token_usage

__all__ = ["Provider", "ProviderConfig", "redact", "require_module"]

# Names of parameters and fields whose value is a credential (matched case-insensitively,
# also as a suffix: "x_api_key", "client_secret", "refresh_token", ...).
_SECRET_NAMES = r"(?:api[_-]?key|apikey|key|token|secret|password|passwd|pwd|signature|sig|credentials?|auth)"
_SECRET_PATTERNS = (
    # Well-known key formats.
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),  # OpenAI, Anthropic
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),  # Google
    re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"),  # GitHub
    re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),  # Slack
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),  # AWS access key ids
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"),  # JWTs
    # Authorization headers with or without a scheme, and bearer tokens anywhere.
    re.compile(r"(?i)(authorization[\"']?\s*[:=]\s*[\"']?(?:basic|bearer|token|digest)?\s*)[^\s\"',}]{4,}"),
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=\-]{8,}"),
    # User and password in URLs: scheme://user:password@host.
    re.compile(r"(?i)(\b[a-z][a-z0-9+.\-]*://)[^\s/@?#\"']+(?=@)"),
    # Credential parameters in query strings and fragments: ?api_key=..., &access_token=...
    re.compile(rf"(?i)([?&;#][^=&#\s\"']*?{_SECRET_NAMES}=)[^&#\s\"']+"),
    # Credential fields in headers, JSON and assignments: api_key: ..., "password": "...".
    re.compile(rf"(?i)(\b[\w-]*?{_SECRET_NAMES}[\"']?\s*[:=]\s*[\"']?)(?!\[REDACTED\])[^\s\"',}}&]{{6,}}"),
)


def redact(text: str, secrets: Sequence[str] = ()) -> str:
    """``text`` with credentials replaced by ``[REDACTED]``.

    Removes the given ``secrets`` and anything that looks like one: OpenAI/Anthropic
    (``sk-...``), Google (``AIza...``), GitHub, Slack and AWS keys, JWTs, bearer
    tokens, ``Authorization`` values, passwords in URLs (``https://user:pw@host``),
    credential query parameters (``?api_key=...``, ``&access_token=...``) and
    credential fields (``"password": "..."``, ``x-api-key: ...``).
    """
    for secret in secrets:
        if secret and len(secret) >= 4:
            text = text.replace(secret, "[REDACTED]")
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(lambda m: (m.group(1) or "" if m.groups() else "") + "[REDACTED]", text)
    return text


def require_module(module: str, extra: str) -> Any:
    """Import an optional SDK or raise :class:`MissingDependencyError` naming the extra."""
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise MissingDependencyError(
            f"This provider needs the {module!r} package: pip install 'orkexa[{extra}]'"
        ) from exc


def _shown(url: str | None) -> str | None:
    """``url`` for ``repr`` and messages: credentials in it are redacted."""
    return None if url is None else redact(url)


# Wording of provider 4xx answers (lowercased), see Provider.invalid_request_reason.
_CONTEXT_LENGTH = re.compile(
    r"context[ _-]?(length|window)|maximum context|too many tokens|prompt is too long|input is too long"
    r"|exceeds? the (model'?s? )?(maximum|context)|input token count|token limit"
)
_MODEL_NOT_FOUND = re.compile(r"model\b[^.]*\bnot found|unknown model|model[^.]*does not exist|no such model")
_MESSAGES = re.compile(
    r"\bmessages?\b|tool_call_id|tool_use|tool_result|tooluse|toolresult|function_call|tool call"
    r"|content type|\bcontents?\b|\brole\b"
)
_PARAMETER = re.compile(
    r"parameter|unsupported|unrecognized|not supported|invalid value|extra (inputs|fields)"
)


@dataclass(repr=False)
class ProviderConfig:
    """Connection settings of one provider instance.

    ``api_key`` is a :class:`~pydantic.SecretStr`: it is never shown in ``repr``,
    logs or error messages, never serialized, and only used by the provider it was
    given to. ``client`` is a preconfigured SDK client (then the other settings are
    not used to build one).
    """

    api_key: SecretStr | None = None
    base_url: str | None = None
    timeout: float | None = None
    max_retries: int | None = None
    client: Any = None
    extra: dict[str, Any] = field(default_factory=dict)
    """Additional SDK client arguments (e.g. ``default_headers``, ``organization``)."""

    def __post_init__(self) -> None:
        if isinstance(self.api_key, str):
            self.api_key = SecretStr(self.api_key)

    def __repr__(self) -> str:
        key = "None" if self.api_key is None else "'**********'"
        return (
            f"ProviderConfig(api_key={key}, base_url={_shown(self.base_url)!r}, timeout={self.timeout!r}, "
            f"max_retries={self.max_retries!r}, client={type(self.client).__name__ if self.client else None})"
        )

    def secret(self) -> str | None:
        return self.api_key.get_secret_value() if self.api_key is not None else None

    def client_kwargs(self) -> dict[str, Any]:
        """``api_key``/``base_url``/``timeout``/``max_retries`` for SDK clients that take them."""
        kwargs: dict[str, Any] = dict(self.extra)
        if self.api_key is not None:
            kwargs["api_key"] = self.secret()
        if self.base_url is not None:
            kwargs["base_url"] = self.base_url
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        if self.max_retries is not None:
            kwargs["max_retries"] = self.max_retries
        return kwargs


class Provider(ABC):
    """Adapter between :class:`~orkexa.core.runtime.AgentCore` and one model API.

    One instance serves one agent; ``asynchronous`` selects the sync or async SDK
    client. Subclasses implement :meth:`capabilities`, :meth:`prepare`,
    :meth:`call`/:meth:`acall` and :meth:`stream`/:meth:`astream`.
    """

    name: ClassVar[str]
    """Registered provider name (``"openai"``); used for history origin and usage."""
    options_type: ClassVar[type | None] = None
    """The typed options class this provider accepts (``None``: none)."""
    native_responses: ClassVar[bool] = False
    """Whether hooks receive native SDK objects (only the ``ClaudeAgent`` facade)."""
    structured_output_with_tools: ClassVar[bool] = True
    """Whether one request may offer tools and also constrain the answer to ``output_format``.

    ``False`` where the constraint suppresses tool calls (Ollama compiles the JSON schema into a
    grammar, so the model can only write JSON and never calls a tool). The runtime then offers the
    tools without the constraint and, once the model answers without a tool call, asks once more
    without tools for the structured answer.
    """
    tool_free_answer: ClassVar[Literal["tool_choice_none", "omit_tools"] | None] = "tool_choice_none"
    """How a request asks for an answer without another tool call while tools were used before.

    ``"tool_choice_none"``: the tools stay declared and ``tool_choice`` is ``"none"`` (APIs that
    honor it). ``"omit_tools"``: no tools are offered (Ollama ignores ``tool_choice``).
    ``None``: neither works (Bedrock Converse has no ``"none"`` and needs the tool config while
    the history holds tool calls). Used by ``Agent(on_max_tool_rounds="answer")``; with
    ``None`` the round limit raises :class:`~orkexa.MaxToolRoundsError` as before.
    """

    def __init__(self, config: ProviderConfig | None = None, asynchronous: bool = False) -> None:
        self.config = config or ProviderConfig()
        self.asynchronous = asynchronous
        self._client: Any = self.config.client

    def without_reasoning(self, request: ModelRequest) -> ModelRequest | None:
        """``request`` with the model's reasoning switched off, or ``None`` if this API cannot do that.

        Used once per turn when a reasoning model spent the whole output limit on reasoning
        (``Agent(on_reasoning_overflow="answer")``). Only adapters whose API has a documented
        switch for it return a request; the default is ``None`` (the response stays an
        :class:`~orkexa.IncompleteResponseError`).
        """
        return None

    def __repr__(self) -> str:
        return f"{type(self).__name__}(asynchronous={self.asynchronous}, config={self.config!r})"

    # ---- client -------------------------------------------------------------------

    @property
    def client(self) -> Any:
        """The SDK client, created on first use."""
        if self._client is None:
            self._client = self.make_client()
        return self._client

    @client.setter
    def client(self, value: Any) -> None:
        self._client = value

    def make_client(self) -> Any:
        """Build the SDK client from :attr:`config` (sync or async)."""
        raise ConfigurationError(f"{type(self).__name__} needs client=")

    @staticmethod
    def env_key(variable: str) -> str | None:
        value = os.environ.get(variable)
        return value or None

    # ---- capabilities and validation ------------------------------------------------

    @abstractmethod
    def capabilities(self, model: str) -> ModelCapabilities:
        """What ``model`` supports through this provider (before user overrides)."""

    def limits(self, model: str) -> ModelLimits:
        """Documented token limits and reasoning settings of ``model`` (all unknown by default)."""
        return ModelLimits()

    def check(self, request: ModelRequest) -> None:  # noqa: B027 - optional hook
        """Provider- and model-specific validation before anything is sent."""

    def server_tool_name(self, definition: Mapping[str, Any]) -> str:
        """The name a server tool definition occupies (for duplicate checks and removal)."""
        return str(definition.get("name") or definition.get("mcp_server_name") or "")

    def parse_block(self, block: ContentBlock) -> Part:
        """Parse a prompt block; providers with native block fields override it."""
        return parse_block(block)

    def check_options(self, options: Any) -> None:
        if options is not None and (self.options_type is None or not isinstance(options, self.options_type)):
            expected = self.options_type.__name__ if self.options_type else "no options"
            raise ConfigurationError(
                f"The {self.name} provider takes {expected}, not {type(options).__name__}"
            )

    # ---- requests -----------------------------------------------------------------

    @abstractmethod
    def prepare(self, request: ModelRequest) -> dict[str, Any]:
        """Encode ``request`` into native SDK keyword arguments."""

    @abstractmethod
    def call(self, native: dict[str, Any], request: ModelRequest) -> ModelResponse:
        """Send one non-streamed request."""

    @abstractmethod
    async def acall(self, native: dict[str, Any], request: ModelRequest) -> ModelResponse:
        """Async :meth:`call`."""

    @abstractmethod
    def stream(self, native: dict[str, Any], request: ModelRequest) -> Iterator[StreamEvent]:
        """Send one streamed request; the last event is a ``ResponseCompleted``."""

    @abstractmethod
    def astream(self, native: dict[str, Any], request: ModelRequest) -> AsyncIterator[StreamEvent]:
        """Async :meth:`stream`."""

    def auto_stream(self, max_tokens: int) -> bool:
        """Whether requests should stream although nobody asked (e.g. long outputs)."""
        return False

    def reasoning_stream(self, model: str, capabilities: ModelCapabilities | None = None) -> ReasoningStream:
        """What the stream of ``model`` transmits of its reasoning (:mod:`orkexa.core.reasoning`).

        ``capabilities`` are the model's capabilities with the caller's overrides (default: the
        provider's own). The default follows the ``reasoning`` capability only: ``none`` for a
        model that does not reason, else ``unknown`` (text is shown if it arrives, never assumed)."""
        caps = capabilities or self.capabilities(model)
        if caps.reasoning == "unsupported":
            return ReasoningStream.none(f"{self.name}:{model} does not reason")
        return ReasoningStream()

    def lenient_tool_input(self, request: ModelRequest, capabilities: ModelCapabilities) -> bool:
        """Whether invalid tool input goes back to the model even with ``raise_tool_errors``.

        True when the provider does not enforce tool schemas, so malformed arguments are
        an expected model error rather than a bug.
        """
        return not capabilities.supports("strict_tool_schemas")

    def request_capabilities(self, request: ModelRequest) -> ModelCapabilities:
        """The capabilities in effect for ``request`` (including the agent's overrides)."""
        caps = request.capabilities
        return caps if isinstance(caps, ModelCapabilities) else self.capabilities(request.model)

    def has_pricing(self, model: str) -> bool:
        """Whether a price for ``model`` is known (for budget pre-checks)."""
        return resolve_pricing(model, provider=self.name) is not None

    def count_tokens(self, request: ModelRequest) -> int:
        raise UnsupportedCapabilityError(
            f"{self.name} does not support token counting", capability="token_counting"
        )

    async def acount_tokens(self, request: ModelRequest) -> int:
        raise UnsupportedCapabilityError(
            f"{self.name} does not support token counting", capability="token_counting"
        )

    # ---- usage and errors ------------------------------------------------------------

    def usage_record(
        self,
        response: ModelResponse,
        *,
        agent: str,
        pricing: PricingInput,
        turn_id: str | None,
        model: str,
        batch: bool = False,
    ) -> UsageRecord | None:
        """The usage record of ``response`` (``None`` if the API reported no usage)."""
        if response.usage is None:
            return None
        return record_token_usage(
            response.usage,
            agent_name=agent,
            model=response.model or model,
            provider=self.name,
            custom_pricing=pricing,
            request_id=response.request_id,
            turn_id=turn_id,
            batch=batch,
            requested_model=model,
        )

    def run_batch(
        self,
        requests: Mapping[str, ModelRequest],
        *,
        poll_interval: float,
        timeout: float | None,
    ) -> tuple[str, dict[str, ModelResponse | str]]:
        """Run requests through a native batch API (capability ``batch``)."""
        raise UnsupportedCapabilityError(f"{self.name} has no batch API in Orkexa", capability="batch")

    def map_error(self, exc: BaseException) -> BaseException | None:
        """Translate an SDK exception into a :class:`ProviderError` (``None``: not an API error)."""
        return None

    #: The environment variables this provider's SDK reads its key from (for error messages).
    credential_variables: ClassVar[tuple[str, ...]] = ()

    def credentials_error(self, exc: BaseException) -> MissingCredentialsError | None:
        """:class:`MissingCredentialsError` when ``exc`` is the SDK's "no credentials" error.

        Agents apply it to every provider failure (also in ``ClaudeAgent``, which otherwise
        raises SDK exceptions unchanged), so a missing key is one Orkexa error everywhere."""
        if not self._is_missing_credentials(exc):
            return None
        return self.missing_credentials()

    def missing_credentials(self) -> MissingCredentialsError:
        """The error for "no credentials found" (adapters raise it instead of the SDK's own)."""
        variables = self.credential_variables
        where = f"set {' or '.join(variables)} or pass api_key=..." if variables else "configure credentials"
        return MissingCredentialsError(
            f"{self.name}: no API key or credentials found; {where}", provider=self.name, variables=variables
        )

    def _is_missing_credentials(self, exc: BaseException) -> bool:
        """Whether ``exc`` is this provider's SDK saying that it found no credentials."""
        return False

    def redact(self, text: str) -> str:
        """``text`` without this provider's API key or anything that looks like a key."""
        return redact(text, [self.config.secret() or ""])

    def provider_error(
        self,
        exc: BaseException,
        *,
        status_code: int | None,
        request_id: str | None = None,
        connection: bool = False,
        hint: str | None = None,
    ) -> ProviderError:
        """A redacted :class:`ProviderError` subclass chosen by HTTP status.

        ``hint`` (what to do about it) is appended to the message, also redacted.
        """
        message = f"{self.name}: {type(exc).__name__}: {exc}"
        if hint:
            message = f"{message.rstrip('.')}; {hint}"
        message = self.redact(message)
        kwargs: dict[str, Any] = {"provider": self.name, "status_code": status_code, "request_id": request_id}
        if connection:
            return ProviderConnectionError(message, retryable=True, **kwargs)
        if status_code in (401, 403):
            return AuthenticationError(message, **kwargs)
        if status_code == 429:
            return RateLimitError(message, retryable=True, **kwargs)
        if status_code is not None and (status_code >= 500 or status_code == 529):
            return ProviderServerError(message, retryable=True, **kwargs)
        if status_code is not None and 400 <= status_code < 500:
            reason = self.invalid_request_reason(status_code, message)
            return InvalidRequestError(message, reason=reason, **kwargs)
        return ProviderError(message, **kwargs)

    def invalid_request_reason(self, status_code: int, text: str) -> InvalidRequestReason | None:
        """What a provider's 4xx answer rejected (:attr:`InvalidRequestError.reason`).

        Read from the status and the text. The default reads the wording providers share
        (context window, unknown model, the messages, a parameter); adapters whose API says
        more precisely override it. ``None`` when the answer does not tell."""
        lowered = text.lower()
        if _CONTEXT_LENGTH.search(lowered) or status_code == 413:
            return "context_length"
        if status_code == 404 or _MODEL_NOT_FOUND.search(lowered):
            return "model_not_found"
        if _MESSAGES.search(lowered):
            return "messages"
        if _PARAMETER.search(lowered):
            return "parameter"
        return None
