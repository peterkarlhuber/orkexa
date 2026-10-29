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
from typing import Any, ClassVar

from pydantic import SecretStr

from ..content import ContentBlock
from ..core.capabilities import ModelCapabilities
from ..core.conversion import parse_block
from ..core.types import ModelRequest, ModelResponse, Part, StreamEvent
from ..errors import (
    AuthenticationError,
    ConfigurationError,
    InvalidRequestError,
    MissingDependencyError,
    ProviderConnectionError,
    ProviderError,
    ProviderServerError,
    RateLimitError,
    UnsupportedCapabilityError,
)
from ..pricing import ModelPricing, resolve_pricing
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

    def __init__(self, config: ProviderConfig | None = None, asynchronous: bool = False) -> None:
        self.config = config or ProviderConfig()
        self.asynchronous = asynchronous
        self._client: Any = self.config.client

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
        pricing: ModelPricing | None,
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
    ) -> ProviderError:
        """A redacted :class:`ProviderError` subclass chosen by HTTP status."""
        message = self.redact(f"{self.name}: {type(exc).__name__}: {exc}")
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
            return InvalidRequestError(message, **kwargs)
        return ProviderError(message, **kwargs)
