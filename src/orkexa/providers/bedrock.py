"""Amazon Bedrock adapter through the official ``boto3`` SDK (``pip install 'orkexa[bedrock]'``).

* API family: the ``bedrock-runtime`` **Converse** and **ConverseStream** operations, the
  model-independent message API of Bedrock. ``InvokeModel`` is not used.
* ``model`` is a foundation model id (``anthropic.claude-sonnet-5-5``), a cross-Region
  inference profile id (``eu.anthropic.claude-sonnet-5-5``) or a Bedrock ARN
  (foundation model, inference profile, application inference profile, provisioned
  throughput, ...). Model ids contain colons (``...-v1:0``); ``"bedrock:<id>"`` keeps them.
* Credentials come only from boto3's credential chain (environment, shared config and
  profiles, SSO, container and instance roles, ``AWS_BEARER_TOKEN_BEDROCK``) or from an
  injected ``boto3.Session``/client. ``api_key=`` is rejected, and the provider never
  reads another provider's key. Endpoint URLs configured in the environment or the
  AWS config file (``AWS_ENDPOINT_URL*``) are ignored, so signed requests and tokens go
  only to the regional AWS endpoint or an explicitly passed ``base_url``.
* boto3 is synchronous. ``AsyncAgent`` runs each Bedrock call in a dedicated worker
  thread, so the event loop never blocks; a cancelled stream is closed. A cancelled
  non-streamed call cannot be aborted inside boto3: it finishes in its thread and its
  result (and usage) is discarded.
* Tool calling is manual (the runtime runs every tool). Tool results go back as
  ``toolResult`` blocks in one user message; parallel tool calls are supported where
  the model returns several ``toolUse`` blocks.
* Reasoning blocks (Claude extended thinking) are kept in ``Part.meta["bedrock"]`` and
  replayed unchanged to the same model, which Claude requires in tool loops.
* ``BedrockOptions(data_residency="eu_only")`` verifies the processing regions of every
  model, inference profile, endpoint and guardrail before the first paid request (see
  :mod:`orkexa.providers.bedrock_residency`).
* Usage: Converse reports input, output, cache read and cache write tokens; they are
  normalized so nothing is counted twice. Prices are looked up for provider
  ``"bedrock"`` only (never Anthropic's first-party prices) and every record carries
  the tariff condition ``aws_region`` (the source region), so a price must declare
  the regions it applies to (``Tariff.of("aws_region", "eu-central-1")``).
"""

from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from contextlib import suppress
from copy import deepcopy
import hashlib
import json
import re
import threading
import time
from typing import Any, ClassVar, TypeVar
from urllib.parse import urlsplit

from ..core.capabilities import ModelCapabilities, Support
from ..core.conversion import document_text, search_result_text, validate_output
from ..core.schema import drop_null_optionals, strict_schema
from ..core.types import (
    DocumentPart,
    ImagePart,
    Message,
    ModelRequest,
    ModelResponse,
    NativePart,
    Part,
    ReasoningDelta,
    ReasoningPart,
    RefusalDetails,
    ResponseCompleted,
    SearchResultPart,
    StopReason,
    StreamEvent,
    TextDelta,
    TextPart,
    TokenUsage,
    ToolCallDelta,
    ToolCallPart,
    ToolCallStarted,
    ToolResultPart,
)
from ..errors import (
    AuthenticationError,
    ConfigurationError,
    DataResidencyError,
    HistoryConversionError,
    InvalidRequestError,
    ProviderConnectionError,
    ProviderError,
    ProviderServerError,
    RateLimitError,
    UnexpectedResponseError,
    UnsupportedCapabilityError,
)
from ..pricing import PricingInput
from ..usage import UsageRecord, record_token_usage
from .base import Provider, ProviderConfig, redact, require_module
from .bedrock_residency import (
    EU_REGIONS,
    ModelRef,
    check_destinations,
    check_endpoint,
    parse_model_id,
    profile_destinations,
    static_destinations,
    verify_guardrail,
)
from .options import BedrockOptions, check_bedrock_fields

__all__ = ["EU_REGIONS", "BedrockProvider", "bedrock_usage"]

_PROVIDER = "bedrock"
_T = TypeVar("_T")

DEFAULT_READ_TIMEOUT = 900.0
"""Seconds boto3 waits for response data (botocore's default of 60 s is too short for long
generations and for the first structured-output request, which compiles a grammar)."""
DEFAULT_CONNECT_TIMEOUT = 10.0
RESIDENCY_CACHE_SECONDS = 300.0
"""How long a verified set of destination regions is reused before it is checked again."""

#: Converse limits checked before a request is sent (Bedrock user guide, Converse API).
MAX_IMAGE_BYTES = 3_750_000
MAX_IMAGES = 20
MAX_DOCUMENT_BYTES = 4_500_000
MAX_DOCUMENTS = 5

_IMAGE_FORMATS = {"image/png": "png", "image/jpeg": "jpeg", "image/gif": "gif", "image/webp": "webp"}
_DOCUMENT_FORMATS = {
    "application/pdf": "pdf",
    "text/csv": "csv",
    "application/msword": "doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.ms-excel": "xls",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "text/html": "html",
    "text/plain": "txt",
    "text/markdown": "md",
}
_SIGNATURES: dict[str, tuple[bytes, ...]] = {
    "png": (b"\x89PNG\r\n\x1a\n",),
    "jpeg": (b"\xff\xd8\xff",),
    "gif": (b"GIF87a", b"GIF89a"),
    "pdf": (b"%PDF-",),
}
_DOCUMENT_NAME = re.compile(r"[^A-Za-z0-9\s\-()\[\]]")
_TOOL_ID = re.compile(r"^[a-zA-Z0-9_.:-]{1,64}$")
_STOP_REASONS: dict[str, StopReason] = {
    "end_turn": "end_turn",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
    "stop_sequence": "stop_sequence",
    "guardrail_intervened": "refusal",
    "content_filtered": "refusal",
    "model_context_window_exceeded": "context_window_exceeded",
}

# Error codes of bedrock-runtime, the bedrock control plane, STS and the AWS signer.
_AUTH_CODES = frozenset(
    {
        "AccessDeniedException",
        "UnrecognizedClientException",
        "InvalidSignatureException",
        "SignatureDoesNotMatch",
        "IncompleteSignature",
        "ExpiredTokenException",
        "ExpiredToken",
        "InvalidClientTokenId",
        "MissingAuthenticationTokenException",
        "UnauthorizedException",
    }
)
_RATE_CODES = frozenset(
    {
        "ThrottlingException",
        "TooManyRequestsException",
        "ServiceQuotaExceededException",
        "ModelNotReadyException",
    }
)
_SERVER_CODES = frozenset(
    {
        "InternalServerException",
        "ServiceUnavailableException",
        "ModelErrorException",
        "ModelStreamErrorException",
        "ModelTimeoutException",
    }
)
_INVALID_CODES = frozenset({"ValidationException", "ResourceNotFoundException", "ConflictException"})

# AWS session tokens, Bedrock API keys and SigV4 signatures, on top of base.redact().
_AWS_SECRETS = (
    re.compile(r"\b(?:IQoJb3JpZ2lu|FwoGZXIvYXdz|FQoGZXIvYXdz)[A-Za-z0-9+/=]{16,}"),
    re.compile(r"\bABSK[A-Za-z0-9+/=]{16,}"),
    re.compile(r"\bbedrock-api-key-[A-Za-z0-9+/=._%-]{8,}"),
    re.compile(r"(?i)(signature=)[0-9a-f]{16,}"),
)


# ---- capabilities ------------------------------------------------------------------------------

#: Claude families on Bedrock (base ids such as ``anthropic.claude-sonnet-5-5`` or
#: ``anthropic.claude-sonnet-4-5-20250929-v1:0``).
_CLAUDE = re.compile(r"^anthropic\.claude-(?:3|(?:opus|sonnet|haiku|fable|mythos)-\d)")
_CLAUDE_5 = re.compile(r"^anthropic\.claude-(?:opus|sonnet|haiku|fable|mythos)-5(?:[-.:]|$)")
#: Claude models whose Bedrock card says they cannot be forced to call a tool.
_CLAUDE_NO_FORCING = re.compile(r"^anthropic\.claude-(?:sonnet-5-5|opus-5-5|fable-5-1|mythos-5-1)(?:[-.:]|$)")
_CLAUDE_45_PLUS = re.compile(r"^anthropic\.claude-(?:opus|sonnet|haiku)-4-[5-9](?:[-.:]|$)")
_CLAUDE_45_46 = re.compile(r"^anthropic\.claude-(?:opus|sonnet|haiku)-4-[56](?:[-.:]|$)")
_CLAUDE_3_4 = re.compile(r"^anthropic\.claude-(?:3|(?:opus|sonnet|haiku)-4)")
_CLAUDE_CACHING = re.compile(r"^anthropic\.claude-(?:3-7-sonnet|(?:opus|sonnet|haiku)-[45])")
_CLAUDE_NO_VISION = re.compile(r"^anthropic\.claude-3-5-haiku")
_NOVA = re.compile(r"^amazon\.nova-(?:micro|lite|pro|premier|2-lite)(?:[-.:]|$)")
_NOVA_TEXT_ONLY = re.compile(r"^amazon\.nova-micro")

_COMMON_REASONS = {
    "system_messages": "Converse has no system messages inside the conversation; use system_prompt",
    "server_tools": "The Bedrock adapter does not support server-side (system) tools",
    "native_memory_tool": "Bedrock has no native memory tool; Orkexa offers memory as a function tool",
    "batch": "Orkexa does not implement Bedrock batch inference (S3 jobs); use evaluate_many(batch=False)",
}


def bedrock_capabilities(base_model: str | None) -> ModelCapabilities:
    """Documented Converse features of a foundation model (``unknown`` when not documented)."""
    values: dict[str, Support] = {
        "system_messages": "unsupported",
        "server_tools": "unsupported",
        "native_memory_tool": "unsupported",
        "batch": "unsupported",
    }
    reasons = dict(_COMMON_REASONS)
    model = base_model or ""
    if _CLAUDE.match(model):
        values.update(tools="supported", parallel_tool_calls="supported", streaming="supported")
        values["document_input"] = "supported"
        if _CLAUDE_NO_VISION.match(model):
            values["image_input"] = "unsupported"
            reasons["image_input"] = f"{model} has no image input"
        else:
            values["image_input"] = "supported"
        if _CLAUDE_CACHING.match(model):
            values["caching"] = "supported"
        if _CLAUDE_5.match(model):
            values.update(
                structured_output="supported",
                strict_tool_schemas="supported",
                reasoning="supported",
                token_counting="unsupported",
            )
            reasons["token_counting"] = f"{model} does not support CountTokens on bedrock-runtime"
            if _CLAUDE_NO_FORCING.match(model):
                values["forced_tool_choice"] = "unsupported"
                reasons["forced_tool_choice"] = (
                    f"{model} on Bedrock cannot be forced to call a tool (tool_choice 'auto' only)"
                )
        elif _CLAUDE_45_PLUS.match(model):
            values.update(structured_output="supported", strict_tool_schemas="supported")
            if _CLAUDE_45_46.match(model):
                values.update(token_counting="supported", forced_tool_choice="supported")
            else:
                values["token_counting"] = "unsupported"
                reasons["token_counting"] = f"{model} does not support CountTokens on bedrock-runtime"
        else:
            values["forced_tool_choice"] = "supported"
    elif _NOVA.match(model):
        values.update(
            tools="supported", streaming="supported", forced_tool_choice="supported", caching="supported"
        )
        values["structured_output"] = "unsupported"
        reasons["structured_output"] = f"{model} does not support structured outputs on Bedrock"
        if _NOVA_TEXT_ONLY.match(model):
            values.update(image_input="unsupported", document_input="unsupported")
            reasons["image_input"] = reasons["document_input"] = f"{model} accepts text only"
        else:
            values.update(image_input="supported", document_input="supported")
    return ModelCapabilities(**values, reasons=reasons)  # type: ignore[arg-type,unused-ignore]


# ---- helpers -----------------------------------------------------------------------------------


def _get(obj: Any, name: str) -> Any:
    return obj.get(name) if isinstance(obj, Mapping) else None


def _to_json(value: Any) -> Any:
    """Native block data as JSON-safe values (bytes become ``{"$bytes": base64}``)."""
    if isinstance(value, (bytes, bytearray)):
        return {"$bytes": base64.b64encode(bytes(value)).decode("ascii")}
    if isinstance(value, Mapping):
        return {str(k): _to_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json(v) for v in value]
    return value


def _from_json(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {"$bytes"} and isinstance(value["$bytes"], str):
            return base64.b64decode(value["$bytes"])
        return {k: _from_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_from_json(v) for v in value]
    return value


def _tool_id(call_id: str) -> str:
    """A Converse-valid ``toolUseId``; foreign ids that do not fit are mapped deterministically."""
    if _TOOL_ID.match(call_id):
        return call_id
    return "orkexa_" + hashlib.sha256(call_id.encode("utf-8")).hexdigest()[:40]


def _request_id(data: Any) -> str | None:
    metadata = _get(data, "ResponseMetadata")
    value = _get(metadata, "RequestId")
    return str(value) if value else None


def bedrock_usage(usage: Any, service_tier: Any = None, cache_ttl: str = "5m") -> TokenUsage | None:
    """Normalized usage: ``input_tokens`` excludes cache reads and cache writes.

    The Bedrock prompt caching guide documents that ``inputTokens`` holds only the
    tokens neither read from nor written to the cache (total input =
    ``inputTokens + cacheReadInputTokens + cacheWriteInputTokens``), so the three are
    taken as they are. ``totalTokens`` is not used. Cache writes are split by the
    ``cacheDetails`` TTLs; writes the details do not cover count for ``cache_ttl``,
    the TTL the request asked for.
    """
    if not isinstance(usage, Mapping):
        return None

    def number(name: str) -> int:
        value = usage.get(name)
        return int(value) if isinstance(value, int) and value > 0 else 0

    inputs, outputs = number("inputTokens"), number("outputTokens")
    read, write = number("cacheReadInputTokens"), number("cacheWriteInputTokens")
    write_5m = write_1h = 0
    for detail in usage.get("cacheDetails") or []:
        count = _get(detail, "inputTokens")
        if isinstance(count, int):
            if _get(detail, "ttl") == "1h":
                write_1h += count
            else:
                write_5m += count
    rest = write - write_5m - write_1h
    if rest > 0:
        if cache_ttl == "1h":
            write_1h += rest
        else:
            write_5m += rest
    tier = _get(service_tier, "type")
    return TokenUsage(
        input_tokens=inputs,
        output_tokens=outputs,
        cache_read_input_tokens=read,
        cache_creation_input_tokens=write,
        cache_creation_5m_input_tokens=write_5m,
        cache_creation_1h_input_tokens=write_1h,
        service_tier=str(tier) if tier else None,
    )


def _secure_endpoint(url: str) -> str:
    parts = urlsplit(url)
    loopback = parts.hostname in ("localhost", "127.0.0.1", "::1")
    if parts.scheme != "https" and not (parts.scheme == "http" and loopback):
        raise ConfigurationError(
            "A custom Bedrock base_url must use https (plain http only for localhost): "
            "signed AWS requests carry the access key id and session token"
        )
    return url


async def _in_thread(
    function: Callable[[threading.Event], _T], on_orphan: Callable[[_T], None] | None = None
) -> _T:
    """Run blocking ``function(cancelled)`` in a dedicated daemon thread (not the loop's executor).

    ``cancelled`` is set when the awaiting task is cancelled, so ``function`` can skip a
    paid request it has not started yet. A result that arrives after the cancellation is
    passed to ``on_orphan`` (e.g. to close a stream that was opened meanwhile).
    """
    loop = asyncio.get_running_loop()
    future: asyncio.Future[_T] = loop.create_future()
    cancelled = threading.Event()

    def orphan(result: Any) -> None:
        if on_orphan is not None:
            with suppress(Exception):
                on_orphan(result)

    def deliver(setter: Callable[[Any], None], value: Any, ok: bool) -> None:
        if not future.done():
            setter(value)
        elif ok:
            orphan(value)

    def run() -> None:
        try:
            result: Any = function(cancelled)
        except BaseException as exc:  # handed to the awaiting task
            with _closed_loop_ok():
                loop.call_soon_threadsafe(deliver, future.set_exception, exc, False)
            return
        if cancelled.is_set():
            orphan(result)
            return
        try:
            loop.call_soon_threadsafe(deliver, future.set_result, result, True)
        except RuntimeError:  # the loop is closed: nobody will receive the result
            orphan(result)

    threading.Thread(target=run, name="orkexa-bedrock", daemon=True).start()
    try:
        return await future
    except asyncio.CancelledError:
        cancelled.set()
        raise


class _closed_loop_ok:
    """Ignore ``RuntimeError`` from a loop that was closed while a thread still ran."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, kind: Any, exc: Any, tb: Any) -> bool:
        return kind is RuntimeError


def _not_cancelled(cancelled: threading.Event | None) -> None:
    if cancelled is not None and cancelled.is_set():
        raise asyncio.CancelledError("cancelled before the Bedrock request was sent")


def _contains_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(v, key) for v in value.values())
    if isinstance(value, list):
        return any(_contains_key(v, key) for v in value)
    return False


_NATIVE_MEDIA = frozenset({"image", "document", "video", "audio"})


# ---- the provider ------------------------------------------------------------------------------


class BedrockProvider(Provider):
    """Provider ``"bedrock"``: Amazon Bedrock Converse through ``boto3``."""

    name: ClassVar[str] = _PROVIDER
    options_type: ClassVar[type | None] = BedrockOptions

    def __init__(self, config: ProviderConfig | None = None, asynchronous: bool = False) -> None:
        super().__init__(config, asynchronous)
        if self.config.api_key is not None:
            raise ConfigurationError(
                "The bedrock provider authenticates with AWS credentials (boto3 credential chain, "
                "BedrockOptions(profile_name=... or session=...), or a client=), not with api_key=; "
                "Orkexa never uses another provider's key for Bedrock"
            )
        if self.config.base_url is not None:
            _secure_endpoint(self.config.base_url)
        self._default = BedrockOptions()
        self._options = self._default
        self._own: Any = None
        """The runtime client this provider built itself (any other client counts as injected)."""
        self._built_for: BedrockOptions | None = None
        self._control: Any = None
        self._destinations: dict[tuple[str, str], tuple[float, tuple[str, ...]]] = {}
        self._guardrails: dict[tuple[str, str], float] = {}
        self._lock = threading.Lock()

    # ---- options and clients ------------------------------------------------------------

    def check_options(self, options: Any) -> None:
        super().check_options(options)
        # The agent's options configure the clients (region, profile, residency).
        self._options = options if isinstance(options, BedrockOptions) else self._default

    def _session(self, options: BedrockOptions) -> Any:
        if options.session is not None:
            return options.session
        boto3 = require_module("boto3", "bedrock")
        return boto3.Session(profile_name=options.profile_name, region_name=options.region_name)

    def _config(self) -> Any:
        config_module = require_module("botocore.config", "bedrock")
        timeout = self.config.timeout
        retries: dict[str, Any] = {"mode": "standard"}
        if self.config.max_retries is not None:
            retries["total_max_attempts"] = self.config.max_retries + 1
        return config_module.Config(
            read_timeout=timeout if timeout is not None else DEFAULT_READ_TIMEOUT,
            connect_timeout=min(timeout, DEFAULT_CONNECT_TIMEOUT) if timeout else DEFAULT_CONNECT_TIMEOUT,
            retries=retries,
            ignore_configured_endpoint_urls=True,
            user_agent_extra="orkexa",
        )

    def make_client(self) -> Any:
        options = self._options
        session = self._session(options)
        kwargs: dict[str, Any] = {"config": self._config()}
        if options.region_name is not None:
            kwargs["region_name"] = options.region_name
        if self.config.base_url is not None:
            kwargs["endpoint_url"] = self.config.base_url
        client = session.client("bedrock-runtime", **kwargs)
        self._own = client
        self._built_for = options
        self._control = None
        return client

    def _same_options(self, options: BedrockOptions) -> bool:
        built = self._built_for
        return built is not None and built == options and built.session is options.session

    def _runtime(self, options: BedrockOptions) -> Any:
        """The runtime client; a client built here is rebuilt when the options changed.

        A client passed as ``client=`` or assigned to ``agent.client`` is never replaced.
        """
        if self._client is not None and self._client is self._own and not self._same_options(options):
            self._client = None
        self._options = options
        return self.client

    def _control_client(self, options: BedrockOptions, region: str, injected: bool) -> Any:
        if options.control_client is not None:
            control_region = getattr(getattr(options.control_client, "meta", None), "region_name", None)
            if control_region != region:
                raise DataResidencyError(
                    f"The control_client uses region {control_region}, but the runtime client uses {region}; "
                    "destination regions depend on the source region"
                )
            return options.control_client
        if injected:
            raise DataResidencyError(
                "With an injected client, data residency checks need "
                "BedrockOptions(control_client=boto3.client('bedrock', region_name=...)) in the same region"
            )
        if self._control is None:
            self._control = self._session(options).client(
                "bedrock", region_name=region, config=self._config()
            )
        return self._control

    def _options_of(self, request: ModelRequest) -> BedrockOptions:
        options = request.options
        return options if isinstance(options, BedrockOptions) else self._default

    # ---- capabilities and checks -------------------------------------------------------------

    def capabilities(self, model: str) -> ModelCapabilities:
        return bedrock_capabilities(parse_model_id(model).base_model)

    def check(self, request: ModelRequest) -> None:
        options = self._options_of(request)
        caps = self.request_capabilities(request)
        where = {"provider": _PROVIDER, "model": request.model}
        if request.tool_choice is not None and request.tool_choice.mode == "none":
            raise ConfigurationError(
                "Bedrock Converse has no tool_choice 'none'; remove the tools for requests without them"
            )
        if request.native_tools:
            caps.require("server_tools", **where)
        if options.prompt_caching:
            caps.require("caching", **where, feature="prompt caching (cache points)")
            if options.cache_ttl == "1h" and _NOVA.match(parse_model_id(request.model).base_model or ""):
                raise UnsupportedCapabilityError(
                    "Amazon Nova models cache only for 5 minutes; use cache_ttl='5m'", capability="caching"
                )
        for tool in request.tools:
            if tool.strict and caps.supports("strict_tool_schemas"):
                strict_schema(tool.input_schema, name=tool.name)  # raises ToolSchemaError early
        if request.output_format is not None:
            strict_schema(request.output_format.model_json_schema(), name=request.output_format.__name__)
        if self._allowed(options) is not None:
            ref = parse_model_id(request.model)
            if ref.kind == "inference-profile" and request.model.rsplit("/", 1)[-1].startswith("global."):
                static_destinations(ref, ref.arn_region or "")  # raises: global profile
        self._encode_messages(request.messages, request, dry_run=True)

    # ---- residency ------------------------------------------------------------------------

    @staticmethod
    def _allowed(options: BedrockOptions) -> frozenset[str] | None:
        if options.allowed_regions is not None:
            return frozenset(options.allowed_regions)
        if options.data_residency == "eu_only":
            return EU_REGIONS
        return None

    def _ensure_residency(self, request: ModelRequest, client: Any, native: Mapping[str, Any]) -> None:
        """Verify where the request in ``native`` would be processed; raises before it is sent."""
        options = self._options_of(request)
        allowed = self._allowed(options)
        if allowed is None:
            return
        injected = client is not self._own
        meta = getattr(client, "meta", None)
        region = getattr(meta, "region_name", None)
        endpoint = getattr(meta, "endpoint_url", None)
        if not injected and self.config.base_url is None:
            endpoint = None  # boto3 built the regional default endpoint itself
        check_endpoint(region, endpoint, allowed)
        assert isinstance(region, str)
        if _contains_key(native.get("messages"), "s3Location") or self._has_url_media(request):
            raise DataResidencyError(
                "S3 and URL media sources cannot be checked for data residency; "
                "send the content as bytes (image(path), document(path))"
            )
        model_id = str(native.get("modelId", request.model))
        key = (region, model_id)
        with self._lock:
            cached = self._destinations.get(key)
        if cached is None or cached[0] < time.monotonic():
            ref = parse_model_id(model_id)
            destinations = static_destinations(ref, region)
            if destinations is None:
                destinations = self._lookup_profile(ref, options, region, injected)
            with self._lock:
                self._destinations[key] = (time.monotonic() + RESIDENCY_CACHE_SECONDS, destinations)
        else:
            destinations = cached[1]
        # Checked on every request: the allowlist may have changed since the lookup.
        check_destinations(destinations, allowed, model_id)
        guardrail = native.get("guardrailConfig")
        if guardrail is not None:
            self._verify_guardrail(dict(guardrail), options, region, injected)

    def _lookup_profile(
        self, ref: ModelRef, options: BedrockOptions, region: str, injected: bool
    ) -> tuple[str, ...]:
        control = self._control_client(options, region, injected)
        try:
            response = control.get_inference_profile(inferenceProfileIdentifier=ref.model_id)
        except Exception as exc:
            raise DataResidencyError(
                f"The destination regions of {ref.model_id} could not be verified "
                f"({self._describe(exc)}); the request was not sent. The caller needs "
                "bedrock:GetInferenceProfile in the source region."
            ) from None
        return profile_destinations(response, ref.model_id)

    def _verify_guardrail(
        self, guardrail: dict[str, Any], options: BedrockOptions, region: str, injected: bool
    ) -> None:
        identifier = str(guardrail.get("guardrailIdentifier") or "")
        if identifier.startswith("arn:"):
            guardrail_region = identifier.split(":")[3] if identifier.count(":") >= 3 else ""
            if guardrail_region != region:
                raise DataResidencyError(f"The guardrail {identifier} is not in {region}")
        version = str(guardrail.get("guardrailVersion") or "")
        key = (region, identifier + ":" + version)
        with self._lock:
            expires = self._guardrails.get(key)
        if expires is not None and expires >= time.monotonic():
            return
        control = self._control_client(options, region, injected)
        args = {"guardrailIdentifier": identifier}
        if version:
            args["guardrailVersion"] = version
        try:
            response = control.get_guardrail(**args)
        except Exception as exc:
            raise DataResidencyError(
                f"The guardrail {identifier} could not be verified ({self._describe(exc)}); "
                "the caller needs bedrock:GetGuardrail"
            ) from None
        verify_guardrail(response, identifier)
        with self._lock:
            self._guardrails[key] = time.monotonic() + RESIDENCY_CACHE_SECONDS

    @staticmethod
    def _has_url_media(request: ModelRequest) -> bool:
        for message in request.messages:
            for part in message.parts():
                children = (
                    part.content
                    if isinstance(part, ToolResultPart) and isinstance(part.content, list)
                    else [part]
                )
                for child in children:
                    if isinstance(child, (ImagePart, DocumentPart)) and child.source.type == "url":
                        return True
        return False

    def _describe(self, exc: BaseException) -> str:
        code = self._error_code(exc)
        return self.redact(code or type(exc).__name__)

    # ---- encoding ---------------------------------------------------------------------------

    def _media_bytes(self, part: ImagePart | DocumentPart) -> bytes:
        try:
            return base64.b64decode(part.source.data or "", validate=True)
        except ValueError:
            raise InvalidRequestError(
                "An image or document has invalid base64 data", provider=_PROVIDER
            ) from None

    def _image(self, part: ImagePart, counts: dict[str, int]) -> dict[str, Any]:
        source = part.source
        counts["images"] += 1
        if counts["images"] > MAX_IMAGES:
            raise InvalidRequestError(
                f"Bedrock Converse accepts at most {MAX_IMAGES} images per message", provider=_PROVIDER
            )
        if source.type == "url" and (source.url or "").startswith("s3://"):
            self._check_s3(counts)
            fmt = _IMAGE_FORMATS.get(source.media_type or "")
            if fmt is None:
                raise UnsupportedCapabilityError(
                    "An S3 image needs media_type image/png, image/jpeg, image/gif or image/webp",
                    capability="image_input",
                )
            return {"image": {"format": fmt, "source": {"s3Location": {"uri": source.url}}}}
        if source.type == "url":
            raise UnsupportedCapabilityError(
                "Bedrock cannot fetch image URLs and Orkexa does not download them; pass the image as "
                "bytes (image(path)) or as an s3:// URI",
                capability="image_input",
            )
        if source.type != "base64":
            raise HistoryConversionError(
                f"File {source.file_id!r} was uploaded to {source.provider or 'anthropic'}; "
                "Bedrock cannot read it"
            )
        fmt = _IMAGE_FORMATS.get(source.media_type or "")
        if fmt is None:
            raise UnsupportedCapabilityError(
                f"Bedrock accepts PNG, JPEG, GIF and WebP images, not {source.media_type!r}",
                capability="image_input",
            )
        data = self._media_bytes(part)
        self._check_size(data, MAX_IMAGE_BYTES, "image")
        self._check_signature(data, fmt, "image")
        return {"image": {"format": fmt, "source": {"bytes": data}}}

    def _document(self, part: DocumentPart, counts: dict[str, int], names: set[str]) -> dict[str, Any]:
        source = part.source
        if part.citations is not None and part.citations.enabled:
            raise UnsupportedCapabilityError(
                "The Bedrock adapter does not support document citations", capability="document_input"
            )
        counts["documents"] += 1
        if counts["documents"] > MAX_DOCUMENTS:
            raise InvalidRequestError(
                f"Bedrock Converse accepts at most {MAX_DOCUMENTS} documents per message", provider=_PROVIDER
            )
        fmt = _DOCUMENT_FORMATS.get(source.media_type or ("application/pdf" if source.type != "file" else ""))
        if fmt is None:
            raise UnsupportedCapabilityError(
                f"Bedrock does not accept documents of type {source.media_type!r} "
                f"(supported: {', '.join(sorted(_DOCUMENT_FORMATS))})",
                capability="document_input",
            )
        block: dict[str, Any] = {"format": fmt, "name": self._document_name(part, counts, names)}
        if source.type == "url" and (source.url or "").startswith("s3://"):
            self._check_s3(counts)
            block["source"] = {"s3Location": {"uri": source.url}}
        elif source.type == "url":
            raise UnsupportedCapabilityError(
                "Bedrock cannot fetch document URLs and Orkexa does not download them; pass the "
                "document as bytes (document(path)) or as an s3:// URI",
                capability="document_input",
            )
        elif source.type == "base64":
            data = self._media_bytes(part)
            self._check_size(data, MAX_DOCUMENT_BYTES, "document")
            self._check_signature(data, fmt, "document")
            block["source"] = {"bytes": data}
        else:
            raise HistoryConversionError(
                f"File {source.file_id!r} was uploaded to {source.provider or 'anthropic'}; "
                "Bedrock cannot read it"
            )
        if part.context:
            block["context"] = part.context
        return {"document": block}

    @staticmethod
    def _check_s3(counts: dict[str, int]) -> None:
        if not counts.get("nova"):
            raise UnsupportedCapabilityError(
                "Only Amazon Nova models read images and documents from S3 locations on Bedrock; "
                "pass the content as bytes",
                capability="image_input",
            )

    @staticmethod
    def _document_name(part: DocumentPart, counts: dict[str, int], names: set[str]) -> str:
        """A neutral, unique name (Converse allows letters, digits, single spaces, - ( ) [ ])."""
        cleaned = _DOCUMENT_NAME.sub(" ", part.title or "")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()[:180] or f"Document {counts['documents']}"
        name, number = cleaned, 2
        while name in names:
            name = f"{cleaned} ({number})"
            number += 1
        names.add(name)
        return name

    @staticmethod
    def _check_size(data: bytes, limit: int, what: str) -> None:
        if not data:
            raise InvalidRequestError(f"An empty {what} cannot be sent to Bedrock", provider=_PROVIDER)
        if len(data) > limit:
            raise InvalidRequestError(
                f"The {what} has {len(data)} bytes; Bedrock Converse accepts at most {limit} bytes",
                provider=_PROVIDER,
            )

    @staticmethod
    def _check_signature(data: bytes, fmt: str, what: str) -> None:
        signatures = _SIGNATURES.get(fmt)
        if fmt == "webp":
            valid = data[:4] == b"RIFF" and data[8:12] == b"WEBP"
        elif signatures is not None:
            valid = data.startswith(signatures)
        else:
            return
        if not valid:
            raise InvalidRequestError(
                f"The {what} content does not match its declared format {fmt!r}", provider=_PROVIDER
            )

    def _user_blocks(self, part: Part, counts: dict[str, int], names: set[str]) -> list[dict[str, Any]]:
        if isinstance(part, TextPart):
            return [{"text": part.text}] if part.text else []
        if isinstance(part, SearchResultPart):
            return [{"text": search_result_text(part)}]
        if isinstance(part, DocumentPart) and part.source.type == "text":
            return [{"text": document_text(part)}]
        if isinstance(part, ImagePart):
            return [self._image(part, counts)]
        if isinstance(part, DocumentPart):
            return [self._document(part, counts, names)]
        if isinstance(part, NativePart) and part.provider == _PROVIDER:
            return [self._native_block(part)]
        raise HistoryConversionError(f"Bedrock cannot receive a {part.type!r} part in a user message")

    @staticmethod
    def _native_block(part: NativePart) -> dict[str, Any]:
        if _NATIVE_MEDIA & set(part.data):
            raise HistoryConversionError(
                "Native Bedrock media blocks are not replayed (they would skip Orkexa's media checks); "
                "use image(...) or document(...) parts"
            )
        block: dict[str, Any] = _from_json(deepcopy(part.data))
        return block

    def _tool_result(
        self, part: ToolResultPart, counts: dict[str, int], names: set[str], status: bool
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        if isinstance(part.content, str):
            content.append({"text": part.content})
        elif part.content:
            for item in part.content:
                content.extend(self._user_blocks(item, counts, names))
        if not content:
            content.append({"text": ""})
        result: dict[str, Any] = {"toolUseId": _tool_id(part.call_id), "content": content}
        if part.is_error:
            if status:
                result["status"] = "error"
            else:
                content.insert(0, {"text": "Error:"})
        return {"toolResult": result}

    def _assistant_blocks(self, part: Part) -> list[dict[str, Any]]:
        extras = part.native(_PROVIDER)
        if isinstance(part, TextPart):
            return [{"text": part.text}] if part.text else []
        if isinstance(part, ToolCallPart):
            return [{"toolUse": {"toolUseId": _tool_id(part.id), "name": part.name, "input": part.arguments}}]
        if isinstance(part, ReasoningPart):
            block = extras.get("block")
            return [_from_json(deepcopy(block))] if isinstance(block, dict) else []
        if isinstance(part, NativePart) and part.provider == _PROVIDER:
            return [self._native_block(part)]
        raise HistoryConversionError(f"A {part.type!r} part cannot be sent to Bedrock")

    def _encode_messages(
        self, messages: Sequence[Message], request: ModelRequest, *, dry_run: bool = False
    ) -> list[dict[str, Any]]:
        """Converse ``messages``: roles alternate (adjacent messages of one role are joined)."""
        base = parse_model_id(request.model).base_model or ""
        nova = bool(_NOVA.match(base))
        status = nova or bool(_CLAUDE_3_4.match(base))
        names: set[str] = set()
        encoded: list[dict[str, Any]] = []
        counts: dict[str, int] = {}
        for message in messages:
            if not (encoded and encoded[-1]["role"] == message.role):
                # Limits apply per Converse message, i.e. after adjacent messages are joined.
                counts = {"images": 0, "documents": 0, "nova": int(nova)}
            if message.role == "system":
                raise UnsupportedCapabilityError(
                    _COMMON_REASONS["system_messages"], capability="system_messages"
                )
            blocks: list[dict[str, Any]] = []
            for part in message.parts():
                if message.role == "assistant":
                    blocks.extend(self._assistant_blocks(part))
                elif isinstance(part, ToolResultPart):
                    blocks.append(self._tool_result(part, counts, names, status))
                else:
                    blocks.extend(self._user_blocks(part, counts, names))
            if not blocks:
                continue
            if encoded and encoded[-1]["role"] == message.role:
                encoded[-1]["content"].extend(blocks)
            else:
                encoded.append({"role": message.role, "content": blocks})
        for converse in encoded:
            content = converse["content"]
            if any("document" in b for b in content) and not any("text" in b for b in content):
                raise InvalidRequestError(
                    "Bedrock Converse needs a text block next to a document; add a question or instruction",
                    provider=_PROVIDER,
                )
        if dry_run:
            return []
        return encoded

    def _tool_config(self, request: ModelRequest, caching: dict[str, Any] | None) -> dict[str, Any] | None:
        if not request.tools:
            return None
        caps = self.request_capabilities(request)
        tools: list[dict[str, Any]] = []
        for tool in request.tools:
            strict = tool.strict and caps.supports("strict_tool_schemas")
            spec: dict[str, Any] = {
                "name": tool.name,
                "inputSchema": {
                    "json": strict_schema(tool.input_schema, name=tool.name) if strict else tool.input_schema
                },
            }
            if tool.description:
                spec["description"] = tool.description
            if strict:
                spec["strict"] = True
            tools.append({"toolSpec": spec})
        if caching is not None and not _NOVA.match(parse_model_id(request.model).base_model or ""):
            tools.append(
                {"cachePoint": dict(caching)}
            )  # Nova accepts cache points only in system and messages
        config: dict[str, Any] = {"tools": tools}
        choice = request.tool_choice
        if choice is not None and choice.mode == "required":
            config["toolChoice"] = {"any": {}}
        elif choice is not None and choice.mode == "tool":
            config["toolChoice"] = {"tool": {"name": choice.name}}
        elif choice is not None and choice.mode == "auto":
            config["toolChoice"] = {"auto": {}}
        return config

    def prepare(self, request: ModelRequest) -> dict[str, Any]:
        options = self._options_of(request)
        if request.native_tools:
            raise UnsupportedCapabilityError(_COMMON_REASONS["server_tools"], capability="server_tools")
        caching: dict[str, Any] | None = None
        if options.prompt_caching:
            caching = {"type": "default"} if options.cache_ttl == "5m" else {"type": "default", "ttl": "1h"}
        messages = self._encode_messages(request.messages, request)
        if caching is not None and messages and messages[-1]["role"] == "user":
            messages[-1]["content"].append({"cachePoint": dict(caching)})
        inference: dict[str, Any] = {"maxTokens": request.max_tokens}
        if options.temperature is not None:
            inference["temperature"] = options.temperature
        if options.top_p is not None:
            inference["topP"] = options.top_p
        if options.stop_sequences:
            inference["stopSequences"] = list(options.stop_sequences)
        native: dict[str, Any] = {
            "modelId": request.model,
            "messages": messages,
            "inferenceConfig": inference,
        }
        if request.system:
            native["system"] = [{"text": request.system}]
            if caching is not None:
                native["system"].append({"cachePoint": dict(caching)})
        tool_config = self._tool_config(request, caching)
        if tool_config is not None:
            native["toolConfig"] = tool_config
        output: dict[str, Any] = {}
        if request.output_format is not None:
            name = re.sub(r"[^A-Za-z0-9_-]", "_", request.output_format.__name__)[:64] or "output"
            schema = strict_schema(request.output_format.model_json_schema(), name=name)
            output["textFormat"] = {
                "type": "json_schema",
                "structure": {"jsonSchema": {"schema": json.dumps(schema), "name": name}},
            }
        if request.effort is not None:
            output["effort"] = request.effort
        if output:
            native["outputConfig"] = output
        if options.guardrail is not None:
            native["guardrailConfig"] = dict(options.guardrail)
        if options.service_tier is not None:
            native["serviceTier"] = {"type": options.service_tier}
        if options.performance_latency is not None:
            native["performanceConfig"] = {"latency": options.performance_latency}
        if options.additional_model_request_fields:
            check_bedrock_fields(options.additional_model_request_fields)  # the dict itself is mutable
            native["additionalModelRequestFields"] = deepcopy(options.additional_model_request_fields)
        if options.request_metadata:
            native["requestMetadata"] = dict(options.request_metadata)
        return native

    # ---- responses -----------------------------------------------------------------------------

    def _strict_schemas(self, request: ModelRequest) -> dict[str, dict[str, Any]]:
        caps = self.request_capabilities(request)
        if not caps.supports("strict_tool_schemas"):
            return {}
        return {
            tool.name: strict_schema(tool.input_schema, name=tool.name)
            for tool in request.tools
            if tool.strict
        }

    def _parts(self, blocks: Sequence[Any], request: ModelRequest) -> tuple[list[Part], bool]:
        schemas = self._strict_schemas(request)
        parts: list[Part] = []
        calls = False
        for block in blocks:
            if not isinstance(block, Mapping):
                continue
            if "text" in block and len(block) == 1:
                parts.append(TextPart(text=str(block["text"])))
            elif "toolUse" in block and _get(block["toolUse"], "type") is not None:
                raise UnexpectedResponseError(
                    "Converse returned a server tool call (unsupported by the Bedrock adapter)"
                )
            elif "toolUse" in block:
                calls = True
                use = block["toolUse"]
                name = str(use.get("name", ""))
                call_id = str(use.get("toolUseId", ""))
                arguments = use.get("input")
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments) if arguments.strip() else {}
                    except json.JSONDecodeError as exc:
                        parts.append(ToolCallPart(id=call_id, name=name, parse_error=f"invalid JSON: {exc}"))
                        continue
                if arguments is None:
                    arguments = {}
                if not isinstance(arguments, dict):
                    parts.append(
                        ToolCallPart(id=call_id, name=name, parse_error="input is not a JSON object")
                    )
                    continue
                if name in schemas:
                    arguments = drop_null_optionals(arguments, schemas[name])
                parts.append(ToolCallPart(id=call_id, name=name, arguments=arguments))
            elif "reasoningContent" in block:
                reasoning = block["reasoningContent"]
                text = _get(_get(reasoning, "reasoningText"), "text")
                part = ReasoningPart(
                    text=str(text) if text else None, redacted="redactedContent" in (reasoning or {})
                )
                parts.append(part.with_native(_PROVIDER, {"block": _to_json(block)}))
            else:
                parts.append(NativePart(provider=_PROVIDER, data=_to_json(block)))
        return parts, calls

    def build(
        self,
        *,
        blocks: Sequence[Any],
        stop: str | None,
        usage: Any,
        tier: Any,
        request: ModelRequest,
        raw: Any,
        request_id: str | None,
    ) -> ModelResponse:
        parts, calls = self._parts(blocks, request)
        stop_reason: StopReason = _STOP_REASONS.get(stop or "", "other")
        if stop_reason == "end_turn" and calls:
            stop_reason = "tool_use"
        stop_details = RefusalDetails(category=stop) if stop_reason == "refusal" else None
        message = Message(role="assistant", content=parts, provider=_PROVIDER, model=request.model)
        parsed, error = None, None
        if request.output_format is not None and stop_reason == "end_turn":
            parsed, error = validate_output(request.output_format, message.text, strict_schema=True)
        return ModelResponse(
            message=message,
            stop_reason=stop_reason,
            provider=_PROVIDER,
            model=request.model,
            native_stop_reason=stop,
            parsed=parsed,
            parse_error=error,
            stop_details=stop_details,
            usage=bedrock_usage(usage, tier, self._options_of(request).cache_ttl),
            request_id=request_id,
            raw=raw,
        )

    def normalize(self, response: Any, request: ModelRequest) -> ModelResponse:
        message = _get(_get(response, "output"), "message")
        return self.build(
            blocks=_get(message, "content") or [],
            stop=_get(response, "stopReason"),
            usage=_get(response, "usage"),
            tier=_get(response, "serviceTier"),
            request=request,
            raw=response,
            request_id=_request_id(response),
        )

    # ---- calls ---------------------------------------------------------------------------------

    def _call_sync(
        self, native: dict[str, Any], request: ModelRequest, cancelled: threading.Event | None = None
    ) -> ModelResponse:
        client = self._runtime(self._options_of(request))
        self._ensure_residency(request, client, native)
        _not_cancelled(cancelled)
        return self.normalize(client.converse(**native), request)

    def call(self, native: dict[str, Any], request: ModelRequest) -> ModelResponse:
        return self._call_sync(native, request)

    async def acall(self, native: dict[str, Any], request: ModelRequest) -> ModelResponse:
        return await _in_thread(lambda cancelled: self._call_sync(native, request, cancelled))

    def _open_stream(
        self, native: dict[str, Any], request: ModelRequest, cancelled: threading.Event | None = None
    ) -> tuple[Any, str | None]:
        client = self._runtime(self._options_of(request))
        self._ensure_residency(request, client, native)
        _not_cancelled(cancelled)
        response = client.converse_stream(**native)
        stream = _get(response, "stream")
        if stream is None:
            raise UnexpectedResponseError("ConverseStream returned no event stream")
        return stream, _request_id(response)

    def stream(self, native: dict[str, Any], request: ModelRequest) -> Iterator[StreamEvent]:
        events, request_id = self._open_stream(native, request)
        state = _BedrockStream(request_id)
        try:
            for event in events:
                yield from state.translate(event)
        finally:
            _close(events)
        yield ResponseCompleted(state.final(self, request))

    async def astream(self, native: dict[str, Any], request: ModelRequest) -> AsyncIterator[StreamEvent]:
        events, request_id = await _in_thread(
            lambda cancelled: self._open_stream(native, request, cancelled),
            on_orphan=lambda opened: _close(opened[0]),  # opened after a cancellation: close it
        )
        state = _BedrockStream(request_id)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        stop = threading.Event()

        def pump() -> None:
            try:
                for event in events:
                    if stop.is_set():
                        return
                    with _closed_loop_ok():
                        loop.call_soon_threadsafe(queue.put_nowait, ("event", event))
                with _closed_loop_ok():
                    loop.call_soon_threadsafe(queue.put_nowait, ("end", None))
            except BaseException as exc:  # handed to the consuming task
                if not stop.is_set():
                    with _closed_loop_ok():
                        loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))

        threading.Thread(target=pump, name="orkexa-bedrock-stream", daemon=True).start()
        try:
            while True:
                kind, item = await queue.get()
                if kind == "end":
                    break
                if kind == "error":
                    raise item
                for translated in state.translate(item):
                    yield translated
        finally:
            stop.set()
            _close(events)  # unblocks the reading thread and releases the connection
        yield ResponseCompleted(state.final(self, request))

    # ---- token counting --------------------------------------------------------------------------

    def _count(self, request: ModelRequest, cancelled: threading.Event | None = None) -> int:
        caps = self.request_capabilities(request)
        caps.require("token_counting", provider=_PROVIDER, model=request.model)
        native = self.prepare(request)
        converse: dict[str, Any] = {k: native[k] for k in ("messages", "system", "toolConfig") if k in native}
        if "additionalModelRequestFields" in native:
            converse["additionalModelRequestFields"] = native["additionalModelRequestFields"]
        client = self._runtime(self._options_of(request))
        self._ensure_residency(request, client, native)
        _not_cancelled(cancelled)
        result = client.count_tokens(modelId=native["modelId"], input={"converse": converse})
        return int(_get(result, "inputTokens") or 0)

    def count_tokens(self, request: ModelRequest) -> int:
        return self._count(request)

    async def acount_tokens(self, request: ModelRequest) -> int:
        return await _in_thread(lambda cancelled: self._count(request, cancelled))

    # ---- usage and errors ------------------------------------------------------------------------

    def _region(self) -> str | None:
        client = self._client
        region = getattr(getattr(client, "meta", None), "region_name", None)
        return region if isinstance(region, str) else None

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
        """Usage priced for provider ``bedrock`` with the tariff condition ``aws_region``."""
        if response.usage is None:
            return None
        region = self._region()
        return record_token_usage(
            response.usage,
            agent_name=agent,
            model=response.model or model,
            provider=_PROVIDER,
            custom_pricing=pricing,
            request_id=response.request_id,
            turn_id=turn_id,
            batch=batch,
            requested_model=model,
            conditions={"aws_region": region or "unknown"},
        )

    def redact(self, text: str) -> str:
        text = redact(text)
        for pattern in _AWS_SECRETS:
            text = pattern.sub(lambda m: (m.group(1) if m.groups() else "") + "[REDACTED]", text)
        return text

    @staticmethod
    def _error_code(exc: BaseException) -> str | None:
        """The AWS error code; exception events of a stream use lowercase names (throttlingException)."""
        response = getattr(exc, "response", None)
        code = _get(_get(response, "Error"), "Code")
        return str(code)[:1].upper() + str(code)[1:] if code else None

    def map_error(self, exc: BaseException) -> BaseException | None:
        module = type(exc).__module__
        if not module.startswith(("botocore", "boto3", "urllib3")):
            return None
        name = type(exc).__name__
        response = getattr(exc, "response", None)
        code = self._error_code(exc)
        metadata = _get(response, "ResponseMetadata")
        status = _get(metadata, "HTTPStatusCode")
        status = status if isinstance(status, int) else None
        request_id = _get(metadata, "RequestId")
        detail = _get(_get(response, "Error"), "Message")
        message = self.redact(f"bedrock: {code or name}: {detail or exc}")
        kwargs: dict[str, Any] = {
            "provider": _PROVIDER,
            "status_code": status,
            "request_id": str(request_id) if request_id else None,
        }
        if code is not None:
            if code in _AUTH_CODES:
                hint = (
                    " (missing IAM permission, or model access not enabled for this account/region)"
                    if code == "AccessDeniedException"
                    else ""
                )
                return AuthenticationError(message + hint, **kwargs)
            if code in _RATE_CODES:
                return RateLimitError(message, retryable=True, **kwargs)
            if code in _SERVER_CODES:
                return ProviderServerError(message, retryable=True, **kwargs)
            if code in _INVALID_CODES:
                return InvalidRequestError(message, **kwargs)
            return self.provider_error(exc, status_code=status, request_id=kwargs["request_id"])
        if name == "NoRegionError":
            return ConfigurationError(
                "bedrock: no AWS region configured; set BedrockOptions(region_name=...) or AWS_DEFAULT_REGION"
            )
        if name == "ParamValidationError":
            return InvalidRequestError(message, **kwargs)
        if any(word in name for word in ("Credential", "Token", "SSO", "Auth")):
            return AuthenticationError(message, **kwargs)
        if any(word in name for word in ("Connect", "Timeout", "ConnectionClosed", "Endpoint", "Protocol")):
            return ProviderConnectionError(message, retryable=True, **kwargs)
        return ProviderError(message, **kwargs)


def _close(events: Any) -> None:
    close = getattr(events, "close", None)
    if close is not None:
        with suppress(Exception):  # a failing close must not hide the stream's result or error
            close()


def _stream_error(name: str, body: Any) -> BaseException:
    """An exception event inside the stream as the matching botocore ``ClientError``."""
    exceptions = require_module("botocore.exceptions", "bedrock")
    code = name[:1].upper() + name[1:]
    message = _get(body, "message") or _get(body, "Message") or ""
    error: BaseException = exceptions.ClientError(
        {"Error": {"Code": code, "Message": str(message)}}, "ConverseStream"
    )
    return error


class _BedrockStream:
    """Accumulates ConverseStream events into the blocks a Converse response would hold."""

    def __init__(self, request_id: str | None) -> None:
        self.request_id = request_id
        self.blocks: dict[int, dict[str, Any]] = {}
        self.stop: str | None = None
        self.stopped = False
        self.usage: Any = None
        self.tier: Any = None
        self.metadata = False

    def _block(self, index: int, kind: str) -> dict[str, Any]:
        block = self.blocks.get(index)
        if block is None:
            block = self.blocks[index] = {"kind": kind}
        return block

    def translate(self, event: Any) -> Iterator[StreamEvent]:
        if not isinstance(event, Mapping):
            return
        for key, body in event.items():
            if key.endswith("Exception"):
                raise _stream_error(key, body)
            index = _get(body, "contentBlockIndex")
            index = index if isinstance(index, int) else 0
            if key == "contentBlockStart":
                use = _get(_get(body, "start"), "toolUse")
                if use is not None:
                    if _get(use, "type") is not None:
                        raise UnexpectedResponseError(
                            "ConverseStream returned a server tool call (unsupported by the Bedrock adapter)"
                        )
                    block = self._block(index, "toolUse")
                    block.update(id=str(use.get("toolUseId", "")), name=str(use.get("name", "")), input="")
                    yield ToolCallStarted(call_id=block["id"], name=block["name"], index=index)
            elif key == "contentBlockDelta":
                yield from self._delta(index, _get(body, "delta") or {})
            elif key == "messageStop":
                self.stopped = True
                self.stop = _get(body, "stopReason")
            elif key == "metadata":
                self.metadata = True
                self.usage = _get(body, "usage")
                self.tier = _get(body, "serviceTier")

    def _delta(self, index: int, delta: Mapping[str, Any]) -> Iterator[StreamEvent]:
        if "text" in delta:
            block = self._block(index, "text")
            block["text"] = block.get("text", "") + str(delta["text"])
            if delta["text"]:
                yield TextDelta(str(delta["text"]))
        elif "toolUse" in delta:
            block = self._block(index, "toolUse")
            piece = str(_get(delta["toolUse"], "input") or "")
            block["input"] = block.get("input", "") + piece
            yield ToolCallDelta(call_id=str(block.get("id", "")), index=index, arguments_delta=piece)
        elif "reasoningContent" in delta:
            block = self._block(index, "reasoning")
            reasoning = delta["reasoningContent"] or {}
            if "text" in reasoning:
                block["text"] = block.get("text", "") + str(reasoning["text"])
                if reasoning["text"]:
                    yield ReasoningDelta(str(reasoning["text"]))
            if "signature" in reasoning:
                block["signature"] = block.get("signature", "") + str(reasoning["signature"])
            if "redactedContent" in reasoning:
                block["redacted"] = block.get("redacted", b"") + bytes(reasoning["redactedContent"])
        else:
            raise UnexpectedResponseError(
                f"ConverseStream returned an unsupported content delta ({', '.join(delta)})"
            )

    def native_blocks(self) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        for index in sorted(self.blocks):
            block = self.blocks[index]
            kind = block["kind"]
            if kind == "text":
                blocks.append({"text": block.get("text", "")})
            elif kind == "toolUse":
                blocks.append(
                    {"toolUse": {"toolUseId": block["id"], "name": block["name"], "input": block["input"]}}
                )
            elif "redacted" in block:
                blocks.append({"reasoningContent": {"redactedContent": block["redacted"]}})
            else:
                text: dict[str, Any] = {"text": block.get("text", "")}
                if "signature" in block:
                    text["signature"] = block["signature"]
                blocks.append({"reasoningContent": {"reasoningText": text}})
        return blocks

    def final(self, provider: BedrockProvider, request: ModelRequest) -> ModelResponse:
        if not self.stopped:
            raise ProviderConnectionError(
                "bedrock: the ConverseStream response ended before messageStop; the answer is incomplete",
                provider=_PROVIDER,
                request_id=self.request_id,
                retryable=True,
            )
        if not self.metadata:
            raise ProviderConnectionError(
                "bedrock: the ConverseStream response ended without its usage metadata",
                provider=_PROVIDER,
                request_id=self.request_id,
                retryable=True,
            )
        return provider.build(
            blocks=self.native_blocks(),
            stop=self.stop,
            usage=self.usage,
            tier=self.tier,
            request=request,
            raw=None,
            request_id=self.request_id,
        )
