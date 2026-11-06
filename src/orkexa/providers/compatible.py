"""OpenAI-compatible endpoints: any ``base_url`` (gateways, Ollama, vLLM, LM Studio, ...).

``Agent(provider="compatible", base_url="http://host:8000/v1", model="...")`` speaks
the Chat Completions protocol by default; ``CompatibleOptions(api="responses")``
selects the Responses API for servers that really implement it. There is no
automatic switching: a failing request is reported, never retried with another
protocol.

Security and capabilities:

* ``OPENAI_API_KEY`` is never read: endpoints get only the ``api_key`` passed to
  this agent (local servers need none).
* Every capability is ``unknown`` unless declared with ``capabilities=``: model
  names say nothing reliable about what a server and model support. The presets
  ``ollama`` and ``vllm`` only set the default ``base_url`` and the few features
  the server software itself provides for every model (streaming, system messages,
  structured output); tool calling still depends on the model and the server
  configuration and must be declared.
* Redirects are never followed: the endpoint was chosen by its ``base_url``, and a
  redirect would send the conversation (and the ``api_key``) on to another address.
  A redirect answer is reported as the endpoint's error. Pass your own ``client=`` to
  change that.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Any, ClassVar

from pydantic import SecretStr

from ..core.capabilities import ModelCapabilities, ModelLimits, Support
from ..core.types import ModelRequest, ModelResponse, StreamEvent
from ..errors import ConfigurationError
from .base import ProviderConfig, _shown, require_module
from .chat_completions import ChatCompletionsProvider, ChatMessageRules
from .openai import OpenAIProvider, ResponsesCodec
from .options import CompatibleOptions, OpenAIOptions

__all__ = ["CompatibleProvider", "OllamaProvider", "VLLMProvider"]

_NO_KEY = "orkexa-no-api-key"


class _CompatibleResponses(OpenAIProvider):
    """The Responses API engine of a compatible endpoint (shares the endpoint's client)."""

    default_base_url: ClassVar[str | None] = None

    def __init__(self, owner: CompatibleProvider) -> None:
        self.owner = owner
        super().__init__(owner.config, owner.asynchronous)
        self.codec = ResponsesCodec(owner.name)

    def check_endpoint(self) -> None:
        return None

    @property  # type: ignore[misc]
    def client(self) -> Any:
        return self.owner.client

    def capabilities(self, model: str) -> ModelCapabilities:
        return self.owner.capabilities(model)

    def limits(self, model: str) -> ModelLimits:
        return self.owner.limits(model)

    @staticmethod
    def _options(request: ModelRequest) -> OpenAIOptions:
        options = request.options
        extra = options.request_options if isinstance(options, CompatibleOptions) else {}
        return OpenAIOptions(store=False, include_encrypted_reasoning=False, request_options=dict(extra))

    def normalize(self, response: Any, request: ModelRequest) -> ModelResponse:
        result = super().normalize(response, request)
        result.provider = self.owner.name
        result.message = result.message.model_copy(update={"provider": self.owner.name})
        return result


class CompatibleProvider(ChatCompletionsProvider):
    """Provider ``"compatible"``: an OpenAI-compatible server at ``base_url``."""

    name: ClassVar[str] = "compatible"
    default_base_url: ClassVar[str | None] = None
    preset: ClassVar[Mapping[str, Support]] = {}

    def __init__(self, config: ProviderConfig | None = None, asynchronous: bool = False) -> None:
        config = config or ProviderConfig()
        if config.base_url is None:
            config.base_url = self.default_base_url
        if config.base_url is None and config.client is None:
            raise ConfigurationError(
                f"provider={self.name!r} needs base_url=..., e.g. 'http://localhost:8000/v1'"
            )
        if config.api_key is None:
            config.api_key = SecretStr(_NO_KEY)  # the SDK would otherwise read OPENAI_API_KEY
        super().__init__(config, asynchronous)
        self._responses = _CompatibleResponses(self)

    def make_client(self) -> Any:
        """An SDK client that never follows a redirect (see the module docstring).

        A ``http_client`` passed in the config's ``extra`` (or a whole ``client=``) is used
        as it is.
        """
        sdk = require_module("openai", "openai")
        kwargs = self.config.client_kwargs()
        if "http_client" not in kwargs:
            factory = sdk.DefaultAsyncHttpxClient if self.asynchronous else sdk.DefaultHttpxClient
            kwargs["http_client"] = factory(follow_redirects=False)
        cls = sdk.AsyncOpenAI if self.asynchronous else sdk.OpenAI
        return cls(**kwargs)

    #: What to check when the endpoint cannot be reached (appended to the connection error).
    start_hint: ClassVar[str] = "check that the server runs and that base_url is right"

    def _endpoint(self) -> str:
        """The configured base URL for messages, without credentials."""
        return _shown(self.config.base_url) or "the endpoint of the client= given"

    def map_error(self, exc: BaseException) -> BaseException | None:
        """Like the base class, with the address and what to do for connection errors and redirects."""
        try:
            sdk = require_module("openai", "openai")
        except ImportError:
            return None
        if isinstance(exc, sdk.APITimeoutError):
            hint = (
                f"{self._endpoint()} did not answer in time; a large local model may need a longer timeout="
            )
            return self.provider_error(exc, status_code=None, connection=True, hint=hint)
        if isinstance(exc, sdk.APIConnectionError):
            hint = f"{self._endpoint()} cannot be reached; {self.start_hint}"
            return self.provider_error(exc, status_code=None, connection=True, hint=hint)
        if isinstance(exc, sdk.APIStatusError) and 300 <= exc.status_code < 400:
            hint = (
                f"{self._endpoint()} answered with a redirect, which Orkexa does not follow for "
                "OpenAI-compatible endpoints; set base_url to the address it redirects to"
            )
            return self.provider_error(
                exc, status_code=exc.status_code, request_id=getattr(exc, "request_id", None), hint=hint
            )
        return super().map_error(exc)

    def capabilities(self, model: str) -> ModelCapabilities:
        return ModelCapabilities.all(
            "unknown",
            reasons={
                "native_memory_tool": "OpenAI-compatible endpoints have no native memory tool",
                "batch": "Orkexa has no batch support for OpenAI-compatible endpoints",
                "token_counting": "OpenAI-compatible endpoints have no standard token counting API",
            },
            native_memory_tool="unsupported",
            batch="unsupported",
            token_counting="unsupported",
            **dict(self.preset),
        )

    @staticmethod
    def _api(request: ModelRequest) -> str:
        options = request.options
        return options.api if isinstance(options, CompatibleOptions) else "chat_completions"

    def check(self, request: ModelRequest) -> None:
        if self._api(request) == "responses":
            if request.effort is not None and not self._options(request).send_reasoning_effort:
                raise ConfigurationError(
                    "effort on an OpenAI-compatible endpoint needs "
                    "CompatibleOptions(send_reasoning_effort=True) "
                    "(and capabilities={'reasoning': 'supported'})"
                )
            self._responses.check(request)
        else:
            super().check(request)

    def server_tool_name(self, definition: Mapping[str, Any]) -> str:
        return self._responses.server_tool_name(definition)

    def prepare(self, request: ModelRequest) -> dict[str, Any]:
        if self._api(request) == "responses":
            native = self._responses.prepare(request)
            native.pop("include", None)
            return native
        return super().prepare(request)

    def call(self, native: dict[str, Any], request: ModelRequest) -> ModelResponse:
        if self._api(request) == "responses":
            return self._responses.call(native, request)
        return super().call(native, request)

    async def acall(self, native: dict[str, Any], request: ModelRequest) -> ModelResponse:
        if self._api(request) == "responses":
            return await self._responses.acall(native, request)
        return await super().acall(native, request)

    def stream(self, native: dict[str, Any], request: ModelRequest) -> Iterator[StreamEvent]:
        if self._api(request) == "responses":
            return self._responses.stream(native, request)
        return super().stream(native, request)

    def astream(self, native: dict[str, Any], request: ModelRequest) -> AsyncIterator[StreamEvent]:
        if self._api(request) == "responses":
            return self._responses.astream(native, request)
        return super().astream(native, request)


class OllamaProvider(CompatibleProvider):
    """Provider ``"ollama"``: Ollama's OpenAI-compatible API (default ``http://localhost:11434/v1``)."""

    name: ClassVar[str] = "ollama"
    default_base_url: ClassVar[str | None] = "http://localhost:11434/v1"
    # Ollama turns response_format into a grammar: with tools offered, the model writes JSON and
    # never calls one.
    structured_output_with_tools: ClassVar[bool] = False
    # Ollama's OpenAI-compatible API reads an assistant message's ``reasoning`` as its thinking
    # (openai/openai.go: ``Message.Reasoning`` -> ``api.Message.Thinking``); the chat templates of
    # thinking models (Qwen 3.x, ...) show it again in the turns of the current tool loop.
    message_rules: ClassVar[ChatMessageRules] = ChatMessageRules(reasoning_field="reasoning")
    start_hint: ClassVar[str] = (
        "start Ollama (`ollama serve`, or `orkexa-local serve` for the managed runtime) or correct base_url"
    )
    preset: ClassVar[Mapping[str, Support]] = {
        "streaming": "supported",
        "system_messages": "supported",
        "structured_output": "supported",
    }


class VLLMProvider(CompatibleProvider):
    """Provider ``"vllm"``: a vLLM OpenAI-compatible server (default ``http://localhost:8000/v1``)."""

    name: ClassVar[str] = "vllm"
    default_base_url: ClassVar[str | None] = "http://localhost:8000/v1"
    start_hint: ClassVar[str] = "start the server (`vllm serve <model>`) or correct base_url"
    preset: ClassVar[Mapping[str, Support]] = {
        "streaming": "supported",
        "system_messages": "supported",
        "structured_output": "supported",
    }
