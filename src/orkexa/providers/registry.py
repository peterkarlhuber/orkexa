"""Provider registry and model resolution.

``Agent(model="openai:gpt-5.5")`` and ``Agent(provider="openai", model="gpt-5.5")``
are equivalent. A ``name:`` prefix only selects a provider if ``name`` is registered,
so model ids that contain colons (Ollama tags such as ``llama3.1:8b``, fine-tuned
``ft:...`` ids) are never split. Without prefix and ``provider=``, the model belongs
to :data:`DEFAULT_PROVIDER` (``"anthropic"``).

A mistyped provider never falls back to Anthropic silently (DX-02): ``"OpenAI:gpt-5.5"``
or ``"openai/gpt-5.5"`` (a registered name in another case or with ``/``) raise
:class:`~orkexa.ConfigurationError` with the corrected name. Agents resolve their model
``strict``: without ``provider=``, any other ``prefix:`` or ``/`` raises too (an Anthropic
model id contains neither), so ``Agent(model="llama3.1:8b")`` asks for ``provider="ollama"``.
"""

from __future__ import annotations

from collections.abc import Callable
import importlib
from typing import Any

from ..errors import ConfigurationError
from .base import Provider, ProviderConfig

__all__ = [
    "DEFAULT_PROVIDER",
    "ProviderFactory",
    "create_provider",
    "provider_names",
    "register_provider",
    "resolve_model",
    "split_model",
]

DEFAULT_PROVIDER = "anthropic"

ProviderFactory = Callable[[ProviderConfig, bool], Provider]
"""``factory(config, asynchronous) -> Provider``."""

_BUILTIN: dict[str, str] = {
    "anthropic": "orkexa.providers.anthropic:AnthropicProvider",
    "openai": "orkexa.providers.openai:OpenAIProvider",
    "gemini": "orkexa.providers.gemini:GeminiProvider",
    "bedrock": "orkexa.providers.bedrock:BedrockProvider",
    "compatible": "orkexa.providers.compatible:CompatibleProvider",
    "ollama": "orkexa.providers.compatible:OllamaProvider",
    "vllm": "orkexa.providers.compatible:VLLMProvider",
    "fake": "orkexa.providers.fake:FakeProvider",
}
_REGISTRY: dict[str, ProviderFactory | str] = dict(_BUILTIN)


def register_provider(name: str, factory: ProviderFactory, *, replace: bool = False) -> None:
    """Register a provider under ``name`` (usable as ``Agent(model="name:model")``).

    ``factory`` is a :class:`Provider` subclass or a callable ``(config, asynchronous)``.
    Built-in names can only be replaced with ``replace=True``.
    """
    if not name or not name.replace("-", "").replace("_", "").isalnum():
        raise ValueError(f"Invalid provider name {name!r}")
    if name in _REGISTRY and not replace:
        raise ValueError(f"Provider {name!r} is already registered; pass replace=True")
    _REGISTRY[name] = factory


def provider_names() -> list[str]:
    return sorted(_REGISTRY)


#: A model id per provider for error messages (examples only, not a catalog).
_EXAMPLE_MODELS = {
    "anthropic": "claude-sonnet-5-5",
    "openai": "gpt-5.5",
    "gemini": "gemini-3.8-flash",
    "bedrock": "eu.anthropic.claude-sonnet-5-5",
    "ollama": "qwen3.5:4b",
}


def split_model(model: str) -> tuple[str | None, str]:
    """``"openai:gpt-5.5"`` -> ``("openai", "gpt-5.5")``; unregistered prefixes are kept."""
    prefix, sep, rest = model.partition(":")
    if sep and prefix in _REGISTRY and rest:
        return prefix, rest
    return None, model


def resolve_model(model: str, provider: str | None = None, *, strict: bool = False) -> tuple[str, str]:
    """The ``(provider, model id)`` an agent uses.

    Without ``provider=`` and without a registered prefix the model belongs to
    :data:`DEFAULT_PROVIDER`, except for a mistyped registered name (``"OpenAI:..."``,
    ``"openai/..."``), which raises. With ``strict`` (agents, evaluators and the dynamic
    agent factory), any unregistered ``prefix:`` and any ``/`` raise as well.

    Raises:
        ConfigurationError: prefix and ``provider=`` disagree, the provider is unknown, or
            (without ``provider=``) the prefix is not a registered provider.
    """
    if not isinstance(model, str) or not model.strip():
        raise ConfigurationError("model must be a non-empty string")
    head, colon, rest = model.partition(":")
    if colon and head in _REGISTRY and not rest.strip():
        raise ConfigurationError(
            f"model={model!r} names the provider {head!r} but no model; write '{head}:<model id>', "
            f"e.g. {head + ':' + _EXAMPLE_MODELS.get(head, 'my-model')!r}"
        )
    prefix, model_id = split_model(model)
    if provider is not None and provider not in _REGISTRY:
        raise ConfigurationError(f"Unknown provider {provider!r}; registered: {', '.join(provider_names())}")
    if prefix is not None and provider is not None and prefix != provider:
        raise ConfigurationError(
            f"model={model!r} names provider {prefix!r} but provider={provider!r} was given"
        )
    if provider is None and prefix is None:
        _check_unprefixed(model, strict=strict)
    return provider or prefix or DEFAULT_PROVIDER, model_id


def _check_unprefixed(model: str, *, strict: bool) -> None:
    """Refuse a model name whose provider part is not a registered provider (no silent default)."""
    known = {name.lower(): name for name in _REGISTRY}
    head, colon, rest = model.partition(":")
    if colon and head.strip().lower() in known and rest:
        right = known[head.strip().lower()]
        raise ConfigurationError(
            f"Unknown provider prefix {head!r} in model {model!r}; did you mean {right + ':' + rest!r}?"
        )
    first, slash, tail = model.partition("/")
    if slash and ":" not in first and first.strip().lower() in known and tail:
        right = known[first.strip().lower()]
        raise ConfigurationError(
            f"Model {model!r} uses '/' to name a provider; did you mean {right + ':' + tail!r}? "
            f"(for a model id with '/', e.g. on vLLM, pass provider=...)"
        )
    if not strict:
        return
    if colon:
        raise ConfigurationError(
            f"Unknown provider prefix {head!r} in model {model!r}; "
            f"registered: {', '.join(provider_names())}. A model id that contains ':' "
            f"(e.g. an Ollama tag) needs provider=..., e.g. provider='ollama' or 'ollama:{model}'"
        )
    if slash:
        raise ConfigurationError(
            f"Model {model!r} has no provider and contains '/'; pass provider=... "
            f"(e.g. 'vllm' or 'compatible') or write it as 'provider:{model}'"
        )


def _factory(name: str) -> ProviderFactory:
    entry = _REGISTRY.get(name)
    if entry is None:
        raise ConfigurationError(f"Unknown provider {name!r}; registered: {', '.join(provider_names())}")
    if isinstance(entry, str):
        module, _, attribute = entry.partition(":")
        loaded: Any = getattr(importlib.import_module(module), attribute)
        _REGISTRY[name] = loaded
        return loaded  # type: ignore[no-any-return]
    return entry


def create_provider(
    name: str, config: ProviderConfig | None = None, *, asynchronous: bool = False
) -> Provider:
    """A new provider instance for one agent."""
    factory = _factory(name)
    provider = factory(config or ProviderConfig(), asynchronous)
    if not isinstance(provider, Provider):
        raise ConfigurationError(f"The factory of provider {name!r} did not return a Provider")
    return provider
