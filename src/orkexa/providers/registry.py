"""Provider registry and model resolution.

``Agent(model="openai:gpt-5.5")`` and ``Agent(provider="openai", model="gpt-5.5")``
are equivalent. A ``name:`` prefix only selects a provider if ``name`` is registered,
so model ids that contain colons (Ollama tags such as ``llama3.1:8b``, fine-tuned
``ft:...`` ids) are never split. Without prefix and ``provider=``, the model belongs
to :data:`DEFAULT_PROVIDER` (``"anthropic"``).
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


def split_model(model: str) -> tuple[str | None, str]:
    """``"openai:gpt-5.5"`` -> ``("openai", "gpt-5.5")``; unregistered prefixes are kept."""
    prefix, sep, rest = model.partition(":")
    if sep and prefix in _REGISTRY and rest:
        return prefix, rest
    return None, model


def resolve_model(model: str, provider: str | None = None) -> tuple[str, str]:
    """The ``(provider, model id)`` an agent uses.

    Raises:
        ConfigurationError: prefix and ``provider=`` disagree, or the provider is unknown.
    """
    if not isinstance(model, str) or not model.strip():
        raise ConfigurationError("model must be a non-empty string")
    prefix, model_id = split_model(model)
    if provider is not None and provider not in _REGISTRY:
        raise ConfigurationError(f"Unknown provider {provider!r}; registered: {', '.join(provider_names())}")
    if prefix is not None and provider is not None and prefix != provider:
        raise ConfigurationError(
            f"model={model!r} names provider {prefix!r} but provider={provider!r} was given"
        )
    return provider or prefix or DEFAULT_PROVIDER, model_id


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
