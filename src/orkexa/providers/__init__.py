"""Provider adapters, the provider registry and typed provider options.

Importing this package does not import any vendor SDK; adapters load their SDK when
they are first used and raise :class:`~orkexa.errors.MissingDependencyError` with
the extra to install if it is missing.
"""

from __future__ import annotations

from .base import Provider, ProviderConfig, redact
from .options import (
    AnthropicOptions,
    BedrockOptions,
    CompatibleOptions,
    GeminiOptions,
    OpenAIOptions,
    ProviderOptions,
)
from .registry import (
    DEFAULT_PROVIDER,
    create_provider,
    provider_names,
    register_provider,
    resolve_model,
    split_model,
)

__all__ = [
    "DEFAULT_PROVIDER",
    "AnthropicOptions",
    "BedrockOptions",
    "CompatibleOptions",
    "GeminiOptions",
    "OpenAIOptions",
    "Provider",
    "ProviderConfig",
    "ProviderOptions",
    "create_provider",
    "provider_names",
    "redact",
    "register_provider",
    "resolve_model",
    "split_model",
]
