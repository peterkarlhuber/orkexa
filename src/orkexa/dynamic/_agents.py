"""Building the agents of the dynamic package for any provider.

Every component (Coding Agent, Agent Architect, Orchestrator, AgentFactory) takes
a model that may be provider-qualified (``"openai:gpt-5.5"``) and per-provider
:class:`ProviderSettings`. Credentials are looked up by the provider of each
model, so a key is never handed to another provider. The legacy ``client`` and
``api_key`` arguments belong to Anthropic and are only used for Anthropic models.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import logging
from typing import Any

from ..agent import Agent
from ..claude import ClaudeAgent
from ..core.types import Effort
from ..providers.base import redact
from ..providers.registry import resolve_model

__all__ = ["ProviderSettings", "build_agent", "same_model"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, repr=False)
class ProviderSettings:
    """Connection settings for the agents of one provider.

    Attributes:
        api_key: The provider's API key (never shown in ``repr``).
        base_url: Endpoint override (required for ``compatible``); credentials in
            it (``user:password@``, ``?api_key=...``) are redacted in ``repr``.
        client: A preconfigured SDK client, used instead of ``api_key``/``base_url``.
        options: Provider options (e.g. :class:`~orkexa.OpenAIOptions`).
        capabilities: Capability overrides, e.g. ``{"tools": "supported"}`` for a
            compatible endpoint.
    """

    api_key: str | None = field(default=None, repr=False)
    base_url: str | None = None
    client: Any | None = field(default=None, repr=False)
    options: Any | None = None
    capabilities: Mapping[str, Any] | None = None

    def __repr__(self) -> str:
        key = None if self.api_key is None else "**********"
        base_url = None if self.base_url is None else redact(self.base_url)
        return (
            f"ProviderSettings(api_key={key!r}, base_url={base_url!r}, "
            f"options={self.options!r}, capabilities={self.capabilities!r})"
        )

    def agent_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        for key in ("api_key", "base_url", "client", "options", "capabilities"):
            value = getattr(self, key)
            if value is not None:
                kwargs[key] = dict(value) if key == "capabilities" else value
        return kwargs


def same_model(a: str, b: str) -> bool:
    """Whether two model names denote the same provider and model id.

    ``"claude-opus-5-5"`` and ``"anthropic:claude-opus-5-5"`` are the same model.
    """
    if a == b:
        return True
    try:
        return resolve_model(a) == resolve_model(b)
    except ValueError:
        return False


def build_agent(
    model: str,
    *,
    settings: Mapping[str, ProviderSettings] | None,
    client: Any | None,
    api_key: str | None,
    effort: Effort | None,
    **kwargs: Any,
) -> Agent[Any]:
    """An agent for ``model`` with the credentials of the model's provider.

    Anthropic models without custom settings become a :class:`~orkexa.ClaudeAgent`
    (the same runtime; it keeps the ``claude-agent`` history format and hooks);
    all other models become a :class:`~orkexa.Agent`. ``client``/``api_key`` (the
    legacy Anthropic arguments) are ignored for other providers. For a model
    whose provider reports reasoning as ``unsupported``, ``effort`` is dropped
    (and logged at INFO level): in the dynamic package it is a cost and quality
    hint chosen by a model (``AgentSpec.effort``), not a requirement. A plain
    :class:`~orkexa.Agent` with ``effort=`` still raises for such a model.
    """
    provider, _ = resolve_model(model)
    own = (settings or {}).get(provider)
    if provider == "anthropic":
        if own is None:
            return ClaudeAgent(model=model, client=client, api_key=api_key, effort=effort, **kwargs)
        merged = own.agent_kwargs()
        if set(merged) <= {"api_key", "client"}:
            return ClaudeAgent(model=model, effort=effort, **merged, **kwargs)
        return Agent(model=model, effort=effort, **merged, **kwargs)
    agent_kwargs = own.agent_kwargs() if own is not None else {}
    if effort is not None:
        probe: Agent[Any] = Agent(model=model, **agent_kwargs)
        caps = probe.adapter.capabilities(probe._model_id()).merged(probe.capability_overrides)
        if caps.reasoning == "unsupported":
            logger.info("%s does not support reasoning effort; effort=%r is not used", model, effort)
            effort = None
    return Agent(model=model, effort=effort, **agent_kwargs, **kwargs)
