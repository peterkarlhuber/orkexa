"""Model selection shared by the examples: providers without an API key are skipped."""

from __future__ import annotations

import os

MODELS = {
    "anthropic": ("ANTHROPIC_API_KEY", os.environ.get("ORKEXA_ANTHROPIC_MODEL", "claude-sonnet-5-5")),
    "openai": ("OPENAI_API_KEY", os.environ.get("ORKEXA_OPENAI_MODEL", "gpt-6.1-sol")),
    "gemini": ("GEMINI_API_KEY", os.environ.get("ORKEXA_GEMINI_MODEL", "gemini-3.8-flash")),
}


def has_key(provider: str) -> bool:
    variable = MODELS[provider][0]
    return bool(os.environ.get(variable) or (provider == "gemini" and os.environ.get("GOOGLE_API_KEY")))


def available_models() -> list[str]:
    """``provider:model`` for every provider whose API key is set."""
    return [f"{provider}:{model}" for provider, (_, model) in MODELS.items() if has_key(provider)]


def model() -> str:
    """The model of the single-provider examples (``ORKEXA_MODEL`` or the first available)."""
    chosen = os.environ.get("ORKEXA_MODEL")
    if chosen:
        return chosen
    models = available_models()
    if not models:
        raise SystemExit("Set ANTHROPIC_API_KEY, OPENAI_API_KEY or GEMINI_API_KEY (or ORKEXA_MODEL).")
    return models[0]
