"""DX-02: a mistyped or unknown provider prefix raises instead of falling back to Anthropic."""

from __future__ import annotations

import pytest

from orkexa import Agent, ClaudeAgent, ConfigurationError, Evaluator
from orkexa.providers import resolve_model
from orkexa.providers.fake import FakeModel


@pytest.mark.parametrize(
    ("model", "hint"),
    [
        ("OpenAI:gpt-4o", "'openai:gpt-4o'"),
        ("Gemini:gemini-3-pro", "'gemini:gemini-3-pro'"),
        ("openai/gpt-4o", "'openai:gpt-4o'"),
        ("ANTHROPIC/claude-opus-5-5", "'anthropic:claude-opus-5-5'"),
    ],
)
def test_a_mistyped_registered_provider_always_raises(model: str, hint: str) -> None:
    with pytest.raises(ConfigurationError, match="did you mean") as caught:
        resolve_model(model)  # also the lenient default
    assert hint in str(caught.value)
    with pytest.raises(ConfigurationError, match="did you mean"):
        Agent(model=model)


@pytest.mark.parametrize("model", ["foo:bar", "llama3.1:8b", "meta-llama/Llama-3-8B", "arn:aws:bedrock:x"])
def test_agents_refuse_unknown_prefixes_and_slashes(model: str) -> None:
    with pytest.raises(ConfigurationError, match="provider"):
        Agent(model=model)
    with pytest.raises(ConfigurationError):
        Evaluator(model=model)
    with pytest.raises(ConfigurationError):
        resolve_model(model, strict=True)


def test_valid_names_still_resolve() -> None:
    assert resolve_model("claude-opus-5-5", strict=True) == ("anthropic", "claude-opus-5-5")
    assert resolve_model("openai:gpt-5.5", strict=True) == ("openai", "gpt-5.5")
    assert resolve_model("llama3.1:8b", "ollama", strict=True) == ("ollama", "llama3.1:8b")
    assert resolve_model("ollama:llama3.1:8b", strict=True) == ("ollama", "llama3.1:8b")
    assert resolve_model("meta-llama/Llama-3-8B", "vllm", strict=True) == ("vllm", "meta-llama/Llama-3-8B")
    assert resolve_model("openai/gpt-oss-20b", "vllm") == ("vllm", "openai/gpt-oss-20b")
    assert resolve_model("bedrock:arn:aws:bedrock:eu-central-1::x/y", strict=True)[0] == "bedrock"
    # The lenient default (parsing stored names) keeps unknown prefixes for the default provider.
    assert resolve_model("llama3.1:8b") == ("anthropic", "llama3.1:8b")
    assert Agent(model="llama3.1:8b", provider="fake", client=FakeModel([])).provider == "fake"
    assert ClaudeAgent(model="claude-haiku-5-5", client=object()).provider == "anthropic"
