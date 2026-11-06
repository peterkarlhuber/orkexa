"""Documented model limits (``ModelLimits``) and the output parameters each adapter really sends.

The scripted clients record every request, so these tests assert the native parameter names:
Anthropic ``max_tokens``, OpenAI ``max_output_tokens``, Gemini ``max_output_tokens`` and
``thinking_config``, Bedrock ``inferenceConfig.maxTokens``, OpenAI-compatible endpoints
``max_tokens``/``max_completion_tokens`` (Ollama maps it to ``num_predict``). No network access.
"""

from __future__ import annotations

import pytest

from orkexa import (
    Agent,
    CompatibleOptions,
    ConfigurationError,
    GeminiOptions,
    ModelLimits,
    UnsupportedCapabilityError,
)
from orkexa.providers import ProviderConfig
from orkexa.providers.anthropic import model_limits
from orkexa.providers.registry import create_provider

from .contract.harness import AnthropicHarness, BedrockHarness, ChatHarness, GeminiHarness, OpenAIHarness


def _limits(provider: str, model: str) -> ModelLimits:
    base_url = None if provider == "bedrock" else "http://localhost:1/v1"
    return create_provider(provider, ProviderConfig(client=object(), base_url=base_url)).limits(model)


# ---- the declarative values -------------------------------------------------------------------


def test_documented_claude_limits() -> None:
    opus = model_limits("claude-opus-5-5")
    assert (opus.context_window, opus.max_output_tokens, opus.source) == (1_000_000, 128_000, "documented")
    assert opus.thinking_modes == ("adaptive",)
    assert model_limits("claude-sonnet-5-5").thinking_modes == ("adaptive", "between_tools")
    haiku = model_limits("claude-haiku-4-5-20251001")
    assert (haiku.context_window, haiku.max_output_tokens, haiku.efforts) == (200_000, 64_000, ())
    assert model_limits("claude-sonnet-4-6").efforts == ("low", "medium", "high", "max")
    assert model_limits("claude-unknown-9") == ModelLimits()


def test_unknown_limits_are_never_guessed() -> None:
    for provider, model in (("openai", "gpt-5.5"), ("gemini", "gemini-3-pro"), ("compatible", "my-model")):
        limits = _limits(provider, model)
        assert limits.max_output_tokens is None and limits.context_window is None
    assert _limits("compatible", "my-model") == ModelLimits()


def test_reasoning_settings_by_provider() -> None:
    assert _limits("gemini", "gemini-3-pro").efforts == ("low", "medium", "high")
    assert _limits("gemini", "gemini-2.5-flash").thinking_budget is True
    assert _limits("gemini", "gemini-2.0-flash").thinking_budget is False
    assert _limits("openai", "gpt-6.1-sol").efforts == ("low", "medium", "high", "xhigh", "max")
    assert _limits("openai", "gpt-4o").efforts == ()
    assert _limits("openai", "gpt-5.5").efforts is None  # not documented to Orkexa
    bedrock = _limits("bedrock", "eu.anthropic.claude-sonnet-5-5")
    assert bedrock.thinking_modes == () and bedrock.efforts == ("low", "medium", "high", "xhigh", "max")


def test_declared_values_fill_gaps_only() -> None:
    declared = ModelLimits().declared(max_output_tokens=4096, context_window=32_768)
    assert (declared.max_output_tokens, declared.context_window, declared.source) == (
        4096,
        32_768,
        "declared",
    )
    documented = model_limits("claude-opus-5-5").declared(max_output_tokens=4096)
    assert documented.max_output_tokens == 128_000  # a documented value is never replaced
    with pytest.raises(ValueError):
        ModelLimits().declared(efforts=3)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ModelLimits(max_output_tokens=0)


# ---- checks before anything is sent -----------------------------------------------------------


def test_max_tokens_above_the_documented_maximum_is_refused_before_sending() -> None:
    harness = AnthropicHarness()
    with pytest.raises(ConfigurationError, match="128000"):
        harness.agent([harness.text("ok")], max_tokens=200_000)
    agent = harness.agent([harness.text("ok")], max_tokens=1000)
    with pytest.raises(ConfigurationError, match="128000"):
        agent.ask("hi", max_tokens=128_001)  # a per-turn limit is checked as well
    assert harness.script.calls == []


def test_haiku_4_5_takes_no_effort_and_no_adaptive_thinking() -> None:
    from orkexa import AnthropicOptions

    with pytest.raises(UnsupportedCapabilityError):
        Agent(model="claude-haiku-4-5", effort="high", client=object())
    harness = AnthropicHarness()
    harness.model = "claude-haiku-4-5"
    with pytest.raises(UnsupportedCapabilityError, match="thinking"):
        harness.agent([harness.text("ok")], effort=None, options=AnthropicOptions(thinking="adaptive"))


def test_effort_outside_the_documented_levels_is_refused() -> None:
    harness = AnthropicHarness()
    harness.model = "claude-sonnet-4-6"
    with pytest.raises(UnsupportedCapabilityError, match="xhigh"):
        harness.agent([harness.text("ok")], effort="xhigh")


def test_sonnet_5_5_rejects_disabled_but_takes_between_tools() -> None:
    from orkexa import AnthropicOptions

    harness = AnthropicHarness()
    harness.model = "claude-sonnet-5-5"
    agent = harness.agent(
        [harness.text("ok")], effort="high", options=AnthropicOptions(thinking="between_tools")
    )
    agent.ask("hi")
    assert harness.script.calls[0]["thinking"] == {"type": "between_tools"}


# ---- the parameters actually sent --------------------------------------------------------------


def test_anthropic_sends_max_tokens_and_effort() -> None:
    harness = AnthropicHarness()
    agent = harness.agent([harness.text("ok")], max_tokens=12_345, effort="high")
    agent.ask("hi")
    call = harness.script.calls[0]
    assert call["max_tokens"] == 12_345
    assert call["output_config"]["effort"] == "high"


def test_anthropic_per_turn_max_tokens_wins() -> None:
    harness = AnthropicHarness()
    agent = harness.agent([harness.text("ok")], max_tokens=4000)
    agent.ask("hi", max_tokens=9000)
    assert harness.script.calls[0]["max_tokens"] == 9000


def test_openai_sends_max_output_tokens_and_reasoning_effort() -> None:
    harness = OpenAIHarness()
    agent = harness.agent([harness.text("ok")], max_tokens=7777, effort="medium")
    agent.ask("hi")
    call = harness.script.calls[0]
    assert call["max_output_tokens"] == 7777
    assert "max_tokens" not in call
    assert call["reasoning"]["effort"] == "medium"


def test_gemini_sends_max_output_tokens_and_thinking_level() -> None:
    harness = GeminiHarness()
    agent = harness.agent([harness.text("ok")], max_tokens=6543, effort="low")
    agent.ask("hi")
    config = harness.script.calls[0]["config"]
    value = config if isinstance(config, dict) else config.model_dump()
    assert value["max_output_tokens"] == 6543
    assert value["thinking_config"]["thinking_level"] in ("LOW", "low")


def test_gemini_2_5_sends_a_thinking_budget() -> None:
    harness = GeminiHarness()
    harness.model = "gemini-2.5-flash"
    agent = harness.agent(
        [harness.text("ok")], max_tokens=5000, effort=None, options=GeminiOptions(thinking_budget=2048)
    )
    agent.ask("hi")
    config = harness.script.calls[0]["config"]
    value = config if isinstance(config, dict) else config.model_dump()
    assert value["thinking_config"]["thinking_budget"] == 2048
    assert value["max_output_tokens"] == 5000


@pytest.mark.parametrize("param", ["max_tokens", "max_completion_tokens"])
def test_compatible_endpoint_sends_the_configured_limit_name(param: str) -> None:
    harness = ChatHarness()
    agent = harness.agent(
        [harness.text("ok")],
        max_tokens=3210,
        options=CompatibleOptions(max_tokens_param=param),  # type: ignore[arg-type]
    )
    agent.ask("hi")
    call = harness.script.calls[0]
    assert call[param] == 3210
    other = "max_completion_tokens" if param == "max_tokens" else "max_tokens"
    assert other not in call


def test_bedrock_sends_max_tokens_in_inference_config() -> None:
    harness = BedrockHarness()
    agent = harness.agent([harness.text("ok")], max_tokens=4321, effort=None)
    agent.ask("hi")
    assert harness.script.calls[0]["inferenceConfig"]["maxTokens"] == 4321


def test_agent_exposes_its_limits() -> None:
    agent = Agent(model="claude-opus-5-5", client=object())
    assert agent.limits.max_output_tokens == 128_000
