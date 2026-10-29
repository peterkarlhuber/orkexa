"""Model capabilities of current OpenAI and Gemini families and Anthropic batch items built like
normal requests (fake clients, no network access)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from anthropic.types import message_create_params
from anthropic.types.beta import message_create_params as beta_message_create_params
from pydantic import BaseModel
import pytest

from orkexa import (
    Agent,
    AnthropicOptions,
    ConfigurationError,
    UnknownCapabilityError,
    UnsupportedCapabilityError,
)

from .contract.harness import AnthropicHarness
from .helpers import add

# ---- OpenAI model families -----------------------------------------------------------------------


@pytest.mark.parametrize("model", ["gpt-6-astra", "gpt-6.1-sol", "gpt-6-sol", "gpt-6.1-terra", "gpt-6-luna"])
@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_gpt6_models_take_every_effort_and_media(model: str, effort: str) -> None:
    agent = Agent(model=f"openai:{model}", effort=effort, client=object())  # type: ignore[arg-type]
    capabilities = agent.capabilities
    assert capabilities.reasoning == capabilities.image_input == capabilities.document_input == "supported"
    assert agent.adapter.prepare(agent._build_request([]))["reasoning"] == {"effort": effort}


def test_unknown_openai_family_needs_a_declared_capability() -> None:
    with pytest.raises(UnknownCapabilityError):
        Agent(model="openai:future-model", effort="high", client=object())
    agent = Agent(
        model="openai:future-model", effort="high", client=object(), capabilities={"reasoning": "supported"}
    )
    assert agent.capabilities.reasoning == "supported"


def test_classic_models_still_reject_effort() -> None:
    with pytest.raises(UnsupportedCapabilityError):
        Agent(model="openai:gpt-4.1", effort="high", client=object())


# ---- Gemini model families -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "model", ["gemini-4-argon", "gemini-3.8-flash", "gemini-3.5-flash-lite", "models/gemini-4-argon"]
)
def test_current_gemini_models_take_a_thinking_level(model: str) -> None:
    agent = Agent(model=f"gemini:{model}", effort="medium", client=object())
    assert agent.capabilities.reasoning == "supported"


def test_gemini_effort_beyond_high_and_old_families_are_refused() -> None:
    with pytest.raises(UnsupportedCapabilityError, match="'high'"):
        Agent(model="gemini:gemini-4-argon", effort="max", client=object())
    with pytest.raises(UnsupportedCapabilityError, match="thinking budget"):
        Agent(model="gemini:gemini-2.5-flash", effort="high", client=object())
    with pytest.raises(UnknownCapabilityError):
        Agent(model="gemini:gemini-exp-next", effort="high", client=object())


# ---- Anthropic batches ---------------------------------------------------------------------------


class _Batches:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.created.append(kwargs)
        return SimpleNamespace(id="batch-1", processing_status="ended")

    def retrieve(self, batch_id: str) -> Any:  # pragma: no cover - the batch has ended
        raise AssertionError

    def results(self, batch_id: str) -> list[Any]:
        return []


def _agent(**kwargs: Any) -> tuple[Agent[Any], _Batches, _Batches]:
    plain, beta = _Batches(), _Batches()
    client = SimpleNamespace(
        messages=SimpleNamespace(batches=plain), beta=SimpleNamespace(messages=SimpleNamespace(batches=beta))
    )
    return Agent(model="anthropic:claude-opus-5-5", client=client, **kwargs), plain, beta


def _keys(typed_dict: Any) -> set[str]:
    return set(typed_dict.__required_keys__ | typed_dict.__optional_keys__)


WEB_SEARCH = {"type": "web_search_20250305", "name": "web_search"}


def test_batch_items_are_built_like_normal_requests() -> None:
    options = AnthropicOptions(
        thinking="adaptive",
        cache_ttl="1h",
        tool_choice={"type": "auto", "disable_parallel_tool_use": True},
        request_options={"metadata": {"user_id": "u1"}, "extra_body": {"service_tier": "auto"}},
    )
    agent, plain, beta = _agent(options=options, effort="high", server_tools=[WEB_SEARCH])
    normal = agent.adapter.prepare(agent._build_request([]))
    agent._run_batch({"case-0": "Analyze this"}, poll_interval=0, timeout=None)
    (created,) = plain.created
    assert beta.created == []
    params = created["requests"][0]["params"]
    assert set(params) <= _keys(message_create_params.MessageCreateParamsNonStreaming)
    for field in ("tools", "tool_choice", "thinking", "cache_control", "output_config", "metadata", "system"):
        assert params[field] == normal[field], field
    assert params["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert params["service_tier"] == "auto"  # extra_body becomes part of the item
    assert params["messages"] == [{"role": "user", "content": "Analyze this"}]


class _Verdict(BaseModel):
    ok: bool


def test_batch_items_with_a_beta_go_to_the_beta_batches() -> None:
    agent, plain, beta = _agent(options=AnthropicOptions(thinking_display="updates"), output_format=_Verdict)
    agent._run_batch({"case-0": "x"}, poll_interval=0, timeout=None)
    assert plain.created == []
    (created,) = beta.created
    assert created["betas"]
    params = created["requests"][0]["params"]
    assert set(params) <= _keys(beta_message_create_params.MessageCreateParamsNonStreaming)
    assert "betas" not in params and "output_format" not in params
    assert params["output_config"]["format"]["type"] == "json_schema"
    assert params["cache_control"] == {"type": "ephemeral"}  # top level, not extra_body


def test_batch_rejects_what_a_batch_item_cannot_do() -> None:
    agent, plain, _ = _agent(tools=[add])
    with pytest.raises(UnsupportedCapabilityError, match="local tools"):
        agent._run_batch({"case-0": "x"}, poll_interval=0, timeout=None)
    agent, plain, _ = _agent(options=AnthropicOptions(request_options={"extra_headers": {"x": "y"}}))
    with pytest.raises(ConfigurationError, match="extra_headers"):
        agent._run_batch({"case-0": "x"}, poll_interval=0, timeout=None)
    assert plain.created == []


def test_evaluator_batch_keeps_the_judge_configuration() -> None:
    from orkexa.evaluation import Criterion, EvaluationCase, Evaluator

    harness = AnthropicHarness()
    batches = _Batches()
    client = harness.client(False)
    client.messages.batches = batches
    evaluator = Evaluator(
        criteria=[Criterion(name="ok", description="Output is acceptable")],
        client=client,
        options=AnthropicOptions(thinking="adaptive", cache_ttl="1h"),
    )
    with pytest.raises(Exception, match="no result"):
        evaluator.evaluate_many([EvaluationCase(task="Task", output="Result")], batch=True)
    params = batches.created[0]["requests"][0]["params"]
    assert params["thinking"] == {"type": "adaptive"}
    assert params["cache_control"]["ttl"] == "1h"
    assert params["output_config"]["format"]["type"] == "json_schema"
