"""Adapter rules that are not part of the shared contract: credentials, endpoints,
missing SDKs, request shapes and schema conversion. No network access."""

from __future__ import annotations

import sys

from pydantic import BaseModel
import pytest

from orkexa import (
    Agent,
    CompatibleOptions,
    ConfigurationError,
    MissingDependencyError,
    OpenAIOptions,
    ToolSchemaError,
    UnknownCapabilityError,
    UnsupportedCapabilityError,
)
from orkexa.core.schema import drop_null_optionals, inline_refs, strict_schema
from orkexa.providers import ProviderConfig, redact
from orkexa.providers.base import require_module
from orkexa.providers.registry import create_provider

from .contract.harness import SECRET, ChatHarness, GeminiHarness, OpenAIHarness
from .helpers import add


class Inner(BaseModel):
    x: int


class Outer(BaseModel):
    a: Inner
    b: list[Inner] = []
    c: str | None = None


# ---- credentials and endpoints -------------------------------------------------------


def test_compatible_needs_base_url() -> None:
    with pytest.raises(ConfigurationError, match="base_url"):
        Agent(provider="compatible", model="m")


def test_compatible_never_reads_openai_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    agent = Agent(provider="compatible", model="m", base_url="http://localhost:9000/v1")
    client = agent.client
    assert client.api_key != SECRET
    assert str(client.base_url).startswith("http://localhost:9000/v1")


def test_compatible_uses_its_own_key() -> None:
    agent = Agent(provider="compatible", model="m", base_url="https://gateway.example/v1", api_key="gw-key")
    assert agent.client.api_key == "gw-key"


@pytest.mark.parametrize(
    ("preset", "url"), [("ollama", "http://localhost:11434/v1"), ("vllm", "http://localhost:8000/v1")]
)
def test_presets_have_default_base_url(preset: str, url: str) -> None:
    agent = Agent(provider=preset, model="m")
    assert str(agent.client.base_url).rstrip("/") == url


def test_openai_custom_base_url_needs_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    with pytest.raises(ConfigurationError, match="never sends OPENAI_API_KEY"):
        Agent(provider="openai", model="gpt-5.5", base_url="https://proxy.example/v1")
    agent = Agent(
        provider="openai", model="gpt-5.5", base_url="https://proxy.example/v1", api_key="proxy-key"
    )
    assert agent.client.api_key == "proxy-key"


def test_openai_ignores_base_url_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://attacker.example/v1")
    agent = Agent(provider="openai", model="gpt-5.5")
    assert "attacker" not in str(agent.client.base_url)


def test_gemini_custom_base_url_needs_api_key() -> None:
    with pytest.raises(ConfigurationError, match="explicit api_key"):
        Agent(provider="gemini", model="gemini-3-pro", base_url="https://proxy.example")


def test_anthropic_custom_base_url_needs_api_key() -> None:
    with pytest.raises(ConfigurationError):
        Agent(provider="anthropic", model="claude-opus-5-5", base_url="https://proxy.example")


def test_api_key_is_hidden_in_repr_and_redacted() -> None:
    config = ProviderConfig(api_key=SECRET)
    assert SECRET not in repr(config)
    provider = create_provider("openai", config)
    assert SECRET not in repr(provider)
    assert redact(f"failed with {SECRET}", [SECRET]) == "failed with [REDACTED]"
    assert redact("key sk-ant-abcdefghijklmnopqrstuv1234") == "key [REDACTED]"


def test_missing_sdk_raises_with_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(MissingDependencyError, match=r"orkexa\[openai\]"):
        require_module("openai", "openai")
    with pytest.raises(MissingDependencyError, match=r"orkexa\[openai\]"):
        Agent(provider="openai", model="gpt-5.5", api_key="k").client  # noqa: B018


def test_no_client_is_created_before_use() -> None:
    agent = Agent(provider="openai", model="gpt-5.5", api_key="k")
    assert agent.adapter._client is None


# ---- capabilities ---------------------------------------------------------------------


def test_compatible_capabilities_are_unknown_until_declared() -> None:
    with pytest.raises(UnknownCapabilityError, match="capabilities="):
        Agent(provider="compatible", model="m", base_url="http://localhost:9000/v1", tools=[add])
    Agent(
        provider="compatible",
        model="m",
        base_url="http://localhost:9000/v1",
        tools=[add],
        capabilities={"tools": "supported"},
    )


def test_compatible_has_no_token_counting() -> None:
    harness = ChatHarness()
    agent = harness.agent([])
    with pytest.raises(UnsupportedCapabilityError):
        agent.count_tokens("hi")


def test_gemini_rejects_unmapped_effort() -> None:
    with pytest.raises((ConfigurationError, UnsupportedCapabilityError)):
        GeminiHarness().agent([], effort="max")


# ---- request shapes --------------------------------------------------------------------


def test_openai_request_is_stateless_with_encrypted_reasoning() -> None:
    harness = OpenAIHarness()
    agent = harness.agent([harness.text("ok")], effort="high")
    agent.ask("hi")
    call = harness.script.calls[0]
    assert call["store"] is False
    assert "reasoning.encrypted_content" in call["include"]
    assert call["reasoning"]["effort"] == "high"
    assert "previous_response_id" not in call


def test_openai_store_option_is_passed() -> None:
    harness = OpenAIHarness()
    agent = harness.agent(
        [harness.text("ok")], options=OpenAIOptions(store=True, include_encrypted_reasoning=False)
    )
    agent.ask("hi")
    assert harness.script.calls[0]["store"] is True


def test_openai_tools_are_strict() -> None:
    harness = OpenAIHarness()
    agent = harness.agent([harness.text("ok")], tools=[add])
    agent.ask("hi")
    (definition,) = harness.script.calls[0]["tools"]
    assert definition["type"] == "function"
    assert definition["strict"] is True
    assert definition["parameters"]["additionalProperties"] is False


def test_openai_count_tokens() -> None:
    harness = OpenAIHarness()
    agent = harness.agent([])
    assert agent.count_tokens("hi") == 321


def test_gemini_disables_automatic_function_calling() -> None:
    harness = GeminiHarness()
    agent = harness.agent([harness.text("ok")], tools=[add])
    agent.ask("hi")
    config = harness.script.calls[0]["config"]
    afc = config["automatic_function_calling"]
    assert (afc["disable"] if isinstance(afc, dict) else afc.disable) is True


def test_gemini_count_tokens() -> None:
    harness = GeminiHarness()
    agent = harness.agent([])
    assert agent.count_tokens("hi") == 321


def test_chat_completions_json_object_fallback_puts_schema_in_prompt() -> None:
    harness = ChatHarness()
    agent = harness.agent(
        [harness.text('{"x": 1}')],
        output_format=Inner,
        options=CompatibleOptions(structured_output="json_object"),
    )
    assert agent.ask("hi") == Inner(x=1)
    call = harness.script.calls[0]
    assert call["response_format"] == {"type": "json_object"}
    assert '"x"' in call["messages"][0]["content"]


def test_chat_completions_json_schema_response_format() -> None:
    harness = ChatHarness()
    agent = harness.agent([harness.text('{"x": 1}')], output_format=Inner)
    agent.ask("hi")
    response_format = harness.script.calls[0]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "Inner"


def test_chat_completions_never_replays_reasoning() -> None:
    harness = ChatHarness()
    agent = harness.agent(
        [harness.tool_calls([("c1", "add", {"a": 1, "b": 2})]), harness.text("3")], tools=[add]
    )
    agent.ask("hi")
    assistant = [m for m in harness.script.calls[1]["messages"] if m["role"] == "assistant"]
    assert all("reasoning" not in m and "reasoning_content" not in m for m in assistant)


# ---- schema conversion -----------------------------------------------------------------


def test_strict_schema_makes_every_property_required_and_closed() -> None:
    schema = strict_schema(Outer.model_json_schema(), name="Outer")
    assert schema["required"] == ["a", "b", "c"]
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["Inner"]["additionalProperties"] is False
    assert {"type": "null"} in schema["properties"]["b"]["anyOf"]


def test_drop_null_optionals_restores_defaults() -> None:
    schema = Outer.model_json_schema()
    assert drop_null_optionals({"a": {"x": 1}, "b": None, "c": None}, schema) == {"a": {"x": 1}}


def test_nullable_property_with_default_is_rejected() -> None:
    class Ambiguous(BaseModel):
        d: int | None = 5

    with pytest.raises(ToolSchemaError):
        strict_schema(Ambiguous.model_json_schema(), name="Ambiguous")


def test_inline_refs_resolves_definitions() -> None:
    schema = inline_refs(Outer.model_json_schema(), name="Outer")
    text = repr(schema)
    assert "$ref" not in text and "$defs" not in text
    assert schema["properties"]["a"]["properties"]["x"] == {"type": "integer"}


def test_inline_refs_rejects_recursion() -> None:
    class Node(BaseModel):
        children: list[Node] = []

    with pytest.raises(ToolSchemaError):
        inline_refs(Node.model_json_schema(), name="Node")


def test_declared_capabilities_drive_the_encoding() -> None:
    """An agent's capabilities= overrides reach the adapter (not only the checks)."""
    harness = ChatHarness()
    kwargs = harness.agent_kwargs()
    kwargs["capabilities"] = {**kwargs["capabilities"], "strict_tool_schemas": "supported"}
    harness.script.responses = [harness.text('{"a": {"x": 1}, "b": null, "c": null}')]
    agent = Agent(provider="ollama", client=harness.client(False), output_format=Outer, **kwargs)
    assert agent.ask("hi") == Outer(a=Inner(x=1))
    response_format = harness.script.calls[0]["response_format"]["json_schema"]
    assert response_format["strict"] is True
    assert response_format["schema"]["required"] == ["a", "b", "c"]
