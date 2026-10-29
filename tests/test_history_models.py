"""Reasoning is bound to the provider *and* model that produced it."""

from __future__ import annotations

from typing import Any
import warnings

import pytest

from orkexa import Agent, HistoryConversionWarning, Message
from orkexa.core.conversion import prepare_history
from orkexa.core.types import ReasoningPart, TextPart, ToolCallPart, ToolResultPart
from orkexa.providers.fake import FakeModel, response, text

from .contract.harness import HARNESSES, GeminiHarness, Harness
from .helpers import add


def _reasoning(messages: list[Message], model: str | None = None) -> list[ReasoningPart]:
    return [
        part
        for message in messages
        if message.role == "assistant" and (model is None or message.model == model)
        for part in message.parts()
        if isinstance(part, ReasoningPart)
    ]


@pytest.mark.parametrize("policy", ["strict", "convert"])
def test_model_switch_drops_reasoning_of_the_old_model(policy: Any) -> None:
    fake = FakeModel(
        [
            response(
                [
                    ReasoningPart(text="model-bound"),
                    ToolCallPart(id="c1", name="add", arguments={"a": 1, "b": 2}),
                ],
                stop_reason="tool_use",
            ),
            text("three", reasoning="also model-bound"),
            text("second answer"),
        ]
    )
    agent = Agent(model="fake:old", client=fake, tools=[add], history_policy=policy)
    assert agent.ask("first") == "three"
    assert [m.model for m in agent.messages if m.role == "assistant"] == ["old", "old"]
    agent.model = "fake:new"
    with pytest.warns(HistoryConversionWarning, match=r"2 fake:old reasoning part\(s\).*fake:new"):
        assert agent.ask("second") == "second answer"
    sent = fake.requests[-1].messages
    assert _reasoning(sent) == []
    # Visible content and the tool call/result pair stay (the fake checks pairing).
    assert [m.text for m in sent if m.role == "assistant"] == ["", "three"]
    assert any(isinstance(p, ToolCallPart) for m in sent for p in m.parts())
    assert any(isinstance(p, ToolResultPart) for m in sent for p in m.parts())
    # The stored history is unchanged: switching back replays the reasoning again.
    assert len(_reasoning(agent.messages, "old")) == 2


def test_reasoning_is_replayed_within_a_turn_and_to_the_same_model() -> None:
    fake = FakeModel(
        [
            response(
                [ReasoningPart(text="plan"), ToolCallPart(id="c1", name="add", arguments={"a": 1, "b": 2})],
                stop_reason="tool_use",
            ),
            text("three"),
            text("again"),
        ]
    )
    agent = Agent(model="fake:m", client=fake, tools=[add])
    with warnings.catch_warnings():
        warnings.simplefilter("error", HistoryConversionWarning)
        agent.ask("first")
        agent.ask("second")
    assert len(_reasoning(fake.requests[1].messages)) == 1
    assert len(_reasoning(fake.requests[2].messages)) == 1


def test_messages_without_a_model_are_treated_as_the_current_model() -> None:
    legacy = [
        Message(role="user", content="hi"),
        Message(role="assistant", content=[ReasoningPart(text="r"), TextPart(text="hello")], provider="fake"),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error", HistoryConversionWarning)
        prepared = prepare_history(legacy, "fake", "strict", "any-model")
    assert prepared == legacy


def test_reasoning_only_message_of_another_model_is_removed() -> None:
    history = [
        Message(role="user", content="hi"),
        Message(role="assistant", content=[ReasoningPart(text="r")], provider="fake", model="a"),
        Message(role="assistant", content=[TextPart(text="hello")], provider="fake", model="a"),
    ]
    with pytest.warns(HistoryConversionWarning):
        prepared = prepare_history(history, "fake", "strict", "b")
    assert [m.text for m in prepared] == ["hi", "hello"]


@pytest.mark.parametrize("cls", HARNESSES, ids=[h.name for h in HARNESSES])
def test_dated_model_versions_keep_reasoning_in_the_tool_loop(cls: type[Harness]) -> None:
    """APIs report a dated version (``gpt-5.5-2026-08-01``); it is the same model."""
    harness = cls()
    requested = harness.agent_kwargs()["model"]
    harness.model = f"{requested}-2026-08-01"  # type: ignore[attr-defined]
    responses = [harness.tool_calls([("c1", "add", {"a": 1, "b": 2})]), harness.text("3")]
    agent = harness.agent(responses, model=requested, tools=[add])
    with warnings.catch_warnings():
        warnings.simplefilter("error", HistoryConversionWarning)
        assert agent.ask("1+2?") == "3"
    assert harness.native_replayed(harness.script.calls[1])
    assert agent.last_response.model == harness.model  # type: ignore[union-attr]
    assert {m.model for m in agent.messages if m.role == "assistant"} == {requested}


def _signatures(call: dict[str, Any]) -> list[Any]:
    return [part.thought_signature for content in call["contents"] for part in content.parts or []]


def test_gemini_thought_signatures_are_only_replayed_to_their_model() -> None:
    harness = GeminiHarness()
    responses = [harness.tool_calls([("c1", "add", {"a": 1, "b": 2})]), harness.text("3"), harness.text("ok")]
    agent = harness.agent(responses, tools=[add])
    agent.ask("1+2?")
    assert any(_signatures(harness.script.calls[1]))
    agent.model = "gemini:gemini-3-flash"
    with pytest.warns(HistoryConversionWarning):
        agent.ask("again")
    assert not any(_signatures(harness.script.calls[2]))


def test_unchanged_histories_are_not_restructured() -> None:
    history = [
        Message(role="user", content="hi"),
        Message(role="assistant", content="A", provider="fake", model="model-one"),
        Message(role="assistant", content="B", provider="fake", model="model-two"),
        Message(role="assistant", content="C", provider="fake", model="model-two"),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error", HistoryConversionWarning)
        assert prepare_history(history, "fake", "strict", "model-one") == history


def test_merging_after_dropped_parts_keeps_models_apart() -> None:
    history = [
        Message(role="user", content="hi"),
        Message(
            role="assistant",
            content=[ReasoningPart(text="r"), TextPart(text="A")],
            provider="other",
            model="x",
        ),
        Message(role="assistant", content=[TextPart(text="B")], provider="fake", model="model-one"),
        Message(role="assistant", content=[TextPart(text="C")], provider="fake", model="model-two"),
    ]
    with pytest.warns(HistoryConversionWarning):
        prepared = prepare_history(history, "fake", "convert", "model-one")
    assert [(m.provider, m.model, m.text) for m in prepared] == [
        (None, None, "hi"),
        ("other", "x", "A"),
        ("fake", "model-one", "B"),
        ("fake", "model-two", "C"),
    ]


def test_dropping_a_foreign_message_keeps_neighbouring_boundaries() -> None:
    history = [
        Message(role="user", content="go"),
        Message(
            role="assistant",
            content=[ToolCallPart(id="t1", name="add", arguments={"a": 1, "b": 2})],
            provider="fake",
            model="m",
        ),
        Message(role="user", content=[ToolResultPart(call_id="t1", content="3")]),
        Message(role="assistant", content=[TextPart(text="three")], provider="fake", model="m"),
        Message(role="assistant", content=[ReasoningPart(text="foreign")], provider="other", model="x"),
        Message(role="assistant", content=[TextPart(text="later")], provider="fake", model="m"),
    ]
    with pytest.warns(HistoryConversionWarning, match="empty message"):
        prepared = prepare_history(history, "fake", "convert", "m")
    assert prepared == history[:4] + history[5:]
