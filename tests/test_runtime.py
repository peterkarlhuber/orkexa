"""The shared runtime, driven by the SDK-free FakeProvider (no network, no vendor SDK)."""

from __future__ import annotations

import asyncio
from decimal import Decimal
import json
from pathlib import Path
import threading
from typing import Any

from pydantic import BaseModel
import pytest

from orkexa import (
    Agent,
    AsyncAgent,
    ConcurrentTurnError,
    ConfigurationError,
    EphemeralMemory,
    IncompleteResponseError,
    MaxToolRoundsError,
    Message,
    ModelPricing,
    RefusalError,
    StructuredOutputError,
    ToolCall,
    ToolDefinitionError,
    ToolExecutionError,
    ToolInputError,
    UnexpectedResponseError,
    UnknownCapabilityError,
    UnsupportedCapabilityError,
    UsageTracker,
)
from orkexa.core.types import (
    ReasoningDelta,
    ResponseCompleted,
    TextDelta,
    TextPart,
    TokenUsage,
    ToolCallDelta,
    ToolCallPart,
    ToolCallStarted,
    ToolResultPart,
    TurnCompleted,
    TurnFailed,
    UsageReported,
)
from orkexa.providers.fake import FakeModel, response, text, tool_calls
from orkexa.providers.registry import resolve_model, split_model

from .helpers import add, async_add, async_slow, broken, lookup, multiply, slow


def agent_with(*script: Any, **kwargs: Any) -> tuple[Agent[Any], FakeModel]:
    model = FakeModel(list(script), capabilities=kwargs.pop("caps", None))
    return Agent(model="fake:m", client=model, **kwargs), model


# ---- tool loop -----------------------------------------------------------------------


def test_multi_round_tool_loop_commits_history() -> None:
    agent, model = agent_with(
        tool_calls(("add", {"a": 1, "b": 2})),
        tool_calls(("multiply", {"a": 3, "b": 4}), prefix="r2"),
        text("3 and 12"),
        tools=[add, multiply],
    )
    assert agent.ask("compute") == "3 and 12"
    roles = [m.role for m in agent.messages]
    assert roles == ["user", "assistant", "user", "assistant", "user", "assistant"]
    results = [p for m in agent.messages for p in m.parts() if isinstance(p, ToolResultPart)]
    assert [(r.call_id, r.name, r.content) for r in results] == [
        ("call_0", "add", "3"),
        ("call_r20", "multiply", "12"),
    ]
    assert len(model.requests) == 3
    assert [t.name for t in model.requests[0].tools] == ["add", "multiply"]
    assert agent.messages[1].provider == "fake" and agent.messages[1].model == "m"
    assert agent.usage.requests == 3 and agent.last_turn_usage.requests == 3


def test_several_calls_in_one_response_keep_order_and_ids_in_parallel() -> None:
    agent, model = agent_with(
        tool_calls(
            ("slow", {"seconds": 0.05}, "a"), ("add", {"a": 1, "b": 1}, "b"), ("slow", {"seconds": 0.0}, "c")
        ),
        text("ok"),
        tools=[slow, add],
        max_parallel_tools=3,
    )
    assert agent.ask("go") == "ok"
    results = model.requests[1].messages[-1].parts()
    assert [(r.call_id, r.content) for r in results] == [("a", "done"), ("b", "2"), ("c", "done")]  # type: ignore[union-attr]


def test_tool_errors_follow_raise_tool_errors() -> None:
    agent, _ = agent_with(tool_calls(("broken", {"x": 1})), tools=[broken])
    with pytest.raises(ToolExecutionError, match="boom"):
        agent.ask("go")
    assert agent.messages == []  # rolled back

    agent, model = agent_with(
        tool_calls(("broken", {"x": 1})), text("sorry"), tools=[broken], raise_tool_errors=False
    )
    assert agent.ask("go") == "sorry"
    result = model.requests[1].messages[-1].parts()[0]
    assert isinstance(result, ToolResultPart) and result.is_error and "boom" in str(result.content)


def test_tool_result_error_always_goes_back_to_the_model() -> None:
    agent, model = agent_with(tool_calls(("lookup", {"key": "x"})), text("unknown"), tools=[lookup])
    assert agent.ask("go") == "unknown"
    result = model.requests[1].messages[-1].parts()[0]
    assert isinstance(result, ToolResultPart) and result.is_error and "unknown key" in str(result.content)


def test_unknown_tool_and_invalid_input() -> None:
    agent, _ = agent_with(tool_calls(("nope", {})), tools=[add])
    with pytest.raises(ToolExecutionError, match="Unknown tool called: nope"):
        agent.ask("go")
    # The provider enforces schemas: invalid input is a bug and raises.
    agent, _ = agent_with(tool_calls(("add", {"a": "x", "b": 1})), tools=[add])
    with pytest.raises(ToolInputError):
        agent.ask("go")
    # It does not: the model gets the error and can retry.
    agent, model = agent_with(
        tool_calls(("add", {"a": "x", "b": 1})),
        text("retry"),
        tools=[add],
        caps={"strict_tool_schemas": "unsupported"},
    )
    assert agent.ask("go") == "retry"
    assert model.requests[1].messages[-1].parts()[0].is_error  # type: ignore[union-attr]


def test_malformed_arguments_are_reported_as_tool_input_error() -> None:
    bad = response([ToolCallPart(id="c", name="add", parse_error="not JSON")], stop_reason="tool_use")
    agent, model = agent_with(bad, text("ok"), tools=[add], raise_tool_errors=False)
    assert agent.ask("go") == "ok"
    assert "not JSON" in str(model.requests[1].messages[-1].parts()[0].content)  # type: ignore[union-attr]


def test_before_and_after_tool_hooks() -> None:
    seen: list[tuple[str, str, bool]] = []

    def deny_multiply(call: ToolCall) -> Any:
        return "not allowed" if call.name == "multiply" else None

    agent, _ = agent_with(
        tool_calls(("add", {"a": 1, "b": 2}), ("multiply", {"a": 2, "b": 2})),
        text("done"),
        tools=[add, multiply],
        before_tool=deny_multiply,
        after_tool=lambda call, content, is_error: seen.append((call.name, content, is_error)),
    )
    agent.ask("go")
    assert seen == [("add", "3", False), ("multiply", "Tool call denied: not allowed", True)]
    agent2, _ = agent_with(
        tool_calls(("add", {"a": 1, "b": 2})), text("x"), tools=[add], before_tool=lambda c: False
    )
    agent2.ask("go")
    assert agent2.messages[2].parts()[0].content == "Tool call denied."  # type: ignore[union-attr]


def test_max_tool_rounds_and_rollback_keep_usage() -> None:
    tracker = UsageTracker()
    agent, _ = agent_with(
        tool_calls(("add", {"a": 1, "b": 1})),
        tool_calls(("add", {"a": 1, "b": 1})),
        tools=[add],
        max_tool_rounds=1,
        usage_tracker=tracker,
    )
    with pytest.raises(MaxToolRoundsError):
        agent.ask("loop")
    assert agent.messages == []
    assert tracker.usage.requests == 2  # costs never disappear from the ledger


def test_incomplete_and_unexpected_stop_reasons() -> None:
    agent, _ = agent_with(text("cut", stop_reason="max_tokens"))
    with pytest.raises(IncompleteResponseError, match="max_tokens"):
        agent.ask("x")
    agent, _ = agent_with(text("cut", stop_reason="context_window_exceeded"))
    with pytest.raises(IncompleteResponseError):
        agent.ask("x")
    agent, _ = agent_with(text("?", stop_reason="other"))
    with pytest.raises(UnexpectedResponseError, match="Unhandled stop_reason"):
        agent.ask("x")
    agent, _ = agent_with(response([TextPart(text="no calls")], stop_reason="tool_use"))
    with pytest.raises(UnexpectedResponseError, match="without tool_use"):
        agent.ask("x")
    assert agent.messages == []


def test_pause_turn_continues() -> None:
    agent, model = agent_with(text("part", stop_reason="pause_turn"), text("rest"))
    assert agent.ask("x") == "rest"
    assert len(model.requests) == 2 and model.requests[1].messages[-1].text == "part"


def test_refusal_modes() -> None:
    agent, _ = agent_with(text("I can't", stop_reason="refusal"))
    with pytest.raises(RefusalError):
        agent.ask("x")
    assert agent.messages == []
    agent, _ = agent_with(text("I can't", stop_reason="refusal"), on_refusal="return")
    assert agent.ask("x") == "I can't"
    assert len(agent.messages) == 2


# ---- structured output ------------------------------------------------------------------


class Answer(BaseModel):
    value: int


def test_structured_output_is_validated_locally() -> None:
    agent, model = agent_with(text('{"value": 42}'), output_format=Answer)
    result = agent.ask("x")
    assert isinstance(result, Answer) and result.value == 42
    assert model.requests[0].output_format is Answer
    agent, _ = agent_with(text("not json"), output_format=Answer)
    with pytest.raises(StructuredOutputError, match="Answer"):
        agent.ask("x")
    agent, _ = agent_with(text("no", stop_reason="refusal"), output_format=Answer, on_refusal="return")
    with pytest.raises(RefusalError):
        agent.ask("x")


def test_output_format_must_be_a_model() -> None:
    with pytest.raises(TypeError):
        Agent(model="fake:m", output_format=dict)  # type: ignore[arg-type]


# ---- streaming and events ------------------------------------------------------------------


def test_streaming_events_and_identical_history() -> None:
    events: list[Any] = []
    texts: list[str] = []
    thoughts: list[str] = []
    script = [tool_calls(("add", {"a": 2, "b": 3})), text("five", reasoning="thinking...")]
    agent, _ = agent_with(
        *script, tools=[add], on_event=events.append, on_text=texts.append, on_thinking=thoughts.append
    )
    assert agent.ask("x") == "five"
    kinds = [type(e) for e in events]
    assert kinds == [
        ToolCallStarted,
        ToolCallDelta,
        ResponseCompleted,
        UsageReported,
        ReasoningDelta,
        TextDelta,
        TextDelta,
        ResponseCompleted,
        UsageReported,
        TurnCompleted,
    ]
    assert "".join(texts) == "five" and thoughts == ["thinking..."]
    plain, _ = agent_with(*script, tools=[add], stream=False)
    plain.ask("x")
    assert [m.model_dump() for m in plain.messages] == [m.model_dump() for m in agent.messages]


def test_turn_failed_event() -> None:
    events: list[Any] = []
    agent, _ = agent_with(text("cut", stop_reason="max_tokens"), on_event=events.append)
    with pytest.raises(IncompleteResponseError):
        agent.ask("x")
    assert isinstance(events[-1], TurnFailed) and isinstance(events[-1].error, IncompleteResponseError)


def test_request_and_response_hooks_get_normalized_objects() -> None:
    requests: list[Any] = []
    responses: list[Any] = []
    agent, _ = agent_with(text("hi"), on_request=requests.append, on_response=responses.append)
    agent.ask("x")
    assert requests[0].model == "m" and responses[0].text == "hi"
    assert agent.last_response is responses[0]


# ---- per-turn options, system messages, capabilities -------------------------------------------


def test_per_turn_overrides_and_system_message() -> None:
    agent, model = agent_with(text("a"), text("b"), effort="low")
    agent.ask("x", effort="high", max_tokens=100, system="Answer briefly.")
    agent.ask("y")
    first, second = model.requests
    assert (first.effort, first.max_tokens) == ("high", 100)
    assert (second.effort, second.max_tokens) == ("low", 16_000)
    assert [m.role for m in first.messages] == ["user", "system"]
    assert agent._effort == "low"
    with pytest.raises(ValueError):
        agent.ask("x", system=" ")
    with pytest.raises(ValueError):
        agent.ask("x", max_tokens=0)


def test_unknown_capability_is_not_assumed() -> None:
    with pytest.raises(UnknownCapabilityError, match="capabilities=\\{'tools': 'supported'\\}"):
        agent_with(tools=[add], caps={"tools": "unknown"})
    agent, _ = agent_with(
        text("ok"), tools=[add], caps={"tools": "unknown"}, capabilities={"tools": "supported"}
    )
    assert agent.ask("x") == "ok"
    with pytest.raises(UnsupportedCapabilityError):
        agent_with(caps={"structured_output": "unsupported"}, output_format=Answer)
    with pytest.raises(UnsupportedCapabilityError):
        agent_with(caps={"reasoning": "unsupported"}, effort="high")
    agent, _ = agent_with(caps={"system_messages": "unsupported"})
    with pytest.raises(UnsupportedCapabilityError):
        agent.ask("x", system="extra")
    agent, _ = agent_with(caps={"image_input": "unsupported"})
    with pytest.raises(UnsupportedCapabilityError):
        agent.ask([{"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}}, "what?"])
    with pytest.raises(ValueError, match="Unknown capability"):
        agent_with(capabilities={"teleport": "supported"})


def test_tool_registration() -> None:
    agent, _ = agent_with(tools=[add])
    with pytest.raises(ValueError, match="Duplicate tool name: add"):
        agent.add_tool(add)
    with pytest.raises(ToolDefinitionError, match="async tools require AsyncAgent"):
        agent.add_tool(async_add)
    agent.add_server_tool({"type": "web_search", "name": "web"})
    with pytest.raises(TypeError):
        agent.add_server_tool({"name": "x"})
    assert agent.remove_tool("web") == {"type": "web_search", "name": "web"}
    assert agent.remove_tool("add") is add
    with pytest.raises(KeyError):
        agent.remove_tool("add")
    assert repr(agent) == "Agent(name='m', model='m', tools=[], messages=0)"


def test_memory_is_a_function_tool_without_native_memory() -> None:
    memory = EphemeralMemory()
    agent, model = agent_with(
        tool_calls(("memory", {"command": "create", "path": "/memories/a.md", "file_text": "x"})),
        text("saved"),
        memory=memory,
        caps={"native_memory_tool": "unsupported"},
    )
    agent.ask("remember")
    assert memory.files() == {"/memories/a.md": "x"}
    assert [t.name for t in model.requests[0].tools] == ["memory"] and model.requests[0].native_tools == []
    agent, model = agent_with(text("ok"), memory=EphemeralMemory())
    agent.ask("x")
    assert model.requests[0].native_tools == [{"type": "memory_20250818", "name": "memory"}]


# ---- model resolution --------------------------------------------------------------------------


def test_model_resolution() -> None:
    assert resolve_model("openai:gpt-5.5") == ("openai", "gpt-5.5")
    assert resolve_model("gpt-5.5", "openai") == ("openai", "gpt-5.5")
    assert resolve_model("claude-opus-5-5") == ("anthropic", "claude-opus-5-5")
    assert resolve_model("llama3.1:8b", "ollama") == ("ollama", "llama3.1:8b")
    assert split_model("ft:gpt-4o:org") == (None, "ft:gpt-4o:org")
    with pytest.raises(ConfigurationError, match="names provider"):
        resolve_model("openai:gpt-5.5", "gemini")
    with pytest.raises(ConfigurationError, match="Unknown provider"):
        resolve_model("x", "nope")
    agent, _ = agent_with(text("ok"))
    agent.model = "openai:gpt-5.5"
    with pytest.raises(ConfigurationError, match="create a new agent"):
        agent.ask("x")


def test_options_must_match_the_provider() -> None:
    from orkexa import OpenAIOptions

    with pytest.raises(ConfigurationError, match="takes no options"):
        Agent(model="fake:m", options=OpenAIOptions())


# ---- history --------------------------------------------------------------------------------------


def test_save_and_load_history(tmp_path: Path) -> None:
    agent, _ = agent_with(tool_calls(("add", {"a": 1, "b": 2})), text("3"), tools=[add])
    agent.ask("x")
    path = tmp_path / "h.json"
    agent.save_history(path)
    data = json.loads(path.read_text())
    assert data["format"] == "orkexa.history" and data["version"] == 1
    other, _ = agent_with()
    other.load_history(path)
    assert other.messages == agent.messages

    legacy = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
    ]
    path.write_text(json.dumps(legacy))
    other.load_history(path)
    assert other.messages[1].provider == "anthropic" and other.messages[1].text == "hello"
    path.write_text(json.dumps({"format": "orkexa.history", "version": 99, "messages": []}))
    with pytest.raises(ValueError, match="unsupported history version"):
        other.load_history(path)
    path.write_text(json.dumps({"x": 1}))
    with pytest.raises(ValueError, match="not a conversation history"):
        other.load_history(path)


def test_initial_messages_and_reset() -> None:
    agent, model = agent_with(
        text("b"), messages=[Message(role="user", content="a"), {"role": "assistant", "content": "A"}]
    )
    agent.ask("next")
    assert [m.text for m in model.requests[0].messages] == ["a", "A", "next"]
    agent.reset()
    assert agent.messages == [] and agent.last_response is None


# ---- usage, pricing, budgets ---------------------------------------------------------------------


def test_usage_is_unpriced_unless_configured() -> None:
    agent, _ = agent_with(text("a"))
    agent.ask("x")
    record = agent.last_usage
    assert record is not None and record.estimated_cost_usd is None and record.provider == "fake"
    assert record.turn_id is not None and "No known pricing" in (record.pricing_note or "")
    priced, _ = agent_with(
        response([TextPart(text="a")], usage=TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)),
        pricing=ModelPricing.of("1", "2", "1", "1", "0.1"),
    )
    priced.ask("x")
    assert priced.usage.estimated_cost_usd == Decimal("3")
    assert priced.last_usage is not None and priced.last_usage.pricing_source == "custom"


def test_tracker_before_request_can_stop_a_turn() -> None:
    class Stop(UsageTracker):
        def before_request(self, **kwargs: Any) -> None:
            if not kwargs["priced"]:
                raise RuntimeError(f"unpriced {kwargs['provider']}:{kwargs['model']}")

    agent, model = agent_with(text("a"), usage_tracker=Stop())
    with pytest.raises(RuntimeError, match="unpriced fake:m"):
        agent.ask("x")
    assert model.requests == []


def test_count_tokens() -> None:
    agent, _ = agent_with()
    assert agent.count_tokens("hello") == 42
    with pytest.raises(ValueError, match="Nothing to count"):
        agent.count_tokens()


# ---- concurrency and timeouts -----------------------------------------------------------------------


def test_concurrent_ask_is_rejected() -> None:
    started = threading.Event()
    release = threading.Event()

    def wait(request: Any) -> Any:
        started.set()
        release.wait(5)
        return text("first")

    agent, _ = agent_with(wait, text("second"))
    results: list[str] = []
    worker = threading.Thread(target=lambda: results.append(agent.ask("one")))
    worker.start()
    started.wait(5)
    with pytest.raises(ConcurrentTurnError):
        agent.ask("two")
    release.set()
    worker.join(5)
    assert results == ["first"] and agent.ask("three") == "second"


def test_sync_tool_timeout() -> None:
    agent, model = agent_with(
        tool_calls(("slow", {"seconds": 2})),
        text("gave up"),
        tools=[slow],
        tool_timeout=0.05,
        raise_tool_errors=False,
    )
    assert agent.ask("x") == "gave up"
    assert "timed out" in str(model.requests[1].messages[-1].parts()[0].content)  # type: ignore[union-attr]


# ---- async -------------------------------------------------------------------------------------------


def test_async_agent_tools_hooks_and_parallelism() -> None:
    async def run() -> None:
        texts: list[str] = []

        async def on_text(chunk: str) -> None:
            texts.append(chunk)

        async def before(call: ToolCall) -> None:
            await asyncio.sleep(0)

        model = FakeModel(
            [tool_calls(("async_add", {"a": 1, "b": 2}), ("add", {"a": 3, "b": 4})), text("3 and 7")]
        )
        agent: AsyncAgent[str] = AsyncAgent(
            model="fake:m",
            client=model,
            tools=[async_add, add],
            on_text=on_text,
            before_tool=before,
            max_parallel_tools=2,
        )
        assert await agent.ask("x") == "3 and 7"
        assert "".join(texts) == "3 and 7"
        results = model.requests[1].messages[-1].parts()
        assert [r.content for r in results] == ["3", "7"]  # type: ignore[union-attr]
        assert await agent.count_tokens() == 42

    asyncio.run(run())


def test_async_cancellation_leaves_history_unchanged() -> None:
    async def run() -> None:
        model = FakeModel([tool_calls(("async_slow", {"seconds": 5}))])
        agent: AsyncAgent[str] = AsyncAgent(model="fake:m", client=model, tools=[async_slow])
        task = asyncio.create_task(agent.ask("x"))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert agent.messages == []
        assert agent.usage.requests == 1
        model.add(text("fresh"))
        assert await agent.ask("again") == "fresh"

    asyncio.run(run())


def test_async_tool_timeout_and_concurrency_guard() -> None:
    async def run() -> None:
        model = FakeModel([tool_calls(("async_slow", {"seconds": 5})), text("gave up")])
        agent: AsyncAgent[str] = AsyncAgent(
            model="fake:m", client=model, tools=[async_slow], tool_timeout=0.05, raise_tool_errors=False
        )
        first = asyncio.create_task(agent.ask("x"))
        await asyncio.sleep(0)
        with pytest.raises(ConcurrentTurnError):
            await agent.ask("y")
        assert await first == "gave up"

    asyncio.run(run())
