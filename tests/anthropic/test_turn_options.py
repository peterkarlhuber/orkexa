"""Refusals, thinking display, mid-conversation system messages, per-turn overrides
and eager input streaming (0.7.0)."""

from __future__ import annotations

import asyncio
from decimal import Decimal
import json
from typing import Any

import anthropic
import httpx2
import pytest

from orkexa import (
    MODEL_PRICING,
    AsyncClaudeAgent,
    ClaudeAgent,
    RefusalError,
    Tool,
    ToolExecutionError,
    ToolInputError,
)

from .conftest import AsyncFakeClient, FakeClient, make_beta_message, make_message


def _text(text: str, **kwargs: Any) -> Any:
    return make_message([{"type": "text", "text": text}], **kwargs)


def _agent(client: Any, **kwargs: Any) -> ClaudeAgent[Any]:
    return ClaudeAgent(client=client, **kwargs)


@Tool
def shout(text: str) -> str:
    """Repeat a text in capital letters.

    Args:
        text: The text to repeat.

    Returns:
        The text in capital letters.
    """
    return text.upper()


def _refusal() -> Any:
    return make_beta_message(
        [{"type": "text", "text": ""}],
        stop_reason="refusal",
        stop_details={"type": "refusal", "category": "cyber", "explanation": "Declined."},
    )


# ---- refusals ----------------------------------------------------------------------


def test_text_refusal_raises_and_keeps_the_history() -> None:
    agent = _agent(FakeClient([_text("Hello"), _refusal()]))
    agent.ask("Hi")
    before = agent.history()
    with pytest.raises(RefusalError) as info:
        agent.ask("Something declined")
    assert info.value.category == "cyber"
    assert agent.history() == before
    assert agent.last_response is not None and agent.last_response.stop_reason == "refusal"
    assert agent.last_turn_usage.requests == 1


def test_on_refusal_return_keeps_the_previous_behavior() -> None:
    agent = _agent(FakeClient([_refusal()]), on_refusal="return")
    assert agent.ask("Hi") == ""
    assert agent.messages[-1]["role"] == "assistant"
    with pytest.raises(ValueError, match="on_refusal"):
        _agent(FakeClient([]), on_refusal="ignore")


def test_async_text_refusal_raises() -> None:
    agent = AsyncClaudeAgent(client=AsyncFakeClient([_refusal()]))  # type: ignore[arg-type]
    with pytest.raises(RefusalError):
        asyncio.run(agent.ask("Hi"))
    assert agent.messages == []


# ---- pricing -------------------------------------------------------------------------


def test_sonnet_55_cache_reads_cost_a_twentieth_of_input() -> None:
    # The official price list (checked 2026-10-31) prices Claude Sonnet 5.5 and
    # Claude Opus 5.5 cache hits at 0.05x the base input price.
    prices = MODEL_PRICING["claude-sonnet-5-5"]
    assert prices.cache_read == Decimal("0.10") == prices.input / 20


# ---- thinking display and the on_thinking hook ---------------------------------------


def test_thinking_display_is_sent_and_updates_add_the_beta() -> None:
    client = FakeClient([_text("a"), _text("b"), _text("c")])
    _agent(client, thinking_display="summarized").ask("Hi")
    _agent(client, thinking="adaptive", thinking_display="omitted").ask("Hi")
    _agent(client, model="claude-opus-5-5", thinking_display="updates").ask("Hi")
    first, second, third = client.messages.create_calls
    assert first["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert "betas" not in first
    assert second["thinking"] == {"type": "adaptive", "display": "omitted"}
    assert third["thinking"] == {"type": "adaptive", "display": "updates"}
    assert third["betas"] == ["thinking-display-updates-2026-08-18"]


def test_thinking_display_validation() -> None:
    with pytest.raises(ValueError, match="thinking_display"):
        _agent(FakeClient([]), thinking_display="full")
    with pytest.raises(ValueError, match="thinking_display"):
        _agent(
            FakeClient([]),
            model="claude-sonnet-5-5",
            effort="low",
            thinking="between_tools",
            thinking_display="summarized",
        )


def test_on_thinking_streams_thinking_and_text_separately() -> None:
    message = make_message(
        [
            {"type": "thinking", "thinking": "Let me add.", "signature": "sig"},
            {"type": "text", "text": "42"},
        ]
    )
    client = FakeClient([message])
    thoughts: list[str] = []
    texts: list[str] = []
    agent = _agent(client, thinking_display="summarized", on_thinking=thoughts.append, on_text=texts.append)
    assert agent.ask("6 * 7?") == "42"
    assert thoughts == ["Let me add."] and texts == ["42"]
    assert client.messages.stream_calls and not client.messages.create_calls


def test_on_thinking_alone_turns_on_streaming() -> None:
    client = FakeClient([_text("ok")])
    _agent(client, on_thinking=lambda _: None).ask("Hi")
    assert client.messages.stream_calls


def _sse(events: list[tuple[str, dict[str, Any]]]) -> bytes:
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events).encode()


def test_stream_events_of_the_real_sdk_reach_the_hooks() -> None:
    usage = {"input_tokens": 10, "output_tokens": 5}
    body = _sse(
        [
            (
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "model": "claude-haiku-5-5",
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": usage,
                    },
                },
            ),
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "thinking", "thinking": "", "signature": ""},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "thinking_delta", "thinking": "Adding."},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "signature_delta", "signature": "sig"},
                },
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            (
                "content_block_start",
                {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
            ),
            (
                "content_block_delta",
                {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "4"}},
            ),
            (
                "content_block_delta",
                {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "2"}},
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": 1}),
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": 5},
                },
            ),
            ("message_stop", {"type": "message_stop"}),
        ]
    )
    requests: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(json.loads(request.content))
        return httpx2.Response(200, content=body, headers={"content-type": "text/event-stream"})

    client = anthropic.Anthropic(
        api_key="test", http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler))
    )
    thoughts: list[str] = []
    texts: list[str] = []
    agent = ClaudeAgent(
        client=client,
        thinking_display="summarized",
        on_thinking=thoughts.append,
        on_text=texts.append,
        tools=[shout],
        eager_input_streaming=True,
    )
    assert agent.ask("6 * 7?") == "42"
    assert thoughts == ["Adding."] and texts == ["4", "2"]
    sent = requests[0]
    assert sent["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert sent["tools"][0]["eager_input_streaming"] is True
    assert agent.messages[-1]["content"][0]["type"] == "thinking"


# ---- mid-conversation system messages -------------------------------------------------


def test_system_message_follows_the_prompt_and_stays_in_the_history() -> None:
    client = FakeClient([_text("Short.")])
    agent = _agent(client, model="claude-opus-5-5")
    agent.ask("Explain caching.", system="Answer in at most ten words.")
    sent = client.messages.create_calls[0]["messages"]
    assert sent[-1] == {"role": "system", "content": "Answer in at most ten words."}
    assert sent[-2]["role"] == "user"
    assert [m["role"] for m in agent.messages] == ["user", "system", "assistant"]


def test_system_message_is_rejected_on_models_without_support() -> None:
    agent = _agent(FakeClient([]), model="claude-sonnet-5")
    with pytest.raises(ValueError, match="system messages"):
        agent.ask("Hi", system="Be brief.")
    with pytest.raises(ValueError, match="empty"):
        _agent(FakeClient([])).ask("Hi", system=" ")


# ---- per-turn overrides -----------------------------------------------------------------


def test_effort_and_max_tokens_apply_to_one_turn_only() -> None:
    client = FakeClient([_text("a"), _text("b"), _text("c")])
    agent = _agent(client, effort="medium")
    agent.ask("Hard question", effort="high", max_tokens=64_000)
    agent.ask("Easy question")
    assert client.messages.stream_calls[0]["output_config"] == {"effort": "high"}
    assert client.messages.stream_calls[0]["max_tokens"] == 64_000
    assert client.messages.create_calls[0]["output_config"] == {"effort": "medium"}
    assert client.messages.create_calls[0]["max_tokens"] == 16_000
    with pytest.raises(ValueError, match="max_tokens"):
        agent.ask("x", max_tokens=0)


def test_overrides_are_reset_after_a_failed_turn() -> None:
    agent = _agent(FakeClient([_refusal()]))
    with pytest.raises(RefusalError):
        agent.ask("Hi", effort="max")
    assert agent._effort == "medium"


def test_async_overrides() -> None:
    client = AsyncFakeClient([_text("a")])
    agent = AsyncClaudeAgent(client=client, model="claude-opus-5-5")  # type: ignore[arg-type]
    asyncio.run(agent.ask("Hi", effort="low", system="Be brief."))
    call = client.messages.create_calls[0]
    assert call["output_config"] == {"effort": "low"}
    assert call["messages"][-1]["role"] == "system"


# ---- model checks ---------------------------------------------------------------------


def test_disabled_thinking_above_high_is_rejected_where_the_api_rejects_it() -> None:
    with pytest.raises(ValueError, match="'high' or below"):
        _agent(FakeClient([]), model="claude-haiku-5-5", thinking="disabled", effort="xhigh")
    agent = _agent(FakeClient([]), model="claude-opus-5", thinking="disabled", effort="high")
    with pytest.raises(ValueError, match="'high' or below"):
        agent.ask("Hi", effort="max")
    _agent(FakeClient([]), model="claude-opus-4-8", thinking="disabled", effort="max")


def test_between_tools_only_on_sonnet_55_up_to_high() -> None:
    with pytest.raises(ValueError, match="claude-sonnet-5-5"):
        _agent(FakeClient([]), model="claude-opus-5-5", thinking="between_tools")
    with pytest.raises(ValueError, match="'high' or below"):
        _agent(FakeClient([]), model="claude-sonnet-5-5", thinking="between_tools", effort="xhigh")
    _agent(FakeClient([]), model="anthropic.claude-sonnet-5-5", thinking="between_tools", effort="low")
    _agent(FakeClient([]), model="my-gateway-model", thinking="between_tools")  # unknown: not checked


# ---- eager input streaming ------------------------------------------------------------------


def test_eager_input_streaming_is_only_set_on_streamed_local_tools() -> None:
    client = FakeClient([_text("a"), _text("b")])
    server_tool = {"type": "web_fetch_20260209", "name": "web_fetch"}
    _agent(client, tools=[shout], server_tools=[server_tool], eager_input_streaming=True).ask("Hi")
    _agent(client, tools=[shout], server_tools=[server_tool], eager_input_streaming=True, stream=True).ask(
        "Hi"
    )
    buffered = client.messages.create_calls[0]["tools"]
    streamed = client.messages.stream_calls[0]["tools"]
    assert "eager_input_streaming" not in buffered[0]
    assert streamed[0]["eager_input_streaming"] is True
    assert "eager_input_streaming" not in streamed[1]


def test_invalid_eager_input_goes_back_to_claude() -> None:
    truncated = make_message(
        [{"type": "tool_use", "id": "toolu_1", "name": "shout", "input": {}}], stop_reason="tool_use"
    )
    client = FakeClient([truncated, _text("Retried.")])
    agent = _agent(client, tools=[shout], eager_input_streaming=True, stream=True)
    assert agent.ask("Shout hi") == "Retried."
    result = agent.messages[2]["content"][0]
    assert result["is_error"] is True and "invalid tool inputs" in result["content"]

    strict = _agent(FakeClient([truncated]), tools=[shout])
    with pytest.raises(ToolInputError):
        strict.ask("Shout hi")
    assert issubclass(ToolInputError, ToolExecutionError)
