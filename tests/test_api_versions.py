"""Provider API contracts of the supported SDK versions: OpenAI response statuses and
background mode, parameters Anthropic API version 2026-10-21 removed, and the real SDK
stream helpers (through a mocked HTTP transport, no network access)."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import anthropic
import httpx2
import openai
from pydantic import BaseModel
import pytest

from orkexa import (
    Agent,
    AnthropicOptions,
    AsyncAgent,
    ClaudeAgent,
    CompatibleOptions,
    ConfigurationError,
    IncompleteResponseError,
    Message,
    OpenAIOptions,
    ProviderServerError,
    UnexpectedResponseError,
    UsageTracker,
)
from orkexa.core.types import ModelRequest
from orkexa.providers.openai import ResponsesCodec

from .contract.harness import AnthropicHarness, CompatibleResponsesHarness, OpenAIHarness

# ---- OpenAI response statuses ----------------------------------------------------------------


@pytest.mark.parametrize("status", ["queued", "in_progress"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_unfinished_openai_response_is_not_an_answer(status: str, asynchronous: bool) -> None:
    harness = OpenAIHarness()
    responses = [harness._response([], status=status)]
    agent = harness.async_agent(responses) if asynchronous else harness.agent(responses)
    with pytest.raises(UnexpectedResponseError, match=f"still '{status}'"):
        asyncio.run(agent.ask("hi")) if asynchronous else agent.ask("hi")
    assert agent.messages == []


def test_cancelled_openai_response_is_not_committed_but_billed() -> None:
    harness = OpenAIHarness()
    tracker = UsageTracker()
    cancelled = harness._response([harness._message("half an answer")], status="cancelled")
    agent = harness.agent([cancelled], usage_tracker=tracker)
    with pytest.raises(UnexpectedResponseError, match="cancelled"):
        agent.ask("hi")
    assert agent.messages == []
    assert len(tracker.records) == 1


@pytest.mark.parametrize("harness_class", [OpenAIHarness, CompatibleResponsesHarness])
def test_background_mode_is_rejected_before_sending(harness_class: type[OpenAIHarness]) -> None:
    harness = harness_class()
    kwargs = harness.agent_kwargs()
    if harness_class is OpenAIHarness:
        kwargs["options"] = OpenAIOptions(request_options={"background": True})
    else:
        kwargs["options"] = CompatibleOptions(api="responses", request_options={"background": True})
    with pytest.raises(ConfigurationError, match="background"):
        Agent(provider=harness.provider, client=harness.client(False), **kwargs)
    agent = harness.agent([harness.text("never")])
    agent.options = kwargs["options"]  # changed later: refused before sending
    with pytest.raises(ConfigurationError, match="background"):
        agent.ask("hi")
    assert harness.script.calls == []


def test_background_false_is_accepted() -> None:
    harness = OpenAIHarness()
    agent = harness.agent([harness.text("ok")], options=OpenAIOptions(request_options={"background": False}))
    assert agent.ask("hi") == "ok"
    assert harness.script.calls[0]["background"] is False


def test_failed_stream_raises_the_provider_error() -> None:
    harness = OpenAIHarness()
    failed = harness._response([], status="failed")
    failed.error = openai.types.responses.ResponseError(code="server_error", message="upstream failed")
    agent = harness.agent([failed], stream=True)
    with pytest.raises(ProviderServerError, match="upstream failed"):
        agent.ask("hi")
    assert agent.messages == []


# ---- the real OpenAI stream helper ---------------------------------------------------------------


def _usage() -> dict[str, Any]:
    return {
        "input_tokens": 10,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 5,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 15,
    }


def _openai_response(status: str, output: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "model": "gpt-5.5",
        "status": status,
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "incomplete_details": {"reason": "max_output_tokens"} if status == "incomplete" else None,
        "usage": _usage(),
    }


def _truncated_stream() -> bytes:
    message = {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "incomplete",
        "content": [{"type": "output_text", "text": "partial", "annotations": []}],
    }
    events = [
        {"type": "response.created", "response": _openai_response("in_progress", [])},
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {**message, "status": "in_progress", "content": []},
        },
        {
            "type": "response.content_part.added",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []},
        },
        {
            "type": "response.output_text.delta",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "delta": "partial",
            "logprobs": [],
        },
        {"type": "response.incomplete", "response": _openai_response("incomplete", [message])},
    ]
    return "".join(
        f"event: {e['type']}\ndata: {json.dumps({**e, 'sequence_number': n})}\n\n"
        for n, e in enumerate(events)
    ).encode()


def _sse(request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=_truncated_stream())


@pytest.mark.parametrize("asynchronous", [False, True])
def test_truncated_stream_through_the_real_openai_sdk(asynchronous: bool) -> None:
    """``ResponseStream.get_final_response()`` raises a bare RuntimeError without ``response.completed``."""
    transport = httpx2.MockTransport(_sse)
    tracker = UsageTracker()
    agent: Agent[Any] | AsyncAgent[Any]
    if asynchronous:
        client: Any = openai.AsyncOpenAI(
            api_key="sk-test", http_client=httpx2.AsyncClient(transport=transport)
        )
        agent = AsyncAgent(model="openai:gpt-5.5", client=client, stream=True, usage_tracker=tracker)
    else:
        client = openai.OpenAI(api_key="sk-test", http_client=httpx2.Client(transport=transport))
        agent = Agent(model="openai:gpt-5.5", client=client, stream=True, usage_tracker=tracker)
    deltas: list[str] = []
    agent.on_text = deltas.append
    with pytest.raises(IncompleteResponseError, match="max_output_tokens"):
        asyncio.run(agent.ask("hi")) if isinstance(agent, AsyncAgent) else agent.ask("hi")
    assert deltas == ["partial"]
    assert agent.messages == []
    assert tracker.usage.requests == 1 and tracker.usage.output_tokens == 5


def test_unknown_content_before_text_keeps_the_text_for_replay() -> None:
    codec = ResponsesCodec("openai")
    item = {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "content": [
            {"type": "future_content", "value": "opaque"},
            {"type": "output_text", "text": "Visible answer"},
        ],
    }
    parts, _, _ = codec.decode([item], ModelRequest(model="gpt-5.5", messages=[]))
    assert [p.text for p in parts] == ["Visible answer"]  # type: ignore[attr-defined]
    replay = codec.encode([Message(role="assistant", content=parts, provider="openai")])
    assert replay == [item]


# ---- Anthropic API version 2026-10-21 ------------------------------------------------------------


@pytest.mark.parametrize("name", ["temperature", "top_p", "top_k"])
def test_removed_sampling_parameters_are_rejected(name: str) -> None:
    with pytest.raises(ConfigurationError, match="2026-10-21"):
        ClaudeAgent(client=object(), request_options={name: 0.5})
    with pytest.raises(ConfigurationError, match=name):
        AnthropicOptions(request_options={name: 0.5})
    agent = ClaudeAgent(client=object())
    agent.request_options[name] = 0.5  # changed after construction: refused before sending
    with pytest.raises(ConfigurationError):
        agent.ask("hi")


def test_sampling_parameters_reach_a_client_pinned_to_the_old_api_version() -> None:
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    client = anthropic.Anthropic(
        api_key="sk-test",
        default_headers={"anthropic-version": "2023-06-01"},
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
    )
    agent = ClaudeAgent(
        model="claude-sonnet-4-6",
        client=client,
        stream=False,
        request_options={"extra_body": {"temperature": 0.2}},
    )
    assert agent.ask("hi") == "ok"
    (request,) = seen
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert json.loads(request.content)["temperature"] == 0.2


class _Answer(BaseModel):
    value: int


def test_beta_token_count_sends_the_output_format_as_api_field() -> None:
    harness = AnthropicHarness()  # checks the arguments against the SDK's count_tokens()
    agent = harness.agent([], output_format=_Answer, options=AnthropicOptions(thinking_display="updates"))
    assert agent.count_tokens("hi") == 321
    (call,) = harness.script.calls
    assert call["betas"] and call["output_config"]["format"]["type"] == "json_schema"
    assert "output_format" not in call
