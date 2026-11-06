"""Live reasoning across providers: what each stream transmits, in order with text and tool calls.

Every provider's recorded stream shape goes through the real SDK (Anthropic, OpenAI Responses,
OpenAI-compatible Chat Completions over HTTP) or the contract harness clients (Gemini, Bedrock)
and through the agent loop, so the order of ``ReasoningStatus``, ``ReasoningDelta``, ``TextDelta``
and tool call events is the order an interface receives. Reasoning text is only ever what the
provider sent: providers that transmit none produce status events at most, never text.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import anthropic
import httpx2
import openai
import pytest

from orkexa import (
    Agent,
    AsyncAgent,
    GeminiOptions,
    ReasoningDelta,
    ReasoningStatus,
    ReasoningStream,
    TextDelta,
    ToolCallDelta,
    ToolCallStarted,
)
from orkexa.core.types import ReasoningPart
from orkexa.providers.base import ProviderConfig
from orkexa.providers.registry import create_provider

from .contract.harness import BedrockHarness, GeminiHarness, Script, converse_response
from .helpers import add


def _shape(events: list[Any]) -> list[tuple[Any, ...]]:
    """The events an interface renders, with consecutive deltas of one kind merged."""
    shape: list[tuple[Any, ...]] = []
    for event in events:
        if isinstance(event, ReasoningStatus):
            item: tuple[Any, ...] = ("status", event.state, event.kind, event.redacted)
        elif isinstance(event, ReasoningDelta):
            item = ("reasoning", event.kind, event.text)
        elif isinstance(event, TextDelta):
            item = ("text", event.text)
        elif isinstance(event, ToolCallStarted):
            item = ("tool", event.name)
        else:
            continue
        if shape and item[0] in ("reasoning", "text") and shape[-1][:-1] == item[:-1]:
            shape[-1] = (*item[:-1], shape[-1][-1] + item[-1])
        else:
            shape.append(item)
    return shape


def _sse(events: list[tuple[str, dict[str, Any]]]) -> bytes:
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events).encode()


def _transport(bodies: list[bytes], seen: list[dict[str, Any]]) -> httpx2.MockTransport:
    queue = list(bodies)

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        return httpx2.Response(200, content=queue.pop(0), headers={"content-type": "text/event-stream"})

    return httpx2.MockTransport(handler)


# ---- Anthropic: summarized thinking, redacted thinking, signatures --------------------------------


def _claude_message(blocks: list[tuple[str, dict[str, Any], list[dict[str, Any]]]], stop: str) -> bytes:
    usage = {"input_tokens": 10, "output_tokens": 5}
    events: list[tuple[str, dict[str, Any]]] = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-opus-5-5",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": usage,
                },
            },
        )
    ]
    for index, (_, start, deltas) in enumerate(blocks):
        events.append(
            ("content_block_start", {"type": "content_block_start", "index": index, "content_block": start})
        )
        for delta in deltas:
            events.append(
                ("content_block_delta", {"type": "content_block_delta", "index": index, "delta": delta})
            )
        events.append(("content_block_stop", {"type": "content_block_stop", "index": index}))
    events.append(
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop, "stop_sequence": None},
                "usage": {"output_tokens": 5},
            },
        )
    )
    events.append(("message_stop", {"type": "message_stop"}))
    return _sse(events)


def _thinking(*pieces: str) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    deltas: list[dict[str, Any]] = [{"type": "thinking_delta", "thinking": piece} for piece in pieces]
    deltas.append({"type": "signature_delta", "signature": "sig"})
    return ("thinking", {"type": "thinking", "thinking": "", "signature": ""}, deltas)


def _text_block(*pieces: str) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    return ("text", {"type": "text", "text": ""}, [{"type": "text_delta", "text": p} for p in pieces])


def test_anthropic_thinking_redacted_thinking_tool_call_and_text_interleave() -> None:
    first = _claude_message(
        [
            _thinking("I should ", "add."),
            ("redacted", {"type": "redacted_thinking", "data": "ENCRYPTED"}, []),
            _text_block("Adding."),
            (
                "tool",
                {"type": "tool_use", "id": "toolu_1", "name": "add", "input": {}},
                [{"type": "input_json_delta", "partial_json": '{"a": 2, "b": 3}'}],
            ),
        ],
        "tool_use",
    )
    second = _claude_message([_thinking("Got 5."), _text_block("Five", ".")], "end_turn")
    seen: list[dict[str, Any]] = []
    client = anthropic.Anthropic(
        api_key="test", http_client=anthropic.DefaultHttpxClient(transport=_transport([first, second], seen))
    )
    events: list[Any] = []
    agent = Agent(model="anthropic:claude-opus-5-5", client=client, tools=[add], on_event=events.append)
    assert agent.ask("2 + 3?") == "Five."
    assert _shape(events) == [
        ("status", "started", "summary", False),
        ("reasoning", "summary", "I should add."),
        ("status", "finished", "summary", False),
        ("status", "started", "summary", True),
        ("status", "finished", "summary", False),
        ("text", "Adding."),
        ("tool", "add"),
        ("status", "started", "summary", False),
        ("reasoning", "summary", "Got 5."),
        ("status", "finished", "summary", False),
        ("text", "Five."),
    ]
    # The signature and the redacted block go back unchanged with the tool result.
    replayed = seen[1]["messages"][1]["content"]
    assert replayed[0] == {"type": "thinking", "thinking": "I should add.", "signature": "sig"}
    assert replayed[1] == {"type": "redacted_thinking", "data": "ENCRYPTED"}


def test_anthropic_omitted_thinking_has_status_but_no_text() -> None:
    body = _claude_message(
        [
            (
                "thinking",
                {"type": "thinking", "thinking": "", "signature": ""},
                [{"type": "signature_delta", "signature": "s"}],
            ),
            _text_block("ok"),
        ],
        "end_turn",
    )
    client = anthropic.Anthropic(
        api_key="test", http_client=anthropic.DefaultHttpxClient(transport=_transport([body], []))
    )
    events: list[Any] = []
    thoughts: list[str] = []
    agent = Agent(
        model="anthropic:claude-opus-5-5", client=client, on_event=events.append, on_thinking=thoughts.append
    )
    agent.ask("hi")
    assert _shape(events) == [
        ("status", "started", "summary", False),
        ("status", "finished", "summary", False),
        ("text", "ok"),
    ]
    assert thoughts == []


# ---- OpenAI Responses: reasoning summaries (the reasoning itself is never sent) ---------------------


def _openai_response(output: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "model": "gpt-5.5",
        "status": "completed",
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": 10,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 30,
            "output_tokens_details": {"reasoning_tokens": 20},
            "total_tokens": 40,
        },
    }


def _openai_stream(items: list[tuple[dict[str, Any], list[dict[str, Any]]]]) -> bytes:
    events: list[dict[str, Any]] = [
        {"type": "response.created", "response": {**_openai_response([]), "status": "in_progress"}}
    ]
    done: list[dict[str, Any]] = []
    for index, (item, middle) in enumerate(items):
        empty = {**item, "summary": []} if item["type"] == "reasoning" else item
        if item["type"] == "message":
            empty = {**item, "content": [], "status": "in_progress"}
        if item["type"] == "function_call":
            empty = {**item, "arguments": ""}
        events.append({"type": "response.output_item.added", "output_index": index, "item": empty})
        events += [{**event, "output_index": index, "item_id": item["id"]} for event in middle]
        events.append({"type": "response.output_item.done", "output_index": index, "item": item})
        done.append(item)
    events.append({"type": "response.completed", "response": _openai_response(done)})
    return "".join(
        f"event: {e['type']}\ndata: {json.dumps({**e, 'sequence_number': n})}\n\n"
        for n, e in enumerate(events)
    ).encode()


def _summary_events(*parts: list[str]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for index, pieces in enumerate(parts):
        part = {"type": "summary_text", "text": ""}
        events.append({"type": "response.reasoning_summary_part.added", "summary_index": index, "part": part})
        for piece in pieces:
            events.append(
                {"type": "response.reasoning_summary_text.delta", "summary_index": index, "delta": piece}
            )
    return events


def _message(text: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    item = {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }
    middle = [
        {
            "type": "response.content_part.added",
            "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []},
        },
        {"type": "response.output_text.delta", "content_index": 0, "delta": text, "logprobs": []},
    ]
    return item, middle


@pytest.mark.parametrize("asynchronous", [False, True])
def test_openai_summary_parts_tool_call_and_text_interleave(asynchronous: bool) -> None:
    reasoning = {
        "type": "reasoning",
        "id": "rs_1",
        "summary": [{"type": "summary_text", "text": "Plan."}, {"type": "summary_text", "text": "Add."}],
        "encrypted_content": "gAAA",
    }
    call = {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": "add",
        "arguments": '{"a": 2, "b": 3}',
        "status": "completed",
    }
    first = _openai_stream(
        [
            (reasoning, _summary_events(["Pl", "an."], ["Add."])),
            (call, [{"type": "response.function_call_arguments.delta", "delta": '{"a": 2, "b": 3}'}]),
        ]
    )
    # The second answer reasons without a summary: a phase, no text.
    second = _openai_stream([({"type": "reasoning", "id": "rs_2", "summary": []}, []), _message("Five.")])
    seen: list[dict[str, Any]] = []
    transport = _transport([first, second], seen)
    events: list[Any] = []
    kwargs: dict[str, Any] = {
        "model": "openai:gpt-5.5",
        "tools": [add],
        "on_event": events.append,
        "effort": "low",
    }
    options = openai_options()
    if asynchronous:
        client: Any = openai.AsyncOpenAI(
            api_key="sk-test", http_client=httpx2.AsyncClient(transport=transport)
        )
        agent: Any = AsyncAgent(client=client, options=options, **kwargs)
        assert asyncio.run(agent.ask("2 + 3?")) == "Five."
    else:
        client = openai.OpenAI(api_key="sk-test", http_client=httpx2.Client(transport=transport))
        agent = Agent(client=client, options=options, **kwargs)
        assert agent.ask("2 + 3?") == "Five."
    assert seen[0]["reasoning"] == {"effort": "low", "summary": "auto"}
    assert _shape(events) == [
        ("status", "started", "summary", False),
        ("reasoning", "summary", "Plan.\n\nAdd."),
        ("status", "finished", "summary", False),
        ("tool", "add"),
        ("status", "started", "summary", False),
        ("status", "finished", "summary", False),
        ("text", "Five."),
    ]
    reasoning_parts = [p for p in agent.messages[1].parts() if isinstance(p, ReasoningPart)]
    assert reasoning_parts[0].text == "Plan.\nAdd."  # the stored summary is the provider's


def openai_options() -> Any:
    from orkexa import OpenAIOptions

    return OpenAIOptions(reasoning_summary="auto")


# ---- Gemini: thought summaries (include_thoughts) ------------------------------------------------


def test_gemini_thought_summaries_tool_call_and_text_interleave() -> None:
    harness = GeminiHarness()
    first = harness._response(
        [
            {"text": "Thinking about ", "thought": True},
            {"text": "the sum.", "thought": True},
            {
                "function_call": {"id": "c1", "name": "add", "args": {"a": 2, "b": 3}},
                "thought_signature": "c2ln",
            },
        ]
    )
    second = harness._response([{"text": "Checked.", "thought": True}, {"text": "Five."}])
    events: list[Any] = []
    agent = harness.agent(
        [first, second], tools=[add], on_event=events.append, options=GeminiOptions(include_thoughts=True)
    )
    assert agent.ask("2 + 3?") == "Five."
    assert harness.script.calls[0]["config"]["thinking_config"] == {"include_thoughts": True}
    assert _shape(events) == [
        ("status", "started", "summary", False),
        ("reasoning", "summary", "Thinking about the sum."),
        ("status", "finished", "summary", False),
        ("tool", "add"),
        ("status", "started", "summary", False),
        ("reasoning", "summary", "Checked."),
        ("status", "finished", "summary", False),
        ("text", "Five."),
    ]


def test_gemini_without_include_thoughts_streams_no_reasoning() -> None:
    harness = GeminiHarness()
    events: list[Any] = []
    agent = harness.agent([harness.text("Five.")], on_event=events.append, stream=True)
    agent.ask("2 + 3?")
    assert _shape(events) == [("text", "Five.")]


def test_gemini_thought_summary_at_the_end_of_the_stream_is_closed() -> None:
    harness = GeminiHarness()
    response = harness._response([{"text": "Done.", "thought": True}])
    events: list[Any] = []
    agent = harness.agent([response], on_event=events.append, stream=True)
    agent.ask("x")
    assert _shape(events)[-1] == ("status", "finished", "summary", False)


# ---- Bedrock Converse: reasoningContent (text, signature, redacted) ------------------------------


def test_bedrock_claude_reasoning_redacted_tool_call_and_text_interleave() -> None:
    harness = BedrockHarness()
    harness.script = Script(
        [
            converse_response(
                [
                    {"reasoningContent": {"reasoningText": {"text": "plan it", "signature": "sig"}}},
                    {"reasoningContent": {"redactedContent": b"\x00secret"}},
                    {"toolUse": {"toolUseId": "t1", "name": "add", "input": {"a": 2, "b": 3}}},
                ],
                "tool_use",
            ),
            converse_response([{"text": "Five."}]),
        ]
    )
    events: list[Any] = []
    agent = Agent(
        provider="bedrock",
        client=harness.client(False),
        **harness.agent_kwargs(),
        tools=[add],
        on_event=events.append,
        stream=True,
    )
    assert agent.ask("2 + 3?") == "Five."
    assert _shape(events) == [
        ("status", "started", "summary", False),
        ("reasoning", "summary", "plan it"),
        ("status", "finished", "summary", False),
        ("status", "started", "summary", True),
        ("status", "finished", "summary", False),
        ("tool", "add"),
        ("text", "Five."),
    ]


def test_bedrock_open_weight_model_reasoning_is_thinking_text() -> None:
    from orkexa.providers.bedrock import _BedrockStream, _reasoning_kind

    assert _reasoning_kind("eu.anthropic.claude-sonnet-4-5-20250929-v1:0") == "summary"
    assert _reasoning_kind("us.deepseek.r1-v1:0") == "thinking"
    state = _BedrockStream("r", _reasoning_kind("us.deepseek.r1-v1:0"))
    events = [
        *state.translate(
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"reasoningContent": {"text": "hm"}}}}
        ),
        *state.translate({"contentBlockStop": {"contentBlockIndex": 0}}),
        *state.translate({"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"text": "A"}}}),
    ]
    assert _shape(events) == [
        ("status", "started", "thinking", False),
        ("reasoning", "thinking", "hm"),
        ("status", "finished", "thinking", False),
        ("text", "A"),
    ]


# ---- OpenAI-compatible servers (Ollama, vLLM): reasoning text deltas ------------------------------


def _chunk(delta: dict[str, Any], finish: str | None = None, usage: dict[str, Any] | None = None) -> str:
    body: dict[str, Any] = {
        "id": "chat-1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "qwen3:4b",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    if usage is not None:
        body["usage"] = usage
    return f"data: {json.dumps(body)}\n\n"


def _chat_stream(*deltas: dict[str, Any], finish: str = "stop") -> bytes:
    usage = {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}
    chunks = [_chunk(delta) for delta in deltas] + [_chunk({}, finish, usage), "data: [DONE]\n\n"]
    return "".join(chunks).encode()


@pytest.mark.parametrize("field", ["reasoning", "reasoning_content"])
def test_compatible_reasoning_deltas_tool_call_and_text_interleave(field: str) -> None:
    first = _chat_stream(
        {"role": "assistant", field: "Need "},
        {field: "a sum."},
        {
            "tool_calls": [
                {"index": 0, "id": "c1", "type": "function", "function": {"name": "add", "arguments": ""}}
            ]
        },
        {"tool_calls": [{"index": 0, "function": {"arguments": '{"a": 2, "b": 3}'}}]},
        finish="tool_calls",
    )
    second = _chat_stream({field: "It is 5."}, {"content": "Fi"}, {"content": "ve."})
    seen: list[dict[str, Any]] = []
    client = openai.OpenAI(
        api_key="x",
        base_url="http://127.0.0.1:11434/v1",
        http_client=httpx2.Client(transport=_transport([first, second], seen)),
    )
    events: list[Any] = []
    agent = Agent(
        model="ollama:qwen3:4b",
        client=client,
        tools=[add],
        capabilities={"tools": "supported", "reasoning": "supported"},
        on_event=events.append,
    )
    assert agent.ask("2 + 3?") == "Five."
    assert _shape(events) == [
        ("status", "started", "thinking", False),
        ("reasoning", "thinking", "Need a sum."),
        ("status", "finished", "thinking", False),
        ("tool", "add"),
        ("status", "started", "thinking", False),
        ("reasoning", "thinking", "It is 5."),
        ("status", "finished", "thinking", False),
        ("text", "Five."),
    ]
    # Reasoning is never sent back to the server.
    assert all("reasoning" not in message for message in seen[1]["messages"])


def test_compatible_model_without_reasoning_streams_text_only() -> None:
    body = _chat_stream({"content": "Hi"})
    client = openai.OpenAI(
        api_key="x",
        base_url="http://127.0.0.1:11434/v1",
        http_client=httpx2.Client(transport=_transport([body], [])),
    )
    events: list[Any] = []
    Agent(model="ollama:llama3.1:8b", client=client, on_event=events.append).ask("hi")
    assert _shape(events) == [("text", "Hi")]
    assert not [e for e in events if isinstance(e, (ReasoningDelta, ReasoningStatus, ToolCallDelta))]


# ---- what each provider declares ---------------------------------------------------------------


def _adapter(name: str, **config: Any) -> Any:
    return create_provider(name, ProviderConfig(client=object(), **config))


@pytest.mark.parametrize(
    ("provider", "model", "expected"),
    [
        ("anthropic", "claude-opus-5-5", ReasoningStream("summary", True, "thinking_display")),
        ("openai", "gpt-5.5", ReasoningStream("summary", True, "reasoning_summary")),
        ("gemini", "gemini-3-pro", ReasoningStream("summary", False, "include_thoughts")),
        ("gemini", "gemini-2.5-flash", ReasoningStream("summary", False, "include_thoughts")),
        ("bedrock", "eu.anthropic.claude-sonnet-4-5-20250929-v1:0", ReasoningStream("summary", True)),
        ("bedrock", "us.deepseek.r1-v1:0", ReasoningStream("unknown", True)),
        ("ollama", "qwen3:4b", ReasoningStream("unknown")),
    ],
)
def test_declared_reasoning_streams(provider: str, model: str, expected: ReasoningStream) -> None:
    assert _adapter(provider).reasoning_stream(model) == expected


def test_declared_reasoning_stream_without_text() -> None:
    assert _adapter("openai").reasoning_stream("gpt-4o").text == "none"
    assert "not a reasoning model" in _adapter("openai").reasoning_stream("gpt-4o").reason
    assert _adapter("gemini").reasoning_stream("gemini-2.0-flash").text == "none"
    ollama = _adapter("ollama")
    caps = ollama.capabilities("qwen3:4b")
    assert ollama.reasoning_stream("qwen3:4b", caps.merged({"reasoning": "supported"})).text == "thinking"
    assert ollama.reasoning_stream("llama3.1:8b", caps.merged({"reasoning": "unsupported"})).text == "none"


def test_fake_provider_scripts_both_kinds_and_omitted_reasoning() -> None:
    from orkexa.providers.fake import FakeModel, text

    events: list[Any] = []
    model = FakeModel([text("a", reasoning="sum", reasoning_kind="summary"), text("b", reasoning="")])
    agent = Agent(model="fake:m", client=model, on_event=events.append)
    agent.ask("x")
    agent.ask("y")
    assert _shape(events) == [
        ("status", "started", "summary", False),
        ("reasoning", "summary", "sum"),
        ("status", "finished", "summary", False),
        ("text", "a"),
        ("status", "started", "thinking", True),
        ("status", "finished", "thinking", False),
        ("text", "b"),
    ]
