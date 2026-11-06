"""Chat Completions messages as servers accept them, and refused requests explained.

Regression tests for Ollama's HTTP 400 ``invalid message content type: <nil>`` in multi-step agent
runs: an assistant turn without text and without tool calls (a reasoning model that only thought,
or an empty answer) was sent as ``{"role": "assistant", "content": null}``. Ollama's
OpenAI-compatible API (``openai/openai.go``, ``FromChatRequest``) accepts a ``null`` or missing
``content`` only next to tool calls; so does the Chat Completions protocol itself.
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx2
import openai
from openai.types.chat import ChatCompletion
from pydantic import BaseModel
import pytest

from orkexa import (
    Agent,
    CompatibleOptions,
    HistoryConversionError,
    InvalidRequestError,
    ProviderConnectionError,
    RateLimitError,
)
from orkexa.core.conversion import message_outline
from orkexa.core.runtime import FORMAT_PROMPT
from orkexa.core.types import Message, ModelRequest, ReasoningPart, TextPart, ToolCallPart, ToolResultPart
from orkexa.providers.base import ProviderConfig
from orkexa.providers.chat_completions import ChatMessageRules, check_chat_messages
from orkexa.providers.registry import create_provider

from .contract.harness import Script
from .helpers import add


def provider(name: str = "ollama") -> Any:
    config: dict[str, Any] = {"base_url": "http://localhost:9/v1"} if name == "compatible" else {}
    return create_provider(name, ProviderConfig(client=object(), **config))


def encode(name: str, messages: list[Message]) -> list[dict[str, Any]]:
    adapter = provider(name)
    return adapter.prepare(ModelRequest(model="m", messages=messages, max_tokens=100))["messages"]


CALL = ToolCallPart(id="call_1", name="add", arguments={"a": 1, "b": 2})
RESULT = ToolResultPart(call_id="call_1", name="add", content="3")


def history(*last: Message) -> list[Message]:
    return [
        Message(role="user", content="Add."),
        Message(role="assistant", content=[ReasoningPart(text="I add."), CALL]),
        Message(role="user", content=[RESULT]),
        *last,
    ]


# ---- encoding ---------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["ollama", "vllm", "compatible"])
def test_a_turn_with_only_reasoning_is_an_empty_text_not_null(name: str) -> None:
    sent = encode(
        name,
        history(
            Message(role="assistant", content=[ReasoningPart(text="It is 3.")]),
            Message(role="user", content=FORMAT_PROMPT),
        ),
    )
    assert sent[3]["role"] == "assistant"
    assert sent[3]["content"] == ""
    assert "tool_calls" not in sent[3]


@pytest.mark.parametrize("name", ["ollama", "vllm", "compatible"])
def test_an_empty_answer_is_an_empty_text(name: str) -> None:
    sent = encode(
        name, history(Message(role="assistant", content=[]), Message(role="user", content="Go on."))
    )
    assert sent[3] == {"role": "assistant", "content": ""}


@pytest.mark.parametrize("name", ["ollama", "vllm", "compatible"])
def test_tool_calls_without_text_keep_null_content(name: str) -> None:
    """The canonical form is unchanged: ``null`` content next to tool calls."""
    sent = encode(name, history())
    assert sent[1]["content"] is None
    assert sent[1]["tool_calls"][0]["function"] == {"name": "add", "arguments": '{"a": 1, "b": 2}'}
    assert sent[2] == {"role": "tool", "tool_call_id": "call_1", "content": "3"}


def test_text_is_kept_as_it_was() -> None:
    sent = encode(
        "ollama",
        history(Message(role="assistant", content=[ReasoningPart(text="t"), TextPart(text="It is 3.")])),
    )
    assert sent[3]["content"] == "It is 3."


@pytest.mark.parametrize("name", ["ollama", "vllm"])
def test_ollama_and_vllm_get_their_reasoning_back_generic_endpoints_do_not(name: str) -> None:
    """Ollama and vLLM read ``reasoning`` of an assistant message as the model's thinking
    (ORK-042: checked in the vLLM 0.33.0 source); generic endpoints declare no such
    field and get none."""
    last = Message(role="assistant", content=[ReasoningPart(text="It is 3.")])
    sent = encode(name, history(last))
    assert sent[1]["reasoning"] == "I add." and sent[1]["content"] is None
    assert sent[3] == {"role": "assistant", "content": "", "reasoning": "It is 3."}
    assert all("reasoning" not in m for m in encode("compatible", history(last)))
    assert provider(name).message_rules == ChatMessageRules(reasoning_field="reasoning")
    assert provider("compatible").message_rules == ChatMessageRules()


def test_redacted_or_textless_reasoning_is_not_replayed() -> None:
    sent = encode(
        "ollama",
        history(
            Message(
                role="assistant",
                content=[ReasoningPart(redacted=True, text="x"), ReasoningPart(), TextPart(text="ok")],
            )
        ),
    )
    assert sent[3] == {"role": "assistant", "content": "ok"}


def test_a_tool_result_without_content_is_an_empty_text() -> None:
    sent = encode(
        "ollama",
        [
            Message(role="user", content="Add."),
            Message(role="assistant", content=[CALL]),
            Message(role="user", content=[ToolResultPart(call_id="call_1", name="add", content=None)]),
        ],
    )
    assert sent[2] == {"role": "tool", "tool_call_id": "call_1", "content": ""}


# ---- the protocol's message rules -----------------------------------------------------------


@pytest.mark.parametrize(
    ("messages", "problem"),
    [
        ([{"role": "user", "content": None}], "message 0 (user): content is missing"),
        ([{"role": "system"}], "message 0 (system): content is missing"),
        (
            [{"role": "user", "content": "a"}, {"role": "assistant", "content": None}],
            "message 1 (assistant): content is missing",
        ),
        (
            [{"role": "user", "content": "a"}, {"role": "assistant", "content": None, "tool_calls": []}],
            "message 1 (assistant): content is missing",
        ),
        (
            [{"role": "user", "content": "a"}, {"role": "tool", "tool_call_id": "x", "content": "1"}],
            "answers no tool call",
        ),
        (
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "", "function": {"name": "f", "arguments": "{}"}}],
                }
            ],
            "has no id",
        ),
        (
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "a", "function": {"name": "f", "arguments": {}}}],
                }
            ],
            "JSON string",
        ),
        ([{"role": "user", "content": 3}], "content of type int"),
        ([{"role": "user", "content": [{"text": "x"}]}], "needs a type"),
        ([{"role": "robot", "content": "x"}], "unknown role"),
    ],
)
def test_messages_the_protocol_does_not_allow_are_refused_locally(
    messages: list[dict[str, Any]], problem: str
) -> None:
    with pytest.raises(HistoryConversionError, match=re.escape(problem)):
        check_chat_messages(messages)


def test_the_refusal_names_the_message_never_its_content() -> None:
    with pytest.raises(HistoryConversionError) as caught:
        check_chat_messages(
            [{"role": "user", "content": "secret customer data"}, {"role": "user", "content": None}]
        )
    assert "secret" not in str(caught.value)


def test_valid_conversations_pass() -> None:
    check_chat_messages(
        [
            {"role": "system", "content": "s"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "a"},
                    {"type": "image_url", "image_url": {"url": "data:"}},
                ],
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "a", "type": "function", "function": {"name": "f", "arguments": "{}"}},
                    {"id": "b", "type": "function", "function": {"name": "f", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "a", "content": "1"},
            {"role": "tool", "tool_call_id": "b", "content": ""},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "go"},
        ]
    )


# ---- streaming: tool calls that share an index ------------------------------------------------


def _chunk(delta: dict[str, Any], finish: str | None = None) -> Any:
    from openai.types.chat import ChatCompletionChunk

    return ChatCompletionChunk.model_validate(
        {
            "id": "c",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "m",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
    )


def _call(index: int, call_id: str, name: str, arguments: str) -> dict[str, Any]:
    return {
        "tool_calls": [
            {
                "index": index,
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ]
    }


def test_complete_tool_calls_in_separate_chunks_with_index_zero_stay_separate() -> None:
    """Servers that send each call complete in its own chunk may number every one ``0``."""
    adapter = provider("ollama")
    chunks = [
        _chunk({"reasoning": "two sums"}),
        _chunk(_call(0, "call_a", "add", '{"a": 1, "b": 2}')),
        _chunk(_call(0, "call_b", "add", '{"a": 3, "b": 4}')),
        _chunk({}, "tool_calls"),
    ]
    adapter.client = type(
        "C",
        (),
        {
            "chat": type(
                "Ch",
                (),
                {"completions": type("Co", (), {"create": staticmethod(lambda **_: iter(chunks))})()},
            )()
        },
    )()
    request = ModelRequest(
        model="m", messages=[Message(role="user", content="x")], max_tokens=10, tools=[add.definition()]
    )
    final = list(adapter.stream(adapter.prepare(request), request))[-1].response
    assert [(c.id, c.arguments) for c in final.tool_calls] == [
        ("call_a", {"a": 1, "b": 2}),
        ("call_b", {"a": 3, "b": 4}),
    ]


def test_deltas_of_one_call_are_still_merged() -> None:
    adapter = provider("ollama")
    chunks = [
        _chunk(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_a",
                        "type": "function",
                        "function": {"name": "add", "arguments": ""},
                    }
                ]
            }
        ),
        _chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"a": 1, '}}]}),
        _chunk({"tool_calls": [{"index": 0, "function": {"arguments": '"b": 2}'}}]}),
        _chunk(
            {
                "tool_calls": [
                    {
                        "index": 1,
                        "id": "call_b",
                        "type": "function",
                        "function": {"name": "add", "arguments": '{"a": 0, "b": 0}'},
                    }
                ]
            }
        ),
        _chunk({}, "tool_calls"),
    ]
    adapter.client = type(
        "C",
        (),
        {
            "chat": type(
                "Ch",
                (),
                {"completions": type("Co", (), {"create": staticmethod(lambda **_: iter(chunks))})()},
            )()
        },
    )()
    request = ModelRequest(
        model="m", messages=[Message(role="user", content="x")], max_tokens=10, tools=[add.definition()]
    )
    final = list(adapter.stream(adapter.prepare(request), request))[-1].response
    assert [(c.id, c.arguments) for c in final.tool_calls] == [
        ("call_a", {"a": 1, "b": 2}),
        ("call_b", {"a": 0, "b": 0}),
    ]


# ---- the agent flow of the failed run ------------------------------------------------------------


class Downtime(BaseModel):
    analysis: str
    downtime_share_percent: float


def _completion(message: dict[str, Any], finish: str) -> ChatCompletion:
    return ChatCompletion.model_validate(
        {
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "qwen3.5:4b",
            "choices": [{"index": 0, "message": {"role": "assistant", **message}, "finish_reason": finish}],
        }
    )


class FakeOllama:
    """Ollama 0.40.14's OpenAI-compatible validation (``FromChatRequest``) in front of scripted answers."""

    def __init__(self, answers: list[Any]) -> None:
        self.script = Script(answers)
        self.chat = type("Chat", (), {"completions": self})()

    def create(self, **kwargs: Any) -> Any:
        for message in kwargs["messages"]:
            if message.get("content") is None and message.get("tool_calls") is None:
                raise openai.BadRequestError(
                    "Error code: 400 - invalid message content type: <nil>",
                    response=httpx2.Response(
                        400, request=httpx2.Request("POST", "http://localhost:11434/v1/chat/completions")
                    ),
                    body={
                        "error": {
                            "message": "invalid message content type: <nil>",
                            "type": "invalid_request_error",
                        }
                    },
                )
        return self.script.next(kwargs)


def _tool_round(n: int) -> ChatCompletion:
    call = {
        "id": f"call_{n}",
        "type": "function",
        "function": {"name": "add", "arguments": json.dumps({"a": n, "b": 1})},
    }
    return _completion({"content": "", "reasoning": f"step {n}", "tool_calls": [call]}, "tool_calls")


def _agent(client: Any, **kwargs: Any) -> Agent[Any]:
    return Agent(
        provider="ollama",
        model="qwen3.5:4b",
        client=client,
        tools=[add],
        output_format=Downtime,
        capabilities={"tools": "supported", "reasoning": "supported"},
        options=CompatibleOptions(send_reasoning_effort=True),
        **kwargs,
    )


def test_the_downtime_analyst_flow_reaches_its_structured_answer() -> None:
    """Five tool rounds with reasoning and no text, a tool phase that ends with reasoning only, then
    the request for the structured answer (FORMAT_PROMPT, no tools): the 6th and 7th requests failed
    with HTTP 400 before."""
    answers = [
        *(_tool_round(n) for n in range(1, 6)),
        _completion({"content": "", "reasoning": "share is 4.65 %"}, "stop"),
    ]
    answers.append(
        _completion(
            {"content": '{"analysis": "ok", "downtime_share_percent": 4.65}', "reasoning": "json"}, "stop"
        )
    )
    client = FakeOllama(answers)
    agent = _agent(client)
    assert agent.ask("Analyze") == Downtime(analysis="ok", downtime_share_percent=4.65)
    calls = client.script.calls
    assert len(calls) == 7
    last = calls[-1]
    assert "tools" not in last and last["response_format"]["type"] == "json_schema"
    assert last["messages"][-2] == {"role": "assistant", "content": "", "reasoning": "share is 4.65 %"}
    assert last["messages"][-1] == {"role": "user", "content": FORMAT_PROMPT}
    # Every tool round's reasoning went back with its tool calls (Qwen's interleaved thinking).
    assert [m.get("reasoning") for m in calls[5]["messages"] if m.get("tool_calls")] == [
        f"step {n}" for n in range(1, 6)
    ]
    assert [m["tool_call_id"] for m in calls[5]["messages"] if m["role"] == "tool"] == [
        f"call_{n}" for n in range(1, 6)
    ]


def test_a_refused_conversation_is_explained_without_its_content() -> None:
    """A provider 400 about the messages is not retried, says what was refused and outlines the
    conversation by roles and part kinds only."""
    client = FakeOllama([])

    def refuse(**kwargs: Any) -> Any:
        raise openai.BadRequestError(
            "Error code: 400 - invalid message content type: <nil>",
            response=httpx2.Response(
                400, request=httpx2.Request("POST", "http://localhost:11434/v1/chat/completions")
            ),
            body=None,
        )

    client.create = refuse  # type: ignore[method-assign]
    agent = _agent(client)
    with pytest.raises(InvalidRequestError) as caught:
        agent.ask("customer secret 4711")
    error = caught.value
    assert error.reason == "messages" and error.status_code == 400 and not error.retryable
    assert error.outline == "#0 user[text]"
    assert "4711" not in str(error)
    assert str(error).endswith("[request messages: #0 user[text]]")


@pytest.mark.parametrize(
    ("status", "text", "reason"),
    [
        (400, "invalid message content type: <nil>", "messages"),
        (
            400,
            "messages with role 'tool' must be a response to a preceeding message with 'tool_calls'",
            "messages",
        ),
        (400, "messages.3.content.0: unexpected `tool_use_id` found in `tool_result` blocks", "messages"),
        (400, "This model's maximum context length is 8192 tokens", "context_length"),
        (400, "prompt is too long: 210000 tokens > 200000 maximum", "context_length"),
        (413, "request entity too large", "context_length"),
        (404, "model 'qwen9:7b' not found", "model_not_found"),
        (400, "Unsupported parameter: 'max_tokens' is not supported with this model", "parameter"),
        (400, "bad", None),
    ],
)
def test_provider_4xx_answers_are_classified(status: int, text: str, reason: str | None) -> None:
    assert provider("ollama").invalid_request_reason(status, text) == reason


def test_transient_failures_stay_retryable_and_unclassified() -> None:
    adapter = provider("ollama")
    request = httpx2.Request("POST", "http://localhost:11434/v1/chat/completions")
    limited = adapter.map_error(
        openai.RateLimitError("slow", response=httpx2.Response(429, request=request), body=None)
    )
    assert isinstance(limited, RateLimitError) and limited.retryable
    down = adapter.map_error(openai.APIConnectionError(request=request))
    assert isinstance(down, ProviderConnectionError) and down.retryable


def test_outline_of_a_long_conversation_keeps_its_ends() -> None:
    messages = [Message(role="user", content="a")] + [
        Message(role="assistant", content=[TextPart(text="b")]) for _ in range(30)
    ]
    outline = message_outline(messages, limit=6)
    assert outline.startswith("#0 user[text] #1 assistant[text] ... 25 more ...")
    assert outline.endswith("#30 assistant[text]")
