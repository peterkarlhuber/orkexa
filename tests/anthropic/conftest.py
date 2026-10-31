"""Test helpers: a fake Anthropic client that replays scripted responses."""

from __future__ import annotations

from collections.abc import AsyncIterator
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

from anthropic.types import Message
from anthropic.types.beta import BetaMessage
import pytest


def make_message(
    content: list[dict[str, Any]],
    *,
    stop_reason: str = "end_turn",
    model: str = "claude-opus-5-5",
    usage: dict[str, Any] | None = None,
) -> Message:
    return Message.model_validate(
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": usage or {"input_tokens": 100, "output_tokens": 50},
        }
    )


def make_beta_message(
    content: list[dict[str, Any]],
    *,
    stop_reason: str = "end_turn",
    model: str = "claude-opus-5-5",
    usage: dict[str, Any] | None = None,
    stop_details: dict[str, Any] | None = None,
) -> BetaMessage:
    return BetaMessage.model_validate(
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "stop_details": stop_details,
            "usage": usage or {"input_tokens": 100, "output_tokens": 50},
        }
    )


class ParsedResponse:
    """Minimal stand-in for anthropic's ParsedMessage."""

    def __init__(self, message: Message, parsed_output: Any) -> None:
        self._message = message
        self.parsed_output = parsed_output

    def __getattr__(self, name: str) -> Any:
        return getattr(self._message, name)


def _events(message: Any) -> list[SimpleNamespace]:
    """The SDK's ``text``/``thinking`` stream events, one per content block."""
    events = []
    for block in message.content:
        if block.type == "text":
            events.append(SimpleNamespace(type="text", text=block.text))
        elif block.type == "thinking":
            events.append(SimpleNamespace(type="thinking", thinking=block.thinking))
    return events


class FakeStream:
    """Stand-in for the SDK's MessageStream (sync and async)."""

    def __init__(self, message: Any) -> None:
        self.message = message

    def __iter__(self) -> Any:
        return iter(_events(self.message))

    def __enter__(self) -> FakeStream:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def get_final_message(self) -> Any:
        return self.message


class AsyncFakeStream:
    def __init__(self, message: Any) -> None:
        self.message = message

    async def __aiter__(self) -> AsyncIterator[SimpleNamespace]:
        for event in _events(self.message):
            yield event

    async def __aenter__(self) -> AsyncFakeStream:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def get_final_message(self) -> Any:
        return self.message


class FakeMessages:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.create_calls: list[dict[str, Any]] = []
        self.parse_calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []
        self.count_calls: list[dict[str, Any]] = []

    def _next(self) -> Any:
        if not self.responses:
            raise AssertionError("No scripted response left")
        return self.responses.pop(0)

    def create(self, **kwargs: Any) -> Any:
        self.create_calls.append(deepcopy(kwargs))
        return self._next()

    def parse(self, **kwargs: Any) -> Any:
        self.parse_calls.append(deepcopy(kwargs))
        return self._next()

    def stream(self, **kwargs: Any) -> FakeStream:
        self.stream_calls.append(deepcopy(kwargs))
        return FakeStream(self._next())

    def count_tokens(self, **kwargs: Any) -> Any:
        self.count_calls.append(deepcopy(kwargs))
        return SimpleNamespace(input_tokens=1234)

    @property
    def all_calls(self) -> list[dict[str, Any]]:
        return self.create_calls + self.parse_calls + self.stream_calls


class FakeClient:
    """``client.messages`` and ``client.beta.messages`` share one script."""

    def __init__(self, responses: list[Any]) -> None:
        self.messages = FakeMessages(responses)
        self.beta = SimpleNamespace(messages=self.messages)


class AsyncFakeMessages(FakeMessages):
    async def create(self, **kwargs: Any) -> Any:  # type: ignore[override]
        return super().create(**kwargs)

    async def parse(self, **kwargs: Any) -> Any:  # type: ignore[override]
        return super().parse(**kwargs)

    def stream(self, **kwargs: Any) -> AsyncFakeStream:  # type: ignore[override]
        self.stream_calls.append(deepcopy(kwargs))
        return AsyncFakeStream(self._next())

    async def count_tokens(self, **kwargs: Any) -> Any:  # type: ignore[override]
        return super().count_tokens(**kwargs)


class AsyncFakeClient:
    def __init__(self, responses: list[Any]) -> None:
        self.messages = AsyncFakeMessages(responses)
        self.beta = SimpleNamespace(messages=self.messages)


@pytest.fixture
def fake_client() -> type[FakeClient]:
    return FakeClient
