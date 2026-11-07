"""``orkexa.testing.fakes.ScriptedClient`` (MUP-ONA B1, plan MP-B1 item j)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from orkexa.testing.fakes import ScriptedCall, ScriptedClient


# MUP-ONA T-FAKES-1 (MP-B1 item j)
def test_routes_answer_per_path_and_record_the_calls() -> None:
    client = ScriptedClient(
        {
            "messages.create": ["first", "second"],
            "beta.messages.batches.retrieve": [SimpleNamespace(id="b-1")],
        },
        attributes={"api_key": "test"},
    )
    assert client.__dict__["api_key"] == "test"  # a plain attribute, not a scripted path
    assert client.messages.create(model="m", max_tokens=1) == "first"
    assert client.beta.messages.batches.retrieve("b-1").id == "b-1"
    assert client.messages.create(model="m") == "second"
    assert [call.path for call in client.calls] == [
        "messages.create",
        "beta.messages.batches.retrieve",
        "messages.create",
    ]
    assert client.calls_to("beta.messages.batches.retrieve") == [
        ScriptedCall("beta.messages.batches.retrieve", ("b-1",))
    ]
    assert client.remaining("messages.create") == 0 and client.remaining() == 0
    with pytest.raises(AssertionError, match=r"no scripted answer left for messages\.create"):
        client.messages.create()
    with pytest.raises(AssertionError, match=r"no scripted answer left for models\.list"):
        client.models.list()


# MUP-ONA T-FAKES-1 (MP-B1 item j)
def test_a_shared_script_answers_in_call_order() -> None:
    client = ScriptedClient(["a", "b"])
    assert client.one() == "a"
    assert client.two.deep() == "b"
    assert client.remaining() == 0


# MUP-ONA T-FAKES-1 (MP-B1 item j)
def test_exceptions_are_raised_and_callables_answer_with_the_call() -> None:
    client = ScriptedClient([TimeoutError("slow"), ValueError, lambda call: call.kwargs["n"] * 2])
    with pytest.raises(TimeoutError, match="slow"):
        client.messages.create()
    with pytest.raises(ValueError):
        client.messages.create()
    assert client.messages.create(n=21) == 42


# MUP-ONA T-FAKES-1 (MP-B1 item j)
def test_arguments_are_recorded_as_they_were_at_the_call() -> None:
    client = ScriptedClient([None])
    messages: list[dict[str, Any]] = [{"role": "user", "content": "hi"}]
    client.messages.create(messages=messages)
    messages.append({"role": "assistant", "content": "later"})
    assert client.calls[0].kwargs == {"messages": [{"role": "user", "content": "hi"}]}


# MUP-ONA T-FAKES-1 (MP-B1 item j)
def test_asynchronous_clients_return_awaitables() -> None:
    client = ScriptedClient({"messages.create": ["done", RuntimeError("boom")]}, asynchronous=True)

    async def run() -> None:
        assert await client.messages.create() == "done"
        pending = client.messages.create()  # recorded now, raised when awaited
        assert len(client.calls) == 2
        with pytest.raises(RuntimeError, match="boom"):
            await pending

    asyncio.run(run())


# MUP-ONA T-FAKES-1 (MP-B1 item j)
def test_private_and_dunder_names_are_not_paths() -> None:
    client = ScriptedClient()
    with pytest.raises(AttributeError):
        _ = client._secret
    with pytest.raises(AttributeError):
        _ = client.messages.__wrapped__
    assert "messages.create" in repr(client.messages.create)
