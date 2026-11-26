"""Tools declare where their content comes from; the agent reports it on every tool call (N-08)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from orkexa import Agent, AsyncAgent, EphemeralMemory, LocalFileStore, Tool, ToolCall
from orkexa.errors import ToolDefinitionError
from orkexa.files import FILES_CONTENT_SOURCE
from orkexa.memory import MEMORY_TOOL_NAME
from orkexa.providers.fake import FakeModel, text, tool_calls
from orkexa.retrieval import INDEX_CONTENT_SOURCE

from .helpers import add


@Tool(content_source="folder")
def read_note(name: str) -> str:
    """Read a note of a granted folder.

    Args:
        name: The note's name.

    Returns:
        The note's text.
    """
    return f"note {name}"


def _seen(script: list[Any], **kwargs: Any) -> list[tuple[str, str | None]]:
    seen: list[tuple[str, str | None]] = []
    model = FakeModel(script)
    agent: Agent[Any] = Agent(
        model="fake:m",
        client=model,
        after_tool=lambda call, _content, _error: seen.append((call.name, call.content_source)),
        **kwargs,
    )
    agent.ask("go")
    return seen


def test_a_tool_declares_its_content_source_and_keeps_it_when_decorated_with_arguments() -> None:
    assert read_note.content_source == "folder" and read_note.name == "read_note"
    assert add.content_source is None
    assert Tool(read_note.function, name="other", content_source="files").content_source == "files"


@pytest.mark.parametrize("bad", ["", "Folder", "with space", "x" * 33, "1st", 3])
def test_an_invalid_content_source_is_refused(bad: Any) -> None:
    with pytest.raises(ToolDefinitionError, match="content source"):
        Tool(read_note.function, content_source=bad)


def test_every_call_reports_the_declared_source(tmp_path: Path) -> None:
    store = LocalFileStore(tmp_path / "files")
    seen = _seen(
        [
            tool_calls(("read_note", {"name": "a"}), ("add", {"a": 1, "b": 2}), ("nope", {})),
            tool_calls(
                ("files", {"command": "list"}), (MEMORY_TOOL_NAME, {"command": "view", "path": "/memories"})
            ),
            text("done"),
        ],
        tools=[read_note, add],
        files=store,
        memory=EphemeralMemory(),
        raise_tool_errors=False,
    )
    assert seen == [
        ("read_note", "folder"),
        ("add", None),
        ("nope", None),  # an unknown tool declares nothing
        ("files", FILES_CONTENT_SOURCE),
        (MEMORY_TOOL_NAME, "memory"),
    ]


def test_a_memory_whose_content_has_no_source_reports_none() -> None:
    memory = EphemeralMemory()
    memory.content_source = None
    seen = _seen(
        [tool_calls((MEMORY_TOOL_NAME, {"command": "view", "path": "/memories"})), text("done")],
        memory=memory,
    )
    assert seen == [(MEMORY_TOOL_NAME, None)]


def test_the_index_tool_declares_its_source() -> None:
    from orkexa import Index

    class Empty(Index):
        def add_chunks(self, chunks: Any) -> list[int]:  # pragma: no cover - not called
            return []

        def search(self, query: str, k: int = 5) -> list[Any]:  # pragma: no cover - not called
            return []

        def delete(self, source: str) -> int:  # pragma: no cover - not called
            return 0

    assert Empty().tool().content_source == INDEX_CONTENT_SOURCE == "index"


def test_the_async_agent_reports_the_source_too() -> None:
    seen: list[ToolCall] = []

    async def main() -> None:
        agent: AsyncAgent[Any] = AsyncAgent(
            model="fake:m",
            client=FakeModel([tool_calls(("read_note", {"name": "a"})), text("done")]),
            tools=[read_note],
            before_tool=lambda call: seen.append(call),
        )
        await agent.ask("go")

    asyncio.run(main())
    assert [(call.name, call.content_source) for call in seen] == [("read_note", "folder")]
