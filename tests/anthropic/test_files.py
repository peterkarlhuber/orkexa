"""Tests for content-block tool results and the raw-file stores."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Callable
import os
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
from typing import Any
import zlib

import pytest

from orkexa import (
    AsyncClaudeAgent,
    ClaudeAgent,
    EphemeralMemory,
    FileStore,
    LocalFileStore,
    SQLiteFileStore,
    Tool,
    ToolContent,
    ToolExecutionError,
    ToolResultError,
    image,
)
from orkexa.content import describe_content, to_tool_content

from .conftest import AsyncFakeClient, FakeClient, make_message


def _png() -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x00"))
        + chunk(b"IEND", b"")
    )


PDF = b"%PDF-1.4\n% minimal\n"


def _calls(*inputs: dict[str, Any], name: str = "files") -> Any:
    return make_message(
        [{"type": "tool_use", "id": f"toolu_{i}", "name": name, "input": p} for i, p in enumerate(inputs)],
        stop_reason="tool_use",
    )


def _text(text: str) -> Any:
    return make_message([{"type": "text", "text": text}])


def _results(client: FakeClient | AsyncFakeClient, request: int = 1) -> list[dict[str, Any]]:
    return list(client.messages.all_calls[request]["messages"][-1]["content"])


# ---- tools returning content blocks ---------------------------------------------------


@Tool
def chart(title: str) -> ToolContent:
    """Render a chart.

    Args:
        title: Chart title.

    Returns:
        The chart image and its title.
    """
    return [f"Chart: {title}", image(_png(), media_type="image/png")]


def test_tool_content_return_values_pass_through_as_blocks() -> None:
    assert chart.returns_content
    result = chart.execute({"title": "Sales"})
    assert isinstance(result, list)
    assert result[0] == {"type": "text", "text": "Chart: Sales"}
    assert result[1]["type"] == "image"


def test_to_tool_content_validates_blocks() -> None:
    assert to_tool_content(["a", "", {"type": "text", "text": "b"}]) == [
        {"type": "text", "text": "a"},
        {"type": "text", "text": "b"},
    ]
    for invalid in ("text", {"type": "text", "text": "x"}, [], [""], [{"type": "tool_use"}], [3]):
        with pytest.raises(ValueError):
            to_tool_content(invalid)
    search = {"type": "search_result", "source": "s", "title": "t", "content": []}
    assert to_tool_content([search]) == [search]
    with pytest.raises(ValueError, match="mixed"):
        to_tool_content([search, "note"])


def test_invalid_tool_content_is_a_tool_execution_error() -> None:
    @Tool
    def broken() -> ToolContent:
        """Return nothing useful.

        Returns:
            Nothing.
        """
        return []

    with pytest.raises(ToolExecutionError, match="invalid return value"):
        broken.execute({})


def test_describe_content_hides_base64() -> None:
    blocks = [
        {"type": "text", "text": "hello"},
        image(_png(), media_type="image/png"),
        {"type": "document", "title": "Q3", "source": {"type": "file", "file_id": "file_1"}},
        {"type": "search_result", "source": "s", "title": "t", "content": [{"type": "text", "text": "x"}]},
        {"type": "other"},
    ]
    summary = describe_content(blocks)
    assert summary.splitlines() == [
        "hello",
        "[image image/png]",
        "[document 'Q3' file_1]",
        "[search_result 's' 't'] x",
        "[other]",
    ]
    assert describe_content("plain") == "plain"


def test_agent_sends_tool_content_and_hooks_get_a_summary() -> None:
    seen: list[str] = []
    client = FakeClient([_calls({"title": "Sales"}, name="chart"), _text("A dot.")])
    agent: ClaudeAgent[str] = ClaudeAgent(
        client=client,  # type: ignore[arg-type]
        tools=[chart],
        after_tool=lambda call, content, is_error: seen.append(content),
    )
    assert agent.ask("Show the chart") == "A dot."
    (result,) = _results(client)
    assert result["content"][1]["type"] == "image"
    assert seen == ["Chart: Sales\n[image image/png]"]


def test_tool_result_errors_reach_claude_even_with_raise_tool_errors() -> None:
    @Tool
    def lookup(key: str) -> str:
        """Look up a key.

        Args:
            key: The key.

        Returns:
            The value.
        """
        raise ToolResultError(f"unknown key {key}")

    client = FakeClient([_calls({"key": "x"}, name="lookup"), _text("Sorry.")])
    agent: ClaudeAgent[str] = ClaudeAgent(client=client, tools=[lookup])  # type: ignore[arg-type]
    agent.ask("x?")
    (result,) = _results(client)
    assert result == {
        "type": "tool_result",
        "tool_use_id": "toolu_0",
        "content": "unknown key x",
        "is_error": True,
    }


# ---- stores ------------------------------------------------------------------------------


StoreFactory = Callable[[Path], FileStore]
STORES: list[StoreFactory] = [
    lambda tmp: LocalFileStore(tmp / "files"),
    lambda tmp: SQLiteFileStore(tmp / "files.db", namespace="peter"),
]


@pytest.fixture(params=STORES, ids=["local", "sqlite"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> FileStore:
    factory: StoreFactory = request.param
    return factory(tmp_path)


def test_add_info_list_get_and_delete(store: FileStore, tmp_path: Path) -> None:
    path = tmp_path / "dot.png"
    path.write_bytes(_png())
    entry = store.add(path, description="a red dot", metadata={"owner": "peter"})
    assert entry.name == "dot.png"
    assert entry.media_type == "image/png"
    assert entry.kind == "image"
    assert entry.size == len(_png())
    assert len(entry.id) == 16
    assert store.info(entry.id) == entry
    assert store.get_bytes(entry.id) == _png()
    assert entry.id in store
    assert len(store) == 1

    pdf = store.add(PDF, name="report.pdf", description="Quarterly report")
    assert pdf.kind == "document"
    assert [e.id for e in store.list()] == [pdf.id, entry.id]
    assert [e.id for e in store.list("quarter")] == [pdf.id]
    assert [e.id for e in store.list("DOT")] == [entry.id]

    store.delete(entry.id)
    assert entry.id not in store
    with pytest.raises(KeyError):
        store.info(entry.id)
    with pytest.raises(KeyError):
        store.get_bytes(entry.id)
    with pytest.raises(KeyError):
        store.delete(entry.id)
    with pytest.raises(KeyError):
        store.info("../etc/passwd")
    assert "../x" not in store


def test_same_bytes_update_the_existing_entry(store: FileStore) -> None:
    first = store.add(b"hello", name="a.txt", description="first", metadata={"a": "1"})
    second = store.add(b"hello", name="b.txt", metadata={"b": "2"})
    assert second.id == first.id
    assert second.name == "b.txt"
    assert second.description == "first"
    assert second.metadata == {"a": "1", "b": "2"}
    assert len(store) == 1
    assert store.get_bytes(first.id) == b"hello"


def test_add_rejects_unsupported_input(store: FileStore) -> None:
    with pytest.raises(ValueError, match="name is required"):
        store.add(b"x")
    with pytest.raises(ValueError, match="media type"):
        store.add(b"x", name="archive")
    with pytest.raises(ValueError, match="unsupported"):
        store.add(b"PK", name="letter.docx")
    with pytest.raises(ValueError, match="UTF-8"):
        store.add(b"\xff\xfe", name="notes.txt")
    store.max_image_bytes = 10
    with pytest.raises(ValueError, match="exceed"):
        store.add(_png(), name="dot.png")


def test_text_types_are_guessed() -> None:
    store = SQLiteFileStore(":memory:")
    assert store.add(b"# Notes", name="notes.md").media_type == "text/markdown"
    assert store.add(b"a,b", name="data.csv").media_type == "text/csv"
    assert store.add(b"x", name="x.bin", media_type="text/plain").media_type == "text/plain"


def test_blocks_for_images_pdfs_and_text(store: FileStore) -> None:
    png = store.add(_png(), name="dot.png")
    pdf = store.add(PDF, name="report.pdf")
    notes = store.add(b"# Notes", name="notes.md")
    image_block = store.block(png.id)
    assert image_block["source"]["media_type"] == "image/png"
    assert base64.b64decode(image_block["source"]["data"]) == _png()
    pdf_block = store.block(pdf.id, citations=True)
    assert pdf_block["source"]["media_type"] == "application/pdf"
    assert pdf_block["title"] == "report.pdf"
    assert pdf_block["citations"] == {"enabled": True}
    assert store.block(notes.id)["source"] == {"type": "text", "media_type": "text/plain", "data": "# Notes"}


def test_files_api_uploads_once_and_then_references_the_file_id(tmp_path: Path) -> None:
    uploads: list[tuple[str, bytes, str]] = []

    def upload(*, file: tuple[str, bytes, str]) -> Any:
        uploads.append(file)
        return SimpleNamespace(id=f"file_{len(uploads)}")

    client = SimpleNamespace(files=SimpleNamespace(upload=upload))
    store = LocalFileStore(tmp_path / "files", files_api=client)
    pdf = store.add(PDF, name="report.pdf")
    assert store.block(pdf.id) == {
        "type": "document",
        "source": {"type": "file", "file_id": "file_1"},
        "title": "report.pdf",
    }
    assert store.block(pdf.id)["source"]["file_id"] == "file_1"
    assert uploads == [("report.pdf", PDF, "application/pdf")]
    assert store.info(pdf.id).api_file_id == "file_1"
    png = store.add(_png(), name="dot.png")
    assert store.block(png.id) == {"type": "image", "source": {"type": "file", "file_id": "file_2"}}


def test_note_to_memory_writes_a_reference() -> None:
    memory = EphemeralMemory()
    store = SQLiteFileStore(":memory:")
    entry = store.add(PDF, name="report.pdf", description="Q3 report", note_to=memory)
    note = memory.files()[f"/memories/files/{entry.id}.md"]
    assert "report.pdf" in note
    assert entry.id in note
    assert "Q3 report" in note


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")
def test_local_store_permissions(tmp_path: Path) -> None:
    store = LocalFileStore(tmp_path / "files")
    entry = store.add(b"secret", name="s.txt")
    assert (tmp_path / "files").stat().st_mode & 0o777 == 0o700
    assert (tmp_path / "files" / "blobs" / entry.id).stat().st_mode & 0o777 == 0o600
    assert (tmp_path / "files" / "catalog.json").stat().st_mode & 0o777 == 0o600
    assert not [name for name in os.listdir(tmp_path / "files") if name.startswith(".tmp-")]


def test_stores_persist_across_instances(tmp_path: Path) -> None:
    LocalFileStore(tmp_path / "files").add(b"hi", name="a.txt")
    assert [e.name for e in LocalFileStore(tmp_path / "files").list()] == ["a.txt"]
    with SQLiteFileStore(tmp_path / "db.sqlite", namespace="a") as first:
        first.add(b"hi", name="a.txt")
    with SQLiteFileStore(tmp_path / "db.sqlite", namespace="a") as again:
        assert len(again) == 1
    with SQLiteFileStore(tmp_path / "db.sqlite", namespace="b") as other:
        assert len(other) == 0
    with pytest.raises(ValueError, match="namespace"):
        SQLiteFileStore(":memory:", namespace="")
    with pytest.raises(ValueError, match="positive"):
        SQLiteFileStore(":memory:", max_image_bytes=0)


# ---- the files tool ------------------------------------------------------------------------


def test_files_tool_lists_and_loads(store: FileStore) -> None:
    tool = store.tool()
    assert tool.name == "files"
    assert tool.schema()["input_schema"]["required"] == ["command"]
    assert tool.execute({"command": "list"}) == [{"type": "text", "text": "No files are stored."}]
    entry = store.add(PDF, name="report.pdf", description="Q3")
    (listing,) = tool.execute({"command": "list"})
    assert f"{entry.id} | report.pdf | application/pdf" in listing["text"]
    assert tool.execute({"command": "list", "query": "nothing"})[0]["text"] == "No stored files match."
    header, block = tool.execute({"command": "load", "file_id": entry.id})
    assert header["text"].startswith(f"File {entry.id}")
    assert block["type"] == "document"
    with pytest.raises(ToolResultError, match="invalid file id"):
        tool.execute({"command": "load"})
    with pytest.raises(ToolResultError, match="no file with id"):
        tool.execute({"command": "load", "file_id": "0" * 16})


def test_agent_with_files_runs_the_tool() -> None:
    store = SQLiteFileStore(":memory:")
    entry = store.add(_png(), name="dot.png")
    client = FakeClient(
        [
            _calls({"command": "list"}, {"command": "load", "file_id": entry.id}, {"command": "load"}),
            _text("It is a dot."),
        ]
    )
    agent: ClaudeAgent[str] = ClaudeAgent(client=client, files=store)  # type: ignore[arg-type]
    assert agent.ask("What is in dot.png?") == "It is a dot."
    assert [tool["name"] for tool in client.messages.create_calls[0]["tools"]] == ["files"]
    listing, loaded, failed = _results(client)
    assert entry.id in listing["content"][0]["text"]
    assert loaded["content"][1]["type"] == "image"
    assert failed["is_error"] is True
    assert agent.files is store
    with pytest.raises(TypeError, match="FileStore"):
        ClaudeAgent(client=client, files="./files")  # type: ignore[arg-type]


def test_async_agent_with_files() -> None:
    store = SQLiteFileStore(":memory:")
    entry = store.add(PDF, name="report.pdf")
    client = AsyncFakeClient([_calls({"command": "load", "file_id": entry.id}), _text("Done.")])
    agent: AsyncClaudeAgent[str] = AsyncClaudeAgent(client=client, files=store)  # type: ignore[arg-type]
    assert asyncio.run(agent.ask("Read the report")) == "Done."
    (loaded,) = _results(client)
    assert loaded["content"][1]["source"]["media_type"] == "application/pdf"
