"""Tests for the memory tool backends and their integration into the agents."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from contextlib import closing
import os
from pathlib import Path
import sqlite3
import sys
import threading
from typing import Any

import pytest

from orkexa import (
    AsyncClaudeAgent,
    ClaudeAgent,
    EphemeralMemory,
    FileMemory,
    MemoryBackend,
    MemoryToolError,
    SQLiteMemory,
    Tool,
    ToolCall,
    ToolExecutionError,
)

from .conftest import AsyncFakeClient, FakeClient, make_message

LISTING = (
    "Here're the files and directories up to 2 levels deep in /memories, "
    "excluding hidden items and node_modules:"
)


@pytest.fixture(params=["ephemeral", "file", "sqlite"])
def memory(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[MemoryBackend]:
    backend: MemoryBackend
    if request.param == "ephemeral":
        backend = EphemeralMemory()
    elif request.param == "file":
        backend = FileMemory(tmp_path / "memory")
    else:
        backend = SQLiteMemory(tmp_path / "memory.db")
    yield backend
    if isinstance(backend, SQLiteMemory):
        backend.close()


def run(memory: MemoryBackend, **command: Any) -> str:
    return memory.execute(command)


def fails(memory: MemoryBackend, message: str, **command: Any) -> None:
    with pytest.raises(MemoryToolError) as info:
        memory.execute(command)
    assert str(info.value) == message


# ---- commands, identical for every backend ---------------------------------------


def test_view_of_an_empty_memory_lists_only_the_root(memory: MemoryBackend) -> None:
    assert run(memory, command="view", path="/memories") == f"{LISTING}\n0B\t/memories"


def test_create_and_view_a_file(memory: MemoryBackend) -> None:
    result = run(memory, command="create", path="/memories/notes.md", file_text="Hello\nWorld\n")
    assert result == "File created successfully at: /memories/notes.md"
    assert run(memory, command="view", path="/memories/notes.md") == (
        "Here's the content of /memories/notes.md with line numbers:\n     1\tHello\n     2\tWorld"
    )
    assert run(memory, command="view", path="/memories/notes.md", view_range=[2, -1]).endswith(
        "with line numbers:\n     2\tWorld"
    )


def test_create_overwrites_and_rejects_directories(memory: MemoryBackend) -> None:
    run(memory, command="create", path="/memories/a/b.md", file_text="one")
    run(memory, command="create", path="/memories/a/b.md", file_text="two")
    assert run(memory, command="view", path="/memories/a/b.md").endswith("1\ttwo")
    fails(memory, "Error: /memories/a is a directory", command="create", path="/memories/a", file_text="")
    fails(
        memory,
        "Error: /memories/a/b.md is a file, not a directory",
        command="create",
        path="/memories/a/b.md/c.md",
        file_text="",
    )


def test_directory_listing_is_two_levels_deep_with_sizes(memory: MemoryBackend) -> None:
    run(memory, command="create", path="/memories/b.md", file_text="x" * 2048)
    run(memory, command="create", path="/memories/a/one.md", file_text="12345")
    run(memory, command="create", path="/memories/a/deep/two.md", file_text="123")
    run(memory, command="create", path="/memories/.hidden", file_text="secret")
    run(memory, command="create", path="/memories/node_modules/x.js", file_text="x")
    assert run(memory, command="view", path="/memories") == "\n".join(
        [
            LISTING,
            "2.0K\t/memories",
            "8B\t/memories/a/",
            "3B\t/memories/a/deep/",
            "5B\t/memories/a/one.md",
            "2.0K\t/memories/b.md",
        ]
    )
    sub = run(memory, command="view", path="/memories/a/")
    assert sub.splitlines()[0].endswith("deep in /memories/a, excluding hidden items and node_modules:")
    assert "3B\t/memories/a/deep/two.md" in sub


def test_view_errors(memory: MemoryBackend) -> None:
    fails(
        memory,
        "The path /memories/missing.md does not exist. Please provide a valid path.",
        command="view",
        path="/memories/missing.md",
    )
    run(memory, command="create", path="/memories/n.md", file_text="a\nb\n")
    fails(
        memory,
        "Error: Invalid `view_range` parameter: [2, 5]. "
        "It should be within the range of lines of the file: [1, 2]",
        command="view",
        path="/memories/n.md",
        view_range=[2, 5],
    )


def test_str_replace(memory: MemoryBackend) -> None:
    run(memory, command="create", path="/memories/p.md", file_text="color: blue\nsize: M\n")
    result = run(memory, command="str_replace", path="/memories/p.md", old_str="blue", new_str="green")
    assert result.startswith("The memory file has been edited.")
    assert "     1\tcolor: green" in result
    run(memory, command="str_replace", path="/memories/p.md", old_str="size: M\n")
    assert run(memory, command="view", path="/memories/p.md").endswith("     1\tcolor: green")


def test_str_replace_errors(memory: MemoryBackend) -> None:
    run(memory, command="create", path="/memories/p.md", file_text="a\nx\nx\n")
    fails(
        memory,
        "No replacement was performed, old_str `zzz` did not appear verbatim in /memories/p.md.",
        command="str_replace",
        path="/memories/p.md",
        old_str="zzz",
        new_str="",
    )
    fails(
        memory,
        "No replacement was performed. Multiple occurrences of old_str `x` in lines: 2, 3. "
        "Please ensure it is unique",
        command="str_replace",
        path="/memories/p.md",
        old_str="x",
        new_str="y",
    )
    fails(
        memory,
        "Error: The path /memories/q.md does not exist. Please provide a valid path.",
        command="str_replace",
        path="/memories/q.md",
        old_str="a",
        new_str="b",
    )


def test_insert(memory: MemoryBackend) -> None:
    run(memory, command="create", path="/memories/todo.md", file_text="one\nthree\n")
    assert (
        run(memory, command="insert", path="/memories/todo.md", insert_line=1, insert_text="two\n")
        == "The file /memories/todo.md has been edited."
    )
    run(memory, command="insert", path="/memories/todo.md", insert_line=0, insert_text="zero")
    run(memory, command="insert", path="/memories/todo.md", insert_line=4, insert_text="four")
    assert run(memory, command="view", path="/memories/todo.md").splitlines()[1:] == [
        "     1\tzero",
        "     2\tone",
        "     3\ttwo",
        "     4\tthree",
        "     5\tfour",
    ]
    fails(
        memory,
        "Error: Invalid `insert_line` parameter: 9. "
        "It should be within the range of lines of the file: [0, 5]",
        command="insert",
        path="/memories/todo.md",
        insert_line=9,
        insert_text="x",
    )
    fails(
        memory,
        "Error: The path /memories/nope.md does not exist",
        command="insert",
        path="/memories/nope.md",
        insert_line=0,
        insert_text="x",
    )


def test_delete(memory: MemoryBackend) -> None:
    run(memory, command="create", path="/memories/dir/a.md", file_text="a")
    run(memory, command="create", path="/memories/dir/sub/b.md", file_text="b")
    run(memory, command="create", path="/memories/dirx.md", file_text="keep")
    assert run(memory, command="delete", path="/memories/dir") == "Successfully deleted /memories/dir"
    assert run(memory, command="view", path="/memories").splitlines()[2:] == ["4B\t/memories/dirx.md"]
    fails(memory, "Error: The path /memories/dir does not exist", command="delete", path="/memories/dir")
    fails(
        memory,
        "Error: cannot delete the /memories directory itself",
        command="delete",
        path="/memories",
    )


def test_rename(memory: MemoryBackend) -> None:
    run(memory, command="create", path="/memories/draft.md", file_text="text")
    run(memory, command="create", path="/memories/dir/a.md", file_text="a")
    run(memory, command="create", path="/memories/dirx.md", file_text="x")
    assert (
        run(memory, command="rename", old_path="/memories/draft.md", new_path="/memories/final/v1.md")
        == "Successfully renamed /memories/draft.md to /memories/final/v1.md"
    )
    run(memory, command="rename", old_path="/memories/dir", new_path="/memories/archive")
    listing = run(memory, command="view", path="/memories")
    assert "/memories/archive/a.md" in listing
    assert "/memories/dirx.md" in listing
    assert "/memories/dir/" not in listing
    fails(
        memory,
        "Error: The destination /memories/dirx.md already exists",
        command="rename",
        old_path="/memories/final/v1.md",
        new_path="/memories/dirx.md",
    )
    fails(
        memory,
        "Error: The path /memories/gone.md does not exist",
        command="rename",
        old_path="/memories/gone.md",
        new_path="/memories/new.md",
    )
    fails(
        memory,
        "Error: cannot move /memories/archive into itself",
        command="rename",
        old_path="/memories/archive",
        new_path="/memories/archive/inner",
    )
    fails(
        memory,
        "Error: cannot rename the /memories directory itself",
        command="rename",
        old_path="/memories",
        new_path="/memories/x",
    )


@pytest.mark.parametrize(
    "path",
    [
        "/memories/../secrets.env",
        "/memories/a/../../x",
        "/memories/%2e%2e/x",
        "/memories/%252e%252e/x",
        "/memories/a\\..\\b",
        "/memories/a//b",
        "/memories/./a",
        "/memories/a\x00b",
        "/etc/passwd",
        "/memoriesx/a",
        "memories/a",
    ],
)
def test_path_traversal_is_rejected(memory: MemoryBackend, path: str) -> None:
    with pytest.raises(MemoryToolError):
        run(memory, command="create", path=path, file_text="x")


def test_invalid_commands(memory: MemoryBackend) -> None:
    with pytest.raises(MemoryToolError, match="invalid memory command"):
        run(memory, command="format_disk", path="/memories")
    with pytest.raises(MemoryToolError, match="file_text"):
        run(memory, command="create", path="/memories/a.md")
    with pytest.raises(MemoryToolError, match="must not be empty"):
        run(memory, command="str_replace", path="/memories/a.md", old_str="", new_str="x")


def test_size_limits(tmp_path: Path) -> None:
    memory = EphemeralMemory(max_file_chars=10, max_view_chars=60)
    with pytest.raises(MemoryToolError, match="the limit is 10"):
        memory.create("/memories/big.md", "x" * 11)
    memory.max_file_chars = 1000
    memory.create("/memories/long.md", "\n".join(f"line {i}" for i in range(50)))
    view = memory.view("/memories/long.md")
    assert view.endswith(
        "[Output truncated at 60 characters (50 lines in total); use view_range to read the rest.]"
    )
    assert len(view.splitlines()) < 10
    with pytest.raises(ValueError):
        EphemeralMemory(max_file_chars=0)


def test_methods_can_seed_and_inspect_a_memory() -> None:
    memory = EphemeralMemory({"/memories/prefs.md": "Language: German\n"})
    memory.insert("/memories/prefs.md", 1, "Tone: concise")
    assert memory.files() == {"/memories/prefs.md": "Language: German\nTone: concise\n"}
    memory.clear()
    assert memory.files() == {}


def test_concurrent_commands_are_serialized(memory: MemoryBackend) -> None:
    memory.create("/memories/log.md", "")

    def append(index: int) -> None:
        memory.insert("/memories/log.md", 0, f"entry {index}")

    threads = [threading.Thread(target=append, args=(i,)) for i in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(memory.view("/memories/log.md").splitlines()) == 21


# ---- persistence and isolation ----------------------------------------------------


def test_file_memory_persists_with_private_permissions(tmp_path: Path) -> None:
    FileMemory(tmp_path / "m").create("/memories/notes/a.md", "Ä persisted\n")
    assert (tmp_path / "m" / "notes" / "a.md").read_text(encoding="utf-8") == "Ä persisted\n"
    assert FileMemory(tmp_path / "m").view("/memories/notes/a.md").endswith("1\tÄ persisted")
    if sys.platform != "win32":
        assert (tmp_path / "m" / "notes" / "a.md").stat().st_mode & 0o777 == 0o600
        assert (tmp_path / "m" / "notes").stat().st_mode & 0o777 == 0o700
    assert not [p for p in (tmp_path / "m").rglob(".tmp-*")]


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")
def test_file_memory_rejects_symlinks_that_leave_the_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret")
    root = tmp_path / "m"
    memory = FileMemory(root)
    os.symlink(outside, root / "link")
    os.symlink(outside / "secret.txt", root / "file-link")
    for path in ("/memories/link/secret.txt", "/memories/file-link"):
        with pytest.raises(MemoryToolError, match="outside the memory directory"):
            memory.view(path)
    with pytest.raises(MemoryToolError, match="outside the memory directory"):
        memory.create("/memories/link/new.txt", "x")
    assert not (outside / "new.txt").exists()
    assert "secret" not in memory.view("/memories")


def test_file_memory_rejects_binary_files(tmp_path: Path) -> None:
    memory = FileMemory(tmp_path)
    (tmp_path / "image.png").write_bytes(b"\x89PNG\xff\xfe")
    with pytest.raises(MemoryToolError, match="not a UTF-8 text file"):
        memory.view("/memories/image.png")


def test_sqlite_memory_persists_and_separates_namespaces(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    with SQLiteMemory(database, namespace="peter") as memory:
        memory.create("/memories/prefs.md", "German")
    with SQLiteMemory(database, namespace="anna") as other:
        assert other.view("/memories") == f"{LISTING}\n0B\t/memories"
        other.create("/memories/prefs.md", "English")
    with SQLiteMemory(database, namespace="peter") as memory:
        assert memory.view("/memories/prefs.md").endswith("1\tGerman")
        memory.delete("/memories/prefs.md")
    with closing(sqlite3.connect(database)) as connection:
        rows = connection.execute("SELECT namespace, path FROM memory_files").fetchall()
    assert rows == [("anna", "prefs.md")]
    with pytest.raises(ValueError):
        SQLiteMemory(database, namespace="")


def test_sqlite_memory_rolls_back_a_failed_command(tmp_path: Path) -> None:
    class Failing(SQLiteMemory):
        def _write(self, key: str, text: str) -> None:
            super()._write(key, text)
            raise OSError("disk full")

    with Failing(tmp_path / "m.db") as memory:
        with pytest.raises(OSError):
            memory.create("/memories/a.md", "x")
        assert memory._read("a.md") is None


def test_sqlite_memory_paths_are_not_like_patterns(tmp_path: Path) -> None:
    with SQLiteMemory(":memory:") as memory:
        memory.create("/memories/a_b/x.md", "1")
        memory.create("/memories/axb/y.md", "2")
        memory.delete("/memories/a_b")
        assert "/memories/axb/" in memory.view("/memories")


# ---- agent integration ---------------------------------------------------------------


def _text(text: str) -> Any:
    return make_message([{"type": "text", "text": text}])


def _memory_calls(*inputs: dict[str, Any]) -> Any:
    return make_message(
        [
            {"type": "tool_use", "id": f"toolu_{i}", "name": "memory", "input": payload}
            for i, payload in enumerate(inputs)
        ],
        stop_reason="tool_use",
    )


def _results(client: FakeClient | AsyncFakeClient, request: int = 1) -> list[dict[str, Any]]:
    return list(client.messages.all_calls[request]["messages"][-1]["content"])


def test_agent_sends_the_tool_and_runs_memory_commands() -> None:
    memory = EphemeralMemory()
    client = FakeClient(
        [
            _memory_calls(
                {"command": "create", "path": "/memories/prefs.md", "file_text": "German\n"},
                {"command": "view", "path": "/memories/missing.md"},
            ),
            _text("Noted."),
        ]
    )
    agent: ClaudeAgent[str] = ClaudeAgent(client=client, memory=memory)  # type: ignore[arg-type]
    assert agent.ask("Remember: I prefer German.") == "Noted."
    assert client.messages.create_calls[0]["tools"] == [{"type": "memory_20250818", "name": "memory"}]
    created, missing = _results(client)
    assert created == {
        "type": "tool_result",
        "tool_use_id": "toolu_0",
        "content": "File created successfully at: /memories/prefs.md",
    }
    assert missing["is_error"] is True
    assert "does not exist" in missing["content"]
    assert memory.files() == {"/memories/prefs.md": "German\n"}
    assert "memory" in repr(agent)


def test_memory_works_next_to_local_tools_and_count_tokens() -> None:
    @Tool
    def echo(text: str) -> str:
        """Echo a text.

        Args:
            text: The text.

        Returns:
            The same text.
        """
        return text

    client = FakeClient([])
    agent: ClaudeAgent[str] = ClaudeAgent(client=client, tools=[echo], memory=EphemeralMemory())  # type: ignore[arg-type]
    agent.count_tokens("Hi")
    names = [tool["name"] for tool in client.messages.count_calls[0]["tools"]]
    assert names == ["echo", "memory"]

    @Tool(name="memory")
    def fake_memory(text: str) -> str:
        """Pretend to be memory.

        Args:
            text: Anything.

        Returns:
            Anything.
        """
        return text

    with pytest.raises(ValueError, match="Duplicate tool name"):
        agent.add_tool(fake_memory)
    with pytest.raises(TypeError, match="MemoryBackend"):
        ClaudeAgent(client=client, memory={"type": "memory_20250818"})  # type: ignore[arg-type]


def test_hooks_see_and_can_deny_memory_calls() -> None:
    memory = EphemeralMemory({"/memories/keep.md": "important"})
    seen: list[tuple[str, str, bool]] = []

    def before(call: ToolCall) -> str | None:
        return "deleting is not allowed" if call.input.get("command") == "delete" else None

    client = FakeClient(
        [
            _memory_calls(
                {"command": "delete", "path": "/memories/keep.md"},
                {"command": "view", "path": "/memories/keep.md"},
            ),
            _text("ok"),
        ]
    )
    agent: ClaudeAgent[str] = ClaudeAgent(
        client=client,  # type: ignore[arg-type]
        memory=memory,
        before_tool=before,
        after_tool=lambda call, content, is_error: seen.append((call.name, content, is_error)),
    )
    agent.ask("Clean up")
    denied, viewed = _results(client)
    assert denied["content"] == "Tool call denied: deleting is not allowed"
    assert denied["is_error"] is True
    assert "important" in viewed["content"]
    assert memory.files() == {"/memories/keep.md": "important"}
    assert [entry[2] for entry in seen] == [True, False]


def test_unexpected_backend_errors_follow_raise_tool_errors() -> None:
    class Broken(EphemeralMemory):
        def _write(self, key: str, text: str) -> None:
            raise OSError("disk full")

    create = {"command": "create", "path": "/memories/a.md", "file_text": "x"}
    client = FakeClient([_memory_calls(create)])
    agent: ClaudeAgent[str] = ClaudeAgent(client=client, memory=Broken())  # type: ignore[arg-type]
    with pytest.raises(ToolExecutionError, match="memory: OSError: disk full"):
        agent.ask("Remember")
    assert agent.messages == []

    client = FakeClient([_memory_calls(create), _text("Sorry.")])
    agent = ClaudeAgent(client=client, memory=Broken(), raise_tool_errors=False)  # type: ignore[arg-type]
    assert agent.ask("Remember") == "Sorry."
    assert _results(client)[0]["is_error"] is True


def test_memory_can_be_shared_by_agents() -> None:
    memory = EphemeralMemory()
    writer = FakeClient(
        [_memory_calls({"command": "create", "path": "/memories/f.md", "file_text": "fact"}), _text("ok")]
    )
    reader = FakeClient([_memory_calls({"command": "view", "path": "/memories/f.md"}), _text("fact")])
    ClaudeAgent(client=writer, memory=memory).ask("Remember")  # type: ignore[arg-type]
    ClaudeAgent(client=reader, memory=memory).ask("Recall")  # type: ignore[arg-type]
    assert _results(reader)[0]["content"].endswith("1\tfact")


def test_async_agent_runs_memory_commands(tmp_path: Path) -> None:
    memory = FileMemory(tmp_path)
    threads: list[str] = []

    def before(call: ToolCall) -> None:
        threads.append(threading.current_thread().name)

    client = AsyncFakeClient(
        [
            _memory_calls(
                {"command": "create", "path": "/memories/a.md", "file_text": "a"},
                {"command": "create", "path": "/memories/b.md", "file_text": "b"},
            ),
            _text("done"),
        ]
    )
    agent: AsyncClaudeAgent[str] = AsyncClaudeAgent(
        client=client,  # type: ignore[arg-type]
        memory=memory,
        max_parallel_tools=2,
        before_tool=before,
    )
    assert asyncio.run(agent.ask("Remember a and b")) == "done"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.md", "b.md"]
    assert [r["content"] for r in _results(client)] == [
        "File created successfully at: /memories/a.md",
        "File created successfully at: /memories/b.md",
    ]
    assert threads


def test_custom_backend_needs_only_storage_methods() -> None:
    class DictBackend(MemoryBackend):
        def __init__(self) -> None:
            super().__init__()
            self.data: dict[str, str] = {}

        def _read(self, key: str) -> str | None:
            return self.data.get(key)

        def _write(self, key: str, text: str) -> None:
            self.data[key] = text

        def _files(self, prefix: str) -> dict[str, int]:
            start = f"{prefix}/" if prefix else ""
            return {k: len(v) for k, v in self.data.items() if k.startswith(start)}

        def _delete(self, key: str) -> None:
            for k in [key, *self._files(key)]:
                self.data.pop(k, None)

        def _rename(self, old: str, new: str) -> None:
            self.data[new] = self.data.pop(old)

    backend = DictBackend()
    backend.create("/memories/x.md", "1")
    backend.rename("/memories/x.md", "/memories/y.md")
    assert backend.data == {"y.md": "1"}
    factory: Callable[[], MemoryBackend] = DictBackend
    assert isinstance(factory(), MemoryBackend)
