"""Persisted data is readable only by its owner: files ``0600``, new directories ``0700``.

Runs under a permissive umask (``022``), so every check proves that Orkexa sets the
modes itself instead of relying on the process environment. POSIX only: on Windows
access is governed by ACLs, not by mode bits.
"""

from __future__ import annotations

from collections.abc import Iterator
import os
from pathlib import Path
import stat
import sys

import pytest

from orkexa import (
    Agent,
    ClaudeAgent,
    FileMemory,
    LocalFileStore,
    SQLiteFileStore,
    SQLiteIndex,
    SQLiteMemory,
    _storage,
)
from orkexa.dynamic import AgentRegistry, AgentSpec, ToolRegistry
from orkexa.providers.fake import FakeModel, text

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")


@pytest.fixture(autouse=True)
def _permissive_umask() -> Iterator[None]:
    previous = os.umask(0o022)
    try:
        yield
    finally:
        os.umask(previous)


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _no_temporary_files(directory: Path) -> None:
    assert not [p.name for p in directory.iterdir() if p.name.endswith(".tmp")]


# ---- histories -------------------------------------------------------------------------------


def _history_agents() -> list[Agent[str] | ClaudeAgent[str]]:
    agent: Agent[str] = Agent(model="fake:m", client=FakeModel([text("hello")]))
    agent.ask("hi")
    claude: ClaudeAgent[str] = ClaudeAgent(client=object(), messages=[{"role": "user", "content": "hi"}])
    return [agent, claude]


@pytest.mark.parametrize("index", [0, 1], ids=["Agent", "ClaudeAgent"])
def test_saved_histories_are_private_new_and_overwritten(tmp_path: Path, index: int) -> None:
    agent = _history_agents()[index]
    new = tmp_path / "new.json"
    agent.save_history(new)
    assert _mode(new) == 0o600

    existing = tmp_path / "existing.json"
    existing.write_text("old")
    existing.chmod(0o644)
    agent.save_history(existing)
    assert _mode(existing) == 0o600 and existing.read_text() != "old"
    _no_temporary_files(tmp_path)


def test_saved_history_replaces_a_symlink_instead_of_following_it(tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("untouched")
    link = tmp_path / "history.json"
    link.symlink_to(outside)
    _history_agents()[0].save_history(link)
    assert outside.read_text() == "untouched"
    assert not link.is_symlink() and _mode(link) == 0o600


def test_failed_history_write_keeps_the_old_file_and_no_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "history.json"
    path.write_text("previous")

    def fail(source: str, target: str) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(_storage.os, "replace", fail)
    with pytest.raises(OSError, match="disk full"):
        _history_agents()[0].save_history(path)
    assert path.read_text() == "previous"
    _no_temporary_files(tmp_path)


# ---- SQLite stores ---------------------------------------------------------------------------


def _use_sqlite_memory(database: Path) -> None:
    memory = SQLiteMemory(database)
    memory.create("/memories/a.md", "note")
    memory.close()


def _use_sqlite_files(database: Path) -> None:
    store = SQLiteFileStore(database)
    store.add(b"content", name="a.txt", media_type="text/plain")
    store.close()


def _use_sqlite_index(database: Path) -> None:
    index = SQLiteIndex(database)
    index.add("some text to index", source="a.md")
    index.close()


STORES = {
    "SQLiteMemory": _use_sqlite_memory,
    "SQLiteFileStore": _use_sqlite_files,
    "SQLiteIndex": _use_sqlite_index,
}


@pytest.mark.parametrize("store", sorted(STORES))
def test_new_sqlite_databases_and_their_journals_are_private(
    tmp_path: Path, store: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sqlite3

    database = tmp_path / "store.db"
    seen: dict[str, int] = {}
    connect = sqlite3.connect

    def watching_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        connection = connect(*args, **kwargs)  # type: ignore[arg-type]
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    monkeypatch.setattr(sqlite3, "connect", watching_connect)
    STORES[store](database)
    # Reopen in WAL mode and keep the connection open, so the -wal and -shm files exist.
    connection = connect(database)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("CREATE TABLE IF NOT EXISTS probe (x)")
    connection.execute("INSERT INTO probe VALUES (1)")
    connection.commit()
    for path in tmp_path.glob("store.db*"):
        seen[path.name] = _mode(path)
    connection.close()
    assert {"store.db", "store.db-wal", "store.db-shm"} <= set(seen)
    assert set(seen.values()) == {0o600}, seen


@pytest.mark.parametrize("store", sorted(STORES))
def test_existing_sqlite_databases_keep_their_mode(tmp_path: Path, store: str) -> None:
    database = tmp_path / "store.db"
    STORES[store](database)
    database.chmod(0o640)  # e.g. deliberately shared with a group
    STORES[store](database)
    assert _mode(database) == 0o640


# ---- directory based stores ------------------------------------------------------------------


def test_file_memory_and_local_file_store_are_private(tmp_path: Path) -> None:
    memory = FileMemory(tmp_path / "nested" / "memory")
    memory.create("/memories/dir/a.md", "note")
    store = LocalFileStore(tmp_path / "other" / "files")
    store.add(b"content", name="a.txt", media_type="text/plain")
    for directory in (
        tmp_path / "nested",
        tmp_path / "nested" / "memory",
        tmp_path / "nested" / "memory" / "dir",
    ):
        assert _mode(directory) == 0o700, directory
    for directory in (
        tmp_path / "other",
        tmp_path / "other" / "files",
        tmp_path / "other" / "files" / "blobs",
    ):
        assert _mode(directory) == 0o700, directory
    files = [p for p in (tmp_path / "nested").rglob("*") if p.is_file()]
    files += [p for p in (tmp_path / "other").rglob("*") if p.is_file()]
    assert files and {_mode(p) for p in files} == {0o600}


# ---- registries ------------------------------------------------------------------------------


def test_registry_records_and_nested_directories_are_private(tmp_path: Path) -> None:
    tools = ToolRegistry(tmp_path / "state" / "tools")
    tools.add_generated(
        name="tool",
        function_name="f",
        description="test",
        code="def f(): return 1",
        tests="def test_f(): assert f() == 1",
        input_schema={},
        output_schema=None,
    )
    tools.approve("tool", 1)  # overwrite of an existing record
    agents = AgentRegistry(tmp_path / "state" / "agents")
    spec = AgentSpec(name="Analyst", description="Test", system_prompt="Test", model="fake:m", effort="low")
    agents.register(spec, "requirement with confidential detail")
    agents.revoke("Analyst")

    for directory in ("state", "state/tools", "state/tools/tool", "state/agents", "state/agents/Analyst"):
        assert _mode(tmp_path / directory) == 0o700, directory
    records = [tmp_path / "state/tools/tool/v1.json", tmp_path / "state/agents/Analyst/v1.json"]
    assert {_mode(p) for p in records} == {0o600}
    _no_temporary_files(tmp_path / "state/tools/tool")


def test_existing_directories_keep_their_mode(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o750)
    shared.chmod(0o750)
    ToolRegistry(shared).add_generated(
        name="tool",
        function_name="f",
        description="test",
        code="def f(): return 1",
        tests="def test_f(): assert f() == 1",
        input_schema={},
        output_schema=None,
    )
    assert _mode(shared) == 0o750 and _mode(shared / "tool") == 0o700


def test_the_process_umask_is_not_changed(tmp_path: Path) -> None:
    _history_agents()[0].save_history(tmp_path / "h.json")
    _use_sqlite_memory(tmp_path / "m.db")
    current = os.umask(0o022)
    assert current == 0o022
