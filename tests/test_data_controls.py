"""Technical data-handling controls that the compliance documentation refers to:
erasure in the SQLite stores, nothing persisted by default, histories and usage
records without credentials or content (deterministic, offline)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from orkexa import (
    Agent,
    EphemeralMemory,
    SQLiteFileStore,
    SQLiteIndex,
    SQLiteMemory,
    UsageRecord,
    UsageTracker,
)
from orkexa.providers.fake import FakeModel, text, tool_calls

from .helpers import add

PERSONAL = "Jane Roe, born 1980-02-03, account DE89370400440532013000"
MARKER = b"DE89370400440532013000"
KEY = "sk-test-0123456789abcdef0123456789abcdef"


def _file_bytes(database: Path) -> bytes:
    """The database with every sidecar file (WAL, journal) that SQLite left behind."""
    return b"".join(p.read_bytes() for p in database.parent.glob(database.name + "*"))


# ---- erasure ---------------------------------------------------------------------------------


def test_sqlite_memory_overwrites_deleted_files(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"
    memory = SQLiteMemory(database)
    memory.create("/memories/customer.md", PERSONAL)
    memory.create("/memories/other.md", "kept")
    memory.delete("/memories/customer.md")
    memory.close()
    assert MARKER not in _file_bytes(database)
    assert SQLiteMemory(database).view("/memories/other.md").endswith("kept")


def test_sqlite_file_store_overwrites_deleted_files(tmp_path: Path) -> None:
    database = tmp_path / "files.db"
    store = SQLiteFileStore(database)
    stored = store.add(PERSONAL.encode(), name="customer.txt", media_type="text/plain")
    store.delete(stored.id)
    store.close()
    assert MARKER not in _file_bytes(database)


def test_sqlite_index_erases_deleted_chunks_after_optimize(tmp_path: Path) -> None:
    database = tmp_path / "index.db"
    index = SQLiteIndex(database)
    index.add(PERSONAL, source="customer.md")
    index.add("Refunds are possible within 30 days.", source="policy.md")
    assert index.delete("customer.md") == 1
    assert index.search("Jane") == []  # gone from results at once
    index.optimize()  # and from the full-text index segments
    assert [hit.source for hit in index.search("refunds")] == ["policy.md"]
    index.close()
    assert MARKER not in _file_bytes(database)


def test_file_memory_delete_removes_the_file(tmp_path: Path) -> None:
    from orkexa import FileMemory

    memory = FileMemory(tmp_path / "memory")
    memory.create("/memories/customer.md", PERSONAL)
    memory.delete("/memories/customer.md")
    assert list((tmp_path / "memory").rglob("*.md")) == []


# ---- persistence and retention ---------------------------------------------------------------


def _agent(**kwargs: object) -> Agent[str]:
    script = FakeModel([tool_calls(("add", {"a": 1, "b": 2})), text("3")])
    return Agent(model="fake:m", client=script, api_key=KEY, tools=[add], **kwargs)  # type: ignore[arg-type]


def test_an_agent_writes_nothing_to_disk_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    agent = _agent(memory=EphemeralMemory())
    assert agent.ask(PERSONAL) == "3"
    assert list(tmp_path.iterdir()) == []


def test_reset_forgets_the_conversation() -> None:
    agent = _agent()
    agent.ask(PERSONAL)
    assert agent.conversation()
    agent.reset()
    assert agent.messages == [] and agent.conversation() == [] and agent.last_response is None


def test_saved_histories_contain_no_credentials(tmp_path: Path) -> None:
    agent = _agent()
    agent.ask(PERSONAL)
    path = tmp_path / "history.json"
    agent.save_history(path)
    saved = path.read_text(encoding="utf-8")
    assert PERSONAL in saved  # the conversation is the controller's data ...
    assert KEY not in saved  # ... the key is not part of it
    assert KEY not in repr(agent)


def test_usage_records_hold_metadata_but_no_content() -> None:
    tracker = UsageTracker()
    agent = _agent(usage_tracker=tracker)
    agent.ask(PERSONAL)
    fields = {f.name for f in dataclasses.fields(UsageRecord)}
    assert not fields & {"prompt", "messages", "content", "text", "output", "input"}
    assert len(tracker.records) == 2
    assert "Jane" not in repr(tracker.records) and KEY not in repr(tracker.records)
