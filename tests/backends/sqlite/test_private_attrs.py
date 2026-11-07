"""Private attributes of the SQLite backends that Nodestro uses, kept by the move (MUP-ONA MP-B2, T-SQL-11).

nodestro-runtime reads chunks through ``SQLiteIndex._connection`` under ``SQLiteIndex._lock``
(``sqlite_tables.KnowledgeChunks``), clears a memory through ``SQLiteMemory._connection``/``_lock``
(``sqlite_tables.clear_memory``), subclasses ``SQLiteMemory`` and calls ``_read``/``_files``
(``memory_spaces.MemorySpace``), and updates catalog entries with ``SQLiteFileStore._save`` under ``_lock``
(``knowledge.py``). These tests pin those names and types until a public alternative exists.
"""

from __future__ import annotations

from pathlib import Path
import sqlite3
import threading

import pytest

from orkexa import SQLiteFileStore, SQLiteIndex, SQLiteMemory
from orkexa.memory import MemoryBackend

RLOCK = type(threading.RLock())


# MUP-ONA T-SQL-11
@pytest.mark.parametrize("cls", [SQLiteIndex, SQLiteFileStore, SQLiteMemory])
def test_connection_and_lock_keep_their_names_and_types(cls: type, tmp_path: Path) -> None:
    backend = cls(tmp_path / "nodestro.db", namespace="nodestro")
    try:
        assert isinstance(backend._connection, sqlite3.Connection)
        assert isinstance(backend._lock, RLOCK)
        assert backend.namespace == "nodestro"
    finally:
        backend.close()


# MUP-ONA T-SQL-11
def test_index_chunks_are_readable_through_the_private_connection(tmp_path: Path) -> None:
    with SQLiteIndex(tmp_path / "nodestro.db", namespace="nodestro") as index:
        index.add("Refunds take five days.", source="file-1", title="Policy")
        with index._lock:
            rows = index._connection.execute(
                "SELECT source, count(*) FROM index_chunks WHERE namespace = ? GROUP BY source",
                (index.namespace,),
            ).fetchall()
            index._connection.execute(
                "UPDATE index_chunks SET file_id = ? WHERE namespace = ? AND source = ?",
                ("file-1", index.namespace, "file-1"),
            )
        assert rows == [("file-1", 1)]
        (hit,) = index.search("refunds", k=1)
        assert hit.file_id == "file-1"


class _BrowsableMemory(MemoryBackend):
    """Like Nodestro's ``BrowsableMemory``: a mixin that reads through the storage methods."""

    def entries(self) -> dict[str, int]:
        with self._lock:
            return self._files("")

    def read(self, key: str) -> str | None:
        with self._lock:
            return self._read(key)


class _MemorySpace(_BrowsableMemory, SQLiteMemory):
    """Like Nodestro's ``MemorySpace(BrowsableMemory, SQLiteMemory)``."""

    def clear(self) -> None:
        with self._lock:
            self._connection.execute("DELETE FROM memory_files WHERE namespace = ?", (self.namespace,))

    def close(self) -> None:
        SQLiteMemory.close(self)


# MUP-ONA T-SQL-11
def test_a_subclass_of_sqlite_memory_uses_the_private_storage_methods(tmp_path: Path) -> None:
    space = _MemorySpace(tmp_path / "nodestro.db", namespace="nodestro:team")
    try:
        space.create("/memories/notes.md", "hello")
        assert space.entries() == {"notes.md": 5}
        assert space.read("notes.md") == "hello"
        space.clear()
        assert space.entries() == {}
    finally:
        space.close()


# MUP-ONA T-SQL-11
def test_file_store_entries_are_saved_through_the_private_save(tmp_path: Path) -> None:
    with SQLiteFileStore(tmp_path / "nodestro.db", namespace="nodestro") as store:
        entry = store.add(b"hello", name="notes.txt")
        with store._lock:
            store._save(entry.model_copy(update={"description": "Notes"}), None)
        assert store.info(entry.id).description == "Notes"
