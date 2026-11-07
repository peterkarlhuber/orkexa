"""MUP-ONA T-API-3: every import path of the SQLite backends type-checks with ``mypy --strict``.

Not a test module: ``test_import_order.py`` runs mypy on this file. It is never executed.
"""

from __future__ import annotations

from typing import assert_type

import orkexa
from orkexa import SQLiteFileStore, SQLiteIndex, SQLiteMemory
import orkexa.backends.sqlite
from orkexa.backends.sqlite import SQLiteFileStore as LeafFileStore
from orkexa.backends.sqlite import SQLiteIndex as LeafIndex
from orkexa.backends.sqlite import SQLiteMemory as LeafMemory
from orkexa.backends.sqlite.files import SQLiteFileStore as ModuleFileStore
from orkexa.backends.sqlite.index import SQLiteIndex as ModuleIndex
from orkexa.backends.sqlite.memory import SQLiteMemory as ModuleMemory
import orkexa.files
from orkexa.files import FileStore
from orkexa.files import SQLiteFileStore as OldFileStore
import orkexa.memory
from orkexa.memory import MemoryBackend
from orkexa.memory import SQLiteMemory as OldMemory
import orkexa.retrieval
from orkexa.retrieval import Index
from orkexa.retrieval import SQLiteIndex as OldIndex


# MUP-ONA T-API-3
def probe() -> None:
    index: Index = OldIndex(":memory:", namespace="probe")
    assert_type(OldIndex(":memory:"), SQLiteIndex)
    assert_type(orkexa.retrieval.SQLiteIndex(":memory:"), LeafIndex)
    assert_type(ModuleIndex(":memory:"), orkexa.SQLiteIndex)
    assert_type(orkexa.backends.sqlite.SQLiteIndex(":memory:"), SQLiteIndex)
    store: FileStore = OldFileStore(":memory:", namespace="probe")
    assert_type(OldFileStore(":memory:"), SQLiteFileStore)
    assert_type(orkexa.files.SQLiteFileStore(":memory:"), LeafFileStore)
    assert_type(ModuleFileStore(":memory:"), orkexa.SQLiteFileStore)
    memory: MemoryBackend = OldMemory(":memory:", namespace="probe")
    assert_type(OldMemory(":memory:"), SQLiteMemory)
    assert_type(orkexa.memory.SQLiteMemory(":memory:"), LeafMemory)
    assert_type(ModuleMemory(":memory:"), orkexa.SQLiteMemory)
    hits: list[orkexa.Hit] = index.search("probe")
    del hits, store, memory


class ProbeMemory(OldMemory):
    """A subclass through the old path, as Nodestro's ``MemorySpace`` does."""

    def clear(self) -> None:
        with self._lock:
            self._connection.execute("DELETE FROM memory_files WHERE namespace = ?", (self.namespace,))
