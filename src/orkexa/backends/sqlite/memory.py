"""The SQLite memory backend :class:`SQLiteMemory` (Python's ``sqlite3``, no extra dependency).

Moved here from :mod:`orkexa.memory` without a behavior change; ``orkexa.SQLiteMemory`` and
``orkexa.memory.SQLiteMemory`` remain valid import paths for the same class.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
import os

from ..._storage import connect_private_database, storage_errors, write_transaction
from ...memory import MemoryBackend

__all__ = ["SQLiteMemory"]


class SQLiteMemory(MemoryBackend):
    """Persistent memory in a SQLite database (Python's ``sqlite3``, no extra dependency).

    Each ``namespace`` is a separate ``/memories`` (one per user or project) in the
    table ``memory_files``. Every command runs in a transaction, and the database
    uses WAL mode, so several processes can share one file.

    Args:
        database: Path of the database file (``":memory:"`` for a temporary one).
        namespace: Which memory in the database this backend works on.
        max_file_chars: See :class:`MemoryBackend`.
        max_view_chars: See :class:`MemoryBackend`.

    A database path that is a symbolic link raises :class:`~orkexa.retrieval.RetrievalError`
    (``storage.symlink``); a damaged file raises it with ``storage.corrupt``.

    The private attributes ``_connection`` (the ``sqlite3.Connection``) and ``_lock`` (the ``threading.RLock``
    of :class:`MemoryBackend`) and the storage methods ``_read`` and ``_files`` are preserved for
    compatibility: Nodestro subclasses this class and uses them. They are not public API.
    """

    def __init__(
        self,
        database: str | os.PathLike[str],
        *,
        namespace: str = "default",
        max_file_chars: int = 100_000,
        max_view_chars: int = 16_000,
    ) -> None:
        super().__init__(max_file_chars=max_file_chars, max_view_chars=max_view_chars)
        if not namespace:
            raise ValueError("namespace must not be empty")
        self.namespace = namespace
        self._connection = connect_private_database(database)  # refuses symlinks and damaged files
        try:
            with storage_errors():
                self._connection.execute(
                    "CREATE TABLE IF NOT EXISTS memory_files ("
                    " namespace TEXT NOT NULL, path TEXT NOT NULL, content TEXT NOT NULL,"
                    " updated_at TEXT NOT NULL, PRIMARY KEY (namespace, path))"
                )
        except BaseException:
            self._connection.close()
            raise

    @property
    def capabilities(self) -> frozenset[str]:
        """``partitioned`` (by ``namespace``) and ``concurrent_writers``.

        See :attr:`MemoryBackend.capabilities`."""
        return frozenset({"partitioned", "concurrent_writers"})

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self._connection.close()

    def __enter__(self) -> SQLiteMemory:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        with write_transaction(self._connection):  # BEGIN IMMEDIATE, rolled back on any error
            yield

    def _read(self, key: str) -> str | None:
        row = self._connection.execute(
            "SELECT content FROM memory_files WHERE namespace = ? AND path = ?", (self.namespace, key)
        ).fetchone()
        return None if row is None else str(row[0])

    def _write(self, key: str, text: str) -> None:
        self._connection.execute(
            "INSERT INTO memory_files (namespace, path, content, updated_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT (namespace, path) DO UPDATE SET content = excluded.content, "
            "updated_at = excluded.updated_at",
            (self.namespace, key, text, datetime.now(UTC).isoformat()),
        )

    def _files(self, prefix: str) -> dict[str, int]:
        start = f"{prefix}/" if prefix else ""
        rows = self._connection.execute(
            "SELECT path, length(CAST(content AS BLOB)) FROM memory_files "
            "WHERE namespace = ? AND substr(path, 1, ?) = ?",
            (self.namespace, len(start), start),
        )
        return {str(path): int(size) for path, size in rows}

    def _delete(self, key: str) -> None:
        self._connection.execute(
            "DELETE FROM memory_files WHERE namespace = ? AND (path = ? OR substr(path, 1, ?) = ?)",
            (self.namespace, key, len(key) + 1, key + "/"),
        )

    def _rename(self, old: str, new: str) -> None:
        self._connection.execute(
            "UPDATE memory_files SET path = ? || substr(path, ?), updated_at = ? "
            "WHERE namespace = ? AND (path = ? OR substr(path, 1, ?) = ?)",
            (
                new,
                len(old) + 1,
                datetime.now(UTC).isoformat(),
                self.namespace,
                old,
                len(old) + 1,
                old + "/",
            ),
        )
