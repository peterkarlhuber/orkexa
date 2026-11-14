"""The SQLite file store :class:`SQLiteFileStore` (Python's ``sqlite3``, no extra dependency).

Moved here from :mod:`orkexa.files` without a behavior change; ``orkexa.SQLiteFileStore`` and
``orkexa.files.SQLiteFileStore`` remain valid import paths for the same class.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import os
from typing import Any

from ..._storage import connect_private_database, storage_errors, write_transaction
from ...files import FileStore, StoredFile

__all__ = ["SQLiteFileStore"]


class SQLiteFileStore(FileStore):
    """Files in a SQLite database (Python's ``sqlite3``, no extra dependency).

    Each ``namespace`` is a separate store in the table ``stored_files``; the file
    can be the same database that ``SQLiteMemory`` or ``SQLiteIndex`` use.

    Args:
        database: Path of the database file (``":memory:"`` for a temporary one).
        namespace: Which store in the database this backend works on.
        files_api: See :class:`FileStore`.
        max_image_bytes: See :class:`FileStore`.
        max_document_bytes: See :class:`FileStore`.

    A database path that is a symbolic link raises :class:`~orkexa.retrieval.RetrievalError`
    (``storage.symlink``); a damaged file raises it with ``storage.corrupt``.

    The private attributes ``_connection`` (the ``sqlite3.Connection``) and ``_lock`` (the ``threading.RLock``
    of :class:`FileStore`) and the storage method ``_save`` are preserved for compatibility: Nodestro updates
    catalog entries through them. They are not public API.
    """

    def __init__(
        self,
        database: str | os.PathLike[str],
        *,
        namespace: str = "default",
        files_api: Any | None = None,
        max_image_bytes: int = 5_000_000,
        max_document_bytes: int = 20_000_000,
    ) -> None:
        super().__init__(
            files_api=files_api, max_image_bytes=max_image_bytes, max_document_bytes=max_document_bytes
        )
        if not namespace:
            raise ValueError("namespace must not be empty")
        self.namespace = namespace
        self._connection = connect_private_database(database)
        try:
            with storage_errors():
                self._connection.execute(
                    "CREATE TABLE IF NOT EXISTS stored_files ("
                    " namespace TEXT NOT NULL, id TEXT NOT NULL, entry TEXT NOT NULL, data BLOB NOT NULL,"
                    " PRIMARY KEY (namespace, id))"
                )
        except BaseException:
            self._connection.close()
            raise

    @property
    def capabilities(self) -> frozenset[str]:
        """``partitioned`` (by ``namespace``), ``concurrent_writers``; see :attr:`FileStore.capabilities`."""
        return frozenset({"partitioned", "concurrent_writers"})

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self._connection.close()

    def __enter__(self) -> SQLiteFileStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        # Two processes adding the same file must not both read "missing" and overwrite each other's entry.
        with write_transaction(self._connection):  # BEGIN IMMEDIATE, rolled back on any error
            yield

    def _save(self, entry: StoredFile, data: bytes | None) -> None:
        if data is None:
            self._connection.execute(
                "UPDATE stored_files SET entry = ? WHERE namespace = ? AND id = ?",
                (entry.model_dump_json(), self.namespace, entry.id),
            )
        else:
            self._connection.execute(
                "INSERT OR REPLACE INTO stored_files (namespace, id, entry, data) VALUES (?, ?, ?, ?)",
                (self.namespace, entry.id, entry.model_dump_json(), data),
            )

    def _entry(self, file_id: str) -> StoredFile | None:
        row = self._connection.execute(
            "SELECT entry FROM stored_files WHERE namespace = ? AND id = ?", (self.namespace, file_id)
        ).fetchone()
        return None if row is None else StoredFile.model_validate_json(row[0])

    def _data(self, file_id: str) -> bytes | None:
        row = self._connection.execute(
            "SELECT data FROM stored_files WHERE namespace = ? AND id = ?", (self.namespace, file_id)
        ).fetchone()
        return None if row is None else bytes(row[0])

    def _entries(self) -> list[StoredFile]:
        rows: Iterator[tuple[str]] = self._connection.execute(
            "SELECT entry FROM stored_files WHERE namespace = ?", (self.namespace,)
        )
        return [StoredFile.model_validate_json(row[0]) for row in rows]

    def _remove(self, file_id: str) -> None:
        self._connection.execute(
            "DELETE FROM stored_files WHERE namespace = ? AND id = ?", (self.namespace, file_id)
        )
