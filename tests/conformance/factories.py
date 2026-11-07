"""Factories of the Orkexa backends for the conformance suite (MUP-ONA B1, plan MP-B1 step 4).

Each factory opens a new instance on the storage of the case: a file or directory in ``request.workdir``,
partitioned by ``request.partition`` where the backend has partitions (the SQLite namespace). ``LEGACY`` maps
a factory to the FX-SQ-0 file it can start from (T-SQL-10): the case's storage is then a copy of that legacy
file.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import shutil
from typing import Any

from orkexa import EphemeralMemory, FileMemory, LocalFileStore, SQLiteFileStore, SQLiteIndex, SQLiteMemory
from orkexa.testing.conformance import Request

LEGACY_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "legacy" / "FX-SQ-0"


def _database(request: Request, name: str, legacy: str | None) -> Path:
    path = request.workdir / name
    if legacy is not None and not path.exists():
        shutil.copyfile(LEGACY_DIR / legacy, path)  # a private copy: the committed fixture stays unchanged
    return path


def sqlite_index_fts(request: Request, *, legacy: str | None = None) -> SQLiteIndex:
    """SQLiteIndex without vectors (no sqlite-vec needed): the embedding model of the request is ignored."""
    return SQLiteIndex(_database(request, "index.db", legacy), namespace=request.partition)


def sqlite_index_vec(request: Request, *, legacy: str | None = None) -> SQLiteIndex:
    """SQLiteIndex with the request's embedding model (vector spaces, sqlite-vec)."""
    return SQLiteIndex(
        _database(request, "index.db", legacy), namespace=request.partition, embeddings=request.embeddings
    )


def sqlite_file_store(request: Request, *, legacy: str | None = None) -> SQLiteFileStore:
    return SQLiteFileStore(_database(request, "files.db", legacy), namespace=request.partition)


def sqlite_memory(request: Request, *, legacy: str | None = None) -> SQLiteMemory:
    return SQLiteMemory(_database(request, "memory.db", legacy), namespace=request.partition)


def local_file_store(request: Request) -> LocalFileStore:
    return LocalFileStore(request.workdir / "files")


def file_memory(request: Request) -> FileMemory:
    return FileMemory(request.workdir / "memory")


def ephemeral_memory(request: Request) -> EphemeralMemory:
    return EphemeralMemory()


FACTORIES: dict[str, tuple[Callable[..., Any], frozenset[str]]] = {
    "sqlite-index-fts": (sqlite_index_fts, frozenset({"partitioned", "concurrent_writers"})),
    "sqlite-index-vec": (
        sqlite_index_vec,
        frozenset({"vectors", "reindex", "partitioned", "concurrent_writers"}),
    ),
    "sqlite-file-store": (sqlite_file_store, frozenset({"partitioned", "concurrent_writers"})),
    "sqlite-memory": (sqlite_memory, frozenset({"partitioned", "concurrent_writers"})),
    "local-file-store": (local_file_store, frozenset()),
    "file-memory": (file_memory, frozenset()),
    "ephemeral-memory": (ephemeral_memory, frozenset()),
}
"""Factory and expected capabilities per target name (the names of tests/conformance/expected_skips.json)."""

LEGACY: dict[str, str] = {
    "sqlite-index-fts": "index_fts.db",
    "sqlite-index-vec": "index_vec.db",
    "sqlite-file-store": "files.db",
    "sqlite-memory": "memory.db",
}
"""The FX-SQ-0 file each SQLite factory starts from in the legacy run (T-SQL-10)."""
