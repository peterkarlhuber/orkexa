"""Integrity of the legacy fixtures FX-SQ-0 (MUP-ONA A0): checksums, and every file opens today."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sqlite3

import pytest

from orkexa import SQLiteFileStore, SQLiteIndex, SQLiteMemory
from orkexa.embeddings import CallableEmbeddings

from .audit3._vec import needs_vec

LEGACY = Path(__file__).resolve().parent / "fixtures" / "legacy"
FIXTURE = LEGACY / "FX-SQ-0"
DATABASES = ("index_fts.db", "index_vec.db", "files.db", "memory.db")


def _manifest() -> dict[str, object]:
    data: dict[str, object] = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))
    return data


def _contents(name: str) -> dict[str, object]:
    contents = _manifest()["contents"]
    assert isinstance(contents, dict)
    value: dict[str, object] = contents[name]
    return value


def _copy(name: str, tmp_path: Path) -> Path:
    """A private copy: opening a database writes to it (WAL), the committed file must stay unchanged."""
    target = tmp_path / name
    shutil.copyfile(FIXTURE / name, target)
    return target


# MUP-ONA T-FX-1
@pytest.mark.parametrize("name", DATABASES)
def test_the_checksums_match(name: str) -> None:
    digest = hashlib.sha256((FIXTURE / name).read_bytes()).hexdigest()
    recorded, file_name = (FIXTURE / f"{name}.sha256").read_text(encoding="utf-8").split()
    assert file_name == name
    assert digest == recorded
    checksums = _manifest()["checksums"]
    assert isinstance(checksums, dict)
    assert checksums[name] == digest


# MUP-ONA T-FX-1
@pytest.mark.parametrize("name", DATABASES)
def test_the_files_are_intact_sqlite_databases(name: str, tmp_path: Path) -> None:
    connection = sqlite3.connect(_copy(name, tmp_path))
    try:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        if name.startswith("index_"):
            connection.execute("INSERT INTO index_fts (index_fts) VALUES ('integrity-check')")
    finally:
        connection.close()


# MUP-ONA T-FX-1
def test_the_full_text_index_opens_and_finds_its_sources(tmp_path: Path) -> None:
    database = _copy("index_fts.db", tmp_path)
    for namespace, sources in _contents("index_fts.db").items():
        with SQLiteIndex(database, namespace=namespace) as index:
            assert index.sources() == sources
    with SQLiteIndex(database, namespace="handbook") as index:
        assert [hit.source for hit in index.search("refund warranty", 1)] == ["warranty.md"]


# MUP-ONA T-FX-1
def test_the_vector_index_opens_for_full_text_search(tmp_path: Path) -> None:
    database = _copy("index_vec.db", tmp_path)
    contents = _contents("index_vec.db")
    for namespace in ("spaces", "legacy"):
        with SQLiteIndex(database, namespace=namespace) as index:
            assert index.sources() == contents[namespace]


# MUP-ONA T-FX-1
@needs_vec
def test_the_vector_index_opens_with_its_vectors(tmp_path: Path) -> None:
    from .fixtures.legacy.make_legacy_sqlite import DIMENSIONS, vectors_for

    database = _copy("index_vec.db", tmp_path)
    model = CallableEmbeddings(vectors_for, name="fx-sq-0-synthetic", dimensions=DIMENSIONS)
    with SQLiteIndex(database, namespace="spaces", embeddings=model) as index:
        status = index.vector_status()
        assert status.complete and status.chunks == 4 and status.embedded == 4
        hits = index.retrieve("humidity", 4, mode="vector")
        assert hits[0].source == "beta.txt"
        assert "zero.txt" not in [hit.source for hit in hits]  # a zero vector has no cosine distance
    with SQLiteIndex(database, namespace="legacy", embedder=vectors_for) as index:
        assert index.vector_status().legacy
        assert index.retrieve("wind", 1, mode="vector")[0].source == "legacy-b.txt"


# MUP-ONA T-FX-1
def test_the_file_store_opens_with_its_entries(tmp_path: Path) -> None:
    database = _copy("files.db", tmp_path)
    for namespace, ids in _contents("files.db").items():
        with SQLiteFileStore(database, namespace=namespace) as store:
            assert sorted(entry.id for entry in store.list()) == sorted(ids)
            assert isinstance(ids, list)
            for file_id in ids:
                entry = store.info(file_id)
                assert hashlib.sha256(store.get_bytes(file_id)).hexdigest() == entry.sha256


# MUP-ONA T-FX-1
def test_the_memory_opens_with_its_files(tmp_path: Path) -> None:
    from .fixtures.legacy.make_legacy_sqlite import MEMORIES

    database = _copy("memory.db", tmp_path)
    for namespace, paths in _contents("memory.db").items():
        memory = SQLiteMemory(database, namespace=namespace)
        try:
            assert isinstance(paths, list)
            for path in paths:
                listing = memory.view(path)
                for line in MEMORIES[namespace][path].splitlines():
                    assert line in listing
        finally:
            memory.close()
