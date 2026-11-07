"""SQLite backend robustness: concurrent writers, query limits, damaged and legacy files (MUP-ONA MP-A10)."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import sqlite3
import stat
import subprocess
import sys
import threading
import time
from types import ModuleType
from typing import Any

import pytest

from orkexa import SQLiteFileStore, SQLiteIndex, SQLiteMemory
from orkexa.embeddings import CallableEmbeddings, EmbeddingIdentity, EmbeddingModel
from orkexa.errors import ToolExecutionError
from orkexa.retrieval import MAX_QUERY_CHARS, MAX_QUERY_WORDS, RetrievalError

from ...audit3._vec import needs_vec

TESTS = Path(__file__).resolve().parents[2]
LEGACY = TESTS / "fixtures" / "legacy" / "FX-SQ-0"
RANKING = TESTS / "fixtures" / "ranking"

ROUNDS = 100
"""Sources each concurrent deleter removes; every round is a chance for the PS-8 race."""

_DELETER = """
import sys, time
from pathlib import Path
from orkexa import SQLiteIndex

database, rounds, go = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
index = SQLiteIndex(database)
print("ready", flush=True)
while not go.exists():
    time.sleep(0.001)
errors = 0
for number in range(rounds):
    try:
        index.delete(f"source-{number}")
    except Exception as exc:
        errors += 1
        print(f"error {type(exc).__name__}: {exc}", flush=True)
index.close()
print(f"errors {errors}", flush=True)
"""


def _fill(database: Path) -> None:
    with SQLiteIndex(database) as index:
        for number in range(ROUNDS):
            for chunk in range(3):
                index.add(f"round {number} chunk {chunk} alpha beta gamma", source=f"source-{number}")


def _assert_intact(database: Path) -> None:
    connection = sqlite3.connect(database)
    try:
        connection.execute("INSERT INTO index_fts (index_fts) VALUES ('integrity-check')")
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        connection.close()
    with SQLiteIndex(database) as index:
        assert len(index) == 0
        assert index.search("alpha beta") == []
        index.add("a fresh chunk about alpha after the race", source="fresh")
        assert [hit.source for hit in index.search("alpha")] == ["fresh"]


def _delete_with_two_instances(database: Path) -> list[str]:
    errors: list[str] = []
    barrier = threading.Barrier(2)

    def work() -> None:
        index = SQLiteIndex(database)
        try:
            barrier.wait()
            for number in range(ROUNDS):
                try:
                    index.delete(f"source-{number}")
                except sqlite3.DatabaseError as exc:
                    errors.append(f"{type(exc).__name__}: {exc}")
        finally:
            index.close()

    threads = [threading.Thread(target=work) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
    return errors


def _delete_with_two_processes(database: Path, tmp_path: Path) -> list[str]:
    go = tmp_path / "go"
    workers = [
        subprocess.Popen(
            [sys.executable, "-c", _DELETER, str(database), str(ROUNDS), str(go)],
            stdout=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    try:
        for worker in workers:
            assert worker.stdout is not None
            assert worker.stdout.readline().strip() == "ready"
        go.touch()  # both workers start deleting at the same moment
        outputs = [worker.communicate(timeout=120)[0] for worker in workers]
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
                worker.wait()
    assert all(worker.returncode == 0 for worker in workers), outputs
    return [line for output in outputs for line in output.splitlines() if line.startswith("error ")]


# MUP-ONA T-SQL-1
def test_two_instances_and_two_processes_delete_the_same_sources(tmp_path: Path) -> None:
    """Two instances (threads) and then two processes delete the same 100 sources at once.

    ``SQLiteIndex.delete`` must read the rows inside its write transaction: otherwise both writers send the
    FTS5 ``'delete'`` for the same rows, and the second one fails with "database disk image is malformed"
    (PS-8).
    """
    threads_db = tmp_path / "threads.db"
    _fill(threads_db)
    thread_errors = _delete_with_two_instances(threads_db)

    processes_db = tmp_path / "processes.db"
    _fill(processes_db)
    process_errors = _delete_with_two_processes(processes_db, tmp_path)

    assert thread_errors == []
    assert process_errors == []
    _assert_intact(threads_db)
    _assert_intact(processes_db)


# ---- T-SQL-2: query limits ----------------------------------------------------------------------------


def _filled_index(path: Path, **options: Any) -> SQLiteIndex:
    index = SQLiteIndex(path, **options)
    index.add("The warranty covers two years. Refunds take fourteen days.", source="warranty.md")
    index.add("Parcels ship every weekday from the central warehouse.", source="shipping.md")
    return index


# MUP-ONA T-SQL-2
def test_an_overlong_query_is_refused_at_once_without_the_lock(tmp_path: Path) -> None:
    index = _filled_index(tmp_path / "index.db")
    words = " ".join(f"word{number}" for number in range(5000))
    query = (words + " ") * (100_000 // (len(words) + 1) + 1)
    query = query[:100_000]
    holding = threading.Event()
    release = threading.Event()

    def hold_the_lock() -> None:  # a long write elsewhere: the check must not wait for it
        with index._lock:
            holding.set()
            release.wait(10)

    holder = threading.Thread(target=hold_the_lock)
    holder.start()
    try:
        assert holding.wait(5)
        for mode in ("fts", None):
            started = time.monotonic()
            with pytest.raises(RetrievalError) as error:
                index.retrieve(query, 5, mode=mode)  # type: ignore[arg-type]
            assert error.value.code == "query.too_long"
            assert time.monotonic() - started < 1.0
    finally:
        release.set()
        holder.join()
    assert [hit.source for hit in index.search("warranty")] == ["warranty.md"]  # a reader is not blocked
    index.close()


# MUP-ONA T-SQL-2
def test_the_query_limits_count_characters_and_distinct_words(tmp_path: Path) -> None:
    index = _filled_index(tmp_path / "index.db")
    assert (MAX_QUERY_CHARS, MAX_QUERY_WORDS) == (1000, 64)
    at_limit = " ".join(f"w{number}" for number in range(MAX_QUERY_WORDS - 1)) + " warranty"
    assert [hit.source for hit in index.search(at_limit)] == ["warranty.md"]
    assert index.search("warranty " * 100) != []  # repeated words count once (600 characters)
    for query in (at_limit + " extra", "x" * (MAX_QUERY_CHARS + 1)):
        with pytest.raises(RetrievalError) as error:
            index.search(query)
        assert error.value.code == "query.too_long"
    with pytest.raises(ToolExecutionError) as failed:  # the search tool checks too
        index.tool().execute({"query": at_limit + " extra"})
    assert isinstance(failed.value.__cause__, RetrievalError)
    assert failed.value.__cause__.code == "query.too_long"
    index.close()


# MUP-ONA T-SQL-2
def test_the_query_limits_are_configurable(tmp_path: Path) -> None:
    index = _filled_index(tmp_path / "index.db", max_query_chars=20, max_query_words=2)
    assert index.search("warranty refunds") != []
    for query in ("warranty refunds days", "warranty" + " " * 20):
        with pytest.raises(RetrievalError, match="at most"):
            index.search(query)
    index.close()
    roomy = _filled_index(tmp_path / "roomy.db", max_query_chars=5000, max_query_words=500)
    assert roomy.search(" ".join(f"w{number}" for number in range(300)) + " warranty") != []
    roomy.close()
    with pytest.raises(ValueError):
        SQLiteIndex(tmp_path / "bad.db", max_query_words=0)


# ---- T-SQL-3: FTS5 syntax is plain text ---------------------------------------------------------------


# MUP-ONA T-SQL-3
@pytest.mark.parametrize(
    "query",
    [
        "NEAR(alpha beta)",
        "text: alpha",
        "title: secret",
        '"alpha',
        "alpha*",
        "^alpha",
        "alpha AND NOT beta",
        "alpha OR",
        "(alpha",
        "{text title}: alpha",
        "alpha -beta",
        "١٢٣ alpha",
        "alpha\x00beta",
        "'",
        "",
    ],
)
def test_fts5_syntax_in_a_query_is_searched_as_words(tmp_path: Path, query: str) -> None:
    with SQLiteIndex(tmp_path / "index.db") as index:
        index.add("alpha beta gamma, item ١٢٣", source="plain.md")
        index.add("zulu yankee xray", source="secret.md", title="secret")
        hits = index.search(query)
        words = {word.lower() for word in query.replace("\\x00", " ").split()}
        if "secret" not in " ".join(words):
            assert all(hit.source == "plain.md" for hit in hits)  # no column filter on the title
        if query in ("text: alpha", "١٢٣ alpha", "NEAR(alpha beta)"):
            assert [hit.source for hit in hits] == ["plain.md"]


# ---- T-SQL-4: extension loading is switched off again -------------------------------------------------


def _extension_loading_enabled(connection: sqlite3.Connection) -> bool:
    try:
        connection.execute("SELECT load_extension('orkexa-no-such-extension')")
    except sqlite3.OperationalError as exc:
        return "not authorized" not in str(exc)
    return True  # pragma: no cover - the extension cannot exist


# MUP-ONA T-SQL-4
@needs_vec
def test_a_failing_sqlite_vec_load_leaves_extension_loading_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sqlite_vec  # type: ignore[import-not-found,import-untyped,unused-ignore]

    def broken(connection: sqlite3.Connection) -> None:
        assert _extension_loading_enabled(connection)  # on while loading
        raise sqlite3.OperationalError("simulated load failure")

    with SQLiteIndex(tmp_path / "index.db") as index:
        monkeypatch.setattr(sqlite_vec, "load", broken)
        with pytest.raises(sqlite3.OperationalError, match="simulated"):
            index._load_sqlite_vec()
        assert not _extension_loading_enabled(index._connection)
        assert not index._vec_loaded
        monkeypatch.undo()
        index._load_sqlite_vec()  # a later attempt works and switches loading off again
        assert index._vec_loaded and not _extension_loading_enabled(index._connection)


# ---- T-SQL-5: query vector dimension ------------------------------------------------------------------


class _WrongQueryLength(EmbeddingModel):
    """Documents get 4 dimensions, queries 3 (a broken model that does not check itself)."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, float(len(text)), 0.5, 0.25] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 2.0, 3.0]

    def identity(self) -> EmbeddingIdentity:
        return EmbeddingIdentity(provider="test", model="wrong-query-length", dimensions=4)


# MUP-ONA T-SQL-5
@needs_vec
@pytest.mark.parametrize("mode", ["vector", "hybrid"])
def test_a_query_vector_of_another_length_is_refused(tmp_path: Path, mode: str) -> None:
    with SQLiteIndex(tmp_path / "spaces.db", embeddings=_WrongQueryLength()) as index:
        index.add("alpha beta", source="a.md")
        with pytest.raises(RetrievalError) as error:
            index.retrieve("alpha", mode=mode)  # type: ignore[arg-type]
        assert error.value.code == "vector.dimension"
    documents = lambda texts: [[1.0, 0.5, 0.25, 0.125] for _ in texts]  # noqa: E731
    queries = lambda texts: [[1.0, 0.5] for _ in texts]  # noqa: E731
    with SQLiteIndex(tmp_path / "legacy.db", embedder=documents, query_embedder=queries) as index:
        index.add("alpha beta", source="a.md")
        with pytest.raises(RetrievalError) as error:
            index.retrieve("alpha", mode=mode)  # type: ignore[arg-type]
        assert error.value.code == "vector.dimension"


# ---- T-SQL-6: symbolic links --------------------------------------------------------------------------

_OPENERS = [SQLiteIndex, SQLiteFileStore, SQLiteMemory]


# MUP-ONA T-SQL-6
@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")
@pytest.mark.parametrize("opener", _OPENERS, ids=lambda cls: cls.__name__)
@pytest.mark.parametrize("suffix", ["", "-wal", "-shm", "-journal"])
def test_a_symlinked_database_path_is_refused(tmp_path: Path, opener: Any, suffix: str) -> None:
    real = tmp_path / "real.db"
    opener(real).close()
    link = tmp_path / "link.db"
    if suffix:
        shutil.copyfile(real, link)
        (tmp_path / f"elsewhere{suffix}").write_bytes(b"")
        os.symlink(tmp_path / f"elsewhere{suffix}", f"{link}{suffix}")
    else:
        os.symlink(real, link)
    before = hashlib.sha256(real.read_bytes()).hexdigest()
    with pytest.raises(RetrievalError) as error:
        opener(link)
    assert error.value.code == "storage.symlink"
    assert str(tmp_path) not in str(error.value)  # no path in the message
    assert hashlib.sha256(real.read_bytes()).hexdigest() == before


# MUP-ONA T-SQL-6
@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")
def test_a_dangling_symlink_is_refused_and_not_created(tmp_path: Path) -> None:
    link = tmp_path / "index.db"
    os.symlink(tmp_path / "target.db", link)
    with pytest.raises(RetrievalError) as error:
        SQLiteIndex(link)
    assert error.value.code == "storage.symlink"
    assert not (tmp_path / "target.db").exists()


# ---- T-SQL-7: damaged files ---------------------------------------------------------------------------


def _damaged(tmp_path: Path, kind: str) -> Path:
    source = tmp_path / "source.db"
    with SQLiteIndex(source) as index:
        for number in range(200):
            index.add(f"chunk {number} alpha beta gamma " * 20, source=f"source-{number}")
    data = source.read_bytes()
    target = tmp_path / f"{kind}.db"
    generator = random.Random(7)
    if kind == "truncated":
        target.write_bytes(data[: len(data) // 3])
    elif kind == "random":
        target.write_bytes(bytes(generator.randrange(256) for _ in range(8192)))
    else:  # scribbled: the header survives, every page gets random bytes
        damaged = bytearray(data)
        for offset in range(4096, len(damaged), 4096):
            damaged[offset + 100 : offset + 400] = bytes(generator.randrange(256) for _ in range(300))
        target.write_bytes(bytes(damaged))
    return target


# MUP-ONA T-SQL-7
@pytest.mark.parametrize("kind", ["truncated", "random", "scribbled"])
def test_a_damaged_index_raises_storage_corrupt(tmp_path: Path, kind: str) -> None:
    path = _damaged(tmp_path, kind)
    with pytest.raises(RetrievalError) as error:
        index = SQLiteIndex(path)
        try:
            index.search("alpha")
            index.delete("source-5")
        finally:
            index.close()
    assert error.value.code == "storage.corrupt"
    assert isinstance(error.value.__cause__, sqlite3.DatabaseError)
    assert str(tmp_path) not in str(error.value)


# MUP-ONA T-SQL-7
@pytest.mark.parametrize("opener", [SQLiteFileStore, SQLiteMemory], ids=lambda cls: cls.__name__)
def test_a_file_that_is_no_database_raises_storage_corrupt(tmp_path: Path, opener: Any) -> None:
    path = _damaged(tmp_path, "random")
    with pytest.raises(RetrievalError) as error:
        opener(path)
    assert error.value.code == "storage.corrupt"


# ---- T-SQL-8: SQLiteFileStore.add from two processes --------------------------------------------------

FILE_ROUNDS = 200

_ADDER = """
import sys, time
from pathlib import Path
from orkexa import SQLiteFileStore

database, worker, rounds, go = sys.argv[1], sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
store = SQLiteFileStore(database)
print("ready", flush=True)
while not go.exists():
    time.sleep(0.001)
for number in range(rounds):
    shared = f"shared file {number}".encode()
    store.add(shared, name=f"shared-{number}.txt", metadata={worker: str(number)})
    store.add(f"own file {worker} {number}".encode(), name=f"{worker}-{number}.txt")
store.close()
print("done", flush=True)
"""


# MUP-ONA T-SQL-8
def test_two_processes_adding_files_lose_no_entry(tmp_path: Path) -> None:
    database = tmp_path / "files.db"
    SQLiteFileStore(database).close()
    go = tmp_path / "go"
    workers = [
        subprocess.Popen(
            [sys.executable, "-c", _ADDER, str(database), name, str(FILE_ROUNDS), str(go)],
            stdout=subprocess.PIPE,
            text=True,
        )
        for name in ("p1", "p2")
    ]
    try:
        for worker in workers:
            assert worker.stdout is not None
            assert worker.stdout.readline().strip() == "ready"
        go.touch()
        outputs = [worker.communicate(timeout=180)[0] for worker in workers]
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
                worker.wait()
    assert all(worker.returncode == 0 for worker in workers), outputs
    with SQLiteFileStore(database) as store:
        entries = store.list()
        assert len(entries) == 3 * FILE_ROUNDS
        shared = [entry for entry in entries if entry.name.startswith("shared-")]
        assert len(shared) == FILE_ROUNDS
        for entry in shared:  # both workers' metadata survived: no read-then-overwrite
            number = entry.name.removeprefix("shared-").removesuffix(".txt")
            assert entry.metadata == {"p1": number, "p2": number}


# ---- T-SQL-9: file modes, secure_delete, WAL ----------------------------------------------------------


# MUP-ONA T-SQL-9
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
@pytest.mark.parametrize("opener", _OPENERS, ids=lambda cls: cls.__name__)
def test_database_files_are_private_with_secure_delete_and_wal(tmp_path: Path, opener: Any) -> None:
    path = tmp_path / "data.db"
    backend = opener(path)
    try:
        connection: sqlite3.Connection = backend._connection
        assert connection.execute("PRAGMA secure_delete").fetchone()[0] == 1
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        connection.execute("CREATE TABLE IF NOT EXISTS probe (x)")
        connection.execute("INSERT INTO probe VALUES (1)")
        for name in ("data.db", "data.db-wal", "data.db-shm"):
            assert stat.S_IMODE((tmp_path / name).stat().st_mode) == 0o600, name
    finally:
        backend.close()


# ---- T-SQL-10: legacy files FX-SQ-0 -------------------------------------------------------------------


def _legacy(name: str, tmp_path: Path) -> Path:
    target = tmp_path / name
    shutil.copyfile(LEGACY / name, target)
    return target


def _legacy_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "make_legacy_sqlite", LEGACY.parent / "make_legacy_sqlite.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# MUP-ONA T-SQL-10
def test_a_legacy_full_text_index_is_searched_written_and_deleted(tmp_path: Path) -> None:
    database = _legacy("index_fts.db", tmp_path)
    with SQLiteIndex(database, namespace="handbook") as index:
        assert index.search("refunds", 1)[0].source == "warranty.md"
        index.add("Gift cards never expire.", source="gift-cards.md")
        assert index.search("gift cards", 1)[0].source == "gift-cards.md"
        assert index.delete("warranty.md") == 1
        assert index.search("refunds") == []
        assert index.sources() == ["gift-cards.md", "privacy.md", "shipping.md"]
    with SQLiteIndex(database, namespace="tickets") as index:
        assert index.sources() == ["ticket-1001", "ticket-1002"]  # the other namespace is untouched
    connection = sqlite3.connect(database)
    try:
        connection.execute("INSERT INTO index_fts (index_fts) VALUES ('integrity-check')")
    finally:
        connection.close()


# MUP-ONA T-SQL-10
@needs_vec
def test_a_legacy_vector_index_is_searched_written_and_deleted(tmp_path: Path) -> None:
    legacy = _legacy_module()
    database = _legacy("index_vec.db", tmp_path)
    model = CallableEmbeddings(legacy.vectors_for, name="fx-sq-0-synthetic", dimensions=legacy.DIMENSIONS)
    with SQLiteIndex(database, namespace="spaces", embeddings=model) as index:
        index.add("Delta station measures wind and humidity.", source="delta.txt")
        assert index.vector_status().complete
        assert index.delete("beta.txt") == 1
        hits = index.retrieve("humidity", 3, mode="hybrid")
        assert hits[0].source == "delta.txt"
    with SQLiteIndex(database, namespace="legacy", embedder=legacy.vectors_for) as index:
        index.add("Legacy chunk about wind farms.", source="legacy-c.txt")
        assert index.delete("legacy-b.txt") == 1
        assert index.retrieve("wind", 1, mode="vector")[0].source == "legacy-c.txt"


# MUP-ONA T-SQL-10
def test_a_legacy_file_store_and_memory_are_read_written_and_deleted(tmp_path: Path) -> None:
    manifest = json.loads((LEGACY / "manifest.json").read_text(encoding="utf-8"))
    files = _legacy("files.db", tmp_path)
    with SQLiteFileStore(files, namespace="alpha") as store:
        first, second = manifest["contents"]["files.db"]["alpha"]
        assert store.get_bytes(first).startswith(b"Plain text note")
        added = store.add(b"new text", name="new.txt")
        store.delete(second)
        assert sorted(entry.id for entry in store.list()) == sorted([first, added.id])
    memory = SQLiteMemory(_legacy("memory.db", tmp_path), namespace="alpha")
    try:
        assert "Phase 2 started." in memory.view("/memories/projects/alpha/status.md")
        memory.create("/memories/new.md", "written by the current orkexa\n")
        memory.delete("/memories/profile.md")
        assert "new.md" in memory.view("/memories")
        assert "profile.md" not in memory.view("/memories")
    finally:
        memory.close()


# ---- T-RANK-1: FX-RANK-0 results are unchanged --------------------------------------------------------


def _ranking_module(name: str) -> ModuleType:
    sys.path.insert(0, str(RANKING))  # measure_sqlite imports make_rank_corpus from its directory
    try:
        spec = importlib.util.spec_from_file_location(name, RANKING / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(RANKING))
    return module


# MUP-ONA T-RANK-1
@pytest.mark.parametrize("vec", [pytest.param(True, marks=needs_vec), False], ids=["sqlite", "sqlite-novec"])
def test_the_ranking_fixture_gives_the_recorded_sqlite_results(tmp_path: Path, vec: bool) -> None:
    """Valid queries keep exactly the results recorded before MP-A10 (FX-RANK-0)."""
    corpus = _ranking_module("make_rank_corpus")
    fixture: dict[str, Any] = corpus.load_fixture()
    measured = _ranking_module("measure_sqlite").measure(fixture, tmp_path, vec=vec)
    recorded = fixture["results"]["sqlite" if vec else "sqlite-novec"]
    assert measured["runs"] == recorded["runs"]
