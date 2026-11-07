"""Concurrency of the PostgreSQL backend over one pool (MUP-ONA B5: T-PG-6, T-PART-7/T-PART-CC).

* **T-PG-6** two tenants with eight threads each over one :class:`~orkexa.backends.postgres.PsycopgSource`
  (fewer connections than threads, so threads wait for the pool): writes, searches in every mode and deletes;
  exact counts and no row of the other tenant.
* **T-PART-CC** (plan 10.4) three tenants with eight threads each over one pool, mixed writes, searches,
  deletes and reindexes for 60 seconds (``ORKEXA_TEST_PART_SECONDS`` changes the duration), with vectors
  ranked by the database (pgvector) and in Python; exact counts, no marker leak. Taken over from
  nodestro-runtime ``tests/test_pg_partitions.py`` (76f9cb2) with Orkexa's classes, as are the two reindex
  cases under steady writes.
* A chunk deleted between a vector store's read of it and the foreign key check (``READ COMMITTED``) fails the
  store; the reindex stores again without it, in a round and in the switch. T-PART-CC hit this race in about
  one of four 60-second runs; the case here makes it deterministic.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
import os
import threading
import time
from typing import Any

import pytest

from orkexa.backends.postgres import PostgresIndex
from orkexa.backends.postgres import index as pg_index
from orkexa.retrieval import RetrievalError
from orkexa.testing.conformance import KeywordEmbeddings

from . import support

pytestmark = pytest.mark.postgres

TENANTS = ("part-a", "part-b", "part-c")
MARKER = "zebramarker"
THREADS = 8


@pytest.fixture
def pooled() -> Iterator[support.Database]:
    """A test schema whose source has fewer connections than the threads of a test."""
    with support.database(max_connections=6, timeout=120.0) as opened:
        yield opened


def _require_ranking(db: support.Database, ranking: str) -> None:
    if ranking == "database" and not PostgresIndex(db.source, TENANTS[0], layout=db.layout).pgvector():
        support.require_pgvector("the database ranks vectors only with pgvector")


# MUP-ONA T-PG-6
def test_two_tenants_with_eight_threads_each_over_one_pool(pooled: support.Database) -> None:
    db = pooled
    model = KeywordEmbeddings("pg6")
    rounds = 12
    errors: list[str] = []
    written: dict[str, set[str]] = {"pg6-a": set(), "pg6-b": set()}
    lock = threading.Lock()

    def make(tenant: str) -> PostgresIndex:
        return PostgresIndex(db.source, tenant, layout=db.layout, namespace="pg6", embeddings=model)

    for tenant in written:
        make(tenant).add(f"Seed note for {tenant} about alpha.", source="seed.md")

    def work(tenant: str, worker: int) -> None:
        try:
            for number in range(rounds):
                source = f"w{worker}-{number}.md"
                make(tenant).add(f"{tenant} worker {worker} note {number} about alpha.", source=source)
                for mode in ("fts", "vector", "hybrid"):
                    hits = make(tenant).retrieve("alpha", 50, mode=mode)  # type: ignore[arg-type]
                    foreign = [hit.text for hit in hits if tenant not in hit.text]
                    if foreign:
                        errors.append(f"{tenant}/{worker} {mode}: foreign rows {foreign[:2]}")
                if number % 3 == 2:
                    if make(tenant).delete(source) != 1:
                        errors.append(f"{tenant}/{worker}: delete of {source} failed")
                else:
                    with lock:
                        written[tenant].add(source)
        except BaseException as exc:  # reported below
            errors.append(f"{tenant}/{worker}: {type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=work, args=(tenant, n)) for tenant in written for n in range(THREADS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(600)
    assert not any(thread.is_alive() for thread in threads)
    assert errors == []
    assert db.source._open <= 6
    for tenant, sources in written.items():
        index = make(tenant)
        assert set(index.sources()) == sources | {"seed.md"}
        status = index.vector_status()
        assert status.chunks == status.embedded == len(sources) + 1


class _Models:
    """The embedding model the tenants' vectors currently belong to (a reindex switches it)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.current = {tenant: KeywordEmbeddings("part-x") for tenant in TENANTS}

    def get(self, tenant: str) -> KeywordEmbeddings:
        with self.lock:
            return self.current[tenant]

    def switch(self, tenant: str, model: KeywordEmbeddings) -> None:
        with self.lock:
            self.current[tenant] = model


# MUP-ONA T-PART-7
@pytest.mark.parametrize("ranking", ["database", "runtime"])
def test_three_tenants_with_eight_threads_each(pooled: support.Database, ranking: str) -> None:
    # MUP-ONA T-PART-CC
    database = pooled
    _require_ranking(database, ranking)
    seconds = float(os.environ.get("ORKEXA_TEST_PART_SECONDS", "60"))
    models = _Models()

    def make(tenant: str) -> PostgresIndex:
        index = PostgresIndex(
            database.source, tenant, layout=database.layout, namespace="cc", embeddings=models.get(tenant)
        )
        if ranking == "runtime":
            index._pgvector = False
        return index

    # The same sources and texts in every tenant (the same chunk texts, sources and space keys); B and C hold
    # a marker.
    for tenant in TENANTS:
        index = make(tenant)
        for number in range(4):
            index.add(f"Base note {number} about alpha policy.", source=f"base-{number}.md")
        if tenant != "part-a":
            index.add(f"The {MARKER} note about alpha policy.", source="marker.md")
    kept: dict[str, set[str]] = {tenant: set() for tenant in TENANTS}
    counts = {"writes": 0, "searches": 0, "deletes": 0, "reindexes": 0, "retries": 0}
    lock = threading.Lock()
    errors: list[str] = []
    deadline = time.monotonic() + seconds

    def current(tenant: str, action: Callable[[PostgresIndex], Any]) -> Any:
        """``action`` on an index with the tenant's current model; a reindex that switched the model meanwhile
        makes a write or search with the old one raise ``reindex_required`` (never a fallback): then again
        with the new."""
        for _attempt in range(20):
            try:
                return action(make(tenant))
            except RetrievalError as exc:
                if exc.code != "reindex_required":
                    raise
                with lock:
                    counts["retries"] += 1
        raise AssertionError(f"{tenant}: still reindex_required after 20 attempts")

    def add(tenant: str, text: str, source: str) -> None:
        current(tenant, lambda index: index.add(text, source=source))

    def reindex(tenant: str, generation: int) -> None:
        model = KeywordEmbeddings(f"part-{tenant}-{generation}")
        for _attempt in range(20):
            try:
                make(tenant).reindex(model, batch_size=8)
                models.switch(tenant, model)
                with lock:
                    counts["reindexes"] += 1
                return
            except RetrievalError as exc:  # chunks added meanwhile: continue the same reindex
                if exc.code != "vector_incomplete":
                    raise
        raise AssertionError(f"{tenant}: reindex {generation} did not complete")

    def work(tenant: str, worker: int) -> None:
        number = 0
        try:
            while time.monotonic() < deadline:
                source = f"w{worker}-{number}.md"
                add(tenant, f"Worker {worker} note {number} about beta policy.", source)
                with lock:
                    counts["writes"] += 1
                    kept[tenant].add(source)
                for mode in ("fts", "vector", "hybrid"):
                    hits = current(
                        tenant,
                        lambda index, mode=mode: index.retrieve(f"alpha policy {MARKER}", 50, mode=mode),
                    )
                    marker = [hit.source for hit in hits if hit.source == "marker.md" or MARKER in hit.text]
                    if tenant == "part-a" and marker:
                        errors.append(f"{tenant} thread {worker} {mode}: marker visible {marker}")
                with lock:
                    counts["searches"] += 3
                if number % 2 == 1:
                    deleted = make(tenant).delete(source)
                    if deleted != 1:
                        errors.append(f"{tenant} thread {worker}: delete of {source} gave {deleted}")
                    with lock:
                        counts["deletes"] += 1
                        kept[tenant].discard(source)
                if worker == 0 and number % 5 == 4:
                    reindex(tenant, number)
                number += 1
        except BaseException as exc:  # reported below
            errors.append(f"{tenant} thread {worker}: {type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=work, args=(tenant, n)) for tenant in TENANTS for n in range(THREADS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(seconds + 300)
    assert not any(thread.is_alive() for thread in threads)
    assert errors == []
    assert counts["writes"] >= len(threads) and counts["reindexes"] >= len(TENANTS), counts

    for tenant in TENANTS:
        index = make(tenant)
        own = {f"base-{number}.md" for number in range(4)} | ({"marker.md"} if tenant != "part-a" else set())
        expected = own | kept[tenant]
        assert set(index.sources()) == expected, tenant
        assert len(index) == len(expected), tenant
        status = index.reindex(
            models.get(tenant)
        )  # completes vectors of chunks written during the last switch
        assert (status.chunks, status.embedded, status.excluded) == (len(expected), len(expected), 0), status
        assert status.active is not None and status.active["model"] == models.get(tenant).name
        found = {hit.source for hit in index.retrieve(MARKER, 50, mode="hybrid")}
        assert ("marker.md" in found) is (tenant != "part-a"), (tenant, found)
    stray = database.query(
        "SELECT count(*) FROM t_chunk_vectors v JOIN t_chunks c ON c.id = v.chunk_id"
        " WHERE v.tenant_id <> c.tenant_id OR v.namespace <> c.namespace"
    )
    assert stray == [(0,)]
    spaces = database.query(
        "SELECT tenant_id, state, count(*) FROM t_vector_spaces GROUP BY 1, 2 ORDER BY 1, 2"
    )
    assert spaces == [(tenant, "active", 1) for tenant in TENANTS]


# MUP-ONA T-PART-7
@pytest.mark.parametrize("ranking", ["database", "runtime"])
def test_a_reindex_finishes_while_writers_keep_adding(db: support.Database, ranking: str) -> None:
    """Every embedding request of the reindex lets a writer add a chunk (with the old model, switching to the
    new one on ``reindex_required``, as the knowledge base does). Before, a chunk added between the last batch
    and the switch failed the switch (``vector_incomplete``) every time, so a reindex under steady writes
    never finished (seen in T-PART-CC). Now the switch holds the namespace's lock that writes take: one call
    finishes, writes made meanwhile wait or land in the new space, and nothing is left without a vector."""
    _require_ranking(db, ranking)
    old, new = KeywordEmbeddings("steady-old"), KeywordEmbeddings("steady-new")

    def make(model: KeywordEmbeddings | None) -> PostgresIndex:
        index = PostgresIndex(db.source, "part-a", layout=db.layout, namespace="steady", embeddings=model)
        if ranking == "runtime":
            index._pgvector = False
        return index

    for number in range(20):
        make(old).add(f"Base note {number} about alpha.", source=f"base-{number}.md")
    writers: list[threading.Thread] = []
    errors: list[str] = []
    waited: list[bool] = []

    def write(number: int) -> None:
        try:
            for model in (old, new):
                try:
                    make(model).add(f"Steady note {number} about beta.", source=f"steady-{number}.md")
                    return
                except RetrievalError as exc:
                    if exc.code != "reindex_required":
                        raise
            errors.append(f"steady-{number}: not written")
        except BaseException as exc:  # reported below
            errors.append(f"steady-{number}: {type(exc).__name__}: {exc}")

    reindexing = threading.current_thread()

    def on_embed(_texts: Any) -> None:
        if threading.current_thread() is not reindexing:
            return  # a writer embedding with the new model after the switch
        writer = threading.Thread(target=write, args=(len(writers),))
        writers.append(writer)
        writer.start()
        writer.join(0.5)  # done at once, unless the switch holds the lock: then it waits for the switch
        waited.append(writer.is_alive())

    new.on_embed = on_embed
    status = make(None).reindex(new, batch_size=4)  # one call: no vector_incomplete
    new.on_embed = None
    for writer in writers:
        writer.join(30)
    assert errors == []
    assert status.active is not None and status.active["model"] == "steady-new" and status.building is None
    assert waited[-1] and not any(waited[:-1]), waited  # only the switch's own batch made a writer wait
    final = make(new).vector_status()
    assert final.chunks == 20 + len(writers) and final.embedded == final.chunks, final
    assert {hit.source for hit in make(new).retrieve("steady beta", 100, mode="vector")} >= {
        f"steady-{number}.md" for number in range(len(writers))
    }


# MUP-ONA T-PART-7
@pytest.mark.parametrize("ranking", ["database", "runtime"])
@pytest.mark.parametrize("switch_chunks", [256, 16])
def test_the_switch_embeds_what_the_last_round_left(
    db: support.Database, ranking: str, switch_chunks: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Writers add more than a batch during every unlocked round (five chunks per request of four), so the
    rounds never get below one batch. Before, the switch embedded at most one batch and failed every attempt
    with ``vector_incomplete`` (T-PART-CC under CI load). Now it embeds the rest while writes wait, up to
    ``_REINDEX_SWITCH_CHUNKS``; above that, writes outpace the model and the reindex still says so."""
    _require_ranking(db, ranking)
    monkeypatch.setattr(pg_index, "_REINDEX_SWITCH_CHUNKS", switch_chunks)
    old, new = KeywordEmbeddings("burst-old"), KeywordEmbeddings("burst-new")

    def make(model: KeywordEmbeddings | None) -> PostgresIndex:
        index = PostgresIndex(
            db.source, "part-b", layout=db.layout, namespace=f"burst-{switch_chunks}", embeddings=model
        )
        if ranking == "runtime":
            index._pgvector = False
        return index

    for number in range(20):
        make(old).add(f"Base note {number} about alpha.", source=f"base-{number}.md")
    writers: list[threading.Thread] = []
    errors: list[str] = []
    added = [0]

    def write(first: int) -> None:
        try:
            for number in range(first, first + 5):
                for model in (old, new):
                    try:
                        make(model).add(f"Burst note {number} about beta.", source=f"burst-{number}.md")
                        break
                    except RetrievalError as exc:
                        if exc.code != "reindex_required":
                            raise
        except BaseException as exc:  # reported below
            errors.append(f"burst-{first}: {type(exc).__name__}: {exc}")

    reindexing = threading.current_thread()

    def on_embed(_texts: Any) -> None:
        if threading.current_thread() is not reindexing:
            return
        writer = threading.Thread(target=write, args=(added[0],))
        added[0] += 5
        writers.append(writer)
        writer.start()
        writer.join(0.05)  # done at once while unlocked; at the switch it waits, and adds after it

    new.on_embed = on_embed
    try:
        if switch_chunks < 20:
            with pytest.raises(RetrievalError) as raised:
                make(None).reindex(new, batch_size=4)
            assert raised.value.code == "vector_incomplete"
        else:
            status = make(None).reindex(new, batch_size=4)  # one call
            assert status.active is not None and status.active["model"] == "burst-new", status
    finally:
        new.on_embed = None
        for writer in writers:
            writer.join(30)
    assert errors == []
    if switch_chunks >= 20:
        final = make(new).vector_status()
        assert final.chunks == 20 + added[0] and final.embedded == final.chunks, final


class _Vanished(Exception):
    """A foreign key violation as psycopg raises it (``psycopg.errors.ForeignKeyViolation``)."""

    sqlstate = "23503"


class _VanishingSource:
    """Wraps a source: the first ``failures`` vector stores in a reindex round (``in_switch=False``) or in the
    switch (``in_switch=True``, the transaction that locked the spaces ``FOR UPDATE``) fail as if their chunk
    had been deleted meanwhile."""

    def __init__(self, inner: Any, *, in_switch: bool, failures: int = 1) -> None:
        self.inner = inner
        self.lock = inner.lock
        self.in_switch = in_switch
        self.failures = failures
        self.failed = 0

    @contextmanager
    def transaction(self, mode: Any = "transaction", *, tenant_id: str | None = None) -> Iterator[Any]:
        with self.inner.transaction(mode, tenant_id=tenant_id) as connection:
            yield _VanishingConnection(self, connection)


class _VanishingConnection:
    def __init__(self, source: _VanishingSource, connection: Any) -> None:
        self.source = source
        self.connection = connection
        self.locked = False

    def execute(self, query: Any, params: Any = None) -> Any:
        text = query.as_string(self.connection)
        self.locked = self.locked or "FOR UPDATE" in text
        store = "INSERT INTO" in text and "SELECT c.tenant_id" in text
        if store and self.locked == self.source.in_switch and self.source.failed < self.source.failures:
            self.source.failed += 1
            raise _Vanished("insert or update violates foreign key constraint (chunk deleted meanwhile)")
        return self.connection.execute(query, params)


# MUP-ONA T-PART-7
@pytest.mark.parametrize("where", ["round", "switch"])
def test_a_chunk_deleted_under_a_vector_store_repeats_the_store(db: support.Database, where: str) -> None:
    psycopg_errors = pytest.importorskip("psycopg.errors")
    assert psycopg_errors.ForeignKeyViolation.sqlstate == _Vanished.sqlstate
    source = _VanishingSource(db.source, in_switch=where == "switch")
    index = PostgresIndex(source, "vanish", layout=db.layout, namespace="vanish")
    for number in range(4):
        index.add(f"Chunk {number} about alpha.", source=f"c{number}.md")
    plain = PostgresIndex(db.source, "vanish", layout=db.layout, namespace="vanish")
    late: list[list[int]] = []

    def add_late(_texts: Any) -> None:  # after the round's high-water mark: left for the switch
        if not late:
            late.append(plain.add("A late chunk about alpha.", source="late.md"))

    status = index.reindex(KeywordEmbeddings("vanish", on_embed=add_late), batch_size=4)
    assert source.failed == 1 and late
    assert (status.chunks, status.embedded, status.building, status.complete) == (5, 5, None, True)


# MUP-ONA T-PART-7
def test_a_store_that_keeps_failing_ends_the_reindex(db: support.Database) -> None:
    source = _VanishingSource(db.source, in_switch=False, failures=pg_index._STORE_ATTEMPTS)
    index = PostgresIndex(source, "vanish", layout=db.layout, namespace="vanish")
    index.add("Chunk about alpha.", source="c.md")
    with pytest.raises(_Vanished):
        index.reindex(KeywordEmbeddings("vanish"), batch_size=4)
    assert source.failed == pg_index._STORE_ATTEMPTS
    assert index.vector_status().active is None
