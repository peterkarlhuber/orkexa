"""The partition guard of :mod:`orkexa.backends.postgres` (MUP-ONA B5: T-GUARD-1..3) and the connection
source.

* **T-GUARD-1..3** the executor refuses a partitioned statement whose ``tenant_id`` is missing, ``None``,
  empty, whitespace, not a string, 10,000 characters long or another tenant with
  ``RetrievalError("partition.missing")`` -- for every partitioned catalog entry, and through the index, file
  store and memory -- and a spy source proves that no transaction was opened (no database access).
* The cases on a real server (marker ``postgres``) check :class:`~orkexa.backends.postgres.PsycopgSource`: its
  modes, joining, rollback (T-PG-TX), the pool limit, broken connections and the transaction-local tenant
  setting (the hook for row-level security).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import json
import threading
from typing import Any

import pytest

from orkexa.backends.postgres import (
    ConnectionSource,
    PostgresFileStore,
    PostgresIndex,
    PostgresLayout,
    PostgresMemory,
    PsycopgSource,
)
from orkexa.backends.postgres._sql import CATALOG, MAX_TENANT_CHARS, Executor, jsonb_text, render
from orkexa.retrieval import RetrievalError

from . import support

TENANT = "tenant-a"
LAYOUT = PostgresLayout("t_")
PARTITIONED = sorted(name for name, entry in CATALOG.items() if entry.partitioned)


class SpySource:
    """A ``ConnectionSource`` that records every transaction and fails the test if a statement runs."""

    def __init__(self) -> None:
        self.opened: list[tuple[str, str | None]] = []

    @contextmanager
    def transaction(self, mode: str = "transaction", *, tenant_id: str | None = None) -> Iterator[Any]:
        self.opened.append((mode, tenant_id))
        yield self

    def execute(self, query: Any, params: Any = None) -> Any:
        raise AssertionError(f"a statement reached the database: {query!r}")


def _params(name: str, **override: Any) -> dict[str, Any]:
    params: dict[str, Any] = dict.fromkeys(CATALOG[name].params, "x")
    params["tenant_id"] = TENANT
    params.update(override)
    return params


#: Tenant values the guard refuses (T-GUARD-1).
REFUSED: dict[str, Any] = {
    "empty": "",
    "none": None,
    "whitespace": " \t\n",
    "10000 characters": "a" * 10_000,
    "129 characters": "a" * (MAX_TENANT_CHARS + 1),
    "foreign tenant": "tenant-b",
    "padded own tenant": f" {TENANT}",
    "NUL character": f"{TENANT}\x00",  # PostgreSQL text cannot hold it (Orkexa addition to B4's guard)
    "number": 1,
    "list": [TENANT],
}


# MUP-ONA T-GUARD-1
# MUP-ONA T-GUARD-2
# MUP-ONA T-GUARD-3
@pytest.mark.parametrize("value", list(REFUSED.values()), ids=list(REFUSED))
@pytest.mark.parametrize("name", PARTITIONED)
def test_the_guard_refuses_an_unbound_tenant_before_any_database_access(name: str, value: Any) -> None:
    spy = SpySource()
    executor = Executor(spy, layout=LAYOUT, tenant_id=TENANT)
    with pytest.raises(RetrievalError) as raised:
        executor.execute(name, _params(name, tenant_id=value))
    assert raised.value.code == "partition.missing"
    assert spy.opened == []


# MUP-ONA T-GUARD-1
@pytest.mark.parametrize("name", PARTITIONED)
def test_the_guard_refuses_a_missing_tenant(name: str) -> None:
    spy = SpySource()
    params = _params(name)
    del params["tenant_id"]
    with pytest.raises(RetrievalError) as raised:
        Executor(spy, layout=LAYOUT, tenant_id=TENANT).execute(name, params)
    assert raised.value.code == "partition.missing"
    assert spy.opened == []


def test_the_guard_also_holds_inside_a_transaction() -> None:
    spy = SpySource()
    executor = Executor(spy, layout=LAYOUT, tenant_id=TENANT)
    with executor.transaction(), pytest.raises(RetrievalError):
        executor.execute("chunks_count", _params("chunks_count", tenant_id="tenant-b"))
    assert spy.opened == [("transaction", TENANT)]  # the block's own transaction; the statement never ran


def test_the_instance_tenant_must_pass_the_guard_too() -> None:
    for value in REFUSED.values():
        if value in ("tenant-b", f" {TENANT}"):
            continue  # valid tenants of their own
        with pytest.raises(ValueError):
            Executor(SpySource(), layout=LAYOUT, tenant_id=value)
        for kind in (PostgresIndex, PostgresFileStore, PostgresMemory):
            with pytest.raises(ValueError):
                kind(SpySource(), value, layout=LAYOUT)


def test_only_catalog_names_with_their_parameters_run() -> None:
    spy = SpySource()
    executor = Executor(spy, layout=LAYOUT, tenant_id=TENANT)
    with pytest.raises(ValueError, match="Unknown statement"):
        executor.execute("SELECT * FROM t_chunks", {"tenant_id": TENANT})
    with pytest.raises(ValueError, match="parameters"):
        executor.execute("files_entry", _params("files_entry", extra=1))
    params = _params("files_entry")
    del params["id"]
    with pytest.raises(ValueError, match="parameters"):
        executor.execute("files_entry", params)
    with pytest.raises(ValueError, match="Unknown statement"):
        render("nope", "t_")
    assert spy.opened == []


def _stores(source: Any) -> list[Any]:
    return [
        PostgresIndex(source, TENANT, layout=LAYOUT, namespace="n"),
        PostgresFileStore(source, TENANT, layout=LAYOUT, namespace="n"),
        PostgresMemory(source, TENANT, layout=LAYOUT, namespace="n"),
    ]


# MUP-ONA T-GUARD-1
@pytest.mark.parametrize("value", ["tenant-b", "", None, " ", "a" * 10_000])
def test_a_store_whose_tenant_was_changed_reaches_no_row(value: Any) -> None:
    """The stores bind ``self.tenant_id``; the executor keeps the tenant they were built for. Changing the
    attribute afterwards (a bug, or an attack on the object) is caught by the guard, not sent to the
    database."""
    spy = SpySource()
    index, files, memory = _stores(spy)
    calls = [
        lambda: len(index),
        lambda: index.sources(),
        lambda: index.delete("s"),
        lambda: index.retrieve("alpha", mode="fts"),
        lambda: files._entries(),
        lambda: files._remove("f"),
        lambda: memory._read("/memories/a"),
        lambda: memory.clear(),
    ]
    for store in (index, files, memory):
        store.tenant_id = value
    for call in calls:
        with pytest.raises(RetrievalError) as raised:
            call()
        assert raised.value.code == "partition.missing"
    assert spy.opened == []


def test_jsonb_text_replaces_only_escaped_nul() -> None:
    assert jsonb_text('{"a": 1}') == '{"a": 1}'
    cleaned = jsonb_text('{"a": "x\\u0000y", "\\u0000k": ["\\u0000"]}')
    assert "\\u0000" not in cleaned
    assert json.loads(cleaned) == {"a": "x\ufffdy", "\ufffdk": ["\ufffd"]}
    assert jsonb_text('"plain"') == '"plain"'


def test_a_source_is_a_connection_source() -> None:
    assert isinstance(SpySource(), ConnectionSource)


# ---- PsycopgSource on a real server
# -------------------------------------------------------------------------


@pytest.mark.postgres
def test_psycopg_source_runs_each_mode(db: support.Database) -> None:
    source = db.source
    assert isinstance(source, ConnectionSource)
    executor = Executor(source, layout=LAYOUT, tenant_id=TENANT)
    key = {"tenant_id": TENANT, "namespace": "n"}
    assert executor.execute("memory_write", {**key, "path": "a", "content": "x"}).rowcount == 1
    with executor.transaction():
        executor.execute("memory_write", {**key, "path": "b", "content": "y"})
        assert executor.execute("memory_files", {**key, "start": ""}).rows == [("a", 1), ("b", 1)]
    with pytest.raises(ValueError), source.transaction("bulk"):  # type: ignore[arg-type]
        pass


# MUP-ONA T-PG-TX
@pytest.mark.postgres
def test_a_failed_transaction_rolls_back_and_the_next_operation_is_clean(db: support.Database) -> None:
    """A forced error after the first statement rolls the transaction back; the next operation works."""
    executor = Executor(db.source, layout=LAYOUT, tenant_id=TENANT)
    key = {"tenant_id": TENANT, "namespace": "n"}
    executor.execute("memory_write", {**key, "path": "keep", "content": "x"})
    with pytest.raises(RuntimeError), executor.transaction():
        executor.execute("memory_clear", key)
        executor.execute("memory_write", {**key, "path": "new", "content": "y"})
        raise RuntimeError("forced")
    assert executor.execute("memory_files", {**key, "start": ""}).rows == [("keep", 1)]
    with pytest.raises(Exception), executor.transaction():  # noqa: B017 - the server's own error
        executor.execute("memory_clear", key)
        executor.execute("files_upsert", {**key, "id": "f", "entry": "not json", "data": b""})
    assert executor.execute("memory_files", {**key, "start": ""}).rows == [("keep", 1)]
    memory = PostgresMemory(db.source, TENANT, layout=LAYOUT, namespace="n")
    memory.create("/memories/after.md", "fine")
    assert "fine" in memory.view("/memories/after.md")


@pytest.mark.postgres
def test_a_transaction_of_the_source_is_joined_by_the_stores(db: support.Database) -> None:
    """Commands of several stores in one ``transaction`` block of the source are one transaction."""
    files = PostgresMemory(db.source, TENANT, layout=LAYOUT, namespace="a")
    other = PostgresMemory(db.source, TENANT, layout=LAYOUT, namespace="b")
    with pytest.raises(RuntimeError), db.source.transaction("transaction", tenant_id=TENANT):
        files.create("/memories/one.md", "1")
        other.create("/memories/two.md", "2")
        raise RuntimeError("roll back both")
    assert files._files("") == {} and other._files("") == {}
    with db.source.transaction("transaction", tenant_id=TENANT):
        files.create("/memories/one.md", "1")
        other.create("/memories/two.md", "2")
    assert files._files("") == {"one.md": 1} and other._files("") == {"two.md": 1}


@pytest.mark.postgres
def test_the_pool_limits_connections_and_waits(db: support.Database) -> None:
    source = PsycopgSource(support.conninfo(db.schema, runtime=False), max_connections=2, timeout=0.5)
    try:
        held = threading.Event()
        release = threading.Event()

        def hold() -> None:
            with source.transaction("transaction"):
                held.set()
                release.wait(10)

        threads = [threading.Thread(target=hold) for _ in range(2)]
        for thread in threads:
            thread.start()
            held.wait(10)
            held.clear()
        with pytest.raises(TimeoutError), source.transaction("read"):
            pass
        release.set()
        for thread in threads:
            thread.join(10)
        with source.transaction("read") as connection:
            assert connection.execute("SELECT 1").fetchall() == [(1,)]
        assert source._open <= 2
    finally:
        source.close()
    with pytest.raises(RuntimeError, match="closed"), source.transaction("read"):
        pass


@pytest.mark.postgres
def test_a_broken_connection_is_not_reused(db: support.Database) -> None:
    source = PsycopgSource(support.conninfo(db.schema, runtime=False), max_connections=1)
    try:
        with source.transaction("read") as connection:
            pid = connection.execute("SELECT pg_backend_pid()").fetchall()[0][0]
        with support.admin() as admin:
            admin.execute("SELECT pg_terminate_backend(%s)", (pid,))
        with pytest.raises(Exception), source.transaction("read") as connection:  # noqa: B017 - psycopg's error
            connection.execute("SELECT 1")
        with source.transaction("read") as connection:
            assert connection.execute("SELECT pg_backend_pid()").fetchall()[0][0] != pid
    finally:
        source.close()


@pytest.mark.postgres
def test_the_tenant_setting_is_off_by_default_and_transaction_local(db: support.Database) -> None:
    conninfo = support.conninfo(db.schema, runtime=False)
    probe = "SELECT current_setting('app.tenant_id', true)"
    plain = PsycopgSource(conninfo)
    hooked = PsycopgSource(conninfo, tenant_setting="app.tenant_id", max_connections=1)
    try:
        with plain.transaction("read", tenant_id=TENANT) as connection:
            assert connection.execute(probe).fetchall() in ([(None,)], [("",)])
        for mode in ("read", "write", "transaction"):
            with hooked.transaction(mode, tenant_id=TENANT) as connection:  # type: ignore[arg-type]
                assert connection.execute(probe).fetchall() == [(TENANT,)]
        # The setting ended with its transaction: the (only) pooled connection of ``hooked`` no longer has it.
        assert len(hooked._idle) == 1
        assert hooked._idle[0].execute(probe).fetchall() in ([(None,)], [("",)])
    finally:
        plain.close()
        hooked.close()
    for bad in ("x'; DROP", "app", "App.tenant", "app.tenant id"):
        with pytest.raises(ValueError):
            PsycopgSource(conninfo, tenant_setting=bad)
    with pytest.raises(ValueError):
        PsycopgSource(conninfo, max_connections=0)
