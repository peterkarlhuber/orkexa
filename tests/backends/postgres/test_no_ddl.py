"""No DDL at runtime; a runtime role without DDL privilege is enough (MUP-ONA B5: T-DDL-1..3).

* **T-DDL-1** an event trigger logs every DDL command of the test database while the stores are built and used
  (index with vectors: add, every search mode, reindex, delete; file store; memory; ``verify_schema``): zero.
* **T-DDL-2** the restricted runtime role (no superuser, owns nothing, no ``CREATE`` on any schema, only
  ``SELECT``/``INSERT``/``UPDATE``/``DELETE`` on the five tables and the version table) passes construction,
  ``verify_schema``, writes, searches and reindex.
* **T-DDL-3** ``migrate`` with the runtime role fails with ``schema.mismatch`` (``reason="permission"``)
  before anything is changed; with the owner it works.

The CI jobs ``pg-a`` and ``pg-b`` also run every PostgreSQL test as the runtime role
(``ORKEXA_TEST_RUNTIME_ROLE=1``).
"""

from __future__ import annotations

import pytest

from orkexa.backends.postgres import (
    PostgresFileStore,
    PostgresIndex,
    PostgresMemory,
    PsycopgSource,
    migrate,
    verify_schema,
)
from orkexa.retrieval import RetrievalError
from orkexa.testing.conformance import KeywordEmbeddings

from . import support

pytestmark = pytest.mark.postgres


def workload(source: PsycopgSource, db: support.Database) -> None:
    """Everything a product does at run time, on ``source``."""
    report = verify_schema(source, db.layout)
    assert report.ok, report.summary()
    index = PostgresIndex(source, "ddl-a", layout=db.layout, embeddings=KeywordEmbeddings("ddl-1"))
    index.add("Refund policy for alpha customers.", source="a.md")
    index.add("Beta release notes.", source="b.md")
    for mode in ("fts", "vector", "hybrid"):
        assert index.retrieve("refund alpha", 5, mode=mode), mode  # type: ignore[arg-type]
    status = index.reindex(KeywordEmbeddings("ddl-2"))
    assert status.active is not None and status.active["model"] == "ddl-2"
    index.refresh()
    index.pgvector()
    assert index.delete("b.md") == 1
    files = PostgresFileStore(source, "ddl-a", layout=db.layout)
    stored = files.add(b"hello", name="h.txt")
    assert files.get_bytes(stored.id) == b"hello"
    files.delete(stored.id)
    memory = PostgresMemory(source, "ddl-a", layout=db.layout)
    memory.create("/memories/a.md", "one")
    memory.rename("/memories/a.md", "/memories/b.md")
    memory.delete("/memories/b.md")


# MUP-ONA T-DDL-1
def test_the_runtime_produces_no_ddl(db: support.Database) -> None:
    with support.ddl_log() as ddl:
        workload(db.source, db)
        assert ddl() == []


# MUP-ONA T-DDL-1
def test_the_event_trigger_sees_ddl() -> None:
    """The trigger of T-DDL-1 is live: a DDL statement during the window is logged."""
    with support.database() as db, support.ddl_log() as ddl:
        db.query("CREATE TABLE ddl_probe (x int)")
        assert ("CREATE TABLE", "postgres") in ddl() or [tag for tag, _ in ddl()] == ["CREATE TABLE"]


# MUP-ONA T-DDL-2
def test_a_role_without_ddl_privilege_is_enough() -> None:
    with support.database(runtime=True) as db:
        with support.admin() as admin:
            role = admin.execute(
                "SELECT rolsuper, rolcreatedb, rolcreaterole, rolbypassrls FROM pg_roles WHERE rolname = %s",
                (support.RUNTIME_ROLE,),
            ).fetchone()
            assert role == (False, False, False, False)
            can_create = admin.execute(
                "SELECT has_schema_privilege(%s, %s, 'CREATE'), has_schema_privilege(%s, 'public', 'CREATE')",
                (support.RUNTIME_ROLE, db.schema, support.RUNTIME_ROLE),
            ).fetchone()
            assert can_create == (False, False)
        with db.source.transaction("read") as connection:
            assert connection.execute("SELECT current_user").fetchall() == [(support.RUNTIME_ROLE,)]
        with support.ddl_log() as ddl:
            workload(db.source, db)
            assert ddl() == []


# MUP-ONA T-DDL-3
def test_migrate_needs_the_migration_role() -> None:
    with support.database(migrated=False) as db:
        db.grant()  # USAGE on the schema, nothing to create with
        runtime = db.runtime_source()
        try:
            with pytest.raises(RetrievalError) as raised:
                migrate(runtime, db.layout)
            assert (raised.value.code, raised.value.reason) == ("schema.mismatch", "permission")
            assert "migration or owner role" in str(raised.value)
            assert (
                db.query(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema = %s", (db.schema,)
                )
                == []
            )
            assert migrate(db.owner, db.layout) == 0  # the owner may
            db.grant()
            with pytest.raises(RetrievalError) as again:
                migrate(runtime, db.layout)  # even on a current schema: the role is checked first
            assert again.value.reason == "permission"
            assert verify_schema(runtime, db.layout).ok
        finally:
            runtime.close()
