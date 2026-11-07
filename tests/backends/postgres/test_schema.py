"""Schema version 1, :func:`verify_schema`, :func:`schema_steps` and :func:`migrate` (MUP-ONA B5: T-SCH-1..3).

* **T-SCH-1** ``verify_schema`` reports a missing table, column and index, a wrong column type (``double
  precision[]`` instead of ``real[]``) and whether pgvector is installed -- and changes nothing (catalog
  snapshot before and after, an event trigger sees no DDL). It never raises.
* **T-SCH-2** schema comparison: one schema migrated by Nodestro's migration files 0001-0010 (test data from
  nodestro-runtime 76f9cb2, checksums checked), another by ``migrate(PostgresLayout("nodestro_"))``; their
  tables, columns, types, indexes and constraints of the knowledge and memory tables are identical. With
  ``ORKEXA_SCHEMA_REPORTS`` set, the comparison is written to ``<dir>/schema-comparison.json`` (CI artifact).
* **T-SCH-3** ``migrate`` is explicit, idempotent, serialized, refuses a newer version and tables it did not
  create, and never installs an extension; ``schema_steps`` is the text it runs.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import threading
from typing import Any

import pytest

from orkexa.backends.postgres import (
    SCHEMA_VERSION,
    PostgresLayout,
    PsycopgSource,
    SchemaReport,
    migrate,
    schema_steps,
    verify_schema,
)
from orkexa.retrieval import RetrievalError

from . import support
from .test_guard import SpySource

MIGRATIONS = Path(__file__).resolve().parents[2] / "fixtures" / "postgres" / "nodestro_migrations"
#: nodestro-runtime tests/fixtures/migration_checksums.json at 76f9cb2 (MUP-ONA T-MIG-1): the copies are those
#: files.
NODESTRO_CHECKSUMS = {
    "0001_documents.sql": "aa3bf7c5b3a8cd52a59f5b09b15b47746263d977a2716038a42421b7c29d5f18",
    "0002_knowledge_memory.sql": "2838b1ea2b91bf1d460e22e2c1373d2a73078dda79fc41f68b62fd941ad03409",
    "0003_identity.sql": "bfe77a13b03b8823621a5258473d479eecd96e9359a87c52a90790547afb77a7",
    "0004_jobs.sql": "6bc223ed073628a2198a278f0a04150b5560795de3a95d49955c569824dccba7",
    "0005_find_indexes.sql": "ad75b8be04d242e435ea689cd0c58265e01f11a0a183b27c41fa6a43c343200b",
    "0006_partial_find_indexes.sql": "f5481069398d91860897e1e164276217be2d91cdff75c2c1c8f364216743f592",
    "0007_axentra_links.sql": "5cdbd4f63be6813a0ccd25c469e865aaaf3f73f5b5bece23402051fb6758b97e",
    "0008_tenant_status.sql": "afd8f83c39c8d82e1b91b099e5d737843008ecd1055077b580c921a46f53b3a8",
    "0009_created_order.sql": "bada36a3af623fb6b4a08e317c3e92cc4b275d8c56db6710c9bf6f2d4661e8ba",
    "0010_knowledge_vectors.sql": "aadda58e1c14e6654bca42394e058ae85ad44309b2f017d9ec2f15409a6732f5",
}
NODESTRO = PostgresLayout("nodestro_")
KNOWLEDGE_TABLES = sorted(NODESTRO.tables.values())
needs_psycopg = pytest.mark.skipif(importlib.util.find_spec("psycopg") is None, reason="needs psycopg")


def _kinds(report: SchemaReport) -> set[tuple[str, str, str]]:
    return {(problem.kind, problem.table, problem.name) for problem in report.problems}


# ---- without a server
# ---------------------------------------------------------------------------------------


@needs_psycopg
def test_schema_steps_are_the_tables_of_0002_and_0010_without_an_extension() -> None:
    steps = schema_steps(support.LAYOUT)
    assert len(steps) == 8
    assert [step.split(" (")[0] for step in steps] == [
        'CREATE TABLE "t_files"',
        'CREATE TABLE "t_chunks"',
        'CREATE INDEX "t_chunks_source" ON "t_chunks"',
        'CREATE INDEX "t_chunks_search" ON "t_chunks" USING gin',
        'CREATE TABLE "t_memory_files"',
        'CREATE TABLE "t_vector_spaces"',
        'CREATE TABLE "t_chunk_vectors"',
        'CREATE INDEX "t_chunk_vectors_chunk" ON "t_chunk_vectors"',
    ]
    text = " ".join(steps)
    assert "EXTENSION" not in text.upper() and " vector" not in text and "::vector" not in text
    assert "embedding real[] NOT NULL" in text
    assert "to_tsvector('simple', search_text)" in text
    assert schema_steps(NODESTRO)[0].startswith('CREATE TABLE "nodestro_files"')


# MUP-ONA T-SCH-1
def test_verify_schema_never_raises() -> None:
    """A source that fails, a missing server or a wrong argument is a problem of the kind ``error``."""

    class Failing(SpySource):
        def execute(self, query: Any, params: Any = None) -> Any:
            raise OSError("connection refused")

    for source, layout in ((Failing(), support.LAYOUT), (SpySource(), "t_")):
        report = verify_schema(source, layout)  # type: ignore[arg-type]
        assert not report.ok
        assert [problem.kind for problem in report.problems] == ["error"]
        assert "error" in report.summary()


def test_migrate_needs_a_layout() -> None:
    with pytest.raises(TypeError):
        migrate(SpySource(), "t_")  # type: ignore[arg-type]


# ---- T-SCH-1
# ------------------------------------------------------------------------------------------------


# MUP-ONA T-SCH-1
@pytest.mark.postgres
def test_verify_schema_accepts_a_migrated_schema(db: support.Database) -> None:
    report = verify_schema(db.source, db.layout)
    assert report.ok, report.summary()
    assert report.version == SCHEMA_VERSION
    assert report.schema == db.schema
    assert report.prefix == "t_"


# MUP-ONA T-SCH-1
@pytest.mark.postgres
def test_verify_schema_reports_pgvector(db: support.Database) -> None:
    report = verify_schema(db.source, db.layout)
    assert report.pgvector is support.has_pgvector()
    assert (report.pgvector_version is not None) is report.pgvector
    if support.PGVECTOR_REQUIRED:
        assert report.pgvector, "ORKEXA_REQUIRE_PGVECTOR=1 but verify_schema reports no pgvector"
    assert ("pgvector installed" in report.summary()) is report.pgvector


#: T-SCH-1: a change by an operator -> the problem verify_schema must report.
DAMAGE = {
    "missing table": ("DROP TABLE t_files", ("missing_table", "t_files", "")),
    "missing column": ("ALTER TABLE t_chunks DROP COLUMN title", ("missing_column", "t_chunks", "title")),
    "missing index": ("DROP INDEX t_chunks_search", ("missing_index", "t_chunks", "t_chunks_search")),
    "wrong type": (
        "ALTER TABLE t_chunk_vectors ALTER COLUMN embedding TYPE double precision[]",
        ("wrong_column", "t_chunk_vectors", "embedding"),
    ),
    "wrong nullability": (
        "ALTER TABLE t_memory_files ALTER COLUMN content DROP NOT NULL",
        ("wrong_column", "t_memory_files", "content"),
    ),
    "wrong index": (
        "DROP INDEX t_chunks_source; CREATE INDEX t_chunks_source ON t_chunks (source)",
        ("wrong_index", "t_chunks", "t_chunks_source"),
    ),
    "newer version": ("INSERT INTO t_orkexa_schema (version) VALUES (2)", ("version", "t_orkexa_schema", "")),
}


# MUP-ONA T-SCH-1
@pytest.mark.postgres
@pytest.mark.parametrize("damage", list(DAMAGE), ids=list(DAMAGE))
def test_verify_schema_reports_a_difference_and_changes_nothing(db: support.Database, damage: str) -> None:
    statement, expected = DAMAGE[damage]
    db.query(statement)
    before = support.catalog_snapshot(db.schema)
    with support.ddl_log() as ddl:
        report = verify_schema(db.source, db.layout)
        assert ddl() == []
    assert support.catalog_snapshot(db.schema) == before
    assert not report.ok
    assert expected in _kinds(report), report.summary()
    assert len(report.problems) == 1 or damage == "missing table", report.summary()


# MUP-ONA T-SCH-1
@pytest.mark.postgres
def test_verify_schema_of_an_empty_schema_reports_every_table(db: support.Database) -> None:
    report = verify_schema(db.source, PostgresLayout("empty_"))
    assert {problem.kind for problem in report.problems} == {"missing_table"}
    assert {problem.table for problem in report.problems} == set(PostgresLayout("empty_").tables.values())
    assert report.version is None and report.schema is None


# ---- T-SCH-2
# ------------------------------------------------------------------------------------------------


def _nodestro_migrations() -> list[tuple[str, str]]:
    files = sorted(MIGRATIONS.glob("*.sql"))
    sums = dict(
        reversed(line.split())
        for line in (MIGRATIONS / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    )
    assert [path.name for path in files] == sorted(NODESTRO_CHECKSUMS)
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == sums[path.name] == NODESTRO_CHECKSUMS[path.name], path.name
    return [(path.name, path.read_text(encoding="utf-8")) for path in files]


# MUP-ONA T-SCH-2
def test_the_nodestro_migrations_are_unchanged_copies() -> None:
    assert len(_nodestro_migrations()) == 10


def _knowledge(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {key: [row for row in rows if row[0] in KNOWLEDGE_TABLES] for key, rows in snapshot.items()}


# MUP-ONA T-SCH-2
@pytest.mark.postgres
def test_migrate_creates_what_nodestros_migrations_create() -> None:
    migrations = _nodestro_migrations()
    support.require_postgres()
    by_nodestro, by_orkexa = support.new_schema(), support.new_schema()
    for schema in (by_nodestro, by_orkexa):
        support.create_schema(schema)
    try:
        owner = PsycopgSource(support.conninfo(by_nodestro, runtime=False))
        try:
            for _name, text in migrations:  # as Nodestro's runner: one transaction per file
                with owner.transaction("transaction") as connection:
                    connection.execute(text)
            nodestro_report = verify_schema(owner, NODESTRO)
        finally:
            owner.close()
        owner = PsycopgSource(support.conninfo(by_orkexa, runtime=False))
        try:
            assert migrate(owner, NODESTRO) == 0
            orkexa_report = verify_schema(owner, NODESTRO)
        finally:
            owner.close()
        nodestro, orkexa = (_knowledge(support.catalog_snapshot(s)) for s in (by_nodestro, by_orkexa))
        folder = os.environ.get("ORKEXA_SCHEMA_REPORTS")
        if folder:
            Path(folder).mkdir(parents=True, exist_ok=True)
            (Path(folder) / "schema-comparison.json").write_text(
                json.dumps(
                    {
                        "test": "T-SCH-2 (tests/backends/postgres/test_schema.py)",
                        "nodestro_migrations": "nodestro-runtime 76f9cb2, 0001-0010",
                        "tables": KNOWLEDGE_TABLES,
                        "identical": nodestro == orkexa,
                        "nodestro": nodestro,
                        "orkexa": orkexa,
                    },
                    indent=1,
                    default=str,
                )
                + "\n",
                encoding="utf-8",
            )
        assert {row[0] for row in orkexa["columns"]} == set(KNOWLEDGE_TABLES)
        for key in ("columns", "indexes", "constraints"):
            assert orkexa[key] == nodestro[key], key
        assert len(orkexa["indexes"]) == 8 and len(orkexa["constraints"]) == 14
        assert nodestro_report.ok and nodestro_report.version is None, nodestro_report.summary()
        assert orkexa_report.ok and orkexa_report.version == 1, orkexa_report.summary()
    finally:
        for schema in (by_nodestro, by_orkexa):
            support.drop_schema(schema)


# ---- T-SCH-3: migrate
# ---------------------------------------------------------------------------------------


# MUP-ONA T-SCH-3
@pytest.mark.postgres
def test_migrate_is_idempotent_and_records_its_version(db: support.Database) -> None:
    before = support.catalog_snapshot(db.schema)
    assert migrate(db.owner, db.layout) == SCHEMA_VERSION
    assert support.catalog_snapshot(db.schema) == before
    assert db.query("SELECT version FROM t_orkexa_schema") == [(1,)]
    with support.admin() as connection:
        extensions = connection.execute("SELECT extname FROM pg_extension ORDER BY 1").fetchall()
    assert {name for (name,) in extensions} <= {"plpgsql", "vector"}  # vector only from the server setup


# MUP-ONA T-SCH-3
@pytest.mark.postgres
def test_migrate_refuses_a_newer_version(db: support.Database) -> None:
    db.query("INSERT INTO t_orkexa_schema (version) VALUES (2)")
    with pytest.raises(RetrievalError) as raised:
        migrate(db.owner, db.layout)
    assert (raised.value.code, raised.value.reason) == ("schema.mismatch", "too_new")


# MUP-ONA T-SCH-3
@pytest.mark.postgres
def test_migrate_refuses_tables_it_did_not_create() -> None:
    with support.database(migrated=False) as db:
        db.query('CREATE TABLE t_files (tenant_id text COLLATE "C" NOT NULL)')
        with pytest.raises(RetrievalError) as raised:
            migrate(db.owner, db.layout)
        assert (raised.value.code, raised.value.reason) == ("schema.mismatch", "unrecorded")
        assert "t_files" in str(raised.value)
        tables = db.query(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = %s", (db.schema,)
        )
        assert tables == [("t_files",)]  # rolled back: no version table either


# MUP-ONA T-SCH-3
@pytest.mark.postgres
def test_concurrent_migrations_apply_once() -> None:
    with support.database(migrated=False) as db:
        results: list[Any] = []

        def run() -> None:
            source = PsycopgSource(support.conninfo(db.schema, runtime=False))
            try:
                results.append(migrate(source, db.layout))
            except Exception as exc:  # reported below
                results.append(exc)
            finally:
                source.close()

        threads = [threading.Thread(target=run) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(60)
        assert sorted(results) == [0, 1, 1, 1], results
        assert verify_schema(db.owner, db.layout).ok


# MUP-ONA T-SCH-3
@pytest.mark.postgres
def test_two_layouts_share_one_schema() -> None:
    with support.database() as db:
        other = PostgresLayout("other_")
        assert migrate(db.owner, other) == 0
        assert verify_schema(db.owner, other).ok and verify_schema(db.owner, db.layout).ok
