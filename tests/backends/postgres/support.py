"""PostgreSQL for the tests of ``orkexa.backends.postgres`` (MUP-ONA B5).

``ORKEXA_TEST_POSTGRES`` names the test database (a libpq connection string of a superuser, the "owner" that
creates schemas, roles and event triggers and runs :func:`~orkexa.backends.postgres.migrate`). Without it the
PostgreSQL tests (marker ``postgres``) skip, unless ``ORKEXA_REQUIRE_POSTGRES=1`` (CI jobs ``pg-a``/``pg-b``),
which turns a missing server into a failure. ``ORKEXA_REQUIRE_PGVECTOR=1`` (job ``pg-b``) fails every test
that needs pgvector instead of skipping it.

Every test gets its own schema (dropped afterwards); its connections use ``search_path=<schema>, public``
(pgvector's type lives in ``public``). The first use sets the server up as a superuser would: the restricted
role :data:`RUNTIME_ROLE` (``LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS``, owns nothing, no
``CREATE`` anywhere) and, where the server has it, ``CREATE EXTENSION vector`` in ``public``.

``ORKEXA_TEST_RUNTIME_ROLE=1`` runs every store, search and ``verify_schema`` as a product's runtime does: as
:data:`RUNTIME_ROLE` with ``SELECT``, ``INSERT``, ``UPDATE`` and ``DELETE`` on the tables of the test schema
only; ``migrate`` still runs as the owner.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import os
import threading
from typing import Any
import uuid

import pytest

from orkexa.backends.postgres import PostgresLayout, PsycopgSource, migrate

DSN = os.environ.get("ORKEXA_TEST_POSTGRES", "")
REQUIRED = os.environ.get("ORKEXA_REQUIRE_POSTGRES") == "1"
PGVECTOR_REQUIRED = os.environ.get("ORKEXA_REQUIRE_PGVECTOR") == "1"
RUNTIME_ROLE_MODE = os.environ.get("ORKEXA_TEST_RUNTIME_ROLE") == "1"
RUNTIME_ROLE = "orkexa_test_runtime"
RUNTIME_PASSWORD = "orkexa-test-runtime-pw"
#: The layout of the tests (MUP-ONA: catalog tests with the prefix ``t_``).
LAYOUT = PostgresLayout("t_")

_setup_lock = threading.Lock()
_setup_done: list[bool] = []


def require_postgres() -> str:
    """The test database, or skip (fail when PostgreSQL is required)."""
    if not DSN:
        if REQUIRED:
            pytest.fail("ORKEXA_REQUIRE_POSTGRES=1 but ORKEXA_TEST_POSTGRES names no database")
        pytest.skip("ORKEXA_TEST_POSTGRES is not set")
    setup_server()
    return DSN


def admin() -> Any:
    """A new autocommit psycopg connection as the owner (close it)."""
    import psycopg

    return psycopg.connect(require_dsn(), autocommit=True)


def require_dsn() -> str:
    if not DSN:
        raise RuntimeError("ORKEXA_TEST_POSTGRES is not set")
    return DSN


def setup_server() -> None:
    """Once per session: the runtime role and, where available, pgvector (the CI's superuser setup)."""
    with _setup_lock:
        if _setup_done:
            return
        with admin() as connection:
            if (
                connection.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (RUNTIME_ROLE,)).fetchone()
                is None
            ):
                # Constant name and password: no value is formatted into the statement.
                connection.execute(
                    "CREATE ROLE orkexa_test_runtime LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS"
                    " PASSWORD 'orkexa-test-runtime-pw'"
                )
            available = connection.execute(
                "SELECT 1 FROM pg_available_extensions WHERE name = 'vector'"
            ).fetchone()
            if available is not None:
                connection.execute("CREATE EXTENSION IF NOT EXISTS vector SCHEMA public")
        _setup_done.append(True)


def has_pgvector() -> bool:
    """Whether pgvector is installed in the test database."""
    require_postgres()
    with admin() as connection:
        return (
            connection.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'").fetchone() is not None
        )


def require_pgvector(reason: str) -> None:
    """Skip a test that needs pgvector without it, or fail it under ``ORKEXA_REQUIRE_PGVECTOR=1``."""
    if has_pgvector():
        return
    if PGVECTOR_REQUIRED:
        pytest.fail(f"ORKEXA_REQUIRE_PGVECTOR=1 but the test database has no pgvector: {reason}")
    pytest.skip(f"the test database has no pgvector: {reason}")


def conninfo(schema: str, *, runtime: bool) -> str:
    """The test database as the owner or the runtime role, with ``schema`` first on the search path."""
    from psycopg.conninfo import conninfo_to_dict, make_conninfo

    parts = conninfo_to_dict(require_dsn())
    if runtime:
        parts.update(user=RUNTIME_ROLE, password=RUNTIME_PASSWORD)
    parts["options"] = f"-c search_path={schema},public"
    return make_conninfo("", **parts)


@dataclass
class Database:
    """One test schema: the owner's source, the source the stores use, the layout."""

    schema: str
    owner: PsycopgSource
    source: PsycopgSource
    layout: PostgresLayout

    def query(self, text: str, params: Any = None) -> list[tuple[Any, ...]]:
        """Rows of ``text`` as the owner (test checks and setup only)."""
        with self.owner.transaction("transaction") as connection:
            cursor = connection.execute(text, params)
            try:
                return [tuple(row) for row in cursor.fetchall()]
            except Exception:  # a statement without rows
                return []

    def runtime_source(self, **kwargs: Any) -> PsycopgSource:
        """A new source as the restricted runtime role on this schema (close it)."""
        return PsycopgSource(conninfo(self.schema, runtime=True), **kwargs)

    def grant(self, role: str = RUNTIME_ROLE) -> None:
        """What a product's operator grants its runtime role: DML on the tables of the schema, nothing
        else."""
        from psycopg import sql

        with admin() as connection:
            connection.execute(
                sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(
                    sql.Identifier(self.schema), sql.Identifier(role)
                )
            )
            connection.execute(
                sql.SQL("GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA {} TO {}").format(
                    sql.Identifier(self.schema), sql.Identifier(role)
                )
            )


def new_schema() -> str:
    return f"orkexa_test_{uuid.uuid4().hex[:16]}"


def create_schema(schema: str) -> None:
    from psycopg import sql

    with admin() as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))


def drop_schema(schema: str) -> None:
    from psycopg import sql

    with admin() as connection:
        connection.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema)))


@contextmanager
def database(
    layout: PostgresLayout = LAYOUT, *, migrated: bool = True, runtime: bool | None = None, **source_args: Any
) -> Iterator[Database]:
    """A fresh test schema, migrated for ``layout`` by the owner; the stores' source is the runtime role under
    ``ORKEXA_TEST_RUNTIME_ROLE=1`` (or ``runtime=True``), else the owner."""
    require_postgres()
    schema = new_schema()
    create_schema(schema)
    owner = PsycopgSource(conninfo(schema, runtime=False), **source_args)
    use_runtime = RUNTIME_ROLE_MODE if runtime is None else runtime
    source = PsycopgSource(conninfo(schema, runtime=True), **source_args) if use_runtime else owner
    opened = Database(schema, owner, source, layout)
    try:
        if migrated:
            migrate(owner, layout)
            opened.grant()
        yield opened
    finally:
        if source is not owner:
            source.close()
        owner.close()
        drop_schema(schema)


@contextmanager
def ddl_log() -> Iterator[Callable[[], list[tuple[str, str]]]]:
    """An event trigger that logs every DDL command of the test database (tag, session role) from here on
    (MUP-ONA T-DDL-1); yields a function that reads the log. Removed afterwards."""
    with admin() as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS public.orkexa_test_ddl_log (tag text NOT NULL, role text NOT NULL)"
        )
        connection.execute(
            "CREATE OR REPLACE FUNCTION public.orkexa_test_log_ddl() RETURNS event_trigger LANGUAGE plpgsql"
            " SECURITY DEFINER AS $$ BEGIN"
            " INSERT INTO public.orkexa_test_ddl_log VALUES (tg_tag, session_user); END $$"
        )
        connection.execute("DROP EVENT TRIGGER IF EXISTS orkexa_test_ddl")
        connection.execute(
            "CREATE EVENT TRIGGER orkexa_test_ddl ON ddl_command_start"
            " EXECUTE FUNCTION public.orkexa_test_log_ddl()"
        )
        connection.execute("TRUNCATE public.orkexa_test_ddl_log")

    def read() -> list[tuple[str, str]]:
        with admin() as connection:
            rows = connection.execute("SELECT tag, role FROM public.orkexa_test_ddl_log").fetchall()
        return [(str(tag), str(role)) for tag, role in rows]

    try:
        yield read
    finally:
        with admin() as connection:
            connection.execute("DROP EVENT TRIGGER IF EXISTS orkexa_test_ddl")


def catalog_snapshot(schema: str) -> dict[str, Any]:
    """Tables, columns, indexes and constraints of ``schema`` as the owner sees them (to prove a check
    changed nothing, and for the schema comparison T-SCH-2)."""
    with admin() as connection:
        columns = connection.execute(
            "SELECT table_name, column_name, ordinal_position, data_type, udt_name, is_nullable,"
            " collation_name, column_default, is_generated, generation_expression, is_identity,"
            " identity_generation"
            " FROM information_schema.columns WHERE table_schema = %s ORDER BY table_name, ordinal_position",
            (schema,),
        ).fetchall()
        indexes = connection.execute(
            "SELECT tablename, indexname, replace(indexdef, %s, '<schema>.') FROM pg_indexes"
            " WHERE schemaname = %s ORDER BY tablename, indexname",
            (schema + ".", schema),
        ).fetchall()
        constraints = connection.execute(
            "SELECT t.relname, c.conname, c.contype::text, pg_get_constraintdef(c.oid) FROM pg_constraint c"
            " JOIN pg_class t ON t.oid = c.conrelid JOIN pg_namespace n ON n.oid = t.relnamespace"
            " WHERE n.nspname = %s ORDER BY t.relname, c.conname",
            (schema,),
        ).fetchall()
    return {
        "columns": [list(row) for row in columns],
        "indexes": [list(row) for row in indexes],
        "constraints": [[*row[:3], str(row[3]).replace(schema + ".", "<schema>.")] for row in constraints],
    }
