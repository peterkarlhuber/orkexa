"""The tables of the PostgreSQL backend as data: schema version 1, its check and its explicit migration.

Schema version 1 is exactly the form of Nodestro's migrations ``0002_knowledge_memory.sql`` and
``0010_knowledge_vectors.sql`` (nodestro-runtime), for any :class:`~orkexa.backends.postgres.PostgresLayout`:
five tables, their constraints and three indexes, vectors as ``real[]``. The pgvector extension is never part
of it: the backend ranks vectors with pgvector where an administrator installed it (``CREATE EXTENSION
vector``), else in Python.

* :func:`verify_schema` checks a database read-only (``information_schema``, ``pg_indexes``,
  ``pg_extension``): it never changes anything and never raises; it returns a :class:`SchemaReport`. A product
  calls it at start.
* :func:`schema_steps` gives the statements of version 1 for a layout, for a product's own migration runner.
* :func:`migrate` applies them, only when called (never by the stores), with a role that may create tables,
  and records the version in ``<prefix>orkexa_schema``. It never runs ``CREATE EXTENSION``.

The runtime role of a product needs no DDL privilege: ``SELECT``, ``INSERT``, ``UPDATE`` and ``DELETE`` on the
five tables (and ``SELECT`` on the version table, for :func:`verify_schema`) are enough.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ...retrieval import RetrievalError
from ._sql import PostgresLayout
from .connection import Connection, ConnectionSource, psycopg_sql

__all__ = ["SCHEMA_VERSION", "SchemaProblem", "SchemaReport", "migrate", "schema_steps", "verify_schema"]

#: The schema version this release of the backend reads and writes.
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class _Column:
    """A column as ``information_schema.columns`` shows it."""

    name: str
    data_type: str  # information_schema data_type; arrays are "ARRAY" with udt_name "_float4"
    udt_name: str
    nullable: bool
    collation: str | None = None
    default: str | None = None
    generated: str | None = None  # generation_expression of a stored generated column
    identity: str | None = None  # identity_generation ("ALWAYS")


_TEXT_C = {"data_type": "text", "udt_name": "text", "collation": "C"}

#: Schema version 1: every table (by its template placeholder) and its columns, as 0002 + 0010 create them.
_TABLES: Mapping[str, tuple[_Column, ...]] = {
    "files": (
        _Column("tenant_id", nullable=False, **_TEXT_C),
        _Column("namespace", nullable=False, **_TEXT_C),
        _Column("id", nullable=False, **_TEXT_C),
        _Column("entry", "jsonb", "jsonb", nullable=False),
        _Column("data", "bytea", "bytea", nullable=False),
    ),
    "chunks": (
        _Column("id", "bigint", "int8", nullable=False, identity="ALWAYS"),
        _Column("tenant_id", nullable=False, **_TEXT_C),
        _Column("namespace", nullable=False, **_TEXT_C),
        _Column("source", nullable=False, **_TEXT_C),
        _Column("title", "text", "text", nullable=True),
        _Column("file_id", nullable=True, **_TEXT_C),
        _Column("metadata", "jsonb", "jsonb", nullable=False),
        _Column("text", "text", "text", nullable=False),
        _Column("search_text", "text", "text", nullable=False),
        _Column(
            "search",
            "tsvector",
            "tsvector",
            nullable=True,
            generated="to_tsvector('simple'::regconfig, search_text)",
        ),
    ),
    "memory": (
        _Column("tenant_id", nullable=False, **_TEXT_C),
        _Column("namespace", nullable=False, **_TEXT_C),
        _Column("path", nullable=False, **_TEXT_C),
        _Column("content", "text", "text", nullable=False),
        _Column(
            "updated_at",
            "timestamp with time zone",
            "timestamptz",
            nullable=False,
            default="clock_timestamp()",
        ),
    ),
    "spaces": (
        _Column("tenant_id", nullable=False, **_TEXT_C),
        _Column("namespace", nullable=False, **_TEXT_C),
        _Column("space", nullable=False, **_TEXT_C),
        _Column("identity", "jsonb", "jsonb", nullable=False),
        _Column("dimensions", "integer", "int4", nullable=False),
        _Column("state", nullable=False, **_TEXT_C),
    ),
    "vectors": (
        _Column("tenant_id", nullable=False, **_TEXT_C),
        _Column("namespace", nullable=False, **_TEXT_C),
        _Column("space", nullable=False, **_TEXT_C),
        _Column("chunk_id", "bigint", "int8", nullable=False),
        _Column("embedding", "ARRAY", "_float4", nullable=False),  # real[] (never pgvector's vector)
    ),
}

#: Schema version 1: every index (its name after the prefix -> table placeholder and the definition after
#: ``USING``, as ``pg_indexes.indexdef`` prints it). Primary keys are indexes too.
_INDEXES: Mapping[str, tuple[str, str]] = {
    "files_pkey": ("files", "btree (tenant_id, namespace, id)"),
    "chunks_pkey": ("chunks", "btree (id)"),
    "chunks_source": ("chunks", "btree (tenant_id, namespace, source)"),
    "chunks_search": ("chunks", "gin (search)"),
    "memory_files_pkey": ("memory", "btree (tenant_id, namespace, path)"),
    "vector_spaces_pkey": ("spaces", "btree (tenant_id, namespace, space)"),
    "chunk_vectors_pkey": ("vectors", "btree (tenant_id, space, chunk_id)"),
    "chunk_vectors_chunk": ("vectors", "btree (chunk_id)"),
}

#: The statements of schema version 1 (0002 + 0010 without comments and without the optional ``CREATE
#: EXTENSION``), with the tables as placeholders (rendered as identifiers by :func:`schema_steps`).
_STEPS: tuple[str, ...] = (
    "CREATE TABLE {files} ("
    ' tenant_id text COLLATE "C" NOT NULL,'
    ' namespace text COLLATE "C" NOT NULL,'
    ' id text COLLATE "C" NOT NULL,'
    " entry jsonb NOT NULL,"
    " data bytea NOT NULL,"
    " PRIMARY KEY (tenant_id, namespace, id),"
    " CHECK (tenant_id <> ''))",
    "CREATE TABLE {chunks} ("
    " id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,"
    ' tenant_id text COLLATE "C" NOT NULL,'
    ' namespace text COLLATE "C" NOT NULL,'
    ' source text COLLATE "C" NOT NULL,'
    " title text,"
    ' file_id text COLLATE "C",'
    " metadata jsonb NOT NULL,"
    " text text NOT NULL,"
    " search_text text NOT NULL,"
    " search tsvector GENERATED ALWAYS AS (to_tsvector('simple', search_text)) STORED,"
    " CHECK (tenant_id <> ''))",
    "CREATE INDEX {chunks_source} ON {chunks} (tenant_id, namespace, source)",
    "CREATE INDEX {chunks_search} ON {chunks} USING gin (search)",
    "CREATE TABLE {memory} ("
    ' tenant_id text COLLATE "C" NOT NULL,'
    ' namespace text COLLATE "C" NOT NULL,'
    ' path text COLLATE "C" NOT NULL,'
    " content text NOT NULL,"
    " updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),"
    " PRIMARY KEY (tenant_id, namespace, path),"
    " CHECK (tenant_id <> ''))",
    "CREATE TABLE {spaces} ("
    ' tenant_id text COLLATE "C" NOT NULL,'
    ' namespace text COLLATE "C" NOT NULL,'
    ' space text COLLATE "C" NOT NULL,'
    " identity jsonb NOT NULL,"
    " dimensions integer NOT NULL CHECK (dimensions > 0),"
    " state text COLLATE \"C\" NOT NULL CHECK (state IN ('active', 'building')),"
    " PRIMARY KEY (tenant_id, namespace, space),"
    " CHECK (tenant_id <> ''))",
    "CREATE TABLE {vectors} ("
    ' tenant_id text COLLATE "C" NOT NULL,'
    ' namespace text COLLATE "C" NOT NULL,'
    ' space text COLLATE "C" NOT NULL,'
    " chunk_id bigint NOT NULL REFERENCES {chunks} (id) ON DELETE CASCADE,"
    " embedding real[] NOT NULL,"
    " PRIMARY KEY (tenant_id, space, chunk_id),"
    " FOREIGN KEY (tenant_id, namespace, space) REFERENCES {spaces} (tenant_id, namespace, space)"
    " ON DELETE CASCADE,"
    " CHECK (tenant_id <> ''))",
    "CREATE INDEX {vectors_chunk} ON {vectors} (chunk_id)",
)

#: The version table of :func:`migrate` (one row per applied version).
_VERSION_TABLE = (
    "CREATE TABLE IF NOT EXISTS {version} (version integer PRIMARY KEY, applied_at timestamptz NOT NULL"
    " DEFAULT now())"
)
_VERSION_READ = "SELECT coalesce(max(version), 0) FROM {version}"
_VERSION_WRITE = "INSERT INTO {version} (version) VALUES (%(version)s)"

# Read-only queries of verify_schema and migrate: every name is a bound value, never part of the text.
_Q_TABLE = (
    "SELECT n.nspname FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace"
    " WHERE c.oid = to_regclass(%(name)s)"
)
_Q_COLUMNS = (
    "SELECT column_name, data_type, udt_name, is_nullable, collation_name, column_default,"
    " generation_expression,"
    " identity_generation FROM information_schema.columns WHERE table_schema = %(schema)s AND table_name ="
    " %(table)s"
)
_Q_INDEXES = (
    "SELECT indexname, indexdef FROM pg_catalog.pg_indexes WHERE schemaname = %(schema)s AND tablename ="
    " %(table)s"
)
_Q_PGVECTOR = "SELECT extversion FROM pg_catalog.pg_extension WHERE extname = 'vector'"
_Q_VERSION = "SELECT coalesce(max(version), 0) FROM {version}"
_Q_CAN_CREATE = "SELECT has_schema_privilege(current_schema(), 'CREATE')"


def _names(layout: PostgresLayout) -> dict[str, str]:
    """Every name a statement of the schema uses: the tables, the named indexes and the version table."""
    return {
        **layout.tables,
        "chunks_source": layout.prefix + "chunks_source",
        "chunks_search": layout.prefix + "chunks_search",
        "vectors_chunk": layout.prefix + "chunk_vectors_chunk",
        "version": layout.version_table,
    }


def _render(template: str, layout: PostgresLayout) -> Any:
    """``template`` with its names as identifiers (``psycopg.sql``)."""
    sql = psycopg_sql()
    return sql.SQL(template).format(**{key: sql.Identifier(name) for key, name in _names(layout).items()})


def schema_steps(layout: PostgresLayout) -> tuple[str, ...]:
    """The statements that create schema version 1 for ``layout``, in order, as SQL text.

    For a product that runs its own migrations. The table and index names are quoted identifiers; the text is
    the form of Nodestro's migrations 0002 and 0010 without comments and without the optional ``CREATE
    EXTENSION vector``. The version table of :func:`migrate` is not part of it.

    Raises:
        ImportError: psycopg is not installed.
    """
    return tuple(str(_render(step, layout).as_string(None)) for step in _STEPS)


@dataclass(frozen=True)
class SchemaProblem:
    """One difference between a database and schema version 1.

    ``kind`` is ``missing_table``, ``missing_column``, ``wrong_column`` (type, nullability, collation,
    default, generation or identity), ``missing_index``, ``wrong_index``, ``version`` (the version table names
    another version) or ``error`` (the check itself failed, e.g. no connection); ``table`` and ``name`` say
    where, ``detail`` what was expected and found.
    """

    kind: str
    table: str
    name: str
    detail: str

    def __str__(self) -> str:
        where = f"{self.table}.{self.name}" if self.name and self.table else (self.table or self.name)
        return f"{self.kind}: {where}: {self.detail}" if where else f"{self.kind}: {self.detail}"


@dataclass(frozen=True)
class SchemaReport:
    """What :func:`verify_schema` found.

    Attributes:
        prefix: The table prefix of the checked layout. schema: The schema that holds the first table found
        (``None``: no table found). version: The version recorded by :func:`migrate` (``None``: no version
        table, e.g. a product that runs its own migrations). pgvector: Whether the pgvector extension is
        installed in the database (``CREATE EXTENSION vector``); the index then ranks vectors in the database
        (mode B), else in Python (mode A). Either is valid. pgvector_version: The installed extension version,
        or ``None``. problems: Every difference from schema version 1; empty when the schema is usable.
    """

    prefix: str
    schema: str | None
    version: int | None
    pgvector: bool
    pgvector_version: str | None = None
    problems: tuple[SchemaProblem, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        """Whether the database holds schema version 1 for the layout (pgvector does not matter)."""
        return not self.problems

    def summary(self) -> str:
        """The report as text, one problem per line."""
        head = (
            f"PostgreSQL schema for prefix {self.prefix!r}: {'ok' if self.ok else 'NOT usable'}"
            f" (version {self.version if self.version is not None else 'not recorded'},"
            f" pgvector {'installed ' + str(self.pgvector_version) if self.pgvector else 'not installed'})"
        )
        return "\n".join([head, *(f"- {problem}" for problem in self.problems)])


def _rows(
    connection: Connection, query: Any, params: Mapping[str, Any] | None = None
) -> list[tuple[Any, ...]]:
    return [tuple(row) for row in connection.execute(query, params).fetchall()]


def _check_table(
    connection: Connection, layout: PostgresLayout, key: str, problems: list[SchemaProblem]
) -> str | None:
    """Check one table of ``layout``; the schema that holds it, or ``None`` when it is missing."""
    table = layout.table(key)
    found = _rows(connection, _Q_TABLE, {"name": table})
    if not found:
        problems.append(SchemaProblem("missing_table", table, "", "the table does not exist"))
        return None
    schema = str(found[0][0])
    columns = {str(row[0]): row for row in _rows(connection, _Q_COLUMNS, {"schema": schema, "table": table})}
    for column in _TABLES[key]:
        row = columns.get(column.name)
        if row is None:
            problems.append(SchemaProblem("missing_column", table, column.name, "the column does not exist"))
            continue
        _, data_type, udt_name, nullable, collation, default, generated, identity = row
        expected = {
            "type": f"{column.data_type} ({column.udt_name})",
            "nullable": column.nullable,
            "collation": column.collation,
            "default": column.default,
            "generated": column.generated,
            "identity": column.identity,
        }
        actual = {
            "type": f"{data_type} ({udt_name})",
            "nullable": nullable == "YES",
            "collation": collation,
            "default": default,
            "generated": generated,
            "identity": identity,
        }
        for what, value in expected.items():
            if actual[what] != value:
                problems.append(
                    SchemaProblem(
                        "wrong_column",
                        table,
                        column.name,
                        f"{what}: expected {value!r}, found {actual[what]!r}",
                    )
                )
    indexes = {
        str(name): str(definition)
        for name, definition in _rows(connection, _Q_INDEXES, {"schema": schema, "table": table})
    }
    for suffix, (index_table, using) in _INDEXES.items():
        if index_table != key:
            continue
        name = layout.prefix + suffix
        definition = indexes.get(name)
        if definition is None:
            problems.append(SchemaProblem("missing_index", table, name, f"expected an index USING {using}"))
        elif not definition.endswith(f" USING {using}"):
            problems.append(
                SchemaProblem("wrong_index", table, name, f"expected USING {using}, found {definition}")
            )
    return schema


def verify_schema(source: ConnectionSource, layout: PostgresLayout) -> SchemaReport:
    """Check read-only that the database of ``source`` holds schema version 1 for ``layout``.

    Reads ``information_schema``, ``pg_indexes``, ``pg_extension`` and the version table of :func:`migrate`
    (when it exists) in one ``read`` transaction; changes nothing, needs no DDL privilege (a table the role
    may not use counts as missing) and never raises: a failing check (no connection, no psycopg) is a problem
    of the kind ``error`` in the report.

    Args:
        source: Gives the connection (the product's runtime role is enough).
        layout: The tables to check.

    Returns:
        A :class:`SchemaReport`; :attr:`SchemaReport.ok` says whether the backend can use the tables.
    """
    prefix = getattr(layout, "prefix", repr(layout))
    problems: list[SchemaProblem] = []
    schema: str | None = None
    version: int | None = None
    pgvector: str | None = None
    try:
        if not isinstance(layout, PostgresLayout):
            raise TypeError(f"layout must be a PostgresLayout, not {type(layout).__name__}")
        with source.transaction("read") as connection:
            for key in _TABLES:
                found = _check_table(connection, layout, key, problems)
                schema = schema or found
            rows = _rows(connection, _Q_PGVECTOR)
            pgvector = str(rows[0][0]) if rows else None
            if _rows(connection, _Q_TABLE, {"name": layout.version_table}):
                version = int(_rows(connection, _render(_Q_VERSION, layout))[0][0])
                if version != SCHEMA_VERSION:
                    problems.append(
                        SchemaProblem(
                            "version",
                            layout.version_table,
                            "",
                            f"expected version {SCHEMA_VERSION}, found {version}",
                        )
                    )
    except Exception as exc:  # never raises: the report says what failed
        problems.append(SchemaProblem("error", "", "", f"{type(exc).__name__}: {exc}".strip()))
    return SchemaReport(prefix, schema, version, pgvector is not None, pgvector, tuple(problems))


def migrate(source: ConnectionSource, layout: PostgresLayout) -> int:
    """Create schema version 1 for ``layout`` where it is missing, and record it; only when called explicitly.

    Runs in one transaction under an advisory lock (concurrent calls wait for each other): creates the version
    table ``<prefix>orkexa_schema`` if needed, applies :func:`schema_steps` if no version is recorded, and
    records version 1. A database that already has version 1 is left unchanged. It never runs ``CREATE
    EXTENSION``: an administrator installs pgvector, if wanted, and the index picks it up
    (``PostgresIndex.refresh``).

    Args:
        source: Gives the connection of a role that may create tables in the current schema (the migration or
            owner role, not the runtime role).
        layout: The tables to create.

    Returns:
        The version the database had before (``0``: none).

    Raises:
        RetrievalError: ``schema.mismatch`` with ``reason`` ``permission`` (the role may not create tables in
            the current schema; nothing was changed), ``too_new`` (a newer version is recorded) or
            ``unrecorded`` (some of the tables exist but no version is recorded: they were created another
            way). ImportError: psycopg is not installed.
    """
    if not isinstance(layout, PostgresLayout):
        raise TypeError(f"layout must be a PostgresLayout, not {type(layout).__name__}")
    with source.transaction("transaction") as connection:
        if not _rows(connection, _Q_CAN_CREATE)[0][0]:
            raise RetrievalError(
                "schema.mismatch",
                "migrate() needs a role that may create tables in the current schema (a migration or owner"
                " role,"
                " not the runtime role); nothing was changed",
                reason="permission",
            )
        connection.execute("SELECT pg_advisory_xact_lock(hashtext(%(name)s))", {"name": layout.version_table})
        connection.execute(_render(_VERSION_TABLE, layout))
        current = int(_rows(connection, _render(_VERSION_READ, layout))[0][0])
        if current > SCHEMA_VERSION:
            raise RetrievalError(
                "schema.mismatch",
                f"The database holds schema version {current} for prefix {layout.prefix!r}; this release"
                " knows"
                f" {SCHEMA_VERSION}",
                reason="too_new",
            )
        if current == SCHEMA_VERSION:
            return current
        existing = [
            layout.table(key) for key in _TABLES if _rows(connection, _Q_TABLE, {"name": layout.table(key)})
        ]
        if existing:
            raise RetrievalError(
                "schema.mismatch",
                f"The tables {', '.join(existing)} exist but no schema version is recorded; they were not"
                " created"
                " by migrate() (check them with verify_schema)",
                reason="unrecorded",
            )
        for step in _STEPS:
            connection.execute(_render(step, layout))
        connection.execute(_render(_VERSION_WRITE, layout), {"version": SCHEMA_VERSION})
        return current
