"""The PostgreSQL storage backends: :class:`PostgresIndex`, :class:`PostgresFileStore` and
:class:`PostgresMemory` (extra ``postgres``: ``pip install "orkexa[postgres]"``).

The classes keep the rows of many tenants in one set of tables and enforce the partition: every statement is a
named entry of a fixed catalog that binds ``tenant_id``, and a guard refuses any statement not bound to the
instance's tenant before it reaches the database. They never check permissions; who may act for a tenant is
the host's decision. Connections come from a :class:`ConnectionSource` (:class:`PsycopgSource` is a ready-made
one); the tables are named by a :class:`PostgresLayout` (a prefix). :func:`verify_schema` checks them
read-only, :func:`migrate` creates them when a product asks for it (:mod:`orkexa.backends.postgres.schema`).

Importing this package needs no psycopg and opens no connection: psycopg is imported when a
:class:`PsycopgSource` is built or the first statement is rendered.
"""

from __future__ import annotations

from ._sql import CATALOG, PostgresLayout, Statement, render
from .connection import (
    MISSING_PSYCOPG,
    MODES,
    Connection,
    ConnectionSource,
    Cursor,
    PsycopgSource,
    TransactionMode,
)
from .files import PostgresFileStore
from .index import MAX_SOURCES, PostgresIndex, PostgresVectorStatus, search_words
from .memory import PostgresMemory
from .schema import SCHEMA_VERSION, SchemaProblem, SchemaReport, migrate, schema_steps, verify_schema

__all__ = [
    "CATALOG",
    "MAX_SOURCES",
    "MISSING_PSYCOPG",
    "MODES",
    "SCHEMA_VERSION",
    "Connection",
    "ConnectionSource",
    "Cursor",
    "PostgresFileStore",
    "PostgresIndex",
    "PostgresLayout",
    "PostgresMemory",
    "PostgresVectorStatus",
    "PsycopgSource",
    "SchemaProblem",
    "SchemaReport",
    "Statement",
    "TransactionMode",
    "migrate",
    "render",
    "schema_steps",
    "search_words",
    "verify_schema",
]
