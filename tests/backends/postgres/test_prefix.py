"""Safe identifiers: the table prefix of :class:`~orkexa.backends.postgres.PostgresLayout` (MUP-ONA B5:
T-PG-ID, T-GUARD-2).

A prefix must match ``^[a-z][a-z0-9_]{0,30}_$``; anything else is refused before a statement is rendered or a
store is built, and the stores take only a :class:`PostgresLayout`, never a bare string.
"""

from __future__ import annotations

import dataclasses
import importlib.util
from typing import Any

import pytest

from orkexa.backends.postgres import PostgresFileStore, PostgresIndex, PostgresLayout, PostgresMemory, render
from orkexa.backends.postgres._sql import TABLES, Executor, check_prefix

from .test_guard import TENANT, SpySource

needs_psycopg = pytest.mark.skipif(importlib.util.find_spec("psycopg") is None, reason="needs psycopg")

BAD_PREFIXES = {
    "injection": "x; DROP",
    "upper case": "A_",
    "no underscore": "a",
    "non-ASCII": "ä_",
    "double quote": 'a"_',
    "single quote": "a'_",
    "33 characters": "a" * 32 + "_",
    "empty": "",
    "starts with a digit": "1a_",
    "starts with underscore": "_a_",
    "space": "a b_",
    "trailing newline": "a_\n",
    "not a string": None,
}


# MUP-ONA T-PG-ID
# MUP-ONA T-GUARD-2
@pytest.mark.parametrize("prefix", list(BAD_PREFIXES.values()), ids=list(BAD_PREFIXES))
def test_a_bad_prefix_is_refused(prefix: Any) -> None:
    spy = SpySource()
    with pytest.raises(ValueError, match="prefix"):
        check_prefix(prefix)
    with pytest.raises(ValueError, match="prefix"):
        PostgresLayout(prefix)
    with pytest.raises(ValueError, match="prefix"):
        render("chunks_count", prefix)
    # A layout whose prefix was changed afterwards (a frozen dataclass, forced) is refused by the executor.
    forged = PostgresLayout("t_")
    object.__setattr__(forged, "prefix", prefix)
    with pytest.raises(ValueError, match="prefix"):
        Executor(spy, layout=forged, tenant_id=TENANT)
    for kind in (PostgresIndex, PostgresFileStore, PostgresMemory):
        with pytest.raises(ValueError, match="prefix"):
            kind(spy, TENANT, layout=forged)
    assert spy.opened == []


@pytest.mark.parametrize("kind", [PostgresIndex, PostgresFileStore, PostgresMemory])
def test_a_store_takes_only_a_layout(kind: Any) -> None:
    for layout in ("t_", None, {"prefix": "t_"}):
        with pytest.raises(TypeError, match="PostgresLayout"):
            kind(SpySource(), TENANT, layout=layout)


@needs_psycopg
@pytest.mark.parametrize("prefix", ["nodestro_", "t_", "a_", "a1_b_", "a" * 31 + "_"])
def test_a_good_prefix_names_the_tables(prefix: str) -> None:
    layout = PostgresLayout(prefix)
    text = render("vectors_store", layout.prefix).as_string(None)
    assert f'"{prefix}chunk_vectors"' in text and f'"{prefix}chunks"' in text
    assert layout.tables == {key: prefix + table for key, table in TABLES.items()}
    assert layout.table("memory") == prefix + "memory_files"
    assert layout.version_table == prefix + "orkexa_schema"
    assert len(layout.version_table) <= 63  # PostgreSQL's identifier limit


def test_the_layout_is_immutable_and_has_a_default() -> None:
    layout = PostgresLayout()
    assert layout.prefix == "orkexa_"
    with pytest.raises(dataclasses.FrozenInstanceError):
        layout.prefix = "x_"  # type: ignore[misc]
    assert PostgresLayout("t_") == PostgresLayout("t_")
