"""The conformance suite against the PostgreSQL backends (MUP-ONA B5: T-CONF-* (PG), T-PART-1..9).

``PostgresIndex``, ``PostgresFileStore`` and ``PostgresMemory`` run every case of
``orkexa.testing.conformance`` (index, vector, reindex, file store, memory and the partition cases of plan
chapter 10.4), with the tenant as the partition and a fresh namespace per case, each target in a test schema
of its own. The index runs twice: ``pg-index`` ranks vectors as the database can (pgvector in ``pg-b``, which
``ORKEXA_REQUIRE_PGVECTOR=1`` demands), ``pg-index-python`` always in Python (mode A), so both paths meet the
same contract on a server with pgvector. No case is an expected failure; skips must match
``expected_skips.json`` for the job named by ``ORKEXA_CONFORMANCE_JOB`` (``pg-a``, ``pg-b``). The CI jobs run
this as superuser and as the restricted runtime role (``ORKEXA_TEST_RUNTIME_ROLE=1``).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from orkexa.backends.postgres import PostgresFileStore, PostgresIndex, PostgresMemory
from orkexa.testing.conformance import (
    ALL_FILE_STORE_CASES,
    ALL_INDEX_CASES,
    ALL_MEMORY_CASES,
    Case,
    Request,
)

from ..backends.postgres import support
from ..backends.postgres.support import orkexa_postgres_gate, tabrion_postgres_url  # noqa: F401  # fixtures
from .adapter import run_and_check

ALL = frozenset({"vectors", "reindex", "partitioned", "concurrent_writers"})
STORAGE = frozenset({"partitioned", "concurrent_writers"})

#: target -> (the cases it runs, the capabilities it must report)
TARGETS: dict[str, tuple[tuple[Case, ...], frozenset[str]]] = {
    "pg-index": (ALL_INDEX_CASES, ALL),
    "pg-index-python": (ALL_INDEX_CASES, ALL),
    "pg-file-store": (ALL_FILE_STORE_CASES, STORAGE),
    "pg-memory": (ALL_MEMORY_CASES, STORAGE),
}

#: T-PART-n (plan 10.4, in its order R, W, U, D, RX, TX, CC, V, J) -> the orkexa cases that prove it.
T_PART: dict[int, tuple[str, ...]] = {
    1: ("partition.index_reads", "partition.files_reads", "partition.memory_reads"),
    2: ("partition.index_writes", "partition.files_writes_and_deletes"),
    3: ("partition.memory_updates_and_deletes",),
    4: (
        "partition.index_deletes",
        "partition.files_writes_and_deletes",
        "partition.memory_updates_and_deletes",
    ),
    5: ("partition.index_reindex",),
    6: ("partition.index_failed_write_rolls_back",),
    7: ("partition.index_concurrent_use",),
    8: (
        "partition.index_hostile_values",
        "partition.files_hostile_values",
        "partition.memory_hostile_values",
    ),
    9: ("partition.index_vector_joins",),
}


def _factory(target: str, db: support.Database) -> Callable[[Request], Any]:
    def index(request: Request) -> PostgresIndex:
        opened = PostgresIndex(
            db.source,
            request.partition,
            layout=db.layout,
            namespace=request.scope,
            embeddings=request.embeddings,
        )
        if target == "pg-index-python":
            opened._pgvector = False  # rank in Python, as a server without pgvector does
        return opened

    def files(request: Request) -> PostgresFileStore:
        return PostgresFileStore(db.source, request.partition, layout=db.layout, namespace=request.scope)

    def memory(request: Request) -> PostgresMemory:
        return PostgresMemory(db.source, request.partition, layout=db.layout, namespace=request.scope)

    return {"pg-index": index, "pg-index-python": index, "pg-file-store": files, "pg-memory": memory}[target]


# MUP-ONA T-CONF-1
# MUP-ONA T-CONF-2
# MUP-ONA T-CONF-3
# MUP-ONA T-CONF-4
# MUP-ONA T-CONF-5
# MUP-ONA T-CONF-6
# MUP-ONA T-PART-1
# MUP-ONA T-PART-2
# MUP-ONA T-PART-3
# MUP-ONA T-PART-4
# MUP-ONA T-PART-5
# MUP-ONA T-PART-6
# MUP-ONA T-PART-7
# MUP-ONA T-PART-8
# MUP-ONA T-PART-9
@pytest.mark.postgres
@pytest.mark.parametrize("target", list(TARGETS))
def test_the_postgres_backends_conform(target: str, tmp_path: Path) -> None:
    with support.database() as db:
        if target == "pg-index":
            ranks = PostgresIndex(db.source, "conformance-probe", layout=db.layout).pgvector()
            if support.PGVECTOR_REQUIRED:
                assert ranks, "ORKEXA_REQUIRE_PGVECTOR=1 but the test database has no pgvector"
        cases, expect = TARGETS[target]
        report = run_and_check(target, _factory(target, db), cases, expect, tmp_path)
        assert report.passed
        assert report.capabilities >= expect
        partition = {case.name for case in cases if case.name.startswith("partition.")}
        assert partition <= set(report.passed)  # never skipped, never expected to fail


def test_every_t_part_case_runs_against_the_postgres_classes() -> None:
    names = {case.name for case in (*ALL_INDEX_CASES, *ALL_FILE_STORE_CASES, *ALL_MEMORY_CASES)}
    assert set(T_PART) == set(range(1, 10))
    targets = {target: {case.name for case in cases} for target, (cases, _) in TARGETS.items()}
    for number, cases in T_PART.items():
        assert set(cases) <= names, f"T-PART-{number}: {set(cases) - names}"
        for name in cases:
            if name.startswith("partition.index_"):
                assert name in targets["pg-index"] and name in targets["pg-index-python"], name
            else:
                assert any(name in found for found in targets.values()), name
