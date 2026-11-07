"""The public conformance suite of the storage contracts (MUP-ONA B1, T-CONF; ADR 0014).

Every backend of ``Index`` (and ``ReindexableIndex``), ``FileStore`` and ``MemoryBackend`` runs the same
cases: the SQLite and local classes in Orkexa's own tests, Nodestro's PostgreSQL classes in nodestro-runtime,
your own backend in yours. A **factory** opens an instance for a :class:`Request`; the caller names the
capabilities it **expects** the backend to report::

    from orkexa import SQLiteIndex
    from orkexa.testing.conformance import ALL_INDEX_CASES, Request, run_cases

    def sqlite_index(request: Request) -> SQLiteIndex:
        return SQLiteIndex(
            request.workdir / "index.db", namespace=request.partition, embeddings=request.embeddings
        )

    report = run_cases(
        sqlite_index,
        ALL_INDEX_CASES,
        expect={"vectors", "reindex", "partitioned", "concurrent_writers"},
        target="sqlite-vec",
    )
    report.check()  # AssertionError with the summary unless every case passed

Cases whose capabilities the backend does not report are skipped (with the reason in the report); compare them
with an expectation (:meth:`Report.check` with ``expected_skips``, :func:`load_expected_skips`), so that a
lost capability fails instead of silently skipping. The case lists:

* :data:`INDEX_CASES` (T-CONF-1, T-CONF-2) and :data:`VECTOR_CASES` (T-CONF-3) in :mod:`.index`;
* :data:`REINDEX_CASES` (T-CONF-4) in :mod:`.reindex`;
* :data:`FILE_STORE_CASES` (T-CONF-5) in :mod:`.files`;
* :data:`MEMORY_CASES` (T-CONF-6) in :mod:`.memory`;
* :data:`PARTITION_INDEX_CASES`, :data:`PARTITION_FILE_STORE_CASES`, :data:`PARTITION_MEMORY_CASES` (plan
  chapter 10.4, T-PART) in :mod:`.partition`.

The package needs no pytest; see the documentation ("Testing helpers and conformance") for a pytest adapter.
"""

from __future__ import annotations

from ._core import (
    CAPABILITIES,
    DEFAULT_PARTITION,
    SECURITY_NEEDS,
    Case,
    CaseFailure,
    Context,
    Factory,
    Outcome,
    Report,
    Request,
    expect,
    expect_code,
    expect_equal,
    expect_error,
    load_expected_skips,
    run_cases,
)
from ._models import KeywordEmbeddings
from .files import FILE_STORE_CASES
from .index import INDEX_CASES, VECTOR_CASES
from .memory import MEMORY_CASES
from .partition import PARTITION_FILE_STORE_CASES, PARTITION_INDEX_CASES, PARTITION_MEMORY_CASES
from .reindex import REINDEX_CASES

__all__ = [
    "ALL_FILE_STORE_CASES",
    "ALL_INDEX_CASES",
    "ALL_MEMORY_CASES",
    "CAPABILITIES",
    "DEFAULT_PARTITION",
    "FILE_STORE_CASES",
    "INDEX_CASES",
    "MEMORY_CASES",
    "PARTITION_FILE_STORE_CASES",
    "PARTITION_INDEX_CASES",
    "PARTITION_MEMORY_CASES",
    "REINDEX_CASES",
    "SECURITY_NEEDS",
    "VECTOR_CASES",
    "Case",
    "CaseFailure",
    "Context",
    "Factory",
    "KeywordEmbeddings",
    "Outcome",
    "Report",
    "Request",
    "expect",
    "expect_code",
    "expect_equal",
    "expect_error",
    "load_expected_skips",
    "run_cases",
]

ALL_INDEX_CASES: tuple[Case, ...] = INDEX_CASES + VECTOR_CASES + REINDEX_CASES + PARTITION_INDEX_CASES
"""Every case for an ``Index`` backend (the ones its capabilities do not cover are skipped)."""
ALL_FILE_STORE_CASES: tuple[Case, ...] = FILE_STORE_CASES + PARTITION_FILE_STORE_CASES
"""Every case for a ``FileStore`` backend."""
ALL_MEMORY_CASES: tuple[Case, ...] = MEMORY_CASES + PARTITION_MEMORY_CASES
"""Every case for a ``MemoryBackend`` backend."""
