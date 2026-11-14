"""Orkexa's core without the PostgreSQL extras (MUP-ONA B5: T-PKG-1, T-PKG-2; plan K12, E-2).

* **T-PKG-1** ``import orkexa``, ``orkexa.backends`` and every backend package work with ``psycopg``,
  ``pglast`` and ``pytest`` blocked, import none of them and open no network connection; and the core suite
  (:data:`CORE_SUITE`: contracts, SQLite, conformance, the PostgreSQL guard and prefix checks, the rule
  engine's skips, the public API) passes in a pytest run with ``psycopg`` and ``pglast`` blocked
  (``tests/packaging/block_extras.py``) and no third-party pytest plugin loaded. The CI job ``core-no-extras``
  runs this file in an installation without any extra (``pip install -r requirements/tabrion.txt . pytest``:
  the core requires the stdlib-only ``tabrion`` since TABRION MIG-ORK-3); the job ``test`` runs it with the
  extras installed but blocked.
* **T-PKG-2** ``import orkexa.backends.postgres`` works without psycopg; the clear message
  ``pip install "orkexa[postgres]"`` comes only on connection setup (``PsycopgSource``) or when a statement is
  rendered; ``verify_schema`` reports it instead of raising.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[2]

#: The core suite of T-PKG-1: what must pass without any extra (provider SDKs, pdf, vector, sqlcheck,
#: postgres).
CORE_SUITE = (
    "tests/test_storage_contracts.py",
    "tests/test_retrieval_modes.py",
    "tests/test_legacy_fixtures.py",
    "tests/test_rank_fixture.py",
    "tests/test_private_storage.py",
    "tests/test_testing_fakes.py",
    "tests/test_public_api.py",
    "tests/test_sqlcheck.py",
    "tests/test_check_skips.py",
    "tests/conformance",
    "tests/backends",
    "tests/flow",
)
#: Tests of the core suite that need a development tool, not an extra.
DEV_TOOLS = ("tests/backends/sqlite/test_import_order.py::test_the_old_import_paths_pass_mypy_strict",)

_BLOCK = """
import sys
BLOCKED = ("psycopg", "psycopg_binary", "psycopg_c", "psycopg_pool", "pglast", "pytest", "_pytest", "pluggy")
for name in BLOCKED:
    sys.modules[name] = None
import socket
def _no_network(*args, **kwargs):
    raise AssertionError("a network connection was opened on import")
socket.socket.connect = _no_network
socket.create_connection = _no_network
"""


def _python(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_BLOCK) + textwrap.dedent(code)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


# MUP-ONA T-PKG-1
def test_the_packages_import_without_extras_and_open_no_connection() -> None:
    result = _python(
        """
        import orkexa, orkexa.backends, orkexa.backends.sqlite, orkexa.backends.postgres
        import orkexa.testing, orkexa.testing.conformance, orkexa.testing.sqlcheck
        from orkexa import SQLiteIndex
        from orkexa.backends.postgres import PostgresIndex, PostgresLayout, verify_schema
        assert PostgresLayout("t_").table("chunks") == "t_chunks"
        watched = {"psycopg", "pglast", "pytest", "_pytest", "tabrion"}  # tabrion: TABRION MIG-ORK-2
        loaded = sorted(m for m in sys.modules if m.split(".")[0] in watched and sys.modules[m] is not None)
        print(loaded)
        """
    )
    assert result.returncode == 0, result.stderr
    # TABRION MIG-ORK-3: the SQLite backends use the stdlib-only tabrion core, never its server layers.
    assert result.stdout.strip() == "['tabrion', 'tabrion._version', 'tabrion.errors', 'tabrion.sqlite']"


# MUP-ONA T-PKG-1
def test_the_core_does_not_import_the_postgres_backend() -> None:
    result = _python(
        """
        import orkexa, orkexa.retrieval, orkexa.files, orkexa.memory, orkexa.embeddings, orkexa.backends
        print(sorted(m for m in sys.modules if m.startswith(("orkexa.backends.postgres", "orkexa.testing"))))
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


# MUP-ONA T-PKG-2
def test_the_postgres_backend_names_the_extra_only_on_use() -> None:
    result = _python(
        """
        from orkexa.backends.postgres import (
            MISSING_PSYCOPG, PostgresFileStore, PostgresIndex, PostgresLayout, PsycopgSource, render,
            schema_steps, verify_schema,
        )
        assert MISSING_PSYCOPG == 'The PostgreSQL backend needs psycopg: pip install "orkexa[postgres]"'
        for attempt in (lambda: PsycopgSource("host=db"), lambda: render("chunks_count", "t_"),
                        lambda: schema_steps(PostgresLayout("t_"))):
            try:
                attempt()
            except ImportError as exc:
                assert str(exc) == MISSING_PSYCOPG and exc.name == "psycopg", exc
            else:
                raise AssertionError("no ImportError")

        class Source:
            def transaction(self, mode="transaction", *, tenant_id=None):
                raise ImportError(MISSING_PSYCOPG)

        report = verify_schema(Source(), PostgresLayout("t_"))
        assert not report.ok and "orkexa[postgres]" in report.summary(), report.summary()
        index = PostgresIndex(Source(), "tenant-a", layout=PostgresLayout("t_"))
        try:
            len(index)
        except ImportError as exc:
            assert str(exc) == MISSING_PSYCOPG
        print("ok")
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


# MUP-ONA T-PKG-1
def test_the_core_suite_passes_without_extras() -> None:
    # The core suite runs as in no CI job of its own: no requirement, no skip list, no report directory.
    own = ("ORKEXA_REQUIRE_", "ORKEXA_CONFORMANCE_", "ORKEXA_CATALOG_", "ORKEXA_SCHEMA_")
    env = {key: value for key, value in os.environ.items() if not key.startswith(own)}
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"  # no third-party pytest plugin (pytest-cov ...)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "tests.packaging.block_extras",
            "-p",
            "no:cacheprovider",
            "-q",
            "-m",
            "not postgres",
            *(f"--deselect={test}" for test in DEV_TOOLS),
            *CORE_SUITE,
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=1200,
        check=False,
    )
    summary = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    assert result.returncode == 0, result.stdout[-5000:] + result.stderr[-2000:]
    assert " passed" in summary and "failed" not in summary and "error" not in summary, summary
