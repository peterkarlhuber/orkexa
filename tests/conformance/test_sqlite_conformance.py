"""The conformance suite against the SQLite backends (MUP-ONA B1: T-CONF-1..6, T-PART, T-SQL-10)."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
import json
from pathlib import Path
import sqlite3
from typing import Any

import pytest

from orkexa import SQLiteFileStore, SQLiteIndex, SQLiteMemory
from orkexa.testing.conformance import (
    ALL_FILE_STORE_CASES,
    ALL_INDEX_CASES,
    ALL_MEMORY_CASES,
    Case,
    Request,
    run_cases,
)

from ..audit3._vec import needs_vec
from .adapter import run_and_check
from .factories import FACTORIES, LEGACY, LEGACY_DIR

CASES: dict[str, tuple[Case, ...]] = {
    "sqlite-index-fts": ALL_INDEX_CASES,
    "sqlite-index-vec": ALL_INDEX_CASES,
    "sqlite-file-store": ALL_FILE_STORE_CASES,
    "sqlite-memory": ALL_MEMORY_CASES,
}
TARGETS = [
    pytest.param("sqlite-index-fts", id="sqlite-index-fts"),
    pytest.param("sqlite-index-vec", id="sqlite-index-vec", marks=needs_vec),
    pytest.param("sqlite-file-store", id="sqlite-file-store"),
    pytest.param("sqlite-memory", id="sqlite-memory"),
]


# MUP-ONA T-CONF-1
# MUP-ONA T-CONF-2
# MUP-ONA T-CONF-3
# MUP-ONA T-CONF-4
# MUP-ONA T-CONF-5
# MUP-ONA T-CONF-6
@pytest.mark.parametrize("target", TARGETS)
def test_the_sqlite_backends_conform(target: str, tmp_path: Path) -> None:
    factory, expect = FACTORIES[target]
    report = run_and_check(target, factory, CASES[target], expect, tmp_path)
    assert report.passed  # the run did something


def _legacy_contents() -> dict[str, Any]:
    manifest = json.loads((LEGACY_DIR / "manifest.json").read_text(encoding="utf-8"))
    contents: dict[str, Any] = manifest["contents"]
    return contents


def _check_legacy_data(target: str, factory: Callable[..., Any], workdir: Path) -> None:
    """The legacy namespaces of the copied FX-SQ-0 file still hold exactly their recorded content."""
    contents = _legacy_contents()[LEGACY[target]]
    request = Request(workdir=workdir, scope="legacy-check")
    for namespace, expected in contents.items():
        if not isinstance(expected, list):
            continue  # manifest metadata (embedding model, dimensions)
        instance = factory(Request(workdir=request.workdir, scope=request.scope, partition=namespace))
        try:
            if isinstance(instance, SQLiteIndex):
                assert instance.sources() == expected, namespace
            elif isinstance(instance, SQLiteFileStore):
                assert sorted(entry.id for entry in instance.list()) == sorted(expected), namespace
            else:
                assert isinstance(instance, SQLiteMemory)
                for path in expected:
                    assert "     1\t" in instance.view(path), (
                        namespace,
                        path,
                    )  # the file exists and has content
        finally:
            instance.close()
    database = next(path for path in workdir.iterdir() if path.suffix == ".db")
    connection = sqlite3.connect(database)
    try:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        if target.startswith("sqlite-index"):
            connection.execute("INSERT INTO index_fts (index_fts) VALUES ('integrity-check')")
    finally:
        connection.close()


def _legacy_factory(target: str, checked: list[Path]) -> Callable[[Request], Any]:
    factory = partial(FACTORIES[target][0], legacy=LEGACY[target])

    def open_on_legacy_copy(request: Request) -> Any:
        if request.workdir not in checked:
            checked.append(request.workdir)
        return factory(request)

    return open_on_legacy_copy


# MUP-ONA T-SQL-10
@pytest.mark.parametrize("target", TARGETS)
def test_the_suite_runs_on_a_copy_of_the_legacy_files(target: str, tmp_path: Path) -> None:
    """Every case runs on its own copy of the FX-SQ-0 file written before B1: the contract holds on old files
    (the next test checks that the legacy content survives the cases)."""
    workdirs: list[Path] = []
    report = run_and_check(
        f"{target}-legacy", _legacy_factory(target, workdirs), CASES[target], FACTORIES[target][1], tmp_path
    )
    assert report.passed
    assert len(workdirs) == len(report.passed) + 1  # one legacy copy per case that ran, and one for the probe


# MUP-ONA T-SQL-10
@pytest.mark.parametrize("target", TARGETS)
def test_a_legacy_copy_keeps_its_content_after_conformance_cases(target: str, tmp_path: Path) -> None:
    """The cases write next to the legacy data (other namespaces), never into it."""
    factory = partial(FACTORIES[target][0], legacy=LEGACY[target])
    shared = tmp_path / "shared"
    shared.mkdir()

    def same_copy(request: Request) -> Any:
        # Every case of this run works on the one legacy copy in ``shared`` (unique partitions per case).
        return factory(
            Request(shared, request.scope, f"{request.partition}-{request.scope}", request.embeddings)
        )

    cases = [case for case in CASES[target] if not case.name.startswith("partition.")]
    report = run_cases(
        same_copy, cases, expect=FACTORIES[target][1], target=f"{target}-shared", workdir=tmp_path
    )
    assert not report.failed, report.summary()
    _check_legacy_data(target, factory, shared)
