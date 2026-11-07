"""The conformance runner (MUP-ONA B1): it catches broken backends (T-CONF-7) and skip drift (T-CONF-8)."""

from __future__ import annotations

from collections.abc import Iterable
import json
from pathlib import Path
from typing import Any

import pytest

from orkexa import FileMemory, SQLiteIndex, SQLiteMemory
from orkexa.retrieval import Hit, RetrievalMode
from orkexa.testing.conformance import (
    ALL_FILE_STORE_CASES,
    ALL_INDEX_CASES,
    ALL_MEMORY_CASES,
    INDEX_CASES,
    MEMORY_CASES,
    PARTITION_INDEX_CASES,
    PARTITION_MEMORY_CASES,
    VECTOR_CASES,
    Case,
    Context,
    Request,
    expect_equal,
    load_expected_skips,
    run_cases,
)

from ..audit3._vec import needs_vec
from .factories import sqlite_index_fts, sqlite_memory

EXPECTED_SKIPS = Path(__file__).resolve().parent / "expected_skips.json"


def _names(cases: Iterable[Case]) -> set[str]:
    return {case.name for case in cases}


# ---- T-CONF-7: the runner refuses what it cannot check ---------------------------------------------------


# MUP-ONA T-CONF-7
def test_fewer_capabilities_than_expected_fail_the_run(tmp_path: Path) -> None:
    report = run_cases(sqlite_index_fts, INDEX_CASES, expect={"vectors", "partitioned"}, workdir=tmp_path)
    assert not report.ok
    assert any("missing vectors" in error for error in report.errors)
    with pytest.raises(AssertionError, match="missing vectors"):
        report.check()


# MUP-ONA T-CONF-7
def test_unknown_expected_capabilities_fail_the_run(tmp_path: Path) -> None:
    report = run_cases(
        sqlite_index_fts, INDEX_CASES[:1], expect={"partitioned", "teleport"}, workdir=tmp_path
    )
    assert any("unknown expected capabilities: teleport" in error for error in report.errors)


class TenantIndex(SQLiteIndex):
    """A backend with tenants that forgets to report ``partitioned``."""

    def __init__(self, path: Path, *, tenant_id: str) -> None:
        super().__init__(path, namespace=tenant_id)

    @property
    def capabilities(self) -> frozenset[str]:
        return frozenset({"concurrent_writers"})


# MUP-ONA T-CONF-7
def test_a_tenant_id_backend_must_report_partitioned(tmp_path: Path) -> None:
    def factory(request: Request) -> TenantIndex:
        return TenantIndex(request.workdir / "index.db", tenant_id=request.partition)

    report = run_cases(factory, INDEX_CASES[:2], expect=set(), workdir=tmp_path)
    assert any("takes tenant_id but does not report the capability 'partitioned'" in e for e in report.errors)
    assert not report.ok


# MUP-ONA T-CONF-7
def test_cases_of_another_contract_fail(tmp_path: Path) -> None:
    report = run_cases(sqlite_memory, INDEX_CASES[:2], expect=set(), workdir=tmp_path)
    assert any("does not implement the contract of the index cases" in e for e in report.errors)
    assert report.failed == [case.name for case in INDEX_CASES[:2]]


# MUP-ONA T-CONF-7
def test_a_factory_that_cannot_open_fails_every_case(tmp_path: Path) -> None:
    def broken(request: Request) -> Any:
        raise OSError("no database here")

    report = run_cases(broken, MEMORY_CASES, expect=set(), workdir=tmp_path)
    assert report.failed == [case.name for case in MEMORY_CASES]
    assert any("could not open an instance: OSError: no database here" in e for e in report.errors)


# MUP-ONA T-CONF-7
def test_duplicate_case_names_fail_the_run(tmp_path: Path) -> None:
    report = run_cases(sqlite_memory, MEMORY_CASES[:1] * 2, expect=set(), workdir=tmp_path)
    assert any("duplicate case names: memory.create_and_view" in e for e in report.errors)


# MUP-ONA T-CONF-7
def test_cases_validate_their_definition() -> None:
    def run(context: Context) -> None:
        return None

    Case("custom.my_check", frozenset({"vectors"}), run)
    with pytest.raises(ValueError, match="Invalid case name"):
        Case("No Dots", frozenset(), run)
    with pytest.raises(ValueError, match="unknown capabilities"):
        Case("custom.check", frozenset({"teleport"}), run)
    with pytest.raises(ValueError, match="a security case may need only"):
        Case("custom.check", frozenset({"concurrent_writers"}), run, security=True)
    with pytest.raises(ValueError, match="unknown contract"):
        Case("custom.check", frozenset(), run, "queue")  # type: ignore[arg-type]


# MUP-ONA T-CONF-7
def test_security_cases_need_only_security_capabilities() -> None:
    every = ALL_INDEX_CASES + ALL_MEMORY_CASES
    security = [case for case in every if case.security]
    assert _names(PARTITION_INDEX_CASES) <= _names(security)
    assert all(case.needs <= {"partitioned", "vectors", "reindex"} for case in security)
    assert len(_names(every)) == len(every)  # names are unique across the lists


# MUP-ONA T-CONF-7
def test_a_failing_case_reports_its_detail(tmp_path: Path) -> None:
    def wrong(context: Context) -> None:
        """Always fails."""
        expect_equal(1, 2, "one")

    report = run_cases(
        sqlite_memory, [Case("custom.wrong", frozenset(), wrong, "memory")], expect=set(), workdir=tmp_path
    )
    assert report.failed == ["custom.wrong"]
    assert "one" in report.summary() and "FAILED custom.wrong" in report.summary()
    data = report.to_dict()
    assert data["ok"] is False and data["outcomes"][0]["status"] == "failed"
    json.dumps(data)


# ---- T-CONF-7: mutants of a correct backend fail the matching cases --------------------------------------


# MUP-ONA T-CONF-7
def test_a_backend_that_ignores_the_partition_fails_the_partition_cases(tmp_path: Path) -> None:
    def one_namespace(request: Request) -> SQLiteIndex:
        return SQLiteIndex(request.workdir / "index.db", namespace="everyone")  # ignores request.partition

    report = run_cases(one_namespace, PARTITION_INDEX_CASES, expect={"partitioned"}, workdir=tmp_path)
    assert {"partition.index_reads", "partition.index_writes", "partition.index_deletes"} <= set(
        report.failed
    )

    def one_memory(request: Request) -> SQLiteMemory:
        return SQLiteMemory(request.workdir / "memory.db", namespace="everyone")

    report = run_cases(one_memory, PARTITION_MEMORY_CASES, expect={"partitioned"}, workdir=tmp_path)
    assert {"partition.memory_reads", "partition.memory_updates_and_deletes"} <= set(report.failed)


class ClaimsPartitions(FileMemory):
    """A one-directory memory that claims partitions it does not have."""

    @property
    def capabilities(self) -> frozenset[str]:
        return frozenset({"partitioned"})


# MUP-ONA T-CONF-7
def test_a_false_partitioned_claim_fails(tmp_path: Path) -> None:
    def factory(request: Request) -> ClaimsPartitions:
        return ClaimsPartitions(request.workdir / "memory")

    report = run_cases(factory, PARTITION_MEMORY_CASES, expect={"partitioned"}, workdir=tmp_path)
    assert "partition.memory_reads" in report.failed


class TruncatingIndex(SQLiteIndex):
    """Cuts long queries instead of refusing them, and claims (wrongly) to check queries itself."""

    def retrieve(
        self,
        query: str,
        k: int = 5,
        *,
        mode: RetrievalMode | None = None,
        sources: Iterable[str] | None = None,
    ) -> list[Hit]:
        return super().retrieve(" ".join(query[:200].split()[:10]), k, mode=mode, sources=sources)

    retrieve.__orkexa_checks_query__ = True  # type: ignore[attr-defined]


class HonestTruncatingIndex(SQLiteIndex):
    """The same cut without the false marker: the contract's wrapper still checks the query first."""

    def retrieve(
        self,
        query: str,
        k: int = 5,
        *,
        mode: RetrievalMode | None = None,
        sources: Iterable[str] | None = None,
    ) -> list[Hit]:
        return super().retrieve(" ".join(query[:200].split()[:10]), k, mode=mode, sources=sources)


LIMIT_CASES = [
    case for case in INDEX_CASES if case.name.startswith("index.query_over") or "huge" in case.name
]


# MUP-ONA T-CONF-7
def test_a_backend_that_skips_the_query_limits_fails(tmp_path: Path) -> None:
    def factory(request: Request) -> TruncatingIndex:
        return TruncatingIndex(request.workdir / "index.db", namespace=request.partition)

    report = run_cases(factory, LIMIT_CASES, expect={"partitioned"}, workdir=tmp_path)
    assert set(report.failed) == _names(LIMIT_CASES)

    def honest(request: Request) -> HonestTruncatingIndex:
        return HonestTruncatingIndex(request.workdir / "index.db", namespace=request.partition)

    report = run_cases(honest, LIMIT_CASES, expect={"partitioned"}, workdir=tmp_path)
    report.check()


class UnroundedIndex(SQLiteIndex):
    """Returns vector scores with more digits than the contract allows."""

    def retrieve(
        self,
        query: str,
        k: int = 5,
        *,
        mode: RetrievalMode | None = None,
        sources: Iterable[str] | None = None,
    ) -> list[Hit]:
        hits = super().retrieve(query, k, mode=mode, sources=sources)
        if (mode or self.mode) != "vector":
            return hits
        return [hit.model_copy(update={"score": hit.score + 3e-9}) for hit in hits]


class DotProductIndex(SQLiteIndex):
    """Scores vectors by an unnormalized dot product (a loud vector wins) instead of the cosine."""

    def retrieve(
        self,
        query: str,
        k: int = 5,
        *,
        mode: RetrievalMode | None = None,
        sources: Iterable[str] | None = None,
    ) -> list[Hit]:
        hits = super().retrieve(query, k, mode=mode, sources=sources)
        if (mode or self.mode) != "vector":
            return hits
        return [
            hit.model_copy(
                update={"score": round(hit.score * (10.0 if "loudvector" in hit.text else 1.0), 6)}
            )
            for hit in hits
        ]


# MUP-ONA T-CONF-7
@needs_vec
@pytest.mark.parametrize(
    ("index_class", "failing"),
    [
        (UnroundedIndex, {"vector.scores_are_rounded_to_6_digits"}),
        (DotProductIndex, {"vector.score_is_the_cosine"}),
    ],
    ids=["unrounded", "dot-product"],
)
def test_wrong_vector_scores_fail(index_class: type[SQLiteIndex], failing: set[str], tmp_path: Path) -> None:
    def factory(request: Request) -> SQLiteIndex:
        return index_class(
            request.workdir / "index.db", namespace=request.partition, embeddings=request.embeddings
        )

    report = run_cases(factory, VECTOR_CASES, expect={"vectors"}, workdir=tmp_path)
    assert failing <= set(report.failed), report.summary()


# ---- T-CONF-8: skips are compared with an expectation per CI job -----------------------------------------


# MUP-ONA T-CONF-8
def test_a_skip_that_is_not_expected_fails(tmp_path: Path) -> None:
    report = run_cases(sqlite_index_fts, VECTOR_CASES[:2], expect={"partitioned"}, workdir=tmp_path)
    assert report.ok and set(report.skipped) == _names(VECTOR_CASES[:2])
    report.check()  # without an expectation, skips are fine
    with pytest.raises(AssertionError, match="skipped but not expected: vector"):
        report.check({VECTOR_CASES[0].name: "no vectors"})
    with pytest.raises(AssertionError, match=r"expected to be skipped but ran: index\.add_and_search"):
        report.check({**report.skipped, "index.add_and_search": "never"})
    report.check({case.name: "SQLiteIndex without embeddings" for case in VECTOR_CASES[:2]})


def _write(tmp_path: Path, jobs: dict[str, Any]) -> Path:
    path = tmp_path / "expected.json"
    path.write_text(json.dumps({"jobs": jobs}), encoding="utf-8")
    return path


# MUP-ONA T-CONF-8
def test_expected_skips_are_read_per_job_and_target(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        {
            "test": {"pg": {"vector.score_is_the_cosine": "pgvector rounds"}, "sqlite": {}},
            "nightly": "test",
            "loop-a": "loop-b",
            "loop-b": "loop-a",
            "bad": {"pg": {"vector.score_is_the_cosine": " "}},
        },
    )
    assert load_expected_skips(path, "test", "pg") == {"vector.score_is_the_cosine": "pgvector rounds"}
    assert load_expected_skips(path, "nightly", "sqlite") == {}
    with pytest.raises(KeyError):
        load_expected_skips(path, "unknown-job", "pg")
    with pytest.raises(KeyError):
        load_expected_skips(path, "test", "unknown-target")
    with pytest.raises(ValueError, match="needs a reason"):
        load_expected_skips(path, "bad", "pg")
    with pytest.raises(ValueError, match="loop"):
        load_expected_skips(path, "loop-a", "pg")


# MUP-ONA T-CONF-8
def test_the_repository_expectations_cover_every_job_and_target() -> None:
    from .factories import FACTORIES, LEGACY

    data = json.loads(EXPECTED_SKIPS.read_text(encoding="utf-8"))
    targets = set(FACTORIES) | {f"{target}-legacy" for target in LEGACY}
    for job in ("test", "lowest-versions", "offline"):
        for target in targets:
            for name in load_expected_skips(EXPECTED_SKIPS, job, target):
                assert name in _names(ALL_INDEX_CASES + ALL_MEMORY_CASES + ALL_FILE_STORE_CASES), name
    assert set(data["jobs"]["test"]) == targets
