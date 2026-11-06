"""FX-RANK-0, the fixed ranking corpus of MUP-ONA (``tests/fixtures/ranking/``).

The fixture holds the results of the code before MP-B1/B3. While :class:`orkexa.SQLiteIndex` is
unchanged, measuring the corpus again gives exactly the recorded SQLite results; a deliberate ranking
change re-measures with ``tests/fixtures/ranking/measure_sqlite.py`` (see the README there).
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sqlite3
import sys
from types import ModuleType
from typing import Any

import pytest

HAS_VEC = importlib.util.find_spec("sqlite_vec") is not None and hasattr(
    sqlite3.connect(":memory:"), "enable_load_extension"
)

RANKING = Path(__file__).parent / "fixtures" / "ranking"


def _module(name: str) -> ModuleType:
    sys.path.insert(0, str(RANKING))  # measure_sqlite imports make_rank_corpus from its directory
    try:
        spec = importlib.util.spec_from_file_location(name, RANKING / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(RANKING))
    return module


@pytest.fixture(scope="module")
def corpus() -> ModuleType:
    return _module("make_rank_corpus")


@pytest.fixture(scope="module")
def fixture(corpus: ModuleType) -> dict[str, Any]:
    data: dict[str, Any] = corpus.load_fixture()
    return data


# MUP-ONA T-RANK-0
def test_fixture_checksum_matches() -> None:
    recorded = (RANKING / "FX-RANK-0.json.sha256").read_text(encoding="utf-8").split()
    assert recorded[1] == "FX-RANK-0.json"
    assert recorded[0] == hashlib.sha256((RANKING / "FX-RANK-0.json").read_bytes()).hexdigest()


# MUP-ONA T-RANK-0
def test_fixture_loads_with_every_backend(fixture: dict[str, Any]) -> None:
    assert fixture["fixture"] == "FX-RANK-0"
    assert len(fixture["chunks"]) == 200
    assert len(fixture["queries"]) == 30
    assert len({chunk["source"] for chunk in fixture["chunks"]}) > 1
    assert set(fixture["results"]) == {"sqlite", "sqlite-novec", "pg-a", "pg-b"}
    for backend, result in fixture["results"].items():
        assert result["meta"]["backend"] == backend
        assert set(result["runs"]) == set(fixture["identities"])
        for run in result["runs"].values():
            assert set(run) == {query["id"] for query in fixture["queries"]}
            for by_mode in run.values():
                assert set(by_mode) == set(fixture["modes"])
                for entry in by_mode.values():
                    assert "error" in entry or len(entry["hits"]) <= fixture["k"]


# MUP-ONA T-RANK-0
def test_corpus_regenerates_identically(corpus: ModuleType, fixture: dict[str, Any]) -> None:
    generated = corpus.header()
    for key, value in generated.items():
        assert fixture[key] == value, key
    assert corpus.dumps(fixture) + "\n" == (RANKING / "FX-RANK-0.json").read_text(encoding="utf-8")


# MUP-ONA T-RANK-0
def test_sqlite_without_sqlite_vec_reproduces_the_recorded_results(
    fixture: dict[str, Any], tmp_path: Path
) -> None:
    measure = _module("measure_sqlite")
    measured = measure.measure(fixture, tmp_path, vec=False)
    recorded = fixture["results"]["sqlite-novec"]
    assert measured["runs"] == recorded["runs"]
    assert measured["notes"] == recorded["meta"]["notes"]


# MUP-ONA T-RANK-0
def test_sqlite_with_sqlite_vec_reproduces_the_recorded_results(
    fixture: dict[str, Any], tmp_path: Path
) -> None:
    if not HAS_VEC:
        pytest.skip("sqlite-vec cannot be loaded here")
    measure = _module("measure_sqlite")
    measured = measure.measure(fixture, tmp_path, vec=True)
    recorded = fixture["results"]["sqlite"]
    assert measured["runs"] == recorded["runs"]
    assert measured["notes"] == recorded["meta"]["notes"]
