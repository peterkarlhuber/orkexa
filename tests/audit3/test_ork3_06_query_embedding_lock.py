"""ORK3-06: a vector search embeds its query without holding the index lock.

Internal backend audit round 3 (2026-11-04). No network.
"""

from __future__ import annotations

from pathlib import Path
import threading
import time

import pytest

from orkexa import SQLiteIndex
from orkexa.embeddings import CallableEmbeddings
from orkexa.retrieval import RetrievalError


def _slow_queries(started: threading.Event, release: threading.Event, name: str = "s") -> CallableEmbeddings:
    def embed(texts: list[str]) -> list[list[float]]:
        if texts and texts[0].startswith("slow"):
            started.set()
            assert release.wait(10)
        return [[1.0, 0.0, 0.0] for _ in texts]

    return CallableEmbeddings(embed, name=name, dimensions=3)


@pytest.mark.parametrize("mode", ["vector", "hybrid"])
def test_writes_do_not_wait_for_a_query_embedding(tmp_path: Path, mode: str) -> None:
    started, release = threading.Event(), threading.Event()
    index = SQLiteIndex(tmp_path / "i.db", embeddings=_slow_queries(started, release))
    index.add("one", source="a")
    index.add("two", source="b")
    hits: list[object] = []
    search = threading.Thread(target=lambda: hits.extend(index.retrieve("slow query", mode=mode)))  # type: ignore[arg-type]
    search.start()
    try:
        assert started.wait(10)
        begin = time.monotonic()
        assert index.delete("b") == 1  # while the query is still being embedded
        assert time.monotonic() - begin < 1.0
    finally:
        release.set()
        search.join(10)
    assert [hit.source for hit in hits] == ["a"]  # type: ignore[attr-defined]
    index.close()


def test_a_reindex_during_the_query_embedding_is_detected(tmp_path: Path) -> None:
    started, release = threading.Event(), threading.Event()
    index = SQLiteIndex(tmp_path / "j.db", embeddings=_slow_queries(started, release))
    index.add("one", source="a")
    errors: list[BaseException] = []

    def search() -> None:
        try:
            index.retrieve("slow query", mode="vector")
        except RetrievalError as exc:
            errors.append(exc)

    thread = threading.Thread(target=search)
    thread.start()
    try:
        assert started.wait(10)
        other = CallableEmbeddings(lambda texts: [[0.0, 1.0, 0.0] for _ in texts], name="other", dimensions=3)
        configured = index.embeddings
        index.reindex(other)  # switches the active space; the search still holds the old model
        index.embeddings = configured
    finally:
        release.set()
        thread.join(10)
    assert len(errors) == 1 and errors[0].code == "reindex_required"  # type: ignore[attr-defined]
    index.close()
