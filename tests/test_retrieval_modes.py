"""Retrieval modes (fts / vector / hybrid), embedding spaces, scoping and reindexing of SQLiteIndex."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sqlite3
import threading

import pytest

from orkexa import SQLiteIndex
from orkexa.embeddings import CallableEmbeddings, EmbeddingIdentity, EmbeddingModel
from orkexa.retrieval import Chunk, RetrievalError

HAS_VEC = importlib.util.find_spec("sqlite_vec") is not None and hasattr(
    sqlite3.connect(":memory:"), "enable_load_extension"
)
pytestmark = pytest.mark.skipif(not HAS_VEC, reason="sqlite-vec cannot be loaded here")

# A tiny "semantic" space: words of one concept share a dimension, so synonyms are close.
CONCEPTS = [
    {"refund", "money", "back", "reimbursement", "erstattung", "geld"},
    {"warranty", "guarantee", "garantie", "defect"},
    {"shipping", "delivery", "lieferung", "versand", "parcel"},
    {"holiday", "vacation", "leave", "urlaub"},
]


class ConceptEmbeddings(EmbeddingModel):
    def __init__(self, name: str = "concepts", digest: str = "d1", flip: bool = False) -> None:
        self.name = name
        self.digest = digest
        self.flip = flip
        self.calls = 0

    def _vector(self, text: str) -> list[float]:
        words = {word.strip(".,?!").lower() for word in text.split()}
        vector = [float(len(words & concept)) for concept in CONCEPTS] + [0.01]
        if self.flip:
            vector.reverse()
        norm = math.sqrt(sum(x * x for x in vector))
        return [x / norm for x in vector]

    def embed_documents(self, texts):  # type: ignore[no-untyped-def]
        self.calls += 1
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def identity(self) -> EmbeddingIdentity:
        return EmbeddingIdentity(
            provider="test", model=self.name, dimensions=len(CONCEPTS) + 1, digest=self.digest
        )


DOCS = [
    Chunk(text="Customers get their money back within 30 days.", source="returns.md"),
    Chunk(text="The guarantee covers every defect for two years.", source="warranty.md"),
    Chunk(text="Parcels leave the warehouse within three days.", source="shipping.md"),
    Chunk(text="Employees have 30 days of vacation per year.", source="hr.md"),
]


def make(tmp_path: Path, model: EmbeddingModel | None = None, **kwargs) -> SQLiteIndex:  # type: ignore[no-untyped-def]
    index = SQLiteIndex(tmp_path / "kb.db", embeddings=model or ConceptEmbeddings(), **kwargs)
    index.add_chunks(DOCS)
    return index


def test_the_three_modes_rank_differently(tmp_path: Path) -> None:
    index = make(tmp_path)
    # No word of the query occurs in any document: full text finds nothing, vectors find the refund rule.
    assert index.retrieve("reimbursement", 2, mode="fts") == []
    vector = index.retrieve("reimbursement", 2, mode="vector")
    assert vector[0].source == "returns.md"
    assert vector[0].ranks == {"vector": 1}
    assert 0.9 < vector[0].score <= 1.0
    # "30 days" matches two documents by words; hybrid adds the semantic ranking on top.
    fts = index.retrieve("30 days vacation", 4, mode="fts")
    assert {hit.source for hit in fts} == {"returns.md", "hr.md", "shipping.md"}
    hybrid = index.retrieve("30 days vacation", 4, mode="hybrid")
    assert hybrid[0].source == "hr.md"
    assert set(hybrid[0].ranks) == {"fts", "vector"}  # really fused from both rankings
    assert len({hit.id for hit in hybrid}) == len(hybrid)  # no duplicates


def test_default_mode_is_hybrid_with_embeddings_and_fts_without(tmp_path: Path) -> None:
    assert make(tmp_path).mode == "hybrid"
    plain = SQLiteIndex(tmp_path / "plain.db")
    plain.add_chunks(DOCS)
    assert plain.mode == "fts"
    with pytest.raises(RetrievalError) as error:
        plain.retrieve("refund", mode="vector")
    assert error.value.code == "vector_unavailable"  # never answered by full text instead


def test_sources_limit_candidates_before_ranking(tmp_path: Path) -> None:
    index = make(tmp_path)
    for mode in ("fts", "vector", "hybrid"):
        hits = index.retrieve("money back guarantee 30 days", 5, mode=mode, sources=["hr.md", "warranty.md"])
        assert hits and {hit.source for hit in hits} <= {"hr.md", "warranty.md"}
    assert index.retrieve("money", 5, mode="vector", sources=[]) == []


def test_tool_uses_mode_and_sources(tmp_path: Path) -> None:
    index = make(tmp_path)
    tool = index.tool(mode="vector", sources=["returns.md"])
    result = tool.function(query="reimbursement", k=3)
    assert result[0]["source"] == "returns.md"
    assert len(result) == 1


def test_changing_the_embedding_model_needs_a_reindex(tmp_path: Path) -> None:
    index = make(tmp_path)
    other = SQLiteIndex(
        tmp_path / "kb.db", embeddings=ConceptEmbeddings(digest="d2")
    )  # same dims, other build
    with pytest.raises(RetrievalError) as error:
        other.retrieve("reimbursement", mode="vector")
    assert error.value.code == "reindex_required"
    with pytest.raises(RetrievalError):
        other.add_chunks([Chunk(text="New refund rule.", source="new.md")])
    assert len(other) == 4  # the failed add left nothing behind
    status = other.reindex()
    assert status.complete and status.active and status.active["digest"] == "d2"
    assert other.retrieve("reimbursement", 1, mode="vector")[0].source == "returns.md"
    # The old space is gone: the first index now reports the mismatch.
    with pytest.raises(RetrievalError):
        index.retrieve("reimbursement", mode="vector")


def test_reindex_is_resumable_and_keeps_the_old_space_until_complete(tmp_path: Path) -> None:
    index = make(tmp_path)
    new = ConceptEmbeddings(name="concepts-v2", flip=True)
    cancel = threading.Event()
    seen: list[tuple[int, int]] = []

    def progress(done: int, total: int) -> None:
        seen.append((done, total))
        cancel.set()  # stop after the first batch

    with pytest.raises(RetrievalError) as error:
        index.reindex(new, batch_size=2, progress=progress, cancel=cancel)
    assert error.value.code == "cancelled"
    status = index.vector_status()
    assert status.active and status.active["model"] == "concepts" and status.embedded == 4
    assert status.building and status.building_embedded == 2
    assert (
        index.retrieve("reimbursement", 1, mode="vector")[0].source == "returns.md"
    )  # old space still serves
    finished = index.reindex(new, batch_size=2)
    assert seen == [(2, 4)]
    assert finished.active and finished.active["model"] == "concepts-v2" and finished.building is None
    assert index.retrieve("reimbursement", 1, mode="vector")[0].source == "returns.md"


def test_first_vectors_for_an_existing_fts_index(tmp_path: Path) -> None:
    plain = SQLiteIndex(tmp_path / "kb.db")
    plain.add_chunks(DOCS)
    model = ConceptEmbeddings()
    vectors = SQLiteIndex(tmp_path / "kb.db", embeddings=model)
    with pytest.raises(RetrievalError) as error:
        vectors.add_chunks([Chunk(text="Extra.", source="x.md")])
    assert error.value.code == "reindex_required"
    vectors.reindex()
    vectors.add_chunks([Chunk(text="Delivery takes a week abroad.", source="abroad.md")])
    assert vectors.vector_status().complete
    assert {hit.source for hit in vectors.retrieve("lieferung", 2, mode="vector")} == {
        "shipping.md",
        "abroad.md",
    }


def test_delete_removes_vectors_and_drop_vectors_keeps_full_text(tmp_path: Path) -> None:
    index = make(tmp_path)
    assert index.delete("returns.md") == 1
    assert all(hit.source != "returns.md" for hit in index.retrieve("money", 4, mode="vector"))
    assert index.drop_vectors() == 3
    assert index.vector_status().active is None
    assert index.retrieve("guarantee", 1, mode="fts")[0].source == "warranty.md"


def test_namespaces_keep_separate_spaces(tmp_path: Path) -> None:
    a = SQLiteIndex(tmp_path / "kb.db", namespace="a", embeddings=ConceptEmbeddings(name="m1"))
    b = SQLiteIndex(
        tmp_path / "kb.db",
        namespace="b",
        embeddings=CallableEmbeddings(
            lambda texts: [[1.0, 0.0, 0.0] for _ in texts], name="m2", dimensions=3
        ),
    )
    a.add_chunks(DOCS[:1])
    b.add_chunks(DOCS[1:2])  # another model with another size in the same file
    assert [hit.source for hit in a.retrieve("money", 5, mode="vector")] == ["returns.md"]
    assert [hit.source for hit in b.retrieve("anything", 5, mode="vector")] == ["warranty.md"]


def test_legacy_embedder_keeps_working_and_honours_modes(tmp_path: Path) -> None:
    model = ConceptEmbeddings()
    index = SQLiteIndex(tmp_path / "old.db", embedder=model, query_embedder=model.query_embedder())
    index.add_chunks(DOCS)
    assert index.mode == "hybrid"
    assert index.search("reimbursement", 1)[0].source == "returns.md"
    assert index.retrieve("reimbursement", 1, mode="vector", sources=["returns.md"])[0].source == "returns.md"
    assert index.retrieve("reimbursement", 1, mode="fts") == []


def test_large_dataset_and_ties_are_deterministic(tmp_path: Path) -> None:
    index = SQLiteIndex(tmp_path / "big.db", embeddings=ConceptEmbeddings())
    index.add_chunks(Chunk(text=f"Parcel {n} delivery note", source=f"s{n % 50}") for n in range(2000))
    first = index.retrieve("delivery", 10, mode="hybrid")
    assert first == index.retrieve("delivery", 10, mode="hybrid")
    assert len(first) == 10
