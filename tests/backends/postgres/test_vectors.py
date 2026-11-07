"""Vectors of the PostgreSQL index in mode A (Python ranks) and mode B (pgvector ranks) (MUP-ONA B5:
T-PG-1..4, T-RANK-1, plan 10.5).

Taken over from nodestro-runtime ``tests/test_pg_drift.py`` (76f9cb2, the ten drift fixes of MP-B3) with
Orkexa's classes: every case runs with vectors ranked by the database (``database``: pgvector, required in
``pg-b`` by ``ORKEXA_REQUIRE_PGVECTOR=1``, skipped in ``pg-a``) and in Python (``runtime``): NaN/infinity/zero
vectors stored by older code are left out and counted (T-PG-3), wrong dimensions, normalized and raw
identities, reindex races, the source limit, the query words; ``refresh()`` after pgvector is removed and
installed again (T-PG-4). T-RANK-1 measures the ranking corpus FX-RANK-0 both ways on a server with pgvector
and checks the equivalence criteria of plan 10.6.
"""

from __future__ import annotations

from collections.abc import Callable
import itertools
import math
from pathlib import Path
from typing import Any

import pytest

from orkexa import Chunk
from orkexa.backends.postgres import MAX_SOURCES, PostgresIndex, verify_schema
from orkexa.embeddings import CallableEmbeddings
from orkexa.retrieval import RetrievalError
from orkexa.testing.conformance import KeywordEmbeddings

from . import support

pytestmark = pytest.mark.postgres

TENANT = "drift-tenant"


@pytest.fixture(params=["database", "runtime"])
def make(request: pytest.FixtureRequest, db: support.Database) -> Callable[..., Any]:
    """Opens a ``PostgresIndex`` that ranks vectors as the server can (``database``: pgvector, which
    ``ORKEXA_REQUIRE_PGVECTOR=1`` demands) or always in Python (``runtime``, a server without pgvector)."""
    if request.param == "database" and not PostgresIndex(db.source, TENANT, layout=db.layout).pgvector():
        support.require_pgvector("the database ranks vectors only with pgvector")

    def open_index(tenant: str = TENANT, **kwargs: Any) -> PostgresIndex:
        index = PostgresIndex(db.source, tenant, layout=db.layout, namespace="drift", **kwargs)
        if request.param == "runtime":
            index._pgvector = False
        index.test_db = db  # type: ignore[attr-defined]
        return index

    return open_index


# ---- drift 1: the embedding identity is read again
# ------------------------------------------------------------


# MUP-ONA T-PG-1
def test_a_model_updated_during_a_reindex_discards_the_new_space(make: Callable[..., Any]) -> None:
    old = KeywordEmbeddings("drift-old")
    index = make(embeddings=old)
    for number in range(3):
        index.add(f"Chunk {number} about alpha.", source=f"c{number}.md")
    new = KeywordEmbeddings("drift-new")

    def update_on_the_server(_texts: Any) -> None:
        new.served_digest = "v2"  # the cached identity is still v1

    new.on_embed = update_on_the_server
    with pytest.raises(RetrievalError) as error:
        index.reindex(new, batch_size=1)
    assert error.value.code == "reindex_required"
    status = index.vector_status()
    assert status.active == old.identity().to_dict() and status.building is None
    assert status.embedded == 3
    new.on_embed = None
    status = index.reindex(new)
    assert status.active is not None and status.active["digest"] == "v2" and status.complete


# ---- drift 2: NaN, infinity and wrong lengths
# ------------------------------------------------------------------


def _set_vector(index: Any, source: str, values: str) -> None:
    """Write a vector by SQL, as older code could store it (``values``: a ``real[]`` literal)."""
    index.test_db.query(
        "UPDATE t_chunk_vectors SET embedding = %s::real[] WHERE tenant_id = %s AND chunk_id IN"
        " (SELECT id FROM t_chunks WHERE tenant_id = %s AND source = %s)",
        (values, index.tenant_id, index.tenant_id, source),
    )


# MUP-ONA T-PG-1
def test_stored_non_finite_vectors_are_left_out_and_kept(make: Callable[..., Any]) -> None:
    index = make(embeddings=KeywordEmbeddings())
    for source, word in (("a.md", "alpha"), ("nan.md", "alpha beta"), ("inf.md", "alpha gamma")):
        index.add(f"A chunk about {word}.", source=source)
    _set_vector(index, "nan.md", "{NaN,1,0,0,0,0,0,0}")
    _set_vector(index, "inf.md", "{-Infinity,1,0,0,0,0,0,0}")
    for mode in ("vector", "hybrid"):
        hits = index.retrieve("alpha", 10, mode=mode)
        assert hits and hits[0].source == "a.md", (mode, hits)
        found = {hit.source for hit in hits if "vector" in hit.ranks}
        assert found == {"a.md"}, (mode, hits)
    stored = index.test_db.query(
        "SELECT c.source, v.embedding[1]::text FROM t_chunk_vectors v"
        " JOIN t_chunks c ON c.id = v.chunk_id"
        " WHERE v.tenant_id = %s AND c.tenant_id = %s AND c.source <> 'a.md' ORDER BY c.source",
        (index.tenant_id, index.tenant_id),
    )
    assert stored == [("inf.md", "-Infinity"), ("nan.md", "NaN")]  # left out, not rewritten
    rows = index.test_db.query(
        "SELECT count(*) FROM t_chunk_vectors WHERE tenant_id = %s", (index.tenant_id,)
    )
    assert rows == [(3,)]


# ---- drift 3: zero vectors
# -------------------------------------------------------------------------------------


# MUP-ONA T-PG-1
def test_zero_vectors_are_stored_but_never_ranked(make: Callable[..., Any]) -> None:
    index = make(embeddings=KeywordEmbeddings())
    index.add("A chunk about alpha.", source="a.md")
    index.add("The zerovector chunk about alpha gardens.", source="zero.md")
    status = index.vector_status()
    assert (status.chunks, status.embedded) == (2, 2)  # stored: the chunk is not missing a vector
    for mode in ("vector", "hybrid"):
        hits = index.retrieve("alpha", 10, mode=mode)
        assert [hit.source for hit in hits if "vector" in hit.ranks] == ["a.md"], (mode, hits)
    assert index.retrieve("zerovector", 10, mode="vector") == []  # a zero query vector finds nothing
    hybrid = index.retrieve("zerovector gardens", 10, mode="hybrid")
    assert [hit.source for hit in hybrid] == ["zero.md"] and hybrid[0].ranks == {"fts": 1}
    assert all(hit.score == hit.score for hit in hybrid)  # never NaN


# ---- drift 4: the cosine in both paths
# -------------------------------------------------------------------------


# MUP-ONA T-PG-1
@pytest.mark.parametrize("normalized", [False, True])
def test_both_paths_score_the_cosine(make: Callable[..., Any], normalized: bool) -> None:
    from orkexa.testing.conformance._models import cosine

    model = KeywordEmbeddings("drift-cosine", normalized=normalized)
    index = make(embeddings=model)
    texts = {
        "loud.md": "A loudvector chunk about alpha.",
        "two.md": "Alpha and beta.",
        "three.md": "Alpha beta gamma.",
    }
    for source, text in texts.items():
        index.add(text, source=source)
    hits = index.retrieve("alpha", 3, mode="vector")
    assert [hit.source for hit in hits] == ["loud.md", "two.md", "three.md"]
    for hit in hits:
        assert abs(hit.score - cosine(model.vector("alpha"), model.vector(texts[hit.source]))) <= 1e-6, hit


# ---- drift 5: the contract's scores
# --------------------------------------------------------------------------


# MUP-ONA T-PG-1
def test_scores_follow_the_contract_in_every_mode(make: Callable[..., Any]) -> None:
    from orkexa.retrieval import fuse_rrf

    index = make(embeddings=KeywordEmbeddings(), fusion="rrf")
    for number, words in enumerate(("alpha", "alpha beta", "alpha beta gamma", "beta gamma delta")):
        index.add(f"Chunk {number}: {words} policy.", source=f"c{number}.md")
    fts = index.retrieve("policy alpha", 4, mode="fts")
    assert [hit.score for hit in fts] == [1 / (60 + rank) for rank in range(1, len(fts) + 1)]
    vector = index.retrieve("alpha beta", 4, mode="vector")
    assert all(hit.score == round(hit.score, 6) for hit in vector)
    hybrid = index.retrieve("alpha beta", 4, mode="hybrid")
    rankings = [
        [hit.id for hit in index.retrieve("alpha beta", 20, mode="fts")],
        [hit.id for hit in index.retrieve("alpha beta", 20, mode="vector")],
    ]
    assert [(hit.id, hit.score) for hit in hybrid] == fuse_rrf(rankings)[:4]


# ---- drift 6: the default mode
# --------------------------------------------------------------------------------


# MUP-ONA T-PG-1
def test_the_default_mode_follows_the_embedding_model(make: Callable[..., Any]) -> None:
    index = make(embeddings=KeywordEmbeddings())
    index.add("A chunk about alpha.", source="a.md")
    index.add("Another chunk about the first topic.", source="b.md")
    assert {name for hit in index.search("alpha", 2) for name in hit.ranks} == {"fts", "vector"}
    assert index.search("alpha", 2) == index.retrieve("alpha", 2, mode="hybrid")
    plain = make(tenant="drift-plain")
    plain.add("A chunk about alpha.", source="a.md")
    assert [hit.ranks for hit in plain.search("alpha")] == [{"fts": 1}]
    assert make(embeddings=KeywordEmbeddings(), mode="fts").search("alpha") == index.retrieve(
        "alpha", 5, mode="fts"
    )


# ---- drift 7: the source limit
# ---------------------------------------------------------------------------------


# MUP-ONA T-PG-1
def test_a_search_names_at_most_50000_sources(make: Callable[..., Any]) -> None:
    index = make(embeddings=KeywordEmbeddings())
    index.add("A chunk about alpha.", source="a.md")
    many = [f"s{number}.md" for number in range(MAX_SOURCES)]
    for mode in ("fts", "vector", "hybrid"):
        with pytest.raises(ValueError):
            index.retrieve("alpha", 3, mode=mode, sources=[*many, "a.md"])
        hits = index.retrieve("alpha", 3, mode=mode, sources=[*many[:-1], "a.md"])
        assert [hit.source for hit in hits] == ["a.md"], mode


# ---- drift 8: the query words of the contract
# -----------------------------------------------------------------


# MUP-ONA T-PG-1
def test_query_words_are_contract_words_with_folded_diacritics(make: Callable[..., Any]) -> None:
    index = make()
    index.add("The café_menu lists omega psi drinks.", source="menu.md")
    index.add("Omega alone, then psi alone.", source="apart.md")
    index.add("A Cafe with a menu.", source="cafe.md")
    assert [hit.source for hit in index.retrieve("omega_ps", 5, mode="fts")] == [
        "menu.md"
    ]  # parts in sequence
    assert [hit.source for hit in index.retrieve("cafe_menu", 5, mode="fts")] == ["menu.md"]
    # diacritics are folded on both sides, as SQLite's unicode61 remove_diacritics 2 does (a documented
    # deviation)
    assert {hit.source for hit in index.retrieve("CAFÉ", 5, mode="fts")} == {"menu.md", "cafe.md"}
    assert index.retrieve("_ __", 5, mode="fts") == []
    for hostile in ("omega & !psi", "omega:* | psi", "'omega' <-> psi", "omega\\ psi"):
        assert {hit.source for hit in index.retrieve(hostile, 5, mode="fts")} == {"menu.md", "apart.md"}, (
            hostile
        )


# ---- drift 9: a chunk deleted during a reindex
# ----------------------------------------------------------------


# MUP-ONA T-PG-1
def test_chunks_deleted_while_a_batch_is_embedded_get_no_vector(make: Callable[..., Any]) -> None:
    index = make()
    for number in range(6):
        index.add(f"Chunk {number} about alpha.", source=f"c{number}.md")
    model = KeywordEmbeddings("drift-delete")
    deleted: list[int] = []

    def delete_two(texts: Any) -> None:  # deletes one chunk of the batch at hand and one of a later batch
        if not deleted:
            deleted.extend((index.delete("c1.md"), index.delete("c4.md")))

    model.on_embed = delete_two
    status = index.reindex(model, batch_size=3)
    assert deleted == [1, 1]
    assert (status.chunks, status.embedded, status.building, status.complete) == (4, 4, None, True)
    rows = index.test_db.query(
        "SELECT count(*) FROM t_chunk_vectors WHERE tenant_id = %s", (index.tenant_id,)
    )
    assert rows == [(4,)]
    assert {hit.source for hit in index.retrieve("alpha", 10, mode="vector")} == {
        "c0.md",
        "c2.md",
        "c3.md",
        "c5.md",
    }


# ---- drift 10: vectors searches leave out are reported
# --------------------------------------------------------


# MUP-ONA T-PG-3
def test_vectors_left_out_are_counted_until_a_reindex_replaces_them(make: Callable[..., Any]) -> None:
    index = make(embeddings=KeywordEmbeddings())
    for source, word in (("a.md", "alpha"), ("nan.md", "beta"), ("inf.md", "gamma"), ("zero.md", "delta")):
        index.add(f"A chunk about {word}.", source=source)
    status = index.vector_status()
    assert (status.legacy, status.excluded) == (False, 0)
    _set_vector(index, "nan.md", "{NaN,1,0,0,0,0,0,0}")
    _set_vector(index, "inf.md", "{1,Infinity,0,0,0,0,0,0}")
    _set_vector(index, "zero.md", "{0,0,0,0,0,0,0,0}")
    status = index.vector_status()
    assert (status.legacy, status.excluded, status.embedded, status.chunks) == (True, 3, 4, 4)
    assert status.model_dump()["excluded"] == 3
    other = make(
        tenant="drift-other", embeddings=KeywordEmbeddings()
    )  # another tenant counts its own vectors
    other.add("A chunk about alpha.", source="a.md")
    assert (other.vector_status().legacy, other.vector_status().excluded) == (False, 0)
    status = index.reindex(KeywordEmbeddings("drift-replacement"))
    assert (status.legacy, status.excluded, status.complete) == (False, 0, True)
    assert {hit.source for hit in index.retrieve("beta gamma delta", 10, mode="vector")} >= {
        "nan.md",
        "inf.md",
    }


# ---- refresh(): pgvector installed or removed while the server runs
# ------------------------------------------


# MUP-ONA T-PG-4
def test_refresh_detects_pgvector_again(db: support.Database) -> None:
    index = PostgresIndex(
        db.source, TENANT, layout=db.layout, namespace="refresh", embeddings=KeywordEmbeddings()
    )
    index.add("A chunk about alpha.", source="a.md")
    index.add("A chunk about beta.", source="b.md")
    detected = index.pgvector()
    index._pgvector = not detected  # a stale detection
    index.refresh()
    assert index.pgvector() is detected
    if not detected:
        support.require_pgvector("T-PG-4 installs pgvector while the index is open")
    expected = [(hit.source, hit.score) for hit in index.retrieve("alpha", 2, mode="vector")]
    try:
        with support.admin() as admin:
            admin.execute("DROP EXTENSION vector")
        index.refresh()
        assert index.pgvector() is False
        assert verify_schema(db.source, db.layout).pgvector is False
        assert [(hit.source, hit.score) for hit in index.retrieve("alpha", 2, mode="vector")] == expected
    finally:
        with support.admin() as admin:
            admin.execute("CREATE EXTENSION IF NOT EXISTS vector SCHEMA public")
    assert index.pgvector() is False  # detected once, until refresh()
    assert verify_schema(db.source, db.layout).pgvector is True  # what an operator sees (MUP-ONA U4)
    index.refresh()
    assert index.pgvector() is True
    assert [(hit.source, hit.score) for hit in index.retrieve("alpha", 2, mode="vector")] == expected


# MUP-ONA T-PG-1
def test_a_model_without_refresh_identity_still_writes_and_reindexes(make: Callable[..., Any]) -> None:
    """A model with only ``embed_documents``, ``embed_query`` and ``identity`` (a duck-typed test double, as
    in nodestro-api's worker tests) caches no identity: writes and reindexes work without
    ``refresh_identity``."""
    keywords = KeywordEmbeddings("drift-duck")

    class Duck:
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [keywords.vector(text) for text in texts]

        def embed_query(self, text: str) -> list[float]:
            return keywords.vector(text)

        def identity(self) -> Any:
            return keywords.identity()

    index = make()
    index.add("A chunk about alpha.", source="a.md")
    status = index.reindex(Duck())
    assert status.complete
    index.embeddings = Duck()
    index.add("A chunk about beta.", source="b.md")
    assert [hit.source for hit in index.retrieve("beta", 1, mode="vector")] == ["b.md"]


# ---- T-RANK-1: pgvector and Python rank alike (plan 10.6)
# ---------------------------------------------------

RANKING = Path(__file__).resolve().parents[2] / "fixtures" / "ranking"
#: Plan 10.6: relative score difference below which two adjacent hits count as tied, and the score tolerance.
TIE_RELATIVE = 1e-6
SCORE_TOLERANCE = 1e-5


def _fixture() -> dict[str, Any]:
    import json

    data: dict[str, Any] = json.loads((RANKING / "FX-RANK-0.json").read_text(encoding="utf-8"))
    return data


def _embeddings(data: dict[str, Any], identity: str) -> CallableEmbeddings:
    """The fixture's embedding model for ``identity``: looks the vector of each text up in the corpus."""
    field = f"vector_{identity}"
    documents = {chunk["text"]: chunk[field] for chunk in data["chunks"]}
    queries = {query["text"]: query[field] for query in data["queries"]}
    spec = data["identities"][identity]
    return CallableEmbeddings(
        lambda texts: [documents[text] for text in texts],
        query_embedder=lambda texts: [queries[text] for text in texts],
        name=spec["name"],
        dimensions=spec["dimensions"],
        provider=spec["provider"],
        normalized=spec["normalized"],
    )


def _measure(index: PostgresIndex, data: dict[str, Any]) -> dict[str, Any]:
    """Every query in every mode: ``{query id: {mode: {"hits": [(key, id, score)]} or {"error": {...}}}}``."""
    results: dict[str, Any] = {}
    for query in data["queries"]:
        by_mode: dict[str, Any] = {}
        for mode in data["modes"]:
            try:
                hits = index.retrieve(
                    query["text"],
                    data["k"],
                    mode=None if mode == "default" else mode,
                    sources=query["sources"],
                )
            except RetrievalError as exc:
                by_mode[mode] = {"error": {"code": exc.code}}
            else:
                by_mode[mode] = {"hits": [(hit.metadata.get("key"), hit.id, hit.score) for hit in hits]}
        results[query["id"]] = by_mode
    return results


def _tied(left: float, right: float) -> bool:
    return abs(left - right) <= TIE_RELATIVE * max(abs(left), abs(right), 1e-12)


def equivalence_problems(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    """Where the results of one query and mode of Python (a) and pgvector (b) differ beyond plan 10.6 (empty:
    equivalent). Taken over from nodestro-runtime ``tests/test_ranking_compat.py`` (76f9cb2)."""
    if "hits" not in a or "hits" not in b:
        same_error = "error" in a and "error" in b and a["error"].get("code") == b["error"].get("code")
        return [] if same_error else [f"error in one path only: {a.get('error')} / {b.get('error')}"]
    hits_a, hits_b = a["hits"], b["hits"]
    problems: list[str] = []
    for name, hits in (("python", hits_a), ("pgvector", hits_b)):
        for (_, left_id, left), (_, right_id, right) in itertools.pairwise(hits):
            if any(math.isnan(score) for score in (left, right)):
                problems.append(f"{name}: NaN score")
            elif left < right:
                problems.append(f"{name}: scores increase ({left} < {right})")
            elif left == right and left_id > right_id:
                problems.append(f"{name}: equal scores not ordered by chunk id ({left_id}, {right_id})")
    if len(hits_a) != len(hits_b):
        return [*problems, f"{len(hits_a)} hits against {len(hits_b)}"]
    scores_a = {key: score for key, _, score in hits_a}
    scores_b = {key: score for key, _, score in hits_b}
    # The last group of tied scores may be cut at k differently; every other key must be in both.
    cut = len(hits_a)
    while cut > 1 and _tied(hits_a[cut - 2][2], hits_a[-1][2]):
        cut -= 1
    for key, _, _ in hits_a[: cut - 1]:
        if key not in scores_b:
            problems.append(f"{key} only in python")
    for key in scores_a.keys() & scores_b.keys():
        if abs(scores_a[key] - scores_b[key]) > SCORE_TOLERANCE:
            problems.append(f"{key}: score {scores_a[key]} against {scores_b[key]}")
    position_b = {key: position for position, (key, _, _) in enumerate(hits_b)}
    for (key_x, _, x), (key_y, _, y) in itertools.combinations(hits_a, 2):
        both = key_x in position_b and key_y in position_b
        if both and not _tied(x, y) and position_b[key_x] > position_b[key_y]:
            problems.append(f"{key_x} before {key_y} in python ({x} > {y}), not in pgvector")
    return problems


def test_the_equivalence_check_sees_a_difference() -> None:
    same = {"hits": [("a", 1, 0.9), ("b", 2, 0.8)]}
    assert equivalence_problems(same, same) == []
    assert equivalence_problems(same, {"hits": [("b", 2, 0.8), ("a", 1, 0.9)]})
    assert equivalence_problems(same, {"hits": [("a", 1, 0.9), ("b", 2, 0.79)]})
    assert equivalence_problems(same, {"error": {"code": "vector_unavailable"}})


# MUP-ONA T-RANK-1
@pytest.mark.parametrize("identity", ["norm", "raw"])
def test_pgvector_and_python_rank_fx_rank_0_alike(db: support.Database, identity: str) -> None:
    support.require_pgvector("T-RANK-1 compares pgvector's ranking with Python's")
    data = _fixture()
    model = _embeddings(data, identity)
    index = PostgresIndex(
        db.source, "fx-rank-tenant", layout=db.layout, namespace="fx-rank", embeddings=model
    )
    chunks = [
        Chunk(text=c["text"], source=c["source"], title=c["title"], metadata={"key": c["key"]})
        for c in data["chunks"]
    ]
    assert index.add_chunks(chunks) == list(range(1, len(chunks) + 1))
    assert index.pgvector()
    by_pgvector = _measure(index, data)
    index._pgvector = False
    by_python = _measure(index, data)
    problems = [
        f"{query}/{mode}: {problem}"
        for query, modes in by_python.items()
        for mode, result in modes.items()
        for problem in equivalence_problems(result, by_pgvector[query][mode])
    ]
    assert problems == []
    assert sum(len(result.get("hits", [])) for modes in by_python.values() for result in modes.values()) > 500
