"""Conformance cases of the ``Index`` contract: search, modes, scores, limits, hostile input, vector checks.

T-CONF-1 (``index.*`` without vectors and with them), T-CONF-2 (input limits and hostile full-text input),
T-CONF-3 (``vector.*``: vector validation, cosine scores, zero vectors). See ``Index.search`` for the
contract.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
import itertools
import math
import time
from typing import Any

from ...retrieval import MAX_QUERY_CHARS, MAX_QUERY_WORDS, Chunk, Hit
from ._core import Case, CaseFailure, Context, expect, expect_code, expect_equal, expect_error
from ._models import KeywordEmbeddings, cosine

__all__ = ["INDEX_CASES", "VECTOR_CASES"]

#: Three documents, one topic word each (one vector axis), all with the word "policy".
DOCS: tuple[tuple[str, str], ...] = (
    ("warranty.md", "Warranty policy: refunds are possible within 30 days of purchase. alpha"),
    ("shipping.md", "Shipping policy: delivery takes five working days, express costs extra. beta"),
    ("privacy.md", "Privacy policy: personal data is deleted after twelve months. gamma"),
)
HOSTILE_QUERIES: tuple[str, ...] = (
    "NEAR(refunds purchase)",
    "text: refunds",
    "title: refunds",
    '"refunds',
    "refunds*",
    "^refunds",
    "refunds AND OR NOT purchase",
    "-refunds +purchase",
    "(refunds) {purchase}",
    "refunds'; DROP TABLE index_chunks; --",
    "refunds % _ \\ ' \"",
    "١٢٣ ٤٥٦ refunds",
    "refunds ​﻿",
    "*",
    '""',
    "NEAR",
)


def _fill(index: Any) -> list[int]:
    ids: list[int] = []
    for source, text in DOCS:
        ids += index.add(
            text, source=source, title=source.removesuffix(".md").title(), metadata={"kind": "doc"}
        )
    return ids


def _count(index: Any) -> int | None:
    return len(index) if hasattr(index, "__len__") else None


def _sources(index: Any) -> list[str] | None:
    sources = getattr(index, "sources", None)
    return list(sources()) if callable(sources) else None


def _modes(context: Context) -> tuple[str, ...]:
    return ("fts", "vector", "hybrid") if "vectors" in context.capabilities else ("fts",)


def _ordered(hits: list[Hit], what: str) -> None:
    scores = [hit.score for hit in hits]
    expect(scores == sorted(scores, reverse=True), f"{what}: scores must not increase: {scores}")
    for left, right in itertools.pairwise(hits):
        if left.score == right.score:
            expect(
                left.id < right.id, f"{what}: equal scores must be ordered by chunk id: {left.id}, {right.id}"
            )


# ---- T-CONF-1: index basics -----------------------------------------------------------------------------


def add_and_search(context: Context) -> None:
    """Added chunks are found with their source, title, metadata and text; ids are distinct integers."""
    index = context.make()
    ids = _fill(index)
    expect(len(ids) == len(set(ids)) == len(DOCS), f"one distinct id per document, got {ids}")
    expect(all(isinstance(chunk_id, int) for chunk_id in ids), f"ids must be integers: {ids}")
    hits = index.search("refunds purchase", 3)
    expect(hits, "search found nothing")
    best = hits[0]
    expect_equal(best.source, "warranty.md", "best hit")
    expect_equal(best.title, "Warranty", "title")
    expect_equal(best.metadata, {"kind": "doc"}, "metadata")
    expect_equal(best.text, DOCS[0][1], "text")
    expect(best.id in ids, f"hit id {best.id} is not an added id")
    _ordered(hits, "search")
    expect(_count(index) in (None, 3), f"len() after three documents: {_count(index)}")
    expect(_sources(index) in (None, sorted(source for source, _ in DOCS)), f"sources(): {_sources(index)}")
    chunk_ids = index.add_chunks(
        [Chunk(text="Opening hours: nine to five.", source="hours.md", file_id="f1")]
    )
    expect_equal(len(chunk_ids), 1, "add_chunks ids")
    hit = index.search("opening hours", 1)[0]
    expect_equal((hit.source, hit.file_id, hit.title), ("hours.md", "f1", None), "add_chunks hit")
    expect_equal(index.add_chunks([Chunk(text="   ", source="blank.md")]), [], "a blank chunk is not stored")


def word_prefixes(context: Context) -> None:
    """Query words match as prefixes, case-insensitively ("refund" finds "refunds")."""
    index = context.make()
    _fill(index)
    for query, source in (("refund", "warranty.md"), ("DELIV", "shipping.md"), ("Personal", "privacy.md")):
        hits = index.retrieve(query, 3, mode="fts")
        expect(hits and hits[0].source == source, f"{query!r} must find {source} first, got {hits}")


def no_words_no_hits(context: Context) -> None:
    """A query without words finds nothing (no error) in full text."""
    index = context.make()
    _fill(index)
    for query in ("", "   ", "?!", "—", '* ^ " ( )'):
        expect_equal(index.search(query), [], f"search({query!r})")
        expect_equal(index.retrieve(query, 3, mode="fts"), [], f"retrieve({query!r}, mode='fts')")


def k_is_checked(context: Context) -> None:
    """``k`` limits the hits; ``k`` below 1 raises ``ValueError``; an unknown mode raises ``ValueError``."""
    index = context.make()
    _fill(index)
    expect(len(index.search("policy", 1)) == 1, "k=1 returns one hit")
    expect(len(index.search("policy", 2)) == 2, "k=2 returns two hits")
    expect_error(lambda: index.search("policy", 0), ValueError, "search(k=0)")
    expect_error(lambda: index.retrieve("policy", 0), ValueError, "retrieve(k=0)")
    expect_error(lambda: index.retrieve("policy", 3, mode="bm25"), ValueError, "an unknown mode")


def delete_source(context: Context) -> None:
    """``delete`` removes every chunk of a source and returns how many; an unknown source gives 0."""
    index = context.make()
    _fill(index)
    index.add("A second warranty chunk about repairs.", source="warranty.md")
    expect_equal(index.delete("warranty.md"), 2, "delete() of a source with two chunks")
    expect_equal(index.search("refunds repairs"), [], "search after delete")
    expect_equal(index.delete("warranty.md"), 0, "delete() again")
    expect_equal(index.delete("unknown.md"), 0, "delete() of an unknown source")
    expect([hit.source for hit in index.search("shipping", 1)] == ["shipping.md"], "other sources stay")
    expect(_count(index) in (None, 2), "len() after delete")
    expect(_sources(index) in (None, ["privacy.md", "shipping.md"]), "sources() after delete")


def add_file_replaces(context: Context) -> None:
    """``add_file`` of the same source replaces its chunks; a file that cannot be read keeps them."""
    index = context.make()
    first = context.file("notes.md", b"Notes: the cafeteria opens at eight.")
    index.add_file(first, source="notes")
    expect([hit.source for hit in index.search("cafeteria")] == ["notes"], "the first version is found")
    second = context.file("notes.md", b"Notes: the library opens at nine.")
    index.add_file(second, source="notes")
    expect_equal(index.search("cafeteria"), [], "the old version is gone")
    hits = index.search("library")
    expect(
        len(hits) == 1 and hits[0].source == "notes" and hits[0].title == "notes.md", f"new version: {hits}"
    )
    broken = context.file("broken.md", b"\xff\xfe not utf-8 \xff")
    expect_error(lambda: index.add_file(broken, source="notes"), ValueError, "a file that is not UTF-8")
    expect(len(index.search("library")) == 1, "a failed add_file keeps the chunks")


def sources_filter(context: Context) -> None:
    """``retrieve(sources=...)`` returns only hits of those sources; an empty list returns nothing."""
    index = (
        context.make(embeddings=KeywordEmbeddings()) if "vectors" in context.capabilities else context.make()
    )
    _fill(index)
    for mode in _modes(context):
        hits = index.retrieve("policy alpha beta gamma", 5, mode=mode, sources=["shipping.md"])
        expect(
            hits and {hit.source for hit in hits} == {"shipping.md"}, f"{mode}: only shipping.md, got {hits}"
        )
        expect_equal(index.retrieve("policy", 5, mode=mode, sources=[]), [], f"{mode}: sources=[]")
        expect_equal(
            index.retrieve("policy", 5, mode=mode, sources=["nope.md"]), [], f"{mode}: unknown source"
        )


def full_text_score(context: Context) -> None:
    """Full-text scores are reciprocal ranks ``1 / (60 + rank)``, with ``ranks={"fts": rank}`` (drift 5)."""
    index = context.make()
    _fill(index)
    for what, hits in (
        ("search", index.search("policy", 3)),
        ("fts", index.retrieve("policy", 3, mode="fts")),
    ):
        expect_equal(len(hits), 3, f"{what}: hits")
        for rank, hit in enumerate(hits, start=1):
            expect(
                math.isclose(hit.score, 1 / (60 + rank), rel_tol=0, abs_tol=1e-12),
                f"{what}: score {hit.score}",
            )
        _ordered(hits, what)
    for rank, hit in enumerate(index.retrieve("policy", 3, mode="fts"), start=1):
        expect_equal(hit.ranks, {"fts": rank}, "ranks of a full-text hit")


def source_limit(context: Context) -> None:
    """A search names at most 50,000 sources; more raise ``ValueError`` (drift 7)."""
    index = context.make()
    _fill(index)
    many = [f"source-{number}.md" for number in range(50_001)]
    expect_error(lambda: index.retrieve("policy", 3, sources=many), ValueError, "50,001 sources")
    hits = index.retrieve("policy", 3, sources=[*many[:999], "privacy.md"])
    expect([hit.source for hit in hits] == ["privacy.md"], f"1,000 sources are accepted, got {hits}")


def query_tokenization(context: Context) -> None:
    """Query words are ``tokenize``/``fold`` words: ``omega_psi`` is one word, not two (drift 8)."""
    index = context.make()
    index.add("Tokens: omega psi appear together here.", source="together.md")
    index.add("Tokens: omega appears alone here.", source="omega.md")
    index.add("Tokens: psi appears alone here.", source="psi.md")
    hits = index.retrieve("omega_psi", 5, mode="fts")
    expect_equal([hit.source for hit in hits], ["together.md"], "hits for 'omega_psi'")
    expect_equal(
        {hit.source for hit in index.retrieve("OMEGA", 5, mode="fts")}, {"together.md", "omega.md"}, "OMEGA"
    )


def default_mode(context: Context) -> None:
    """With an embedding model ``search``/``retrieve`` rank ``hybrid``; without one ``fts`` (drift 6)."""
    model = KeywordEmbeddings()
    index = context.make(embeddings=model)
    _fill(index)
    # "first" is a synonym of the topic "alpha": no document contains the word, the vectors find it.
    for what, hits in (("search", index.search("first", 3)), ("retrieve", index.retrieve("first", 3))):
        expect(
            hits and hits[0].source == "warranty.md", f"{what} with embeddings must rank hybrid, got {hits}"
        )
        expect("vector" in hits[0].ranks, f"{what}: the best hit has a vector rank, got {hits[0].ranks}")
    plain = context.make("conformance-b")
    _fill(plain)
    expect_equal(plain.search("first"), [], "search without embeddings ranks full text")


def vector_search(context: Context) -> None:
    """``vector`` ranks by similarity (ranks ``{"vector": n}``); ``hybrid`` fuses scores into [0, 1]."""
    index = context.make(embeddings=KeywordEmbeddings())
    _fill(index)
    hits = index.retrieve("beta", 3, mode="vector")
    expect(hits and hits[0].source == "shipping.md", f"vector: shipping.md first, got {hits}")
    expect_equal(hits[0].ranks, {"vector": 1}, "vector ranks")
    _ordered(hits, "vector")
    hybrid = index.retrieve("beta delivery", 3, mode="hybrid")
    expect(hybrid and hybrid[0].source == "shipping.md", f"hybrid: shipping.md first, got {hybrid}")
    expect(all(0.0 <= hit.score <= 1.0 for hit in hybrid), f"hybrid scores in [0, 1]: {hybrid}")
    expect(set(hybrid[0].ranks) == {"fts", "vector"}, f"hybrid ranks of a hit both found: {hybrid[0].ranks}")
    _ordered(hybrid, "hybrid")


def no_fallback(context: Context) -> None:
    """``vector`` and ``hybrid`` never fall back to full text when there are no vectors."""
    plain = context.make()
    _fill(plain)
    for mode in ("vector", "hybrid"):
        expect_code(
            partial(plain.retrieve, "refunds", 3, mode=mode),
            f"{mode} without an embedding model",
            "vector_unavailable",
            "mode_unsupported",
        )
    if "vectors" in context.capabilities:
        empty = context.make("conformance-b", embeddings=KeywordEmbeddings())
        expect_code(
            lambda: empty.retrieve("alpha", 3, mode="vector"), "vector without vectors", "vector_unavailable"
        )


# ---- T-CONF-2: input limits and hostile input ---------------------------------------------------------


def _all_ways(index: Any, context: Context, query: str) -> list[tuple[str, Callable[[], object]]]:
    ways: list[tuple[str, Callable[[], object]]] = [("search", lambda: index.search(query))]
    ways += [(f"retrieve {mode}", partial(index.retrieve, query, 3, mode=mode)) for mode in _modes(context)]
    return ways


def _limit_index(context: Context) -> Any:
    index = context.make(embeddings=KeywordEmbeddings())
    _fill(index)
    return index


def query_characters(context: Context) -> None:
    """A query over ``MAX_QUERY_CHARS`` (1000) characters raises ``query.too_long``; 1000 are accepted."""
    index = _limit_index(context)
    for what, call in _all_ways(index, context, "refunds " + "x" * (MAX_QUERY_CHARS - 7)):
        expect_code(call, f"{what} with {MAX_QUERY_CHARS + 1} characters", "query.too_long")
    for _, call in _all_ways(index, context, "refunds " + "x" * (MAX_QUERY_CHARS - 8)):
        call()  # exactly at the limit


def query_words_limit(context: Context) -> None:
    """A query over ``MAX_QUERY_WORDS`` (64) distinct words raises ``query.too_long``; 64 or repeats pass."""
    index = _limit_index(context)
    too_many = " ".join(f"w{number}" for number in range(MAX_QUERY_WORDS + 1))
    for what, call in _all_ways(index, context, too_many):
        expect_code(call, f"{what} with {MAX_QUERY_WORDS + 1} words", "query.too_long")
    for query in (" ".join(f"w{number}" for number in range(MAX_QUERY_WORDS)), "refunds " * 100):
        for _, call in _all_ways(index, context, query):
            call()


def huge_query(context: Context) -> None:
    """A 100,000-character, 5,000-word query is refused with ``query.too_long`` within a second."""
    index = _limit_index(context)
    query = " ".join(f"word{number}" for number in range(5_000)).ljust(100_000, "x")
    for what, call in _all_ways(index, context, query):
        started = time.monotonic()
        expect_code(call, what, "query.too_long")
        expect(time.monotonic() - started < 1.0, f"{what}: the refusal took longer than a second")


def hostile_queries(context: Context) -> None:
    """FTS syntax and odd characters in a query are searched as words: no error, no column filter."""
    index = (
        context.make(embeddings=KeywordEmbeddings()) if "vectors" in context.capabilities else context.make()
    )
    _fill(index)
    for query in HOSTILE_QUERIES:
        for what, call in _all_ways(index, context, query):
            try:
                call()
            except Exception as exc:
                raise CaseFailure(f"{what}({query!r}) raised {type(exc).__name__}: {exc}") from exc
    for query in ("NEAR(refunds purchase)", "text: refunds", "title: refunds", "refunds'; DROP TABLE x; --"):
        hits = index.retrieve(query, 3, mode="fts")
        expect(
            hits and hits[0].source == "warranty.md", f"{query!r} must find warranty.md as words, got {hits}"
        )
    expect(index.search("refunds"), "the index still works")


def hostile_values(context: Context) -> None:
    """Texts, sources, titles and metadata with quotes, semicolons and SQL are stored and found as data."""
    index = context.make()
    text = "injectionprobe ' ; -- $$ \\ % _ \" ` /* */ DROP TABLE x; SELECT 1 OR 1=1"
    chunk = Chunk(
        text=text,
        source="src'; DROP TABLE index_chunks; --",
        title='Title "with" quotes',
        metadata={"key'": 'value"; DELETE', "x": "$$"},
    )
    index.add_chunks([chunk, Chunk(text="unrelated text " + "z" * 200_000, source="big.md")])
    hits = index.search("injectionprobe")
    expect(len(hits) == 1, f"one hit, got {hits}")
    hit = hits[0]
    expect_equal(
        (hit.text, hit.source, hit.title, hit.metadata),
        (text, chunk.source, chunk.title, chunk.metadata),
        "round trip",
    )
    expect_equal(index.delete(chunk.source), 1, "delete of the hostile source")
    expect([h.source for h in index.search("unrelated")] == ["big.md"], "a 200,000-character chunk is kept")


# ---- T-CONF-3: vector checks ------------------------------------------------------------------------------


def _vector_index(context: Context, model: KeywordEmbeddings | None = None) -> Any:
    index = context.make(embeddings=model or KeywordEmbeddings())
    _fill(index)
    return index


def non_finite_documents(context: Context) -> None:
    """A document vector with NaN or infinity raises ``vector.invalid``; nothing is stored (drift 2)."""
    index = _vector_index(context)
    before = _count(index)
    for marker in ("nanvector", "infvector", "neginfvector"):
        expect_code(
            partial(index.add, f"Bad {marker} chunk.", source=f"{marker}.md"), marker, "vector.invalid"
        )
        expect_equal(index.retrieve(marker, 3, mode="fts"), [], f"no {marker} chunk is stored")
    expect_equal(_count(index), before, "len() after refused vectors")


def non_finite_query(context: Context) -> None:
    """A query vector with NaN or infinity raises ``vector.invalid`` (drift 2)."""
    index = _vector_index(context)
    for marker in ("nanvector", "infvector"):
        for mode in ("vector", "hybrid"):
            expect_code(partial(index.retrieve, marker, 3, mode=mode), f"{mode} {marker}", "vector.invalid")


def wrong_dimension_documents(context: Context) -> None:
    """A document vector of another length raises ``vector.dimension``; nothing is stored (drift 2)."""
    index = _vector_index(context)
    before = _count(index)
    expect_code(
        lambda: index.add("A shortvector chunk.", source="short.md"), "short vector", "vector.dimension"
    )
    expect_equal(_count(index), before, "len() after a refused vector")
    expect_equal(index.retrieve("shortvector", 3, mode="fts"), [], "no short chunk is stored")


def wrong_dimension_query(context: Context) -> None:
    """A query vector of another length raises ``vector.dimension`` (drift 2)."""
    index = _vector_index(context)
    for mode in ("vector", "hybrid"):
        expect_code(partial(index.retrieve, "shortvector", 3, mode=mode), mode, "vector.dimension")


def zero_documents(context: Context) -> None:
    """A document with a zero vector is stored for full text but is never a vector hit (drift 3)."""
    index = _vector_index(context)
    index.add("The zerovector chunk about gardens.", source="zero.md")
    hits = index.retrieve("alpha", 10, mode="vector")
    expect("zero.md" not in {hit.source for hit in hits}, f"a zero vector is never a vector hit: {hits}")
    expect(hits and hits[0].source == "warranty.md", f"the other vectors still rank: {hits}")
    expect(all(not math.isnan(hit.score) for hit in hits), "no NaN score")
    expect(
        [hit.source for hit in index.retrieve("gardens", 3, mode="fts")] == ["zero.md"], "full text finds it"
    )


def zero_query(context: Context) -> None:
    """A query with a zero vector finds nothing by vector (no error) (drift 3)."""
    index = _vector_index(context)
    expect_equal(index.retrieve("zerovector", 5, mode="vector"), [], "vector hits of a zero query")


def _scored(context: Context, docs: list[tuple[str, str]], query: str, *, check_cosine: bool) -> list[Hit]:
    model = KeywordEmbeddings()
    index = context.make(embeddings=model)
    for source, text in docs:
        index.add(text, source=source)
    hits: list[Hit] = index.retrieve(query, len(docs), mode="vector")
    expect_equal(len(hits), len(docs), "vector hits")
    vectors = {source: model.vector(text) for source, text in docs}
    for hit in hits if check_cosine else ():
        expected = cosine(model.vector(query), vectors[hit.source])
        expect(
            math.isclose(hit.score, expected, abs_tol=1e-5),
            f"{hit.source}: score {hit.score}, cosine {expected}",
        )
    return hits


def cosine_scores(context: Context) -> None:
    """Vector scores are cosine similarities, whatever the vector length (not dot products) (drift 4)."""
    docs = [
        ("loud.md", "A loudvector chunk about alpha."),
        ("plain.md", "A plain chunk about alpha and beta."),
        ("other.md", "Another chunk about gamma."),
    ]
    hits = _scored(context, docs, "alpha", check_cosine=True)
    expect_equal(hits[0].source, "loud.md", "the parallel vector ranks first")


def rounded_scores(context: Context) -> None:
    """Vector scores are rounded to 6 digits (drift 5)."""
    docs = [("one.md", "alpha"), ("two.md", "alpha beta gamma"), ("three.md", "beta delta")]
    hits = _scored(context, docs, "alpha beta", check_cosine=False)
    for hit in hits:
        expect(
            hit.score == round(hit.score, 6), f"{hit.source}: score {hit.score!r} is not rounded to 6 digits"
        )


def vector_ties(context: Context) -> None:
    """Equal vector scores are ordered by chunk id, ascending; normalized models rank by cosine too."""
    model = KeywordEmbeddings("conformance-normalized", normalized=True)
    index = context.make(embeddings=model)
    ids = [index.add(f"Chunk {number} about epsilon.", source=f"tie-{number}.md")[0] for number in range(4)]
    hits = index.retrieve("epsilon", 4, mode="vector")
    expect_equal([hit.id for hit in hits], sorted(ids), "order of equal scores")
    expect(all(math.isclose(hit.score, 1.0, abs_tol=1e-6) for hit in hits), f"cosine 1: {hits}")


def _case(name: str, run: Callable[[Context], None], *needs: str, security: bool = False) -> Case:
    return Case(name, frozenset(needs), run, "index", security)


INDEX_CASES: tuple[Case, ...] = (
    _case("index.add_and_search", add_and_search),
    _case("index.word_prefixes", word_prefixes),
    _case("index.no_words_no_hits", no_words_no_hits),
    _case("index.k_is_checked", k_is_checked),
    _case("index.delete_removes_a_source", delete_source),
    _case("index.add_file_replaces_the_source", add_file_replaces),
    _case("index.sources_limit_the_hits", sources_filter),
    _case("index.full_text_score_is_the_reciprocal_rank", full_text_score),
    _case("index.too_many_sources_are_refused", source_limit),
    _case("index.query_words_follow_tokenize", query_tokenization),
    _case("index.vector_modes_never_fall_back", no_fallback),
    _case("index.default_mode_is_hybrid_with_embeddings", default_mode, "vectors"),
    _case("index.vector_and_hybrid_search", vector_search, "vectors"),
    _case("index.query_over_the_character_limit", query_characters, security=True),
    _case("index.query_over_the_word_limit", query_words_limit, security=True),
    _case("index.huge_query_is_refused_quickly", huge_query, security=True),
    _case("index.hostile_queries_are_words", hostile_queries, security=True),
    _case("index.hostile_values_are_data", hostile_values, security=True),
)
"""T-CONF-1 and T-CONF-2: the ``Index`` contract (the vector cases need ``vectors``)."""

VECTOR_CASES: tuple[Case, ...] = (
    _case("vector.non_finite_document_refused", non_finite_documents, "vectors", security=True),
    _case("vector.non_finite_query_refused", non_finite_query, "vectors", security=True),
    _case("vector.wrong_dimension_document_refused", wrong_dimension_documents, "vectors", security=True),
    _case("vector.wrong_dimension_query_refused", wrong_dimension_query, "vectors", security=True),
    _case("vector.zero_document_is_never_a_hit", zero_documents, "vectors", security=True),
    _case("vector.zero_query_finds_nothing", zero_query, "vectors", security=True),
    _case("vector.score_is_the_cosine", cosine_scores, "vectors"),
    _case("vector.scores_are_rounded_to_6_digits", rounded_scores, "vectors"),
    _case("vector.ties_go_to_the_lower_chunk_id", vector_ties, "vectors"),
)
"""T-CONF-3: vector validation and vector scores (need ``vectors``)."""
