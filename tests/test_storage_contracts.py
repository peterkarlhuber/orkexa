"""The generic storage contracts of MUP-ONA B1 (O1, O4): tokenize/fold, fuse_rrf, validate_vector, query
limits for every Index, capabilities, ReindexableIndex and the one set of error codes. No network."""

from __future__ import annotations

import ast
from collections.abc import Iterable
import inspect
import math
from pathlib import Path
import re
import time

import pytest

from orkexa import EphemeralMemory, FileMemory, LocalFileStore, SQLiteFileStore, SQLiteIndex, SQLiteMemory
from orkexa.embeddings import EmbeddingIdentity, validate_vector
from orkexa.retrieval import (
    MAX_QUERY_CHARS,
    MAX_QUERY_WORDS,
    RETRIEVAL_ERROR_CODES,
    Chunk,
    Hit,
    Index,
    ReindexableIndex,
    RetrievalError,
    VectorError,
    fold,
    fuse_rrf,
    query_words,
    tokenize,
)

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = [
    "",
    "Refund policy, 30 days!",
    "alpha_beta gamma__delta _x_",
    "Ünïcödé Straße İstanbul ΣΊΣΥΦΟΣ",
    "١٢٣ ٤٥ digits ⅷ ² ½",
    'NEAR(a b) text: x " * ^ -y (z) {c d} AND OR NOT',
    "日本語のテキスト 中文 한국어",
    "tab\tnew\nline\x00nul",
    "café cafe CAFÉ Café",
    "emoji 🙂 and ZWJ‍joined",
]


# MUP-ONA T-CONF-1
@pytest.mark.parametrize("text", SAMPLES)
def test_tokenize_and_fold_reproduce_the_earlier_query_words(text: str) -> None:
    earlier = list(dict.fromkeys(word.lower() for word in re.findall(r"\w+", text, re.UNICODE)))
    assert query_words(text) == earlier
    assert tokenize(text) == re.findall(r"\w+", text)
    assert [fold(word) for word in tokenize(text)] == [word.lower() for word in tokenize(text)]


# MUP-ONA T-CONF-1
def test_tokenize_keeps_underscores_and_fold_keeps_diacritics() -> None:
    assert tokenize("alpha_beta, gamma") == ["alpha_beta", "gamma"]
    assert fold("CAFÉ") == "café"


def _earlier_rrf(rankings: list[list[int]]) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (60 + rank + 1)
    best = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    return [(chunk_id, scores[chunk_id]) for chunk_id in best]


# MUP-ONA T-CONF-1
@pytest.mark.parametrize(
    "rankings",
    [[], [[3, 1, 2]], [[1, 2, 3], [3, 2, 1]], [[5, 4], [4, 5], [9]], [[7, 8, 9, 10], [10, 11, 7]]],
)
def test_fuse_rrf_is_the_earlier_fusion_bit_for_bit(rankings: list[list[int]]) -> None:
    assert fuse_rrf(rankings) == _earlier_rrf(rankings)


# MUP-ONA T-CONF-1
def test_fuse_rrf_scores_and_ties() -> None:
    assert fuse_rrf([[4, 2]]) == [(4, 1 / 61), (2, 1 / 62)]
    assert fuse_rrf([[1, 2], [2, 1]]) == [(1, 1 / 61 + 1 / 62), (2, 1 / 62 + 1 / 61)]  # tie: lower id first
    assert fuse_rrf([[1]], k=0) == [(1, 1.0)]
    with pytest.raises(ValueError):
        fuse_rrf([[1]], k=-1)


IDENTITY = EmbeddingIdentity(provider="test", model="m", dimensions=3, normalized=False)


# MUP-ONA T-CONF-3
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_validate_vector_refuses_non_finite_values(bad: float) -> None:
    with pytest.raises(VectorError) as error:
        validate_vector([1.0, bad, 0.0], IDENTITY)
    assert error.value.code == "vector.invalid"
    assert isinstance(error.value, RetrievalError) and isinstance(error.value, ValueError)
    assert "finite" in str(error.value)


# MUP-ONA T-CONF-3
def test_validate_vector_refuses_values_that_are_no_numbers() -> None:
    with pytest.raises(VectorError) as error:
        validate_vector([1.0, None, 0.0], IDENTITY)  # type: ignore[list-item]
    assert error.value.code == "vector.invalid"


# MUP-ONA T-CONF-3
@pytest.mark.parametrize("vector", [[1.0, 0.0], [1.0, 0.0, 0.0, 0.0], []])
def test_validate_vector_checks_the_dimension(vector: list[float]) -> None:
    with pytest.raises(VectorError) as error:
        validate_vector(vector, IDENTITY)
    assert error.value.code == "vector.dimension"


# MUP-ONA T-CONF-3
def test_validate_vector_discards_a_zero_vector_and_normalizes() -> None:
    assert validate_vector([0.0, 0.0, -0.0], IDENTITY) is None
    assert validate_vector([3.0, 0.0, 4.0], IDENTITY) == [0.6, 0.0, 0.8]
    normalized = EmbeddingIdentity(provider="test", model="m", dimensions=3, normalized=True)
    assert validate_vector([3.0, 0.0, 4.0], normalized) == [3.0, 0.0, 4.0]  # the model's promise is kept
    assert validate_vector([2, 0]) == [1.0, 0.0]  # without identity: no dimension check, always scaled
    unit = validate_vector([1e-200, 1e-200, 0.0], IDENTITY)
    assert unit is not None and math.isclose(math.hypot(*unit), 1.0)


class _ListIndex(Index):
    """The smallest Index: a list, full text by substring; used to check what the base class adds."""

    def __init__(self, **kwargs: int) -> None:
        super().__init__(**kwargs)
        self.chunks: list[Chunk] = []
        self.searched: list[str] = []

    def add_chunks(self, chunks: Iterable[Chunk]) -> list[int]:
        start = len(self.chunks)
        self.chunks += list(chunks)
        return list(range(start, len(self.chunks)))

    def search(self, query: str, k: int = 5) -> list[Hit]:
        self.searched.append(query)
        found = [(i, c) for i, c in enumerate(self.chunks) if query and query in c.text]
        return [Hit(id=i, score=1.0, **c.model_dump()) for i, c in found[:k]]

    def delete(self, source: str) -> int:
        before = len(self.chunks)
        self.chunks = [chunk for chunk in self.chunks if chunk.source != source]
        return before - len(self.chunks)


class _NoInitIndex(_ListIndex):
    def __init__(self) -> None:  # does not call Index.__init__
        self.chunks = []
        self.searched = []


# MUP-ONA T-CONF-2
def test_every_index_subclass_checks_the_query_limits_first() -> None:
    index = _ListIndex()
    index.add("Refunds are possible within 30 days.", source="a")
    assert index.search("Refunds")
    for query in ("x" * (MAX_QUERY_CHARS + 1), " ".join(f"w{n}" for n in range(MAX_QUERY_WORDS + 1))):
        for call in (lambda q=query: index.search(q), lambda q=query: index.search(query=q, k=1)):
            with pytest.raises(RetrievalError) as error:
                call()
            assert error.value.code == "query.too_long"
        with pytest.raises(RetrievalError):
            index.retrieve(query, 3, sources=["a"])  # the base retrieve, through the wrapped search
    assert index.searched == ["Refunds"]  # the subclass's search never saw an overlong query
    assert index.search("x" * MAX_QUERY_CHARS) == []  # at the limit: accepted
    assert index.search(" ".join(f"w{n}" for n in range(MAX_QUERY_WORDS))) == []
    assert index.search("_".join(f"w{n}" for n in range(200))) == []  # one word for tokenize


# MUP-ONA T-CONF-2
def test_the_query_limits_are_constructor_options_and_survive_a_missing_super_init() -> None:
    small = _ListIndex(max_query_chars=5, max_query_words=2)
    with pytest.raises(RetrievalError):
        small.search("abcdef")
    with pytest.raises(RetrievalError):
        small.search("a b c")
    assert small.search("a b") == []
    with pytest.raises(ValueError, match="positive"):
        _ListIndex(max_query_chars=0)
    with pytest.raises(RetrievalError):
        _NoInitIndex().search("x" * (MAX_QUERY_CHARS + 1))


# MUP-ONA T-CONF-2
def test_an_overlong_query_is_refused_quickly() -> None:
    index = _ListIndex()
    started = time.monotonic()
    with pytest.raises(RetrievalError):
        index.search("word " * 20_000)
    assert time.monotonic() - started < 1.0


# MUP-ONA T-CONF-2
def test_sqlite_index_keeps_its_own_checks_and_is_not_wrapped_twice() -> None:
    for name in ("search", "retrieve"):
        method = inspect.getattr_static(SQLiteIndex, name)
        assert not hasattr(method, "__wrapped__"), name
        assert getattr(method, "__orkexa_checks_query__", False)
    assert hasattr(inspect.getattr_static(_ListIndex, "search"), "__wrapped__")


# MUP-ONA T-CONF-7
def test_capabilities_of_the_orkexa_implementations(tmp_path: Path) -> None:
    assert _ListIndex().capabilities == frozenset()
    with SQLiteIndex(tmp_path / "i.db") as index:
        assert index.capabilities == {"partitioned", "concurrent_writers"}
        assert isinstance(index, ReindexableIndex)
    assert not isinstance(_ListIndex(), ReindexableIndex)
    with SQLiteFileStore(tmp_path / "f.db") as store:
        assert store.capabilities == {"partitioned", "concurrent_writers"}
    with SQLiteMemory(tmp_path / "m.db") as memory:
        assert memory.capabilities == {"partitioned", "concurrent_writers"}
    assert LocalFileStore(tmp_path / "files").capabilities == frozenset()
    assert EphemeralMemory().capabilities == frozenset()
    assert FileMemory(tmp_path / "memory").capabilities == frozenset()


# MUP-ONA T-CONF-7
def test_sqlite_index_capabilities_follow_its_vector_configuration(tmp_path: Path) -> None:
    pytest.importorskip("sqlite_vec")
    from orkexa.embeddings import CallableEmbeddings

    try:
        index = SQLiteIndex(
            tmp_path / "v.db",
            embeddings=CallableEmbeddings(lambda texts: [[1.0, 0.0] for _ in texts], name="m", dimensions=2),
        )
    except RuntimeError as exc:  # this Python's sqlite3 cannot load extensions
        pytest.skip(str(exc))
    with index:
        assert index.capabilities == {"partitioned", "concurrent_writers", "vectors", "reindex"}
    with SQLiteIndex(tmp_path / "l.db", embedder=lambda texts: [[1.0, 0.0] for _ in texts]) as legacy:
        assert legacy.capabilities == {"partitioned", "concurrent_writers", "vectors"}


def _documented_codes() -> set[str]:
    doc = inspect.getdoc(RetrievalError) or ""
    return set(re.findall(r"^\* ``([a-z_.]+)``:", doc, re.MULTILINE))


def _raised_codes() -> set[str]:
    """Every literal code passed to RetrievalError/VectorError (and the PDF helper) in the package source."""
    found: set[str] = set()
    for path in (ROOT / "src" / "orkexa").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            name = getattr(node.func, "id", getattr(node.func, "attr", ""))
            first = node.args[0]
            if name in ("RetrievalError", "VectorError") and isinstance(first, ast.Constant):
                found.add(str(first.value))
            if name == "_too_complex":
                found.add("document.too_complex")
    return found


# MUP-ONA T-CONF-1
def test_the_error_codes_live_in_one_place() -> None:
    assert _documented_codes() == set(RETRIEVAL_ERROR_CODES)
    assert _raised_codes() <= set(RETRIEVAL_ERROR_CODES)
    assert {
        "partition.missing",
        "query.too_long",
        "vector.invalid",
        "vector.dimension",
        "schema.mismatch",
    } <= set(RETRIEVAL_ERROR_CODES)
    assert "document.too_complex" in RETRIEVAL_ERROR_CODES
