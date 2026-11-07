"""Conformance cases of ``ReindexableIndex`` (T-CONF-4): space states, identity change, abort, races.

Every case needs the capabilities ``vectors`` and ``reindex``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import threading
from typing import Any

from ...retrieval import RetrievalError
from ._core import Case, Context, expect, expect_code, expect_equal, expect_error
from ._models import KeywordEmbeddings, ModelFailure
from .index import DOCS

__all__ = ["REINDEX_CASES"]


def _fill(index: Any) -> None:
    for source, text in DOCS:
        index.add(text, source=source)


def first_vectors(context: Context) -> None:
    """``reindex`` gives a full-text index its first vectors; then ``active`` is the model and complete."""
    index = context.make()
    _fill(index)
    model = KeywordEmbeddings()
    status = index.reindex(model, batch_size=2)
    expect_equal(status.active, model.identity().to_dict(), "active identity")
    expect_equal(
        (status.chunks, status.embedded, status.building), (3, 3, None), "chunks, embedded, building"
    )
    expect(status.complete, "the status is complete")
    expect_equal(index.vector_status(), status, "vector_status() after reindex")
    hits = index.retrieve("third", 1, mode="vector")
    expect(hits and hits[0].source == "privacy.md", f"vector search after reindex: {hits}")


def identity_change(context: Context) -> None:
    """A new model needs ``reindex``; until then writes and searches with it raise ``reindex_required``."""
    old, new = KeywordEmbeddings("conformance-old"), KeywordEmbeddings("conformance-new")
    index = context.make(embeddings=old)
    _fill(index)
    expect_equal(index.vector_status().active, old.identity().to_dict(), "active before")
    other = context.make(embeddings=new)
    expect_code(
        lambda: other.add("Another chunk about beta.", source="x.md"),
        "a write with another model",
        "reindex_required",
    )
    expect_code(
        lambda: other.retrieve("beta", 3, mode="vector"), "a search with another model", "reindex_required"
    )
    expect_code(
        lambda: other.retrieve("beta", 3, mode="hybrid"), "hybrid with another model", "reindex_required"
    )
    status = index.reindex(new)
    expect_equal(status.active, new.identity().to_dict(), "active after")
    expect_equal((status.embedded, status.building), (3, None), "embedded, building after")
    expect(other.retrieve("beta", 1, mode="vector")[0].source == "shipping.md", "the new model searches")
    expect_code(
        lambda: context.make(embeddings=old).retrieve("beta", 3, mode="vector"),
        "the old model",
        "reindex_required",
    )


def interrupted(context: Context) -> None:
    """An interrupted reindex keeps the old space ``active`` and searchable; calling it again completes it."""
    old = KeywordEmbeddings("conformance-old")
    index = context.make(embeddings=old)
    _fill(index)
    new = KeywordEmbeddings("conformance-new", fail_after=1)
    expect_error(lambda: index.reindex(new, batch_size=1), ModelFailure, "the failing model")
    status = index.vector_status()
    expect_equal(status.active, old.identity().to_dict(), "active after the abort")
    expect_equal(status.embedded, 3, "the old vectors stay")
    expect(status.building == new.identity().to_dict(), f"the new space is building: {status.building}")
    expect_equal(status.building_embedded, 1, "vectors of the finished batch")
    hits = index.retrieve("gamma", 1, mode="vector")
    expect(hits and hits[0].source == "privacy.md", f"the old space still searches: {hits}")
    new.fail_after = None
    status = index.reindex(new, batch_size=1)
    expect_equal(
        (status.active, status.embedded, status.building), (new.identity().to_dict(), 3, None), "resumed"
    )


def cancel(context: Context) -> None:
    """A cancelled reindex raises ``cancelled``, keeps ``active``; ``discard_reindex`` drops ``building``."""
    old = KeywordEmbeddings("conformance-old")
    index = context.make(embeddings=old)
    _fill(index)
    stop = threading.Event()
    stop.set()
    expect_code(
        lambda: index.reindex(KeywordEmbeddings("conformance-new"), cancel=stop), "cancel", "cancelled"
    )
    expect_equal(index.vector_status().active, old.identity().to_dict(), "active after cancel")
    index.discard_reindex()
    status = index.vector_status()
    expect_equal(
        (status.active, status.building, status.embedded), (old.identity().to_dict(), None, 3), "discarded"
    )


def identity_refresh(context: Context) -> None:
    """Writes and reindex re-read the model's identity first: an updated model is never filed under its old
    identity (drift 1)."""
    model = KeywordEmbeddings(digest="v1")
    model.identity()  # cached as v1 ...
    model.served_digest = "v2"  # ... while the server already has v2
    index = context.make(embeddings=model)
    index.add("A first chunk about alpha.", source="a.md")
    expect_equal((index.vector_status().active or {}).get("digest"), "v2", "the identity of the first write")
    model.served_digest = "v3"  # updated on the server; the cached identity is still v2
    expect_code(
        lambda: index.add("A second chunk about beta.", source="b.md"),
        "a write after an update",
        "reindex_required",
    )
    expect_equal(index.vector_status().chunks, 1, "the refused write stored nothing")
    index.reindex(model)
    expect_equal((index.vector_status().active or {}).get("digest"), "v3", "the identity after reindex")


def deleted_during_embedding(context: Context) -> None:
    """A chunk deleted while its batch is embedded gets no vector; the reindex completes (drift 9)."""
    index = context.make()
    _fill(index)
    deleted: list[int] = []

    def delete_once(texts: Sequence[str]) -> None:
        if not deleted:
            deleted.append(index.delete("shipping.md"))

    model = KeywordEmbeddings(on_embed=delete_once)
    status = index.reindex(model, batch_size=10)
    expect_equal(deleted, [1], "the chunk was deleted during the embedding")
    expect_equal(
        (status.chunks, status.embedded, status.building), (2, 2, None), "chunks, embedded, building"
    )
    expect(
        "shipping.md" not in {h.source for h in index.retrieve("beta", 5, mode="vector")}, "no deleted hit"
    )


def added_during_reindex(context: Context) -> None:
    """Chunks added during a reindex get vectors too: it completes with every chunk embedded, or raises
    ``vector_incomplete`` and completes when called again."""
    index = context.make()
    _fill(index)
    writer = context.make()
    added: list[Any] = []

    def add_once(texts: Sequence[str]) -> None:
        if not added:
            added.append(writer.add("A late chunk about delta.", source="late.md"))

    model = KeywordEmbeddings(on_embed=add_once)
    try:
        status = index.reindex(model, batch_size=10)
    except RetrievalError as exc:
        expect_equal(exc.code, "vector_incomplete", "the code of an incomplete reindex")
        status = index.reindex(model, batch_size=10)
    expect_equal(len(added), 1, "a chunk was added during the reindex")
    expect_equal((status.chunks, status.embedded, status.building), (4, 4, None), "after the reindex")
    hits = index.retrieve("fourth", 1, mode="vector")
    expect(hits and hits[0].source == "late.md", f"the late chunk has a vector: {hits}")


def _case(name: str, run: Callable[[Context], None]) -> Case:
    return Case(name, frozenset({"vectors", "reindex"}), run, "index")


REINDEX_CASES: tuple[Case, ...] = (
    _case("reindex.first_vectors_for_a_full_text_index", first_vectors),
    _case("reindex.identity_change_needs_a_reindex", identity_change),
    _case("reindex.interrupted_reindex_keeps_the_old_space", interrupted),
    _case("reindex.cancel_and_discard", cancel),
    _case("reindex.identity_is_refreshed_before_writing", identity_refresh),
    _case("reindex.chunk_deleted_during_embedding_gets_no_vector", deleted_during_embedding),
    _case("reindex.chunks_added_during_a_reindex_get_vectors", added_during_reindex),
)
"""T-CONF-4: the ``ReindexableIndex`` contract."""
