"""Partition (tenant) isolation cases, MUP-ONA plan chapter 10.4 (T-PART), for partitioned backends.

Three partitions A, B and C hold intentionally identical documents, sources, file contents (so identical file
ids), memory paths, namespaces and embedding identities; B and C additionally hold a marker (``zebramarker``).
Every case needs ``partitioned`` (the vector and reindex variants also ``vectors`` or ``reindex``) and is a
security case. A partition value the backend cannot use must be rejected (``ValueError``, ``TypeError`` or
``RetrievalError`` ``partition.*``) or see exactly nothing.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
import threading
from typing import Any

from ...errors import MemoryToolError
from ...retrieval import Chunk, RetrievalError
from ._core import Case, CaseFailure, Context, expect, expect_equal, expect_error
from ._models import KeywordEmbeddings, ModelFailure
from .index import DOCS

__all__ = ["PARTITION_FILE_STORE_CASES", "PARTITION_INDEX_CASES", "PARTITION_MEMORY_CASES"]

A, B, C = "conformance-a", "conformance-b", "conformance-c"
MARKER = "zebramarker"
MARKER_DOC = ("marker.md", f"Confidential {MARKER} notes of this partition about delta.")
MARKER_BYTES = f"{MARKER} secret file".encode()
SHARED_BYTES = b"Shared handbook text"
MARKER_PATH = f"/memories/secret/{MARKER}.md"
#: Partition values that must be rejected or see nothing (T-PART-V).
HOSTILE_PARTITIONS: tuple[Any, ...] = (
    None,
    "",
    " ",
    "\t",
    "x" * 10_000,
    "%",
    "_",
    "'",
    "conformance-%",
    "conformance-_",
    "conformance-b'--",
    "conformance-b ",
    " conformance-b",
    "CONFORMANCE-B",
    "conformance-b\x00",
    "*",
)


def _rejected(error: BaseException) -> bool:
    if isinstance(error, RetrievalError):
        return error.code.startswith("partition.")
    return isinstance(error, ValueError | TypeError)


def _open_hostile(context: Context, value: Any, **kwargs: Any) -> Any | None:
    """An instance for a hostile partition value, or ``None`` when the backend rejects the value."""
    try:
        return context.make(value, **kwargs)
    except Exception as exc:
        if _rejected(exc):
            return None
        raise CaseFailure(f"partition {value!r:.40}: {type(exc).__name__}: {exc}") from exc


def _count(index: Any) -> int | None:
    return len(index) if hasattr(index, "__len__") else None


def _sources(index: Any) -> list[str] | None:
    sources = getattr(index, "sources", None)
    return list(sources()) if callable(sources) else None


# ---- indexes ----------------------------------------------------------------------------------------------


def _indexes(context: Context, model: KeywordEmbeddings | None = None) -> dict[str, Any]:
    indexes = {name: context.make(name, embeddings=model) for name in (A, B, C)}
    for name, index in indexes.items():
        for source, text in DOCS:
            index.add(text, source=source)
        if name != A:
            index.add(MARKER_DOC[1], source=MARKER_DOC[0])
    return indexes


def _modes(context: Context) -> tuple[str, ...]:
    return ("fts", "vector", "hybrid") if "vectors" in context.capabilities else ("fts",)


def _fingerprint(index: Any, context: Context) -> tuple[Any, ...]:
    hits = index.retrieve(f"policy {MARKER} alpha beta gamma delta", 50, mode="fts")
    status = None
    if "vectors" in context.capabilities and hasattr(index, "vector_status"):
        current = index.vector_status()
        status = (current.chunks, current.embedded, current.active, current.building)
    return (_count(index), _sources(index), sorted((hit.id, hit.source, hit.text) for hit in hits), status)


def _no_marker(hits: Any, what: str) -> None:
    for hit in hits:
        expect(
            MARKER not in hit.text and hit.source != MARKER_DOC[0],
            f"{what}: a hit of another partition: {hit}",
        )


def index_reads(context: Context) -> None:
    """T-PART-R: searches in every mode, ``len``, ``sources()`` and ``vector_status`` see only their
    partition."""
    model = KeywordEmbeddings() if "vectors" in context.capabilities else None
    indexes = _indexes(context, model)
    a = indexes[A]
    own = {source for source, _ in DOCS}
    for mode in _modes(context):
        for query in (MARKER, f"{MARKER} delta fourth", "policy confidential notes"):
            hits = a.retrieve(query, 20, mode=mode)
            _no_marker(hits, f"{mode} {query!r}")
            expect(
                {hit.source for hit in hits} <= own, f"{mode} {query!r}: sources {[h.source for h in hits]}"
            )
    _no_marker(a.search(MARKER), "search() for the marker in A")
    expect_equal(a.retrieve(MARKER, 10, mode="fts"), [], "full-text search for the marker in A")
    expect(_count(a) in (None, 3), f"len() of A: {_count(a)}")
    expect(_sources(a) in (None, sorted(own)), f"sources() of A: {_sources(a)}")
    if model is not None:
        status = a.vector_status()
        expect_equal((status.chunks, status.embedded), (3, 3), "vector_status() of A")
    expect(indexes[B].search(MARKER), "B finds its own marker")


def index_writes(context: Context) -> None:
    """T-PART-W: adding the same source, the same texts and the same file in A leaves B and C unchanged."""
    indexes = _indexes(context)
    before = {name: _fingerprint(indexes[name], context) for name in (B, C)}
    a = indexes[A]
    a.add("Warranty policy, second edition: refunds within 60 days. alpha", source="warranty.md")
    a.add_chunks([Chunk(text=MARKER_DOC[1].replace(MARKER, "plain"), source=MARKER_DOC[0])])
    a.add_file(context.file("warranty.md", b"Warranty policy file version."), source="warranty.md")
    for name in (B, C):
        expect_equal(
            _fingerprint(indexes[name], context), before[name], f"partition {name} after writes in A"
        )


def index_deletes(context: Context) -> None:
    """T-PART-D: ``delete(source)`` in A removes only A's chunks; a source only B and C have gives 0 in A."""
    indexes = _indexes(context)
    before = {name: _fingerprint(indexes[name], context) for name in (B, C)}
    expect_equal(indexes[A].delete("warranty.md"), 1, "delete in A")
    expect_equal(indexes[A].delete(MARKER_DOC[0]), 0, "delete of a source A does not have")
    for name in (B, C):
        expect_equal(
            _fingerprint(indexes[name], context), before[name], f"partition {name} after deletes in A"
        )


def index_reindex(context: Context) -> None:
    """T-PART-RX: a reindex in A, finished or aborted, leaves B and C ``active`` with their vectors."""
    model = KeywordEmbeddings("conformance-old")
    indexes = _indexes(context, model)
    before = {name: _fingerprint(indexes[name], context) for name in (B, C)}
    ranking = {name: [h.id for h in indexes[name].retrieve("delta", 10, mode="vector")] for name in (B, C)}
    indexes[A].reindex(KeywordEmbeddings("conformance-new"))
    failing = KeywordEmbeddings("conformance-newer", fail_after=1)
    expect_error(lambda: indexes[A].reindex(failing, batch_size=1), ModelFailure, "the aborted reindex in A")
    for name in (B, C):
        expect_equal(
            _fingerprint(indexes[name], context), before[name], f"partition {name} after reindexes in A"
        )
        current = [h.id for h in indexes[name].retrieve("delta", 10, mode="vector")]
        expect_equal(current, ranking[name], f"vector ranking of {name}")
        expect_equal(
            indexes[name].vector_status().active, model.identity().to_dict(), f"active space of {name}"
        )


def index_rollback(context: Context) -> None:
    """T-PART-TX: a write in A that fails after its first statement rolls back; B is unchanged; A works on."""
    indexes = _indexes(context, KeywordEmbeddings())
    a = indexes[A]
    before_a, before_b = _fingerprint(a, context), _fingerprint(indexes[B], context)
    chunks = [
        Chunk(text="A good chunk about eta.", source="tx.md"),
        Chunk(text="A shortvector chunk.", source="tx.md"),
    ]
    expect_error(lambda: a.add_chunks(chunks), Exception, "a write with an invalid second vector")
    expect_equal(_fingerprint(a, context), before_a, "A after the failed write")
    expect_equal(_fingerprint(indexes[B], context), before_b, "B after the failed write in A")
    a.add("A good chunk about eta.", source="tx.md")
    expect(
        [hit.source for hit in a.retrieve("eta", 1, mode="vector")] == ["tx.md"], "A works after the rollback"
    )


def index_concurrency(context: Context) -> None:
    """T-PART-CC: three partitions written, searched and deleted from several threads at once: exact counts,
    no marker in A."""
    _indexes(context)
    errors: list[BaseException] = []

    def work(partition: str, worker: int) -> None:
        try:
            index = context.make(partition)
            for number in range(8):
                index.add(f"Worker {worker} note {number}: policy text.", source=f"w{worker}-{number}.md")
                hits = index.retrieve(f"policy {MARKER}", 30, mode="fts")
                if partition == A:
                    _no_marker(hits, f"thread {worker}")
                if number % 2 == 0:
                    expect_equal(index.delete(f"w{worker}-{number}.md"), 1, f"delete by thread {worker}")
        except BaseException as exc:  # reported below
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(p, n)) for n, p in enumerate([A, B, C] * 3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(300)
    expect_equal([f"{type(e).__name__}: {e}" for e in errors], [], "errors of the threads")
    for partition, base in ((A, 3), (B, 4), (C, 4)):
        index = context.make(partition)
        expect(_count(index) in (None, base + 12), f"len() of {partition}: {_count(index)}")
        found = {hit.source for hit in index.retrieve("worker", 50, mode="fts")}
        expect_equal(len(found), 12, f"worker sources of {partition}")


def index_hostile_partitions(context: Context) -> None:
    """T-PART-V: empty, ``None``, blank, 10,000-character, wildcard, quoted or foreign-looking partition
    values are rejected or see nothing."""
    _indexes(context)
    for value in HOSTILE_PARTITIONS:
        index = _open_hostile(context, value)
        if index is None:
            continue
        for query in (MARKER, "policy refunds"):
            expect_equal(index.retrieve(query, 10, mode="fts"), [], f"hits in partition {value!r:.40}")
        expect(_count(index) in (None, 0), f"len() in partition {value!r:.40}")
        expect(_sources(index) in (None, []), f"sources() in partition {value!r:.40}")


def index_vector_joins(context: Context) -> None:
    """T-PART-J: with the same embedding identity (same space key) in A and B, vector and hybrid searches of A
    join only A's rows."""
    indexes = _indexes(context, KeywordEmbeddings())
    for mode in ("vector", "hybrid"):
        hits = indexes[A].retrieve("delta fourth", 10, mode=mode)
        _no_marker(hits, mode)
        expect({hit.source for hit in hits} <= {source for source, _ in DOCS}, f"{mode}: {hits}")
    best = indexes[B].retrieve("delta fourth", 1, mode="vector")
    expect(best and best[0].source == MARKER_DOC[0], f"B finds its marker by vector: {best}")


# ---- file stores ------------------------------------------------------------------------------------------


def _stores(context: Context) -> tuple[dict[str, Any], str, str]:
    stores = {name: context.make(name) for name in (A, B, C)}
    shared = marker = ""
    for name, store in stores.items():
        shared = store.add(SHARED_BYTES, name="handbook.txt", description="Handbook").id
        if name != A:
            marker = store.add(MARKER_BYTES, name=f"{MARKER}.txt", description=MARKER).id
    return stores, shared, marker


def _store_print(store: Any) -> list[tuple[str, str, str, bytes]]:
    return sorted((e.id, e.name, e.description, store.get_bytes(e.id)) for e in store.list())


def files_reads(context: Context) -> None:
    """T-PART-R: A never sees B's files, even with their id; identical bytes have the same id in each."""
    stores, shared, marker = _stores(context)
    a = stores[A]
    expect(marker not in a, "B's file id in A")
    expect_error(lambda: a.info(marker), KeyError, "info() of B's file in A")
    expect_error(lambda: a.get_bytes(marker), KeyError, "get_bytes() of B's file in A")
    expect_equal([e.id for e in a.list()], [shared], "list() of A")
    expect_equal(a.list(MARKER), [], "list() of A filtered on the marker")
    expect(MARKER not in a.catalog(), "the catalog of A")
    expect_equal(len(a), 1, "len() of A")
    expect_equal(stores[B].get_bytes(marker), MARKER_BYTES, "B reads its own file")


def files_writes(context: Context) -> None:
    """T-PART-W/D: renaming, adding and deleting in A (same ids) leaves B and C unchanged."""
    stores, shared, marker = _stores(context)
    before = {name: _store_print(stores[name]) for name in (B, C)}
    a = stores[A]
    a.add(SHARED_BYTES, name="renamed.txt", description="changed in A")
    a.add(b"Only in A", name="a.txt")
    a.delete(shared)
    expect_error(lambda: a.delete(marker), KeyError, "delete of B's file id in A")
    for name in (B, C):
        expect_equal(_store_print(stores[name]), before[name], f"partition {name} after writes in A")


def files_hostile_partitions(context: Context) -> None:
    """T-PART-V: hostile partition values are rejected or see no file."""
    _, shared, marker = _stores(context)
    for value in HOSTILE_PARTITIONS:
        store = _open_hostile(context, value)
        if store is None:
            continue
        expect_equal(store.list(), [], f"files in partition {value!r:.40}")
        expect(shared not in store and marker not in store, f"ids in partition {value!r:.40}")


# ---- memories ---------------------------------------------------------------------------------------------


def _memories(context: Context) -> dict[str, Any]:
    memories = {name: context.make(name) for name in (A, B, C)}
    for name, memory in memories.items():
        memory.create("/memories/shared.md", "shared line one\nshared line two\n")
        memory.create("/memories/dir/note.md", "a note")
        if name != A:
            memory.create(MARKER_PATH, f"{MARKER} content")
    return memories


def _memory_print(memory: Any) -> tuple[str, ...]:
    return (
        memory.view("/memories"),
        memory.view("/memories/shared.md"),
        memory.view("/memories/dir/note.md"),
        memory.view(MARKER_PATH),
    )


def memory_reads(context: Context) -> None:
    """T-PART-R: A's listing and reads never show B's files."""
    memories = _memories(context)
    a = memories[A]
    expect(MARKER not in a.view("/memories"), "the listing of A")
    expect_error(lambda: a.view(MARKER_PATH), MemoryToolError, "B's file in A")
    expect_error(lambda: a.view("/memories/secret"), MemoryToolError, "B's directory in A")
    expect(MARKER in memories[B].view(MARKER_PATH), "B reads its own file")


def memory_updates(context: Context) -> None:
    """T-PART-U/D: ``str_replace``, ``insert``, ``rename`` and ``delete`` on the same paths in A leave B and C
    unchanged."""
    memories = _memories(context)
    before = {name: _memory_print(memories[name]) for name in (B, C)}
    a = memories[A]
    a.str_replace("/memories/shared.md", "line one", "line ONE")
    a.insert("/memories/shared.md", 0, "inserted in A")
    a.rename("/memories/dir", "/memories/moved")
    a.rename("/memories/shared.md", "/memories/renamed.md")
    a.delete("/memories/moved")
    expect_error(lambda: a.delete(MARKER_PATH), MemoryToolError, "delete of B's file in A")
    expect_error(
        lambda: a.rename(MARKER_PATH, "/memories/stolen.md"), MemoryToolError, "rename of B's file in A"
    )
    a.create(MARKER_PATH, "A's own file at B's path")
    for name in (B, C):
        expect_equal(_memory_print(memories[name]), before[name], f"partition {name} after commands in A")


def memory_hostile_partitions(context: Context) -> None:
    """T-PART-V: hostile partition values are rejected or see no memory file."""
    _memories(context)
    for value in HOSTILE_PARTITIONS:
        memory = _open_hostile(context, value)
        if memory is None:
            continue
        listing = memory.view("/memories")
        expect("shared.md" not in listing and MARKER not in listing, f"listing in partition {value!r:.40}")
        expect_error(partial(memory.view, "/memories/shared.md"), MemoryToolError, f"read in {value!r:.40}")


def _case(name: str, contract: Any, run: Callable[[Context], None], *needs: str) -> Case:
    return Case(name, frozenset({"partitioned", *needs}), run, contract, security=True)


PARTITION_INDEX_CASES: tuple[Case, ...] = (
    _case("partition.index_reads", "index", index_reads),
    _case("partition.index_writes", "index", index_writes),
    _case("partition.index_deletes", "index", index_deletes),
    _case("partition.index_reindex", "index", index_reindex, "vectors", "reindex"),
    _case("partition.index_failed_write_rolls_back", "index", index_rollback, "vectors"),
    _case("partition.index_concurrent_use", "index", index_concurrency),
    _case("partition.index_hostile_values", "index", index_hostile_partitions),
    _case("partition.index_vector_joins", "index", index_vector_joins, "vectors"),
)
"""T-PART-R, -W, -D, -RX, -TX, -CC, -V and -J for indexes."""

PARTITION_FILE_STORE_CASES: tuple[Case, ...] = (
    _case("partition.files_reads", "files", files_reads),
    _case("partition.files_writes_and_deletes", "files", files_writes),
    _case("partition.files_hostile_values", "files", files_hostile_partitions),
)
"""T-PART-R, -W, -D and -V for file stores."""

PARTITION_MEMORY_CASES: tuple[Case, ...] = (
    _case("partition.memory_reads", "memory", memory_reads),
    _case("partition.memory_updates_and_deletes", "memory", memory_updates),
    _case("partition.memory_hostile_values", "memory", memory_hostile_partitions),
)
"""T-PART-R, -U, -D and -V for memory backends."""
