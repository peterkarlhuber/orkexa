"""Conformance cases of the ``FileStore`` contract (T-CONF-5): content-derived ids, media types and limits,
catalog, delete, id rules, rollback of a refused add, concurrent writers."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
import hashlib
import threading
import time
from typing import Any

from ...errors import ToolResultError
from ._core import Case, Context, expect, expect_equal, expect_error

__all__ = ["FILE_STORE_CASES"]

PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\rIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)
PDF = b"%PDF-1.4\n% \x00\xff binary \x00 bytes\n%%EOF\n"
BAD_IDS = ("", "../../etc/passwd", "0123456789abcdeg", "0123456789ABCDEF", "0123456789abcdef0", "/tmp/x", "*")


def add_and_read(context: Context) -> None:
    """``add`` returns a catalog entry whose id is derived from the bytes; ``info``/``get_bytes`` read it."""
    store = context.make()
    entry = store.add(b"Plain text note", name="note.txt", description="A note", metadata={"k": "v"})
    digest = hashlib.sha256(b"Plain text note").hexdigest()
    expect_equal((entry.id, entry.sha256, entry.size), (digest[:16], digest, 15), "id, sha256, size")
    expect_equal(
        (entry.name, entry.media_type, entry.description), ("note.txt", "text/plain", "A note"), "entry"
    )
    expect_equal(entry.metadata, {"k": "v"}, "metadata")
    expect_equal(store.info(entry.id), entry, "info()")
    expect_equal(store.get_bytes(entry.id), b"Plain text note", "get_bytes()")
    expect(entry.id in store, "the id is in the store")
    expect_equal(len(store), 1, "len()")


def same_bytes(context: Context) -> None:
    """Adding the same bytes again returns the same entry with the new name and description."""
    store = context.make()
    first = store.add(b"same bytes", name="a.txt", description="first", metadata={"a": "1"})
    second = store.add(b"same bytes", name="b.txt", metadata={"b": "2"})
    expect_equal(second.id, first.id, "id of the same bytes")
    expect_equal((second.name, second.description), ("b.txt", "first"), "name updated, description kept")
    expect_equal(second.metadata, {"a": "1", "b": "2"}, "metadata merged")
    expect_equal(len(store), 1, "len()")
    expect_equal(store.info(first.id).name, "b.txt", "the stored entry")


def add_path_and_binary(context: Context) -> None:
    """Files are added from a path or bytes; images and PDFs with any bytes round-trip exactly."""
    store = context.make()
    path = context.file("diagram.png", PNG)
    image = store.add(path, description="A pixel")
    expect_equal((image.name, image.media_type), ("diagram.png", "image/png"), "image entry")
    pdf = store.add(PDF, name="scan.pdf")
    expect_equal(pdf.media_type, "application/pdf", "pdf media type")
    expect_equal((store.get_bytes(image.id), store.get_bytes(pdf.id)), (PNG, PDF), "bytes")
    markdown = store.add(b"# Heading\n", name="notes.md")
    expect_equal(markdown.media_type, "text/markdown", "markdown media type")


def refusals(context: Context) -> None:
    """Unsupported types, text that is not UTF-8, oversized files and bytes without a name are refused."""
    store = context.make()
    expect_error(lambda: store.add(b"x", name="tool.exe"), ValueError, "an executable")
    expect_error(lambda: store.add(b"x", name="doc.docx"), ValueError, "a Word file")
    expect_error(lambda: store.add(b"\xff\xfe\xfa", name="bad.txt"), ValueError, "text that is not UTF-8")
    expect_error(
        lambda: store.add(b"x" * 20_000_001, name="big.txt"), ValueError, "a document over the limit"
    )
    expect_error(
        lambda: store.add(PNG + b"\x00" * 5_000_000, name="big.png"), ValueError, "an image over the limit"
    )
    expect_error(lambda: store.add(b"x"), ValueError, "bytes without a name")
    expect_error(lambda: store.add(b"x", name="blob"), ValueError, "an unknown media type without media_type")
    expect_equal(len(store), 0, "nothing is stored after refusals (rollback)")
    expect_equal(store.list(), [], "list() after refusals")


def catalog(context: Context) -> None:
    """``list`` returns the newest entry first and filters on name and description, case-insensitively."""
    store = context.make()
    old = store.add(b"old", name="old-report.txt", description="Quarterly REPORT")
    time.sleep(0.01)  # created_at must differ
    new = store.add(b"new", name="new.txt", description="Minutes")
    expect_equal([e.id for e in store.list()], [new.id, old.id], "newest first")
    expect_equal([e.id for e in store.list("report")], [old.id], "filter on name and description")
    expect_equal([e.id for e in store.list("MINUTES")], [new.id], "case-insensitive")
    expect_equal(store.list("nothing"), [], "no match")
    text = store.catalog()
    expect(old.id in text and new.id in text, "the catalog names both files")


def delete(context: Context) -> None:
    """``delete`` removes a file and its bytes; unknown ids raise ``KeyError``."""
    store = context.make()
    keep = store.add(b"keep", name="keep.txt")
    gone = store.add(b"gone", name="gone.txt")
    store.delete(gone.id)
    expect(gone.id not in store, "the deleted id is gone")
    expect_error(lambda: store.info(gone.id), KeyError, "info() of a deleted file")
    expect_error(lambda: store.get_bytes(gone.id), KeyError, "get_bytes() of a deleted file")
    expect_error(lambda: store.delete(gone.id), KeyError, "delete() again")
    expect_equal([e.id for e in store.list()], [keep.id], "the other file stays")
    again = store.add(b"gone", name="gone.txt")
    expect_equal(store.get_bytes(again.id), b"gone", "a deleted file can be added again")


def ids_are_not_paths(context: Context) -> None:
    """Only 16 lowercase hex digits are ids; anything else (paths, wildcards) is unknown, also to the tool."""
    store = context.make()
    store.add(b"secret", name="secret.txt")
    for bad in BAD_IDS:
        expect(bad not in store, f"{bad!r} must not be in the store")
        expect_error(partial(store.info, bad), KeyError, f"info({bad!r})")
        expect_error(partial(store.get_bytes, bad), KeyError, f"get_bytes({bad!r})")
        expect_error(partial(store.delete, bad), KeyError, f"delete({bad!r})")
    tool = store.tool()
    for bad in BAD_IDS:
        expect_error(
            partial(tool.execute, {"command": "load", "file_id": bad}), ToolResultError, f"load {bad!r}"
        )
    expect_equal(len(store), 1, "the stored file is untouched")


def tool(context: Context) -> None:
    """The ``files`` tool lists the catalog and loads a file by id."""
    store = context.make()
    entry = store.add(b"Tool text", name="tool.txt", description="for the tool")
    files = store.tool()
    listing = files.execute({"command": "list"})
    expect(entry.id in str(listing), f"list names the file: {listing}")
    loaded = files.execute({"command": "load", "file_id": entry.id})
    expect(entry.id in str(loaded[0]), f"load returns the header first: {loaded}")


def second_instance(context: Context) -> None:
    """A second instance on the same storage and partition sees the files of the first."""
    first = context.make()
    entry = first.add(b"shared", name="shared.txt")
    second = context.make()
    expect_equal(second.get_bytes(entry.id), b"shared", "bytes seen by a second instance")
    second.delete(entry.id)
    expect(entry.id not in first, "a delete by the second instance is seen by the first")


def concurrent_adds(context: Context) -> None:
    """Two instances adding files from several threads at once lose no entry (``concurrent_writers``)."""
    stores = [context.make(), context.make()]
    errors: list[BaseException] = []

    def work(store: Any, worker: int) -> None:
        try:
            for number in range(10):
                store.add(f"file {worker}-{number}".encode(), name=f"f-{worker}-{number}.txt")
        except BaseException as exc:  # reported below
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(stores[w % 2], w)) for w in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(120)
    expect_equal(errors, [], "errors of the writers")
    expect_equal(len(stores[0]), 40, "entries seen by the first instance")
    expect_equal(len(stores[1]), 40, "entries seen by the second instance")


def _case(name: str, run: Callable[[Context], None], *needs: str, security: bool = False) -> Case:
    return Case(name, frozenset(needs), run, "files", security)


FILE_STORE_CASES: tuple[Case, ...] = (
    _case("files.add_and_read", add_and_read),
    _case("files.same_bytes_share_one_entry", same_bytes),
    _case("files.paths_and_binary_round_trip", add_path_and_binary),
    _case("files.refusals_store_nothing", refusals),
    _case("files.catalog_order_and_filter", catalog),
    _case("files.delete", delete),
    _case("files.ids_are_not_paths", ids_are_not_paths, security=True),
    _case("files.tool_lists_and_loads", tool),
    _case("files.second_instance_sees_the_files", second_instance),
    _case("files.concurrent_adds_lose_nothing", concurrent_adds, "concurrent_writers"),
)
"""T-CONF-5: the ``FileStore`` contract."""
