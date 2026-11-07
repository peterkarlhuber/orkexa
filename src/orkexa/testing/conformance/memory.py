"""Conformance cases of the ``MemoryBackend`` contract (T-CONF-6): the six commands, path rules, limits,
failed commands that change nothing, concurrent writers."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
import threading
from typing import Any

from ...errors import MemoryToolError
from ._core import Case, Context, expect, expect_equal, expect_error

__all__ = ["MEMORY_CASES"]

#: Paths that must never be accepted: they leave /memories, hide a traversal or hold control characters.
BAD_PATHS: tuple[str, ...] = (
    "/etc/passwd",
    "memories/x.md",
    "/memoriesx/x.md",
    "/memories/../x.md",
    "/memories/a/../../x.md",
    "/memories/%2e%2e/x.md",
    "/memories/%252e%252e/x.md",
    "/memories/a%2fb.md",
    "/memories/a\\b.md",
    "/memories/./x.md",
    "/memories/x\x00.md",
    "/memories/x\n.md",
    "/memories/x\x7f.md",
)


def _snapshot(memory: Any) -> str:
    return str(memory.view("/memories"))


def _read(memory: Any, path: str) -> str:
    """The content of a memory file without line numbers."""
    text = memory.view(path)
    return "\n".join(line.split("\t", 1)[1] if "\t" in line else "" for line in text.split("\n")[1:])


def create_and_view(context: Context) -> None:
    """``create`` writes a file; ``view`` shows it with line numbers, or a directory listing."""
    memory = context.make()
    expect_equal(
        memory.create("/memories/notes/todo.md", "one\ntwo\n"),
        "File created successfully at: /memories/notes/todo.md",
        "create",
    )
    view = memory.view("/memories/notes/todo.md")
    expect("     1\tone" in view and "     2\ttwo" in view, f"numbered lines: {view!r}")
    listing = memory.view("/memories")
    expect("/memories/notes/" in listing and "/memories/notes/todo.md" in listing, f"listing: {listing!r}")
    memory.create("/memories/notes/todo.md", "replaced")
    expect("replaced" in memory.view("/memories/notes/todo.md"), "create overwrites")
    expect_equal(_read(memory, "/memories/notes/todo.md"), "replaced", "content")
    ranged = memory.view("/memories/notes/todo.md", (1, 1))
    expect("replaced" in ranged, f"view_range: {ranged!r}")
    expect_error(
        lambda: memory.view("/memories/notes/todo.md", (2, 3)),
        MemoryToolError,
        "a view_range outside the file",
    )
    expect_error(lambda: memory.view("/memories/missing.md"), MemoryToolError, "view of a missing file")


def path_rules(context: Context) -> None:
    """Paths outside ``/memories``, traversals (also URL-encoded), backslashes and control characters are
    refused by every command; nothing is written."""
    memory = context.make()
    memory.create("/memories/keep.md", "keep")
    before = _snapshot(memory)
    for path in BAD_PATHS:
        calls: list[tuple[str, Callable[[], object]]] = [
            ("create", partial(memory.create, path, "x")),
            ("view", partial(memory.view, path)),
            ("str_replace", partial(memory.str_replace, path, "keep", "x")),
            ("insert", partial(memory.insert, path, 0, "x")),
            ("delete", partial(memory.delete, path)),
            ("rename from", partial(memory.rename, path, "/memories/y.md")),
            ("rename to", partial(memory.rename, "/memories/keep.md", path)),
        ]
        for command, call in calls:
            expect_error(call, MemoryToolError, f"{command} {path!r}")
    expect_error(lambda: memory.create("/memories", "x"), MemoryToolError, "create of the root")
    expect_error(lambda: memory.delete("/memories"), MemoryToolError, "delete of the root")
    expect_error(lambda: memory.rename("/memories", "/memories/x"), MemoryToolError, "rename of the root")
    expect_equal(_snapshot(memory), before, "the memory after refused paths")


def str_replace(context: Context) -> None:
    """``str_replace`` replaces exactly one occurrence; none or several are errors that change nothing."""
    memory = context.make()
    memory.create("/memories/a.md", "alpha beta\nbeta gamma\n")
    expect_error(
        lambda: memory.str_replace("/memories/a.md", "beta", "x"), MemoryToolError, "two occurrences"
    )
    expect_error(lambda: memory.str_replace("/memories/a.md", "delta", "x"), MemoryToolError, "no occurrence")
    expect_error(lambda: memory.str_replace("/memories/a.md", "", "x"), MemoryToolError, "an empty old_str")
    expect_error(
        lambda: memory.str_replace("/memories/missing.md", "a", "b"), MemoryToolError, "a missing file"
    )
    expect_equal(_read(memory, "/memories/a.md"), "alpha beta\nbeta gamma", "unchanged after errors")
    memory.str_replace("/memories/a.md", "alpha beta", "alpha BETA")
    expect_equal(_read(memory, "/memories/a.md"), "alpha BETA\nbeta gamma", "after the replacement")


def insert(context: Context) -> None:
    """``insert`` adds lines after ``insert_line`` (0: at the start); a line outside the file is an error."""
    memory = context.make()
    memory.create("/memories/a.md", "one\nthree\n")
    memory.insert("/memories/a.md", 1, "two")
    memory.insert("/memories/a.md", 0, "zero")
    expect_equal(_read(memory, "/memories/a.md"), "zero\none\ntwo\nthree", "after inserts")
    expect_error(lambda: memory.insert("/memories/a.md", 9, "x"), MemoryToolError, "a line outside the file")
    expect_error(lambda: memory.insert("/memories/missing.md", 0, "x"), MemoryToolError, "a missing file")
    expect_equal(_read(memory, "/memories/a.md"), "zero\none\ntwo\nthree", "unchanged after errors")


def delete(context: Context) -> None:
    """``delete`` removes a file or a directory with its contents; a missing path is an error."""
    memory = context.make()
    memory.create("/memories/dir/a.md", "a")
    memory.create("/memories/dir/sub/b.md", "b")
    memory.create("/memories/keep.md", "keep")
    memory.delete("/memories/dir/sub/b.md")
    expect_error(lambda: memory.view("/memories/dir/sub/b.md"), MemoryToolError, "the deleted file")
    memory.delete("/memories/dir")
    expect_error(
        lambda: memory.view("/memories/dir/a.md"), MemoryToolError, "a file of the deleted directory"
    )
    expect_error(lambda: memory.delete("/memories/dir"), MemoryToolError, "delete of a missing path")
    expect_equal(_read(memory, "/memories/keep.md"), "keep", "other files stay")


def rename(context: Context) -> None:
    """``rename`` moves files and directories; it never overwrites and never moves a directory into itself."""
    memory = context.make()
    memory.create("/memories/a.md", "a")
    memory.create("/memories/dir/b.md", "b")
    memory.create("/memories/taken.md", "taken")
    memory.rename("/memories/a.md", "/memories/new/a2.md")
    expect_equal(_read(memory, "/memories/new/a2.md"), "a", "the moved file")
    expect_error(lambda: memory.view("/memories/a.md"), MemoryToolError, "the old path")
    memory.rename("/memories/dir", "/memories/moved")
    expect_equal(_read(memory, "/memories/moved/b.md"), "b", "a file of the moved directory")
    expect_error(
        lambda: memory.rename("/memories/new/a2.md", "/memories/taken.md"), MemoryToolError, "overwrite"
    )
    expect_error(
        lambda: memory.rename("/memories/moved", "/memories/moved/inner"), MemoryToolError, "into itself"
    )
    expect_error(
        lambda: memory.rename("/memories/missing.md", "/memories/x.md"), MemoryToolError, "a missing source"
    )
    expect_equal(_read(memory, "/memories/taken.md"), "taken", "the destination is untouched")


def files_and_directories(context: Context) -> None:
    """A file is not a directory: no file below a file, no ``create`` over a directory."""
    memory = context.make()
    memory.create("/memories/a.md", "a")
    expect_error(lambda: memory.create("/memories/a.md/b.md", "b"), MemoryToolError, "a file below a file")
    memory.create("/memories/dir/b.md", "b")
    expect_error(lambda: memory.create("/memories/dir", "x"), MemoryToolError, "create over a directory")
    expect_equal(_read(memory, "/memories/dir/b.md"), "b", "the directory's file")


def size_limits(context: Context) -> None:
    """A file over ``max_file_chars`` (default 100,000) is refused by every write; nothing changes."""
    memory = context.make()
    limit = int(getattr(memory, "max_file_chars", 100_000))
    memory.create("/memories/a.md", "x" * (limit - 10))
    expect_error(
        lambda: memory.create("/memories/b.md", "y" * (limit + 1)), MemoryToolError, "create over the limit"
    )
    expect_error(
        lambda: memory.insert("/memories/a.md", 0, "z" * 20), MemoryToolError, "insert over the limit"
    )
    old = "x" * (limit - 10)
    expect_error(
        lambda: memory.str_replace("/memories/a.md", old, "z" * (limit + 1)), MemoryToolError, "str_replace"
    )
    expect_error(lambda: memory.view("/memories/b.md"), MemoryToolError, "the refused file")
    memory.str_replace("/memories/a.md", old, "unchanged until here")  # matches only the unchanged content
    expect_equal(_read(memory, "/memories/a.md"), "unchanged until here", "content")


def execute_payloads(context: Context) -> None:
    """``execute`` runs tool inputs and refuses unknown commands and missing or extra fields."""
    memory = context.make()
    expect(
        "created" in memory.execute({"command": "create", "path": "/memories/x.md", "file_text": "x"}),
        "create",
    )
    expect("x" in memory.execute({"command": "view", "path": "/memories/x.md"}), "view")
    for payload in (
        {"command": "format", "path": "/memories/x.md"},
        {"command": "create", "path": "/memories/y.md"},
        {"command": "view"},
        {},
    ):
        expect_error(partial(memory.execute, payload), MemoryToolError, f"execute({payload!r})")


def second_instance(context: Context) -> None:
    """Another instance on the same storage and partition sees the files (``concurrent_writers`` backends)."""
    first = context.make()
    first.create("/memories/shared.md", "shared")
    second = context.make()
    expect_equal(_read(second, "/memories/shared.md"), "shared", "seen by a second instance")
    second.delete("/memories/shared.md")
    expect_error(
        lambda: first.view("/memories/shared.md"), MemoryToolError, "a delete by the second instance"
    )


def concurrent_inserts(context: Context) -> None:
    """Two instances inserting into one file from several threads lose no line (``concurrent_writers``)."""
    memories = [context.make(), context.make()]
    memories[0].create("/memories/log.md", "")
    errors: list[BaseException] = []

    def work(memory: Any, worker: int) -> None:
        try:
            for number in range(15):
                memory.insert("/memories/log.md", 0, f"line {worker}-{number}")
        except BaseException as exc:  # reported below
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(memories[w % 2], w)) for w in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(120)
    expect_equal(errors, [], "errors of the writers")
    lines = [line for line in _read(memories[1], "/memories/log.md").split("\n") if line]
    expect_equal(len(lines), 60, "lines after 60 concurrent inserts")
    expect_equal(len(set(lines)), 60, "distinct lines")


def _case(name: str, run: Callable[[Context], None], *needs: str, security: bool = False) -> Case:
    return Case(name, frozenset(needs), run, "memory", security)


MEMORY_CASES: tuple[Case, ...] = (
    _case("memory.create_and_view", create_and_view),
    _case("memory.paths_never_leave_the_root", path_rules, security=True),
    _case("memory.str_replace", str_replace),
    _case("memory.insert", insert),
    _case("memory.delete", delete),
    _case("memory.rename", rename),
    _case("memory.files_are_not_directories", files_and_directories),
    _case("memory.size_limits_change_nothing", size_limits),
    _case("memory.execute_validates_payloads", execute_payloads),
    _case("memory.second_instance_sees_the_files", second_instance, "concurrent_writers"),
    _case("memory.concurrent_inserts_lose_nothing", concurrent_inserts, "concurrent_writers"),
)
"""T-CONF-6: the ``MemoryBackend`` contract."""
