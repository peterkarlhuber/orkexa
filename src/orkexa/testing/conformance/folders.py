"""Conformance cases of folder grants (T-CONF-9; Nodestro N-63): read-only access to granted folders.

A product grants a folder to its agents, and its tools then list, read and search the files below it.
These cases check such an implementation from the outside, through the operations of
:class:`FolderAccess`: what can be granted at all, the path guard (:mod:`orkexa.fs` and the attack catalog
of :mod:`orkexa.testing.pathguard`), include and exclude patterns, size limits, revocation and expiry, and
the access log. The reference implementation is :class:`orkexa.fs.SafeRoot` under a small grant store
(Orkexa's own tests); Nodestro runs the same cases through its folder tools.

The factory opens a :class:`FolderAccess` for a :class:`~orkexa.testing.conformance.Request`. It must
treat ``request.workdir / DATA_FOLDER`` as the product's own data folder (it may create it): that folder
can never be granted and nothing inside it can be read. Refusals raise an exception with a ``code``
attribute: the :mod:`orkexa.fs` code (an :class:`~orkexa.fs.FsError` has it), or for a grant that ended
one of :data:`GRANT_ENDED_CODES`, for a folder that cannot be granted one of :data:`NOT_GRANTABLE_CODES`.
"""

from __future__ import annotations

import builtins
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from functools import partial
import os
from pathlib import Path
import time
from typing import Protocol, runtime_checkable
import unicodedata

from .. import pathguard
from ._core import Case, Context, expect, expect_equal, expect_error

__all__ = [
    "DATA_FOLDER",
    "FOLDER_CASES",
    "GRANT_ENDED_CODES",
    "NOT_GRANTABLE_CODES",
    "FolderAccess",
]

#: The name of the product's data folder in the request's working directory.
DATA_FOLDER = "data"
#: Codes of a refusal because the grant was revoked or expired (the product's own codes may differ from
#: ``fs.revoked``: a grant model that just finds no grant any more says so).
GRANT_ENDED_CODES = frozenset({"fs.revoked", "access.missing_grant", "access.denied"})
#: Codes of a folder that cannot be granted: the path guard's, a safety limit of the product's grant model,
#: or the product's own refusal of the folder.
NOT_GRANTABLE_CODES = frozenset(
    {"fs.protected_root", "fs.network_path", "fs.invalid_path", "access.safety_limit", "folder.not_grantable"}
)


@runtime_checkable
class FolderAccess(Protocol):
    """Read-only access to granted folders, as an agent's tools use it.

    ``grant`` stores a grant of the absolute folder ``path`` and returns its id; a folder that cannot be
    granted raises (:data:`NOT_GRANTABLE_CODES`). ``list`` returns the relative paths of the entries of a
    folder below the grant (``""``: the granted folder), ``read`` the text of a file, ``search`` the
    relative paths of the files whose name or text contains ``text`` (case-insensitive). ``revoke`` ends a
    grant at once. ``accesses`` is the access log of a grant: one ``(operation, outcome)`` per call, with
    ``operation`` one of ``list``, ``read``, ``search`` and ``outcome`` one of ``success``, ``denied``,
    ``failure``. The log never holds a path or a file name.
    """

    def grant(
        self,
        path: str,
        *,
        include: Sequence[str] = (),
        exclude: Sequence[str] = (),
        max_file_bytes: int | None = None,
        expires_at: datetime | None = None,
    ) -> str: ...

    def revoke(self, grant: str) -> None: ...

    def list(self, grant: str, path: str = "") -> builtins.list[str]: ...

    def read(self, grant: str, path: str | bytes) -> str: ...

    def search(self, grant: str, text: str) -> builtins.list[str]: ...

    def accesses(self, grant: str) -> builtins.list[tuple[str, str]]: ...


def _tree(context: Context, name: str = "granted") -> Path:
    """A folder with a few files below the working directory."""
    root = context.workdir / name
    (root / "notes").mkdir(parents=True)
    (root / "readme.md").write_text("Project overview\n", encoding="utf-8", newline="\n")
    (root / "notes" / "plan.txt").write_text("The budget is 42 units\n", encoding="utf-8", newline="\n")
    (root / "notes" / "todo.md").write_text("call the supplier\n", encoding="utf-8", newline="\n")
    return root


def _code(error: BaseException) -> object:
    return getattr(error, "code", None)


def _refused(call: Callable[[], object], what: str, codes: Sequence[str] | frozenset[str]) -> BaseException:
    error = expect_error(call, Exception, what)
    expect(
        _code(error) in set(codes), f"{what}: expected one of {sorted(codes)}, got {_code(error)!r}: {error}"
    )
    return error


def _read_bytes(access: FolderAccess, grant: str, path: str | bytes) -> bytes:
    return access.read(grant, path).encode("utf-8")


def _data(context: Context) -> Path:
    folder = context.workdir / DATA_FOLDER
    folder.mkdir(exist_ok=True)
    (folder / "store.db").write_bytes(pathguard.CANARY)
    (folder / "sub").mkdir(exist_ok=True)
    return folder


def list_read_search(context: Context) -> None:
    """A granted folder is listed, read and searched; the answers name relative paths."""
    access = context.make()
    grant = access.grant(str(_tree(context)))
    expect_equal(sorted(access.list(grant)), ["notes", "readme.md"], "the listing of the granted folder")
    expect_equal(sorted(access.list(grant, "notes")), ["notes/plan.txt", "notes/todo.md"], "a subfolder")
    expect_equal(access.read(grant, "notes/plan.txt"), "The budget is 42 units\n", "a file's text")
    expect_equal(sorted(access.search(grant, "BUDGET")), ["notes/plan.txt"], "a search by text")
    expect_equal(sorted(access.search(grant, "todo")), ["notes/todo.md"], "a search by name")
    expect_equal(access.search(grant, "nowhere-to-be-found"), [], "a search without a hit")


def data_folder_not_grantable(context: Context) -> None:
    """The product's data folder, a folder inside it and the same folder through a link cannot be granted."""
    access = context.make()
    data = _data(context)
    _refused(lambda: access.grant(str(data)), "granting the data folder", NOT_GRANTABLE_CODES)
    _refused(lambda: access.grant(str(data / "sub")), "granting a folder inside it", NOT_GRANTABLE_CODES)
    link = context.workdir / "data-link"
    try:
        os.symlink(data, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        return  # no symbolic links here (Windows without the privilege)
    _refused(lambda: access.grant(str(link)), "granting it through a link", NOT_GRANTABLE_CODES)


def data_folder_inside_a_grant(context: Context) -> None:
    """A grant of a folder that holds the data folder never lists, reads or finds anything inside it."""
    access = context.make()
    _tree(context)
    _data(context)
    grant = access.grant(str(context.workdir))
    expect(DATA_FOLDER not in access.list(grant), "the data folder is not listed")
    inside = ["fs.blocked", "access.safety_limit"]
    _refused(lambda: access.read(grant, f"{DATA_FOLDER}/store.db"), "reading inside it", inside)
    _refused(lambda: access.list(grant, DATA_FOLDER), "listing it", inside)
    found = access.search(grant, pathguard.CANARY.decode())
    expect_equal(found, [], "a search for the content of the data folder")


def system_folders_not_grantable(context: Context) -> None:
    """File-system roots, the home folder and the system folders that exist here cannot be granted."""
    from ...fs import system_folders

    access = context.make()
    candidates = [os.path.abspath(os.sep), str(Path.home()), *system_folders()]
    for folder in dict.fromkeys(candidates):
        if os.path.isdir(folder):
            _refused(partial(access.grant, folder), f"granting {folder}", NOT_GRANTABLE_CODES)


def attack_catalog(context: Context) -> None:
    """Every name and link attack of the path-guard catalog is refused with its code (N10-01 to N10-23)."""
    access = context.make()
    layout = pathguard.build_layout(context.workdir / "attack")
    grant = access.grant(str(layout.root))
    # Any exception with the expected code is a refusal: a product's tools raise their own errors.
    findings = pathguard.check_reader(partial(_read_bytes, access, grant), layout, refusal=Exception)
    expect_equal([f"{item.case} {item.path!r}: {item.problem}" for item in findings], [], "attack findings")
    listed = access.list(grant)
    for name in ("link_file", "link_dir", "chain1", "junction"):
        if name in listed:
            _refused(partial(access.list, grant, name), f"listing through {name}", ["fs.link"])
    found = access.search(grant, pathguard.CANARY.decode())
    expect_equal(found, [], "a search for the content outside the folder")
    found = access.search(grant, pathguard.BLOCKED_CANARY.decode())
    expect_equal(found, [], "a search for the content of blocked files")


def swap_race(context: Context) -> None:
    """A file or folder swapped for a link to the outside while it is read never leaks it (N10-04)."""
    access = context.make()
    for folder in (False, True):
        layout = pathguard.build_layout(context.workdir / f"race-{int(folder)}")
        grant = access.grant(str(layout.root))
        findings = pathguard.swap_race(
            partial(_read_bytes, access, grant),
            layout,
            iterations=300,
            folder=folder,
            min_swaps=3,
        )
        expect_equal([f"{item.path}: {item.problem}" for item in findings], [], "swap race findings")


def exclude_patterns(context: Context) -> None:
    """Excluded files and folders are never listed, read or found, in any case or Unicode spelling."""
    access = context.make()
    root = _tree(context)
    (root / "private").mkdir()
    (root / "private" / "salary.txt").write_text("secret budget\n", encoding="utf-8", newline="\n")
    (root / "notes" / "server.key").write_text("secret budget\n", encoding="utf-8", newline="\n")
    nfd = unicodedata.normalize("NFD", "café")
    (root / nfd).mkdir()
    (root / nfd / "menu.txt").write_text("secret budget\n", encoding="utf-8", newline="\n")
    grant = access.grant(str(root), exclude=["private", "**/*.key", "café"])
    expect_equal(sorted(access.list(grant)), ["notes", "readme.md"], "the listing without excluded entries")
    expect_equal(sorted(access.list(grant, "notes")), ["notes/plan.txt", "notes/todo.md"], "a subfolder")
    expect_equal(access.search(grant, "secret budget"), [], "a search for the text of excluded files")
    for path in (
        "private/salary.txt",
        "PRIVATE/salary.txt",
        "Private/Salary.TXT",
        "notes/server.key",
        "notes/SERVER.KEY",
        "café/menu.txt",
        f"{nfd}/menu.txt",
        "CAFÉ/menu.txt",
    ):
        _refused(partial(access.read, grant, path), f"reading {path!r}", ["fs.excluded", "fs.not_found"])


def include_patterns(context: Context) -> None:
    """With include patterns only matching files are listed, read and found; folders stay listed."""
    access = context.make()
    root = _tree(context)
    grant = access.grant(str(root), include=["**/*.md"])
    expect_equal(sorted(access.list(grant)), ["notes", "readme.md"], "the listing")
    expect_equal(access.list(grant, "notes"), ["notes/todo.md"], "a subfolder")
    _refused(
        lambda: access.read(grant, "notes/plan.txt"), "reading a file that is not included", ["fs.excluded"]
    )
    expect_equal(access.search(grant, "budget"), [], "a search for the text of a file that is not included")


def size_limit(context: Context) -> None:
    """A file above the size limit is refused with a message that names the limit; a smaller one is read."""
    access = context.make()
    root = _tree(context)
    (root / "big.txt").write_text("x" * 1001, encoding="utf-8", newline="\n")
    (root / "small.txt").write_text("x" * 1000, encoding="utf-8", newline="\n")
    grant = access.grant(str(root), max_file_bytes=1000)
    error = _refused(
        lambda: access.read(grant, "big.txt"), "reading a file above the limit", ["fs.too_large"]
    )
    expect("1000" in str(error), f"the message names the limit: {error}")
    expect_equal(len(access.read(grant, "small.txt")), 1000, "a file at the limit")


def revoked(context: Context) -> None:
    """A revoked grant refuses every later call (N10-21)."""
    access = context.make()
    grant = access.grant(str(_tree(context)))
    expect_equal(access.read(grant, "readme.md"), "Project overview\n", "before the revocation")
    access.revoke(grant)
    _refused(lambda: access.read(grant, "readme.md"), "reading after the revocation", GRANT_ENDED_CODES)
    _refused(lambda: access.list(grant), "listing after the revocation", GRANT_ENDED_CODES)
    _refused(lambda: access.search(grant, "budget"), "searching after the revocation", GRANT_ENDED_CODES)


def expired(context: Context) -> None:
    """A grant with an expiry refuses every call once it has expired."""
    access = context.make()
    grant = access.grant(str(_tree(context)), expires_at=datetime.now(UTC) + timedelta(seconds=1.5))
    expect_equal(access.read(grant, "readme.md"), "Project overview\n", "before the expiry")
    time.sleep(2.0)
    _refused(lambda: access.read(grant, "readme.md"), "reading after the expiry", GRANT_ENDED_CODES)


def root_replaced(context: Context) -> None:
    """A granted folder that is moved away and replaced by another folder is refused (N10-05)."""
    access = context.make()
    root = _tree(context)
    grant = access.grant(str(root))
    expect_equal(access.read(grant, "readme.md"), "Project overview\n", "before the swap")
    root.rename(context.workdir / "moved")
    root.mkdir()
    (root / "readme.md").write_text("Planted\n", encoding="utf-8", newline="\n")
    _refused(
        lambda: access.read(grant, "readme.md"),
        "reading a replaced folder",
        ["fs.root_changed", "fs.not_found"],
    )


def unicode_names(context: Context) -> None:
    """A file stored with a decomposed name is read with the composed name (W06-2)."""
    access = context.make()
    root = _tree(context)
    decomposed = root / unicodedata.normalize("NFD", "Übersicht.txt")
    decomposed.write_text("Inhalt\n", encoding="utf-8", newline="\n")
    grant = access.grant(str(root))
    expect_equal(access.read(grant, "Übersicht.txt"), "Inhalt\n", "the composed name")
    expect("Übersicht.txt" in access.list(grant), "the listing names it composed")


def access_log(context: Context) -> None:
    """Every call is in the grant's access log with its outcome, and the log holds no path or file name."""
    access = context.make()
    root = _tree(context)
    grant = access.grant(str(root), exclude=["private"])
    access.list(grant)
    access.read(grant, "notes/plan.txt")
    access.search(grant, "budget")
    expect_error(lambda: access.read(grant, "private/salary.txt"), Exception, "an excluded file")
    log = access.accesses(grant)
    expect_equal(
        log,
        [("list", "success"), ("read", "success"), ("search", "success"), ("read", "denied")],
        "the access log",
    )
    text = repr(log)
    for name in ("plan", "notes", "salary", "private", str(root)):
        expect(name not in text, f"the access log names {name!r}")
    other = access.grant(str(_tree(context, "other")))
    expect_equal(access.accesses(other), [], "the log of another grant")


def _case(name: str, run: Callable[[Context], None], *, security: bool = False) -> Case:
    return Case(name, frozenset(), run, "folders", security)


FOLDER_CASES: tuple[Case, ...] = (
    _case("folders.list_read_search", list_read_search),
    _case("folders.data_folder_not_grantable", data_folder_not_grantable, security=True),
    _case("folders.data_folder_inside_a_grant", data_folder_inside_a_grant, security=True),
    _case("folders.system_folders_not_grantable", system_folders_not_grantable, security=True),
    _case("folders.attack_catalog", attack_catalog, security=True),
    _case("folders.swap_race", swap_race, security=True),
    _case("folders.exclude_patterns", exclude_patterns, security=True),
    _case("folders.include_patterns", include_patterns, security=True),
    _case("folders.size_limit", size_limit, security=True),
    _case("folders.revoked", revoked, security=True),
    _case("folders.expired", expired, security=True),
    _case("folders.root_replaced", root_replaced, security=True),
    _case("folders.unicode_names", unicode_names),
    _case("folders.access_log", access_log, security=True),
)
"""T-CONF-9: folder grants (:class:`FolderAccess`)."""
