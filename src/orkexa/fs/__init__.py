"""A path guard for reading files below one folder: :class:`SafeRoot` (Nodestro plan N-10, design P0-2).

A product grants a folder; an agent's tools then name files *relative* to it. :class:`SafeRoot` makes sure
that such a name can only ever read a regular file that really lies below the folder, on Linux, macOS and
Windows::

    from orkexa.fs import SafeRoot

    root = SafeRoot("/home/ada/Documents/Project", protected=[data_dir], exclude=["**/*.key"])
    root.list("")                        # names, kinds and sizes; links are shown, never followed
    root.read_bytes("notes/plan.md")     # bytes, at most max_file_bytes
    root.extract_text("specs/offer.pdf") # text within a page, character, time and memory budget

How it holds (details in ``docs/guides/path-guard.md``):

* **Names first.** Relative paths only, ``/``-separated, NFC-normalized; ``..``, absolute paths, drives, UNC
  and device prefixes, ``:`` (alternate data streams), ``\\``, reserved device names (``CON``, ``NUL.txt``),
  trailing dots and spaces, 8.3 short names, look-alike separators, control characters and bad UTF-8 are
  refused with the same rules on every operating system.
* **Descriptors, not path strings.** The root is reopened for every call and must still be the folder that was
  checked (its identity and, on macOS and Windows, its final path). Each component is then opened relative to
  its folder without following links: ``openat`` with ``O_NOFOLLOW`` (``openat2`` with ``RESOLVE_*`` on Linux)
  on POSIX, handles with ``FILE_FLAG_OPEN_REPARSE_POINT`` and no ``FILE_SHARE_DELETE`` on Windows, where the
  final path of every handle must be the root plus exactly the names asked for. The file is read through the
  very descriptor that was checked (no time-of-check to time-of-use gap).
* **Only regular files** are read: FIFOs, sockets and devices are refused without blocking. A file with more
  than one hard link is refused unless ``allow_hard_links`` (A-15). Mounts below the root are refused.
* **Blocklist and protected folders.** :data:`DEFAULT_BLOCKED` (``.ssh``, keychains, credentials ...) is
  compared NFC plus case-folded; ``protected`` folders (the product's data folder) are compared by identity,
  so no spelling reaches them. A system folder, a drive or file-system root, the home folder itself, any
  user's home folder, a blocked folder (also given through a link), a protected folder and any folder inside
  one cannot be a root at all (``fs.protected_root``).

Every refusal is an :class:`FsError` with a stable ``code`` (:data:`FS_ERROR_CODES`). This module uses only
the standard library (``ctypes`` on Windows and for ``openat2``); PDF text extraction needs ``orkexa[pdf]``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import os
from pathlib import Path
from stat import S_ISLNK
import sys
import threading
import time
from typing import Any, Literal
import unicodedata

from ._names import (
    DEFAULT_BLOCKED,
    FS_ERROR_CODES,
    MAX_DEPTH,
    MAX_NAME_BYTES,
    MAX_PATH_BYTES,
    FsError,
    Pattern,
    State,
    allowed_folders,
    display_name,
    fold,
    home_parents,
    inside,
    is_short_name,
    path_parts,
    patterns,
    root_syntax,
    split_relative,
    system_folders,
)
from ._types import Info

if sys.platform == "win32":
    from . import _windows as _backend
else:
    from . import _posix as _backend

__all__ = [
    "DEFAULT_BLOCKED",
    "DEFAULT_BROWSE_FOLDERS",
    "DEFAULT_LIST_SECONDS",
    "DEFAULT_MAX_ENTRIES",
    "DEFAULT_MAX_FILE_BYTES",
    "FS_ERROR_CODES",
    "MAX_DEPTH",
    "MAX_NAME_BYTES",
    "MAX_PATH_BYTES",
    "PDF_SUFFIXES",
    "TEXT_SUFFIXES",
    "Entry",
    "ExtractedText",
    "Folder",
    "FolderListing",
    "FsError",
    "SafeRoot",
    "TextBudget",
    "allowed_folders",
    "browse_folder",
    "browse_problem",
    "display_name",
    "fold",
    "is_short_name",
    "root_problem",
    "split_relative",
    "system_folders",
]

#: Largest file :meth:`SafeRoot.read_bytes` reads unless the root or the call says otherwise (50 MiB).
DEFAULT_MAX_FILE_BYTES = 50 * 1024 * 1024
#: Most entries :meth:`SafeRoot.list` reads from one folder (hidden ones included).
DEFAULT_MAX_ENTRIES = 10_000
#: Longest time one :meth:`SafeRoot.list` may take, in seconds.
DEFAULT_LIST_SECONDS = 10.0
#: File name suffixes :meth:`SafeRoot.extract_text` reads as text (UTF-8, or UTF-16 with a byte order mark).
TEXT_SUFFIXES = frozenset(
    {
        ".txt", ".text", ".md", ".markdown", ".rst", ".csv", ".tsv", ".json", ".jsonl", ".yaml", ".yml",
        ".toml", ".ini", ".cfg", ".conf", ".log", ".xml", ".html", ".htm", ".css", ".js", ".ts", ".py",
        ".java", ".c", ".h", ".cpp", ".cs", ".go", ".rs", ".rb", ".php", ".sh", ".ps1", ".sql", ".tex",
        ".srt", ".vtt",
    }
)  # fmt: skip
#: File name suffixes :meth:`SafeRoot.extract_text` reads as PDF (in an isolated process, with budgets).
PDF_SUFFIXES = frozenset({".pdf"})

Kind = Literal["file", "dir", "link", "other"]


@dataclass(frozen=True)
class Entry:
    """One entry of a folder listing (:meth:`SafeRoot.list`) or the answer of :meth:`SafeRoot.stat`.

    ``name`` is the name as stored; ``path`` the relative path to pass back (NFC). ``kind`` is ``file``,
    ``dir``, ``link`` (a symbolic link, junction or mount point: listed, never followed) or ``other`` (FIFO,
    socket, device). ``size`` is in bytes for files, else 0. ``ambiguous`` marks names that equal another
    entry's after NFC normalization and case folding (only the exact name opens). ``display`` is the name with
    bidirectional controls and zero-width characters escaped, for showing it to a person.
    """

    name: str
    path: str
    kind: Kind
    size: int
    ambiguous: bool = False

    @property
    def display(self) -> str:
        return display_name(self.name)


@dataclass(frozen=True)
class TextBudget:
    """Budgets of one :meth:`SafeRoot.extract_text` call.

    ``max_bytes`` caps the file read (``None``: the root's ``max_file_bytes``); ``max_chars`` the text
    returned; ``max_pages``, ``timeout`` (seconds) and ``max_memory_mb`` the PDF extraction, which runs in a
    separate process that is killed at the time budget (:func:`orkexa.retrieval.pdf_pages_isolated`). The
    memory budget is enforced on Linux (``RLIMIT_AS``), macOS (the process is killed when its physical
    footprint passes the budget) and Windows (a job object with a process memory limit); where it cannot be
    enforced a PDF is refused (``fs.unsupported_type``) rather than read without it.
    """

    max_bytes: int | None = None
    max_chars: int = 1_000_000
    max_pages: int = 500
    timeout: float = 30.0
    max_memory_mb: int = 512

    def __post_init__(self) -> None:
        if self.max_bytes is not None and self.max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        if self.max_chars < 1 or self.max_pages < 1 or self.timeout <= 0 or self.max_memory_mb < 16:
            raise ValueError("Text budgets must be positive (max_memory_mb at least 16)")


@dataclass(frozen=True)
class ExtractedText:
    """The text of one file (:meth:`SafeRoot.extract_text`); ``pages`` holds a PDF's pages."""

    path: str
    kind: Literal["text", "pdf"]
    text: str
    size: int
    pages: tuple[str, ...] | None = None


def _identity(info: Info) -> tuple[int, int]:
    return (info.dev, info.ino)


def _ancestors(path: str) -> list[str]:
    """``path`` and every folder above it, nearest first."""
    found = [path]
    current = path
    while True:
        parent = os.path.dirname(current)
        if not parent or parent == current:
            return found
        found.append(parent)
        current = parent


def _home(home: str | os.PathLike[str] | None) -> str | None:
    if home is not None:
        return os.path.realpath(os.fspath(home))
    try:
        return os.path.realpath(Path.home())
    except (RuntimeError, KeyError, OSError):
        return None


def _first_link(path: str) -> tuple[str, tuple[str, ...]] | None:
    """The first component of ``path`` (absolute) that is a link or junction, and the names after it."""
    parts = Path(path).parts
    current = parts[0] if parts else path
    for index, name in enumerate(parts[1:], 1):
        current = os.path.join(current, name)
        if _is_link(current):
            return current, tuple(parts[index + 1 :])
    return None


def _is_link(path: str) -> bool:
    """A symbolic link, or on Windows a junction (mount point reparse point)."""
    try:
        found = os.lstat(path)
    except OSError:
        return False
    return S_ISLNK(found.st_mode) or getattr(found, "st_reparse_tag", 0) in (0xA0000003, 0xA000000C)


def _link_spellings(path: str) -> list[str]:
    """``path`` as given (absolute) and every path it turns into as its links are resolved one at a time.

    ``~/.ssh`` that is a link to ``/vault/keys`` yields ``~/.ssh`` and ``/vault/keys``; a link to ``~/.ssh``
    yields its own name and ``~/.ssh``. Used to apply blocked names to every spelling of a root (review N2).
    """
    current = os.path.abspath(path)
    found = [current]
    for _ in range(40):  # the link limit of Linux
        link = _first_link(current)
        if link is None:
            break
        head, rest = link
        try:
            target = os.readlink(head)
        except (OSError, ValueError):
            break
        if target.startswith("\\\\?\\"):
            target = target[4:]
        current = os.path.normpath(os.path.join(os.path.dirname(head), target, *rest))
        found.append(current)
    return found


def _identities(paths: Iterable[str]) -> set[tuple[int, int]]:
    return {_identity(item) for item in map(_backend.stat_path, paths) if item is not None}


def root_problem(
    path: str,
    info: Info | None = None,
    *,
    protected: Iterable[str | os.PathLike[str]] = (),
    home: str | os.PathLike[str] | None = None,
    blocked: Iterable[str] = DEFAULT_BLOCKED,
    given: str | os.PathLike[str] | None = None,
) -> str | None:
    """Why the canonical folder ``path`` cannot be a root (a message), or ``None``.

    Refused:

    * a drive or file-system root;
    * the home folder itself or any folder that contains it (``home``, default: the user's home folder), the
      folders that hold every user's home folder (``/home``, ``/Users``, ``C:\\Users``) and every folder
      directly inside them (any user's home folder, ``/home/bob``, ``C:\\Users\\Public``);
    * a system folder (:func:`system_folders`) or a folder inside one, except the temporary and
      removable-media folders of :data:`~orkexa.fs._names.POSIX_ALLOWED_FOLDERS`;
    * a folder that the ``blocked`` patterns starting with ``**/`` cover by its absolute path (``~/.ssh``,
      ``~/Library/Keychains`` or a folder inside one), by the canonical path and, with ``given``, by the path
      as it was given and every path its links lead through (a ``~/.ssh`` that is a link elsewhere);
    * a ``protected`` folder or a folder inside one.

    Folders are compared by name (component-wise, NFC plus case-folded) and by identity (device and inode, or
    volume and file index), so case, Unicode normalization, 8.3 names and links cannot alias them. ``path``
    must already be canonical (:class:`SafeRoot` does that).
    """
    parts = path_parts(path)
    if len(parts) <= (1 if os.name == "nt" else 0):
        return "The root of a drive or file system cannot be a root"
    chain = [_backend.stat_path(folder) for folder in _ancestors(path)]
    ids = [_identity(item) if item is not None else None for item in chain]
    if info is not None:
        ids[0] = _identity(info)
    real_home = _home(home)
    if real_home and (inside(real_home, path) or ids[0] in _identities(_ancestors(real_home))):
        if path_parts(real_home) == parts:
            return "The home folder itself cannot be a root; choose a folder inside it"
        return "A folder that contains the home folder cannot be a root; choose a folder inside it"
    parents = home_parents()
    parent_ids = _identities(parents)
    if any(path_parts(parent) == parts for parent in parents) or ids[0] in parent_ids:
        return "The folder of all home folders cannot be a root; choose a folder inside your home folder"
    # Every folder directly inside them is some user's home (N1); a parent inside another one (%PUBLIC% in
    # C:\Users) is itself such a home, whose own folders (Documents ...) may be granted.
    outer = [
        parent
        for parent in parents
        if not any(inside(parent, other) and path_parts(parent) != path_parts(other) for other in parents)
    ]
    if any(path_parts(parent) == parts[:-1] for parent in outer) or (
        len(ids) > 1 and ids[1] in _identities(outer)
    ):
        return "A user's home folder cannot be a root; choose a folder inside it"
    spellings = [parts]
    if given:
        walked = _link_spellings(os.fspath(given))
        # The walk reads the links again, after the root was resolved: a link flipped in between would let
        # the names of one chain vouch for another folder. The chain must end at the folder that was resolved.
        end = _backend.stat_path(walked[-1])
        if ids[0] is not None and (end is None or _identity(end) != ids[0]):
            return "The links on the way to the root changed while it was checked"
        spellings += [path_parts(item) for item in walked]
    for pattern in patterns(blocked):
        if pattern.floating and any(pattern.covers(spelling) for spelling in spellings):
            return "A blocked folder (credentials, keys, keychains) or a folder inside one cannot be a root"
    for folder in protected:
        text = os.fspath(folder)
        found = _backend.stat_path(text)
        real = os.path.realpath(text)
        if inside(path, text) or inside(path, real) or (found is not None and _identity(found) in ids):
            return "A protected folder (or a folder inside one) cannot be a root"
    allowed = _identities(allowed_folders())
    system = _identities(system_folders())
    for identity in ids:
        if identity in allowed:
            break
        if identity in system:
            return "A system folder (or a folder inside one) cannot be a root"
    else:
        if any(inside(path, folder) for folder in system_folders()) and not any(
            inside(path, folder) for folder in allowed_folders()
        ):
            return "A system folder (or a folder inside one) cannot be a root"
    return None


class SafeRoot:
    """Read-only access to the files below one folder, refusing everything that could lead outside it.

    Args:
        path: The folder (absolute). It is canonicalized once (links resolved; on Windows the final path,
            which also expands 8.3 short names) and refused when it cannot be a root (:func:`root_problem`,
            ``fs.protected_root``), when it is a network path or drive (``fs.network_path``) or a Windows
            device path (``fs.invalid_path``).
        include: Patterns (:class:`~orkexa.fs._names.Pattern` syntax: ``*``, ``?``, ``**``, anchored at the
            root unless they start with ``**/``) a file must match to be read; empty means every file.
        exclude: Patterns of files and folders that are never read or listed (``fs.excluded``).
        blocked: Patterns that are refused as ``fs.blocked``; :data:`DEFAULT_BLOCKED` unless replaced. A
            pattern starting with ``**/`` is matched against the absolute path, so it also applies when the
            root itself lies in or above a blocked folder (``~/.config`` as root still blocks ``gh/...``).
        protected: Folders (the product's data folder ...) that can neither be the root nor be reached below
            it. They are compared by name below the root (also before they exist) and by identity, both looked
            up again on every call, so a protected folder that is created, deleted or recreated later is
            still refused.
        home: The user's home folder, which neither it nor a folder containing it may be the root of (default:
            the home folder of this process's user). Every other user's home folder (any folder directly in
            ``/home``, ``/Users``, ``C:\\Users`` ...) is refused as well.
        max_file_bytes: The size limit of a file (``fs.too_large``).
        max_entries: The most entries one listing reads, hidden ones included (``fs.too_many_entries``).
        max_list_seconds: The longest one listing may take (``fs.too_many_entries``, reason ``timeout``).
        allow_hard_links: Read files with more than one hard link (default: refused with ``fs.hard_link``,
            since a hard link can be the very file of another folder; A-15).
        allow_cloud_files: Read OneDrive cloud placeholders on Windows, which may download them (default:
            refused with ``fs.cloud_file``).
        writable: Allow :meth:`create_file` and :meth:`rename` (default: read only, ``fs.read_only``).
        expect_identity: The ``(device, inode)`` recorded when the folder was granted (:attr:`identity`); a
            different folder at the same path raises ``fs.root_changed``.
        on_open: Called while a path is opened, after each component, with the names the file system stored
            for the components opened so far (the root itself is ``()``). On a case-insensitive file system
            a request may name an entry in another spelling (case, dotless i, ignorable characters); these
            are the entry's own names, so a product checks its own rules (a person's deny) against what is
            really opened. It raises to refuse; the path is then neither read nor listed.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        include: Iterable[str] = (),
        exclude: Iterable[str] = (),
        blocked: Iterable[str] = DEFAULT_BLOCKED,
        protected: Iterable[str | os.PathLike[str]] = (),
        home: str | os.PathLike[str] | None = None,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        max_list_seconds: float = DEFAULT_LIST_SECONDS,
        allow_hard_links: bool = False,
        allow_cloud_files: bool = False,
        expect_identity: tuple[int, int] | None = None,
        writable: bool = False,
        on_open: Callable[[tuple[str, ...]], None] | None = None,
    ) -> None:
        if max_file_bytes < 0 or max_entries < 1 or max_list_seconds <= 0:
            raise ValueError(
                "max_file_bytes must not be negative, max_entries and max_list_seconds must be positive"
            )
        text = root_syntax(os.fspath(path))
        protected_paths = [os.fspath(folder) for folder in protected]
        blocked_patterns = patterns(blocked)
        blocked_texts = [pattern.text for pattern in blocked_patterns]
        checks: dict[str, Any] = {
            "protected": protected_paths, "home": home, "blocked": blocked_texts, "given": text
        }  # fmt: skip
        try:
            canonical, info = _backend.canonical(text)
        except FsError as error:
            # A folder this user may not open (such as /root) is still reported as not grantable.
            problem = root_problem(os.path.realpath(text), **checks)
            if error.code == "fs.permission" and problem is not None:
                raise FsError("fs.protected_root", problem) from None
            raise
        problem = root_problem(canonical, info, **checks)
        if problem is not None:
            raise FsError("fs.protected_root", problem)
        if expect_identity is not None and tuple(expect_identity) != _identity(info):
            raise FsError("fs.root_changed", "The root folder is not the folder that was granted")
        self._path = canonical
        self._identity = _identity(info)
        self._include = patterns(include)
        self._exclude = patterns(exclude)
        self._blocked = blocked_patterns
        root_parts = path_parts(canonical)
        # Patterns starting with **/ continue from the absolute path of the root (H2); the others are anchored
        # at the root.
        self._blocked_start: list[tuple[Pattern, State]] = [
            (pattern, pattern.run(root_parts)[0] if pattern.floating else pattern.start())
            for pattern in blocked_patterns
        ]
        self._protected_paths = protected_paths
        self._protected_rel: list[tuple[str, ...]] = []
        self._protected_ids: set[tuple[int, int]] = set()
        for folder in protected_paths:
            for spelling in dict.fromkeys([os.path.abspath(folder), os.path.realpath(folder)]):
                if inside(spelling, canonical):
                    self._protected_rel.append(path_parts(spelling)[len(root_parts) :])
            found = _backend.stat_path(folder)
            if found is not None:
                self._protected_ids.add(_identity(found))
        self.max_file_bytes = max_file_bytes
        self.max_entries = max_entries
        self.max_list_seconds = max_list_seconds
        self.allow_hard_links = allow_hard_links
        self.allow_cloud_files = allow_cloud_files
        self.writable = writable
        self._on_open = on_open
        self._revoked = threading.Event()

    def __repr__(self) -> str:
        # The path names a person's folders: it is not shown (use .path where it is needed).
        return f"SafeRoot(<redacted>, writable={getattr(self, 'writable', False)})"

    @property
    def path(self) -> str:
        """The canonical root folder."""
        return self._path

    @property
    def identity(self) -> tuple[int, int]:
        """``(device, inode)`` on POSIX, ``(volume serial, file index)`` on Windows; keep it with a grant."""
        return self._identity

    # ---- revocation ----------------------------------------------------------------------------------------

    def revoke(self) -> None:
        """Refuse every later call (``fs.revoked``); a read, listing or extraction in progress stops too."""
        self._revoked.set()

    @property
    def revoked(self) -> bool:
        return self._revoked.is_set()

    def _check_revoked(self, shown: str | None = None) -> None:
        if self._revoked.is_set():
            raise FsError("fs.revoked", "The root was revoked", path=shown)

    # ---- checks without I/O --------------------------------------------------------------------------------

    def check(self, path: str | bytes, *, kind: Literal["file", "dir"] = "file") -> tuple[str, ...]:
        """The components of ``path`` after the name rules and the patterns, without file-system access."""
        self._check_revoked()
        parts = split_relative(path, allow_empty=kind == "dir")
        shown = "/".join(parts)
        folded = tuple(fold(part) for part in parts)
        if any(pattern.run(folded, state)[1] for pattern, state in self._blocked_start):
            raise FsError("fs.blocked", "This path is blocked", path=shown)
        if any(folded[: len(rel)] == rel for rel in self._protected_rel):
            self._protected_ids |= _identities(self._protected_paths)  # remember it, should it move
            raise FsError("fs.blocked", "This path is inside a protected folder", path=shown)
        if any(pattern.covers(folded) for pattern in self._exclude):
            raise FsError("fs.excluded", "This path is excluded", path=shown)
        if kind == "file" and self._include and not any(pattern.matches(folded) for pattern in self._include):
            raise FsError("fs.excluded", "This file is not included", path=shown)
        return parts

    def _states(self, folded: tuple[str, ...]) -> list[tuple[State, bool]]:
        """What every pattern (blocked, exclude, include) knows after the folder ``folded``."""
        runs = [pattern.run(folded, state) for pattern, state in self._blocked_start]
        runs += [pattern.run(folded) for pattern in (*self._exclude, *self._include)]
        return runs

    def _on_node(self, shown: str) -> Callable[[int, Info, tuple[str, ...]], None]:
        # Looked up again on every call: a protected folder that was created or recreated since is refused.
        # Every identity seen is kept, so the folder stays protected after it is moved within the root.
        self._protected_ids |= _identities(self._protected_paths)
        protected = set(self._protected_ids)

        on_open = self._on_open

        def check(index: int, info: Info, names: tuple[str, ...]) -> None:
            del index
            if _identity(info) in protected:
                raise FsError("fs.blocked", "This path is inside a protected folder", path=shown)
            if on_open is not None:
                on_open(tuple(unicodedata.normalize("NFC", name) for name in names))

        return check

    def _walk(self, parts: tuple[str, ...], shown: str, final: str) -> Any:
        return _backend.walk(
            self._path,
            self._identity,
            parts,
            final=final,
            on_node=self._on_node(shown),
            shown=shown,
            allow_cloud=self.allow_cloud_files,
        )

    # ---- operations ----------------------------------------------------------------------------------------

    def stat(self, path: str | bytes) -> Entry:
        """Kind and size of one file or folder (links are refused, not reported)."""
        parts = self.check(path, kind="dir")
        shown = "/".join(parts)
        with self._walk(parts, shown, "any") as (_, info):
            kind: Kind = "dir" if info.kind == "dir" else "file"
            if kind == "file" and info.nlink > 1 and not self.allow_hard_links:
                raise FsError("fs.hard_link", "The file has more than one hard link", path=shown)
            return Entry(
                name=parts[-1] if parts else "",
                path=shown,
                kind=kind,
                size=info.size if kind == "file" else 0,
            )

    def list(self, path: str | bytes = "") -> list[Entry]:
        """The entries of a folder below the root (``""`` is the root), sorted by name.

        Entries that could never be opened are left out: names that break the name rules, blocked and excluded
        paths, and protected folders. Files not matched by ``include`` are left out as well; folders stay,
        since files below them may match. Links are listed as ``link`` and never followed. A file with more
        than one hard link is listed with size 0 unless ``allow_hard_links``. At most ``max_entries`` entries
        are read (hidden ones count too) within ``max_list_seconds``; beyond either the listing is refused
        (``fs.too_many_entries``), and a revoked root stops it.
        """
        parts = self.check(path, kind="dir")
        shown = "/".join(parts)
        found: list[Entry] = []
        deadline = time.monotonic() + self.max_list_seconds
        with self._walk(parts, shown, "dir") as (node, _):
            for read, (name, kind, size, links) in enumerate(_backend.listing(node, self._path, parts), 1):
                self._check_revoked(shown)
                if read > self.max_entries:
                    raise FsError(
                        "fs.too_many_entries",
                        f"The folder has more than {self.max_entries} entries",
                        path=shown,
                        reason="entries",
                    )
                if time.monotonic() > deadline:
                    raise FsError(
                        "fs.too_many_entries",
                        f"Listing the folder took longer than {self.max_list_seconds:g} seconds",
                        path=shown,
                        reason="timeout",
                    )
                child = "/".join([*parts, name])
                try:
                    self.check(child, kind="dir" if kind == "dir" else "file")
                except FsError:
                    continue
                if kind == "file" and links > 1 and not self.allow_hard_links:
                    size = 0  # the size of a file that may live elsewhere is not shown (A-15)
                found.append(Entry(name=name, path=_nfc(child), kind=kind, size=size))  # type: ignore[arg-type]
        keys: dict[str, int] = {}
        for entry in found:
            keys[fold(entry.name)] = keys.get(fold(entry.name), 0) + 1
        result = [
            Entry(entry.name, entry.path, entry.kind, entry.size, ambiguous=keys[fold(entry.name)] > 1)
            for entry in found
        ]
        return sorted(result, key=lambda entry: (fold(entry.name), entry.name))

    def read_bytes(self, path: str | bytes, *, max_bytes: int | None = None) -> bytes:
        """The content of a regular file below the root, read through the descriptor that was checked."""
        limit = self.max_file_bytes if max_bytes is None else min(max_bytes, self.max_file_bytes)
        parts = self.check(path, kind="file")
        shown = "/".join(parts)
        with self._walk(parts, shown, "file") as (node, info):
            if info.kind != "file":
                raise FsError("fs.not_regular_file", "Not a regular file", path=shown)
            if info.nlink > 1 and not self.allow_hard_links:
                raise FsError("fs.hard_link", "The file has more than one hard link", path=shown)
            if info.size > limit:
                raise FsError(
                    "fs.too_large",
                    f"The file has {info.size} bytes; at most {limit} bytes can be read",
                    path=shown,
                )
            data = _backend.read(node, limit, self._revoked.is_set, shown)
        if len(data) > limit:
            raise FsError("fs.too_large", f"The file has more than {limit} bytes", path=shown)
        return data

    # ---- writing (only with writable=True) ---------------------------------------------------------------

    def _writable(self, shown: str) -> None:
        if not self.writable:
            raise FsError("fs.read_only", "This root is read only", path=shown)

    def create_file(self, path: str | bytes, data: bytes) -> None:
        """Create a new file below the root with ``data``; an existing entry is never replaced or followed.

        The folder must exist. POSIX: ``openat`` with ``O_CREAT | O_EXCL | O_NOFOLLOW`` in the checked
        folder's descriptor; Windows: ``CREATE_NEW`` below the held folder handles, then the final path of the
        new handle. Raises ``fs.exists`` when the name is taken and ``fs.too_large`` beyond
        ``max_file_bytes``.
        """
        parts = self.check(path, kind="file")
        shown = "/".join(parts)
        self._writable(shown)
        if len(data) > self.max_file_bytes:
            raise FsError("fs.too_large", f"At most {self.max_file_bytes} bytes can be written", path=shown)
        with self._walk(parts[:-1], shown, "dir") as (node, _):
            _backend.create(node, self._path, parts[:-1], parts[-1], bytes(data), shown)

    def rename(self, source: str | bytes, target: str | bytes) -> None:
        """Rename a file or folder below the root; an existing ``target`` is never replaced.

        Refused: links (``fs.link``); files with several hard links unless ``allow_hard_links``; protected
        folders and the folders above them, as source or as target, also before they exist (``fs.blocked``);
        a file that could not be read under its old or
        its new name (blocked, excluded, not included); a folder whose new name would change what the
        blocked, exclude or include patterns decide for anything below it (``fs.excluded``), so a rename
        never makes a hidden file readable. Linux: ``renameat2(RENAME_NOREPLACE)``; macOS:
        ``renameatx_np(RENAME_EXCL)``, both on the descriptors of the checked folders; Windows:
        ``MoveFileExW`` without ``REPLACE_EXISTING`` below the held folder handles (which pin the folders,
        not the entry itself: see the guide).
        """
        src = self.check(source, kind="dir")
        dst = self.check(target, kind="dir")
        shown = "/".join(src)
        self._writable(shown)
        if not src or not dst:
            raise FsError("fs.invalid_path", "The root itself cannot be renamed", path=shown)
        folded_src = tuple(fold(part) for part in src)
        folded_dst = tuple(fold(part) for part in dst)
        if any(rel[: len(folded_src)] == folded_src for rel in self._protected_rel):
            raise FsError("fs.blocked", "A protected folder or a folder above one is not renamed", path=shown)
        if any(rel[: len(folded_dst)] == folded_dst for rel in self._protected_rel):
            # Also while the protected folder does not exist yet: nothing is planted where it will be (N3).
            raise FsError(
                "fs.blocked", "Nothing is renamed to a protected folder or a folder above one", path=shown
            )
        kind = self.stat(source).kind  # refuses links on the way and protected folders, checks the kind
        if kind == "file":
            self.check(source, kind="file")
            self.check(target, kind="file")
        elif self._states(folded_src) != self._states(folded_dst):
            raise FsError(
                "fs.excluded",
                "Renaming this folder would change which blocked, include or exclude patterns apply",
                path=shown,
            )
        with self._walk(src[:-1], shown, "dir") as (src_node, _):
            entry_kind, links = _backend.entry(src_node, src[-1], shown, root=self._path, parts=src[:-1])
            if entry_kind == "link":
                raise FsError("fs.link", "A link is not renamed", path=shown)
            if entry_kind == "file" and links > 1 and not self.allow_hard_links:
                raise FsError("fs.hard_link", "The file has more than one hard link", path=shown)
            with self._walk(dst[:-1], "/".join(dst), "dir") as (dst_node, _):
                _backend.rename(
                    src_node,
                    src[-1],
                    dst_node,
                    dst[-1],
                    kind=entry_kind,
                    shown=shown,
                    root=self._path,
                    parts=(src[:-1], dst[:-1]),
                )

    def extract_text(self, path: str | bytes, *, budget: TextBudget | None = None) -> ExtractedText:
        """The text of a text file or a PDF below the root, within ``budget``.

        Text files (:data:`TEXT_SUFFIXES`) are decoded as UTF-8 (or UTF-16 with a byte order mark); a file
        with NUL bytes is binary (``fs.unsupported_type``). PDFs (:data:`PDF_SUFFIXES`) go through
        :func:`orkexa.retrieval.pdf_pages_isolated`: a separate process with a minimal environment, killed at
        ``budget.timeout``, with ``budget.max_pages``, ``max_chars`` and ``max_memory_mb``. Exceeding a budget
        raises ``fs.too_complex`` with ``reason`` ``pages``, ``chars``, ``timeout``, ``memory``, ``limit`` or
        ``crash``; an unreadable PDF ``fs.unreadable``. Other suffixes raise ``fs.unsupported_type``. A root
        revoked during the extraction raises ``fs.revoked`` instead of returning the text.
        """
        budget = budget or TextBudget()
        parts = self.check(path, kind="file")
        shown = "/".join(parts)
        suffix = os.path.splitext(fold(parts[-1]))[1]
        if suffix not in TEXT_SUFFIXES and suffix not in PDF_SUFFIXES:
            raise FsError(
                "fs.unsupported_type", "No text extraction exists for this kind of file", path=shown
            )
        if suffix in PDF_SUFFIXES and not _memory_budget_enforced():
            raise FsError(
                "fs.unsupported_type",
                "The PDF memory budget cannot be enforced on this operating system",
                path=shown,
            )
        data = self.read_bytes(shown, max_bytes=budget.max_bytes)
        if suffix in PDF_SUFFIXES:
            pages = _pdf_pages(data, budget, shown)
            self._check_revoked(shown)
            return ExtractedText(shown, "pdf", "\n\n".join(pages), len(data), tuple(pages))
        text = _decode(data, shown)
        if len(text) > budget.max_chars:
            raise FsError(
                "fs.too_complex",
                f"The text has more than {budget.max_chars} characters",
                path=shown,
                reason="chars",
            )
        self._check_revoked(shown)
        return ExtractedText(shown, "text", text, len(data))


def _memory_budget_enforced() -> bool:
    """Whether :func:`orkexa.retrieval.pdf_pages_isolated` enforces ``max_memory_mb`` on this system (on
    macOS only where the memory of a process can be measured)."""
    if sys.platform == "darwin":
        from .._pdf_launch import _memory_meter

        meter = _memory_meter()
        return meter is None or meter(os.getpid()) is not None
    return sys.platform.startswith("linux") or sys.platform == "win32"


def _nfc(text: str) -> str:
    import unicodedata

    return unicodedata.normalize("NFC", text)


def _decode(data: bytes, shown: str) -> str:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", errors="replace")
    if b"\x00" in data[:8192]:
        raise FsError("fs.unsupported_type", "The file is binary, not text", path=shown)
    return data.decode("utf-8-sig", errors="replace")


def _pdf_pages(data: bytes, budget: TextBudget, shown: str) -> list[str]:
    from ..retrieval import RetrievalError, pdf_pages_isolated

    if not data.lstrip()[:5] == b"%PDF-":
        raise FsError("fs.unreadable", "The file is not a PDF", path=shown)
    try:
        return pdf_pages_isolated(
            data,
            timeout=budget.timeout,
            max_memory_mb=budget.max_memory_mb,
            max_pages=budget.max_pages,
            max_chars=budget.max_chars,
        )
    except RetrievalError as exc:
        if exc.reason == "unenforced":
            raise FsError("fs.unsupported_type", str(exc), path=shown) from exc
        raise FsError("fs.too_complex", str(exc), path=shown, reason=exc.reason) from exc
    except ImportError:
        raise FsError(
            "fs.unsupported_type", 'Reading PDFs needs pypdf: pip install "orkexa[pdf]"', path=shown
        ) from None
    except ValueError:
        raise FsError("fs.unreadable", "The PDF cannot be read", path=shown) from None


# ---- the folder browser ------------------------------------------------------------------------------------

#: Most sub-folders :func:`browse_folder` answers for one folder; beyond, the answer is ``truncated``.
DEFAULT_BROWSE_FOLDERS = 500
#: Windows' hidden and system attributes (``FILE_ATTRIBUTE_HIDDEN``, ``FILE_ATTRIBUTE_SYSTEM``).
_HIDDEN_ATTRIBUTES = 0x2 | 0x4
#: The home folders of the administrator account (Linux, macOS), which are system folders themselves.
_ROOT_HOMES = ("/root", "/var/root")


@dataclass(frozen=True)
class Folder:
    """One sub-folder of a :class:`FolderListing`.

    ``name`` is the name as stored (show it with :func:`display_name`); ``path`` the canonical absolute path
    to browse next. ``link`` marks a symbolic link or junction: ``path`` is then its target, which passed the
    same checks as every other folder (:func:`browse_problem`).
    """

    name: str
    path: str
    link: bool = False


@dataclass(frozen=True)
class FolderListing:
    """The answer of :func:`browse_folder`: the canonical folder, its parent (``None`` at the root of a file
    system or drive), its sub-folders sorted by name, and whether the listing stopped early
    (``truncated``)."""

    path: str
    parent: str | None
    folders: tuple[Folder, ...]
    truncated: bool = False


class _Browser:
    """The checks of :func:`browse_problem`, with the folders' identities looked up once per listing."""

    def __init__(
        self,
        *,
        protected: Iterable[str | os.PathLike[str]],
        home: str | os.PathLike[str] | None,
        blocked: Iterable[str],
    ) -> None:
        self.home = _home(home)
        found = _backend.stat_path(self.home) if self.home else None
        self.home_id = _identity(found) if found is not None else None
        self.home_open = _home_open(self.home)
        parents = home_parents()
        self.outer = [
            parent
            for parent in parents
            if not any(inside(parent, other) and path_parts(parent) != path_parts(other) for other in parents)
        ]
        self.outer_ids = _identities(self.outer)
        self.system_ids = _identities(system_folders())
        self.allowed_ids = _identities(allowed_folders())
        self.blocked = [pattern for pattern in patterns(blocked) if pattern.floating]
        self.protected = [os.fspath(folder) for folder in protected]
        self.protected_ids = _identities(self.protected)

    def chain(self, path: str, info: Info | None) -> list[tuple[int, int] | None]:
        """The identities of ``path`` and every folder above it, nearest first."""
        found = [_backend.stat_path(folder) for folder in _ancestors(path)]
        ids = [_identity(item) if item is not None else None for item in found]
        if info is not None:
            ids[0] = _identity(info)
        return ids

    def problem(
        self, path: str, ids: list[tuple[int, int] | None], spellings: Iterable[str] = ()
    ) -> str | None:
        parts = path_parts(path)
        every = [parts, *(path_parts(item) for item in spellings)]
        if any(pattern.covers(spelling) for pattern in self.blocked for spelling in every):
            return "A blocked folder (credentials, keys, keychains) or a folder inside one is not shown"
        for folder in self.protected:
            if inside(path, folder) or inside(path, os.path.realpath(folder)):
                return "A protected folder (or a folder inside one) is not shown"
        if self.protected_ids & {identity for identity in ids if identity is not None}:
            return "A protected folder (or a folder inside one) is not shown"
        if self.home_open and self.home and (inside(path, self.home) or self.home_id in ids):
            return None
        own = path_parts(self.home) if self.home else None
        for parent in self.outer:
            above = path_parts(parent)
            if len(parts) > len(above) and parts[: len(above)] == above and parts[: len(above) + 1] != own:
                return "Another user's home folder (or a folder inside one) is not shown"
        for index in range(len(ids) - 1):
            if ids[index + 1] in self.outer_ids and ids[index] != self.home_id:
                return "Another user's home folder (or a folder inside one) is not shown"
        for identity in ids:
            if identity in self.allowed_ids:
                return None
            if identity in self.system_ids:
                return "A system folder (or a folder inside one) is not shown"
        if any(inside(path, folder) for folder in system_folders()) and not any(
            inside(path, folder) for folder in allowed_folders()
        ):
            return "A system folder (or a folder inside one) is not shown"
        return None

    def walked(self, given: str, end: tuple[int, int] | None) -> list[str] | None:
        """The spellings of ``given`` as its links are resolved, or ``None`` when they no longer end at the
        folder that was resolved (a link changed in between)."""
        walked = _link_spellings(given)
        found = _backend.stat_path(walked[-1])
        if end is not None and (found is None or _identity(found) != end):
            return None
        return walked


def browse_problem(
    path: str,
    info: Info | None = None,
    *,
    protected: Iterable[str | os.PathLike[str]] = (),
    home: str | os.PathLike[str] | None = None,
    blocked: Iterable[str] = DEFAULT_BLOCKED,
    given: str | os.PathLike[str] | None = None,
) -> str | None:
    """Why the folder browser (:func:`browse_folder`) neither lists nor opens the canonical folder ``path``
    (a message), or ``None``.

    Unlike :func:`root_problem` this allows the folders a person passes on the way to their own: the root of
    a file system or drive, the folders that hold the home folders, and the user's own home folder (``home``,
    default: the home folder of this process's user) with everything below it. Refused:

    * a system folder (:func:`system_folders`) or a folder inside one, except the temporary and
      removable-media folders of :func:`allowed_folders` and the user's own home folder (also ``/root``),
      unless that home folder is a file-system root or holds a system folder;
    * another user's home folder (a folder directly inside ``/home``, ``/Users``, ``C:\\Users`` ... that is
      not ``home``) or a folder inside one;
    * a folder that the ``blocked`` patterns starting with ``**/`` cover by its canonical path or, with
      ``given``, by the path as given and every path its links lead through;
    * a ``protected`` folder or a folder inside one.

    Folders are compared by name (NFC plus case-folded) and by identity, as in :func:`root_problem`.
    """
    browser = _Browser(protected=protected, home=home, blocked=blocked)
    ids = browser.chain(path, info)
    spellings: list[str] = []
    if given:
        walked = browser.walked(os.fspath(given), ids[0])
        if walked is None:
            return "The links on the way to the folder changed while it was checked"
        spellings = walked
    return browser.problem(path, ids, spellings)


def browse_folder(
    path: str | os.PathLike[str],
    *,
    protected: Iterable[str | os.PathLike[str]] = (),
    home: str | os.PathLike[str] | None = None,
    blocked: Iterable[str] = DEFAULT_BLOCKED,
    max_folders: int = DEFAULT_BROWSE_FOLDERS,
    max_entries: int = DEFAULT_MAX_ENTRIES,
    max_seconds: float = DEFAULT_LIST_SECONDS,
) -> FolderListing:
    """The sub-folders of an absolute folder, for a folder picker: folder names and paths, never files.

    ``path`` is canonicalized (links resolved; on Windows the final path) and refused with
    ``fs.not_browsable`` when :func:`browse_problem` names a reason, with ``fs.invalid_path`` or
    ``fs.network_path`` as a root would be, and with ``fs.not_found``, ``fs.not_a_directory`` or
    ``fs.permission``. The folder is then opened once and must still be the folder that was checked
    (identity and final path, else ``fs.root_changed``); its entries are read through that open folder.

    Left out of the answer: files and every other entry that is not a folder; names starting with ``.``
    (and on Windows names starting with ``$`` and folders with the hidden or system attribute); names that
    break the rules of :func:`split_relative`; and every folder that :func:`browse_problem` refuses. A
    symbolic link or junction is resolved and shown (``link``) only when its target is a folder that passes
    the same checks; its ``path`` is the target. At most ``max_entries`` entries are read within
    ``max_seconds`` and at most ``max_folders`` folders answered; beyond any of them the answer is
    ``truncated``.

    The answer is a snapshot: a folder may change after it was listed, so the folder to browse next is
    checked again when it is opened.
    """
    if max_folders < 1 or max_entries < 1 or max_seconds <= 0:
        raise ValueError("max_folders, max_entries and max_seconds must be positive")
    text = root_syntax(os.fspath(path))
    canonical, info = _backend.canonical(text)
    browser = _Browser(protected=protected, home=home, blocked=blocked)
    ids = browser.chain(canonical, info)
    walked = browser.walked(text, ids[0])
    if walked is None:
        raise FsError("fs.root_changed", "The links on the way to the folder changed while it was checked")
    problem = browser.problem(canonical, ids, walked)
    if problem is not None:
        raise FsError("fs.not_browsable", problem)
    folders: list[Folder] = []
    truncated = False
    deadline = time.monotonic() + max_seconds
    node, close = _open_folder(canonical, _identity(info))
    try:
        for read, (name, kind, _size, _links) in enumerate(_backend.listing(node, canonical, ()), 1):
            if read > max_entries or time.monotonic() > deadline:
                truncated = True
                break
            if kind not in ("dir", "link") or _hidden_name(name):
                continue
            try:
                split_relative(name)
            except FsError:
                continue
            child = os.path.join(canonical, name)
            if kind == "dir":
                found = _child_identity(node, name, child)
                if found is not None and browser.problem(child, [found, *ids]) is None:
                    folders.append(Folder(name=name, path=child))
                continue
            target = _link_target(browser, child)
            if target is not None:
                folders.append(Folder(name=name, path=target, link=True))
    finally:
        close(node)
    folders.sort(key=lambda folder: (fold(folder.name), folder.name))
    if len(folders) > max_folders:
        folders, truncated = folders[:max_folders], True
    parent = os.path.dirname(canonical)
    return FolderListing(
        path=canonical,
        parent=parent if parent and parent != canonical else None,
        folders=tuple(folders),
        truncated=truncated,
    )


def _home_open(home: str | None) -> bool:
    """Whether everything below the user's own home folder may be browsed, even when it lies in or is a
    system folder (a service account in /var/lib, the administrator's /root): not when the home folder is a
    file-system root or holds a system folder (an account whose home is ``/``, ``/var`` or ``/usr/sbin``),
    which would open the system."""
    if not home or len(path_parts(home)) <= (1 if os.name == "nt" else 0):
        return False
    roots = [path_parts(os.path.realpath(item)) for item in _ROOT_HOMES] if os.name != "nt" else []
    if path_parts(home) in roots:
        return True
    return not any(inside(folder, home) for folder in system_folders())


def _hidden_name(name: str) -> bool:
    return name.startswith(".") or (os.name == "nt" and name.startswith("$"))


def _open_folder(path: str, identity: tuple[int, int]) -> tuple[int, Callable[[int], None]]:
    """The folder ``path`` opened (a descriptor or handle) after it was checked, and how to close it."""
    if sys.platform == "win32":
        handle, _ = _backend._open_root(path, identity, True)
        return handle, _backend._close
    fd, _ = _backend._open_root(path, identity)
    return fd, os.close


def _child_identity(node: int, name: str, child: str) -> tuple[int, int] | None:
    """The identity of the folder ``name`` in the open folder ``node``; ``None`` when it is gone, hidden or
    cannot be looked at. Never follows a link."""
    if sys.platform == "win32":
        try:
            if getattr(os.lstat(child), "st_file_attributes", 0) & _HIDDEN_ATTRIBUTES:
                return None
        except OSError:
            return None
        found = _backend.stat_path(child)
        return _identity(found) if found is not None else None
    try:
        st = os.stat(name, dir_fd=node, follow_symlinks=False)
    except OSError:
        return None
    return (st.st_dev, st.st_ino)


def _link_target(browser: _Browser, link: str) -> str | None:
    """The canonical target of the link or junction ``link`` when it is a folder the browser shows."""
    try:
        if sys.platform == "win32" and getattr(os.lstat(link), "st_file_attributes", 0) & _HIDDEN_ATTRIBUTES:
            return None
        target, info = _backend.canonical(root_syntax(link))
    except (FsError, OSError, ValueError):
        return None
    ids = browser.chain(target, info)
    walked = browser.walked(link, ids[0])
    if walked is None or browser.problem(target, ids, walked) is not None:
        return None
    return target
