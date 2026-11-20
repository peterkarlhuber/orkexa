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
  so no spelling reaches them. A system folder, a drive or file-system root, the home folder itself, a
  protected folder and any folder inside one cannot be a root at all (``fs.protected_root``).

Every refusal is an :class:`FsError` with a stable ``code`` (:data:`FS_ERROR_CODES`). This module uses only
the standard library (``ctypes`` on Windows and for ``openat2``); PDF text extraction needs ``orkexa[pdf]``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import threading
from typing import Any, Literal

from ._names import (
    DEFAULT_BLOCKED,
    FS_ERROR_CODES,
    MAX_NAME_BYTES,
    MAX_PATH_BYTES,
    FsError,
    allowed_folders,
    display_name,
    fold,
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
    "DEFAULT_MAX_ENTRIES",
    "DEFAULT_MAX_FILE_BYTES",
    "FS_ERROR_CODES",
    "MAX_NAME_BYTES",
    "MAX_PATH_BYTES",
    "PDF_SUFFIXES",
    "TEXT_SUFFIXES",
    "Entry",
    "ExtractedText",
    "FsError",
    "SafeRoot",
    "TextBudget",
    "display_name",
    "is_short_name",
    "root_problem",
    "split_relative",
    "system_folders",
]

#: Largest file :meth:`SafeRoot.read_bytes` reads unless the root or the call says otherwise (50 MiB).
DEFAULT_MAX_FILE_BYTES = 50 * 1024 * 1024
#: Most entries :meth:`SafeRoot.list` returns for one folder.
DEFAULT_MAX_ENTRIES = 10_000
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
    separate process that is killed at the time budget (:func:`orkexa.retrieval.pdf_pages_isolated`; the
    memory limit is enforced on Linux only).
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


def root_problem(
    path: str, info: Info | None = None, *, protected: Iterable[str | os.PathLike[str]] = ()
) -> str | None:
    """Why the canonical folder ``path`` cannot be a root (a message), or ``None``.

    Refused: a drive or file-system root, the home folder itself, a system folder (:func:`system_folders`) or
        a
    folder inside one (except the temporary and removable-media folders of
    :data:`~orkexa.fs._names.POSIX_ALLOWED_FOLDERS`), and a ``protected`` folder or a folder inside one.
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
    try:
        home = os.path.realpath(Path.home())
    except (RuntimeError, KeyError, OSError):
        home = None
    if home:
        home_info = _backend.stat_path(home)
        if path_parts(home) == parts or (home_info is not None and _identity(home_info) == ids[0]):
            return "The home folder itself cannot be a root; choose a folder inside it"
    for folder in protected:
        text = os.fspath(folder)
        found = _backend.stat_path(text)
        real = os.path.realpath(text) if found is not None else text
        if inside(path, text) or inside(path, real) or (found is not None and _identity(found) in ids):
            return "A protected folder (or a folder inside one) cannot be a root"
    allowed = {_identity(item) for item in map(_backend.stat_path, allowed_folders()) if item is not None}
    system = {_identity(item) for item in map(_backend.stat_path, system_folders()) if item is not None}
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
        blocked: Patterns that are refused as ``fs.blocked``; :data:`DEFAULT_BLOCKED` unless replaced.
        protected: Folders (the product's data folder ...) that can neither be the root nor be reached below
            it.
        max_file_bytes: The size limit of a file (``fs.too_large``).
        max_entries: The most entries one listing returns (``fs.too_many_entries`` beyond).
        allow_hard_links: Read files with more than one hard link (default: refused with ``fs.hard_link``,
            since a hard link can be the very file of another folder; A-15).
        allow_cloud_files: Read OneDrive cloud placeholders on Windows, which may download them (default:
            refused with ``fs.cloud_file``).
        writable: Allow :meth:`create_file` and :meth:`rename` (default: read only, ``fs.read_only``).
        expect_identity: The ``(device, inode)`` recorded when the folder was granted (:attr:`identity`); a
            different folder at the same path raises ``fs.root_changed``.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        include: Iterable[str] = (),
        exclude: Iterable[str] = (),
        blocked: Iterable[str] = DEFAULT_BLOCKED,
        protected: Iterable[str | os.PathLike[str]] = (),
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        allow_hard_links: bool = False,
        allow_cloud_files: bool = False,
        expect_identity: tuple[int, int] | None = None,
        writable: bool = False,
    ) -> None:
        if max_file_bytes < 0 or max_entries < 1:
            raise ValueError("max_file_bytes must not be negative and max_entries must be positive")
        text = root_syntax(os.fspath(path))
        protected_paths = [os.fspath(folder) for folder in protected]
        try:
            canonical, info = _backend.canonical(text)
        except FsError as error:
            # A folder this user may not open (such as /root) is still reported as not grantable.
            problem = root_problem(os.path.realpath(text), protected=protected_paths)
            if error.code == "fs.permission" and problem is not None:
                raise FsError("fs.protected_root", problem) from None
            raise
        problem = root_problem(canonical, info, protected=protected_paths)
        if problem is not None:
            raise FsError("fs.protected_root", problem)
        if expect_identity is not None and tuple(expect_identity) != _identity(info):
            raise FsError("fs.root_changed", "The root folder is not the folder that was granted")
        self._path = canonical
        self._identity = _identity(info)
        self._include = patterns(include)
        self._exclude = patterns(exclude)
        self._blocked = patterns(blocked)
        self._protected: list[tuple[str, tuple[int, int]]] = []
        for folder in protected_paths:
            try:
                real, found = _backend.canonical(root_syntax(folder))
            except FsError:
                continue  # a protected folder that does not exist cannot be reached either
            self._protected.append((real, _identity(found)))
        self._protected_ids = {identity for _, identity in self._protected}
        self.max_file_bytes = max_file_bytes
        self.max_entries = max_entries
        self.allow_hard_links = allow_hard_links
        self.allow_cloud_files = allow_cloud_files
        self.writable = writable
        self._revoked = threading.Event()

    def __repr__(self) -> str:
        return f"SafeRoot({self._path!r})"

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
        """Refuse every later call (``fs.revoked``); a read in progress stops at its next chunk (1 MiB)."""
        self._revoked.set()

    @property
    def revoked(self) -> bool:
        return self._revoked.is_set()

    # ---- checks without I/O --------------------------------------------------------------------------------

    def check(self, path: str | bytes, *, kind: Literal["file", "dir"] = "file") -> tuple[str, ...]:
        """The components of ``path`` after the name rules and the patterns, without file-system access."""
        if self._revoked.is_set():
            raise FsError("fs.revoked", "The root was revoked")
        parts = split_relative(path, allow_empty=kind == "dir")
        shown = "/".join(parts)
        folded = tuple(fold(part) for part in parts)
        if any(pattern.covers(folded) for pattern in self._blocked):
            raise FsError("fs.blocked", "This path is blocked", path=shown)
        if any(pattern.covers(folded) for pattern in self._exclude):
            raise FsError("fs.excluded", "This path is excluded", path=shown)
        if kind == "file" and self._include and not any(pattern.matches(folded) for pattern in self._include):
            raise FsError("fs.excluded", "This file is not included", path=shown)
        return parts

    def _on_node(self, shown: str) -> object:
        def check(index: int, info: Info, names: tuple[str, ...]) -> None:
            del index, names
            if _identity(info) in self._protected_ids:
                raise FsError("fs.blocked", "This path is inside a protected folder", path=shown)

        return check

    # ---- operations ----------------------------------------------------------------------------------------

    def stat(self, path: str | bytes) -> Entry:
        """Kind and size of one file or folder (links are refused, not reported)."""
        parts = self.check(path, kind="dir")
        shown = "/".join(parts)
        with _backend.walk(
            self._path,
            self._identity,
            parts,
            final="any",
            on_node=self._on_node(shown),  # type: ignore[arg-type]
            shown=shown,
            allow_cloud=self.allow_cloud_files,
        ) as (_, info):
            kind: Kind = "dir" if info.kind == "dir" else "file"
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
        since files below them may match. Links are listed as ``link`` and never followed.
        """
        parts = self.check(path, kind="dir")
        shown = "/".join(parts)
        found: list[Entry] = []
        with _backend.walk(
            self._path,
            self._identity,
            parts,
            final="dir",
            on_node=self._on_node(shown),  # type: ignore[arg-type]
            shown=shown,
            allow_cloud=self.allow_cloud_files,
        ) as (node, _):
            for name, kind, size in _backend.listing(node, self._path, parts):
                child = "/".join([*parts, name])
                try:
                    self.check(child, kind="dir" if kind == "dir" else "file")
                except FsError:
                    continue
                if kind == "dir" and any(
                    inside(os.path.join(self._path, *parts, name), real) for real, _ in self._protected
                ):
                    continue
                if len(found) >= self.max_entries:
                    raise FsError(
                        "fs.too_many_entries",
                        f"The folder has more than {self.max_entries} entries",
                        path=shown,
                    )
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
        with _backend.walk(
            self._path,
            self._identity,
            parts,
            final="file",
            on_node=self._on_node(shown),  # type: ignore[arg-type]
            shown=shown,
            allow_cloud=self.allow_cloud_files,
        ) as (node, info):
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

    def _walk_folder(self, parts: tuple[str, ...], shown: str) -> Any:
        return _backend.walk(
            self._path,
            self._identity,
            parts,
            final="dir",
            on_node=self._on_node(shown),  # type: ignore[arg-type]
            shown=shown,
            allow_cloud=self.allow_cloud_files,
        )

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
        with self._walk_folder(parts[:-1], shown) as (node, _):
            _backend.create(node, self._path, parts[:-1], parts[-1], bytes(data), shown)

    def rename(self, source: str | bytes, target: str | bytes) -> None:
        """Rename a file or folder below the root; an existing ``target`` is never replaced.

        Links are not renamed (``fs.link``), nor files with several hard links unless ``allow_hard_links``,
        nor protected folders. Linux: ``renameat2(RENAME_NOREPLACE)``; macOS: ``renameatx_np(RENAME_EXCL)``,
        both on the descriptors of the checked folders; Windows: ``MoveFileExW`` without ``REPLACE_EXISTING``
        below the held folder handles.
        """
        src = self.check(source, kind="dir")
        dst = self.check(target, kind="dir")
        shown = "/".join(src)
        self._writable(shown)
        if not src or not dst:
            raise FsError("fs.invalid_path", "The root itself cannot be renamed", path=shown)
        kind = self.stat(source).kind  # refuses links on the way and protected folders, checks the kind
        if kind == "file":
            self.check(target, kind="file")
        with self._walk_folder(src[:-1], shown) as (src_node, _):
            entry_kind, links = _backend.entry(src_node, src[-1], shown, root=self._path, parts=src[:-1])
            if entry_kind == "link":
                raise FsError("fs.link", "A link is not renamed", path=shown)
            if entry_kind == "file" and links > 1 and not self.allow_hard_links:
                raise FsError("fs.hard_link", "The file has more than one hard link", path=shown)
            with self._walk_folder(dst[:-1], "/".join(dst)) as (dst_node, _):
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
        ``budget.timeout``, with ``budget.max_pages``, ``max_chars`` and (Linux) ``max_memory_mb``. Exceeding
        a budget raises ``fs.too_complex`` with ``reason`` ``pages``, ``chars``, ``timeout``, ``memory``,
        ``limit`` or ``crash``; an unreadable PDF ``fs.unreadable``. Other suffixes raise
        ``fs.unsupported_type``.
        """
        budget = budget or TextBudget()
        parts = self.check(path, kind="file")
        shown = "/".join(parts)
        suffix = os.path.splitext(fold(parts[-1]))[1]
        if suffix not in TEXT_SUFFIXES and suffix not in PDF_SUFFIXES:
            raise FsError(
                "fs.unsupported_type", "No text extraction exists for this kind of file", path=shown
            )
        data = self.read_bytes(shown, max_bytes=budget.max_bytes)
        if suffix in PDF_SUFFIXES:
            pages = _pdf_pages(data, budget, shown)
            return ExtractedText(shown, "pdf", "\n\n".join(pages), len(data), tuple(pages))
        text = _decode(data, shown)
        if len(text) > budget.max_chars:
            raise FsError(
                "fs.too_complex",
                f"The text has more than {budget.max_chars} characters",
                path=shown,
                reason="chars",
            )
        return ExtractedText(shown, "text", text, len(data))


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
        raise FsError("fs.too_complex", str(exc), path=shown, reason=exc.reason) from exc
    except ImportError:
        raise FsError(
            "fs.unsupported_type", 'Reading PDFs needs pypdf: pip install "orkexa[pdf]"', path=shown
        ) from None
    except ValueError:
        raise FsError("fs.unreadable", "The PDF cannot be read", path=shown) from None
