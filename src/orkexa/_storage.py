"""Owner-private files and directories for the data Orkexa persists.

Histories, memory, file stores, search indexes and registries can hold personal data,
so Orkexa creates them readable only by the user that runs it, independent of the
process umask (which it never changes):

* new directories: ``0700``; existing directories are left as they are;
* files written by replacement (histories, registry records): ``0600`` on every write;
* new SQLite databases: ``0600``; SQLite gives its journal and WAL files the same mode.

SQLite databases are opened by :func:`connect_private_database`: a database path (or its journal, WAL
or shared-memory file) that is a symbolic link is refused (``storage.symlink``), and a damaged file is
reported as ``storage.corrupt`` instead of a raw ``sqlite3.DatabaseError``
(see :class:`~orkexa.retrieval.RetrievalError`).

On Windows the mode bits only control the read-only flag; access is governed by the
ACLs the new file inherits from its directory (by default private below the user
profile).
"""

from __future__ import annotations

from collections.abc import Iterator
import contextlib
import logging
import os
from pathlib import Path
import sqlite3
import tempfile

PRIVATE_FILE_MODE = 0o600
PRIVATE_DIR_MODE = 0o700

logger = logging.getLogger("orkexa.storage")

# The primary result codes of a damaged database (SQLITE_CORRUPT, SQLITE_NOTADB); extended codes such as
# SQLITE_CORRUPT_VTAB keep them in their low byte.
_CORRUPT_CODES = (11, 26)
_SQLITE_SIDE_FILES = ("-journal", "-wal", "-shm")


def make_private_dirs(directory: str | os.PathLike[str]) -> None:
    """Create ``directory`` and its missing parents with mode ``0700``.

    Existing directories keep their permissions; the umask can only make a new
    directory stricter, never looser.
    """
    missing: list[Path] = []
    current = Path(directory)
    while not current.exists():
        missing.append(current)
        if current.parent == current:
            break
        current = current.parent
    for path in reversed(missing):
        with contextlib.suppress(FileExistsError):
            path.mkdir(mode=PRIVATE_DIR_MODE)


def write_private_text(path: str | os.PathLike[str], text: str) -> None:
    """Replace ``path`` atomically with ``text`` in a new owner-only (``0600``) file.

    Readers see the old or the new content, never a torn file. A symbolic link at
    ``path`` is replaced, not followed. The temporary file is removed on failure.
    """
    target = Path(path)
    descriptor, temporary = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def create_private_file(path: str | os.PathLike[str]) -> None:
    """Create an empty owner-only file if ``path`` does not exist yet.

    Used before SQLite opens a database, so that the database (and the journal and
    WAL files SQLite derives from it) is private from the start. An existing file
    keeps its permissions. ``":memory:"`` is ignored.
    """
    if os.fspath(path) == ":memory:":
        return
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, PRIVATE_FILE_MODE)
    except FileExistsError:
        return
    os.close(descriptor)


def check_database_path(database: str | os.PathLike[str]) -> None:
    """Refuse a database path that is a symbolic link, or whose journal, WAL or shared-memory file is one.

    Raises :class:`~orkexa.retrieval.RetrievalError` (``storage.symlink``). A link would let whoever can write
    the directory redirect the database (and its private ``0600`` files) to another file. The message names no
    path; the path is logged at debug level. ``":memory:"`` is accepted.
    """
    path = os.fspath(database)
    if path == ":memory:":
        return
    for candidate in (path, *(path + suffix for suffix in _SQLITE_SIDE_FILES)):
        if os.path.islink(candidate):
            from .retrieval import RetrievalError

            logger.debug("refusing the symbolic link %s", candidate)
            raise RetrievalError(
                "storage.symlink",
                "The database path or one of its journal files is a symbolic link; use the real path",
            )


def is_corruption(error: BaseException) -> bool:
    """Whether ``error`` reports a damaged database file (not a lock, a constraint or a bad statement)."""
    if not isinstance(error, sqlite3.DatabaseError):
        return False
    code = getattr(error, "sqlite_errorcode", None)
    if code is not None:
        return bool(code & 0xFF in _CORRUPT_CODES)
    text = str(error).lower()
    return "malformed" in text or "not a database" in text


@contextlib.contextmanager
def storage_errors() -> Iterator[None]:
    """Report a damaged database as :class:`~orkexa.retrieval.RetrievalError` (``storage.corrupt``).

    The original ``sqlite3.DatabaseError`` stays attached as the cause. Other errors pass unchanged.
    """
    try:
        yield
    except sqlite3.DatabaseError as exc:
        if not is_corruption(exc):
            raise
        from .retrieval import RetrievalError

        raise RetrievalError(
            "storage.corrupt", f"The database file is damaged or not a SQLite database ({exc})"
        ) from exc


def connect_private_database(database: str | os.PathLike[str]) -> sqlite3.Connection:
    """Open a SQLite database the way every Orkexa SQLite class does.

    Refuses symbolic links (:func:`check_database_path`), creates a new file with mode ``0600``
    (:func:`create_private_file`), enables ``secure_delete`` and, for files, WAL mode. The connection runs in
    autocommit mode (``isolation_level=None``) and may be used from several threads under the caller's lock.
    A damaged file raises ``storage.corrupt``.
    """
    check_database_path(database)
    create_private_file(database)  # owner-only (0600) from the start; journals inherit it
    connection = sqlite3.connect(database, isolation_level=None, check_same_thread=False, timeout=30)
    try:
        with storage_errors():
            # Deleted rows are overwritten instead of left in free pages of the file.
            connection.execute("PRAGMA secure_delete=ON")
            if os.fspath(database) != ":memory:":
                connection.execute("PRAGMA journal_mode=WAL")
    except BaseException:
        connection.close()
        raise
    return connection
