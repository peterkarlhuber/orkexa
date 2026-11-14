"""Owner-private files and directories for the data Orkexa persists.

Histories, memory, file stores, search indexes and registries can hold personal data,
so Orkexa creates them readable only by the user that runs it, independent of the
process umask (which it never changes):

* new directories: ``0700``; existing directories are left as they are;
* files written by replacement (histories, registry records): ``0600`` on every write;
* new SQLite databases: ``0600``; SQLite gives its journal and WAL files the same mode.

SQLite databases are opened by :func:`connect_private_database` through ``tabrion.sqlite`` (TABRION
MIG-ORK-3, profile :data:`SQLITE_PROFILE`): a database path (or its journal, WAL or shared-memory file) that
is a symbolic link, belongs to another user or is writable by group or others is refused
(``storage.symlink``), and a damaged file is reported as ``storage.corrupt`` instead of a raw
``sqlite3.DatabaseError`` (see :class:`~orkexa.retrieval.RetrievalError`). :func:`write_transaction` and
:func:`read_transaction` are the transaction blocks of the SQLite backends.

On Windows the mode bits only control the read-only flag; access is governed by the
ACLs the new file inherits from its directory (by default private below the user
profile).
"""

from __future__ import annotations

from collections.abc import Iterator
import contextlib
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import TYPE_CHECKING

from tabrion import sqlite as _tabrion_sqlite
from tabrion.errors import Kind, UnsafePath, classify
from tabrion.sqlite import JournalMode, SQLiteProfile, Synchronous

if TYPE_CHECKING:
    from .retrieval import RetrievalError

PRIVATE_FILE_MODE = 0o600
PRIVATE_DIR_MODE = 0o700

SQLITE_PROFILE = SQLiteProfile(
    journal_mode=JournalMode.WAL,
    synchronous=Synchronous.FULL,
    foreign_keys=True,
    busy_timeout=30.0,
    secure_delete=True,
)
"""The profile of every Orkexa SQLite database (TABRION MIG-ORK-3, plan doc 06 field 4).

WAL, ``synchronous=FULL``, foreign keys on, busy timeout 30 seconds, ``secure_delete`` on (deleted rows are
overwritten instead of left in free pages of the file); Tabrion's defaults for the rest: ``trusted_schema``
off, new files ``0600``, existing files of another user or writable by group or others refused."""


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


_UNSAFE_PATHS: dict[str, tuple[str, str]] = {
    "sqlite.symlink": (
        "symlink",
        "The database path or one of its journal files is a symbolic link; use the real path",
    ),
    "sqlite.unsafe_owner": (
        "other_owner",
        "The database file or one of its journal files belongs to another user; Orkexa opens only files of"
        " the user that runs it (change the owner with chown)",
    ),
    "sqlite.unsafe_mode": (
        "writable_by_others",
        "The database file or one of its journal files is writable by group or others; remove those"
        " permissions from the database file and its -wal, -shm and -journal files (chmod go-w)",
    ),
    "sqlite.unsafe_path": (
        "unsafe_path",
        "The database path is no regular file of this user or changed while it was opened; use the real"
        " path to a regular file",
    ),
}
"""``tabrion.sqlite.UnsafePath`` codes -> (``RetrievalError.reason``, message) of ``storage.symlink``."""


def _unsafe_path(error: UnsafePath) -> RetrievalError:
    """``storage.symlink`` for a refusal of ``tabrion.sqlite``; ``reason`` names the rule that refused."""
    from .retrieval import RetrievalError

    reason, message = _UNSAFE_PATHS.get(error.code, _UNSAFE_PATHS["sqlite.unsafe_path"])
    return RetrievalError("storage.symlink", message, reason=reason)


def is_corruption(error: BaseException) -> bool:
    """Whether ``error`` reports a damaged database file (not a lock, a constraint or a bad statement).

    A ``sqlite3.DatabaseError`` that ``tabrion.errors.classify`` reports as ``Kind.CORRUPT``
    (``SQLITE_CORRUPT``, ``SQLITE_NOTADB`` and their extended codes).
    """
    if not isinstance(error, sqlite3.DatabaseError):
        return False
    return classify(error).kind is Kind.CORRUPT


@contextlib.contextmanager
def storage_errors() -> Iterator[None]:
    """Report a damaged database as :class:`~orkexa.retrieval.RetrievalError` (``storage.corrupt``).

    The original ``sqlite3.DatabaseError`` stays attached as the cause. A path that ``tabrion.sqlite``
    refuses (``UnsafePath``) is ``storage.symlink``. Other errors pass unchanged.
    """
    try:
        yield
    except UnsafePath as exc:
        raise _unsafe_path(exc) from None
    except sqlite3.DatabaseError as exc:
        if not is_corruption(exc):
            raise
        from .retrieval import RetrievalError

        raise RetrievalError(
            "storage.corrupt", f"The database file is damaged or not a SQLite database ({exc})"
        ) from exc


def connect_private_database(database: str | os.PathLike[str]) -> sqlite3.Connection:
    """Open a SQLite database the way every Orkexa SQLite class does.

    ``tabrion.sqlite.connect`` under :data:`SQLITE_PROFILE`: refuses symbolic links and files of another
    user or writable by group or others (``storage.symlink``), creates a new file with mode ``0600``,
    enables ``secure_delete``, ``synchronous=FULL``, foreign keys, a busy timeout of 30 seconds and, for
    files, WAL mode. The connection runs in autocommit mode (``isolation_level=None``) and may be used from
    several threads under the caller's lock. A damaged file raises ``storage.corrupt``.
    """
    if not os.fspath(database):
        # tabrion would open an anonymous temporary database for ""; Orkexa keeps refusing it as before.
        raise FileNotFoundError("The database path is empty")
    with storage_errors():
        return _tabrion_sqlite.connect(database, profile=SQLITE_PROFILE, check_same_thread=False)


@contextlib.contextmanager
def write_transaction(connection: sqlite3.Connection) -> Iterator[None]:
    """A write transaction (``BEGIN IMMEDIATE``): committed on a normal exit, rolled back on any error.

    ``tabrion.sqlite.write_transaction``; a damaged file raises ``storage.corrupt``. Callers hold their lock.
    """
    with storage_errors(), _tabrion_sqlite.write_transaction(connection):
        yield


@contextlib.contextmanager
def read_transaction(connection: sqlite3.Connection) -> Iterator[None]:
    """A read transaction: one snapshot of the file (WAL mode), with ``PRAGMA query_only`` for the block.

    ``tabrion.sqlite.read_transaction``; a damaged file raises ``storage.corrupt``. Callers hold their lock.
    """
    with storage_errors(), _tabrion_sqlite.read_transaction(connection):
        yield
