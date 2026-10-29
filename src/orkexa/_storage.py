"""Owner-private files and directories for the data Orkexa persists.

Histories, memory, file stores, search indexes and registries can hold personal data,
so Orkexa creates them readable only by the user that runs it, independent of the
process umask (which it never changes):

* new directories: ``0700``; existing directories are left as they are;
* files written by replacement (histories, registry records): ``0600`` on every write;
* new SQLite databases: ``0600``; SQLite gives its journal and WAL files the same mode.

On Windows the mode bits only control the read-only flag; access is governed by the
ACLs the new file inherits from its directory (by default private below the user
profile).
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
import tempfile

PRIVATE_FILE_MODE = 0o600
PRIVATE_DIR_MODE = 0o700


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
