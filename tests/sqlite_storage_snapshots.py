"""Before/after evidence of Orkexa's SQLite connections (TABRION MIG-ORK-3).

MIG-ORK-3 moves the opening of Orkexa's SQLite databases (path rules, file modes, pragmas) and the transaction
blocks of :class:`~orkexa.SQLiteIndex`, :class:`~orkexa.SQLiteFileStore` and :class:`~orkexa.SQLiteMemory` to
``tabrion.sqlite``. This module records what a caller sees of them:

- ``pragmas``: the settings of each backend's connection on a new file and on ``":memory:"``;
- ``files``: the modes of a new database and its ``-wal``/``-shm`` files (umask 022 and 002);
- ``existing``: whether an existing database with a given mode opens, and its mode afterwards;
- ``symlink``: the refusal of a linked database path, of each linked side file and of a dangling link;
- ``corrupt``: the answer for a file that is no database and for a truncated one;
- ``transactions``: rollback of a failing write block, and a write inside the index's read block;
- ``codes``: :data:`orkexa.retrieval.RETRIEVAL_ERROR_CODES`.

``tests/fixtures/sqlite_storage_before.json`` was written by the code *before* MIG-ORK-3::

    python tests/sqlite_storage_snapshots.py write   # only on the code before MIG-ORK-3

``tests/test_sqlite_storage_snapshots.py`` compares the current code with it; the intended differences are
listed there one by one.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
import json
import os
from pathlib import Path
import random
import shutil
import sqlite3
import stat
import sys
import tempfile
from typing import Any

BEFORE = Path(__file__).resolve().parent / "fixtures" / "sqlite_storage_before.json"
PRAGMAS = (
    "journal_mode",
    "synchronous",
    "foreign_keys",
    "busy_timeout",
    "secure_delete",
    "trusted_schema",
    "query_only",
)
BACKENDS = ("SQLiteIndex", "SQLiteFileStore", "SQLiteMemory")
POSIX = os.name == "posix"
SUFFIXES = ("", "-wal", "-shm")


def _backend(name: str) -> Any:
    import orkexa

    return getattr(orkexa, name)


def _use(name: str, path: Path | str) -> None:
    """Open the backend, write one entry and close it."""
    backend = _backend(name)(path)
    try:
        _write_one(name, backend)
    finally:
        backend.close()


def _write_one(name: str, backend: Any) -> None:
    if name == "SQLiteIndex":
        backend.add("some text to index", source="a.md")
    elif name == "SQLiteFileStore":
        backend.add(b"content", name="a.txt", media_type="text/plain")
    else:
        backend.create("/memories/a.md", "note")


@contextmanager
def _umask(mask: int) -> Iterator[None]:
    previous = os.umask(mask)
    try:
        yield
    finally:
        os.umask(previous)


def _mode(path: Path) -> str | None:
    try:
        return oct(stat.S_IMODE(os.lstat(path).st_mode))
    except FileNotFoundError:
        return None


def _outcome(call: Callable[[], Any], directory: Path) -> dict[str, Any]:
    from orkexa.retrieval import RetrievalError

    try:
        value = call()
    except RetrievalError as exc:
        cause = exc.__cause__
        return {
            "code": exc.code,
            "reason": exc.reason,
            "message": str(exc).replace(str(directory), "<dir>"),
            "cause": type(cause).__name__ if cause is not None else None,
            "path_in_message": str(directory) in str(exc),
        }
    except Exception as exc:
        return {"raised": type(exc).__name__, "message": str(exc).replace(str(directory), "<dir>")}
    return {"ok": value}


def _open_and_close(name: str, path: Path | str) -> str:
    _backend(name)(path).close()
    return "opened"


def pragmas(directory: Path) -> dict[str, Any]:
    found: dict[str, Any] = {}
    for name in BACKENDS:
        for label, path in (("file", directory / f"pragmas-{name}.db"), ("memory", ":memory:")):
            backend = _backend(name)(path)
            try:
                connection: sqlite3.Connection = backend._connection
                values: dict[str, Any] = {
                    pragma: connection.execute(f"PRAGMA {pragma}").fetchone()[0] for pragma in PRAGMAS
                }
                values["isolation_level"] = connection.isolation_level
                values["in_transaction"] = connection.in_transaction
                found[f"{name}:{label}"] = values
            finally:
                backend.close()
    return found


def files(directory: Path) -> dict[str, Any]:
    found: dict[str, Any] = {}
    if not POSIX:
        return found
    for mask in (0o022, 0o002):
        with _umask(mask):
            for name in BACKENDS:
                base = directory / f"umask{mask:03o}-{name}"
                base.mkdir()
                backend = _backend(name)(base / "data.db")
                try:
                    _write_one(name, backend)
                    modes = {suffix or "db": _mode(base / f"data.db{suffix}") for suffix in SUFFIXES}
                finally:
                    backend.close()
                found[f"umask_{mask:03o}:{name}"] = modes
    return found


def existing(directory: Path) -> dict[str, Any]:
    found: dict[str, Any] = {}
    if not POSIX:
        return found
    for mode in (0o600, 0o640, 0o644, 0o660, 0o664, 0o666):
        for name in BACKENDS:
            path = directory / f"existing-{mode:03o}-{name}.db"
            sqlite3.connect(path).close()
            os.chmod(path, mode)
            outcome = _outcome(lambda name=name, path=path: _open_and_close(name, path), directory)
            found[f"{mode:03o}:{name}"] = {"outcome": outcome, "mode_after": _mode(path)}
    return found


def symlink(directory: Path) -> dict[str, Any]:
    found: dict[str, Any] = {}
    if not POSIX:
        return found
    for name in BACKENDS:
        for suffix in ("", "-wal", "-shm", "-journal"):
            base = directory / f"link-{name}{suffix or '-main'}"
            base.mkdir()
            real = base / "real.db"
            _use(name, real)
            link = base / "link.db"
            if suffix:
                shutil.copyfile(real, link)
                (base / f"elsewhere{suffix}").write_bytes(b"")
                os.symlink(base / f"elsewhere{suffix}", f"{link}{suffix}")
            else:
                os.symlink(real, link)
            found[f"{name}:{suffix or 'main'}"] = _outcome(
                lambda name=name, link=link: _open_and_close(name, link), directory
            )
        base = directory / f"dangling-{name}"
        base.mkdir()
        os.symlink(base / "target.db", base / "link.db")
        outcome = _outcome(lambda name=name, base=base: _open_and_close(name, base / "link.db"), directory)
        outcome["target_created"] = (base / "target.db").exists()
        found[f"{name}:dangling"] = outcome
    return found


def corrupt(directory: Path) -> dict[str, Any]:
    found: dict[str, Any] = {}
    generator = random.Random(7)
    noise = bytes(generator.randrange(256) for _ in range(8192))
    source = directory / "corrupt-source.db"
    _use("SQLiteIndex", source)
    data = source.read_bytes()
    for name in BACKENDS:
        for kind, content in (("random", noise), ("truncated", data[:50])):
            path = directory / f"corrupt-{kind}-{name}.db"
            path.write_bytes(content)
            opening = lambda name=name, path=path: _open_and_close(name, path)  # noqa: E731
            found[f"{name}:{kind}"] = _outcome(opening, directory)
    return found


def _state(connection: sqlite3.Connection, outcome: dict[str, Any]) -> dict[str, Any]:
    return {
        "outcome": outcome,
        "rows": connection.execute("SELECT count(*) FROM probe").fetchone()[0],
        "in_transaction": connection.in_transaction,
        "query_only": connection.execute("PRAGMA query_only").fetchone()[0],
    }


def _transaction_cases(name: str, backend: Any, directory: Path) -> dict[str, Any]:
    connection: sqlite3.Connection = backend._connection
    connection.execute("CREATE TABLE probe (x)")
    block = backend._write if name == "SQLiteIndex" else backend._transaction

    def failing() -> None:
        with block():
            connection.execute("INSERT INTO probe VALUES (1)")
            raise RuntimeError("block failed")

    def committed() -> int:
        with block():
            connection.execute("INSERT INTO probe VALUES (2)")
            inside = connection.in_transaction
        return int(inside)

    found = {
        "failing_write": _state(connection, _outcome(failing, directory)),
        "committed_write": _state(connection, _outcome(committed, directory)),
    }
    if name == "SQLiteIndex":

        def write_in_read() -> str:
            with backend._read():
                connection.execute("INSERT INTO probe VALUES (3)")
            return "written"

        found["write_in_read"] = _state(connection, _outcome(write_in_read, directory))
    return found


def transactions(directory: Path) -> dict[str, Any]:
    found: dict[str, Any] = {}
    for name in BACKENDS:
        backend = _backend(name)(directory / f"tx-{name}.db")
        try:
            for case, value in _transaction_cases(name, backend, directory).items():
                found[f"{name}:{case}"] = value
        finally:
            backend.close()
    return found


def codes() -> dict[str, str]:
    from orkexa.retrieval import RETRIEVAL_ERROR_CODES

    return dict(RETRIEVAL_ERROR_CODES)


def collect() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        sections: dict[str, Callable[[Path], Any]] = {
            "pragmas": pragmas,
            "files": files,
            "existing": existing,
            "symlink": symlink,
            "corrupt": corrupt,
            "transactions": transactions,
        }
        result: dict[str, Any] = {}
        for key, function in sections.items():
            part = directory / key
            part.mkdir()
            result[key] = function(part)
        result["codes"] = codes()
        return result


def dumps(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


if __name__ == "__main__":
    if sys.argv[1:] == ["write"]:
        BEFORE.write_text(dumps(collect()), encoding="utf-8")
    else:
        sys.stdout.write(dumps(collect()))
