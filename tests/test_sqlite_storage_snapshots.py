"""TABRION MIG-ORK-3: Orkexa's SQLite on ``tabrion.sqlite`` behaves as before, except where intended.

``tests/fixtures/sqlite_storage_before.json`` is the behavior of the code before MIG-ORK-3
(``tests/sqlite_storage_snapshots.py``). The current code must give the same answers, except the differences
listed in :data:`INTENDED`, each with its reason. Error codes, file modes of new databases, the refusal of
linked paths, the answers for damaged files and the transaction blocks are unchanged.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
import sys
from typing import Any

import pytest
from tabrion.errors import UnsafePath
from tabrion.sqlite import SQLiteProfile

from orkexa import SQLiteFileStore, SQLiteIndex, SQLiteMemory, _storage
from orkexa.retrieval import RetrievalError

from . import sqlite_storage_snapshots as snapshots

POSIX = os.name == "posix"
OPENERS = [SQLiteIndex, SQLiteFileStore, SQLiteMemory]

_WRITABLE = {
    "cause": None,
    "code": "storage.symlink",
    "message": "The database file or one of its journal files is writable by group or others; remove those"
    " permissions (chmod go-w)",
    "path_in_message": False,
    "reason": "writable_by_others",
}

#: The intended differences: (path in the snapshot, before, after, reason).
INTENDED: list[tuple[tuple[str, ...], Any, Any, str]] = []
for _backend in snapshots.BACKENDS:
    for _label in ("file", "memory"):
        INTENDED.append(
            (
                ("pragmas", f"{_backend}:{_label}", "foreign_keys"),
                0,
                1,
                "SQLITE_PROFILE (doc 06 MIG-ORK-3 field 4)",
            )
        )
        INTENDED.append(
            (
                ("pragmas", f"{_backend}:{_label}", "trusted_schema"),
                1,
                0,
                "Tabrion's default: no SQL functions with side effects in the schema",
            )
        )
    for _suffix in ("main", "-wal", "-shm", "-journal", "dangling"):
        INTENDED.append(
            (
                ("symlink", f"{_backend}:{_suffix}", "reason"),
                None,
                "symlink",
                "storage.symlink names the rule that refused the path (reason)",
            )
        )
    if POSIX:
        for _mode in ("660", "664", "666"):
            INTENDED.append(
                (
                    ("existing", f"{_mode}:{_backend}", "outcome"),
                    {"ok": "opened"},
                    _WRITABLE,
                    "E-NOD-SD: an existing database writable by group or others is refused (storage.symlink)",
                )
            )
INTENDED.append(
    (
        ("transactions", "SQLiteIndex:write_in_read", "outcome"),
        {"ok": "written"},
        {"message": "attempt to write a readonly database", "raised": "OperationalError"},
        "tabrion.sqlite.read_transaction sets PRAGMA query_only for the index's read block",
    )
)
INTENDED.append(
    (
        ("transactions", "SQLiteIndex:write_in_read", "rows"),
        2,
        1,
        "the write inside the read block is refused, so its row is missing",
    )
)


def _set(data: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    for key in path[:-1]:
        data = data[key]
    data[path[-1]] = value


def _get(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    for key in path:
        data = data[key]
    return data


@pytest.fixture(scope="module")
def current() -> dict[str, Any]:
    return snapshots.collect()


def test_the_baseline_has_the_intended_before_values() -> None:
    before = json.loads(snapshots.BEFORE.read_text(encoding="utf-8"))
    for path, old, _new, reason in INTENDED:
        assert _get(before, path) == old, (path, reason)


def test_everything_else_is_unchanged(current: dict[str, Any]) -> None:
    expected = json.loads(snapshots.BEFORE.read_text(encoding="utf-8"))
    for path, _old, new, _reason in INTENDED:
        _set(expected, path, new)
    # File modes and links are POSIX only; elsewhere these sections of the baseline are not comparable.
    if not POSIX:
        for section in ("files", "existing", "symlink"):
            expected.pop(section), current.pop(section)
    assert snapshots.dumps(current) == snapshots.dumps(expected)


def test_the_profile_is_the_one_of_the_plan() -> None:
    """Doc 06 MIG-ORK-3 field 4; equal to Tabrion's ``SQLiteProfile.orkexa()``."""
    profile = _storage.SQLITE_PROFILE
    assert profile.journal_mode == "wal" and profile.synchronous == "full"
    assert profile.foreign_keys is True and profile.busy_timeout == 30.0 and profile.secure_delete is True
    assert profile.file_mode == 0o600 and profile.allow_shared_file is False
    assert profile == SQLiteProfile.orkexa()


@pytest.mark.parametrize(
    ("code", "reason"),
    [
        ("sqlite.symlink", "symlink"),
        ("sqlite.unsafe_owner", "other_owner"),
        ("sqlite.unsafe_mode", "writable_by_others"),
        ("sqlite.unsafe_path", "unsafe_path"),
        ("sqlite.some_future_code", "unsafe_path"),
    ],
)
def test_every_refusal_of_tabrion_is_storage_symlink(code: str, reason: str) -> None:
    with pytest.raises(RetrievalError) as error, _storage.storage_errors():
        raise UnsafePath(code)
    assert error.value.code == "storage.symlink" and error.value.reason == reason
    assert error.value.__cause__ is None


@pytest.mark.skipif(not POSIX, reason="file owners are POSIX only")
@pytest.mark.parametrize("opener", OPENERS, ids=lambda cls: cls.__name__)
def test_an_existing_database_of_another_user_is_refused(
    tmp_path: Path, opener: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """E-NOD-SD: a database owned by another user is refused (``reason`` ``other_owner``)."""
    database = tmp_path / "data.db"
    opener(database).close()
    owner = database.stat().st_uid
    monkeypatch.setattr(os, "geteuid", lambda: owner + 1)
    with pytest.raises(RetrievalError) as error:
        opener(database)
    assert error.value.code == "storage.symlink" and error.value.reason == "other_owner"
    assert str(tmp_path) not in str(error.value)


@pytest.mark.skipif(not POSIX, reason="file modes are POSIX only")
@pytest.mark.parametrize("opener", OPENERS, ids=lambda cls: cls.__name__)
def test_a_side_file_writable_by_others_is_refused(tmp_path: Path, opener: Any) -> None:
    database = tmp_path / "data.db"
    opener(database).close()
    wal = tmp_path / "data.db-wal"
    wal.write_bytes(b"")
    wal.chmod(0o666)
    with pytest.raises(RetrievalError) as error:
        opener(database)
    assert error.value.code == "storage.symlink" and error.value.reason == "writable_by_others"


@pytest.mark.skipif(not POSIX, reason="file modes are POSIX only")
@pytest.mark.parametrize("opener", OPENERS, ids=lambda cls: cls.__name__)
def test_an_existing_group_readable_database_keeps_its_mode(tmp_path: Path, opener: Any) -> None:
    """Existing files are not changed with chmod; readable by group or others is still accepted."""
    for mode in (0o640, 0o644):
        database = tmp_path / f"data-{mode:o}.db"
        sqlite3.connect(database).close()
        database.chmod(mode)
        opener(database).close()
        assert database.stat().st_mode & 0o777 == mode


def test_a_nested_write_block_is_refused_and_leaves_the_outer_one_intact(tmp_path: Path) -> None:
    """T-NOTX: Tabrion's blocks never join a transaction they did not open."""
    from tabrion.errors import UsageError

    memory = SQLiteMemory(tmp_path / "m.db")
    try:
        with memory._transaction():
            with pytest.raises(UsageError), memory._transaction():
                pass
            assert memory._connection.in_transaction
        assert not memory._connection.in_transaction
    finally:
        memory.close()


def test_the_core_imports_tabrion_sqlite_but_not_its_server_modules() -> None:
    """Field 7: the core loads ``tabrion.sqlite``, never ``tabrion.postgres`` or the SQLAlchemy layer."""
    import subprocess

    probe = (
        "import sys, orkexa, orkexa.backends.sqlite\n"
        "print(sorted(m for m in sys.modules if m == 'tabrion' or m.startswith('tabrion.')))\n"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    loaded = set(json.loads(result.stdout.replace("'", '"')))
    assert "tabrion.sqlite" in loaded
    assert not {
        name for name in loaded if name.startswith(("tabrion.postgres", "tabrion._sa", "tabrion.sqlalchemy"))
    }
