"""SQLite backend robustness: concurrent writers, query limits, damaged and legacy files (MUP-ONA MP-A10)."""

from __future__ import annotations

from pathlib import Path
import sqlite3
import subprocess
import sys
import threading

import pytest

from orkexa import SQLiteIndex

ROUNDS = 100
"""Sources each concurrent deleter removes; every round is a chance for the PS-8 race."""

_DELETER = """
import sys, time
from pathlib import Path
from orkexa import SQLiteIndex

database, rounds, go = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
index = SQLiteIndex(database)
print("ready", flush=True)
while not go.exists():
    time.sleep(0.001)
errors = 0
for number in range(rounds):
    try:
        index.delete(f"source-{number}")
    except Exception as exc:
        errors += 1
        print(f"error {type(exc).__name__}: {exc}", flush=True)
index.close()
print(f"errors {errors}", flush=True)
"""


def _fill(database: Path) -> None:
    with SQLiteIndex(database) as index:
        for number in range(ROUNDS):
            for chunk in range(3):
                index.add(f"round {number} chunk {chunk} alpha beta gamma", source=f"source-{number}")


def _assert_intact(database: Path) -> None:
    connection = sqlite3.connect(database)
    try:
        connection.execute("INSERT INTO index_fts (index_fts) VALUES ('integrity-check')")
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        connection.close()
    with SQLiteIndex(database) as index:
        assert len(index) == 0
        assert index.search("alpha beta") == []
        index.add("a fresh chunk about alpha after the race", source="fresh")
        assert [hit.source for hit in index.search("alpha")] == ["fresh"]


def _delete_with_two_instances(database: Path) -> list[str]:
    errors: list[str] = []
    barrier = threading.Barrier(2)

    def work() -> None:
        index = SQLiteIndex(database)
        try:
            barrier.wait()
            for number in range(ROUNDS):
                try:
                    index.delete(f"source-{number}")
                except sqlite3.DatabaseError as exc:
                    errors.append(f"{type(exc).__name__}: {exc}")
        finally:
            index.close()

    threads = [threading.Thread(target=work) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
    return errors


def _delete_with_two_processes(database: Path, tmp_path: Path) -> list[str]:
    go = tmp_path / "go"
    workers = [
        subprocess.Popen(
            [sys.executable, "-c", _DELETER, str(database), str(ROUNDS), str(go)],
            stdout=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    try:
        for worker in workers:
            assert worker.stdout is not None
            assert worker.stdout.readline().strip() == "ready"
        go.touch()  # both workers start deleting at the same moment
        outputs = [worker.communicate(timeout=120)[0] for worker in workers]
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
                worker.wait()
    assert all(worker.returncode == 0 for worker in workers), outputs
    return [line for output in outputs for line in output.splitlines() if line.startswith("error ")]


# MUP-ONA T-SQL-1
@pytest.mark.xfail(strict=True, reason="PS-8")
def test_two_instances_and_two_processes_delete_the_same_sources(tmp_path: Path) -> None:
    """Two instances (threads) and then two processes delete the same 100 sources at once.

    ``SQLiteIndex.delete`` must read the rows inside its write transaction: otherwise both writers send the
    FTS5 ``'delete'`` for the same rows, and the second one fails with "database disk image is malformed"
    (PS-8).
    """
    threads_db = tmp_path / "threads.db"
    _fill(threads_db)
    thread_errors = _delete_with_two_instances(threads_db)

    processes_db = tmp_path / "processes.db"
    _fill(processes_db)
    process_errors = _delete_with_two_processes(processes_db, tmp_path)

    assert thread_errors == []
    assert process_errors == []
    _assert_intact(threads_db)
    _assert_intact(processes_db)
