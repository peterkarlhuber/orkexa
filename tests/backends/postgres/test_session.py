"""TABRION MIG-ORK-2: the session settings, ``ping`` and the safe error texts of the PostgreSQL backend.

* :class:`~orkexa.backends.postgres.PsycopgSource` sets ``statement_timeout``,
  ``idle_in_transaction_timeout``, ``lock_timeout`` and ``application_name`` on every new connection with
  ``tabrion.postgres`` (``session_configurator``); with none of them (the default) a connection is exactly
  what ``psycopg.connect`` gives (the ``pg_settings`` diff is empty) and Tabrion is not imported.
* :meth:`~orkexa.backends.postgres.PsycopgSource.ping` answers ``True`` or ``False`` and never raises for a
  database or connection error.
* :func:`~orkexa.backends.postgres.verify_schema` reports a database error by Tabrion's ``safe_message``: no
  server text, no statement, no connection parameters.

Every test is marked ``postgres``: each needs psycopg (the ``postgres`` extra), most a server.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import os
import socket
import subprocess
import sys
import threading
import time
from typing import Any

import pytest

from orkexa.backends.postgres import PostgresMemory, PsycopgSource, verify_schema

from . import support

pytestmark = pytest.mark.postgres

TENANT = "tenant-a"
SETTINGS = {
    "statement_timeout": 1.5,
    "idle_in_transaction_timeout": 60,
    "lock_timeout": 0.25,
    "application_name": "orkexa-mig-ork-2",
}
EXPECTED = {
    "statement_timeout": "1500ms",
    "idle_in_transaction_session_timeout": "1min",
    "lock_timeout": "250ms",
    "application_name": "orkexa-mig-ork-2",
}
_SHOW = (
    "SELECT current_setting('statement_timeout'), current_setting('idle_in_transaction_session_timeout'),"
    " current_setting('lock_timeout'), current_setting('application_name')"
)


def _python(code: str) -> subprocess.CompletedProcess[str]:
    """``python -I -c code`` from the repository root (no PYTHON* variables): a fresh ``sys.modules``."""
    env = {key: value for key, value in os.environ.items() if not key.startswith("PYTHON")}
    return subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, env=env, timeout=300, check=False
    )


@contextmanager
def opened(**kwargs: Any) -> Iterator[PsycopgSource]:
    source = PsycopgSource(**kwargs)
    try:
        yield source
    finally:
        source.close()


def _show(source: PsycopgSource, mode: str = "read") -> dict[str, str]:
    with source.transaction(mode) as connection:  # type: ignore[arg-type]
        row = connection.execute(_SHOW).fetchall()[0]
    return dict(zip(EXPECTED, row, strict=True))


@contextmanager
def _cancel_after(connection: Any, seconds: float) -> Iterator[None]:
    """Cancel the statement of ``connection`` after ``seconds`` (it fails with ``QueryCanceled``), so that
    a test whose limit did not work fails instead of hanging."""
    pid = connection.execute("SELECT pg_backend_pid()").fetchall()[0][0]

    def cancel() -> None:
        with support.admin() as admin:
            admin.execute("SELECT pg_cancel_backend(%s)", (pid,))

    timer = threading.Timer(seconds, cancel)
    timer.start()
    try:
        yield
    finally:
        timer.cancel()


def _free_port() -> int:
    """A local TCP port nothing listens on (a stopped server)."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


# ---- without a server
# -------------------------------------------------------------------------------------------------


def test_the_default_source_imports_no_tabrion() -> None:
    """Import isolation: ``import orkexa`` loads neither psycopg nor Tabrion, and a source without session
    settings imports psycopg only."""
    result = _python(
        "import sys\n"
        "import orkexa, orkexa.backends.postgres\n"
        "from orkexa.backends.postgres import PsycopgSource\n"
        "watched = ('psycopg', 'tabrion')\n"
        "print(sorted(m for m in sys.modules if m.split('.')[0] in watched))\n"
        "PsycopgSource('host=db')\n"
        "print(sorted(m for m in sys.modules if m.split('.')[0] == 'tabrion'))\n"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["[]", "[]"]


def test_session_settings_and_ping_name_the_extra_without_tabrion() -> None:
    result = _python(
        "import sys\n"
        "sys.modules['tabrion'] = None\n"
        "from orkexa.backends.postgres import PsycopgSource\n"
        "source = PsycopgSource('host=db')\n"  # no setting: works without Tabrion, as before
        "for attempt in (lambda: PsycopgSource('host=db', statement_timeout=1), source.ping):\n"
        "    try:\n"
        "        attempt()\n"
        "    except ImportError as exc:\n"
        "        assert 'pip install -r requirements/tabrion.txt \"orkexa[postgres]\"' in str(exc), exc\n"
        "        assert exc.name == 'tabrion', exc\n"
        "    else:\n"
        "        raise AssertionError('no ImportError')\n"
        "print('ok')\n"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_invalid_session_settings_are_refused_at_construction() -> None:
    for name in ("statement_timeout", "idle_in_transaction_timeout", "lock_timeout"):
        for bad in (-1, -0.001, float("nan"), float("inf"), True, "1", 2_147_483.648, 1e306, 10**400):
            with pytest.raises(ValueError, match=name):
                PsycopgSource("host=db", **{name: bad})
    for bad_name in ("ünïcode", "x" * 64, "tab\there"):
        with pytest.raises(ValueError, match="application_name") as caught:
            PsycopgSource("host=db", application_name=bad_name)
        assert bad_name not in str(caught.value)  # the setting's name only, never its value
    # The limits themselves are fine; nothing connects before the first transaction.
    PsycopgSource("host=db", statement_timeout=0, lock_timeout=2_147_483, application_name="x" * 63)


def test_ping_refuses_a_bad_timeout() -> None:
    source = PsycopgSource("host=db")
    for bad in (0, -1, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="timeout"):
            source.ping(bad)


# ---- on a real server
# -------------------------------------------------------------------------------------------------


def test_the_settings_reach_every_connection_and_mode(db: support.Database) -> None:
    conninfo = support.conninfo(db.schema, runtime=False)
    with opened(conninfo=conninfo, max_connections=2, tenant_setting="app.tenant_id", **SETTINGS) as source:
        for mode in ("read", "write", "transaction"):
            assert _show(source, mode) == EXPECTED
        # Two connections at once: the second one is new and configured too.
        with source.transaction("transaction") as first:
            holder: list[dict[str, str]] = []
            thread = threading.Thread(target=lambda: holder.append(_show(source)))
            thread.start()
            thread.join(10)
            assert first.execute(_SHOW).fetchall()[0] == tuple(EXPECTED.values())
        assert holder == [EXPECTED]
        assert source._open == 2
        # The stores work unchanged on a configured source.
        memory = PostgresMemory(source, TENANT, layout=db.layout, namespace="mig-ork-2")
        memory.create("/memories/a.md", "configured")
        assert "configured" in memory.view("/memories/a.md")


def test_without_settings_a_connection_is_unchanged(db: support.Database) -> None:
    """MIG-ORK-2 field 10: the default changes nothing (the ``pg_settings`` diff to a plain connection is
    empty)."""
    import psycopg

    conninfo = support.conninfo(db.schema, runtime=False)
    query = "SELECT name, setting FROM pg_settings ORDER BY name"
    with psycopg.connect(conninfo, autocommit=True) as plain:
        expected = plain.execute(query).fetchall()
    with opened(conninfo=conninfo) as source:
        assert source._configure is None
        with source.transaction("read") as connection:
            assert connection.execute(query).fetchall() == expected


def test_the_statement_timeout_cancels_a_long_statement(db: support.Database) -> None:
    import psycopg

    with opened(
        conninfo=support.conninfo(db.schema, runtime=False), max_connections=1, statement_timeout=0.2
    ) as source:
        started = time.monotonic()
        with pytest.raises(psycopg.errors.QueryCanceled), source.transaction("read") as connection:
            connection.execute("SELECT pg_sleep(10)")
        assert time.monotonic() - started < 5  # not the end of pg_sleep
        # The connection is still good and still configured.
        with source.transaction("read") as connection:
            rows = connection.execute("SELECT current_setting('statement_timeout')").fetchall()
            assert rows == [("200ms",)]
        assert source._open == 1


def test_the_lock_timeout_ends_a_wait_for_a_lock(db: support.Database) -> None:
    import psycopg

    db.query("CREATE TABLE mig_ork_2_locked (id int)")
    with support.admin() as admin, admin.transaction():
        admin.execute(f'LOCK TABLE "{db.schema}".mig_ork_2_locked IN ACCESS EXCLUSIVE MODE')
        with opened(conninfo=support.conninfo(db.schema, runtime=False), lock_timeout=0.2) as source:
            started = time.monotonic()
            with (
                pytest.raises(psycopg.errors.LockNotAvailable),
                source.transaction("read") as connection,
                _cancel_after(connection, 10),  # without the setting the wait would never end
            ):
                connection.execute("SELECT * FROM mig_ork_2_locked")
            assert time.monotonic() - started < 5


def test_the_idle_in_transaction_timeout_ends_a_forgotten_transaction(db: support.Database) -> None:
    import psycopg

    with opened(
        conninfo=support.conninfo(db.schema, runtime=False),
        max_connections=1,
        idle_in_transaction_timeout=0.2,
    ) as source:
        with pytest.raises(psycopg.Error), source.transaction("transaction") as connection:
            connection.execute("SELECT 1")
            time.sleep(1.5)
            connection.execute("SELECT 1")
        # The broken connection was discarded; the next transaction gets a new one.
        with source.transaction("read") as connection:
            assert connection.execute("SELECT 1").fetchall() == [(1,)]


def test_a_connection_whose_settings_fail_is_closed(db: support.Database) -> None:
    with opened(
        conninfo=support.conninfo(db.schema, runtime=False), max_connections=1, lock_timeout=1
    ) as source:
        seen: list[Any] = []

        def failing(connection: Any) -> None:
            seen.append(connection)
            raise RuntimeError("configure failed")

        source._configure = failing
        with pytest.raises(RuntimeError, match="configure failed"), source.transaction("read"):
            pass
        assert seen[0].closed and source._open == 0 and not source._idle


def test_ping_answers_true_and_keeps_a_running_transaction(db: support.Database) -> None:
    with opened(conninfo=support.conninfo(db.schema, runtime=False), max_connections=1, **SETTINGS) as source:
        assert source.ping() is True
        assert source.ping(0.5) is True
        with source.transaction("transaction") as connection:
            connection.execute("CREATE TEMPORARY TABLE mig_ork_2_ping (id int)")
            assert source.ping() is True  # the joined connection, under a savepoint
            connection.execute("INSERT INTO mig_ork_2_ping VALUES (1)")
            assert connection.execute("SELECT count(*) FROM mig_ork_2_ping").fetchall() == [(1,)]
        assert source._open == 1 and len(source._idle) == 1
        assert _show(source) == EXPECTED  # ping's SET LOCAL ended with its transaction


def test_ping_is_false_for_a_stopped_server_and_never_raises() -> None:
    conninfo = f"host=127.0.0.1 port={_free_port()} user=nobody dbname=nothing"
    with opened(conninfo=conninfo, statement_timeout=1) as source:
        started = time.monotonic()
        assert source.ping(2) is False
        assert time.monotonic() - started < 10
        assert source._open == 0
    with pytest.raises(RuntimeError, match="closed"):
        source.ping()


def test_ping_replaces_a_connection_the_server_dropped(db: support.Database) -> None:
    with opened(conninfo=support.conninfo(db.schema, runtime=False), max_connections=2) as source:
        with source.transaction("read") as first, source.transaction("read") as second:
            pids = [c.execute("SELECT pg_backend_pid()").fetchall()[0][0] for c in (first, second)]
        assert len(source._idle) == 2
        with support.admin() as admin:
            for pid in pids:
                admin.execute("SELECT pg_terminate_backend(%s)", (pid,))
        assert source.ping() is True  # both stale connections were discarded, a new one answered
        assert source._open == 1


def test_ping_is_false_when_no_connection_is_free(db: support.Database) -> None:
    with opened(conninfo=support.conninfo(db.schema, runtime=False), max_connections=1, timeout=30) as source:
        held = threading.Event()
        release = threading.Event()

        def hold() -> None:
            with source.transaction("transaction"):
                held.set()
                release.wait(10)

        thread = threading.Thread(target=hold)
        thread.start()
        try:
            assert held.wait(10)
            started = time.monotonic()
            assert source.ping(0.3) is False  # waits at most the ping's timeout, not the source's
            assert time.monotonic() - started < 5
        finally:
            release.set()
            thread.join(10)
        assert source.ping() is True


def test_verify_schema_reports_a_database_error_without_server_text(db: support.Database) -> None:
    secret = "mig-ork-2-secret-value"
    statement = (
        f"DO $$ BEGIN RAISE EXCEPTION '{secret}' USING DETAIL = '{secret}-detail', ERRCODE = '23505'; END $$"
    )

    class Raising:
        def __init__(self, inner: PsycopgSource) -> None:
            self.inner = inner

        @contextmanager
        def transaction(self, mode: str = "transaction", *, tenant_id: str | None = None) -> Iterator[Any]:
            with self.inner.transaction(mode, tenant_id=tenant_id) as connection:  # type: ignore[arg-type]
                connection.execute(statement)
                yield connection

    report = verify_schema(Raising(db.source), db.layout)  # type: ignore[arg-type]
    assert [problem.kind for problem in report.problems] == ["error"]
    text = report.summary()
    assert "UniqueViolation" in text and "SQLSTATE 23505" in text, text
    assert secret not in text and "DETAIL" not in text and "RAISE" not in text, text


def test_verify_schema_reports_a_failed_connection_without_its_parameters() -> None:
    secret = "mig-ork-2-password"
    port = _free_port()
    conninfo = f"host=127.0.0.1 port={port} user=nobody password={secret} dbname=nothing connect_timeout=2"
    with opened(conninfo=conninfo) as source:
        report = verify_schema(source, support.LAYOUT)
    text = report.summary()
    assert [problem.kind for problem in report.problems] == ["error"]
    assert "OperationalError" in text and "connection error" in text, text
    assert secret not in text and "127.0.0.1" not in text and "nobody" not in text, text
