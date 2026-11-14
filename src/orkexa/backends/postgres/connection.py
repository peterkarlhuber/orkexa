"""Where the PostgreSQL classes get their connections: the :class:`ConnectionSource` protocol and
:class:`PsycopgSource`, a ready-made source on psycopg 3.

The PostgreSQL backend never opens, pools or locks a connection by itself. A host passes a
:class:`ConnectionSource`, and every statement runs inside one of its transactions. The protocol has one
method, :meth:`ConnectionSource.transaction`, a context manager for one transaction that yields a
:class:`Connection` (anything with psycopg's ``execute``; a psycopg connection is one).

``mode`` says what the transaction will hold, so a source can run it the cheapest way it knows without
changing its meaning:

* ``"read"``: reads only. A source may run each statement on its own (autocommit, each with its own snapshot)
  and repeat a read whose connection was lost;
* ``"write"``: one statement that changes rows. It is atomic on its own, so a source may run it in autocommit;
  it is never repeated (it may have been applied);
* ``"transaction"``: several statements, all applied or none (``BEGIN`` ... ``COMMIT``; ``ROLLBACK`` when the
  block raises).

A source that is already inside a transaction of the calling thread joins it in every mode. ``tenant_id`` is
the tenant the statements run for. It is the optional hook for row-level security: a source that sets
``set_config('app.tenant_id', <tenant>, true)`` does so at the start of the transaction (transaction-local, so
it must open a transaction even for ``read`` and ``write``).

psycopg is imported only here, and only when a :class:`PsycopgSource` is built or a statement is rendered
(:func:`psycopg_sql`): ``import orkexa.backends.postgres`` works without the ``postgres`` extra and opens no
connection. ``tabrion.postgres`` (also in the extra ``postgres``, TABRION MIG-ORK-2) is imported only when a
:class:`PsycopgSource` sets session settings or is pinged.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
import math
import re
import threading
import time
from types import ModuleType
from typing import Any, Literal, Protocol, runtime_checkable

__all__ = [
    "MISSING_PSYCOPG",
    "MODES",
    "Connection",
    "ConnectionSource",
    "Cursor",
    "PsycopgSource",
    "TransactionMode",
    "psycopg_sql",
]

#: What one transaction of a :class:`ConnectionSource` holds (see the module docstring).
TransactionMode = Literal["read", "write", "transaction"]
MODES: frozenset[str] = frozenset({"read", "write", "transaction"})
#: The message of the :class:`ImportError` raised when psycopg is missing.
MISSING_PSYCOPG = 'The PostgreSQL backend needs psycopg: pip install "orkexa[postgres]"'
#: The message of the :class:`ImportError` raised when Tabrion's PostgreSQL helpers are missing.
_MISSING_TABRION = 'Session settings and ping of PsycopgSource need Tabrion: pip install "orkexa[postgres]"'
#: A setting name for :class:`PsycopgSource` ``tenant_setting``: ``<prefix>.<name>`` (a custom setting).
_SETTING = re.compile(r"[a-z_][a-z0-9_]*\.[a-z_][a-z0-9_]*")
#: The largest timeout PostgreSQL accepts (``INT_MAX`` milliseconds).
_MAX_TIMEOUT_MS = 2_147_483_647


class Cursor(Protocol):
    """What :meth:`Connection.execute` returns (a psycopg cursor is one)."""

    @property
    def rowcount(self) -> int:
        """Rows the statement returned or changed."""
        ...

    def fetchone(self) -> Any:
        """The next row, or ``None``."""
        ...

    def fetchall(self) -> Sequence[Any]:
        """The remaining rows."""
        ...


class Connection(Protocol):
    """A connection inside one transaction (a psycopg connection is one)."""

    def execute(self, query: Any, params: Any = None) -> Cursor:
        """Run ``query`` (text or a ``psycopg.sql.Composable``) with ``params`` bound, never interpolated."""
        ...


@runtime_checkable
class ConnectionSource(Protocol):
    """Gives the PostgreSQL classes their transactions (see the module docstring)."""

    def transaction(
        self, mode: TransactionMode = "transaction", *, tenant_id: str | None = None
    ) -> AbstractContextManager[Connection]:
        """A context manager for one transaction of the kind ``mode``, for the statements of ``tenant_id``."""
        ...


def _psycopg() -> ModuleType:
    """The psycopg package, or an :class:`ImportError` that names the extra."""
    try:
        import psycopg
    except ModuleNotFoundError as exc:
        if exc.name not in {"psycopg", None}:
            raise
        raise ImportError(MISSING_PSYCOPG, name="psycopg") from None
    return psycopg


def _require_tabrion() -> None:
    """Import ``tabrion.postgres`` (extra ``postgres``), or raise an :class:`ImportError` naming the extra."""
    try:
        import tabrion.postgres  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name not in {"tabrion", "tabrion.postgres", None}:
            raise
        raise ImportError(_MISSING_TABRION, name="tabrion") from None


def _milliseconds(name: str, seconds: float | None) -> int | None:
    """A timeout in seconds (``0`` turns it off) in the milliseconds PostgreSQL counts; ``None`` stays."""
    if seconds is None:
        return None
    if isinstance(seconds, bool) or not isinstance(seconds, int | float) or not math.isfinite(seconds):
        raise ValueError(f"{name} must be a finite number of seconds or None")
    if seconds < 0:
        raise ValueError(f"{name} must be at least 0 seconds")
    milliseconds = math.ceil(seconds * 1000)
    if milliseconds > _MAX_TIMEOUT_MS:
        raise ValueError(f"{name} must be at most {_MAX_TIMEOUT_MS / 1000} seconds")
    return milliseconds


def psycopg_sql() -> ModuleType:
    """``psycopg.sql`` (statements are rendered with it), or an :class:`ImportError` that names the extra."""
    _psycopg()
    from psycopg import sql

    return sql


class PsycopgSource:
    """A :class:`ConnectionSource` on one PostgreSQL database with psycopg 3 and a small pool of its own.

    Connections are opened on demand, up to ``max_connections`` at once; a thread that needs one more waits up
    to ``timeout`` seconds (then :class:`TimeoutError`). Connections run in autocommit: ``read`` and ``write``
    run their statement on its own, ``transaction`` inside ``BEGIN`` ... ``COMMIT`` (``ROLLBACK`` when the
    block raises). A ``transaction`` block is joined by every :meth:`transaction` the same thread opens inside
    it, so a host can make the commands of several stores one transaction. A connection that broke is closed
    instead of being reused.

    Session settings (TABRION MIG-ORK-2) are set once on every new connection, right after it is opened,
    with ``tabrion.postgres`` (one ``set_config(..., false)`` round trip). A setting left at ``None`` (the
    default for all of them) is not touched, so the server's and the role's defaults apply as before. A
    connection whose settings fail is closed and the error propagates. :meth:`ping` checks the server.

    Args:
        conninfo: A libpq connection string or URL (``"host=db dbname=app user=app"``).
        max_connections: Most connections open at once.
        timeout: Seconds a thread waits for a free connection.
        tenant_setting: A custom setting (``"app.tenant_id"``) to set to the tenant at the start of every
            transaction, transaction-local (``set_config(..., true)``), as the hook for row-level security.
            Then ``read`` and ``write`` run in a transaction too. ``None`` (default): nothing is set.
        statement_timeout: Seconds a statement may run before the server cancels it (``statement_timeout``;
            ``0`` turns it off). ``None`` (default): unchanged.
        idle_in_transaction_timeout: Seconds a session may stay idle inside a transaction before the server
            ends it (``idle_in_transaction_session_timeout``; ``0`` turns it off). ``None`` (default):
            unchanged.
        lock_timeout: Seconds a statement waits for a lock (``lock_timeout``; ``0`` turns it off). ``None``
            (default): unchanged.
        application_name: The ``application_name`` of the connections (printable ASCII, at most 63
            characters). ``None`` (default): unchanged.
        **connect_args: Further keyword arguments of ``psycopg.connect`` (``autocommit`` is always ``True``).

    Raises:
        ImportError: psycopg is not installed, or Tabrion is not installed and a session setting is given
            (``pip install "orkexa[postgres]"``).
        ValueError: ``max_connections`` below 1, an invalid ``tenant_setting``, a timeout that is negative,
            not a finite number or longer than PostgreSQL allows, or an ``application_name`` that is not
            printable ASCII of at most 63 characters.
    """

    def __init__(
        self,
        conninfo: str = "",
        *,
        max_connections: int = 10,
        timeout: float = 30.0,
        tenant_setting: str | None = None,
        statement_timeout: float | None = None,
        idle_in_transaction_timeout: float | None = None,
        lock_timeout: float | None = None,
        application_name: str | None = None,
        **connect_args: Any,
    ) -> None:
        self._psycopg = _psycopg()
        if max_connections < 1:
            raise ValueError("max_connections must be at least 1")
        if tenant_setting is not None and not _SETTING.fullmatch(tenant_setting):
            raise ValueError(f"Invalid tenant_setting {tenant_setting!r}: it must look like 'app.tenant_id'")
        statement_ms = _milliseconds("statement_timeout", statement_timeout)
        idle_ms = _milliseconds("idle_in_transaction_timeout", idle_in_transaction_timeout)
        lock_ms = _milliseconds("lock_timeout", lock_timeout)
        #: The session settings of every new connection (``None``: none are set, Tabrion is not imported).
        self._configure: Any = None
        if any(value is not None for value in (statement_ms, idle_ms, lock_ms, application_name)):
            _require_tabrion()
            from tabrion.errors import ConfigurationError
            from tabrion.postgres import SessionSettings, session_configurator

            try:
                # Only these four; every other setting stays as the server and the role set it. Validated
                # once, here (R-SET allowlist): a wrong value fails the constructor, not a connection.
                self._configure = session_configurator(
                    SessionSettings(
                        statement_timeout_ms=statement_ms,
                        lock_timeout_ms=lock_ms,
                        idle_in_transaction_timeout_ms=idle_ms,
                        transaction_timeout_ms=None,
                        application_name=application_name,
                        search_path=None,
                        timezone=None,
                    )
                )
            except ConfigurationError as exc:
                # The setting's name only, never its value (Tabrion's rule for configuration errors).
                setting = str(exc.details.get("setting", "session setting")).removesuffix("_ms")
                raise ValueError(f"Invalid {setting} for PsycopgSource") from None
        self.conninfo = conninfo
        self.max_connections = max_connections
        self.timeout = timeout
        self.tenant_setting = tenant_setting
        self._connect_args = {**connect_args, "autocommit": True}
        self._idle: list[Any] = []
        self._open = 0
        self._closed = False
        self._available = threading.Condition()
        self._local = threading.local()
        #: The lock the stores on this source hold for their commands (their default ``lock``): the read,
        #: change and write of one command never interleave with another store's in this process.
        self.lock = threading.RLock()

    def _acquire(self, wait: float | None = None, connect_args: dict[str, Any] | None = None) -> Any:
        wait = self.timeout if wait is None else wait
        deadline = time.monotonic() + wait
        with self._available:
            while True:
                if self._closed:
                    raise RuntimeError("This PsycopgSource is closed")
                if self._idle:
                    return self._idle.pop()
                if self._open < self.max_connections:
                    self._open += 1
                    break
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError(f"No free PostgreSQL connection within {wait} seconds")
                self._available.wait(left)
        try:
            arguments = {**self._connect_args, **(connect_args or {})}
            connection = self._psycopg.connect(self.conninfo, **arguments)
        except BaseException:
            with self._available:
                self._open -= 1
                self._available.notify()
            raise
        if self._configure is not None:
            try:
                self._configure(connection)
            except BaseException:
                with self._available:
                    self._open -= 1
                    self._available.notify()
                connection.close()
                raise
        return connection

    def _release(self, connection: Any) -> None:
        reusable = (
            not connection.closed
            and not connection.broken
            and connection.info.transaction_status == self._psycopg.pq.TransactionStatus.IDLE
        )
        with self._available:
            if reusable and not self._closed:
                self._idle.append(connection)
            else:
                self._open -= 1
                connection.close()
            self._available.notify()

    @contextmanager
    def transaction(
        self, mode: TransactionMode = "transaction", *, tenant_id: str | None = None
    ) -> Iterator[Connection]:
        """One transaction of the kind ``mode`` (see the module docstring), for the statements of
        ``tenant_id``."""
        if mode not in MODES:
            raise ValueError(f"Unknown transaction mode {mode!r}")
        joined: Any = getattr(self._local, "connection", None)
        if joined is not None:
            yield joined
            return
        connection = self._acquire()
        try:
            if mode == "transaction" or self.tenant_setting is not None:
                with connection.transaction():
                    if self.tenant_setting is not None:
                        connection.execute(
                            "SELECT set_config(%s, %s, true)", (self.tenant_setting, tenant_id or "")
                        )
                    self._local.connection = connection
                    try:
                        yield connection
                    finally:
                        self._local.connection = None
            else:
                yield connection
        finally:
            self._release(connection)

    def ping(self, timeout: float = 5.0) -> bool:
        """Whether the server answers within ``timeout`` seconds, with ``tabrion.postgres.ping``.

        Runs ``SELECT`` on a connection of the pool under ``SET LOCAL statement_timeout`` in a transaction
        of its own that is rolled back (a savepoint inside a ``transaction`` block of this thread, which it
        joins). A connection that has to be opened for it gets ``connect_timeout`` = ``timeout`` (rounded up
        to whole seconds) unless ``connect_args`` name one. Never raises for a database or connection error,
        nor when no connection is free within ``timeout``: the answer is ``False``.

        Raises:
            ImportError: Tabrion is not installed (``pip install "orkexa[postgres]"``).
            ValueError: ``timeout`` is not a positive, finite number of seconds.
            RuntimeError: The source is closed.
        """
        _require_tabrion()
        from tabrion.postgres import ping

        if isinstance(timeout, bool) or not isinstance(timeout, int | float) or not math.isfinite(timeout):
            raise ValueError("timeout must be a positive, finite number of seconds")
        if timeout <= 0:
            raise ValueError("timeout must be a positive, finite number of seconds")
        joined: Any = getattr(self._local, "connection", None)
        if joined is not None:
            return bool(ping(joined, timeout=timeout).ok)
        deadline = time.monotonic() + timeout
        connect_args = {"connect_timeout": math.ceil(timeout)}
        if "connect_timeout" in self._connect_args:
            connect_args = {}
        # An idle connection the server has dropped (a restart) fails the ping and is discarded: then the
        # next one, at most every pooled connection and one new one, so that a stale pool does not report a
        # healthy server as down.
        for _ in range(self.max_connections + 1):
            left = deadline - time.monotonic()
            if left <= 0:
                return False
            try:
                connection = self._acquire(left, connect_args)
            except (TimeoutError, self._psycopg.Error):
                return False
            try:
                if ping(connection, timeout=left).ok:
                    return True
                if not (connection.closed or connection.broken):
                    return False
            finally:
                self._release(connection)
        return False

    def close(self) -> None:
        """Close every idle connection; connections in use are closed when they are given back."""
        with self._available:
            self._closed = True
            idle, self._idle = self._idle, []
            self._open -= len(idle)
            self._available.notify_all()
        for connection in idle:
            connection.close()

    def __enter__(self) -> PsycopgSource:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
