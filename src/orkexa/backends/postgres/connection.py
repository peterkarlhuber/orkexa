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
connection.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
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
#: A setting name for :class:`PsycopgSource` ``tenant_setting``: ``<prefix>.<name>`` (a custom setting).
_SETTING = re.compile(r"[a-z_][a-z0-9_]*\.[a-z_][a-z0-9_]*")


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

    Args:
        conninfo: A libpq connection string or URL (``"host=db dbname=app user=app"``).
        max_connections: Most connections open at once.
        timeout: Seconds a thread waits for a free connection.
        tenant_setting: A custom setting (``"app.tenant_id"``) to set to the tenant at the start of every
            transaction, transaction-local (``set_config(..., true)``), as the hook for row-level security.
            Then ``read`` and ``write`` run in a transaction too. ``None`` (default): nothing is set.
        **connect_args: Further keyword arguments of ``psycopg.connect`` (``autocommit`` is always ``True``).

    Raises:
        ImportError: psycopg is not installed (``pip install "orkexa[postgres]"``).
        ValueError: ``max_connections`` below 1 or an invalid ``tenant_setting``.
    """

    def __init__(
        self,
        conninfo: str = "",
        *,
        max_connections: int = 10,
        timeout: float = 30.0,
        tenant_setting: str | None = None,
        **connect_args: Any,
    ) -> None:
        self._psycopg = _psycopg()
        if max_connections < 1:
            raise ValueError("max_connections must be at least 1")
        if tenant_setting is not None and not _SETTING.fullmatch(tenant_setting):
            raise ValueError(f"Invalid tenant_setting {tenant_setting!r}: it must look like 'app.tenant_id'")
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

    def _acquire(self) -> Any:
        deadline = time.monotonic() + self.timeout
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
                    raise TimeoutError(f"No free PostgreSQL connection within {self.timeout} seconds")
                self._available.wait(left)
        try:
            return self._psycopg.connect(self.conninfo, **self._connect_args)
        except BaseException:
            with self._available:
                self._open -= 1
                self._available.notify()
            raise

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
