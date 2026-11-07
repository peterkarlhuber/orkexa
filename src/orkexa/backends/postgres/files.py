"""An Orkexa :class:`~orkexa.FileStore` of one tenant and namespace on PostgreSQL (table
``<prefix>files``)."""

from __future__ import annotations

import builtins
import threading
from typing import Any

from orkexa import FileStore, StoredFile

from ._sql import Executor, PostgresLayout, check_namespace, check_tenant, jsonb_text
from .connection import ConnectionSource

__all__ = ["PostgresFileStore"]


class PostgresFileStore(FileStore):
    """An Orkexa :class:`~orkexa.FileStore` of one tenant and namespace (table ``<prefix>files``).

    Args:
        source: Gives the connections (:class:`~orkexa.backends.postgres.ConnectionSource`).
        tenant_id: The partition: every statement binds it, and the partition guard refuses any other value.
            A non-empty string of at most 128 characters.
        layout: Names the tables (:class:`~orkexa.backends.postgres.PostgresLayout`).
        namespace: Which store of the tenant this backend works on.
        lock: The lock the store's commands hold (default: a lock of this store). A host whose source takes a
            lock of its own for writes passes that lock, so a command never waits for it while holding
            another.
        files_api: See :class:`~orkexa.FileStore`.
        max_image_bytes: See :class:`~orkexa.FileStore`.
        max_document_bytes: See :class:`~orkexa.FileStore`.

    Raises:
        ValueError: An invalid ``tenant_id`` or an empty ``namespace``.
        TypeError: ``layout`` is not a :class:`~orkexa.backends.postgres.PostgresLayout`.

    The store enforces the partition; it never checks permissions (the host decides who may use a tenant).
    """

    def __init__(
        self,
        source: ConnectionSource,
        tenant_id: str,
        *,
        layout: PostgresLayout,
        namespace: str = "default",
        lock: threading.RLock | None = None,
        files_api: Any | None = None,
        max_image_bytes: int = 5_000_000,
        max_document_bytes: int = 20_000_000,
    ) -> None:
        super().__init__(
            files_api=files_api, max_image_bytes=max_image_bytes, max_document_bytes=max_document_bytes
        )
        self.tenant_id = check_tenant(tenant_id)
        self.namespace = check_namespace(namespace)
        self.source = source
        self._sql = Executor(source, layout=layout, tenant_id=tenant_id)
        self.layout = layout
        if lock is not None:
            self._lock = lock

    def _key(self, **params: Any) -> dict[str, Any]:
        return {"tenant_id": self.tenant_id, "namespace": self.namespace, **params}

    @property
    def capabilities(self) -> frozenset[str]:
        """``partitioned`` (by tenant) and ``concurrent_writers`` (:attr:`orkexa.FileStore.capabilities`)."""
        return frozenset({"partitioned", "concurrent_writers"})

    def close(self) -> None:
        """Nothing to do: the connections belong to the source."""

    def _save(self, entry: StoredFile, data: bytes | None) -> None:
        if data is None:
            self._sql.execute(
                "files_update_entry", self._key(entry=jsonb_text(entry.model_dump_json()), id=entry.id)
            )
        else:
            self._sql.execute(
                "files_upsert", self._key(id=entry.id, entry=jsonb_text(entry.model_dump_json()), data=data)
            )

    def _entry(self, file_id: str) -> StoredFile | None:
        rows = self._sql.execute("files_entry", self._key(id=file_id)).rows
        return StoredFile.model_validate_json(rows[0][0]) if rows else None

    def _data(self, file_id: str) -> bytes | None:
        rows = self._sql.execute("files_data", self._key(id=file_id)).rows
        return bytes(rows[0][0]) if rows else None

    def _entries(self) -> builtins.list[StoredFile]:
        rows = self._sql.execute("files_entries", self._key()).rows
        return [StoredFile.model_validate_json(row[0]) for row in rows]

    def _remove(self, file_id: str) -> None:
        self._sql.execute("files_delete", self._key(id=file_id))
