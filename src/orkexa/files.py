"""Stores for raw files (PDFs, images, text) that the model loads on demand.

The memory tool holds text notes; originals such as a scanned contract or a
chart live in a :class:`FileStore`. With ``files=`` an agent offers the model the
tool ``files``: ``list`` shows the catalog, ``load`` brings one original into the
conversation as a document or image block, only when the model needs it::

    from orkexa import Agent, LocalFileStore

    store = LocalFileStore("./data/files")
    report = store.add("q3-report.pdf", description="Quarterly report Q3 2026")

    agent = Agent(model="openai:gpt-5.5", files=store)
    agent.ask("Which risks does the Q3 report name?")  # the model loads report.id

Two backends share one implementation:

* :class:`LocalFileStore` keeps the files below a directory.
* :class:`SQLiteFileStore` keeps them in a SQLite database, one ``namespace``
  per user or project (it can share the database file with ``SQLiteMemory``).

With ``files_api=client`` (an ``anthropic.Anthropic``) a store uploads each file
to Anthropic's Files API the first time it is loaded and from then on sends only
the ``file_id`` instead of the base64 bytes. The original stays in the store.

The model never passes paths, only the ids the store hands out. For other storage
(object storage, a database server) subclass :class:`FileStore` and implement
its storage methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from datetime import datetime, timezone
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import re
import sqlite3
import threading
from typing import TYPE_CHECKING, Any, Literal
import uuid

from pydantic import BaseModel, Field

from ._storage import create_private_file, make_private_dirs
from .content import ContentBlock, ToolContent, document, image, uploaded_file
from .errors import ToolResultError
from .tools import Tool

if TYPE_CHECKING:
    from .memory import MemoryBackend

__all__ = ["FILES_TOOL_NAME", "FileStore", "LocalFileStore", "SQLiteFileStore", "StoredFile"]

FILES_TOOL_NAME = "files"
IMAGE_TYPES = ("image/jpeg", "image/png", "image/gif", "image/webp")
_ID = re.compile(r"[0-9a-f]{16}")
_EXTRA_TEXT_TYPES = {".md": "text/markdown", ".csv": "text/csv", ".json": "text/plain", ".yaml": "text/plain"}


class StoredFile(BaseModel):
    """Catalog entry of a stored file."""

    id: str
    name: str
    media_type: str
    size: int
    sha256: str
    created_at: datetime
    description: str = ""
    metadata: dict[str, str] = Field(default_factory=dict)
    api_file_id: str | None = None
    """The Files API id once the file was uploaded (``files_api=``)."""

    @property
    def kind(self) -> Literal["document", "image"]:
        return "image" if self.media_type in IMAGE_TYPES else "document"

    def line(self) -> str:
        """One catalog line as Claude sees it."""
        description = f" - {self.description}" if self.description else ""
        return f"{self.id} | {self.name} | {self.media_type} | {_format_size(self.size)}{description}"


# Python's mimetypes reads the registry on Windows, which may not know Office files; these are named here so a
# Word file is reported as unsupported on every operating system rather than as an unknown type.
_OFFICE_TYPES = {
    ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xls": "application/vnd.ms-excel",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".ppt": "application/vnd.ms-powerpoint",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}


def _format_size(size: int) -> str:
    if size < 1024:
        return f"{size}B"
    kilobytes = size / 1024
    return f"{kilobytes:.1f}KB" if kilobytes < 1024 else f"{kilobytes / 1024:.1f}MB"


def _media_type(name: str, media_type: str | None) -> str:
    if media_type:
        return media_type
    suffix = Path(name).suffix.lower()
    guessed = mimetypes.guess_type(name)[0] or _EXTRA_TEXT_TYPES.get(suffix) or _OFFICE_TYPES.get(suffix)
    if guessed is None:
        raise ValueError(f"Cannot tell the media type of {name!r}; pass media_type")
    return guessed


def _check_id(file_id: object) -> str:
    if not isinstance(file_id, str) or not _ID.fullmatch(file_id):
        raise ToolResultError(f"Error: invalid file id {file_id!r}; use an id from the files list")
    return file_id


class FileStore(ABC):
    """Keeps raw files and gives Claude the tool to list and load them.

    Accepted are images (JPEG, PNG, GIF, WebP), PDFs and text files, the formats
    Claude can read. A file's id is derived from its content, so adding the same
    bytes again returns the existing entry (with updated name/description).

    Methods are thread-safe (a lock), so a store can be shared by several agents.

    Args:
        files_api: An ``anthropic.Anthropic`` client. If given, ``load`` uploads a
            file to the Files API once and then references its ``file_id``.
        max_image_bytes: Largest image accepted by :meth:`add`.
        max_document_bytes: Largest PDF or text file accepted by :meth:`add`.
    """

    def __init__(
        self,
        *,
        files_api: Any | None = None,
        max_image_bytes: int = 5_000_000,
        max_document_bytes: int = 20_000_000,
    ) -> None:
        if max_image_bytes < 1 or max_document_bytes < 1:
            raise ValueError("max_image_bytes and max_document_bytes must be positive")
        self.files_api = files_api
        self.max_image_bytes = max_image_bytes
        self.max_document_bytes = max_document_bytes
        self._lock = threading.RLock()

    # ---- storage, implemented by subclasses ---------------------------------

    @abstractmethod
    def _save(self, entry: StoredFile, data: bytes | None) -> None:
        """Create or update ``entry``; ``data`` is ``None`` when only the entry changes."""

    @abstractmethod
    def _entry(self, file_id: str) -> StoredFile | None:
        """The catalog entry of ``file_id``, or ``None``."""

    @abstractmethod
    def _data(self, file_id: str) -> bytes | None:
        """The bytes of ``file_id``, or ``None``."""

    @abstractmethod
    def _entries(self) -> list[StoredFile]:
        """All catalog entries."""

    @abstractmethod
    def _remove(self, file_id: str) -> None:
        """Delete ``file_id`` and its bytes."""

    # ---- public API ------------------------------------------------------------

    def add(
        self,
        source: str | os.PathLike[str] | bytes,
        *,
        name: str | None = None,
        media_type: str | None = None,
        description: str = "",
        metadata: Mapping[str, str] | None = None,
        note_to: MemoryBackend | None = None,
    ) -> StoredFile:
        """Store a file and return its catalog entry.

        Args:
            source: A path or the raw bytes.
            name: Display name; defaults to the file name (required for bytes).
            media_type: Guessed from the name if omitted.
            description: What the file is; shown to Claude in the catalog.
            metadata: Extra string fields kept with the entry.
            note_to: A memory backend; writes a short note
                ``/memories/files/<id>.md`` so Claude finds the file from its memory.
        """
        if isinstance(source, bytes):
            if not name:
                raise ValueError("name is required when adding bytes")
            data = source
        else:
            path = Path(source)
            data = path.read_bytes()
            name = name or path.name
        kind = _media_type(name, media_type)
        if kind == "text/plain" or kind.startswith("text/"):
            try:
                data.decode("utf-8")
            except UnicodeDecodeError:
                raise ValueError(f"{name}: text files must be UTF-8") from None
        elif kind not in IMAGE_TYPES and kind != "application/pdf":
            raise ValueError(f"{name}: unsupported media type {kind!r}; use an image, a PDF or a text file")
        limit = self.max_image_bytes if kind in IMAGE_TYPES else self.max_document_bytes
        if len(data) > limit:
            raise ValueError(f"{name}: {len(data)} bytes exceed the limit of {limit} bytes")
        digest = hashlib.sha256(data).hexdigest()
        with self._lock:
            existing = self._entry(digest[:16])
            if existing is not None:
                entry = existing.model_copy(
                    update={
                        "name": name,
                        "description": description or existing.description,
                        "metadata": {**existing.metadata, **(metadata or {})},
                    }
                )
                self._save(entry, None)
            else:
                entry = StoredFile(
                    id=digest[:16],
                    name=name,
                    media_type=kind,
                    size=len(data),
                    sha256=digest,
                    created_at=datetime.now(timezone.utc),
                    description=description,
                    metadata=dict(metadata or {}),
                )
                self._save(entry, data)
        if note_to is not None:
            note_to.create(f"/memories/files/{entry.id}.md", self._note(entry))
        return entry

    def info(self, file_id: str) -> StoredFile:
        """The catalog entry of ``file_id``; raises ``KeyError`` if it is unknown."""
        with self._lock:
            entry = self._entry(file_id) if _ID.fullmatch(file_id) else None
        if entry is None:
            raise KeyError(file_id)
        return entry

    def get_bytes(self, file_id: str) -> bytes:
        """The original bytes of ``file_id``; raises ``KeyError`` if it is unknown."""
        with self._lock:
            data = self._data(file_id) if _ID.fullmatch(file_id) else None
        if data is None:
            raise KeyError(file_id)
        return data

    def list(self, query: str | None = None) -> list[StoredFile]:
        """All entries, newest first; ``query`` filters on name and description (case-insensitive)."""
        with self._lock:
            entries = self._entries()
        if query:
            needle = query.lower()
            entries = [e for e in entries if needle in e.name.lower() or needle in e.description.lower()]
        return sorted(entries, key=lambda e: e.created_at, reverse=True)

    def delete(self, file_id: str) -> None:
        """Remove a file; raises ``KeyError`` if it is unknown.

        A copy uploaded to the Files API (``files_api=``) is not deleted; delete it
        with the client first, using ``info(file_id).api_file_id``.
        """
        with self._lock:
            self.info(file_id)
            self._remove(file_id)

    def __contains__(self, file_id: object) -> bool:
        return isinstance(file_id, str) and bool(_ID.fullmatch(file_id)) and self._entry(file_id) is not None

    def __len__(self) -> int:
        return len(self._entries())

    def block(self, file_id: str, *, citations: bool = False) -> ContentBlock:
        """The document or image block for ``file_id``, e.g. to put a file into a prompt yourself."""
        entry = self.info(file_id)
        title = entry.name if entry.kind == "document" else None
        if self.files_api is not None:
            api_file_id = entry.api_file_id or self._upload(entry)
            return uploaded_file(api_file_id, kind=entry.kind, title=title, citations=citations)
        data = self.get_bytes(file_id)
        if entry.kind == "image":
            return image(data, media_type=entry.media_type)
        media_type = "application/pdf" if entry.media_type == "application/pdf" else "text/plain"
        return document(data, media_type=media_type, title=title, citations=citations)

    def _upload(self, entry: StoredFile) -> str:
        uploaded = self.files_api.files.upload(  # type: ignore[union-attr]
            file=(entry.name, self.get_bytes(entry.id), entry.media_type)
        )
        with self._lock:
            self._save(entry.model_copy(update={"api_file_id": uploaded.id}), None)
        return str(uploaded.id)

    @staticmethod
    def _note(entry: StoredFile) -> str:
        lines = [f"# File {entry.name}", "", f"- id: {entry.id} (load it with the files tool)"]
        lines.append(f"- type: {entry.media_type}, {_format_size(entry.size)}")
        if entry.description:
            lines.append(f"- description: {entry.description}")
        return "\n".join(lines) + "\n"

    # ---- tool -------------------------------------------------------------------

    def catalog(self, query: str | None = None) -> str:
        """The text the ``list`` command returns."""
        entries = self.list(query)
        if not entries:
            return "No stored files match." if query else "No files are stored."
        header = "id | name | type | size - description"
        return "\n".join([header, *(entry.line() for entry in entries)])

    def tool(self, *, name: str = FILES_TOOL_NAME, citations: bool = False) -> Tool:
        """The tool an agent offers for this store (``ClaudeAgent(files=store)`` adds it).

        Args:
            name: Tool name.
            citations: Load documents with citations enabled.
        """
        store = self

        def files(command: Literal["list", "load"], file_id: str = "", query: str = "") -> ToolContent:
            """List stored files (PDFs, images, text) or load one original into the conversation.

            Use list to see which files exist, then load a file only when you need its
            content. A loaded file stays in the conversation, so do not load it twice.

            Args:
                command: list shows the catalog; load returns the file itself.
                file_id: The id of the file to load (from list); required for load.
                query: Optional filter for list on file name and description.

            Returns:
                The catalog as text, or the loaded file as a document or image.
            """
            if command == "list":
                return [store.catalog(query or None)]
            entry = store._entry(_check_id(file_id))
            if entry is None:
                raise ToolResultError(f"Error: no file with id {file_id!r}; use list to see the files")
            header = f"File {entry.id}: {entry.name} ({entry.media_type})"
            return [header, store.block(entry.id, citations=citations)]

        return Tool(files, name=name)


class LocalFileStore(FileStore):
    """Files below a directory: ``root/blobs/<id>`` plus the catalog ``root/catalog.json``.

    The directory is created on demand (mode ``0o700``, files ``0o600``); writes are
    atomic. Use one ``root`` per user or project.

    Args:
        root: The directory that holds the files.
        files_api: See :class:`FileStore`.
        max_image_bytes: See :class:`FileStore`.
        max_document_bytes: See :class:`FileStore`.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        files_api: Any | None = None,
        max_image_bytes: int = 5_000_000,
        max_document_bytes: int = 20_000_000,
    ) -> None:
        super().__init__(
            files_api=files_api, max_image_bytes=max_image_bytes, max_document_bytes=max_document_bytes
        )
        self.root = Path(root)
        make_private_dirs(self.root / "blobs")
        os.chmod(self.root, 0o700)
        self._catalog_path = self.root / "catalog.json"

    def _read_catalog(self) -> dict[str, StoredFile]:
        if not self._catalog_path.is_file():
            return {}
        raw = json.loads(self._catalog_path.read_text(encoding="utf-8"))
        return {key: StoredFile.model_validate(value) for key, value in raw.items()}

    def _atomic_write(self, target: Path, data: bytes) -> None:
        temporary = target.parent / f".tmp-{uuid.uuid4().hex}"
        try:
            descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    def _write_catalog(self, catalog: dict[str, StoredFile]) -> None:
        payload = {key: entry.model_dump(mode="json") for key, entry in catalog.items()}
        self._atomic_write(self._catalog_path, json.dumps(payload, indent=1).encode("utf-8"))

    def _save(self, entry: StoredFile, data: bytes | None) -> None:
        if data is not None:
            self._atomic_write(self.root / "blobs" / entry.id, data)
        catalog = self._read_catalog()
        catalog[entry.id] = entry
        self._write_catalog(catalog)

    def _entry(self, file_id: str) -> StoredFile | None:
        return self._read_catalog().get(file_id)

    def _data(self, file_id: str) -> bytes | None:
        blob = self.root / "blobs" / file_id
        return blob.read_bytes() if file_id in self._read_catalog() and blob.is_file() else None

    def _entries(self) -> list[StoredFile]:
        return list(self._read_catalog().values())

    def _remove(self, file_id: str) -> None:
        catalog = self._read_catalog()
        catalog.pop(file_id, None)
        self._write_catalog(catalog)
        (self.root / "blobs" / file_id).unlink(missing_ok=True)


class SQLiteFileStore(FileStore):
    """Files in a SQLite database (Python's ``sqlite3``, no extra dependency).

    Each ``namespace`` is a separate store in the table ``stored_files``; the file
    can be the same database that ``SQLiteMemory`` or ``SQLiteIndex`` use.

    Args:
        database: Path of the database file (``":memory:"`` for a temporary one).
        namespace: Which store in the database this backend works on.
        files_api: See :class:`FileStore`.
        max_image_bytes: See :class:`FileStore`.
        max_document_bytes: See :class:`FileStore`.
    """

    def __init__(
        self,
        database: str | os.PathLike[str],
        *,
        namespace: str = "default",
        files_api: Any | None = None,
        max_image_bytes: int = 5_000_000,
        max_document_bytes: int = 20_000_000,
    ) -> None:
        super().__init__(
            files_api=files_api, max_image_bytes=max_image_bytes, max_document_bytes=max_document_bytes
        )
        if not namespace:
            raise ValueError("namespace must not be empty")
        self.namespace = namespace
        create_private_file(database)  # owner-only (0600) from the start; journals inherit it
        self._connection = sqlite3.connect(
            database, isolation_level=None, check_same_thread=False, timeout=30
        )
        # Deleted rows are overwritten instead of left in free pages of the file.
        self._connection.execute("PRAGMA secure_delete=ON")
        if str(database) != ":memory:":
            self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS stored_files ("
            " namespace TEXT NOT NULL, id TEXT NOT NULL, entry TEXT NOT NULL, data BLOB NOT NULL,"
            " PRIMARY KEY (namespace, id))"
        )

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self._connection.close()

    def __enter__(self) -> SQLiteFileStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _save(self, entry: StoredFile, data: bytes | None) -> None:
        if data is None:
            self._connection.execute(
                "UPDATE stored_files SET entry = ? WHERE namespace = ? AND id = ?",
                (entry.model_dump_json(), self.namespace, entry.id),
            )
        else:
            self._connection.execute(
                "INSERT OR REPLACE INTO stored_files (namespace, id, entry, data) VALUES (?, ?, ?, ?)",
                (self.namespace, entry.id, entry.model_dump_json(), data),
            )

    def _entry(self, file_id: str) -> StoredFile | None:
        row = self._connection.execute(
            "SELECT entry FROM stored_files WHERE namespace = ? AND id = ?", (self.namespace, file_id)
        ).fetchone()
        return None if row is None else StoredFile.model_validate_json(row[0])

    def _data(self, file_id: str) -> bytes | None:
        row = self._connection.execute(
            "SELECT data FROM stored_files WHERE namespace = ? AND id = ?", (self.namespace, file_id)
        ).fetchone()
        return None if row is None else bytes(row[0])

    def _entries(self) -> list[StoredFile]:
        rows: Iterator[tuple[str]] = self._connection.execute(
            "SELECT entry FROM stored_files WHERE namespace = ?", (self.namespace,)
        )
        return [StoredFile.model_validate_json(row[0]) for row in rows]

    def _remove(self, file_id: str) -> None:
        self._connection.execute(
            "DELETE FROM stored_files WHERE namespace = ? AND id = ?", (self.namespace, file_id)
        )
