"""Retrieval-augmented generation (RAG): a search index the model queries as a tool.

An :class:`Index` holds text chunks of a larger corpus (handbooks, reports,
tickets). With ``index=`` an agent of any provider offers the model the tool
``search``. Its hits are neutral search results: Anthropic receives them as
``search_result`` blocks (so the answer carries API-native citations); OpenAI,
Gemini and compatible endpoints receive them as attributed text, because their
APIs have no search-result blocks::

    from orkexa import Agent, SQLiteIndex, citations, retrieval_sources

    index = SQLiteIndex("agent.db", namespace="handbook")
    index.add_file("handbook.md")

    agent = Agent(model="gemini:gemini-3-pro", index=index)
    answer = agent.ask("How long is the warranty?")
    for source in retrieval_sources(agent):  # hits the search tool really returned
        print(source.source, source.title)
    for cite in citations(agent.last_response):  # API-native citations, if any
        print(cite.source, cite.cited_text)

Three kinds of source information are kept apart: :func:`citations` returns
citations the provider API attached (``kind="api_native"``),
:func:`retrieval_sources` returns the passages Orkexa's search tool actually
delivered in the last turn (``kind="retrieval"``), and source claims the model
writes into its text are neither and are never parsed as sources.

:class:`SQLiteIndex` uses SQLite's built-in FTS5 full-text search (BM25 ranking)
and needs no extra package. Pass ``embedder=`` for vector search with
`sqlite-vec <https://github.com/asg017/sqlite-vec>`_ (``pip install
"orkexa[vector]"``); the index then combines both rankings (hybrid search).
PDF text extraction uses ``pypdf`` (``pip install "orkexa[pdf]"``).

Memory and RAG complement each other: memory is the notebook the model writes
itself, the index is the library your application fills. With ``store=`` a
chunk remembers the The :class:`~orkexa.FileStore` id of its original, so
the model can load the whole file after a hit.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
import io
import json
import os
from pathlib import Path
import re
import sqlite3
import struct
import threading
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from .content import ContentBlock, ToolContent
from .core.types import Citation, Message, ModelResponse, SearchResultPart, TextPart, ToolResultPart
from .tools import Tool

if TYPE_CHECKING:
    from .files import FileStore

__all__ = [
    "SEARCH_TOOL_NAME",
    "Chunk",
    "Citation",
    "Embedder",
    "Hit",
    "Index",
    "SQLiteIndex",
    "citations",
    "retrieval_sources",
    "split_text",
]

SEARCH_TOOL_NAME = "search"
Embedder = Callable[[list[str]], Sequence[Sequence[float]]]
"""Turns texts into vectors, e.g. a Voyage AI or sentence-transformers call."""

_WORD = re.compile(r"\w+", re.UNICODE)
_RRF_K = 60


class Chunk(BaseModel):
    """A piece of text to index."""

    text: str
    source: str
    """Where the text comes from: a file name, URL or id; shown in citations."""
    title: str | None = None
    file_id: str | None = None
    """The FileStore id of the original, if it is stored."""
    metadata: dict[str, str] = Field(default_factory=dict)


class Hit(Chunk):
    """A search result: a stored chunk with its id and score (higher is better)."""

    id: int
    score: float

    def block(self, *, citations: bool = True) -> ContentBlock:
        """This hit as a ``search_result`` content block."""
        content: list[ContentBlock] = [{"type": "text", "text": self.text}]
        if self.file_id:
            note = f"(Original: file {self.file_id}, loadable with the files tool)"
            content.append({"type": "text", "text": note})
        return {
            "type": "search_result",
            "source": self.source,
            "title": self.title or self.source,
            "content": content,
            "citations": {"enabled": citations},
        }


def citations(response: Any) -> list[Citation]:
    """The API-native citations of a response (e.g. ``agent.last_response``).

    Accepts a normalized :class:`~orkexa.core.types.ModelResponse` (every provider:
    Anthropic search-result citations, OpenAI URL/file annotations, ...) or a native
    Anthropic message (only ``search_result_location`` citations, as in ``claude-agent``).
    """
    if isinstance(response, ModelResponse):
        return response.citations
    found: list[Citation] = []
    for block in getattr(response, "content", None) or []:
        for cite in getattr(block, "citations", None) or []:
            if getattr(cite, "type", None) == "search_result_location":
                found.append(
                    Citation(
                        source=cite.source,
                        title=getattr(cite, "title", None),
                        cited_text=cite.cited_text,
                        search_result_index=getattr(cite, "search_result_index", None),
                    )
                )
    return found


def retrieval_sources(conversation: Any, *, tool_name: str | None = None) -> list[Citation]:
    """The search hits the ``search`` tool returned in the last turn (every provider).

    ``conversation`` is an agent (``Agent``, ``ClaudeAgent``, ...) or a list of
    neutral :class:`~orkexa.Message` objects. The result lists each delivered
    passage once, as a :class:`Citation` with ``kind="retrieval"``: verified
    sources the model *received*, not a claim that the answer used them.
    """
    name = tool_name or SEARCH_TOOL_NAME
    if hasattr(conversation, "conversation"):
        messages: list[Message] = conversation.conversation()
    else:
        messages = list(conversation)
    start = 0
    for index, message in enumerate(messages):
        if message.role == "user" and any(not isinstance(p, ToolResultPart) for p in message.parts()):
            start = index
    found: list[Citation] = []
    seen: set[tuple[str, str]] = set()
    for message in messages[start:]:
        for part in message.parts():
            if not isinstance(part, ToolResultPart) or part.is_error or part.name not in (name, None):
                continue
            if not isinstance(part.content, list):
                continue
            for item in part.content:
                if not isinstance(item, SearchResultPart):
                    continue
                text = "\n".join(p.text for p in item.content if isinstance(p, TextPart))
                key = (item.source, text)
                if key in seen:
                    continue
                seen.add(key)
                found.append(
                    Citation(source=item.source, title=item.title, cited_text=text, kind="retrieval")
                )
    return found


def split_text(text: str, *, max_chars: int = 1500, overlap: int = 200) -> list[str]:
    """Split text into chunks of at most about ``max_chars`` at paragraph boundaries.

    Paragraphs longer than ``max_chars`` are split at sentence, then word
    boundaries. Each chunk after the first starts with up to ``overlap``
    characters of the previous one, so a passage cut at a boundary stays findable.
    """
    if max_chars < 100 or not 0 <= overlap < max_chars // 2:
        raise ValueError("max_chars must be at least 100 and overlap below max_chars / 2")
    pieces: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if paragraph:
            pieces.extend(_split_long(paragraph, max_chars - overlap))
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}\n\n{piece}" if current else piece
        if len(candidate) <= max_chars - overlap or not current:
            current = candidate
        else:
            chunks.append(current)
            current = piece
    if current:
        chunks.append(current)
    if overlap:
        for position in range(len(chunks) - 1, 0, -1):
            tail = chunks[position - 1][-overlap:]
            cut = tail.find(" ")
            tail = tail[cut + 1 :] if 0 <= cut < len(tail) - 1 else tail
            chunks[position] = f"{tail.strip()} {chunks[position]}"
    return chunks


def _split_long(paragraph: str, limit: int) -> list[str]:
    if len(paragraph) <= limit:
        return [paragraph]
    parts: list[str] = []
    current = ""
    for unit in re.split(r"(?<=[.!?])\s+", paragraph):
        words = [unit] if len(unit) <= limit else unit.split()
        for word in words:
            while len(word) > limit:  # a single overlong token
                parts.append(word[:limit])
                word = word[limit:]
            candidate = f"{current} {word}" if current else word
            if len(candidate) <= limit:
                current = candidate
            else:
                parts.append(current)
                current = word
    if current:
        parts.append(current)
    return parts


def _pdf_pages(data: bytes) -> list[str]:
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found,unused-ignore]
    except ImportError:
        raise ImportError(
            'Indexing PDFs needs pypdf: pip install "orkexa[pdf]" (formerly "claude-agent[pdf]")'
        ) from None
    return [page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages]


class Index(ABC):
    """A searchable collection of text chunks; subclass it for other search backends.

    Subclasses implement :meth:`add_chunks`, :meth:`search` and :meth:`delete`;
    chunking, file ingestion and the tool come from this class.

    Args:
        max_chars: Chunk size for :meth:`add` and :meth:`add_file`.
        overlap: Chunk overlap, see :func:`split_text`.
    """

    def __init__(self, *, max_chars: int = 1500, overlap: int = 200) -> None:
        split_text("", max_chars=max_chars, overlap=overlap)  # validate
        self.max_chars = max_chars
        self.overlap = overlap

    @abstractmethod
    def add_chunks(self, chunks: Iterable[Chunk]) -> list[int]:
        """Store ready-made chunks and return their ids."""

    @abstractmethod
    def search(self, query: str, k: int = 5) -> list[Hit]:
        """The ``k`` best chunks for ``query``, best first."""

    @abstractmethod
    def delete(self, source: str) -> int:
        """Remove all chunks of ``source``; returns how many were removed."""

    def add(
        self,
        text: str,
        *,
        source: str,
        title: str | None = None,
        file_id: str | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> list[int]:
        """Split ``text`` into chunks and index them; returns the chunk ids."""
        return self.add_chunks(
            Chunk(text=piece, source=source, title=title, file_id=file_id, metadata=dict(metadata or {}))
            for piece in split_text(text, max_chars=self.max_chars, overlap=self.overlap)
        )

    def add_file(
        self,
        path: str | os.PathLike[str],
        *,
        store: FileStore | None = None,
        source: str | None = None,
        title: str | None = None,
        description: str = "",
        metadata: Mapping[str, str] | None = None,
    ) -> list[int]:
        """Index a text, Markdown or PDF file (PDFs page by page, with ``page`` metadata).

        With ``store=`` the original is also stored there, and every chunk keeps
        its file id, so Claude can load the whole file after a hit. Indexing the
        same ``source`` again replaces its chunks.

        Args:
            path: The file.
            store: Optional FileStore for the original.
            source: Citation source; defaults to the file name.
            title: Citation title; defaults to the file name.
            description: Description for the FileStore entry.
            metadata: Extra string fields for every chunk.
        """
        file = Path(path)
        data = file.read_bytes()
        source = source or file.name
        title = title or file.name
        file_id = store.add(data, name=file.name, description=description).id if store is not None else None
        self.delete(source)
        if file.suffix.lower() == ".pdf":
            ids: list[int] = []
            for number, page in enumerate(_pdf_pages(data), start=1):
                if page.strip():
                    extra = {**(metadata or {}), "page": str(number)}
                    page_title = f"{title}, p. {number}"
                    ids += self.add(page, source=source, title=page_title, file_id=file_id, metadata=extra)
            return ids
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError(f"{file.name}: only UTF-8 text and PDF files can be indexed") from None
        return self.add(text, source=source, title=title, file_id=file_id, metadata=metadata)

    def blocks(self, query: str, k: int = 5, *, citations: bool = True) -> list[ContentBlock]:
        """Search results as ``search_result`` blocks to put into a prompt yourself.

        Example: ``agent.ask([*index.blocks(question), question])``.
        """
        return [hit.block(citations=citations) for hit in self.search(query, k)]

    def tool(self, *, name: str = SEARCH_TOOL_NAME, max_k: int = 20, citations: bool = True) -> Tool:
        """The tool an agent offers for this index (``ClaudeAgent(index=index)`` adds it).

        Args:
            name: Tool name.
            max_k: Upper bound for the number of results Claude may request.
            citations: Enable citations on the returned search results.
        """
        index = self

        def search(query: str, k: int = 5) -> ToolContent:
            """Search the knowledge base and return the most relevant passages with their sources.

            Use specific keywords from the question. Search again with other words if
            the results do not answer it. Cite the passages you use.

            Args:
                query: What to look for, in keywords or a short question.
                k: How many passages to return (1 to 20).

            Returns:
                The best matching passages as search results, or a note that nothing matched.
            """
            hits = index.search(query, max(1, min(k, max_k)))
            if not hits:
                return ["No results found."]
            return [hit.block(citations=citations) for hit in hits]

        return Tool(search, name=name)


def _fts_query(query: str) -> str:
    """Natural language -> an FTS5 query matching any word or word prefix ("refund" finds "refunds").

    BM25 then ranks chunks that match more and rarer words higher.
    """
    words = dict.fromkeys(word.lower() for word in _WORD.findall(query))
    return " OR ".join(f'"{word}"*' for word in words)


class SQLiteIndex(Index):
    """A search index in a SQLite database: FTS5 full-text search, optionally plus vectors.

    Each ``namespace`` is a separate index (tables ``index_chunks``/``index_fts``),
    so one database file can hold several corpora next to ``SQLiteMemory`` and
    ``SQLiteFileStore``.

    Args:
        database: Path of the database file (``":memory:"`` for a temporary one).
        namespace: Which index in the database this object works on.
        embedder: Optional function ``list[str] -> list[vector]`` for documents;
            enables vector search with sqlite-vec and hybrid ranking.
        query_embedder: Optional separate function for queries (some embedding
            models embed queries differently); defaults to ``embedder``.
        tokenizer: The FTS5 tokenizer, e.g. ``"porter unicode61"`` for English stemming.
            Only used when the tables are created.
        max_chars: Chunk size, see :func:`split_text`.
        overlap: Chunk overlap, see :func:`split_text`.
    """

    def __init__(
        self,
        database: str | os.PathLike[str],
        *,
        namespace: str = "default",
        embedder: Embedder | None = None,
        query_embedder: Embedder | None = None,
        tokenizer: str = "unicode61 remove_diacritics 2",
        max_chars: int = 1500,
        overlap: int = 200,
    ) -> None:
        super().__init__(max_chars=max_chars, overlap=overlap)
        if not namespace:
            raise ValueError("namespace must not be empty")
        if not re.fullmatch(r"[\w ]+", tokenizer):
            raise ValueError(f"Invalid FTS5 tokenizer: {tokenizer!r}")
        self.namespace = namespace
        self.embedder = embedder
        self.query_embedder = query_embedder or embedder
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            database, isolation_level=None, check_same_thread=False, timeout=30
        )
        # Deleted rows are overwritten instead of left in free pages of the file.
        self._connection.execute("PRAGMA secure_delete=ON")
        if str(database) != ":memory:":
            self._connection.execute("PRAGMA journal_mode=WAL")
        try:
            self._connection.executescript(
                "CREATE TABLE IF NOT EXISTS index_chunks ("
                " id INTEGER PRIMARY KEY, namespace TEXT NOT NULL, source TEXT NOT NULL,"
                " title TEXT, file_id TEXT, metadata TEXT NOT NULL, text TEXT NOT NULL);"
                "CREATE INDEX IF NOT EXISTS index_chunks_source ON index_chunks (namespace, source);"
                "CREATE VIRTUAL TABLE IF NOT EXISTS index_fts USING fts5("
                f" text, title, content='index_chunks', content_rowid='id', tokenize='{tokenizer}');"
                "CREATE TABLE IF NOT EXISTS index_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
            )
        except sqlite3.OperationalError as exc:
            if "fts5" in str(exc).lower():
                raise RuntimeError("This Python's sqlite3 lacks FTS5, which SQLiteIndex needs") from exc
            raise
        if embedder is not None:
            self._load_sqlite_vec()

    def _load_sqlite_vec(self) -> None:
        try:
            import sqlite_vec  # type: ignore[import-not-found,import-untyped,unused-ignore]
        except ImportError:
            raise ImportError(
                "Vector search needs sqlite-vec: "
                'pip install "orkexa[vector]" (formerly "claude-agent[vector]")'
            ) from None
        try:
            self._connection.enable_load_extension(True)
            sqlite_vec.load(self._connection)
            self._connection.enable_load_extension(False)
        except AttributeError:
            raise RuntimeError(
                "This Python's sqlite3 cannot load extensions, which sqlite-vec needs"
            ) from None

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self._connection.close()

    def optimize(self) -> None:
        """Merge the full-text index, erasing the terms of deleted chunks from the file.

        :meth:`delete` removes chunks from search results at once, but FTS5 keeps the
        terms of deleted rows in its index segments until they are merged. Call this
        after deleting data that must be erased. It rewrites the full-text index of
        every namespace in the database, so it takes time proportional to its size.
        """
        with self._lock:
            self._connection.execute("INSERT INTO index_fts (index_fts) VALUES ('optimize')")

    def __enter__(self) -> SQLiteIndex:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __len__(self) -> int:
        row = self._connection.execute(
            "SELECT count(*) FROM index_chunks WHERE namespace = ?", (self.namespace,)
        ).fetchone()
        return int(row[0])

    def sources(self) -> list[str]:
        """All indexed sources."""
        rows = self._connection.execute(
            "SELECT DISTINCT source FROM index_chunks WHERE namespace = ? ORDER BY source", (self.namespace,)
        )
        return [str(row[0]) for row in rows]

    # ---- vectors ----------------------------------------------------------------

    def _vector_table(self, dimensions: int) -> str:
        row = self._connection.execute("SELECT value FROM index_meta WHERE key = 'dimensions'").fetchone()
        if row is None:
            self._connection.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS index_vec USING vec0("
                f" namespace TEXT partition key, embedding float[{dimensions}] distance_metric=cosine)"
            )
            self._connection.execute(
                "INSERT INTO index_meta (key, value) VALUES ('dimensions', ?)", (str(dimensions),)
            )
        elif int(row[0]) != dimensions:
            raise ValueError(f"The index holds {row[0]}-dimensional vectors, the embedder gave {dimensions}")
        return "index_vec"

    @staticmethod
    def _vector(values: Sequence[float]) -> bytes:
        return struct.pack(f"{len(values)}f", *values)

    def _has_vectors(self) -> bool:
        row = self._connection.execute("SELECT value FROM index_meta WHERE key = 'dimensions'").fetchone()
        return row is not None

    # ---- Index API ----------------------------------------------------------------

    def add_chunks(self, chunks: Iterable[Chunk]) -> list[int]:
        chunks = [chunk for chunk in chunks if chunk.text.strip()]
        if not chunks:
            return []
        vectors = list(self.embedder([chunk.text for chunk in chunks])) if self.embedder is not None else None
        if vectors is not None and len(vectors) != len(chunks):
            raise ValueError("The embedder must return one vector per text")
        ids: list[int] = []
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                for position, chunk in enumerate(chunks):
                    cursor = self._connection.execute(
                        "INSERT INTO index_chunks (namespace, source, title, file_id, metadata, text)"
                        " VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            self.namespace,
                            chunk.source,
                            chunk.title,
                            chunk.file_id,
                            json.dumps(chunk.metadata),
                            chunk.text,
                        ),
                    )
                    chunk_id = int(cursor.lastrowid or 0)
                    self._connection.execute(
                        "INSERT INTO index_fts (rowid, text, title) VALUES (?, ?, ?)",
                        (chunk_id, chunk.text, chunk.title or ""),
                    )
                    if vectors is not None:
                        table = self._vector_table(len(vectors[position]))
                        self._connection.execute(
                            f"INSERT INTO {table} (rowid, namespace, embedding) VALUES (?, ?, ?)",
                            (chunk_id, self.namespace, self._vector(vectors[position])),
                        )
                    ids.append(chunk_id)
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
            self._connection.execute("COMMIT")
        return ids

    def delete(self, source: str) -> int:
        with self._lock:
            rows = self._connection.execute(
                "SELECT id, text, title FROM index_chunks WHERE namespace = ? AND source = ?",
                (self.namespace, source),
            ).fetchall()
            if not rows:
                return 0
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                for chunk_id, text, title in rows:
                    self._connection.execute(
                        "INSERT INTO index_fts (index_fts, rowid, text, title) VALUES ('delete', ?, ?, ?)",
                        (chunk_id, text, title or ""),
                    )
                    if self._has_vectors():
                        self._connection.execute("DELETE FROM index_vec WHERE rowid = ?", (chunk_id,))
                    self._connection.execute("DELETE FROM index_chunks WHERE id = ?", (chunk_id,))
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
            self._connection.execute("COMMIT")
        return len(rows)

    def search(self, query: str, k: int = 5) -> list[Hit]:
        if k < 1:
            raise ValueError("k must be positive")
        candidates = max(k * 4, 20)
        with self._lock:
            ranked = [self._fts_ids(query, candidates)]
            if self.query_embedder is not None and self._has_vectors():
                ranked.append(self._vector_ids(query, candidates))
            scores: dict[int, float] = {}
            for ranking in ranked:
                for rank, chunk_id in enumerate(ranking):
                    scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (_RRF_K + rank + 1)
            best = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)[:k]
            return [self._hit(chunk_id, scores[chunk_id]) for chunk_id in best]

    def _fts_ids(self, query: str, limit: int) -> list[int]:
        expression = _fts_query(query)
        if not expression:
            return []
        rows = self._connection.execute(
            "SELECT index_fts.rowid FROM index_fts JOIN index_chunks ON index_chunks.id = index_fts.rowid"
            " WHERE index_fts MATCH ? AND index_chunks.namespace = ? ORDER BY bm25(index_fts) LIMIT ?",
            (expression, self.namespace, limit),
        )
        return [int(row[0]) for row in rows]

    def _vector_ids(self, query: str, limit: int) -> list[int]:
        assert self.query_embedder is not None
        vector = next(iter(self.query_embedder([query])))
        rows = self._connection.execute(
            "SELECT rowid FROM index_vec WHERE embedding MATCH ? AND k = ? AND namespace = ?"
            " ORDER BY distance",
            (self._vector(vector), limit, self.namespace),
        )
        return [int(row[0]) for row in rows]

    def _hit(self, chunk_id: int, score: float) -> Hit:
        source, title, file_id, metadata, text = self._connection.execute(
            "SELECT source, title, file_id, metadata, text FROM index_chunks WHERE id = ?", (chunk_id,)
        ).fetchone()
        return Hit(
            id=chunk_id,
            score=score,
            source=source,
            title=title,
            file_id=file_id,
            metadata=json.loads(metadata),
            text=text,
        )
