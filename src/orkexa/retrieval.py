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
import math
import os
from pathlib import Path
import re
import sqlite3
import struct
import threading
from typing import TYPE_CHECKING, Any, Literal, get_args

from pydantic import BaseModel, Field

from ._storage import create_private_file
from .content import ContentBlock, ToolContent
from .core.types import Citation, Message, ModelResponse, SearchResultPart, TextPart, ToolResultPart
from .errors import OrkexaError
from .tools import Tool

if TYPE_CHECKING:
    from .embeddings import EmbeddingIdentity, EmbeddingModel
    from .files import FileStore

__all__ = [
    "HYBRID_FUSIONS",
    "LEXICAL_WEIGHT",
    "RETRIEVAL_MODES",
    "SEARCH_TOOL_NAME",
    "Chunk",
    "Citation",
    "Embedder",
    "Hit",
    "HybridFusion",
    "Index",
    "RetrievalError",
    "RetrievalMode",
    "SQLiteIndex",
    "VectorStatus",
    "citations",
    "fuse_hybrid",
    "lexical_evidence",
    "query_words",
    "retrieval_sources",
    "split_text",
    "word_weight",
]

SEARCH_TOOL_NAME = "search"
Embedder = Callable[[list[str]], Sequence[Sequence[float]]]
"""Turns texts into vectors, e.g. a Voyage AI or sentence-transformers call."""

RetrievalMode = Literal["fts", "vector", "hybrid"]
"""How an index ranks: ``fts`` full text (BM25), ``vector`` embeddings only, ``hybrid`` both."""
RETRIEVAL_MODES: tuple[str, ...] = get_args(RetrievalMode)

HybridFusion = Literal["evidence", "rrf"]
"""How ``hybrid`` combines its two rankings: ``evidence`` (default) weighs the full-text side by how
much of the query a chunk contains (:func:`fuse_hybrid`); ``rrf`` is the earlier reciprocal rank fusion."""
HYBRID_FUSIONS: tuple[str, ...] = get_args(HybridFusion)

_WORD = re.compile(r"\w+", re.UNICODE)
_RRF_K = 60
_MAX_SOURCES = 50_000
#: The share of the full-text evidence in ``evidence`` fusion (the vector side gets the rest).
LEXICAL_WEIGHT = 0.5


class RetrievalError(OrkexaError, RuntimeError):
    """A search could not run as requested; never answered with another mode instead.

    ``code``: ``mode_unsupported`` (the index cannot rank this way), ``vector_unavailable``
    (no embedding model or no vectors), ``reindex_required`` (the vectors belong to another
    embedding model), ``vector_incomplete`` (not every chunk has a vector yet).
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def check_mode(mode: str) -> RetrievalMode:
    if mode not in RETRIEVAL_MODES:
        raise ValueError(f"Unknown retrieval mode {mode!r}; use one of {', '.join(RETRIEVAL_MODES)}")
    return mode  # type: ignore[return-value]


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
    ranks: dict[str, int] = Field(default_factory=dict)
    """1-based rank of this chunk in each ranking that found it (``fts``, ``vector``)."""

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
    Anthropic message (only ``search_result_location`` citations).
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
        raise ImportError('Indexing PDFs needs pypdf: pip install "orkexa[pdf]"') from None
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

    def retrieve(
        self,
        query: str,
        k: int = 5,
        *,
        mode: RetrievalMode | None = None,
        sources: Iterable[str] | None = None,
    ) -> list[Hit]:
        """The ``k`` best chunks, optionally ranked another way and limited to ``sources``.

        The base implementation supports only full-text ranking (``mode`` ``None`` or
        ``"fts"``) and filters ``sources`` after :meth:`search`, fetching more candidates
        until it has ``k`` hits. Indexes that can filter before ranking override this.
        """
        if mode not in (None, "fts"):
            raise RetrievalError("mode_unsupported", f"{type(self).__name__} supports only full-text search")
        if sources is None:
            return self.search(query, k)
        allowed = set(sources)
        if not allowed:
            return []
        limit = max(k * 20, 50)
        while True:
            hits = self.search(query, limit)
            found = [hit for hit in hits if hit.source in allowed]
            if len(found) >= k or len(hits) < limit or limit >= 20_000:
                return found[:k]
            limit *= 4

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

    def tool(
        self,
        *,
        name: str = SEARCH_TOOL_NAME,
        max_k: int = 20,
        citations: bool = True,
        mode: RetrievalMode | None = None,
        sources: Iterable[str] | None = None,
    ) -> Tool:
        """The tool an agent offers for this index (``ClaudeAgent(index=index)`` adds it).

        Args:
            name: Tool name.
            max_k: Upper bound for the number of results Claude may request.
            citations: Enable citations on the returned search results.
            mode: How the tool ranks (see :meth:`retrieve`); ``None``: the index's default.
            sources: Limit the tool to these sources (checked before ranking where the index can).
        """
        index = self
        allowed = None if sources is None else list(sources)
        if mode is not None:
            check_mode(mode)

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
            count = max(1, min(k, max_k))
            if mode is None and allowed is None:
                hits = index.search(query, count)
            else:
                hits = index.retrieve(query, count, mode=mode, sources=allowed)
            if not hits:
                return ["No results found."]
            return [hit.block(citations=citations) for hit in hits]

        return Tool(search, name=name)


def _fts_query(query: str) -> str:
    """Natural language -> an FTS5 query matching any word or word prefix ("refund" finds "refunds").

    BM25 then ranks chunks that match more and rarer words higher.
    """
    return " OR ".join(f'"{word}"*' for word in query_words(query))


def query_words(query: str) -> list[str]:
    """The distinct lower-case words of ``query`` that full-text search matches (each as a prefix)."""
    return list(dict.fromkeys(word.lower() for word in _WORD.findall(query)))


def word_weight(matching: int, total: int) -> float:
    """The inverse document frequency of a query word that ``matching`` of ``total`` chunks contain.

    The BM25 form ``ln(1 + (N - n + 0.5) / (n + 0.5))``: always positive, highest for a word no chunk
    contains. Such a word cannot be found, so it lowers the share of every chunk (see
    :func:`lexical_evidence`): a query in another language than the corpus has little lexical evidence.
    """
    matching = max(0, min(matching, total))
    return math.log(1.0 + (total - matching + 0.5) / (matching + 0.5))


def lexical_evidence(weights: Mapping[str, float], found: Mapping[int, Iterable[str]]) -> dict[int, float]:
    """How strongly full text supports each chunk: the squared IDF-weighted share of the query it contains.

    ``weights`` maps every query word to its :func:`word_weight`; ``found`` maps chunk ids to the query
    words they contain. A chunk with all query words scores 1, one that shares a single common or
    accidental word (typical of a query in another language: ``de``, ``la``, a number) almost 0. Squaring
    keeps such partial matches from outweighing the vector ranking while a match of the whole query
    still counts fully.
    """
    mass = sum(weights.values())
    if mass <= 0:
        return {chunk_id: 0.0 for chunk_id in found}
    return {
        chunk_id: (sum(weights.get(word, 0.0) for word in set(words)) / mass) ** 2
        for chunk_id, words in found.items()
    }


def fuse_hybrid(
    vector: Sequence[tuple[int, float]],
    lexical: Mapping[int, float],
    *,
    lexical_weight: float = LEXICAL_WEIGHT,
) -> list[tuple[int, float]]:
    """Evidence-weighted hybrid ranking: ``(1 - w) * vector + w * lexical`` per chunk, best first.

    ``vector`` holds the ``(chunk id, cosine similarity)`` candidates of the vector ranking; their
    similarities are min-max normalized over these candidates (best 1, last 0; one candidate or equal
    similarities: 1), so the scale of an embedding model does not matter. ``lexical`` holds the
    :func:`lexical_evidence` of chunks in [0, 1]. A chunk missing from one side counts 0 there. A
    query without lexical evidence keeps the vector order; a chunk that contains the whole query rises
    above vector neighbours that contain only part of it. Ties go to the lower chunk id.
    """
    if not 0.0 <= lexical_weight <= 1.0:
        raise ValueError("lexical_weight must be between 0 and 1")
    similarities = dict(vector)
    low = min(similarities.values(), default=0.0)
    spread = max(similarities.values(), default=0.0) - low
    scores: dict[int, float] = {}
    for chunk_id in {*similarities, *lexical}:
        similarity = similarities.get(chunk_id)
        normalized = 0.0 if similarity is None else (1.0 if spread <= 0 else (similarity - low) / spread)
        evidence = min(max(lexical.get(chunk_id, 0.0), 0.0), 1.0)
        scores[chunk_id] = round((1.0 - lexical_weight) * normalized + lexical_weight * evidence, 9)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


class VectorStatus(BaseModel):
    """The vector state of one namespace of a :class:`SQLiteIndex`.

    ``active`` is the embedding identity searches use; ``building`` a reindex in progress
    (resumable). ``embedded`` counts chunks with a vector in the active space.
    """

    chunks: int
    active: dict[str, Any] | None = None
    embedded: int = 0
    building: dict[str, Any] | None = None
    building_embedded: int = 0
    legacy: bool = False
    """Vectors of an ``embedder=`` callable (one shared table, no model identity)."""

    @property
    def complete(self) -> bool:
        return self.active is not None and self.embedded >= self.chunks


class SQLiteIndex(Index):
    """A search index in a SQLite database: FTS5 full-text search, optionally plus vectors.

    Each ``namespace`` is a separate index (tables ``index_chunks``/``index_fts``),
    so one database file can hold several corpora next to ``SQLiteMemory`` and
    ``SQLiteFileStore``.

    Vectors come from an :class:`~orkexa.embeddings.EmbeddingModel` (``embeddings=``):
    each namespace keeps its vectors in a *space* bound to the model's identity, ranks
    them by exact cosine distance (sqlite-vec) and limits candidates to the requested
    sources before ranking. Changing the model needs :meth:`reindex`; until it has
    finished, searches keep using the old, complete space. The older ``embedder=``
    callables use sqlite-vec's ``vec0`` table as before.

    Args:
        database: Path of the database file (``":memory:"`` for a temporary one).
        namespace: Which index in the database this object works on.
        embeddings: Optional embedding model (enables ``vector`` and ``hybrid`` search).
        mode: Default ranking of :meth:`search` and the tool: ``fts`` (default without
            embeddings), ``vector`` or ``hybrid`` (default with embeddings).
        embedder: Older alternative to ``embeddings``: a function ``list[str] -> list[vector]``.
        query_embedder: Optional separate function for queries; defaults to ``embedder``.
        tokenizer: The FTS5 tokenizer, e.g. ``"porter unicode61"`` for English stemming.
            Only used when the tables are created.
        max_chars: Chunk size, see :func:`split_text`.
        overlap: Chunk overlap, see :func:`split_text`.
        fusion: How ``hybrid`` combines full text and vectors: ``evidence`` (default,
            :func:`fuse_hybrid`) or ``rrf`` (reciprocal rank fusion, k=60, the earlier behavior).
        lexical_weight: The full-text share of ``evidence`` fusion (0: vector order only).
    """

    def __init__(
        self,
        database: str | os.PathLike[str],
        *,
        namespace: str = "default",
        embeddings: EmbeddingModel | None = None,
        mode: RetrievalMode | None = None,
        embedder: Embedder | None = None,
        query_embedder: Embedder | None = None,
        tokenizer: str = "unicode61 remove_diacritics 2",
        max_chars: int = 1500,
        overlap: int = 200,
        fusion: HybridFusion = "evidence",
        lexical_weight: float = LEXICAL_WEIGHT,
    ) -> None:
        super().__init__(max_chars=max_chars, overlap=overlap)
        if not namespace:
            raise ValueError("namespace must not be empty")
        if fusion not in HYBRID_FUSIONS:
            raise ValueError(f"fusion must be one of {', '.join(HYBRID_FUSIONS)}, not {fusion!r}")
        if not 0.0 <= lexical_weight <= 1.0:
            raise ValueError("lexical_weight must be between 0 and 1")
        self.fusion: HybridFusion = fusion
        self.lexical_weight = lexical_weight
        if not re.fullmatch(r"[\w ]+", tokenizer):
            raise ValueError(f"Invalid FTS5 tokenizer: {tokenizer!r}")
        if embeddings is not None and embedder is not None:
            raise ValueError("Pass either embeddings= or embedder=, not both")
        self.namespace = namespace
        self.embeddings = embeddings
        self.embedder = embedder
        self.query_embedder = query_embedder or embedder
        has_vectors = embeddings is not None or embedder is not None
        self.mode: RetrievalMode = check_mode(mode) if mode else ("hybrid" if has_vectors else "fts")
        self._lock = threading.RLock()
        self._vec_loaded = False
        create_private_file(database)  # owner-only (0600) from the start; journals inherit it
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
                "CREATE TABLE IF NOT EXISTS index_spaces ("
                " namespace TEXT NOT NULL, space TEXT NOT NULL, identity TEXT NOT NULL,"
                " dimensions INTEGER NOT NULL, state TEXT NOT NULL, PRIMARY KEY (namespace, space));"
                "CREATE TABLE IF NOT EXISTS index_vectors ("
                " space TEXT NOT NULL, chunk_id INTEGER NOT NULL, embedding BLOB NOT NULL,"
                " PRIMARY KEY (space, chunk_id));"
                "CREATE INDEX IF NOT EXISTS index_vectors_chunk ON index_vectors (chunk_id);"
            )
        except sqlite3.OperationalError as exc:
            if "fts5" in str(exc).lower():
                raise RuntimeError("This Python's sqlite3 lacks FTS5, which SQLiteIndex needs") from exc
            raise
        if has_vectors:
            self._load_sqlite_vec()

    def _load_sqlite_vec(self) -> None:
        if self._vec_loaded:
            return
        try:
            import sqlite_vec  # type: ignore[import-not-found,import-untyped,unused-ignore]
        except ImportError:
            raise ImportError('Vector search needs sqlite-vec: pip install "orkexa[vector]"') from None
        try:
            self._connection.enable_load_extension(True)
            sqlite_vec.load(self._connection)
            self._connection.enable_load_extension(False)
        except AttributeError:
            raise RuntimeError(
                "This Python's sqlite3 cannot load extensions, which sqlite-vec needs"
            ) from None
        self._vec_loaded = True

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

    # ---- legacy vectors (embedder= callables) ----------------------------------------

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
        if row is None:
            return False
        exists = self._connection.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'index_vec'"
        ).fetchone()  # a restored backup has the meta row but no derived vec0 table
        return exists is not None

    # ---- vector spaces (embeddings=) -------------------------------------------------

    def _space(self, state: str) -> tuple[str, EmbeddingIdentity] | None:
        from .embeddings import EmbeddingIdentity

        row = self._connection.execute(
            "SELECT space, identity FROM index_spaces WHERE namespace = ? AND state = ?",
            (self.namespace, state),
        ).fetchone()
        return (str(row[0]), EmbeddingIdentity.from_dict(json.loads(row[1]))) if row else None

    def _count_vectors(self, space: str) -> int:
        row = self._connection.execute(
            "SELECT count(*) FROM index_vectors v JOIN index_chunks c ON c.id = v.chunk_id"
            " WHERE v.space = ? AND c.namespace = ?",
            (space, self.namespace),
        ).fetchone()
        return int(row[0])

    def vector_status(self) -> VectorStatus:
        """Which embedding model the vectors of this namespace belong to, and how complete they are."""
        with self._lock:
            active = self._space("active")
            building = self._space("building")
            return VectorStatus(
                chunks=len(self),
                active=active[1].to_dict() if active else None,
                embedded=self._count_vectors(active[0]) if active else 0,
                building=building[1].to_dict() if building else None,
                building_embedded=self._count_vectors(building[0]) if building else 0,
                legacy=self._has_vectors(),
            )

    def _active_for(self, identity: EmbeddingIdentity) -> str:
        """The active space of this namespace for ``identity`` (created when there is none)."""
        current = self._space("active")
        if current is None:
            space = f"{self.namespace}:{identity.key}"
            building = self._space("building")
            if building is not None:
                if building[0] == space:
                    return space  # a first reindex runs: its final check picks these chunks up too
                raise RetrievalError("reindex_required", "A reindex for another embedding model is running")
            if len(self):
                # Chunks without vectors exist: they would be missing from vector search.
                raise RetrievalError(
                    "reindex_required",
                    "The index holds chunks without vectors; call reindex() to embed them first",
                )
            self._connection.execute(
                "INSERT OR REPLACE INTO index_spaces (namespace, space, identity, dimensions, state)"
                " VALUES (?, ?, ?, ?, 'active')",
                (self.namespace, space, json.dumps(identity.to_dict()), identity.dimensions),
            )
            return space
        space, stored = current
        if stored.key != identity.key:
            raise RetrievalError(
                "reindex_required",
                f"The vectors belong to {stored.model} ({stored.digest[:12] or 'no digest'}), not to "
                f"{identity.model} ({identity.digest[:12] or 'no digest'}); reindex first",
            )
        return space

    def reindex(
        self,
        embeddings: EmbeddingModel | None = None,
        *,
        batch_size: int = 32,
        progress: Callable[[int, int], None] | None = None,
        cancel: threading.Event | None = None,
    ) -> VectorStatus:
        """Embed every chunk with ``embeddings`` (default: this index's model) and switch to it.

        The new vectors are written to a separate space in batches; searches keep using
        the old space until all chunks are embedded, then the switch happens in one
        transaction and the old vectors are deleted. Interrupted or cancelled, it can be
        called again and continues where it stopped. Chunk texts and the full-text index
        stay as they are (no re-upload or re-chunking).

        Args:
            embeddings: The model to switch to; it becomes this index's model.
            batch_size: Chunks per embedding request and transaction.
            progress: Called with (embedded, total) after each batch.
            cancel: Stops between batches with :class:`RetrievalError` (``cancelled``).
        """
        model = embeddings or self.embeddings
        if model is None:
            raise RetrievalError("vector_unavailable", "reindex() needs an embedding model")
        if self.embedder is not None:
            raise RetrievalError("mode_unsupported", "An index with embedder= cannot switch to vector spaces")
        self._load_sqlite_vec()
        identity = model.identity()
        space = f"{self.namespace}:{identity.key}"
        with self._lock:
            active = self._space("active")
            if active is None or active[0] != space:
                building = self._space("building")
                if building is not None and building[0] != space:
                    self._discard(building[0])
                self._connection.execute(
                    "INSERT OR IGNORE INTO index_spaces (namespace, space, identity, dimensions, state)"
                    " VALUES (?, ?, ?, ?, 'building')",
                    (self.namespace, space, json.dumps(identity.to_dict()), identity.dimensions),
                )
        total = len(self)
        while True:
            if cancel is not None and cancel.is_set():
                raise RetrievalError(
                    "cancelled", "The reindex was cancelled; call reindex() again to continue"
                )
            with self._lock:
                rows = self._connection.execute(
                    "SELECT c.id, c.text FROM index_chunks c WHERE c.namespace = ? AND NOT EXISTS"
                    " (SELECT 1 FROM index_vectors v WHERE v.space = ? AND v.chunk_id = c.id)"
                    " ORDER BY c.id LIMIT ?",
                    (self.namespace, space, batch_size),
                ).fetchall()
            if not rows:
                break
            vectors = model.embed_documents([str(row[1]) for row in rows])
            if len(vectors) != len(rows):
                raise RetrievalError(
                    "vector_unavailable", "The embedding model returned a wrong number of vectors"
                )
            with self._lock:
                self._connection.execute("BEGIN IMMEDIATE")
                try:
                    for (chunk_id, text), vector in zip(rows, vectors, strict=True):
                        if len(vector) != identity.dimensions:
                            raise RetrievalError(
                                "vector_unavailable", "The embedding model changed its vector size"
                            )
                        # The batch was embedded without the lock: a chunk deleted meanwhile gets no
                        # vector, and neither does a new chunk that reused its id (SQLite reuses the
                        # largest rowid); the next round embeds that one with its own text.
                        self._connection.execute(
                            "INSERT OR REPLACE INTO index_vectors (space, chunk_id, embedding)"
                            " SELECT ?, id, ? FROM index_chunks WHERE id = ? AND namespace = ? AND text = ?",
                            (space, self._vector(vector), chunk_id, self.namespace, text),
                        )
                except BaseException:
                    self._connection.execute("ROLLBACK")
                    raise
                self._connection.execute("COMMIT")
                done = self._count_vectors(space)
            if progress is not None:
                progress(done, total)
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                missing = self._connection.execute(
                    "SELECT count(*) FROM index_chunks c WHERE c.namespace = ? AND NOT EXISTS"
                    " (SELECT 1 FROM index_vectors v WHERE v.space = ? AND v.chunk_id = c.id)",
                    (self.namespace, space),
                ).fetchone()[0]
                if missing:  # chunks added meanwhile: the caller runs reindex() again
                    raise RetrievalError(
                        "vector_incomplete", f"{missing} chunks were added during the reindex"
                    )
                old = self._connection.execute(
                    "SELECT space FROM index_spaces WHERE namespace = ? AND state = 'active' AND space != ?",
                    (self.namespace, space),
                ).fetchall()
                for (previous,) in old:
                    self._connection.execute("DELETE FROM index_vectors WHERE space = ?", (previous,))
                    self._connection.execute(
                        "DELETE FROM index_spaces WHERE namespace = ? AND space = ?",
                        (self.namespace, previous),
                    )
                self._connection.execute(
                    "UPDATE index_spaces SET state = 'active' WHERE namespace = ? AND space = ?",
                    (self.namespace, space),
                )
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
            self._connection.execute("COMMIT")
        self.embeddings = model
        return self.vector_status()

    def _discard(self, space: str) -> None:
        self._connection.execute("DELETE FROM index_vectors WHERE space = ?", (space,))
        self._connection.execute(
            "DELETE FROM index_spaces WHERE namespace = ? AND space = ?", (self.namespace, space)
        )

    def discard_reindex(self) -> None:
        """Drop an unfinished reindex (the active vectors stay)."""
        with self._lock:
            building = self._space("building")
            if building is not None:
                self._discard(building[0])

    def drop_vectors(self) -> int:
        """Delete all vectors of this namespace (full-text search stays); returns how many."""
        with self._lock:
            spaces = self._connection.execute(
                "SELECT space FROM index_spaces WHERE namespace = ?", (self.namespace,)
            ).fetchall()
            removed = 0
            for (space,) in spaces:
                removed += self._count_vectors(space)
                self._discard(space)
            return removed

    # ---- Index API ----------------------------------------------------------------

    def add_chunks(self, chunks: Iterable[Chunk]) -> list[int]:
        chunks = [chunk for chunk in chunks if chunk.text.strip()]
        if not chunks:
            return []
        texts = [chunk.text for chunk in chunks]
        vectors: list[Sequence[float]] | None = None
        identity = None
        if self.embeddings is not None:
            identity = self.embeddings.identity()
            vectors = list(self.embeddings.embed_documents(texts))
        elif self.embedder is not None:
            vectors = list(self.embedder(texts))
        if vectors is not None and len(vectors) != len(chunks):
            raise ValueError("The embedder must return one vector per text")
        ids: list[int] = []
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                space = self._active_for(identity) if identity is not None else None
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
                    if vectors is not None and space is not None:
                        if identity is not None and len(vectors[position]) != identity.dimensions:
                            raise ValueError("The embedding model returned a vector of another size")
                        # REPLACE: a vector left behind for this id by a deleted chunk is stale.
                        self._connection.execute(
                            "INSERT OR REPLACE INTO index_vectors (space, chunk_id, embedding)"
                            " VALUES (?, ?, ?)",
                            (space, chunk_id, self._vector(vectors[position])),
                        )
                    elif vectors is not None:
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
            legacy = self._has_vectors() and self._vec_loaded
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                for chunk_id, text, title in rows:
                    self._connection.execute(
                        "INSERT INTO index_fts (index_fts, rowid, text, title) VALUES ('delete', ?, ?, ?)",
                        (chunk_id, text, title or ""),
                    )
                    if legacy:
                        self._connection.execute("DELETE FROM index_vec WHERE rowid = ?", (chunk_id,))
                    self._connection.execute("DELETE FROM index_vectors WHERE chunk_id = ?", (chunk_id,))
                    self._connection.execute("DELETE FROM index_chunks WHERE id = ?", (chunk_id,))
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
            self._connection.execute("COMMIT")
        return len(rows)

    def search(self, query: str, k: int = 5) -> list[Hit]:
        return self.retrieve(query, k)

    def retrieve(
        self,
        query: str,
        k: int = 5,
        *,
        mode: RetrievalMode | None = None,
        sources: Iterable[str] | None = None,
    ) -> list[Hit]:
        """The ``k`` best chunks ranked by ``mode`` (default: the index's), within ``sources``.

        ``sources`` limits the candidates before any ranking. ``vector`` and ``hybrid``
        raise :class:`RetrievalError` when vectors are missing or belong to another
        embedding model; they never fall back to full-text search. Scores: ``fts`` gives
        reciprocal-rank values, ``vector`` the cosine similarity and ``hybrid`` its fused
        score in [0, 1] (:func:`fuse_hybrid`; reciprocal-rank-fusion values with ``fusion="rrf"``).
        ``hybrid`` takes ``max(4 * k, 20)`` candidates from each ranking; ``Hit.ranks`` gives a
        hit's position in each ranking whose candidates hold it.
        """
        if k < 1:
            raise ValueError("k must be positive")
        mode = check_mode(mode) if mode else self.mode
        allowed = None if sources is None else list(dict.fromkeys(sources))
        if allowed is not None and not allowed:
            return []
        if allowed is not None and len(allowed) > _MAX_SOURCES:
            raise ValueError(f"At most {_MAX_SOURCES} sources per search")
        candidates = max(k * 4, 20)
        with self._lock:
            ranked: dict[str, list[tuple[int, float]]] = {}
            if mode in ("fts", "hybrid"):
                ranked["fts"] = [(chunk_id, 0.0) for chunk_id in self._fts_ids(query, candidates, allowed)]
            if mode in ("vector", "hybrid"):
                ranked["vector"] = self._vector_ranking(query, candidates, allowed)
            if mode == "vector":
                return [
                    self._hit(chunk_id, similarity, {"vector": rank + 1})
                    for rank, (chunk_id, similarity) in enumerate(ranked["vector"][:k])
                ]
            ranks: dict[int, dict[str, int]] = {}
            for name, ranking in ranked.items():
                for rank, (chunk_id, _) in enumerate(ranking):
                    ranks.setdefault(chunk_id, {})[name] = rank + 1
            if mode == "hybrid" and self.fusion == "evidence":
                lexical = self._lexical_evidence(query, list(ranks))
                fused = fuse_hybrid(ranked["vector"], lexical, lexical_weight=self.lexical_weight)
                return [self._hit(chunk_id, score, ranks[chunk_id]) for chunk_id, score in fused[:k]]
            scores: dict[int, float] = {}
            for ranking in ranked.values():
                for rank, (chunk_id, _) in enumerate(ranking):
                    scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (_RRF_K + rank + 1)
            best = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:k]
            return [self._hit(chunk_id, scores[chunk_id], ranks[chunk_id]) for chunk_id in best]

    def _lexical_evidence(self, query: str, chunk_ids: list[int]) -> dict[int, float]:
        """:func:`lexical_evidence` of ``chunk_ids``; word weights count the chunks of this namespace."""
        total = len(self)
        weights: dict[str, float] = {}
        found: dict[int, list[str]] = {chunk_id: [] for chunk_id in chunk_ids}
        candidates = json.dumps(chunk_ids)
        for word in query_words(query):
            expression = f'"{word}"*'
            matching = self._connection.execute(
                "SELECT count(*) FROM index_fts JOIN index_chunks ON index_chunks.id = index_fts.rowid"
                " WHERE index_fts MATCH ? AND index_chunks.namespace = ?",
                (expression, self.namespace),
            ).fetchone()[0]
            weights[word] = word_weight(int(matching), total)
            if matching:
                for (chunk_id,) in self._connection.execute(
                    "SELECT rowid FROM index_fts WHERE index_fts MATCH ?"
                    " AND rowid IN (SELECT value FROM json_each(?))",
                    (expression, candidates),
                ):
                    found[int(chunk_id)].append(word)
        return lexical_evidence(weights, found)

    @staticmethod
    def _source_filter(allowed: list[str] | None) -> tuple[str, tuple[Any, ...]]:
        if allowed is None:
            return "", ()
        return " AND index_chunks.source IN (SELECT value FROM json_each(?))", (json.dumps(allowed),)

    def _fts_ids(self, query: str, limit: int, allowed: list[str] | None = None) -> list[int]:
        expression = _fts_query(query)
        if not expression:
            return []
        where, extra = self._source_filter(allowed)
        rows = self._connection.execute(
            "SELECT index_fts.rowid FROM index_fts JOIN index_chunks ON index_chunks.id = index_fts.rowid"
            f" WHERE index_fts MATCH ? AND index_chunks.namespace = ?{where}"
            " ORDER BY bm25(index_fts), index_fts.rowid LIMIT ?",
            (expression, self.namespace, *extra, limit),
        )
        return [int(row[0]) for row in rows]

    def _vector_ranking(self, query: str, limit: int, allowed: list[str] | None) -> list[tuple[int, float]]:
        if self.embeddings is None:
            if self.query_embedder is not None and self._has_vectors():
                return self._legacy_vector_ranking(query, limit, allowed)
            raise RetrievalError(
                "vector_unavailable", "This index has no embedding model; use full-text search"
            )
        active = self._space("active")
        if active is None:
            raise RetrievalError("vector_unavailable", "The index has no vectors yet; run reindex() first")
        space, stored = active
        identity = self.embeddings.identity()
        if identity.key != stored.key:
            raise RetrievalError(
                "reindex_required",
                f"The vectors belong to {stored.model}, the index is configured for "
                f"{identity.model}; reindex first",
            )
        self._load_sqlite_vec()
        vector = self.embeddings.embed_query(query)
        where, extra = self._source_filter(allowed)
        rows = self._connection.execute(
            "SELECT index_vectors.chunk_id, vec_distance_cosine(index_vectors.embedding, ?) AS distance"
            " FROM index_vectors JOIN index_chunks ON index_chunks.id = index_vectors.chunk_id"
            f" WHERE index_vectors.space = ? AND index_chunks.namespace = ?{where}"
            " ORDER BY distance, index_vectors.chunk_id LIMIT ?",
            (self._vector(vector), space, self.namespace, *extra, limit),
        )
        return [(int(row[0]), round(1.0 - float(row[1]), 6)) for row in rows]

    def _legacy_vector_ranking(
        self, query: str, limit: int, allowed: list[str] | None
    ) -> list[tuple[int, float]]:
        assert self.query_embedder is not None
        vector = next(iter(self.query_embedder([query])))
        fetch = (
            limit if allowed is None else min(max(limit * 20, 200), 4096)
        )  # vec0 k-NN cannot filter sources
        rows = self._connection.execute(
            "SELECT rowid, distance FROM index_vec WHERE embedding MATCH ? AND k = ? AND namespace = ?"
            " ORDER BY distance",
            (self._vector(vector), fetch, self.namespace),
        ).fetchall()
        ranking = [(int(row[0]), round(1.0 - float(row[1]), 6)) for row in rows]
        if allowed is not None:
            keep = set(allowed)
            sources = dict(
                self._connection.execute(
                    "SELECT id, source FROM index_chunks WHERE id IN (SELECT value FROM json_each(?))",
                    (json.dumps([chunk_id for chunk_id, _ in ranking]),),
                ).fetchall()
            )
            ranking = [item for item in ranking if sources.get(item[0]) in keep]
        return ranking[:limit]

    def _vector_ids(self, query: str, limit: int) -> list[int]:
        return [chunk_id for chunk_id, _ in self._legacy_vector_ranking(query, limit, None)]

    def _hit(self, chunk_id: int, score: float, ranks: dict[str, int] | None = None) -> Hit:
        source, title, file_id, metadata, text = self._connection.execute(
            "SELECT source, title, file_id, metadata, text FROM index_chunks WHERE id = ?", (chunk_id,)
        ).fetchone()
        return Hit(
            id=chunk_id,
            score=score,
            ranks=dict(ranks or {}),
            source=source,
            title=title,
            file_id=file_id,
            metadata=json.loads(metadata),
            text=text,
        )
