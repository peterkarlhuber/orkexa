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
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import functools
import inspect
import io
import json
import math
import os
from pathlib import Path
import re
import signal
import sqlite3
import struct
import threading
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar, get_args, runtime_checkable

from pydantic import BaseModel, Field

from ._storage import connect_private_database, storage_errors
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
    "MAX_QUERY_CHARS",
    "MAX_QUERY_WORDS",
    "RETRIEVAL_ERROR_CODES",
    "RETRIEVAL_MODES",
    "SEARCH_TOOL_NAME",
    "Chunk",
    "Citation",
    "Embedder",
    "Hit",
    "HybridFusion",
    "Index",
    "PdfLimits",
    "ReindexableIndex",
    "RetrievalError",
    "RetrievalMode",
    "SQLiteIndex",
    "VectorError",
    "VectorStatus",
    "citations",
    "fold",
    "fuse_hybrid",
    "fuse_rrf",
    "lexical_evidence",
    "pdf_pages",
    "pdf_pages_isolated",
    "query_words",
    "retrieval_sources",
    "split_text",
    "tokenize",
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
_RRF_K = 60  # the reciprocal rank fusion constant (fuse_rrf's default k)
_MAX_SOURCES = 50_000
#: The share of the full-text evidence in ``evidence`` fusion (the vector side gets the rest).
LEXICAL_WEIGHT = 0.5
#: The longest search query an :class:`Index` accepts, in characters (``max_query_chars=`` changes it).
MAX_QUERY_CHARS = 1000
#: The most distinct query words (:func:`query_words`) an :class:`Index` accepts (``max_query_words=``).
MAX_QUERY_WORDS = 64

RETRIEVAL_ERROR_CODES: Mapping[str, str] = {
    "mode_unsupported": "the index cannot rank this way",
    "vector_unavailable": "no embedding model or no vectors",
    "reindex_required": "the vectors belong to another embedding model",
    "vector_incomplete": "not every chunk has a vector yet",
    "cancelled": "a reindex was cancelled",
    "query.too_long": "the query exceeds the character or word limit of the index",
    "vector.invalid": "a vector holds NaN or infinity",
    "vector.dimension": "a vector's length differs from the embedding model's or the stored vectors'",
    "partition.missing": "an operation of a partitioned backend has no valid partition (tenant) value",
    "schema.mismatch": "a database schema differs from what the backend needs",
    "storage.symlink": "a SQLite database path is a symbolic link",
    "storage.corrupt": "a SQLite database file is damaged or not a database",
    "document.too_complex": "a PDF exceeds a PdfLimits budget",
}
"""Every :attr:`RetrievalError.code` of every storage backend, with its meaning (MUP-ONA O4).

Products map these codes, never message texts. ``partition.missing`` and ``schema.mismatch`` are raised by
partitioned database backends (PostgreSQL); the SQLite classes never raise them."""


class RetrievalError(OrkexaError, RuntimeError):
    """A search could not run as requested; never answered with another mode instead.

    ``code`` is one of the keys of :data:`RETRIEVAL_ERROR_CODES`, the one code set of every storage backend
    (MUP-ONA O4; a test keeps this list and that mapping equal):

    * ``mode_unsupported``: the index cannot rank this way.
    * ``vector_unavailable``: no embedding model or no vectors.
    * ``reindex_required``: the vectors belong to another embedding model.
    * ``vector_incomplete``: not every chunk has a vector yet.
    * ``cancelled``: a :meth:`SQLiteIndex.reindex` was cancelled.
    * ``query.too_long``: the query exceeds the index's character or word limit
      (:data:`MAX_QUERY_CHARS`, :data:`MAX_QUERY_WORDS`); checked before any work or lock.
    * ``vector.invalid``: a vector holds NaN or infinity (raised as :class:`VectorError`).
    * ``vector.dimension``: a vector's length differs from the embedding model's or the stored vectors'
      (raised as :class:`VectorError`).
    * ``partition.missing``: a partitioned backend got no valid partition (tenant) value; raised before any
      database access (PostgreSQL backends).
    * ``schema.mismatch``: a database schema differs from what the backend needs (PostgreSQL backends).
    * ``storage.symlink``: a SQLite database path (or its journal, WAL or shared-memory file) is a
      symbolic link; :class:`SQLiteIndex`, :class:`~orkexa.SQLiteFileStore` and :class:`~orkexa.SQLiteMemory`
      refuse it.
    * ``storage.corrupt``: a SQLite database file is damaged or not a database; the
      ``sqlite3.DatabaseError`` is the ``__cause__``.
    * ``document.too_complex``: a PDF exceeds a :class:`PdfLimits` budget (:func:`pdf_pages`,
      :func:`pdf_pages_isolated`, :meth:`Index.add_file`); ``reason`` says which: ``pages``, ``chars``,
      ``timeout``, ``memory``, ``limit`` (a limit of pypdf itself) or ``crash`` (the isolated extraction
      ended without a result).

    Messages never contain file paths.
    """

    def __init__(self, code: str, message: str, *, reason: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.reason = reason
        """Which limit was hit, for ``document.too_complex``; otherwise ``None``."""


class VectorError(RetrievalError, ValueError):
    """A refused vector: ``vector.invalid`` or ``vector.dimension``.

    Raised by :func:`~orkexa.embeddings.validate_vector`. Also a ``ValueError``, as :class:`SQLiteIndex`
    raised for such vectors before the codes existed.
    """


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


@dataclass(frozen=True)
class PdfLimits:
    """Budgets for extracting the text of one PDF (:func:`pdf_pages`, :func:`pdf_pages_isolated`).

    ``max_pages`` and ``max_chars`` are checked between pages by both functions; ``timeout`` (seconds, wall
    clock) and ``max_memory_mb`` only by :func:`pdf_pages_isolated`, which enforces them on a separate
    process (the memory limit on Linux only). Exceeding one raises :class:`RetrievalError`
    (``document.too_complex``) with ``reason`` ``pages``, ``chars``, ``timeout`` or ``memory``.
    """

    max_pages: int = 2000
    max_chars: int = 5_000_000
    timeout: float = 30.0
    max_memory_mb: int = 512

    def __post_init__(self) -> None:
        if self.max_pages < 1 or self.max_chars < 1 or self.timeout <= 0 or self.max_memory_mb < 16:
            raise ValueError("PDF limits must be positive (max_memory_mb at least 16)")


def _too_complex(reason: str, message: str) -> RetrievalError:
    return RetrievalError("document.too_complex", message, reason=reason)


def pdf_pages(
    data: bytes, *, max_pages: int = PdfLimits.max_pages, max_chars: int = PdfLimits.max_chars
) -> list[str]:
    """The text of each page of a PDF, in this process, within a page and a character budget.

    Raises :class:`RetrievalError` (``document.too_complex``, ``reason`` ``pages``, ``chars`` or ``limit``
    for a limit of pypdf itself) between pages; a PDF that pypdf cannot read raises pypdf's error. The
    budgets are cooperative: a single page that makes pypdf loop or allocate cannot be stopped here. Use
    :func:`pdf_pages_isolated` for untrusted files. Needs ``pypdf`` (``pip install "orkexa[pdf]"``).
    """
    PdfLimits(max_pages=max_pages, max_chars=max_chars)  # validate
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found,unused-ignore]
        from pypdf.errors import LimitReachedError  # type: ignore[import-not-found,unused-ignore]
    except ImportError:
        raise ImportError('Indexing PDFs needs pypdf: pip install "orkexa[pdf]"') from None
    try:
        reader = PdfReader(io.BytesIO(data))
        trailer: Any = reader.trailer
        declared = trailer["/Root"]["/Pages"].get("/Count", 0)  # checked before the page tree is read
        if isinstance(declared, int) and declared > max_pages:
            raise _too_complex("pages", f"The PDF declares {declared} pages; at most {max_pages} are allowed")
        pages: list[str] = []
        total = 0
        for number, page in enumerate(reader.pages, start=1):
            if number > max_pages:
                raise _too_complex("pages", f"The PDF has more than {max_pages} pages")
            text = page.extract_text() or ""
            total += len(text)
            if total > max_chars:
                raise _too_complex("chars", f"The PDF has more than {max_chars} characters of text")
            pages.append(text)
    except LimitReachedError as exc:
        raise _too_complex("limit", f"pypdf refused the PDF: {exc}") from exc
    return pages


_pdf_pages = pdf_pages
"""The former private name of :func:`pdf_pages`, kept as an alias (Nodestro imports it). It is removed no
earlier than in the next-but-one minor version, announced in the CHANGELOG."""


def pdf_pages_isolated(
    data: bytes,
    *,
    timeout: float = PdfLimits.timeout,
    max_memory_mb: int = PdfLimits.max_memory_mb,
    max_pages: int = PdfLimits.max_pages,
    max_chars: int = PdfLimits.max_chars,
) -> list[str]:
    """The text of each page of a PDF, extracted in a separate process that is stopped at the budget.

    Unlike :func:`pdf_pages`, termination does not depend on pypdf cooperating: the extraction runs in a new
    interpreter (``python -I -m orkexa._pdf_worker``) with a minimal environment (no credentials, keys or
    proxies), no inherited file descriptors, an empty private working directory and lowered resource limits;
    after ``timeout`` seconds its process group gets ``SIGTERM`` and two seconds later ``SIGKILL``. Exceeding
    a budget, running out of memory, a crash or an oversized output raise :class:`RetrievalError`
    (``document.too_complex``) with ``reason`` ``timeout``, ``memory``, ``pages``, ``chars``, ``limit`` or
    ``crash``; a PDF that pypdf cannot read raises ``ValueError``.

    Remaining limits, by design: the child runs as the same operating-system user, so a code-execution flaw
    in pypdf could still reach that user's files and the network (no namespaces, seccomp or containers);
    macOS does not enforce the memory limit (CPU, open-file and file-size limits apply); Windows has no
    resource limits or process groups (the time budget still applies, the child is killed). Starting the
    interpreter costs roughly 0.1-0.3 seconds per document. Needs ``pypdf`` (``pip install "orkexa[pdf]"``).
    """
    from ._pdf_launch import run_worker

    limits = PdfLimits(max_pages=max_pages, max_chars=max_chars, timeout=timeout, max_memory_mb=max_memory_mb)
    run = run_worker(
        data,
        timeout=limits.timeout,
        max_memory_mb=limits.max_memory_mb,
        max_pages=limits.max_pages,
        max_chars=limits.max_chars,
    )
    if run.timed_out:
        raise _too_complex("timeout", f"Extracting the PDF took longer than {limits.timeout:g} seconds")
    if run.output_exceeded:
        raise _too_complex("chars", "The PDF text is larger than the character budget allows")
    result = run.result
    if result is None:
        cpu_signal = getattr(signal, "SIGXCPU", None)
        if run.returncode is not None and cpu_signal is not None and run.returncode == -cpu_signal:
            raise _too_complex("timeout", "Extracting the PDF used more CPU time than allowed")
        raise _too_complex("crash", f"The PDF extraction ended without a result (exit code {run.returncode})")
    if result.get("ok") is True:
        pages = result.get("pages")
        if isinstance(pages, list) and all(isinstance(page, str) for page in pages):
            return pages
        raise _too_complex("crash", "The PDF extraction returned an invalid result")
    reason = str(result.get("reason", "crash"))
    message = str(result.get("message", "The PDF cannot be extracted"))
    if reason == "invalid":
        raise ValueError(message)
    if reason not in ("pages", "chars", "memory", "limit"):
        reason = "crash"
    raise _too_complex(reason, message)


_Method = TypeVar("_Method", bound=Callable[..., Any])


def _checks_query(method: _Method) -> _Method:
    """Marks a ``search``/``retrieve`` that calls :meth:`Index.check_query` itself (not wrapped again)."""
    setattr(method, "__orkexa_checks_query__", True)  # noqa: B010 - a marker attribute on a function
    return method


def _with_query_check(method: Callable[..., Any]) -> Callable[..., Any]:
    """``method`` (a subclass's ``search`` or ``retrieve``), checking its query against the limits first."""

    @functools.wraps(method)
    def checked(self: Index, *args: Any, **kwargs: Any) -> Any:
        query = args[0] if args else kwargs.get("query")
        if isinstance(query, str):
            self.check_query(query)
        return method(self, *args, **kwargs)

    return _checks_query(checked)


class Index(ABC):
    """A searchable collection of text chunks; subclass it for other search backends.

    Subclasses implement :meth:`add_chunks`, :meth:`search` and :meth:`delete`;
    chunking, file ingestion and the tool come from this class. The contract every
    backend shares (score semantics, query limits, error codes) is described at
    :meth:`search`; ``orkexa.testing.conformance`` checks a backend against it.

    The query limits apply to every subclass: when a subclass is defined, its own
    ``search`` and ``retrieve`` are wrapped so that they call :meth:`check_query`
    before anything else (no embedding, lock or SQL for an overlong query).

    Args:
        max_chars: Chunk size for :meth:`add` and :meth:`add_file`.
        overlap: Chunk overlap, see :func:`split_text`.
        max_query_chars: Longest accepted query (default :data:`MAX_QUERY_CHARS`); a longer one raises
            :class:`RetrievalError` (``query.too_long``).
        max_query_words: Most distinct query words (:func:`query_words`) accepted (default
            :data:`MAX_QUERY_WORDS`).
    """

    def __init__(
        self,
        *,
        max_chars: int = 1500,
        overlap: int = 200,
        max_query_chars: int = MAX_QUERY_CHARS,
        max_query_words: int = MAX_QUERY_WORDS,
    ) -> None:
        split_text("", max_chars=max_chars, overlap=overlap)  # validate
        if max_query_chars < 1 or max_query_words < 1:
            raise ValueError("max_query_chars and max_query_words must be positive")
        self.max_chars = max_chars
        self.overlap = overlap
        self.max_query_chars = max_query_chars
        self.max_query_words = max_query_words

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for name in ("search", "retrieve"):
            method = cls.__dict__.get(name)
            if inspect.isfunction(method) and not getattr(method, "__orkexa_checks_query__", False):
                setattr(cls, name, _with_query_check(method))

    @property
    def capabilities(self) -> frozenset[str]:
        """What this index can do beyond full-text search, as configured (MUP-ONA O1).

        ``vectors``: vector and hybrid search; ``reindex``: :class:`ReindexableIndex` (vector spaces bound to
        one embedding model); ``partitioned``: one storage holds several partitions (tenants, or SQLite
        namespaces) and an instance sees only its own; ``concurrent_writers``: several instances or processes
        may write the same storage at once. The base class claims none.
        """
        return frozenset()

    def check_query(self, query: str) -> None:
        """Raise :class:`RetrievalError` (``query.too_long``) if ``query`` exceeds this index's limits.

        Counts characters, then distinct :func:`query_words`; runs before any embedding, lock or SQL.
        """
        _check_query(
            query,
            getattr(self, "max_query_chars", MAX_QUERY_CHARS),  # a subclass may skip Index.__init__
            getattr(self, "max_query_words", MAX_QUERY_WORDS),
        )

    @abstractmethod
    def add_chunks(self, chunks: Iterable[Chunk]) -> list[int]:
        """Store ready-made chunks and return their ids."""

    @abstractmethod
    def search(self, query: str, k: int = 5) -> list[Hit]:
        """The ``k`` best chunks for ``query``, best first, ranked the index's default way.

        The contract every backend shares (MUP-ONA O1); a higher score is better:

        * full text (``fts``): the reciprocal rank ``1 / (60 + rank)`` of a hit (rank 1 is the best), never a
          raw BM25 or ``ts_rank`` value;
        * ``vector``: the cosine similarity of the query and chunk vectors, rounded to 6 digits, whatever
          the length of the vectors; a vector without direction (all zeros) is never a hit;
        * ``hybrid``: the fused score of :func:`fuse_hybrid` in [0, 1] (:func:`fuse_rrf` with
          ``fusion="rrf"``).

        Equal scores are ordered by chunk id, ascending. With an embedding model the default ranking is
        ``hybrid``, without one ``fts``. Query words are :func:`query_words` (:func:`tokenize`,
        :func:`fold`), each matched as a word prefix; FTS syntax in a query is searched as words. A query over
        the limits raises :class:`RetrievalError` (``query.too_long``, see :meth:`check_query`); ``k`` below 1
        raises ``ValueError``. Errors carry a code of :data:`RETRIEVAL_ERROR_CODES`.
        """

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
        same ``source`` again replaces its chunks (in one transaction in :class:`SQLiteIndex`);
        a file that cannot be read keeps the old chunks.

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
        if file.suffix.lower() == ".pdf":
            # pdf_pages with its default budgets (PdfLimits), called by its alias so that code which replaces
            # retrieval._pdf_pages (tests, older integrations) keeps working.
            pages = [(number, page) for number, page in enumerate(_pdf_pages(data), start=1) if page.strip()]
        else:
            try:
                pages = [(0, data.decode("utf-8"))]
            except UnicodeDecodeError:
                raise ValueError(f"{file.name}: only UTF-8 text and PDF files can be indexed") from None
        chunks: list[Chunk] = []
        for number, page in pages:
            extra = {**(metadata or {}), "page": str(number)} if number else dict(metadata or {})
            page_title = f"{title}, p. {number}" if number else title
            chunks += (
                Chunk(text=piece, source=source, title=page_title, file_id=file_id, metadata=extra)
                for piece in split_text(page, max_chars=self.max_chars, overlap=self.overlap)
            )
        return self._replace(source, chunks)

    def _replace(self, source: str, chunks: list[Chunk]) -> list[int]:
        """Replace all chunks of ``source`` with ``chunks``; indexes that can make it atomic override this."""
        self.delete(source)
        return self.add_chunks(chunks)

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


def _check_query(query: str, max_chars: int, max_words: int) -> None:
    """Refuse a query over the limits with ``query.too_long`` (before any embedding, lock or SQL)."""
    if len(query) > max_chars:
        raise RetrievalError(
            "query.too_long", f"The query has {len(query)} characters; at most {max_chars} are allowed"
        )
    words = len(query_words(query))
    if words > max_words:
        raise RetrievalError(
            "query.too_long", f"The query has {words} distinct words; at most {max_words} are allowed"
        )


def _validate_document_vector(vector: Sequence[float], identity: EmbeddingIdentity) -> None:
    """:func:`~orkexa.embeddings.validate_vector` for a vector about to be stored."""
    from .embeddings import validate_vector

    validate_vector(vector, identity)


def _fts_query(query: str) -> str:
    """Natural language -> an FTS5 query matching any word or word prefix ("refund" finds "refunds").

    BM25 then ranks chunks that match more and rarer words higher.
    """
    return " OR ".join(f'"{word}"*' for word in query_words(query))


def tokenize(text: str) -> list[str]:
    """The words of ``text`` in order, as every backend's query side sees them (MUP-ONA O1).

    A word is a run of word characters (``\\w``: letters, digits and the underscore, in any script), so
    ``alpha_beta`` is one word and FTS syntax such as ``NEAR(a b)``, ``text:`` or ``"`` never reaches a
    full-text engine as an operator. Matched as a prefix (:func:`query_words`), a word with an underscore
    finds its parts in sequence (``alpha beta``), not each part alone.
    """
    return _WORD.findall(text)


def fold(word: str) -> str:
    """The form in which a query word is matched: lower case (MUP-ONA O1).

    Diacritics are kept: a full-text tokenizer that removes them (SQLite's ``unicode61 remove_diacritics 2``)
    does so for the query and the documents alike.
    """
    return word.lower()


def query_words(query: str) -> list[str]:
    """The distinct folded words of ``query`` (:func:`tokenize`, :func:`fold`) that full-text search matches.

    Each word is matched as a prefix (``refund`` finds ``refunds``); the query limit :data:`MAX_QUERY_WORDS`
    counts these words.
    """
    return list(dict.fromkeys(fold(word) for word in tokenize(query)))


def fuse_rrf(rankings: Iterable[Sequence[int]], *, k: int = _RRF_K) -> list[tuple[int, float]]:
    """Reciprocal rank fusion: each chunk scores ``sum(1 / (k + rank))`` over the rankings that hold it.

    ``rankings`` are chunk ids, best first (rank 1 is the first); the result is ``(chunk id, score)``, best
    first, ties to the lower chunk id. With one ranking it gives the full-text score of the contract
    (:meth:`Index.search`); ``SQLiteIndex(fusion="rrf")`` fuses its full-text and vector rankings with it.
    """
    if k < 0:
        raise ValueError("k must not be negative")
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


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


@runtime_checkable
class ReindexableIndex(Protocol):
    """An index whose vectors belong to one embedding model and can move to another (MUP-ONA O1).

    Implemented by :class:`Index` subclasses that report the capability ``reindex`` (:class:`SQLiteIndex`, the
    PostgreSQL indexes). The vectors of a namespace live in a *space* bound to an
    :class:`~orkexa.embeddings.EmbeddingIdentity`; :meth:`vector_status` shows its states:

    * ``active``: the space searches use; writes embed into it and refuse another model
      (``reindex_required``); searches with another model raise ``reindex_required`` too, never fall back.
    * ``building``: a :meth:`reindex` in progress or interrupted; resumable, and searches keep using
      ``active`` until every chunk has a vector in the new space. Then the switch happens in one
      transaction and the old space is deleted.

    Both re-read the model's identity (:meth:`~orkexa.embeddings.EmbeddingModel.refresh_identity`) before they
    write vectors, so a model replaced on its server is never filed under its old identity; a chunk deleted
    while its batch was embedded gets no vector, and chunks added meanwhile make :meth:`reindex` raise
    ``vector_incomplete``.
    """

    def vector_status(self) -> VectorStatus:
        """The active and building spaces of this namespace and how many chunks have vectors in each."""
        ...

    def reindex(
        self,
        embeddings: EmbeddingModel | None = None,
        *,
        batch_size: int = 32,
        progress: Callable[[int, int], None] | None = None,
        cancel: threading.Event | None = None,
    ) -> VectorStatus:
        """Embed every chunk with ``embeddings`` (default: the index's model) and switch to that space."""
        ...

    def discard_reindex(self) -> None:
        """Drop an unfinished reindex (the active space stays)."""
        ...


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
        max_query_chars: Longest accepted query (default :data:`MAX_QUERY_CHARS`); a longer one raises
            :class:`RetrievalError` (``query.too_long``) before any work.
        max_query_words: Most distinct query words accepted (default :data:`MAX_QUERY_WORDS`); full-text and
            hybrid search run one or two full-text queries per word.

    A database path that is a symbolic link raises ``storage.symlink``; a damaged file ``storage.corrupt``.
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
        max_query_chars: int = MAX_QUERY_CHARS,
        max_query_words: int = MAX_QUERY_WORDS,
    ) -> None:
        super().__init__(
            max_chars=max_chars,
            overlap=overlap,
            max_query_chars=max_query_chars,
            max_query_words=max_query_words,
        )
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
        self._connection = connect_private_database(database)
        try:
            with storage_errors():
                self._create_tables(tokenizer)
            if has_vectors:
                self._load_sqlite_vec()
        except BaseException:
            self._connection.close()
            raise

    def _create_tables(self, tokenizer: str) -> None:
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

    def _load_sqlite_vec(self) -> None:
        if self._vec_loaded:
            return
        try:
            import sqlite_vec  # type: ignore[import-not-found,import-untyped,unused-ignore]
        except ImportError:
            raise ImportError('Vector search needs sqlite-vec: pip install "orkexa[vector]"') from None
        try:
            self._connection.enable_load_extension(True)
        except AttributeError:
            raise RuntimeError(
                "This Python's sqlite3 cannot load extensions, which sqlite-vec needs"
            ) from None
        try:
            sqlite_vec.load(self._connection)
        finally:
            # Extension loading stays enabled only while sqlite-vec loads, also when loading fails.
            self._connection.enable_load_extension(False)
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
        with self._lock, storage_errors():
            self._connection.execute("INSERT INTO index_fts (index_fts) VALUES ('optimize')")

    @property
    def capabilities(self) -> frozenset[str]:
        """``partitioned`` (by ``namespace``) and ``concurrent_writers`` always; ``vectors`` and ``reindex``
        with ``embeddings=``, only ``vectors`` with an ``embedder=`` callable (:attr:`Index.capabilities`)."""
        found = {"partitioned", "concurrent_writers"}
        if self.embeddings is not None:
            found |= {"vectors", "reindex"}
        elif self.embedder is not None:
            found.add("vectors")
        return frozenset(found)

    def __enter__(self) -> SQLiteIndex:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __len__(self) -> int:
        with self._lock, storage_errors():
            row = self._connection.execute(
                "SELECT count(*) FROM index_chunks WHERE namespace = ?", (self.namespace,)
            ).fetchone()
        return int(row[0])

    def sources(self) -> list[str]:
        """All indexed sources."""
        with self._lock, storage_errors():
            rows = self._connection.execute(
                "SELECT DISTINCT source FROM index_chunks WHERE namespace = ? ORDER BY source",
                (self.namespace,),
            ).fetchall()
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
        from .embeddings import validate_vector

        # NaN or infinity raise VectorError (vector.invalid). A zero vector is stored as before: it has no
        # cosine distance (NULL), so searches leave it out, as validate_vector's None (discard) asks.
        validate_vector(values)
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
        with self._lock, storage_errors():
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
        model.refresh_identity()  # the model's current build, not a cached one
        identity = model.identity()
        space = f"{self.namespace}:{identity.key}"
        with self._lock, self._write():
            # Read and write in one transaction: another process may start or switch a reindex meanwhile.
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
        model.refresh_identity()
        if model.identity().key != identity.key:
            # The model changed on the server while it embedded: the new space may mix two builds.
            with self._lock, self._write():
                self._discard(space)
            raise RetrievalError(
                "reindex_required",
                f"{identity.model} changed during the reindex; its new vectors were discarded,"
                " call reindex() again",
            )
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

    @contextmanager
    def _write(self) -> Iterator[None]:
        """A write transaction (``BEGIN IMMEDIATE``): every read inside it sees the state it writes on.

        Two instances or processes must not read rows outside the transaction and write based on them
        (PS-8). Rolled back on any error; a damaged file raises ``storage.corrupt``. Callers hold the lock.
        """
        with storage_errors():
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
            self._connection.execute("COMMIT")

    def _discard(self, space: str) -> None:
        self._connection.execute("DELETE FROM index_vectors WHERE space = ?", (space,))
        self._connection.execute(
            "DELETE FROM index_spaces WHERE namespace = ? AND space = ?", (self.namespace, space)
        )

    def discard_reindex(self) -> None:
        """Drop an unfinished reindex (the active vectors stay)."""
        with self._lock, self._write():
            building = self._space("building")
            if building is not None:
                self._discard(building[0])

    def drop_vectors(self) -> int:
        """Delete all vectors of this namespace (full-text search stays); returns how many."""
        with self._lock, self._write():
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
        return self._store(None, chunks)

    def _replace(self, source: str, chunks: list[Chunk]) -> list[int]:
        """Delete ``source`` and add ``chunks`` in one transaction: concurrent replacements never mix."""
        return self._store(source, chunks)

    def _store(self, replace: str | None, chunks: Iterable[Chunk]) -> list[int]:
        """Add ``chunks`` (after deleting the source ``replace``, if given) in one write transaction.

        The chunks are embedded first, without the lock and outside the transaction (an embedding call can
        take long).
        """
        chunks = [chunk for chunk in chunks if chunk.text.strip()]
        if not chunks and replace is None:
            return []
        texts = [chunk.text for chunk in chunks]
        vectors: list[Sequence[float]] | None = None
        identity = None
        if chunks and self.embeddings is not None:
            self.embeddings.refresh_identity()  # the model's current build, not a cached one
            identity = self.embeddings.identity()
            vectors = list(self.embeddings.embed_documents(texts))
        elif chunks and self.embedder is not None:
            vectors = list(self.embedder(texts))
        if vectors is not None and len(vectors) != len(chunks):
            raise ValueError("The embedder must return one vector per text")
        ids: list[int] = []
        with self._lock, self._write():
            if replace is not None:
                self._delete_rows(replace)
            if not chunks:
                return []
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
                    if identity is not None:
                        _validate_document_vector(vectors[position], identity)
                    # REPLACE: a vector left behind for this id by a deleted chunk is stale.
                    self._connection.execute(
                        "INSERT OR REPLACE INTO index_vectors (space, chunk_id, embedding) VALUES (?, ?, ?)",
                        (space, chunk_id, self._vector(vectors[position])),
                    )
                elif vectors is not None:
                    table = self._vector_table(len(vectors[position]))
                    self._connection.execute(
                        f"INSERT INTO {table} (rowid, namespace, embedding) VALUES (?, ?, ?)",
                        (chunk_id, self.namespace, self._vector(vectors[position])),
                    )
                ids.append(chunk_id)
        return ids

    def delete(self, source: str) -> int:
        # PS-8: the rows are read inside the write transaction. Read before it, two instances or processes
        # both sent the FTS5 'delete' for the same rows, and the second one damaged the full-text index.
        with self._lock, self._write():
            return self._delete_rows(source)

    def _delete_rows(self, source: str) -> int:
        """Delete the chunks of ``source``; runs inside :meth:`_write`."""
        rows = self._connection.execute(
            "SELECT id, text, title FROM index_chunks WHERE namespace = ? AND source = ?",
            (self.namespace, source),
        ).fetchall()
        if not rows:
            return 0
        legacy = self._has_vectors() and self._vec_loaded
        for chunk_id, text, title in rows:
            self._connection.execute(
                "INSERT INTO index_fts (index_fts, rowid, text, title) VALUES ('delete', ?, ?, ?)",
                (chunk_id, text, title or ""),
            )
            if legacy:
                self._connection.execute("DELETE FROM index_vec WHERE rowid = ?", (chunk_id,))
            self._connection.execute("DELETE FROM index_vectors WHERE chunk_id = ?", (chunk_id,))
            self._connection.execute("DELETE FROM index_chunks WHERE id = ?", (chunk_id,))
        return len(rows)

    @_checks_query
    def search(self, query: str, k: int = 5) -> list[Hit]:
        return self.retrieve(query, k)

    @_checks_query
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

        A query longer than ``max_query_chars`` or with more than ``max_query_words`` distinct words
        raises ``query.too_long`` before it is embedded or the index is locked; a query vector of
        another length than the stored vectors raises ``vector.dimension``.
        """
        if k < 1:
            raise ValueError("k must be positive")
        self.check_query(query)
        mode = check_mode(mode) if mode else self.mode
        allowed = None if sources is None else list(dict.fromkeys(sources))
        if allowed is not None and not allowed:
            return []
        if allowed is not None and len(allowed) > _MAX_SOURCES:
            raise ValueError(f"At most {_MAX_SOURCES} sources per search")
        candidates = max(k * 4, 20)
        # The query is embedded before the lock is taken: an embedding call can take long (a
        # network request) and must not block writes and other searches of this index.
        vector = self._query_vector(query) if mode in ("vector", "hybrid") else None
        with self._lock, storage_errors():
            ranked: dict[str, list[tuple[int, float]]] = {}
            if mode in ("fts", "hybrid"):
                ranked["fts"] = [(chunk_id, 0.0) for chunk_id in self._fts_ids(query, candidates, allowed)]
            if mode in ("vector", "hybrid"):
                assert vector is not None
                ranked["vector"] = self._vector_ranking(vector, candidates, allowed)
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
            fused = fuse_rrf([[chunk_id for chunk_id, _ in ranking] for ranking in ranked.values()])
            return [self._hit(chunk_id, score, ranks[chunk_id]) for chunk_id, score in fused[:k]]

    def _lexical_evidence(self, query: str, chunk_ids: list[int]) -> dict[int, float]:
        """:func:`lexical_evidence` of ``chunk_ids``; word weights count the chunks of this namespace.

        Runs two full-text queries per word, for at most ``max_query_words`` words.
        """
        total = len(self)
        weights: dict[str, float] = {}
        found: dict[int, list[str]] = {chunk_id: [] for chunk_id in chunk_ids}
        candidates = json.dumps(chunk_ids)
        for word in query_words(query)[: self.max_query_words]:
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

    def _active_space(self, identity: EmbeddingIdentity) -> str:
        """The active space for ``identity``, or :class:`RetrievalError` (no vectors / other model)."""
        active = self._space("active")
        if active is None:
            raise RetrievalError("vector_unavailable", "The index has no vectors yet; run reindex() first")
        space, stored = active
        if identity.key != stored.key:
            raise RetrievalError(
                "reindex_required",
                f"The vectors belong to {stored.model}, the index is configured for "
                f"{identity.model}; reindex first",
            )
        return space

    def _query_vector(self, query: str) -> Sequence[float]:
        """The query's vector, embedded without holding the lock (after the cheap checks under it)."""
        if self.embeddings is None:
            with self._lock:
                legacy = self.query_embedder is not None and self._has_vectors()
            if not legacy:
                raise RetrievalError(
                    "vector_unavailable", "This index has no embedding model; use full-text search"
                )
            assert self.query_embedder is not None
            return next(iter(self.query_embedder([query])))
        identity = self.embeddings.identity()
        with self._lock:
            self._active_space(identity)
        return self.embeddings.embed_query(query)

    def _vector_ranking(
        self, vector: Sequence[float], limit: int, allowed: list[str] | None
    ) -> list[tuple[int, float]]:
        if self.embeddings is None:
            if self.query_embedder is not None and self._has_vectors():
                return self._legacy_vector_ranking(vector, limit, allowed)
            raise RetrievalError(
                "vector_unavailable", "This index has no embedding model; use full-text search"
            )
        # Checked again under the lock: a reindex may have switched the space meanwhile.
        space = self._active_space(self.embeddings.identity())
        stored = self._connection.execute(
            "SELECT dimensions FROM index_spaces WHERE namespace = ? AND space = ?", (self.namespace, space)
        ).fetchone()
        self._check_dimensions(vector, int(stored[0]))
        self._load_sqlite_vec()
        where, extra = self._source_filter(allowed)
        # A zero vector has no cosine distance (NULL): it is left out instead of breaking the ranking.
        rows = self._connection.execute(
            "SELECT chunk_id, distance FROM (SELECT index_vectors.chunk_id AS chunk_id,"
            " vec_distance_cosine(index_vectors.embedding, ?) AS distance"
            " FROM index_vectors JOIN index_chunks ON index_chunks.id = index_vectors.chunk_id"
            f" WHERE index_vectors.space = ? AND index_chunks.namespace = ?{where})"
            " WHERE distance IS NOT NULL ORDER BY distance, chunk_id LIMIT ?",
            (self._vector(vector), space, self.namespace, *extra, limit),
        )
        return [(int(row[0]), round(1.0 - float(row[1]), 6)) for row in rows]

    @staticmethod
    def _check_dimensions(vector: Sequence[float], dimensions: int) -> None:
        if len(vector) != dimensions:
            raise VectorError(
                "vector.dimension",
                f"The query vector has {len(vector)} dimensions, the stored vectors {dimensions}",
            )

    def _legacy_vector_ranking(
        self, vector: Sequence[float], limit: int, allowed: list[str] | None
    ) -> list[tuple[int, float]]:
        row = self._connection.execute("SELECT value FROM index_meta WHERE key = 'dimensions'").fetchone()
        if row is not None:
            self._check_dimensions(vector, int(row[0]))
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
        assert self.query_embedder is not None
        vector = next(iter(self.query_embedder([query])))
        return [chunk_id for chunk_id, _ in self._legacy_vector_ranking(vector, limit, None)]

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
