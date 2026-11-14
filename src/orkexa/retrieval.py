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
from dataclasses import dataclass
import functools
import inspect
import io
import math
import os
from pathlib import Path
import re
import signal
import threading
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar, get_args, runtime_checkable

from pydantic import BaseModel, Field

from .content import ContentBlock, ToolContent
from .core.types import Citation, Message, ModelResponse, SearchResultPart, TextPart, ToolResultPart
from .errors import OrkexaError
from .tools import Tool

if TYPE_CHECKING:
    from .backends.sqlite.index import SQLiteIndex as SQLiteIndex
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
      refuse it. Since TABRION MIG-ORK-3 also an existing file of another user or one writable by group or
      others; ``reason`` says which: ``symlink``, ``other_owner``, ``writable_by_others`` or
      ``unsafe_path`` (no regular file, or the path changed while it was opened).
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
        """Which limit was hit, for ``document.too_complex``; which rule refused the path, for
        ``storage.symlink``; otherwise ``None``."""


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


# MUP-ONA B2: SQLiteIndex lives in orkexa.backends.sqlite.index and is re-exported here, so every
# import path returns the same class. The backend imports this module, so the re-export is resolved on first
# access (PEP 562) instead of by a module-level import, which would be circular. Type checkers see the import
# under TYPE_CHECKING above; this block is hidden from them so that an unknown name stays an error.
if not TYPE_CHECKING:

    def __getattr__(name: str) -> object:
        if name == "SQLiteIndex":
            from .backends.sqlite.index import SQLiteIndex

            return SQLiteIndex
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    def __dir__() -> list[str]:
        return sorted({*globals(), "SQLiteIndex"})
