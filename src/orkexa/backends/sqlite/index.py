"""The SQLite search index :class:`SQLiteIndex`: FTS5 full-text search, optionally plus vectors (sqlite-vec).

Moved here from :mod:`orkexa.retrieval` without a behavior change; ``orkexa.SQLiteIndex`` and
``orkexa.retrieval.SQLiteIndex`` remain valid import paths for the same class.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
import json
import os
import re
import sqlite3
import struct
import threading
from typing import TYPE_CHECKING, Any

from ..._storage import connect_private_database, storage_errors
from ...retrieval import (
    _MAX_SOURCES,
    HYBRID_FUSIONS,
    LEXICAL_WEIGHT,
    MAX_QUERY_CHARS,
    MAX_QUERY_WORDS,
    Chunk,
    Embedder,
    Hit,
    HybridFusion,
    Index,
    RetrievalError,
    RetrievalMode,
    VectorError,
    VectorStatus,
    _checks_query,
    _validate_document_vector,
    check_mode,
    fuse_hybrid,
    fuse_rrf,
    lexical_evidence,
    query_words,
    word_weight,
)

if TYPE_CHECKING:
    from ...embeddings import EmbeddingIdentity, EmbeddingModel

__all__ = ["SQLiteIndex"]


def _fts_query(query: str) -> str:
    """Natural language -> an FTS5 query matching any word or word prefix ("refund" finds "refunds").

    BM25 then ranks chunks that match more and rarer words higher.
    """
    return " OR ".join(f'"{word}"*' for word in query_words(query))


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

    The private attributes ``_connection`` (the ``sqlite3.Connection``) and ``_lock`` (the ``threading.RLock``
    every operation holds while it uses the connection) are preserved for compatibility: Nodestro reads chunks
    through them. They are not public API.
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
        from ...embeddings import validate_vector

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
        from ...embeddings import EmbeddingIdentity

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
