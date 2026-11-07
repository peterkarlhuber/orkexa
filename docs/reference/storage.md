# Memory, files and retrieval

## The storage contracts

Every backend of `Index`, `FileStore` and `MemoryBackend` (the SQLite and local classes here, Nodestro's
PostgreSQL classes, your own) shares the contract below
([ADR 0014](../architecture/decisions/0014-storage-contracts-and-conformance.md)).

**Scores** (higher is better; equal scores are ordered by chunk id, ascending):

| Ranking | Score |
|---|---|
| full text (`fts`) | the reciprocal rank `1 / (60 + rank)`, rank 1 being the best hit; never a raw BM25 or `ts_rank` value |
| `vector` | the cosine similarity of query and chunk vector, rounded to 6 digits, whatever the vector length; a zero vector is never a hit, a zero query finds nothing |
| `hybrid` | `fuse_hybrid` in [0, 1] (`fusion="evidence"`, the default), or `fuse_rrf` with `fusion="rrf"` |

With an embedding model the default ranking of `search()` is `hybrid`, without one `fts`. `vector` and `hybrid`
never fall back to full text: they raise `RetrievalError` (`vector_unavailable`, `reindex_required`).

**Queries:** the query words are `query_words(query)`, that is `tokenize` (runs of `\w`, so FTS syntax is searched
as words) and `fold` (lower case), each matched as a word prefix. A query longer than `MAX_QUERY_CHARS` (1000)
characters or with more than `MAX_QUERY_WORDS` (64) distinct words raises `RetrievalError("query.too_long")`
before any embedding, lock or SQL, in **every** `Index` subclass (`Index.check_query`; the limits are
`Index.__init__` options). A search may name at most 50,000 sources.

**Vectors:** `validate_vector` checks every vector a backend stores or searches with: a wrong length raises
`vector.dimension`, NaN or infinity `vector.invalid` (both as `VectorError`, also a `ValueError`), a zero vector
is discarded.

**Capabilities:** `capabilities` (a `frozenset[str]`) says what an instance can do beyond the base contract:

| Capability | Meaning | Reported by |
|---|---|---|
| `vectors` | vector and hybrid search | `SQLiteIndex` with `embeddings=` or `embedder=` |
| `reindex` | `ReindexableIndex`: vector spaces bound to one embedding model, `reindex()` | `SQLiteIndex` with `embeddings=` |
| `partitioned` | one storage holds several partitions (tenants, or SQLite namespaces); an instance sees only its own | the three SQLite classes |
| `concurrent_writers` | several instances or processes may write the same storage at once | the three SQLite classes |

**Error codes** (`RetrievalError.code`, the same for every backend; map codes, never message texts):

| Code | Meaning |
|---|---|
| `mode_unsupported` | the index cannot rank this way |
| `vector_unavailable` | no embedding model or no vectors |
| `reindex_required` | the vectors belong to another embedding model |
| `vector_incomplete` | not every chunk has a vector yet (also: chunks added during a reindex) |
| `cancelled` | a reindex was cancelled |
| `query.too_long` | the query exceeds the character or word limit |
| `vector.invalid` | a vector holds NaN or infinity |
| `vector.dimension` | a vector's length differs from the model's or the stored vectors' |
| `partition.missing` | a partitioned backend got no valid partition (tenant) value (PostgreSQL backends) |
| `schema.mismatch` | the database schema differs from what the backend needs (PostgreSQL backends) |
| `storage.symlink` | a SQLite database path is a symbolic link |
| `storage.corrupt` | a SQLite database file is damaged or not a database |
| `document.too_complex` | a PDF exceeds a `PdfLimits` budget (`reason` says which) |

## Memory

::: orkexa.MemoryBackend

::: orkexa.EphemeralMemory

::: orkexa.FileMemory

::: orkexa.SQLiteMemory

## Files

::: orkexa.FileStore

::: orkexa.LocalFileStore

::: orkexa.SQLiteFileStore

## Retrieval

::: orkexa.Index

::: orkexa.SQLiteIndex

::: orkexa.retrieval.ReindexableIndex

::: orkexa.retrieval.VectorStatus

::: orkexa.retrieval.RetrievalError

::: orkexa.retrieval.VectorError

::: orkexa.retrieval.RETRIEVAL_ERROR_CODES

::: orkexa.retrieval.MAX_QUERY_CHARS

::: orkexa.retrieval.MAX_QUERY_WORDS

::: orkexa.retrieval.tokenize

::: orkexa.retrieval.fold

::: orkexa.retrieval.query_words

::: orkexa.retrieval.fuse_rrf

::: orkexa.retrieval.fuse_hybrid

::: orkexa.embeddings.validate_vector

::: orkexa.retrieval.PdfLimits

::: orkexa.retrieval.pdf_pages

::: orkexa.retrieval.pdf_pages_isolated

::: orkexa.citations

::: orkexa.retrieval_sources

::: orkexa.split_text
