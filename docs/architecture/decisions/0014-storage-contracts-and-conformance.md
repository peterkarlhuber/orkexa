# ADR 0014 – Storage contracts, error codes and the conformance suite

Status: accepted (2026-11-07, MUP-ONA B1; decisions O1, O2 and O4 confirmed at gate G0 on 2026-11-06)

## Context

`Index`, `FileStore` and `MemoryBackend` defined only signatures. What a backend had to do beyond them lived in
`SQLiteIndex` alone, and Nodestro's PostgreSQL classes copied or re-derived it: the tokenization of queries
(`\w+` here, `[^\W_]+` plus diacritic folding there), reciprocal rank fusion (a second `_RRF_K = 60`), the
NaN/Inf check (SQLite only), query limits (added to `SQLiteIndex` by MUP-ONA A10, absent elsewhere) and the score
scales (cosine rounded to 6 digits here, raw dot products or `ts_rank` there). Products matched error message
texts because only some errors had codes. Nothing could check a backend against the contracts; the only reusable
pattern was `orkexa.flow.conformance` for flow drivers.

The MUP-ONA plan (chapters 4 and 5, packages B1 to B6) moves the SQLite classes to `orkexa.backends.sqlite` (B2),
fixes ten documented PostgreSQL deviations in Nodestro (B3) and adds `orkexa.backends.postgres` (B5). All of that
needs one definition of the contracts and one suite that measures every backend against it.

## Decision

### O1: contract additions and score semantics (`orkexa.retrieval`, `orkexa.embeddings`)

* **Score semantics** are part of the contract (`Index.search`): full text scores the reciprocal rank
  `1 / (60 + rank)`; vector search the cosine similarity rounded to 6 digits, whatever the vector length; hybrid
  the `fuse_hybrid` score in [0, 1] (`fuse_rrf` with `fusion="rrf"`). Equal scores are ordered by chunk id. With an
  embedding model the default ranking is `hybrid`, without one `fts`.
* **`tokenize(text)` and `fold(word)`** are public; `query_words` uses them. `tokenize` is exactly the earlier
  `\w+` and `fold` exactly `str.lower` (diacritics are kept: tokenizers that remove them do it on both sides), so
  SQLite queries and their results are bit-identical. Nodestro switches its query side to this definition in B3.
* **`fuse_rrf(rankings, k=60)`** is the one reciprocal rank fusion; `SQLiteIndex` uses it, summing in the same
  order as before (bit-identical floats).
* **`validate_vector(vector, identity)`** (`orkexa.embeddings`) is the one vector check: wrong length raises
  `vector.dimension`, NaN/Inf `vector.invalid` (both as `VectorError`, which is a `RetrievalError` and a
  `ValueError`, because `SQLiteIndex` raised `ValueError` before), a zero vector returns `None` (discard: never a
  hit, a zero query finds nothing), otherwise the vector scaled to unit length unless `identity.normalized`.
  `SQLiteIndex` calls it for every vector it stores or searches with but keeps storing the original values and
  keeps leaving zero vectors out at search time (a `NULL` cosine distance): the same results as before.
* **Query limits are lifted into `Index`.** `MAX_QUERY_CHARS = 1000` and `MAX_QUERY_WORDS = 64` (A10) become
  constructor options of `Index` (`max_query_chars=`, `max_query_words=`) and `Index.check_query()`. To make them
  apply to *every* backend ("via the contract", PS-9), `Index.__init_subclass__` wraps a subclass's own `search`
  and `retrieve` so that they call `check_query` first; a subclass that skips `Index.__init__` gets the defaults.
  `SQLiteIndex` marks its methods as checking themselves, so its order of checks is unchanged.
* **`capabilities: frozenset[str]`** on `Index`, `FileStore` and `MemoryBackend` (empty in the base classes), with
  the vocabulary `vectors`, `reindex`, `partitioned`, `concurrent_writers`. The SQLite classes report
  `partitioned` (by `namespace`) and `concurrent_writers`; `SQLiteIndex` adds `vectors` and `reindex` as configured
  (`embeddings=`; `embedder=` gives only `vectors`). `LocalFileStore`, `EphemeralMemory` and `FileMemory` report
  none (one root per user; several instances on one root may lose a concurrent change).
* **`ReindexableIndex`** is a runtime-checkable protocol (`vector_status`, `reindex`, `discard_reindex`) that
  documents the space states `active`/`building`, the identity refresh before writes and the reindex race rules.
* **`pdf_pages`** is public since A1; `orkexa.retrieval._pdf_pages` stays an alias without a warning. Following
  the release rules (O3), the `DeprecationWarning` comes with the next minor version and the removal no earlier
  than the one after.

### O2: layout of the test helpers

* `orkexa.testing` (docstring only), `orkexa.testing.fakes` (`ScriptedClient`) and `orkexa.testing.conformance`
  (`Case`, `run_cases`, `Report`, the case lists of `index`, `reindex`, `files`, `memory`, `partition`). No core
  module imports `orkexa.testing` (import-linter contract since B0); the package needs no pytest, so the suite
  runs under any test runner. Orkexa's own pytest adapter lives in `tests/conformance/`.
* There is no `orkexa.testing.security` and no registry or factory of backends (E-5): the caller passes a factory
  callable and the capabilities it expects.

### O4: one set of error codes

`orkexa.retrieval.RETRIEVAL_ERROR_CODES` maps every `RetrievalError.code` of every storage backend to its
meaning; the `RetrievalError` docstring lists the same codes (a test keeps both and every code raised in the
source in step). New: `vector.invalid`, `partition.missing` and `schema.mismatch` (for the PostgreSQL backends);
`query.too_long`, `vector.dimension`, `storage.symlink`, `storage.corrupt` and `document.too_complex` came with A1
and A10. Products map codes, never message texts.

### The conformance suite (`orkexa.testing.conformance`)

* A **case** has a name, the capabilities it needs and a `run(context)`; `run_cases(factory, cases, expect=...)`
  returns a `Report` of passed, failed and skipped cases. The factory gets a `Request` (a fresh working
  directory, a unique `scope`, the partition value and an optional embedding model) and returns a new instance on
  the storage of that case.
* **Capabilities are declared by the caller.** If the backend reports fewer capabilities than `expect`, the run
  fails; a backend whose constructor takes `tenant_id` must report `partitioned`. A case whose needs the backend
  does not meet is skipped with the reason, and the caller compares the skipped cases with an expectation file
  (`tests/conformance/expected_skips.json` in Orkexa), so a lost capability never passes silently.
* **Security cases** (partition isolation per plan chapter 10.4, input limits, vector checks) can only be switched
  off by `partitioned` or, for vector checks, `vectors`; input limits need nothing.
* Score semantics, default mode, identity refresh, vector validation, source limit, query tokenization and the
  reindex race rules are cases of their own, so that each documented PostgreSQL deviation (plan annex A, drift
  points) fails a named case until B3 fixes it.

## Consequences

* Purely additive for users: new names only (T-API-1 snapshot extended), no changed signatures, SQLite results
  unchanged (the existing suite and FX-RANK-0 stay green unchanged).
* Behavior change for other `Index` subclasses: a query over 1000 characters or 64 distinct words now raises
  `query.too_long` (pass larger limits to `Index.__init__` to keep longer queries). This is the intended PS-9 fix
  for every backend, including Nodestro's PostgreSQL index once it updates its Orkexa pin.
* `VectorError` is also a `ValueError`: code that caught `ValueError` for invalid vectors keeps working; the A10
  query-dimension error (`vector.dimension`) is now a `VectorError` too and therefore also a `ValueError`.
* Rejected alternatives: moving `RetrievalError` into `orkexa.errors` (would change its module and pickled
  name for no gain); capabilities declared per class (an `SQLiteIndex` without embeddings cannot rank vectors);
  enforcing the query limits only in `SQLiteIndex` and asking other backends to call `check_query` (would leave
  PS-9 open in every backend that forgets it).
