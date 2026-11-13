# ADR 0016 – Responsibilities in the product family: Orkexa owns embedding and retrieval

Status: accepted (2026-11-13, owner decision; binding)

## Context

Five products share code and data paths: Nodestro, Orkexa, Axentra, Tabrion and Tabrion Studio. Retrieval code has
moved between them. nodestro-studio `docs/adr/0002-storage-backends.md`, point 5, placed the PostgreSQL
implementations of Orkexa's bases (`PostgresIndex`, `PostgresFileStore`, `PostgresMemory`) inside nodestro-runtime
and stated "Orkexa is not changed". MUP-ONA B5 then moved the generic PostgreSQL backend into Orkexa
(`orkexa.backends.postgres`, [ADR 0015](0015-statement-catalog-and-sql-rule-engine.md)), and in B6 Nodestro
switched to it: nodestro-runtime `src/nodestro_runtime/postgres_tables.py` imports `orkexa.backends.postgres` and
its `PostgresIndex` subclasses `orkexa.backends.postgres.PostgresIndex`. With Tabrion arriving as a shared database
layer, the question of which product owns chunking, vectors and search needed a binding answer.

## Decision

Each product has one responsibility:

* **Nodestro** – application-specific knowledge management: documents, knowledge collections, access assignments,
  and integrating RAG results into AI agents and workflows.
* **Orkexa** – provider-independent embedding and retrieval infrastructure: text segmentation (chunking),
  vectorization, indexing, full-text search, semantic search and hybrid search.
* **Axentra** – reusable SaaS backend infrastructure: identity and user management, multi-tenancy,
  authentication, authorization, roles and permissions, API keys, auditing, and other general backend services
  such as billing, events and background jobs.
* **Tabrion** – shared, secure database infrastructure for SQLite and PostgreSQL: connections, transactions,
  security checks and extension support. It has no embedding or retrieval logic of its own and is not a vector
  database.
* **Tabrion Studio** – a graphical environment to develop and inspect databases, with read-only support for vector
  data, embedding metadata, indexes and database extensions. It has no embedding or retrieval logic.

The existing decisions on migration runners and connection pools stay unchanged: they are not unified, and each
product keeps its own runner and pool. In Orkexa these are `orkexa.backends.postgres.schema.migrate` (one
transaction under an advisory lock) and `orkexa.backends.postgres.connection.PsycopgSource`.

This ADR supersedes the placement in nodestro-studio ADR-0002, point 5: the PostgreSQL retrieval backend lives in
Orkexa (`src/orkexa/backends/postgres/index.py`), and Nodestro adds only its product-specific parts on top of it.

## State in Orkexa (master, 2026-11-13)

* **Chunking:** `orkexa.retrieval.split_text`; the `Index` contract and its scores are in `src/orkexa/retrieval.py`
  ([ADR 0014](0014-storage-contracts-and-conformance.md)).
* **SQLite** (`src/orkexa/backends/sqlite/index.py`): FTS5 for full text; vectors with sqlite-vec (extra `vector`),
  ranked by exact cosine distance with `vec_distance_cosine` over the vectors of the active space. The older
  `embedder=` callables keep using a `vec0` table, which also searches exhaustively.
* **PostgreSQL** (`src/orkexa/backends/postgres/`, extra `postgres`): vectors are stored as `real[]` columns, never
  as pgvector's `vector` type (`schema.py`). When pgvector is installed, the database ranks them with `<=>` (exact
  cosine distance, `_sql.py`); without it, Python ranks the same rows (`_vector.py`). Full text uses `tsvector` and
  `ts_rank`.
* **No approximate nearest-neighbor (ANN) index** exists today on either backend; every vector search is exact.
* **Dependencies:** Orkexa depends on no Nodestro, Axentra or Studio package; the import contract "Orkexa depends
  on no Axentra, Nodestro or Studio package (K-33)" in `pyproject.toml` enforces this in CI. Tabrion is the database
  foundation and is used through version ranges, not pins. Today the only Tabrion requirement is in the `dev` extra,
  `tabrion[testing,postgres]>=0.1,<0.2`, used by the PostgreSQL test support (`tests/backends/postgres/support.py`,
  TABRION MIG-ORK-1); it is development and test only. The runtime package, including the `postgres` extra
  (`psycopg>=3.2,<4`), does not require Tabrion, and the core may not import it until MIG-ORK-3 (contract "Core
  modules do not import tabrion until MIG-ORK-3"). MIG-ORK-2 is planned to add `tabrion[postgres]` to the extra
  `postgres`.

## Consequences

* New retrieval features (chunking strategies, embedding providers, ANN indexes, ranking and fusion) belong in
  Orkexa, behind its contracts and conformance suite, and not in Nodestro, Axentra or Tabrion.
* Tabrion may supply connections, transactions, security checks and extension detection (for example whether
  pgvector or sqlite-vec is available) to Orkexa's backends, but the vector schema, queries and ranking stay in
  Orkexa. Tabrion Studio shows vector data and embedding metadata but never computes embeddings or searches.
* Nodestro keeps documents, collections, access assignments and RAG integration, and builds them on Orkexa's
  `Index`; it does not reimplement search.
* Tenant identity, permissions and audit come from the host (Axentra or Nodestro); Orkexa enforces only the
  partition of a tenant's data (ADR 0015), not who may use a tenant.
* Rejected alternatives: making Tabrion a vector database (it would duplicate Orkexa's ranking and couple the
  database layer to embedding models); keeping PostgreSQL retrieval in Nodestro (other hosts could not reuse it,
  which is what B5 fixed); unifying migration runners and pools across products (not decided; out of scope here).
