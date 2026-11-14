# PostgreSQL backends

`orkexa.backends.postgres` holds `PostgresIndex`, `PostgresFileStore` and `PostgresMemory`: the storage contracts
of [Memory, files and retrieval](storage.md) on PostgreSQL, with many tenants in one set of tables. The classes
come from Nodestro's generic PostgreSQL backend (nodestro-runtime `pg_generic`, MUP-ONA B4) without a change of
behavior; Nodestro switches to them in MUP-ONA B6.

```bash
pip install -r requirements/tabrion.txt "orkexa[postgres]"  # psycopg 3 and the pinned Tabrion; psycopg needs libpq
```

The extra also names `tabrion[postgres]>=0.1,<0.2` (TABRION MIG-ORK-2). Tabrion is not on a package index yet:
install it with `pip install -r requirements/tabrion.txt "orkexa[postgres]"` (pip 25.0 or newer; see
`requirements/tabrion.txt` in the repository), or let the product's own Tabrion pin satisfy the range.

`import orkexa` never imports this package, and `import orkexa.backends.postgres` works without psycopg and opens
no connection. psycopg is imported when a connection is first needed (`PsycopgSource(...)`, `migrate`); without it
that raises `ImportError` with the message `The PostgreSQL backend needs psycopg: pip install "orkexa[postgres]"`
(`MISSING_PSYCOPG`). Tabrion is imported only by the session settings and `ping` of `PsycopgSource` and by the
error texts of `verify_schema`; a `PsycopgSource` without session settings works without it.

## A first index

```python
from orkexa.backends.postgres import PostgresIndex, PostgresLayout, PsycopgSource, migrate, verify_schema

layout = PostgresLayout(prefix="app_")  # tables app_chunks, app_files, ...
source = PsycopgSource("host=db dbname=app user=app_runtime", max_connections=10)

report = verify_schema(source, layout)  # read-only, never raises
if not report.ok:
    raise SystemExit(report.summary())  # run your migration (or migrate() as the owner) first

index = PostgresIndex(source, "tenant-42", layout=layout, namespace="handbook")
index.add("Refunds are possible within 30 days.", source="refunds.md")
hits = index.search("refund", k=5)
```

Each instance works on one tenant (`tenant_id`, the partition) and one namespace. The host decides which tenant a
request may use and passes it in; see [the security model](#security-model-partition-yes-permissions-no).

## Layout and prefix

`PostgresLayout(prefix="orkexa_")` names the tables. Only the prefix is configurable; the column names are fixed
(MUP-ONA E-12). The prefix must match `^[a-z][a-z0-9_]{0,30}_$` (lower case, ends with `_`, at most 32 characters),
so every name fits PostgreSQL's 63-byte limit; any other value raises `ValueError`, and a store refuses an object
that is not a `PostgresLayout` (`TypeError`).

| Key | Table (prefix `p_`) | Holds |
|---|---|---|
| `files` | `p_files` | the files of `PostgresFileStore` |
| `chunks` | `p_chunks` | the chunks of `PostgresIndex` and their search words (`tsvector`) |
| `memory_files` | `p_memory_files` | the files of `PostgresMemory` |
| `vector_spaces` | `p_vector_spaces` | one vector space per embedding model and namespace |
| `chunk_vectors` | `p_chunk_vectors` | the vectors (`real[]`) |

`layout.table(key)`, `layout.tables` and `layout.version_table` (`p_orkexa_schema`, written only by `migrate`)
give the names. Several layouts can live side by side in one schema. Nodestro uses the prefix `nodestro_`.

## Connections: `ConnectionSource`

The stores never open connections themselves. They take a `ConnectionSource`, a protocol with one method:

```python
def transaction(self, mode: TransactionMode = "transaction", *, tenant_id: str | None = None)
    -> ContextManager[Connection]: ...
```

`mode` is `"read"`, `"write"` or `"transaction"` (`MODES`); `tenant_id` is the partition of the statement, for a
source that sets it on the session (row-level security). A host with its own pool passes its own source.

`PsycopgSource(conninfo, *, max_connections=10, timeout=30.0, tenant_setting=None, statement_timeout=None,
idle_in_transaction_timeout=None, lock_timeout=None, application_name=None, **connect_args)` is the source
that comes with Orkexa: psycopg 3 with a small pool of its own (autocommit connections, opened on demand, a broken
connection is discarded). A `transaction` block is joined by every `transaction` the same thread opens inside it,
so the commands of several stores can form one transaction. With `tenant_setting="app.tenant_id"` it sets that
setting to the tenant at the start of every transaction (`set_config(..., true)`), as the hook for row-level
security. The stores on one `PsycopgSource` share its `lock` by default, so two instances in one process never
lose each other's read-modify-write change (pass `lock=` to use another lock). Processes do not share the lock: a
host that runs `PostgresMemory` or `PostgresFileStore` for one tenant and namespace in several processes at once
serializes those commands itself, as with Nodestro's classes.

Session settings (TABRION MIG-ORK-2): `statement_timeout`, `idle_in_transaction_timeout` and `lock_timeout` (in
seconds; `0` turns a limit off) and `application_name` are set once on every new connection, right after it is
opened, with `tabrion.postgres` (one `set_config(..., false)` round trip, Tabrion's allowlist of settings). They
are all `None` by default: then nothing is set and a connection is exactly what `psycopg.connect` gives, as
before. Invalid values raise `ValueError` when the source is built; a connection whose settings fail is closed.
A timeout ends the statement with psycopg's error (`QueryCanceled`, `LockNotAvailable`), as any server error.

```python
source = PsycopgSource(
    "host=db dbname=app user=app_runtime",
    statement_timeout=30,
    idle_in_transaction_timeout=300,
    lock_timeout=10,
    application_name="my-product",
)
if not source.ping(timeout=5):  # True or False, never an exception for a database or connection error
    raise SystemExit("PostgreSQL is not reachable")
```

`ping(timeout=5.0)` runs `tabrion.postgres.ping` on a connection of the pool (inside a `transaction` block of the
same thread on that block's connection, under a savepoint): `False` when the server does not answer within
`timeout`, the connection fails or no connection is free in time. A pooled connection the server has dropped is
discarded and the next one is tried. `verify_schema` describes a database error with Tabrion's `safe_message`
(kind and SQLSTATE, never the server's text, the statement or connection parameters).

## Statement catalog and partition guard

Every statement the stores run is an entry of the statement catalog (`CATALOG`, 43 entries: `Statement(name,
template, params, kind, partitioned, rows)`), rendered for a layout by `render(name, prefix)` with `psycopg.sql.Identifier`.
Values are always bound parameters, never part of the text. The executor runs only catalog entries; before a
partitioned entry runs, the partition guard checks that its `tenant_id` parameter is exactly the tenant of the
instance (a non-empty string of at most 128 characters without U+0000) and otherwise raises
`RetrievalError("partition.missing")` without running anything. Orkexa's tests check every entry with the tenant
SQL rule engine ([`orkexa.testing.sqlcheck`](testing.md#the-tenant-sql-rule-engine)), mutate it to prove the check
fires, and compare the catalog rendered with `nodestro_` with Nodestro's catalog
([ADR 0015](../architecture/decisions/0015-statement-catalog-and-sql-rule-engine.md)).

## Schema version 1

Schema version 1 (`SCHEMA_VERSION`) is exactly the form of Nodestro's migrations `0002_knowledge_memory.sql` and
`0010_knowledge_vectors.sql`, for any prefix: five tables, their constraints and three indexes, vectors as
`real[]`. pgvector is never part of the schema.

The backend never runs DDL at run time (MUP-ONA E-13): the runtime role needs only `SELECT`, `INSERT`, `UPDATE` and
`DELETE` on the five tables (and `SELECT` on the version table for `verify_schema`), no `CREATE`, no ownership.

| Function | Use |
|---|---|
| `verify_schema(source, layout) -> SchemaReport` | checks the tables, columns (type, nullability, collation, default, generation), indexes and the recorded version, read-only (`information_schema`, `pg_indexes`, `pg_extension`); never changes anything and never raises. A product calls it at start. |
| `schema_steps(layout) -> tuple[str, ...]` | the statements of version 1 for a layout, for a product's own migration runner |
| `migrate(source, layout) -> int` | applies them, only when called (never by a store), and records version 1 in `<prefix>orkexa_schema`; returns the version found before (`0`: none) |

`SchemaReport` has `ok`, `summary()`, `version`, `pgvector` (installed and its version) and `problems`, a tuple of
`SchemaProblem(kind, table, name, detail)` with `kind` one of `missing_table`, `missing_column`, `wrong_column`,
`missing_index`, `wrong_index`, `version`, `error` (the check itself failed, for example for lack of rights).

`migrate` runs in one transaction under an advisory lock (concurrent calls wait for each other) and is idempotent.
It raises `RetrievalError("schema.mismatch")` with a `reason`:

| `reason` | Meaning |
|---|---|
| `permission` | the role may not create tables in the current schema (the runtime role); checked first, nothing changed |
| `too_new` | a newer schema version is recorded |
| `unrecorded` | some of the tables exist, but no version is recorded: they were created another way |

It never runs `CREATE EXTENSION`.

## Vectors: pgvector or Python (modes A and B)

`PostgresIndex(..., embeddings=model)` stores one vector per chunk in the vector space of the model's identity and
supports `vector` and `hybrid` search and `reindex()` (`capabilities` `vectors`, `reindex`, `partitioned`,
`concurrent_writers`), with the scores of the storage contract.

* **Mode B:** where an administrator installed pgvector (`CREATE EXTENSION vector`), the database ranks the vectors
  (an exact cosine distance over the `real[]` column cast to `vector`).
* **Mode A:** without pgvector, Python ranks the same rows with the same result.

The index detects pgvector once (`index.pgvector()`); `index.refresh()` makes it detect it again after the
extension was installed or removed while the server runs. Searches are limited to the allowed sources before
ranking in both modes. Orkexa's CI runs every PostgreSQL test in both modes (jobs `pg-a` on `postgres:17` and
`pg-b` on `pgvector/pgvector:pg17`) and checks that both rank the reference fixture alike (T-RANK-1).

## Security model: partition yes, permissions no

The backends enforce the partition and nothing else (MUP-ONA E-6):

* **Partition.** Every statement on a tenant table is bound to the instance's tenant (catalog, rule engine, guard
  and the partition tests of the conformance suite); one tenant never reads, changes or deletes another's rows,
  also not with hostile values (`'`, `;`, `$$`, U+0000, 1 MB of text) in any argument.
* **Permissions** are the host's: which user may use which tenant, namespace or source, workflows and migrations.
  A host that wants a second line of defense can enable row-level security with `tenant_setting`.
* **Roles.** Run the stores with a role that owns nothing and may not create anything; run `migrate` (or your own
  runner with `schema_steps`) with the owner or a migration role.

PostgreSQL text cannot hold U+0000: the index stores U+FFFD in its place, the file store keeps bytes as they are,
and the memory refuses such a text (`MemoryToolError`) without writing.

## Reference

::: orkexa.backends.postgres.PostgresLayout

::: orkexa.backends.postgres.ConnectionSource

::: orkexa.backends.postgres.PsycopgSource

::: orkexa.backends.postgres.PostgresIndex

::: orkexa.backends.postgres.PostgresFileStore

::: orkexa.backends.postgres.PostgresMemory

::: orkexa.backends.postgres.verify_schema

::: orkexa.backends.postgres.schema_steps

::: orkexa.backends.postgres.migrate

::: orkexa.backends.postgres.SchemaReport

::: orkexa.backends.postgres.SchemaProblem
