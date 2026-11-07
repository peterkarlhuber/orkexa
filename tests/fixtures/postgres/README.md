# PostgreSQL backend fixtures (MUP-ONA B5)

Test data for `orkexa.backends.postgres`, taken from nodestro-runtime `76f9cb2` (runtime PR #138 merged, MUP-ONA
B4), where the backend came from. The tests are in `tests/backends/postgres/`.

| File | Purpose |
|---|---|
| `nodestro_catalog.json` | Nodestro's statement catalog (`src/nodestro_runtime/pg_generic/_sql.py` at `76f9cb2`), every entry rendered with the prefix `nodestro_`, with its parameters, kind and flags. T-PAR-2 (`test_catalog.py`) compares Orkexa's catalog rendered with `nodestro_` against it. |
| `make_nodestro_catalog.py` | Generator of `nodestro_catalog.json` from a nodestro-runtime source tree (`git archive`). |
| `nodestro_migrations/0001..0010_*.sql` | Byte-identical copies of `src/nodestro_runtime/migrations/postgres/` at `76f9cb2`. T-SCH-2 (`test_schema.py`) applies them to one schema and `migrate(PostgresLayout("nodestro_"))` to another and compares the knowledge and memory tables. |
| `nodestro_migrations/SHA256SUMS`, `*.sha256` | Checksums (`sha256sum` format); the tests check them. The migration checksums equal nodestro-runtime's `tests/fixtures/migration_checksums.json` (MUP-ONA T-MIG-1). |

Do not edit these files. A new copy comes from nodestro-runtime, with its commit named here and in the generator
call (`origin` in `nodestro_catalog.json`).
