# SQL rule engine fixtures (MUP-ONA B5)

Test data for `orkexa.testing.sqlcheck`, taken from nodestro-runtime `2ef5b3b` (runtime PR #137, MUP-ONA B4), where
the engine came from. The tests are in `tests/test_sqlcheck.py`.

| File | Purpose |
|---|---|
| `sql_parity_before.json` | Byte-identical copy of nodestro-runtime `tests/fixtures/sql_parity_before.json`: the 46 statements of `nodestro_runtime/postgres_tables.py` as Nodestro's tenant SQL check saw them before B4, with Nodestro's tenant tables. T-SQLC-1 renames the tables to `t_*` and the column to `org_id`. |
| `tenant_filter_cases.json` | Nodestro's 112 mandatory cases and 37 controls (plus the 4 `ANY` samples of A11) from nodestro-runtime `tests/test_tenant_filters.py`, as the statements Nodestro's check gives its engine and the engine's findings (T-SQLCHECK). |
| `make_tenant_filter_cases.py` | Generator of `tenant_filter_cases.json` from a nodestro-runtime source tree (standard library and pglast). |
| `*.sha256` | Checksums (`sha256sum` format); `tests/test_sqlcheck.py` checks them. |

Do not edit the JSON files. A new copy comes from nodestro-runtime, with its commit named here and in the generator
call (`origin` in `tenant_filter_cases.json`).
