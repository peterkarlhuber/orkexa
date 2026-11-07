# ADR 0015 – Statement catalog, partition guard and the tenant SQL rule engine

Status: accepted (2026-11-07, MUP-ONA B5; decision O5 confirmed at gate G0 on 2026-11-06, binding as E-14)

## Context

Orkexa gains PostgreSQL backends (`orkexa.backends.postgres`) that keep many tenants in one set of tables. The
whole security promise of such a backend is the partition: every statement on a tenant table must be bound to one
tenant, and the tenant must be the one the instance was built for. Nodestro proved this with a checker that
rebuilt SQL from Python source code (f-strings, string concatenation, helper calls) and judged it; that
reconstruction was fragile, Nodestro-specific and could not be offered to other hosts. MUP-ONA B4 moved
Nodestro's generic PostgreSQL code to a finite statement catalog with an executor and a partition guard, and
separated the generic rules from the reconstruction. B5 brings both to Orkexa.

## Decision (O5)

* **A statement catalog.** Every statement of the PostgreSQL backends is an entry of `CATALOG`
  (`Statement(name, template, params, kind, partitioned, rows)`): a SQL template whose table names are
  placeholders, rendered for a `PostgresLayout` with `psycopg.sql.Identifier`, and whose values are all named
  parameters. No other code of the backend runs SQL; the only exceptions are the constant statements of
  `schema.py` (catalog queries, the migration steps) and of `connection.py` (`set_config`), which a test lists.
* **An executor with a partition guard.** The stores run catalog entries by name only. Before a partitioned entry
  runs, the guard checks that its `tenant_id` parameter is a usable tenant id and exactly the tenant of the
  instance; otherwise it raises `RetrievalError("partition.missing")` without touching the database.
* **A generic rule engine in `orkexa.testing.sqlcheck`.** `check_statement(sql, *, tenant_tables, column,
  placeholder, allow_any=False)` decides with pglast whether a statement binds every tenant table to one tenant
  parameter. It has no allowlist and no configuration file, imports no other Orkexa module and is a test helper
  (extra `sqlcheck`), not a runtime dependency.
* **Proof in Orkexa's CI**, not only in Nodestro's: every catalog entry passes the engine (T-CAT-1), a mutation of
  each entry is caught (T-CAT-2, uploaded as `mutation-probe.json`), the catalog holds exactly the listed entries
  (T-CAT-3), no other code runs SQL and none uses `sql.Literal` (T-CAT-4), the guard refuses a missing or foreign tenant (T-GUARD-1..3), and
  the catalog rendered with the prefix `nodestro_` equals Nodestro's catalog of runtime 76f9cb2, exactly and after
  pglast normalization (T-PAR-2, `tests/fixtures/postgres/nodestro_catalog.json`). The dynamic partition cases of
  the conformance suite (T-PART-1..9) run against real servers with and without pgvector, as superuser and as a
  runtime role without DDL rights.
* **What stays in Nodestro:** the reconstruction of SQL from Python source code, for Nodestro's own
  product-specific statements, which uses the same engine.

## Consequences

* The SQL a backend can run is a finite, reviewable list. A new statement is a catalog entry and is checked by the
  engine and the mutation probe automatically; a statement outside the catalog cannot run.
* Hosts can check their own SQL with the same engine without adopting Orkexa's backends.
* The guard is a run-time check of every partitioned statement (a dictionary lookup and a comparison); no
  measurable cost next to a database round trip.
* The engine cannot see what the server does behind a name (functions, views, triggers, rules) or whether the
  bound parameter holds the right tenant; the guard covers the latter for the catalog, the host covers who may use
  a tenant (MUP-ONA E-6: partition yes, permissions no).
* Rejected alternatives: an allowlist of cross-tenant statements in the engine (a configuration that silently
  grows; a host decides such statements itself); building SQL from fragments at run time (cannot be checked
  before it runs); row-level security as the only line of defense (stays optional, see `tenant_setting`, and is
  assessed for Nodestro after B6, O9).
