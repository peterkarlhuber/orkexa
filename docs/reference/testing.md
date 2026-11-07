# Testing helpers and conformance

`orkexa.testing` holds helpers for your tests. Orkexa's core never imports it (an import-linter contract checks
this), and it needs no pytest.

## Scripted SDK clients

`ScriptedClient` stands in for a provider SDK client such as `anthropic.Anthropic` or `openai.OpenAI`. Every
method path (`client.messages.create`, `client.beta.messages.batches.retrieve`) answers from a script and is
recorded, so a test needs neither the network nor hand-written doubles:

```python
from types import SimpleNamespace

from orkexa import Agent
from orkexa.testing.fakes import ScriptedClient

ended = SimpleNamespace(id="batch-1", processing_status="ended")
client = ScriptedClient({"messages.batches.create": [ended], "messages.batches.results": [[]]})
agent = Agent(model="anthropic:claude-opus-5-5", client=client)
...
(call,) = client.calls_to("messages.batches.create")
assert call.kwargs["requests"][0]["custom_id"]
```

An answer is returned as it is; an exception (instance or class) is raised; a callable gets the
`ScriptedCall` and returns the answer. A call with no answer left fails with `AssertionError`.
`asynchronous=True` returns awaitables (for `AsyncAnthropic`-style clients). For whole agent turns use the
scripted provider `FakeModel` (`orkexa.providers.fake`).

::: orkexa.testing.fakes.ScriptedClient

::: orkexa.testing.fakes.ScriptedCall

## The storage conformance suite

`orkexa.testing.conformance` checks a backend of `Index`, `FileStore` or `MemoryBackend` against the contract in
[Memory, files and retrieval](storage.md#the-storage-contracts). Orkexa runs it against its SQLite and local
classes, Nodestro against its PostgreSQL classes, and you can run it against your own.

A **factory** opens a new instance for a `Request`:

| Field | Meaning |
|---|---|
| `workdir` | an empty directory of this case; every instance of the case gets the same one (open the same storage) |
| `scope` | a fresh identifier of this case run; a backend on a shared server uses it to keep runs apart (a database name, a table prefix, a namespace) |
| `partition` | the partition (tenant) the instance belongs to; partitioned backends must keep partitions apart |
| `embeddings` | the embedding model for an index (`None`: full text only) |

`run_cases(factory, cases, expect=...)` first opens a probe instance and compares its `capabilities` with
`expect`: a backend that reports fewer capabilities than expected fails the run, as does a constructor that
takes `tenant_id` without reporting `partitioned`. Then each case runs with a fresh `workdir` and `scope`; a case
whose `needs` the backend does not report is skipped, with the reason in the report.

| Case list | Contract | Plan IDs |
|---|---|---|
| `INDEX_CASES` | add, search, delete, sources, scores, query limits, hostile input | T-CONF-1, T-CONF-2 |
| `VECTOR_CASES` | vector checks (NaN, infinity, zero, dimension), cosine scores, rounding, ties | T-CONF-3 |
| `REINDEX_CASES` | `ReindexableIndex`: states, identity change, interruption, races | T-CONF-4 |
| `FILE_STORE_CASES` | `FileStore` | T-CONF-5 |
| `MEMORY_CASES` | `MemoryBackend`, path rules, limits | T-CONF-6 |
| `PARTITION_INDEX_CASES`, `PARTITION_FILE_STORE_CASES`, `PARTITION_MEMORY_CASES` | partitions never see each other | T-PART |

`ALL_INDEX_CASES`, `ALL_FILE_STORE_CASES` and `ALL_MEMORY_CASES` combine them per contract. Security cases
(partitions, query limits, vector checks) can only depend on `partitioned`, `vectors` and `reindex`; the report
marks them when they are skipped.

### A pytest adapter

The package needs no pytest. A thin adapter in your test suite runs the cases, compares the skipped cases with
an expectation per CI job, and fails with the report's summary:

```python
from pathlib import Path

import pytest

from orkexa import SQLiteIndex
from orkexa.testing.conformance import ALL_INDEX_CASES, Request, load_expected_skips, run_cases

EXPECTED = Path(__file__).parent / "expected_skips.json"


def sqlite_index(request: Request) -> SQLiteIndex:
    return SQLiteIndex(
        request.workdir / "index.db", namespace=request.partition, embeddings=request.embeddings
    )


def test_the_index_conforms(tmp_path: Path) -> None:
    report = run_cases(
        sqlite_index,
        ALL_INDEX_CASES,
        expect={"vectors", "reindex", "partitioned", "concurrent_writers"},
        target="sqlite-index",
        workdir=tmp_path,
    )
    report.check(load_expected_skips(EXPECTED, "test", "sqlite-index"))
```

The expectation file maps each CI job to its targets and the cases they skip, each with a reason; a job may
name another job whose expectations it shares:

```json
{
  "jobs": {
    "test": {"sqlite-index": {}, "my-store": {"files.concurrent_adds_lose_nothing": "one process only"}},
    "nightly": "test"
  }
}
```

A case that is skipped but not listed, or listed but run, fails `report.check()`: a lost capability cannot
turn into a silent skip. `report.to_dict()` gives the report as JSON (Orkexa's CI saves it as an artifact).

### Own cases

A `Case` has a stable name (`<area>.<what_it_checks>`), the capabilities it needs, a run function that takes a
`Context`, and the contract its instances implement. `Context.make(partition, embeddings=...)` opens an instance
on the case's storage; `expect`, `expect_equal`, `expect_error` and `expect_code` raise `CaseFailure`.

::: orkexa.testing.conformance.run_cases

::: orkexa.testing.conformance.Request

::: orkexa.testing.conformance.Report

::: orkexa.testing.conformance.Case

::: orkexa.testing.conformance.Context

::: orkexa.testing.conformance.Outcome

::: orkexa.testing.conformance.load_expected_skips

::: orkexa.testing.conformance.KeywordEmbeddings

## The tenant SQL rule engine

`orkexa.testing.sqlcheck` checks that PostgreSQL statements on tables partitioned by tenant are bound to one
tenant: every reference to such a table must be filtered by `<partition column> = <parameter>` as a top-level
`AND` condition, writes must set the column from a parameter and never move a row to another tenant. Use it in
your tests on the statements your code sends (a statement catalog, rendered queries, SQL rebuilt from source).
It needs PostgreSQL's parser, pglast:

```bash
pip install "orkexa[sqlcheck]"
```

Importing the module needs no parser; without pglast the first `check_statement` call raises `ImportError`
naming the extra.

```python
from orkexa.testing.sqlcheck import check_statement

TENANT_TABLES = {"docs", "notes"}


def test_the_statements_are_bound_to_one_tenant() -> None:
    for sql in [
        "SELECT body FROM docs WHERE tenant_id = %s AND id = %s",
        "INSERT INTO notes (tenant_id, doc, text) VALUES (%s, %s, %s)",
    ]:
        assert check_statement(sql, tenant_tables=TENANT_TABLES) == [], sql


findings = check_statement(
    "SELECT body FROM docs WHERE id = %s OR tenant_id = %s", tenant_tables=TENANT_TABLES
)
# [Finding(kind='leak', position=17,
#          message='docs is not bound to one tenant by a top-level `tenant_id = <param>` conjunct')]
```

| Parameter | Meaning |
|---|---|
| `tenant_tables` | the tables partitioned by tenant (any case; a schema-qualified name counts by its last part) |
| `column` | the partition column, an unquoted lower-case name (default `tenant_id`) |
| `placeholder` | `"psycopg"` (`%s`, `%(name)s`; `%%` is `%`, the default) or `"postgres"` (`$1`) |
| `allow_any` | `column = ANY(<param>)` binds too (off by default: the array may hold several tenants) |

Each `Finding` has a `kind`, a `position` (character offset in the text as given, best effort) and a `message`:

| Kind | Meaning |
|---|---|
| `leak` | a table reference or write that may reach another tenant's rows |
| `unparsed` | the parser rejects the text (never an exception) |
| `unsupported` | a statement the rules cannot decide: `TRUNCATE`, `COPY`, `MERGE` and the like on a tenant table, or SQL in a string (`DO $$ ... $$`, `query_to_xml('...')`) that names one |

A statement passes only when the list is empty: `unparsed` and `unsupported` are failures too. There is no
allowlist and no configuration file; a statement that is cross-tenant by design is the caller's decision.
Schema statements (`CREATE`, `ALTER`, `CREATE INDEX`, `GRANT`, `DROP`, ...) pass. What the engine cannot see:
whether the parameter holds the right tenant, SQL inside functions, views, triggers and rules on the server, and
tables reached under a name not in `tenant_tables`. The module documentation lists the rules in full.

The engine imports only the standard library and pglast, and no other Orkexa module (an import-linter contract
checks this), so a host can use it without Orkexa's other dependencies. It comes from Nodestro's tenant SQL check
(MUP-ONA B4/B5); Orkexa's tests run Nodestro's 112 mandatory leak cases and 37 controls against it.

::: orkexa.testing.sqlcheck.check_statement

::: orkexa.testing.sqlcheck.Finding
