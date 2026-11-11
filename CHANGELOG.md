# Changelog

## 0.2.0 – unreleased

First release of Orkexa. The version was `0.1.0` until MUP-ONA B0; no 0.1.0 was published, so its entries are
listed here. The development version is `0.2.0.dev0`.

### Added: blocked-module tests per Tabrion edge (TABRION QA-18, 2026-11-08)

* `tests/packaging/test_optional_edges.py` (AC-3): one fresh `python -I` per optional edge of the K-33 list, with
  its modules blocked (`sys.modules[name] = None`): the core imports without `tabrion` (until MIG-ORK-3), and the
  core and `orkexa.backends.postgres` import without psycopg and `tabrion.postgres` (the backend names
  `orkexa[postgres]` only on use). A meta test keeps the edges equal to K-33 and to `pyproject.toml`; a mutation
  probe per edge adds a top-level import to a copy of the package and expects red. Tests only.

### Release rules (MUP-ONA B0, 2026-11-06)

* Tags are named `orkexa-vX.Y.Z` (annotated, on the release's merge commit) and are set only after the owner
  approves; until then the merge commit's SHA stands for the version. No tag exists yet.
* Versions follow SemVer from 0.2.0. A kept alias (such as `orkexa.retrieval._pdf_pages`) is removed only in a minor
  release, with an entry here that names the alias and its replacement.
* Nothing is published on PyPI.
* `tests/test_public_api.py` (T-API-1) fixes the public names of `orkexa`, `orkexa.retrieval`, `orkexa.files`,
  `orkexa.memory`, `orkexa.embeddings` and the deeper paths Nodestro imports, and checks that re-exports are the same
  objects. import-linter (in the `dev` extra, `lint-imports` in CI) keeps the core modules from importing
  `orkexa.backends.postgres` or `orkexa.testing` (T-PKG-0). See "Versioning and releases" in
  `docs/development/releasing.md`.

### Added: import contracts for the Tabrion edge list K-33 and their mutation probe (Tabrion QA-16, 2026-11-11)

* Two new import-linter contracts. Orkexa never imports Axentra, Nodestro or a Studio package (`axentra*`,
  `nodestro*`, `tabrion_studio`). The core never imports `tabrion` until Tabrion MIG-ORK-3 is merged (owner decision
  E-ORK-CORE). The core is the module list of "Core modules do not import orkexa.backends.postgres" plus
  `orkexa.testing`, which ships in the core wheel; the PostgreSQL backend may import `tabrion` (K-33:
  `orkexa[postgres]` -> `tabrion[postgres]`). `[tool.importlinter]`
  now sets `include_external_packages = true`, which import-linter requires for forbidden packages outside Orkexa.
* `tests/architecture/test_forbidden_probe.py` copies the package, adds one forbidden import per case (15 cases,
  every contract at least once) and requires `lint-imports` to name the broken contract. It also checks that the
  backend's edge to `tabrion` stays green, and scans for `axentra*`, `nodestro*`, `levuvo*` and `*_studio` imports
  by prefix. `tests/test_public_api.py` now expects six contracts.

### Fixed: local thinking models running out of tokens or tool rounds (2026-11-08)

* Ollama: reasoning is sent back only on the latest assistant message (`ChatMessageRules.reasoning_replay`,
  `"all"` by default, `"latest"` for Ollama). Since the o400 fix (3f63b03) every earlier thinking text was replayed
  and rendered into the prompt, so a multi-step tool conversation grew by thousands of tokens per round; when it
  passed the context window, Ollama silently dropped the middle of the conversation (the task included) and small
  qwen3.5 models started calling the same tool again. Other OpenAI-compatible servers keep replaying everything.
* New agent options `on_max_tool_rounds` and `on_reasoning_overflow` (`"raise"` by default, unchanged behavior)
  with `"answer"`: one tool-free final request at the last tool round, and one request without reasoning after a
  response stopped at `max_tokens` while reasoning. New event `LimitHandled`; new provider hooks
  `Provider.tool_free_answer` (Ollama ignores `tool_choice="none"`, so tools are omitted there; Bedrock has neither)
  and `Provider.without_reasoning` (Ollama: `reasoning_effort="none"`).
* `IncompleteResponseError` carries `stop_reason`, `max_tokens`, `output_tokens`, `reasoning` and `answered`.

### Fixed: two test races seen on master CI (2026-11-07)

* `test_cancel_kills_the_container` waited for any container with the owner label, so it could stop on the
  self-test container (same label), which ends a moment later; the next `docker ps` then listed nothing
  (`assert []`, macOS on Intel). It now waits for the tool's own `orkexa-sandbox-*` container. The fake `docker`
  replaces its state files in one step and appends each log line with one write, and `FakeDocker.calls` reads only
  complete lines, so a test that polls while the fake writes no longer reads a torn file.
* The managed-server tests' `_alive` checked that `/proc/<pid>/stat` exists and then read it; init can reap the
  killed, orphaned runner between the two, which raised `FileNotFoundError` (Python 3.14). A vanished entry now
  counts as gone. Test: `test_alive_counts_a_process_reaped_during_the_check_as_gone`.
* `orkexa-local serve` could ignore a SIGTERM: its handler raised `KeyboardInterrupt`, and when the signal arrived
  while a finalizer ran (an HTTP response closed during the status check), Python dropped the exception
  ("Exception ignored in ...") and the server kept running (seen as a timeout of
  `test_cli_serve_stops_the_server_on_sigterm` on macOS). The handler now only sets a flag that the serve loop checks.
  Test: `test_cli_serve_stops_on_a_sigterm_delivered_inside_a_finalizer`.

### Fixed: a search no longer fails when another instance deletes a ranked chunk (2026-11-07)

* `SQLiteIndex.retrieve` reads its ranking and the rows of its hits in one read transaction (one WAL snapshot).
  Before, an instance or process that deleted a ranked chunk between the two reads made the search fail with
  `TypeError: cannot unpack non-iterable NoneType object` (seen once in T-PART-CC, `sqlite-index-fts`).
  Test: `test_a_chunk_deleted_by_another_instance_during_a_search_is_no_error`.

### Added: the PostgreSQL backends `orkexa.backends.postgres` (MUP-ONA B5, 2026-11-07)

* Added: `PostgresIndex`, `PostgresFileStore` and `PostgresMemory`, the storage contracts on PostgreSQL with many
  tenants in one set of tables (`tenant_id` is the partition, MUP-ONA E-6: partition yes, permissions no), full
  text (`tsvector`) and vectors (`real[]`, ranked by pgvector where it is installed, else in Python), reindex, and
  the capabilities `partitioned` and `concurrent_writers` (plus `vectors` and `reindex` with an embedding model).
  See `docs/reference/backends-postgres.md`.
* Added: `PostgresLayout(prefix="orkexa_")` names the five tables (only the prefix is configurable, E-12);
  `ConnectionSource` (protocol) and `PsycopgSource` (psycopg 3 with a small pool of its own, transactions joined
  per thread, optional transaction-local `tenant_setting` for row-level security).
* Added: the statement catalog (`CATALOG`, `Statement`, `render`) with an executor and a partition guard: the
  stores run only catalog entries, values are always bound, and a statement whose `tenant_id` is not the
  instance's raises `RetrievalError("partition.missing")` before it runs
  (ADR 0015, `docs/architecture/decisions/0015-statement-catalog-and-sql-rule-engine.md`).
* Added: schema version 1 as data (exactly Nodestro's migrations 0002 and 0010 for any prefix): `verify_schema`
  (read-only, never raises, returns a `SchemaReport` of `SchemaProblem`s), `schema_steps` and `migrate` (only when
  called, under an advisory lock, version table `<prefix>orkexa_schema`; `schema.mismatch` with reason
  `permission`, `too_new` or `unrecorded`). No DDL at run time and no `CREATE EXTENSION` (E-13).
* Taken over from nodestro-runtime 76f9cb2 (`pg_generic`); the first commit is the unchanged code. Changes: a
  `PostgresLayout` instead of a prefix string, Orkexa's docstrings and line length, the catalog texts unchanged.
  Three fixes found by the new tests: a tenant id with U+0000 is refused (PostgreSQL text cannot hold it); the
  stores of one `PsycopgSource` share its `lock` by default, so two instances in one process no longer lose each
  other's memory or file change; and a chunk deleted between a reindex's vector store and its foreign key check
  no longer ends the reindex with a foreign key violation (the store runs again without it; T-PART-CC hit this
  in about one of four runs).
* New extra `postgres = ["psycopg>=3.2,<4"]`; `dev` gains `psycopg[binary]`. psycopg is imported on the first
  connection only: `import orkexa` never imports the package, `import orkexa.backends.postgres` works without
  psycopg, and `PsycopgSource()` then raises `ImportError` naming `orkexa[postgres]`. The first import-linter
  contract names the SQLite subpackage instead of all backends, so the PostgreSQL modules may import each other.
* tests: `tests/backends/postgres/` (T-CAT-1..4, T-GUARD-1..3, T-PAR-2 against Nodestro's catalog, T-PG-ID,
  T-SCH-1..3 with Nodestro's migrations as fixtures, T-DDL-1..3, T-PG-5, T-PG-6, T-PG-4, T-RANK-1),
  `tests/conformance/test_postgres_conformance.py` (T-CONF-*, T-PART-1..9) and
  `tests/packaging/test_core_without_extras.py` (T-PKG-1, T-PKG-2). Server tests carry the marker `postgres`.
* CI: jobs `pg-a` (`postgres:17`) and `pg-b` (`pgvector/pgvector:pg17`, `ORKEXA_REQUIRE_PGVECTOR=1`), images by
  digest, Python 3.11 and 3.14, each as superuser and as a runtime role without DDL rights, with
  `ORKEXA_REQUIRE_POSTGRES=1` and skip lists; job `core-no-extras`. The other jobs deselect `-m postgres`; the
  lowest-versions job pins psycopg and psycopg-binary 3.2.0; the audit covers `postgres`.

### Added: the tenant SQL rule engine `orkexa.testing.sqlcheck` (MUP-ONA B5, 2026-11-07)

* Added: `orkexa.testing.sqlcheck.check_statement(sql, *, tenant_tables, column="tenant_id",
  placeholder="psycopg", allow_any=False)` returns a list of `Finding(kind, position, message)` with `kind` in
  `leak`, `unparsed`, `unsupported`: every reference to a table partitioned by tenant must be bound by
  `<column> = <param>` as a top-level `AND` condition (or by equality with a bound reference), and writes must
  neither omit nor move the partition column. No allowlist and no configuration file; the table set, the column
  and the placeholder style (`%s` or `$1`) are parameters. Also exported: `KINDS`, `PLACEHOLDERS`,
  `ANY_REFUSED`, `MISSING_PGLAST`, `to_postgres_params`. See "The tenant SQL rule engine" in
  `docs/reference/testing.md`.
* Taken over without a change of behavior from nodestro-runtime's `sqlcheck_rules` (2ef5b3b, runtime PR #137);
  the first commit is the unchanged file. Only docstrings, line wrapping and the missing-pglast message changed.
* New extra `sqlcheck = ["pglast>=7,<9"]`, also in `dev`. The engine imports pglast only on first use: `import
  orkexa` and `import orkexa.testing.sqlcheck` work without it, and the first check raises `ImportError` naming
  `orkexa[sqlcheck]`.
* A fourth import-linter contract keeps the engine free of other Orkexa modules (T-PKG-0 now expects 4).
* tests: `tests/test_sqlcheck.py` with the carried-over engine tests (T-SQLC-1: foreign column `org_id` and tables
  `t_docs`/`t_notes`, Nodestro's 46 parity statements renamed), T-SQLCHECK (Nodestro's 112 mandatory cases and 37
  controls of `tests/test_tenant_filters.py` as data in `tests/fixtures/sqlcheck/tenant_filter_cases.json`, with
  its generator; the engine gives the very same findings) and T-PKG-1 for the engine (pglast blocked). CI: the
  Windows ARM64 job installs pglast, the lowest-versions job pins `pglast==7.20`, the audit covers `sqlcheck`.

### Changed: the managed Ollama ends with its application (2026-11-07)

* `ManagedOllama` ties the server's lifetime to the process that started it: on Linux and macOS a watcher process
  (`orkexa/local/_lifeline.py`, the same interpreter, a session of its own) stops the server's process group when
  the application ends without stopping it (crash, OOM kill, `SIGKILL`); on Windows the server runs in a job object
  with kill-on-close, so it and its runners end with the application. The reclaim of a server named in
  `ollama.state.json` at the next start stays as the second layer (POSIX).
* tests: opt-in live smoke tests for OpenAI, Gemini and vLLM (`tests/test_provider_live.py`, `pytest -m live`);
  `tests/test_sandbox_lifecycle.py` uses a temporary directory of its own (another test run on the machine made it
  fail).

### Changed: the SQLite backends moved to `orkexa.backends.sqlite` (MUP-ONA B2, 2026-11-07)

* Moved, no behavior change: `SQLiteIndex`, `SQLiteFileStore` and `SQLiteMemory` are now defined in
  `orkexa.backends.sqlite` (modules `index`, `files`, `memory`), their canonical path. `orkexa.SQLiteIndex`,
  `orkexa.retrieval.SQLiteIndex`, `orkexa.files.SQLiteFileStore`, `orkexa.memory.SQLiteMemory` and the other old
  import paths return the very same classes (`is`), also for type checkers. Only `__module__` changes (for example
  `orkexa.backends.sqlite.index` instead of `orkexa.retrieval`); constructors, SQL, tables, pragmas, file
  permissions and the database format stay as they were. The private helper `_fts_query` moved along
  (`orkexa.backends.sqlite.index._fts_query`).
* The contract modules resolve these names on first access (PEP 562 `__getattr__`) and never import a backend when
  they load; a third import-linter contract checks it. The private attributes Nodestro uses (`_connection`,
  `_lock`, `_save`, `_read`, `_files`) are kept and marked "preserved for compatibility".
* tests: T-API-1 (identity of every import path), T-API-2 (each module imported first in a fresh interpreter),
  T-API-3 (`mypy --strict` over the old paths, `tests/backends/sqlite/mypy_probe.py`), T-SQL-11 (the private
  attributes).

### Added: generic storage contracts (MUP-ONA B1, 2026-11-07)

* Added: the score semantics of every `Index` backend in `Index.search` (full text: reciprocal rank
  `1 / (60 + rank)`; vector: cosine rounded to 6 digits; hybrid: `fuse_hybrid`; ties by chunk id) and
  ADR 0014 (`docs/architecture/decisions/0014-storage-contracts-and-conformance.md`).
* Added: `orkexa.retrieval.tokenize`, `fold` (the query side of full-text search; `query_words` uses them, results
  unchanged) and `fuse_rrf(rankings, k=60)` (`SQLiteIndex` uses it, results unchanged).
* Added: `orkexa.embeddings.validate_vector(vector, identity)`: `vector.dimension`, `vector.invalid` (NaN or
  infinity), `None` for a zero vector, unit length unless `identity.normalized`. `SQLiteIndex` calls it without a
  change of results. The new `VectorError` (a `RetrievalError` and a `ValueError`) carries these codes.
* Added: `RETRIEVAL_ERROR_CODES`, the one code set of every storage backend, with the new codes `vector.invalid`,
  `partition.missing` and `schema.mismatch` (PostgreSQL backends).
* Added: `capabilities` (`vectors`, `reindex`, `partitioned`, `concurrent_writers`) on `Index`, `FileStore` and
  `MemoryBackend`; the SQLite classes report `partitioned` and `concurrent_writers`, `SQLiteIndex` also `vectors`
  and `reindex` as configured. Added: the protocol `orkexa.retrieval.ReindexableIndex`.
* Changed: the query limits `MAX_QUERY_CHARS`/`MAX_QUERY_WORDS` moved from `SQLiteIndex` into `Index`
  (`max_query_chars=`/`max_query_words=`, `Index.check_query`) and apply to every `Index` subclass: their
  `search` and `retrieve` refuse an overlong query with `query.too_long` before anything else. `SQLiteIndex`
  behaves as before.
* Changed: an invalid vector in `SQLiteIndex` raises `VectorError` (still a `ValueError`, now with a code); a query
  vector of another length raises it too (still a `RetrievalError` with `vector.dimension`).
* Added: the public conformance suite `orkexa.testing.conformance` (no pytest needed): `run_cases(factory, cases,
  expect=...)` runs the cases of `Index` (`INDEX_CASES`, `VECTOR_CASES`, `REINDEX_CASES`), `FileStore`
  (`FILE_STORE_CASES`), `MemoryBackend` (`MEMORY_CASES`) and partitions (`PARTITION_*_CASES`) against a backend
  and returns a `Report`; a backend that reports fewer capabilities than expected, or takes `tenant_id` without
  reporting `partitioned`, fails the run. `load_expected_skips` compares the skipped cases with an expectation per
  CI job (`tests/conformance/expected_skips.json` here). Every SQLite and local class passes, also on copies of
  the FX-SQ-0 legacy files. See "Testing helpers and conformance" in the API reference.
* Added: `orkexa.testing.fakes.ScriptedClient`, a scripted, recording stand-in for provider SDK clients;
  `tests/test_capabilities_and_batches.py` uses it instead of hand-written doubles.

### Added: PDF budgets and isolated PDF extraction (MUP-ONA A1, 2026-11-06)

* Added: `orkexa.retrieval.pdf_pages(data, *, max_pages=2000, max_chars=5_000_000)` stops between pages and raises
  `RetrievalError("document.too_complex")` with `reason` `pages`, `chars` or `limit` (a pypdf limit);
  `Index.add_file` uses it for every backend.
* Added: `pdf_pages_isolated(data, *, timeout=30.0, max_memory_mb=512, max_pages, max_chars)` extracts in a
  separate interpreter (`python -I -m orkexa._pdf_worker`, started with `subprocess.Popen`, not `fork` or
  `multiprocessing`) with an allowlisted environment, no inherited descriptors, an empty private working directory
  and resource limits (Linux: memory, CPU, open files, file size); at the timeout the process group gets `SIGTERM`,
  then `SIGKILL`. Reasons: `timeout`, `memory`, `pages`, `chars`, `limit`, `crash`. `PdfLimits` holds the defaults.
* Changed: very large PDFs are refused (more than 2000 pages or 5,000,000 characters of text); the text of other
  PDFs is unchanged. `RetrievalError` has a `reason` attribute (`None` except for `document.too_complex`).
* Changed: the `pdf` extra needs `pypdf>=6.21.4`, the first release without a known advisory (OSV, 2026-11-06).
* Deprecation notice: `orkexa.retrieval._pdf_pages` stays as an alias of `pdf_pages` (no warning yet); it is
  removed no earlier than in the next-but-one minor version.

### Fixed: concurrent SQLite deletes; Added: query limits and storage error codes (MUP-ONA A10, 2026-11-06)

* Fixed (PS-8): two `SQLiteIndex` instances or processes deleting the same source no longer fail with
  "database disk image is malformed": `delete()` reads the rows inside its write transaction. Indexing a file
  again (`add_file`) replaces its chunks in one transaction, starting a reindex and dropping vectors are
  transactions, and `SQLiteFileStore.add()`/`delete()` run in `BEGIN IMMEDIATE`, so two processes adding the same
  file no longer overwrite each other's metadata. `__len__`, `sources()` and `FileStore.__contains__` take the lock.
* Changed: `SQLiteIndex` refuses a query over `MAX_QUERY_CHARS` (1000 characters) or `MAX_QUERY_WORDS` (64
  distinct words) with `RetrievalError("query.too_long")`, checked before embedding and locking; the limits are
  `orkexa.retrieval` constants and constructor arguments (`max_query_chars=`, `max_query_words=`).
* Changed: a query vector of another length than the stored vectors raises `RetrievalError("vector.dimension")`.
* Changed: `SQLiteIndex`, `SQLiteFileStore` and `SQLiteMemory` refuse a database path that is a symbolic link (or
  whose `-wal`, `-shm` or `-journal` file is one) with `RetrievalError("storage.symlink")`, and report a damaged
  file as `RetrievalError("storage.corrupt")` (the `sqlite3.DatabaseError` is the cause) instead of a raw
  `sqlite3.DatabaseError`. Messages contain no paths. The codes are listed in the `RetrievalError` docstring and
  in the guide (`docs/guides/memory-files-rag.md`, "SQLite limits and error codes").
* Fixed: extension loading is switched off again when loading sqlite-vec fails.
* Valid queries return the same results as before (FX-RANK-0); the database format is unchanged.

### Tests: legacy fixtures and skip expectations (MUP-ONA A0, 2026-11-06)

* tests: legacy fixtures and skip expectations. `tests/fixtures/legacy/FX-SQ-0` holds SQLite databases written by
  today's `SQLiteIndex` (with and without sqlite-vec), `SQLiteFileStore` and `SQLiteMemory` with their SHA-256
  checksums, generated by `tests/fixtures/legacy/make_legacy_sqlite.py`; `tests/test_legacy_fixtures.py` checks them.
* CI compares the skipped tests of every pytest job with `tests/ci/skipped-<job>.txt` (`scripts/check_skips.py`):
  a new or a vanished skip fails the job.
* `tests/backends/sqlite/test_robustness.py` reproduces the concurrent FTS5 delete defect (PS-8) with two instances
  and two processes, as a strict expected failure until it is fixed.

### Changed: OpenAI-compatible endpoints no longer follow redirects (2026-11-06)

* `compatible`, `ollama` and `vllm` clients refuse HTTP redirects (a 3xx answer is an error that names the address),
  so a redirect cannot send the conversation or the key to another host. A caller's own `http_client` keeps its own
  redirect setting. Connection errors and timeouts name the endpoint (without credentials) and how to start the
  server (`ollama serve` or `orkexa-local serve`, `vllm serve <model>`).
* vLLM gets reasoning back in the history (`reasoning` field), as Ollama does.

### Added: registry refresh, Bedrock session guard, documented limits, Python 3.14 (2026-11-06)

* `ToolRegistry.refresh()` and `AgentRegistry.refresh()` re-read records written by other processes; a new version
  number is taken from both memory and disk, so two processes no longer reuse a version (#24).
* `BedrockOptions(require_session=True)` refuses to build a session from the host's credential chain.
* Documented `max_output_tokens` (and context windows where documented) for current OpenAI, Gemini and Bedrock
  Claude models; models without a documented number stay unknown.
* Python 3.14 is supported and tested in CI (#25). The sandbox static check also refuses the annotation evaluation
  helpers of 3.14 (`annotationlib`) and code hidden in string annotations.
* `docker/sandbox-ca.Dockerfile` (the sandbox image, kept in sync by a test) takes an optional extra CA as a
  build secret (`pip-ca`; BuildKit or Podman) for builds behind a TLS-intercepting proxy; the CA is not kept in
  the image. `docker/sandbox.Dockerfile` stays buildable with the legacy builder.
* `tests/test_upkeep.py` flags dated snapshots (built-in prices, the Bedrock capability table, the sandbox base
  image) older than 90 days; the release workflow fails on them. The release guide has an upkeep checklist.
* A weekly and manual workflow (`ollama-live.yml`) runs the live tests against the pinned Ollama runtime.

### Changed: `orkexa-local` and local-server messages (2026-11-06)

* Every `orkexa-local` option has a help text; any Orkexa error ends in one line, `error (<code>): <message>`, and
  exit code 1, without a traceback. `status` and the install, serve and convert messages name the next step.
* A model id with a provider prefix but no model (`anthropic:`) is refused with an example.
* The managed Ollama server writes its log and state files privately (mode 0600), rotates its log at 10 MB, reports
  a busy port and a failed start, and its folders are private.

### Changed: documentation and examples (2026-11-06)

* The architecture page covers Bedrock, orchestration, `orkexa.local`, embeddings and retrieval; ADR 0013 records
  the Bedrock credential, endpoint and residency design (#26).
* `examples/bedrock_agent.py` is now `examples/14_bedrock_agent.py`.

### Fixed: multi-step Ollama runs with thinking models no longer fail with HTTP 400 (2026-11-06)

* A turn in which a thinking model only reasoned (no text, no tool call) was kept as `content: null`; Ollama refuses
  that ("invalid message content type: <nil>") when the next request (for example the structured-answer request
  after the tool phase) sends it back. Assistant content is now text, `null` only with tool calls, and `""` only for a
  turn with neither. Adapters declare their message rules (`ChatMessageRules`); Ollama gets its reasoning back in
  the `reasoning` field. `check_chat_messages()` checks every Chat Completions request before it is sent and raises
  `HistoryConversionError` naming the message index and role.
* `InvalidRequestError.reason` (`messages`, `context_length`, `model_not_found`, `parameter`) and `.outline`, a
  structure-only outline of a refused conversation (roles and part kinds, never content). Live tests against a real
  Ollama: `pytest -m live` with `ORKEXA_OLLAMA_LIVE_URL`.

### Added: model limits, resumable local setup and live reasoning for every provider (2026-11-06)

* `ModelLimits` and `Provider.limits(model)` (context window, maximum output tokens, documented effort levels,
  thinking modes and budget). A request whose `max_tokens` is above the documented maximum, or whose effort the
  model does not document, is refused before it is sent; Anthropic refuses a thinking mode the model does not accept.
* Live reasoning for every adapter: `ReasoningStatus` (started/finished, kind, redacted) and `ReasoningDelta.kind`
  (`thinking` or `summary`) from Anthropic, OpenAI Responses, Gemini, Bedrock and Chat Completions endpoints
  (Ollama, vLLM). `Provider.reasoning_stream()` declares what a model's stream transmits. OpenAI
  `reasoning.summary` and Gemini `include_thoughts` are sent only to models that reason.
* The Ollama runtime download resumes after an interruption (HTTP Range), checks free space first, reports its
  phases (download, verify, unpack), names `network` and `disk_full` errors, and holds an install lock. Pull streams
  can be cancelled at once, also while Ollama sends nothing.

### Fixed: the runtime archive cannot place links outside the runtime through other links (audit ORK3-01)

* `install_runtime` checked each link of a `.tar.*` archive on its own, before any link existed, so a
  chain of links (one pointing to `.`, another through it with `..`) could point outside the runtime
  folder. Every link is now checked again against the links that exist: its folder before it is
  created, and all links once the last one exists. The pinned upstream archives are not affected.

### Fixed: `build_model` never removes a working model when a build fails (audit ORK3-09)

* The model is built and checked against the reference vectors under a temporary name and copied to the
  recipe's name only after it passed; the temporary model is removed in every case. Before, the build
  replaced an installed model of the same name first and deleted it on a mismatch (and left it on an
  unexpected error). Inputs larger than the size caps are refused (`too_large`), and malformed
  `safetensors` headers raise `OllamaError("invalid_weights")` instead of raw Python errors.

### Fixed: a managed Ollama left running by a crashed application is stopped on the next start (audit ORK3-05)

* `ManagedOllama` records the server's executable and its own process in `ollama.state.json`. `start()`
  reads that file: a server whose owner is gone and that is still `<binary> serve` leading its own process
  group is stopped with its runners before a new one starts (POSIX), instead of a second server being
  started next to it. A server of another running process is left alone; a stale file is removed.
  Windows keeps the earlier behavior. Without `/proc` (macOS) the command line is read with
  `ps -ww`, so a long install path is not cut off.

### Fixed: vector search no longer blocks the index while it embeds the query (audit ORK3-06)

* `SQLiteIndex.retrieve` in `vector` and `hybrid` mode embedded the query while holding the index lock, so
  a slow embedding server blocked adds, deletes, reindexing and other searches of the same index. The
  query is now embedded outside the lock, after the cheap checks; the active space is checked again under
  the lock before ranking, so a reindex that switched models meanwhile raises `reindex_required`.

### Fixed: vectors of an updated embedding model are not filed under its old identity (audit ORK3-07)

* `OllamaEmbeddings` resolved the model digest once per object, so after `ollama pull` (or a rebuild) a
  long-lived index wrote vectors of the new build under the old identity, mixing two vector spaces. Indexes
  now refresh the identity before they write (`add`, start and end of `reindex()`): adding to a space of
  another build raises `reindex_required`, and a model that changes during `reindex()` discards the new,
  mixed vectors (the active space stays). Searches re-resolve it after `identity_ttl` (default 60 s).
  `EmbeddingModel.refresh_identity()` is new (a no-op by default).

### Fixed: the managed Ollama server does not outlive a stop (audit ORK3-04)

* `ManagedOllama.stop()` now also ends the server's process group on POSIX, so model runners do not
  outlive a server that had to be killed; `orkexa-local serve` stops the server on `SIGTERM` (e.g. from a
  service manager), not only on Ctrl+C; `host="::1"` works (IPv6 port probe and bracketed address).

### Fixed: NaN, infinite and zero vectors (audit ORK3-03)

* A vector with NaN or infinity is refused (`ValueError` when it would be stored or searched;
  `OllamaClient.embed` raises `OllamaError("invalid_response")`). Before, it was stored, and a single such
  vector, or a zero vector, made every `vector` and `hybrid` search of the namespace fail with a
  `TypeError`. Zero vectors have no cosine distance and are now left out of the vector ranking.

### Fixed: a chunk deleted during `reindex()` no longer leaves a vector behind (audit ORK3-02)

* `reindex()` embeds a batch without holding the index lock. A chunk deleted meanwhile still got a
  vector; because SQLite reuses the largest row id, the next chunk then either could not be added
  (`UNIQUE constraint failed`) or silently got the deleted chunk's vector. Vectors are now written only
  for chunks that still exist with the embedded text, and adding a chunk replaces a stale vector of
  its id left by an earlier version.

### Added: local models with Ollama, embeddings and retrieval modes

* `orkexa.local`: `OllamaClient` (models, `/api/show` capabilities, downloads with progress and cancel,
  GGUF import, embeddings; loopback only unless `allow_remote=True`, no redirects), `install_runtime`
  (the pinned Ollama 0.40.14 from a SHA-256-verified archive, offline or with `download=True`),
  `ManagedOllama` (a child `ollama serve` on a free loopback port with its own model folder and
  `OLLAMA_NO_CLOUD=1`; an existing Ollama is only detected), offline model packages
  (`export_package`/`import_package`/`verify_package`, every file checked by SHA-256) and the
  `orkexa-local` command. Nothing runs, installs or downloads on `import orkexa` or `Agent(...)`.
  `install_runtime(artifact=artifact_for(system, machine))` also prepares the runtime of another
  platform (`find_binary(root, system)` finds `ollama.exe` of a Windows runtime on any host).
* `orkexa.embeddings`: the provider-independent `EmbeddingModel` with an `EmbeddingIdentity`
  (provider, model, digest, dimensions, instructions) and `OllamaEmbeddings` (`/api/embed`, batches,
  `truncate=false`, dimension checks, measured usage). `embedder=` callables keep working.
* Retrieval modes `fts`, `vector` and `hybrid`: `Index.retrieve(query, k, mode=, sources=)` and
  `index.tool(mode=, sources=)`. `SQLiteIndex(embeddings=...)` keeps vectors per namespace in a space
  bound to the model's identity, ranks by exact cosine similarity, filters sources before ranking and
  never falls back to full text. `reindex()` switches models safely (batches, resumable, atomic switch).
  `Hit.ranks` shows each ranking's position. The legacy `vec0` path also honours `mode`.
* Local pricing: requests to provider `ollama` get a zero *API* price with source `"local"`
  (`UsageRecord.local_inference`, `UsageSummary.local_requests`), so strict pricing and budgets do not
  refuse or ignore them as unpriced. Local operating costs are not recorded and must not be shown as 0.
  New extra `local` (`zstandard`).
* `orkexa.local.build_model` with `GGUFRecipe`: builds an Ollama model from pinned upstream
  `safetensors` weights and a pinned template GGUF header, verifies every input by SHA-256 and the
  result against reference vectors, and removes a model that does not match. `MDENSEON` builds LightOn
  mDenseOn (Apache-2.0, multilingual and cross-lingual retrieval) as `mdenseon:307m`;
  `orkexa-local build mdenseon`. New extra `convert` (`gguf`).
* `OllamaEmbeddings(document_prefix=, query_prefix=)` and `KNOWN_PREFIXES` for models trained with
  plain prefixes; `EmbeddingIdentity` records them (keys of spaces without prefixes are unchanged).

### Changed: Ollama agents with tools and structured output call their tools

* Ollama compiles `response_format` into a grammar, so with tools offered a model could only write
  JSON and never called a tool: a local agent with `output_format` and a calculator estimated every
  number itself (measured with `qwen3.5:4b`). For providers with
  `Provider.structured_output_with_tools = False` (now `ollama`), the agent offers the tools without
  the schema and, after the first answer without a tool call, makes one more request without tools
  and with the schema (`FORMAT_PROMPT`; not counted as a tool round, resumable). Other providers
  keep the single constrained request.
* `FORMAT_PROMPT` asks for the text in the language of the conversation: with the English request
  alone, `qwen3.5:4b` wrote a German workflow's analysis in English.

### Changed: hybrid search weighs full-text hits by their evidence

* `SQLiteIndex` `hybrid` mode fused BM25 and vector rankings by reciprocal rank fusion (k=60), which
  ignores how strong a full-text match is: with queries in another language than the documents, a chunk
  that shared one accidental word with the query (`days`, `de`, a number) outranked the right vector hit.
  The default is now evidence-weighted score fusion: `(1 - w) * vector + w * lexical`, with the cosine
  similarity min-max normalized over the vector candidates and `lexical` the squared IDF-weighted share
  of the query's words a chunk contains (`orkexa.retrieval.fuse_hybrid`, `lexical_evidence`,
  `word_weight`, `query_words`). On a 120-query eval (German/English lexical queries with codes and
  names, paraphrases, cross-lingual queries in six languages, hard negatives; qwen3-embedding:0.6b)
  hybrid recall@1 rose from 0.53 to 0.93 (fts 0.47, vector 0.89): as good as fts on lexical queries
  (0.98) and close to vector on cross-lingual ones.
* `SQLiteIndex(fusion="rrf")` keeps the earlier fusion; `lexical_weight=` sets `w` (default 0.5).
  `hybrid` hits now carry the fused score in [0, 1] instead of a reciprocal-rank value. `fts` and
  `vector` rank and score exactly as before.

### Changed: an unknown provider prefix raises instead of falling back to Anthropic (DX-02)

* `Agent(model="OpenAI:gpt-4o")`, `"openai/gpt-4o"` or `"foo:bar"` silently went to
  Anthropic (and only failed with a missing `anthropic` extra or at the API). A registered
  name in another case or with `/` now raises `ConfigurationError` with the corrected name,
  also in `resolve_model()`. Agents, `Evaluator` and the dynamic agent factory resolve
  `strict`: without `provider=`, any other `prefix:` or `/` raises (pass `provider="ollama"`
  for `llama3.1:8b`). `resolve_model(model)` without `strict=True` keeps the default provider
  for other unknown prefixes, so code that only parses stored names is unaffected.

### Changed: a missing API key raises `MissingCredentialsError` for every provider (DX-01)

* Without a key each SDK raised its own exception (anthropic a `TypeError` at request time,
  openai an `openai.OpenAIError`, google-genai a `ValueError`), which `except OrkexaError` did
  not catch. Now every provider raises `MissingCredentialsError` (a subclass of
  `AuthenticationError`) naming the provider and the environment variables to set
  (`error.variables`), without a link to the SDK exception; `ClaudeAgent` too.

### Changed: a turn no longer deep-copies the whole conversation (audit ORK2-09)

* Every `ask()` deep-copied every committed message, so the per-turn overhead grew with the
  history (300 turns: 18 ms for the last turn, 77 % of it copying). A turn now extends a new
  list holding the committed messages (last turn of 300: 1.5 ms). The runtime and the
  provider adapters never change a message in place (contract-tested for every adapter), so
  a failed turn still leaves the committed conversation untouched; `conversation()` and
  `history()` still return copies. Hooks that receive `ModelRequest.messages` must treat
  them as read-only.

### Changed: stopping a Docker sandbox call no longer always waits 3 seconds (audit ORK2-08)

* `stop_container` watched the container name for 3 s after every timeout or cancellation.
  When `docker kill` succeeded the container existed, so no create request can still be in
  flight: the watch now ends as soon as the name is confirmed gone. When the kill failed
  (the client may have died while creating it) the full 3 s watch remains.

### Added: a limit on tool calls per model response (audit ORK2-04)

* `max_tool_rounds` did not bound tool calls: one response with 500 calls ran all of them.
  New `max_tool_calls_per_round` (default 64, `None`: no limit): a response with more calls
  raises `MaxToolCallsError` (a subclass of `MaxToolRoundsError`) before any of its calls
  runs. A turn runs at most `max_tool_rounds × max_tool_calls_per_round` calls.

### Changed: turn checkpoints are bound to their content; single-use resume is opt-in (audit ORK2-03)

* `TurnCheckpoint` has two new optional fields, `nonce` and `digest`. The digest covers the
  conversation, every call with its arguments, the stored results and the waiting calls;
  `resume()` refuses a checkpoint changed after it was issued, and one whose `calls` differ
  from the calls in its conversation. Checkpoints written before (no digest) still resume
  on agents without a key. The checkpoint version stays 1.
* New `checkpoint_key=` (agents): signs checkpoints with an HMAC and refuses unsigned or
  altered ones, so approved call arguments cannot be changed in storage.
* New `resume(..., claim=)` and `CheckpointClaims`: a claim keyed by the new
  `TurnCheckpoint.id` refuses a second resume of the same checkpoint before anything runs.
  Without a claim, resuming again still works (and runs the waiting calls again), as
  before; the guide now warns about it.

### Fixed: a failing parallel tool call in AsyncAgent cancels its siblings (audit ORK2-02)

* With `max_parallel_tools > 1`, the first failing call ended `AsyncAgent.ask()` while its
  siblings kept running: they committed side effects and called `after_tool` after the turn
  had failed, even during the next turn. The siblings are now cancelled and awaited before
  the error is raised (the orchestration driver's pattern, now shared as
  `orkexa._aio.gather_or_cancel`), as the sync `Agent` already waited for all calls.

### Fixed: a timed-out or cancelled tool call stops its sandbox call (audit ORK2-01)

* `Agent(tool_timeout=...)` abandoned the worker thread and cancelling `AsyncAgent.ask()`
  left `asyncio.to_thread` running, so a `SandboxedTool`'s process or container kept running
  until the sandbox's own timeout (and `max_parallel_tools` no longer bounded live
  sandboxes). The drivers now bind a per-call cancel check, linked to the caller's
  `cancel_scope`, and set it on timeout or cancellation: the process is killed and the
  container removed. Plain Python tool code in a thread still cannot be stopped.

### Fixed: the Docker self-test reports the container's own environment

* The self-test read the environment from `os.environ`, which also holds what Python sets
  itself. In an image without `LANG` (such as uv's images), C locale coercion (PEP 538) adds
  `LC_CTYPE`, so the self-test refused every daemon with "unexpected environment variables:
  LC_CTYPE". It now reads `/proc/self/environ`, the environment the container started the
  process with. Every variable the container is given is still reported and checked;
  `ALLOWED_ENVIRONMENT` is unchanged.

### Changed: new official logo

* The README, the documentation site (header, landing page, favicon) and the social
  preview use the new official Orkexa logo, with a white-wordmark variant on dark
  surfaces (README in dark mode, docs in the dark scheme, the navy docs header). The
  published PyPI description keeps the light logo only.
* `assets/branding/` holds the two new masters (`orkexa-logo-transparent.png`,
  `orkexa-logo-white.png`) and the shared derivation script `derive_branding.py`;
  `build.py` regenerates every derived file from them. The old logo
  (`orkexa-logo-original.png`) and its derivatives, including `orkexa-mark.png`, are
  removed.
* The GitHub social preview (`assets/branding/social-preview.png`, 1280 × 640) now uses the
  template shared with Nodestro and Axentra (`social_preview.py`): the white-wordmark lockup
  on a dark background with a blue glow and the caption "One agent runtime for every major
  model provider".

### Changed: the Docker self-test checks mounts, seccomp and AppArmor (security review DS-8, DS-9, DS-10)

* `DockerSandbox.self_test()` reads the container's whole mount table (`/proc/self/mountinfo`) and refuses any
  mount Docker or Podman does not add themselves, read-only or not, e.g. a volume from `daemon.json` or Podman's
  `containers.conf`/`mounts.conf` (`unexpected ... mount at <path>`). Allowed: the read-only root, `/proc` with its
  read-only and masked paths, `/dev`, `/dev/pts`, `/dev/mqueue`, rootless device nodes, a read-only `/sys` and
  `/sys/fs/cgroup`, the sandbox's tmpfs file systems, `/etc/hosts`, `/etc/hostname`, `/etc/resolv.conf`,
  `/run/.containerenv` and a read-only `--init` binary.
* The self-test calls `add_key`, `bpf`, `perf_event_open`, `userfaultfd` and `move_pages`, which the default
  seccomp profiles of Docker and Podman block, and refuses the host if any succeeds (a permissive custom profile
  passed the old `Seccomp: 2` check). The probes exist for x86-64 and arm64; other architectures are refused.
* Where the host reports AppArmor, the container must run under an enforcing profile (not `unconfined` or
  `complain`); otherwise the label is reported as the new fact `apparmor_label`. A stack of the engine's container
  profile (`docker-default`, `containers-default-<version>`) with the OCI runtime's unconfined helper profile
  (`crun`, `runc`; Ubuntu 24.04 Podman reports `containers-default-0.57.4-apparmor1//&crun (mixed)`) counts as
  enforcing, because every member of a stack confines the process; any other member or order is refused.
  `evaluate_self_test()` has the new keyword `apparmor`; the report format is version 2.
* `run_args()` adds root-owned, empty `noexec` tmpfs file systems on `/run` and `/var/tmp` (1 MiB): Podman mounted
  writable, executable tmpfs file systems there for every read-only container.
* `DockerProbeError` details and `docker ps` errors are redacted (`redact_docker_output()`, exported from
  `orkexa.dynamic.docker`): no URL user info or passwords, no non-standard socket paths, no user names from home
  directories, no tokens in query strings or `Authorization` values. Redaction happens before the 500-character cut.
* `docker/sandbox.Dockerfile` pins `python:3.12-slim` by the digest of its multi-arch index and installs the new
  `docker/sandbox-requirements.txt` (exact versions, sha256 per wheel, x86-64 and arm64) with `--require-hashes`;
  `tests/test_dockerfiles.py` refuses a shipped Dockerfile with an unpinned `FROM` or an unhashed `pip install`.

### Added: pausing a turn before a model request; adjustable budgets

* `RequestSuspended(reason, ref)` (exported from `orkexa`): a usage tracker may raise it
  from `check_request()`, `before_request()` or `try_reserve()`. The request is not sent,
  recorded or reserved; the turn ends with `TurnSuspended`, whose `TurnCheckpoint` has
  the new field `stage="request"` (plus `reason` and `ref`). `Agent.resume()` /
  `AsyncAgent.resume()` admit the same request again and continue the turn without
  re-sending earlier requests. Tool-stage checkpoints have `stage="tools"` (the default;
  `CHECKPOINT_VERSION` stays 1, so existing checkpoints still resume). Native batch
  requests cannot be suspended (ORK-022).
* `TurnCheckpoint.calls` and `waiting` may be empty for `stage="request"`; a
  `stage="tools"` checkpoint still needs at least one call and one waiting call.
* `BudgetExceededError` (and its subclasses) has optional keyword-only details:
  `limit_usd`, `spent_usd`, `reserved_usd`, `requested_usd`, `tracker` (the tracker whose
  limit blocked, possibly a parent) and `reason` (`"used_up"`, `"request_too_large"`,
  `"exceeded"`). `BudgetTracker` fills them at every refusal; raising it with a message
  only keeps working.
* `BudgetTracker.set_limit(limit_usd)` changes the limit thread-safely, returns the
  previous one and wakes requests that wait for budget.
* `tool_results_of_last_turn(agent)` is public (exported from `orkexa` and
  `orkexa.evaluation`): the text `refine()` gives a judge, for applications that run their
  own (for example resumable) ask-evaluate-revise loop.

### Python 3.11 minimum

* Python 3.11 is now the minimum (3.10 reached end of life in October 2026): `requires-python`, classifiers, ruff
  `py311`, mypy 3.11; CI tests 3.11 to 3.13, the lowest-versions job runs on 3.11. The docs extra needs
  `mkdocstrings>=1.0.7` and allows `pymdown-extensions` up to 12.
* The modules the Docker sandbox runner imports (`orkexa.dynamic.docker.RUNNER_MODULES`) stay importable on
  Python 3.10, because they run with the Python of the user's sandbox image (ruff `UP` is off for them).

### Removed: native macOS sandbox (Docker-first, ADR 0012)

* The experimental Seatbelt `MacOSSandbox` (ORK-014) is removed; `DockerSandbox` is the sandbox for untrusted
  code on all platforms. An independent review found it not releasable (MS-1: same-user processes can read a
  sandboxed process's environment through `KERN_PROCARGS2`; MS-2 to MS-10).
* `orkexa.dynamic.MacOSSandbox` remains as a deprecated placeholder for one release: `probe()` always reports
  `available=False`, `reason="removed"`; its capabilities claim no isolation; constructing it emits a
  `DeprecationWarning`; `validate()` / `call()` raise `SandboxError` / `ToolExecutionError` without running
  anything. It will be deleted later.
* The `Platforms` workflow no longer requires a native sandbox on macOS (`ORKEXA_REQUIRE_NATIVE_SANDBOX` is gone).

### Added: Amazon Bedrock provider

* Provider `"bedrock"` (`pip install "orkexa[bedrock]"`, also part of `orkexa[all]`):
  the Bedrock Converse and ConverseStream API through `boto3`, with `BedrockOptions`
  (region, profile or session, inference parameters, guardrail, service tier, latency,
  prompt caching, model-specific fields, request metadata). Model ids may be
  foundation model ids, cross-Region inference profile ids or Bedrock ARNs.
* Credentials come only from boto3's credential chain (roles, SSO, profiles,
  `AWS_BEARER_TOKEN_BEDROCK`) or an injected session or client; `api_key=` is
  rejected, `AWS_ENDPOINT_URL*` settings are ignored and a custom `base_url` must use
  `https`. `AsyncAgent` runs Bedrock calls in worker threads.
* `BedrockOptions(data_residency="eu_only")` and `allowed_regions=` verify the region,
  endpoint, every destination region of an inference profile (`GetInferenceProfile`)
  and the guardrail before the first paid request, reject global profiles and fail
  closed. New `DataResidencyError` (a `ConfigurationError`).
* Capabilities per model family (Claude 5.x, Claude 4.5–4.8, older Claude, Amazon
  Nova; `unknown` for other models and opaque ARNs).
* Usage is priced for provider `bedrock` only, never with Anthropic's first-party
  prices; every record carries the tariff condition `aws_region` (the source region).
  Orkexa ships no Bedrock prices. `record_token_usage()` takes an additive
  `conditions=` argument for such provider-reported tariff conditions.
* Example `examples/bedrock_agent.py`; documentation of credentials, the differences
  between the Anthropic API and Claude on Bedrock, EU data residency with an example
  SCP, invocation logging and Bedrock prices.

### Changed: Docker sandbox proves its host before code runs (ORK-019 to ORK-021)

* New `orkexa.dynamic.docker`: `inspect_host()` reads `docker info`/`docker version`
  (and the native `podman info`) into a `DockerHost`; `preflight()` refuses a daemon
  with the reason codes `os-type` (no Linux containers), `limits` (no `MemoryLimit`,
  `PidsLimit`, `CpuCfsQuota` or `CpuCfsPeriod`, e.g. rootless without cgroup
  delegation), `seccomp` (none or `unconfined`) and `rootless-cgroup` (rootless on
  cgroup v1). The engine's name is reported, never trusted. `host_fingerprint()`.
* `DockerSandbox.self_test()` runs one container of exactly the sandbox's shape and
  checks user 65534, `CapEff`/`CapPrm`/`CapBnd` 0, `NoNewPrivs` 1, `Seccomp` 2, only
  `lo`, a read-only root, `noexec`/`nosuid`/`nodev` `/tmp` and `/work`, the cgroup v2
  `memory.max`, `memory.swap.max`, `pids.max` and `cpu.max`, `RLIMIT_NOFILE`,
  `RLIMIT_FSIZE`, `RLIMIT_CORE`, the tmpfs sizes and the environment (only the
  engine's and the image's own variables; secret-like image variables are refused).
  cgroup v1 is refused (reason `self-test`). A pass is remembered per daemon
  fingerprint, image id and shape. `DockerSandbox.probe()` and `available()` (same
  signatures; `available()` gains an optional `image`) are true only after preflight
  and self-test, and every run refuses otherwise. `clear_self_test_cache()`.
* One container shape for every call: `ContainerLimits` and `run_args()` add
  `--pull never`, `--ipc private`, `--shm-size`, `--ulimit nofile/fsize/core`,
  `--log-driver none`, `nodev` and a tmpfs `/work` (uid 65534, mode 0700), and the
  labels `org.orkexa.sandbox=<owner>` and `org.orkexa.deadline=<epoch>`; no `-v`, no
  `-e`. Calls run the checked image by its id (`resolve_image_id()`); `pin_base()`
  pulls base images only by digest and checks `RepoDigests`.
* `DockerSandbox(transport="stdin")` is the new default: code, payload and the runner
  modules (`runner_files()`, `runner_fingerprint()`, `copy_runner()`) travel in a JSON
  envelope on stdin, so nothing of the host's file system is mounted and the daemon
  may run in a VM or on another host. `transport="bind"` keeps the old mounts and
  emits a `DeprecationWarning`. New `owner=` (the label value).
* `DockerSandbox(environ=...)`: the environment of every `docker` CLI process the
  sandbox starts (preflight, image inspection, self-test, runs, stop, reap), e.g. only
  `PATH`, `HOME` and `DOCKER_HOST`; default `None` inherits this process's. It never
  reaches the container.
* `Sandbox.cancel()` and timeouts kill and remove the call's container by name;
  `list_sandbox_containers()`, `reap()` and `DockerSandbox.reap()` remove containers
  past their deadline label (after a crash of the host process).
* `python -m orkexa.dynamic.docker --self-test` (also `--preflight`, `--reap`,
  `--json`) checks a computer; see `docs/development/docker-manual-verification.md`.
* CI: the `docker` job of the `Platforms` workflow is a matrix of rootful and
  rootless Docker (cgroup delegation of cpu, memory and pids) on ubuntu-24.04, a
  rootless daemon without cpu/pids delegation that must be refused, Podman through
  `podman system service`, rootful Docker on ubuntu-24.04-arm and a non-blocking
  Colima job on macos-15-intel. The real-container tests (pytest marker `docker`:
  self-test, fork bomb, OOM kill, network, read-only root, file-size limit, cancel,
  reaper) fail instead of skipping with `ORKEXA_REQUIRE_DOCKER=1`.

### Fixed: sandbox output is bounded in the host

* `SubprocessSandbox` and `DockerSandbox` read the runner's output with
  `communicate_bounded()`: at most `response_limit(max_output_chars)` bytes of stdout and
  the last 64 KiB of stderr stay in the host process. Code that wrote straight to the
  inherited stdout (`os.write(1, ...)`) could previously make the host buffer any amount
  (the child's memory limit does not cover the host). A run that writes more is stopped
  (`SandboxError`, the container is removed). `communicate_bounded()`, `run_bounded()` and
  `response_limit()` are public in `orkexa.dynamic` for other sandbox backends.

### Added: turn checkpoints (Adaptive Orchestration M6b)

* A `before_tool` hook may raise `ToolCallSuspended(reason, ref)`: the call does not
  run, the turn ends with `TurnSuspended` and its JSON `TurnCheckpoint` (conversation,
  results of calls that ran, waiting calls). `Agent.resume(checkpoint)` /
  `AsyncAgent.resume()` continue the turn: only the waiting calls pass `before_tool`
  again (`ToolCall.resumed=True`), calls that ran are not repeated, and the model is
  not called again before all results of the round are in. Checkpoints are bound to
  the agent's name, model, prompt, tools and limits (`CheckpointError`).

### Added: dynamic teams and paused delegations (Adaptive Orchestration M6b, ADR 0011)

* Orchestration protocol version 2 (version 1 records still read and replay):
  `Parallel` (several members at once, `max_parallel`), `Staff` (a team assembled from
  the pool while the orchestration runs, `staffing`, `max_team`), member `Parameter`s
  with validated `values`, an `Outcome` per assignment (`succeeded`, `failed`,
  `rejected`, `cancelled`, `expired`, `waiting`) and `Pending` delegations: a member
  that waits pauses the orchestration, which later continues the same execution
  (`ref`) without asking the model again (`complete_pending()`).
* Drivers take `execute(Delegation)`; `DelegationSuspended` pauses, `DelegationEnded`
  reports rejected, cancelled or expired members. The async driver runs parallel
  assignments concurrently and cancels the others on a budget error.

### Added: orchestration protocol (Adaptive Orchestration M4, ADR 0010)

* `orkexa.orchestration`: a model chooses which member of a fixed `Pool` works next.
  Two decisions, `Delegate` and `Finish` (no "create": members are selected, never
  generated); `validate()` checks each one against pool, `OrchestrationLimits`
  (`max_decisions`, `max_per_member`, `max_answer_chars`) and history with stable
  `orchestration.*` codes. `OrchestrationState` records every step (decision, verdict,
  result or error) as JSON; `replay()` re-validates a record without model calls.
* Reference drivers `run_orchestration()` and `arun_orchestration()` with `decide` and
  `delegate` callables; refused decisions are explained to the model, member failures
  are recorded, a `BudgetExceededError` (and any exception in `stop_on`) ends the
  orchestration.

### Added: budget reservations (Adaptive Orchestration AO-12)

* `BudgetTracker` reserves an upper bound of every request's cost before it is sent
  and releases it once the usage is recorded (`try_reserve()`, `release()`,
  `reserved_usd`, `Reservation`). A request that does not fit next to the requests in
  flight waits for them (`WaitForBudget` effect; async agents never block the event
  loop). **Behavior change:** agents sharing a tracker no longer overshoot it by one
  request each when they run concurrently.
* `orkexa.core.cost_bound.request_cost_bound()`: input size and `max_tokens` at the
  highest rates of a price (tiers, tariffs, cache classes); `None` (never 0) for
  requests without a bound, counted in `unbounded_requests`.
* `on_oversized="exclusive"` (default: an oversized request runs alone) or `"reject"`
  (refused before it is sent); `oversized_requests`; `hard_maximum`.
* Child scopes: `BudgetTracker(limit, parent=..., prefix=...)` passes every record,
  check and reservation on to the parent, so neither limit can be bypassed.
* Native batches are still admitted as one request without a reservation.

### Added: flow kernel (`orkexa.flow`)

* `orkexa.flow`, a deterministic, I/O-free workflow kernel (ADR 0009, guide
  `docs/guides/flows.md`): `Flow`, `Node`, `Edge`, `Loop`, `Limits`, `validate()`
  with stable `flow.*` issue codes, and the pure functions `start()`, `complete()`,
  `advance()`, `cancel()`, `resume()` and `outcome()` over a JSON-serializable
  `FlowState` (contract `1.0`; events `orkexa.flow-event/1`).
* Invocations (`node/path/attempt`) instead of one result per node: branching ports,
  `join="all"`/`"any"`, suspension and resumption, stops from results, loop edges with
  a mandatory `max_iterations` (nested loops, `on_exhausted="fail"`/`"continue"`), and
  limits for invocations, parallelism, iterations and wall time.
* `run_flow()`/`run_flow_sync()`, an asyncio reference driver with one handler per node
  type, and `orkexa.flow.conformance` with scripted cases for other drivers.
* Example `13_flow_quality_loop.py` (offline).

### Added: pricing engine (PRICING-1)

* `orkexa.pricing` is now a package with a provider-independent pricing engine
  (guide: `docs/guides/pricing.md`): `ModelPrice` (Decimal rates per million tokens,
  explicit currency, explicit aliases, opt-in dated ids, long-context tiers, tariffs
  for service tier/batch/speed/data residency/regions, per-unit fees, special token
  rates, source, version, validity period, check date and quality), immutable,
  versioned and JSON-serializable `PriceSnapshot` (format `orkexa.price-snapshot/1`,
  content hash), isolated thread-safe `PriceResolver` with snapshot history, pinned
  `PricingBasis`, `MeteredUsage`, `calculate_cost()` with an itemized `CostBreakdown`
  and `PriceReference`s, `PricePolicy`, `round_cost()`, `PricingError` and
  `PriceNotFoundError`.
* Agents accept `pricing=` as `ModelPrice`, `PriceResolver` or `PriceSnapshot` (and
  `ModelPricing` as before); `UsageTracker(price_resolver=...)` prices every agent that
  shares it. Each request is priced with one pinned snapshot.
* `UsageRecord`: `requested_model`, `pricing_version`, `pricing_quality`,
  `pricing_conditions`, `known_cost_usd`, `missing_cost_components`, `cost`,
  `cost_complete`, `known_cost`, `to_dict()`/`from_dict()`. `UsageSummary`:
  `known_cost_usd`, `missing_cost_components`, `cost_complete`.
* `BudgetTracker(price_resolver=..., price_policy=..., on_price_issue=...)`,
  `UsageTracker.check_request()`, `UnreliablePriceError`, `UnreliablePriceWarning`.
* Provider adapters report what prices depend on: the OpenAI/Chat Completions service
  tier, fee-bearing OpenAI hosted-tool calls, audio tokens, Gemini audio/image tokens
  and Google Search grounding, Anthropic code execution.
* The built-in Anthropic catalog (checked 2026-10-31) adds Claude Opus 4.5,
  Sonnet 4.5, Mythos 5 and Mythos 5.1, fast mode (2x), US data residency (1.1x) and
  batch as tariffs. Example `12_pricing_snapshots.py` (offline).

### Changed: pricing corrections

* Claude Sonnet 5.5 cache reads cost $0.10 per million tokens (0.05x input, per the
  official price list), not $0.20.
* Anthropic fast mode and US-only data residency are priced instead of unknown.
* A reported non-standard OpenAI/compatible `service_tier` (`flex`, `priority`, ...)
  makes the cost unknown unless the price has that tariff (it was priced as standard).
* Unpriced server-tool fees, audio/image tokens and Anthropic code execution without
  web tools make a cost incomplete (`estimated_cost_usd=None`, known part in
  `known_cost_usd`) instead of being left out silently.
* The reported model id is resolved before the configured one; `pricing_source` is set
  whenever a price was used.
* `BudgetTracker.spent_usd` includes the known part of incomplete costs;
  `on_unpriced="reject"` also raises after a request whose cost turned out unknown
  (it is recorded first).
* `Provider.usage_record(pricing=...)` receives a pinned `PricingBasis`; custom
  providers pass it on to `record_token_usage(custom_pricing=...)` unchanged.
  Migration notes: `docs/guides/pricing.md`, section "Backward compatibility and migration".

### Added

* Shared agent runtime (`Agent`, `AsyncAgent`) as a sans-I/O state machine with
  synchronous and asynchronous drivers; provider adapters for Anthropic (Messages API),
  OpenAI (Responses API), Google Gemini (`google-genai`) and OpenAI-compatible endpoints
  (Chat Completions or Responses; presets `ollama` and `vllm`); `FakeProvider` for tests.
* Provider registry (`"provider:model"`, `register_provider`), typed provider options
  (`AnthropicOptions`, `OpenAIOptions`, `GeminiOptions`, `CompatibleOptions`) and a
  capability system with `supported` / `unsupported` / `unknown` and overrides.
* Neutral history (`Message` and parts with lossless provider metadata), versioned
  history files, `history_policy="strict" | "convert"` for provider switches.
* Neutral stream events (`on_event`), provider error hierarchy (`ProviderError`,
  `RateLimitError`, ...) with redacted messages.
* Strict-schema conversion for OpenAI and inlined schemas for Gemini.
* Usage across providers: `provider`, `turn_id`, `workflow`, `pricing_source` on
  `UsageRecord`, `usage_by_provider()`, `register_pricing()`; unknown cost is `None`.
* `orkexa.dynamic`: `ProviderSettings`, provider-qualified models in all components
  and `DynamicPolicy.allowed_models`; `BudgetTracker(on_unpriced=...)` with a pre-request
  check; `UnpricedModelError`, `UnpricedUsageWarning`, `UnsafeOptInWarning`.
* `retrieval_sources()`: verified search hits of the last turn, for every provider.
* `ClaudeAgent`/`AsyncClaudeAgent`: Anthropic-native agents with Messages API
  histories, native SDK objects in hooks and SDK exceptions, on the shared runtime.
* Docs (quickstart, provider guide, API reference, feature matrix),
  11 examples, CI for Python 3.10 to 3.13, packaging tests in fresh virtual environments.

### Defaults

* `AutoApprover()` warns (`UnsafeOptInWarning`) unless `acknowledge_unsafe=True`.
* `ToolRegistry()` uses `DockerSandbox()`;
  `SubprocessSandbox()` warns unless `acknowledge_unsafe=True` (see Security below).
* History and registry files are `0600`, new SQLite databases `0600`.
* `BudgetTracker` with a limit warns about requests without known price.
* `BudgetTracker` refuses a request when the budget is exactly used up
  (`spent_usd >= limit_usd`, e.g. a limit of 0); the limit is documented as a soft limit
  for concurrent agents.
* `ToolRegistry.add_generated()` validates the tool name (`[A-Za-z0-9_-]{1,128}`) and
  never writes outside the registry directory; stored records with a name that does
  not match their directory are skipped.

### Security (independent release audit, 2026-10-29)

* **Provider errors no longer chain the SDK exception** (SEC-01). `Agent` and
  `AsyncAgent` raise `ProviderError` with `__cause__` and `__context__` set to `None`
  in synchronous, asynchronous and streamed turns, token counting and batches, so
  tracebacks and `logging.exception()` cannot print the SDK exception, its request or
  its `Authorization` header. `provider`, `status_code`, `request_id`, `retryable` and
  the redacted message are kept. Redaction also covers bearer and basic credentials,
  `x-api-key`, Google, GitHub, Slack and AWS keys, JWTs, URL user info, credential
  query parameters and JSON credential fields.
* **Generated code runs in `DockerSandbox` by default** (SEC-02). `ToolRegistry()`
  without `sandbox=` fails closed when Docker or the image is missing; no code runs.
  `SubprocessSandbox()` emits `UnsafeOptInWarning` unless created with
  `acknowledge_unsafe=True`, and its run directory stays `0700`. The static check
  rejects access to modules through attributes of allowed modules (`calendar.sys`,
  `dataclasses.inspect`, `json.codecs`, `enum.bltns`, `typing_extensions._types` ...),
  and a test walks every reachable module so new exposures fail CI. Example 11 stops
  without Docker unless `ORKEXA_ALLOW_SUBPROCESS_SANDBOX=1`.
* **Stored data is owner-private** (PRI-01 to PRI-03). `save_history()` (all agents),
  `ToolRegistry` and `AgentRegistry` write `0600` files atomically; new SQLite
  databases of `SQLiteMemory`, `SQLiteFileStore` and `SQLiteIndex` (and so their
  journal and WAL files) are `0600`; directories Orkexa creates, including missing
  parents, are `0700`. The umask is not changed; existing databases and directories
  keep their mode.
* **`base_url` credentials are redacted** in the `repr` of `ProviderConfig` and
  `ProviderSettings` (SEC-03); requests use the URL unchanged.

#### Migration

* Build the sandbox image (`docker build -t orkexa-sandbox -f docker/sandbox.Dockerfile .`),
  or, for trusted or reviewed code only, pass
  `ToolRegistry(path, sandbox=SubprocessSandbox(acknowledge_unsafe=True))`.
* Code that read `ProviderError.__cause__` uses the error's attributes instead.
* Setups that share histories, registries or new databases with other users grant
  access explicitly. To make databases created by earlier versions private, run
  `chmod 600 <database>*`.

See "Migrating to the secure defaults" in the security guide.

### Fixed during review of the first implementation

* Error texts of failed OpenAI responses (`status="failed"`) are redacted; the runtime
  also redacts every `ProviderError` an adapter builds from native response text.
* Reasoning (and Gemini thought signatures) of another model of the same provider is
  no longer replayed after `agent.model` changes; `Message.model` now records the
  requested model id, so dated versions reported by the API do not count as a switch.
* `UsageTracker.usage_by_turn()`, `usage_by_workflow()` and `check()` exist as
  documented; `orkexa.usage.workflow()` labels requests, and `Orchestrator.ask()`
  labels each run including delegated agents (`last_workflow`).
* `ToolBuilder.as_tools()`' `load_tool` respects `DynamicPolicy.allowed_agent_tools`.
* `AgentRegistry` validates agent names like `ToolRegistry` validates tool names and
  skips stored records whose name does not match their directory.
* Native batches pass the usage tracker's pre-request check (budget, unpriced models)
  before submission, and every batch result is recorded before a budget error is raised.
* `prepare_history()` keeps message boundaries and roles: it never merges messages and
  only removes a message when none of its parts can be replayed.
* `AgentRegistry` and `ToolRegistry` persist a record before they change their
  in-memory state, so a failed write (full disk, rejected path) leaves both unchanged;
  record files are replaced atomically.
* Batch usage no longer counts in `last_turn_usage` (it covers the most recent `ask()`).
* A request that was answered is recorded even when `on_response` (or a hook of the
  final stream event) raises.
* `reset()` and `load_history()` raise `ConcurrentTurnError` while a turn runs instead
  of being undone when the turn commits.
* `revoke()` of both registries documents and tests its behavior on a failed write: the
  active version is revoked last, so it stays active until a retry completes.
* Dropping `effort` for models without reasoning in the dynamic package is logged.
* OpenAI Responses: only `completed` responses are answers; `queued`, `in_progress`
  and `cancelled` raise `UnexpectedResponseError` without changing the history
  (previously an empty answer was committed), and `request_options={"background":
  True}` is rejected. A stream that ends with `response.incomplete` or
  `response.failed` raises `IncompleteResponseError` or `ProviderServerError` and
  records its usage, instead of the SDK's bare `RuntimeError`.
* Anthropic: `temperature`, `top_p` and `top_k` in `request_options` raise
  `ConfigurationError` (API version 2026-10-21 of `anthropic>=2.0` removed them).
  `count_tokens()` with `output_format` and a beta (e.g. `thinking_display="updates"`)
  sends the schema as `output_config.format`; the beta method has no `output_format`.
* Text of a Responses message item is replayed even when an unknown content type
  precedes it.
* OpenAI capabilities cover the GPT-6 family (`gpt-6-astra`, `gpt-6.1-sol`,
  `gpt-6.1-terra`, `gpt-6-luna`, ...): reasoning effort and image/document input are
  `supported` instead of `unknown`.
* Gemini 4 models (`gemini-4-argon`) map `effort` to `thinking_level` like Gemini 3
  instead of raising `UnknownCapabilityError`. Docs and examples use current model ids
  (`gpt-6.1-sol`, `gemini-3.8-flash`).
* Anthropic batch items are built like normal requests: thinking, cache TTL, server
  tools, native `tool_choice`, `request_options` (with `extra_body` merged into the
  item) and betas (through the beta batch API) are no longer dropped. Local tools and
  `extra_headers`/`extra_query`/`timeout` are rejected before submission.
* CI also runs the test suite with the lowest supported dependency versions.
* `SQLiteMemory`, `SQLiteFileStore` and `SQLiteIndex` open their connections with
  `PRAGMA secure_delete=ON`: deleted content no longer stays readable in free pages of
  the database file. New `SQLiteIndex.optimize()` also erases the terms of deleted
  chunks from the FTS5 index segments.
* Tests marked `live` run only when selected with `pytest -m live`; an API key in the
  environment no longer makes a plain `pytest` call a paid API.
* Docs extra: `pymdown-extensions>=11.0.1,<12` (fixes PYSEC-2026-3609 and
  PYSEC-2026-3654) and `mkdocs<2`.
* Repository: brand assets, README and documentation overhaul (compliance pages,
  troubleshooting, development and release guides), community files, issue and pull
  request templates, Dependabot, a dependency audit and `twine check` in CI, and a
  release workflow prepared for PyPI trusted publishing.
