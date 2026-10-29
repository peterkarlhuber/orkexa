# Source inventory of the reference `claude-agent` 0.8.0

This inventory was taken from the reference repository `peterkarlhuber/claude-agent`
(`pyproject.toml`: version **0.8.0**, commit `6d8551a`) before any Orkexa code was
written. Every feature below is a migration item; its Orkexa counterpart, test and
status are tracked in [feature-parity.md](feature-parity.md).

Baseline of the reference (Python 3.12, `pip install -e ".[dev]"`, anthropic 2.3.0,
pydantic 2.14):

| Command | Result |
| --- | --- |
| `pytest` | **326 passed, 2 skipped** (Docker end-to-end, live API) in 14 s |
| `ruff check .` / `ruff format --check .` | clean |
| `mypy` (strict, 32 files) | clean |

## Package layout (`src/claude_agent`, ~6,000 lines)

| Module | Lines | Responsibility |
| --- | ---: | --- |
| `__init__.py` | 121 | 51 public exports, `__version__` from distribution metadata |
| `_base.py` | 661 | `_AgentBase`: configuration, validation, history, usage, request building, response processing, tool-call helpers (no I/O) |
| `agent.py` | 176 | `ClaudeAgent`: sync API calls, streaming, tool execution (thread pool), `ask()`, `count_tokens()` |
| `async_agent.py` | 119 | `AsyncClaudeAgent`: same loop with `await`, `asyncio.gather` + semaphore |
| `tools.py` | 291 | `@Tool` decorator: docstring parsing, Pydantic arg model, strict schema, I/O validation |
| `content.py` | 244 | content-block helpers `text/image/document/uploaded_file`, `to_content`, `ToolContent`, `describe_content` |
| `errors.py` | 101 | exception hierarchy rooted at `ClaudeAgentError` |
| `usage.py` | 251 | `UsageRecord`, `UsageSummary`, `UsageTracker`, `record_api_usage`, `summarize_usage` |
| `pricing.py` | 80 | `ModelPricing`, `MODEL_PRICING`, `resolve_pricing` (Haiku 5.5 long-context tier) |
| `memory.py` | 675 | memory-tool command engine + `EphemeralMemory`, `FileMemory`, `SQLiteMemory` |
| `files.py` | 499 | `FileStore`, `LocalFileStore`, `SQLiteFileStore`, `StoredFile`, `files` tool, Files API upload |
| `retrieval.py` | 569 | `Index`, `SQLiteIndex` (FTS5/BM25 + sqlite-vec hybrid via RRF), `Chunk`, `Hit`, `Citation`, `citations()`, `split_text()`, PDF ingestion |
| `evaluation.py` | 437 | `Criterion`, `CriterionResult`, `Evaluation`, `EvaluationCase`, `Evaluator` (+ Message Batches), `Refinement`, `refine()`, `tool_results_of_last_turn()` |
| `dynamic/` | ~2,400 | dynamic tools/agents: see below |

## Agent core (`_base.py`, `agent.py`, `async_agent.py`)

| Feature | Location | Semantics to preserve |
| --- | --- | --- |
| Constructor (33 keyword-only params) | `_AgentBase.__init__` | defaults: `model="claude-haiku-5-5"`, `system_prompt="You are a helpful assistant."`, `effort="medium"`, `cache_control=True`, `cache_ttl="5m"`, `max_tokens=16000`, `max_tool_rounds=8`, `max_parallel_tools=1`, `raise_tool_errors=True`, `stream=None`, `on_refusal="raise"` |
| Validation at construction | `_AgentBase.__init__`, `_check_model_options` | `TypeError` for non-BaseModel `output_format`, non-`MemoryBackend`/`FileStore`/`Index`; `ValueError` for limits, `cache_ttl`, `tool_choice`, `thinking_display`, `on_refusal`, `fallbacks`, reserved `request_options` |
| Model-specific checks | `_check_model_options`, `_check_system_message` | forced `tool_choice` rejected on Opus 5.5 / Sonnet 5.5 / Fable 5.1 / Mythos 5.1; `thinking="disabled"` rejected on those and Fable 5 / Mythos 5; allowed only up to effort `high` on Opus 5 / Haiku 5.5; `between_tools` only on Sonnet 5.5 up to `high` (unknown models unchecked); system messages rejected on Sonnet 5 / Opus 4.6–4.7 / Sonnet 4.6 / Haiku 4.5; model ids normalized by `model_family()` (`anthropic.` prefix, `[...]` suffix, date suffix) |
| Request building | `_request_args` | `model, max_tokens, system, messages`; `output_config.effort`; `thinking {type, display}`; top-level `cache_control` (or `extra_body.cache_control` with `output_format`, SDK `parse()` limitation); `tools` + `tool_choice` only if tools exist; SDK `output_format`; `fallbacks`; deep-merge of `request_options` (dicts merged); `betas` |
| Beta routing | `_betas`, `_api` | any beta → `client.beta.messages`; automatic betas for `thinking_display="updates"` and both `fallbacks` variants |
| Streaming selection | `_use_stream` | `stream` param, else streamed when `on_text`/`on_thinking` set or `max_tokens > 16000` |
| Stream hooks | `_stream_hook`, `_call_api` | SDK events `text` → `on_text(delta)`, `thinking` → `on_thinking(delta)` (non-empty only); `_request_id` copied from stream headers |
| Eager input streaming | `_eager`, `_tool_definitions` | `eager_input_streaming=True` only on streamed requests and only on local tools |
| Tool definitions order | `_tool_definitions` | local tools, then server tools, then memory tool (`memory_20250818`) |
| Turn transaction | `_start_turn`, `ask` | history deep-copied at start; committed only on a final response; `system=` appends a `role: "system"` message after the prompt; per-turn `effort`/`max_tokens` overrides reset in `finally` |
| Response processing | `_process_response` | usage recorded **before** any check; `max_tokens`/`model_context_window_exceeded` → `IncompleteResponseError`; assistant blocks stored with `model_dump(exclude_none=True)`; `tool_use`/`pause_turn` count as rounds (`MaxToolRoundsError`); `pause_turn` resumes without user message; unknown stop reasons → `UnexpectedResponseError`; refusal → `RefusalError(stop_details)` unless `on_refusal="return"`; text joined with `"\n"`; structured output from `parsed_output` |
| Tool execution | `ClaudeAgent._run_tool(s)` | `before_tool` (`None`/`True` allow; `False` → "Tool call denied."; string → "Tool call denied: <reason>"); `ToolResultError` always back to model; `ToolInputError` raised unless `raise_tool_errors=False` or eager input; `ToolExecutionError` raised unless `raise_tool_errors=False`; `after_tool(call, describe_content(content), is_error)`; results of one turn in one user message, in call order |
| Parallel tools | `ClaudeAgent._run_tools` | thread pool of `min(max_parallel_tools, n)` with `copy_context()` per call; first failure re-raised |
| Async variant | `AsyncClaudeAgent` | awaitable hooks; async tools awaited, sync tools via `asyncio.to_thread`; memory commands in a worker thread; semaphore of `max_parallel_tools` |
| Sync agent rejects async tools | `ClaudeAgent._check_tool` | `ToolDefinitionError("...async tools require AsyncClaudeAgent")` |
| History API | `history`, `reset`, `save_history`, `load_history`, `messages=` | JSON list of Messages-API dicts; load validates roles `user/assistant/system` |
| Token counting | `_count_tokens_args`, `count_tokens` | subset of request fields; `betas` add `context_management`, `mcp_servers`; prompt appended without starting a turn; error on empty history |
| Usage properties | `usage`, `last_usage`, `last_turn_usage`, `usage_records`, `reset_usage` | per-agent records; shared `usage_tracker` keeps its ledger |
| Tool management | `add_tool`, `add_server_tool`, `remove_tool`, `_tool_names` | duplicate names rejected across local/server/memory; server tool name from `name` or `mcp_server_name` |
| `repr` | `__repr__` | `ClaudeAgent(name=..., model=..., tools=[...], messages=N)` |

## Tools and content

| Feature | Location | Notes |
| --- | --- | --- |
| `@Tool`, `@Tool()`, `@Tool(name=..., strict=...)` | `tools.Tool` | Google docstring (`Args:`/`Returns:` mandatory), type hints mandatory (`Any`/`object` rejected), keyword-compatible params only, async functions allowed, async generators rejected |
| Strict input schema | `_check_schema_supported` | every object `additionalProperties: false`; arbitrary maps rejected |
| Validation | `Tool._validate_input`, `_serialize_output` | Pydantic strict **JSON** mode on input; strict output validation; `ToolContent` returns blocks |
| Execution | `execute`, `aexecute` | wraps handler errors in `ToolExecutionError`, passes `ToolResultError` through; `aexecute` runs sync tools in a thread |
| Schema | `Tool.schema()` | `{name, description, input_schema, strict}`; description includes `Returns:` text |
| Content helpers | `content.py` | base64/URL/text sources, PDF/text documents with `title`, `context`, `citations`; Files API references |
| `ToolContent` | `content.ToolContent`, `to_tool_content` | allowed block types `text/image/document/search_result`; search results not mixable |

## Memory, files, retrieval

| Feature | Location | Notes |
| --- | --- | --- |
| Memory command engine | `MemoryBackend.execute/view/create/str_replace/insert/delete/rename` | exact Anthropic memory-tool messages, 2-level listing, truncation, size limits, RLock, path validation (also URL-encoded) |
| Backends | `EphemeralMemory`, `FileMemory` (0o700/0o600, atomic, symlink-safe), `SQLiteMemory` (WAL, transaction per command, namespaces) | |
| Native tool | `MemoryBackend.definition` | `{"type": "memory_20250818", "name": "memory"}` (Anthropic client-side tool) |
| File stores | `FileStore.add/info/get_bytes/list/delete/block/catalog/tool`, `LocalFileStore`, `SQLiteFileStore` | content-addressed ids (sha256[:16]), size limits, `note_to=` memory note |
| Files API | `FileStore(files_api=anthropic_client)` | uploads once, stores `api_file_id`, then sends `file` source blocks (**Anthropic-specific**) |
| Index | `Index.add/add_file/blocks/tool`, `SQLiteIndex` | FTS5 BM25 with prefix OR-query, optional sqlite-vec vectors with cosine, reciprocal-rank fusion (k=60), PDF pages via pypdf |
| Citations | `Hit.block`, `citations(response)` | hits rendered as Anthropic `search_result` blocks; `citations()` reads `search_result_location` citations from a native response (**Anthropic-specific**) |

## Evaluation

| Feature | Location | Notes |
| --- | --- | --- |
| Judge | `Evaluator(criteria, model="claude-opus-5-5", effort="medium", max_tokens=4096, ...)` | tool-less `ClaudeAgent[_Verdict]`, fresh history per evaluation, random-tag prompt injection guard; pass computed in Python |
| Batch | `Evaluator.evaluate_many(batch=True)` | **Anthropic Message Batches**: `client.messages.batches.create/retrieve/results`, `transform_schema(_Verdict)` as `output_config.format`, polling with timeout, usage priced at 50 % (`batch=True`) |
| Refinement | `refine(agent, prompt, evaluator, ...)` | evaluator-optimizer loop in the same conversation; `tool_results_of_last_turn()` reads Anthropic `tool_use`/`tool_result` blocks |

## Dynamic agents (`dynamic/`)

| Component | Location | Provider coupling in 0.8.0 |
| --- | --- | --- |
| `CodingAgent`, `ToolProposal` | `coding_agent.py` | builds `ClaudeAgent`; `client: Anthropic` |
| `AgentArchitect` | `architect.py` | builds `ClaudeAgent` |
| `AgentSpec`, `ToolRequirement` | `agent_spec.py` | `model` is an unqualified Claude id |
| `AgentFactory` | `factory.py` | builds `ClaudeAgent`; passes one `client`/`api_key` to every agent |
| `AgentRegistry`, `AgentRecord` | `agent_registry.py` | JSON per version; provider-neutral |
| `ToolRegistry`, `ToolRecord` | `tool_registry.py` | sha256-checked persistence; provider-neutral |
| `ToolBuilder` | `tool_builder.py` | static check → sandbox tests → optional LLM review → approval → registry; `attach`/`revoke`/`as_tools` typed on `ClaudeAgent` |
| `SandboxedTool` | `sandboxed_tool.py` | host-side structural payload check, every call in a sandbox |
| `Sandbox`, `SubprocessSandbox`, `DockerSandbox` | `sandbox.py` | temp dir prefix `claude-agent-sandbox-`, package mount `/opt/claude-agent/claude_agent`, image `claude-agent-sandbox` (**hardcoded legacy paths**) |
| Sandbox runner | `_sandbox_runner.py` | registers `claude_agent` as bare namespace to import `claude_agent.tools.Tool` without the SDK (**hardcoded import**) |
| Static check | `static_check.py` | AST allowlist of imports, forbidden names/attributes, dunder strings, top-level statement rules |
| Approval | `approval.py` | `Approver`, `AutoApprover`, `DenyApprover`, `CallbackApprover`, `ConsoleApprover`, `LLMApprover`, `share_usage_tracker` |
| Review | `review.py` | `TOOL_REVIEW_CRITERIA`, `AGENT_REVIEW_CRITERIA`, `review_tool`, `review_agent` |
| Policy | `policy.py` | `DynamicPolicy` (allowed models default Claude ids), `BudgetTracker` (unpriced = 0), `RESERVED_TOOL_NAMES`, `clamp_effort` |
| Orchestrator | `orchestrator.py` | `ClaudeAgent` with `list_agents`/`create_agent`/`run_agent`; `_depth` ContextVar recursion limit |
| Output formats | `output_registry.py` | named Pydantic models |
| Errors | `dynamic/errors.py` | 8 classes under `DynamicAgentError` |

## Tests, examples, docs, CI, packaging

* `tests/` (15 modules, ~4,900 lines): fake Anthropic client (`conftest.py`) recording
  `create/parse/stream/count_tokens` kwargs and replaying `anthropic.types.Message`
  objects; one real-SDK streaming test via `httpx2.MockTransport`; live test opt-in
  (`-m live`, `ANTHROPIC_API_KEY`).
* `examples/` (7 scripts): weather, investment, async research, memory, RAG, evaluator, orchestrator.
* `docs/` (MkDocs Material + mkdocstrings): getting started, 18 guides, reference, project pages.
* CI: GitHub Actions matrix Python 3.10–3.13 (ruff, format, mypy, pytest-cov), docs build, release workflow.
* Packaging: hatchling, `anthropic>=1.0,<3` as a hard dependency, extras `pdf`, `vector`, `docs`, `dev`; `py.typed`; `docker/sandbox.Dockerfile`.

## Security mechanisms present in 0.8.0

* strict tool input/output validation (Pydantic strict JSON mode, `additionalProperties: false`);
* memory path validation (traversal, URL encoding, control chars) and `FileMemory` symlink confinement;
* generated code: AST static check, tests in sandbox, approval required (no default approver), sha256-verified registry persistence;
* `SubprocessSandbox`: empty environment, temp dir, `-I -B`, rlimits (CPU, AS, FSIZE, NPROC=0, CORE);
* `DockerSandbox`: no network, read-only FS, `--cap-drop ALL`, `no-new-privileges`, uid 65534, pids/memory/cpu limits, no env passthrough;
* LLM review can only make decisions stricter; random-tag prompt-injection guard in the judge prompt;
* recursion limit via ContextVar, `max_agents`, `max_generated_tools`, shared `BudgetTracker`.
