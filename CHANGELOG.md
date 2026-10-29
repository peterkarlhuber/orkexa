# Changelog

## 0.1.0 – unreleased

First release of Orkexa, the provider-independent successor of `claude-agent` 0.8.0.

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
* `python -m orkexa.compat` diagnostics; distribution `orkexa-claude-compat` providing
  unchanged `import claude_agent`.
* Docs (quickstart, provider guide, migration guide, API reference, feature matrix),
  12 examples, CI for Python 3.10 to 3.13, packaging tests in fresh virtual environments.

### Changed compared to claude-agent 0.8.0

* Sandbox temp directory prefix, Docker image and mount path use `orkexa`
  (`orkexa-sandbox-`, `orkexa-sandbox`, `/opt/orkexa/orkexa`).
* `AutoApprover()` warns (`UnsafeOptInWarning`) unless `acknowledge_unsafe=True`.
* `BudgetTracker` with a limit warns about requests without known price.
* Install hints name the `orkexa[...]` extras.
* `BudgetTracker` refuses a request when the budget is exactly used up
  (`spent_usd >= limit_usd`, e.g. a limit of 0); the limit is documented as a soft limit
  for concurrent agents.
* `ToolRegistry.add_generated()` validates the tool name (`[A-Za-z0-9_-]{1,128}`) and
  never writes outside the registry directory; stored records with a name that does
  not match their directory are skipped (0.8.0 accepted names such as `../x`).

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
  final stream event) raises; 0.8.0 lost its usage.
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
