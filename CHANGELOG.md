# Changelog

## 0.1.0 – unreleased

First release of Orkexa.

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
