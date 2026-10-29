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
* Dropping `effort` for models without reasoning in the dynamic package is logged.
