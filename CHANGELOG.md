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
