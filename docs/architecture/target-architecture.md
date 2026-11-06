# Target architecture

> *An agent is not a Claude, GPT or Gemini agent, but an Orkexa agent with an
> exchangeable model provider.*

## Overview

```text
Application
    |
Agent / AsyncAgent            ClaudeAgent / AsyncClaudeAgent (Anthropic facades)
    |                                   |
    +----------- AgentCore (orkexa.core.runtime) ----------+
    |   turn state machine (sans-I/O generator)            |
    |   conversation copy / commit / rollback              |
    |   tool registry, approval hooks, error policy        |
    |   streaming events + hooks, usage + budget, limits   |
    |   memory / files / index tools, structured output    |
    +------------------------------------------------------+
    |
Provider protocol (orkexa.providers.base) + capabilities (orkexa.core.capabilities)
    |
    +-- AnthropicProvider     -> Anthropic Messages API        (anthropic SDK)
    +-- OpenAIProvider        -> OpenAI Responses API          (openai SDK)
    +-- GeminiProvider        -> Gemini generate_content       (google-genai SDK)
    +-- BedrockProvider       -> Amazon Bedrock Converse        (boto3)
    +-- ChatCompletionsProvider / CompatibleProvider (+ Ollama, vLLM presets)
    |                         -> Chat Completions or Responses on any base_url
    +-- FakeProvider          -> scripted, no SDK (tests)
```

## Responsibilities

### `orkexa.core.types` – the only internal data model

* `Message(role, content: str | list[Part], provider, model)`.
* `Part` (discriminated union): `TextPart`, `ImagePart`, `DocumentPart`,
  `SearchResultPart`, `ToolCallPart`, `ToolResultPart`, `ReasoningPart`, `NativePart`.
  Every part has `meta: {provider: {...}}` for lossless native extras (thinking
  signatures, OpenAI item ids and encrypted reasoning, Gemini thought signatures,
  Anthropic `cache_control`/citations).
* `ToolDefinition`, `ToolChoice`, `ModelRequest`, `ModelResponse`, `StopReason`,
  `RefusalDetails`, `StreamEvent` (`TextDelta`, `ReasoningDelta`, `ToolCallStarted`,
  `ToolCallDelta`, `ResponseCompleted`, `UsageReported`, `TurnCompleted`, `TurnFailed`),
  `Citation`.
* Input content blocks use the Messages API dict format (`image()`, `document()`, …)
  and are parsed into parts at the boundary.

### `orkexa.core.runtime.AgentCore` – the one runtime

* Holds configuration, the committed conversation, tools, usage records.
* `_turn()` is a generator that yields two effects, `CallModel(request)` and
  `RunTools(calls)`, and receives their results. It contains *all* business rules of
  a turn: history copy, per-turn overrides, round limits, stop-reason handling,
  refusals, incomplete responses, structured output, commit, usage recording, budget
  pre-checks. No provider branches.
* `Agent` (sync) and `AsyncAgent` (async) are I/O drivers of about 100 lines each:
  they perform the effects (provider call or stream, tool execution in threads or
  tasks, awaiting async hooks). See [ADR 0001](decisions/0001-single-runtime.md).
* Tool-call helpers shared by both drivers: hook argument, denial interpretation,
  error classification (`ToolResultError` always returned; `ToolInputError` and
  `ToolExecutionError` per `raise_tool_errors`), result construction.

### Providers – adapters only

A provider implements (see `orkexa.providers.base.Provider`):

| Method | Purpose |
| --- | --- |
| `capabilities(model)` | `ModelCapabilities` with `supported` / `unsupported` / `unknown` per feature |
| `check(request)` | model- and option-specific validation before anything is sent |
| `prepare(request) -> native` | encode history, tools, schemas, options into SDK kwargs |
| `call(native)` / `acall(native)` | one non-streamed request → `ModelResponse` |
| `stream(native)` / `astream(native)` | normalized `StreamEvent`s ending with `ResponseCompleted` |
| `count_tokens(request)` / `acount_tokens` | native token counting where available |
| `record_usage(response, agent=..., pricing=...)` | normalized `UsageRecord` |
| `map_error(exc)` | SDK exception → normalized `ProviderError` (secrets redacted) |
| `parse_block(dict)` | provider-specific input block parsing (Anthropic keeps extras) |
| optional `run_batch(...)` | native batch processing (Anthropic Message Batches) |
| `limits(model)` | documented token limits and reasoning settings (`ModelLimits`; unknown is `None`) |

No adapter contains a tool loop. Automatic SDK tool execution is never used
(Gemini: `automatic_function_calling.disable=True`; no Python callables are passed to
any SDK).

### Capabilities

`ModelCapabilities` lists: `tools`, `forced_tool_choice`, `parallel_tool_calls`,
`strict_tool_schemas`, `structured_output`, `streaming`, `image_input`,
`document_input`, `reasoning`, `caching`, `server_tools`, `native_memory_tool`,
`token_counting`, `batch`, `system_messages` (mid-conversation). The runtime calls
`require()` before a feature is used: `unsupported` → `UnsupportedCapabilityError`;
`unknown` → `UnknownCapabilityError` with the override to use. Users override per
agent with `capabilities={"tools": "supported"}`. Model names are never used to
*enable* a feature on a compatible endpoint. See [ADR 0003](decisions/0003-capabilities.md).

### Provider selection

`Agent(model="openai:gpt-5.5")` or `Agent(provider="openai", model="gpt-5.5")`. A
prefix is only recognized if it names a registered provider, so Ollama tags
(`llama3.1:8b`) and fine-tuned ids (`ft:gpt-…`) are not split. Prefix and `provider=`
must agree (`ConfigurationError`). Unprefixed models without `provider=` resolve to
the documented default provider `anthropic`. `register_provider(name, factory)` adds
providers without touching the runtime.

### Conversation, history and cross-provider rules

* The committed history is a list of `Message`s; a turn works on a copy and commits
  only on a final response. Failed turns leave the history unchanged, but usage of
  every API response is recorded *before* any check (costs never disappear from the
  ledger); executed tools are not undone (documented).
* Assistant messages carry their origin `provider`/`model`. Before encoding, the
  adapter calls `prepare_history(messages, provider, policy)`:
  meta of other providers is ignored; foreign `ReasoningPart`/`NativePart`s and
  provider file references raise `HistoryConversionError` (`history_policy="strict"`,
  default) or are dropped with a warning (`history_policy="convert"`). Tool calls,
  tool results, text and media convert losslessly.
* `save_history()` writes `{"format": "orkexa.history", "version": 1, "messages": [...]}`;
  `load_history()` also reads plain Messages-API lists through the Anthropic codec.
  `ClaudeAgent` writes the plain list. See
  [ADR 0002](decisions/0002-neutral-history.md).

### Tools and schemas

`@Tool` is unchanged. `Tool.definition()` yields the neutral definition; adapters
convert it (`orkexa.core.schema`):

* Anthropic: the JSON schema as generated.
* OpenAI strict function calling: optional properties become required + nullable;
  a `null` argument for such a property is removed before local validation, so the
  Python default applies; ambiguous cases (nullable field with non-`None` default)
  raise `ToolSchemaError`; `default` keywords are stripped (applied locally).
* Gemini: `$ref`/`$defs` inlined (recursive schemas raise `ToolSchemaError`); Gemini
  does not enforce schemas, so Orkexa's local validation is the guard and invalid
  arguments are returned to the model as tool errors.
* Chat Completions: like OpenAI.

All arguments are always validated locally by Pydantic in strict JSON mode before a
tool runs.

### Structured output

Native everywhere it exists (Anthropic `messages.parse`, OpenAI `responses.parse`,
Gemini `response_json_schema`, Chat Completions `response_format: json_schema`) plus
local validation of the returned JSON where the SDK does not parse. The only
fallback is the explicit `CompatibleOptions(structured_output="json_object")`
(schema in the instructions, JSON mode, local validation). Refusals raise
`RefusalError`, incomplete outputs `IncompleteResponseError`, invalid JSON
`StructuredOutputError`.

### Streaming and hooks

Adapters translate SDK events into `TextDelta`, `ReasoningDelta`, `ToolCallStarted`,
`ToolCallDelta` and a final `ResponseCompleted` that carries the same `ModelResponse`
a non-streamed call returns, so history and results are identical. Hooks:
`on_text`, `on_thinking`, `on_event` (all normalized events), `on_request`,
`on_response`, `before_tool`, `after_tool`. Sync or awaitable in `AsyncAgent`.
`ClaudeAgent` passes native request kwargs and native responses to `on_request` /
`on_response`.

### Usage, pricing, budgets

`UsageRecord` holds token counts, cost and the labels `provider`, `turn_id`,
`workflow`, `pricing_source`. Normalization (no double counting):

| Provider | `input_tokens` | `cache_read_input_tokens` | `cache_creation_input_tokens` | `output_tokens` | `thinking_tokens` |
| --- | --- | --- | --- | --- | --- |
| Anthropic | as reported | as reported | as reported | as reported (includes thinking) | `output_tokens_details.thinking_tokens` |
| OpenAI Responses / Chat | input − cached − cache-write | `cached_tokens` | `cache_write_tokens` | as reported (includes reasoning) | `reasoning_tokens` |
| Gemini | prompt − cached + tool-use prompt | `cached_content_token_count` | 0 | candidates + thoughts | `thoughts_token_count` |
| Bedrock | input (excludes cache reads and writes) | `cacheReadInputTokens` | `cacheWriteInputTokens` (split by 5 min / 1 h lifetime) | as reported | — |

Costs come from the pricing engine (`orkexa.pricing`, ADR 0008): immutable,
versioned `PriceSnapshot`s of `ModelPrice`s, isolated `PriceResolver`s and one pinned
`PricingBasis` per request. Costs use `Decimal`; unknown or incomplete costs give
`estimated_cost_usd=None` (never 0) with a `pricing_note`, the known part in
`known_cost_usd` and the price basis (`pricing_version`, `pricing_quality`,
`pricing_conditions`, itemized `cost`). Only Anthropic prices ship (sourced); other
providers are priced through `pricing=`, a tracker's `price_resolver` or
`register_pricing()`; no Bedrock prices ship (they differ by region, profile and tier). `BudgetTracker` checks before every request and after every
recorded response; unpriced requests follow `on_unpriced="ignore" | "warn" | "reject"`,
price-policy violations `on_price_issue`.

### Errors and secrets

Generic agents map SDK errors to `ProviderError` subclasses with `provider`,
`status_code`, `request_id`, `retryable`; messages are redacted (configured keys,
`sk-…`, `AIza…`, bearer tokens). `ClaudeAgent` re-raises SDK exceptions unchanged.
API keys are kept in `ProviderConfig` (`SecretStr`, hidden in `repr`), never
serialized, never passed to sandboxes, and never shared between providers.
`CompatibleProvider` never reads `OPENAI_API_KEY`; the OpenAI/Anthropic providers
refuse a non-default `base_url` without an explicit `api_key` (no implicit
environment key to third-party hosts). OpenAI-compatible endpoints never follow a
redirect. See [ADR 0007](decisions/0007-credentials-and-endpoints.md). Bedrock uses only
AWS credentials, ignores endpoint URLs from the environment and verifies data residency
before the first paid request ([ADR 0013](decisions/0013-bedrock-credentials-and-residency.md)).

### Concurrency and cancellation

One agent = one conversation. A second concurrent `ask()` raises
`ConcurrentTurnError`. Async cancellation leaves the committed history untouched;
plain sync tool code already running in threads finishes in the background. `tool_timeout`
bounds every tool call (async: cancelled; sync: abandoned thread, documented). A sandbox
call is stopped in both cases: the drivers bind a per-call cancel check (linked to the
caller's `cancel_scope`) and set it on timeout or cancellation, so its process and
container are killed. A failing parallel call in `AsyncAgent` cancels and awaits its
siblings, like the sync driver and the orchestration driver.

### Subsystems

Memory, files, retrieval, evaluation and the dynamic package are flat modules
([ADR 0005](decisions/0005-module-layout.md)) and use
only the generic agent contract. Anthropic-only behaviors (native memory tool,
Files API, `search_result` citations, Message Batches) are selected through
capabilities, never through provider names in the runtime.

`orkexa.flow` follows the same split as the agent runtime: a sans-I/O kernel decides
what runs next in a workflow and drivers execute it
([ADR 0009](decisions/0009-flow-kernel.md)). It depends only on the data model and
Pydantic, never on an agent or provider.

`orkexa.orchestration` defines how a model chooses which member of a fixed pool works
next (`Delegate`, `Parallel`, `Staff`, `Finish`), validates every decision against the
pool and its limits, records it as plain JSON and replays it
([ADR 0010](decisions/0010-orchestration-protocol.md),
[ADR 0011](decisions/0011-dynamic-teams-and-paused-delegations.md)). Its reference driver
never runs a member itself: the caller's `delegate` does, with its own permissions.

Retrieval ranks by full text (`fts`, BM25), by vectors (`vector`) or both (`hybrid`);
a mode the index cannot serve raises instead of falling back to another. Vectors come
from an `EmbeddingModel` (`orkexa.embeddings`: `OllamaEmbeddings`, `CallableEmbeddings`)
whose identity is stored with the index, so vectors of two models are never mixed.

`orkexa.local` manages local models with Ollama: the management API (`OllamaClient`,
loopback only unless allowed), a pinned runtime installed from a verified archive
(`install_runtime`), a managed server on a free loopback port (`ManagedOllama`), offline
model packages and pinned model recipes (`build_model`). Nothing there runs on import;
`orkexa-local` is its command line.

## Package layout

```text
src/orkexa/
  __init__.py            public API
  agent.py, async_agent.py   generic agents (I/O drivers)
  claude.py              ClaudeAgent / AsyncClaudeAgent facades
  core/types.py          data model
  core/capabilities.py   Support, ModelCapabilities
  core/conversion.py     prompt parsing, cross-provider history, history files
  core/runtime.py        AgentCore (turn state machine)
  core/schema.py         provider schema conversion/validation
  core/checkpoint.py     signed turn checkpoints (pausing at a tool call or request)
  providers/base.py      Provider, ProviderConfig
  providers/options.py   typed per-provider options (AnthropicOptions, BedrockOptions, ...)
  providers/registry.py  registry and model resolution
  providers/anthropic.py, anthropic_codec.py, openai.py, gemini.py, chat_completions.py,
    compatible.py, fake.py
  providers/bedrock.py, bedrock_residency.py   Converse adapter, region checks
  pricing/               price model, snapshots, resolver, cost, built-in Anthropic prices
  tools.py, content.py, errors.py, usage.py
  memory.py, files.py, retrieval.py, embeddings.py, evaluation.py
  dynamic/               dynamic tools and agents, sandboxes, registries
  dynamic/docker/        Docker host checks, container shape, self-test
  flow/                  flow kernel, reference driver, conformance cases
  orchestration/         orchestration protocol and reference driver
  local/                 Ollama client, runtime install, managed server, packages, CLI
```

## Decisions

* [ADR 0001 – single sans-I/O runtime](decisions/0001-single-runtime.md)
* [ADR 0002 – neutral history with lossless provider codecs](decisions/0002-neutral-history.md)
* [ADR 0003 – capabilities: unknown is not supported](decisions/0003-capabilities.md)
* [ADR 0005 – flat module layout](decisions/0005-module-layout.md)
* [ADR 0006 – provider SDKs, API families and versions](decisions/0006-provider-apis.md)
* [ADR 0007 – credentials and endpoints](decisions/0007-credentials-and-endpoints.md)
* [ADR 0008 – pricing engine](decisions/0008-pricing-engine.md)
* [ADR 0009 – flow kernel](decisions/0009-flow-kernel.md)
* [ADR 0010 – orchestration protocol](decisions/0010-orchestration-protocol.md)
* [ADR 0011 – dynamic teams, parallel delegation and paused delegations](decisions/0011-dynamic-teams-and-paused-delegations.md)
* [ADR 0012 – MacOSSandbox removed (Docker-first)](decisions/0012-macos-sandbox-removed.md)
* [ADR 0013 – Amazon Bedrock: credentials, endpoints and data residency](decisions/0013-bedrock-credentials-and-residency.md)

ADR 0004 (a separate distribution for legacy imports) was withdrawn on 2026-10-31,
when that compatibility layer was removed; its number is not reused.

## Design risks

| Risk | Mitigation |
| --- | --- |
| Lossy Anthropic ↔ neutral conversion changes request bytes | codec keeps unknown blocks/fields; `tests/anthropic` compares request dicts exactly |
| Behavior drift of `ClaudeAgent` | `tests/anthropic` drives it through a fake Anthropic client in CI |
| OpenAI strict-schema rules reject Orkexa tool schemas | explicit transformation with tests; ambiguous schemas fail at request build |
| Gemini thought signatures lost between rounds | signatures stored in part meta and asserted in contract tests |
| Compatible endpoints overstate features | capabilities default to `unknown`; presets are documented as typical, not guaranteed |
| Newer SDK majors (openai 3.x, google-genai 2.x, anthropic 2.x) change APIs | version ranges pinned to tested majors; adapters use only documented fields |
