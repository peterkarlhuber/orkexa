# Public API compatibility: `claude_agent` 0.8.0 → `orkexa`

Scope: every symbol in `claude_agent.__all__` (51) and `claude_agent.dynamic.__all__`
(48), plus the documented submodule imports used by the reference docs and tests
(`claude_agent.content`, `.evaluation`, `.retrieval`, `.usage`, `.memory`, `.files`,
`.pricing`, `.tools`, `.errors`, `.agent`, `.async_agent`, `.dynamic.*`).

Strategy legend:

* **same** – the object is moved to Orkexa unchanged (identical signature, defaults,
  return types, exceptions and side effects). `claude_agent.X` *is* `orkexa.X`.
* **facade** – thin compatibility class over the shared runtime with the legacy
  signature and observable behavior (request dicts, history format, hooks).
* **extended** – legacy signature and behavior kept; new optional parameters or
  fields added with defaults that reproduce the legacy behavior.
* **generalized** – accepts the generic agent contract instead of `ClaudeAgent`;
  legacy calls keep working.

The compatibility claim for every row is enforced by running the **unchanged
reference test suite** against Orkexa (see `tests/legacy/`, the only deviations are
listed at the end of this page).

## Agents

| Symbol | Signature (0.8.0) | Orkexa | Strategy / notes |
| --- | --- | --- | --- |
| `ClaudeAgent` | 33 keyword-only params (see inventory), generic `ClaudeAgent[OutputT]` | `orkexa.ClaudeAgent` | **facade** over `orkexa.Agent` with the Anthropic adapter. Same params and defaults (`model="claude-haiku-5-5"`, `effort="medium"`, …). `messages`/`history()`/`save_history()` keep the Messages-API dict format; `on_request` receives the native request kwargs, `on_response` and `last_response` the native SDK message; SDK exceptions are not re-mapped. |
| `AsyncClaudeAgent` | same | `orkexa.AsyncClaudeAgent` | **facade** over `orkexa.AsyncAgent`. |
| `ClaudeAgent.ask(prompt, *, system=None, effort=None, max_tokens=None) -> OutputT` | | same | `str` without `output_format`, validated model instance with it. |
| `count_tokens(prompt=None) -> int` | | same | native `messages.count_tokens` (beta namespace with betas). |
| `add_tool`, `add_server_tool`, `remove_tool`, `reset`, `reset_usage`, `history`, `save_history`, `load_history` | | same | `load_history` additionally accepts the Orkexa v1 history format. |
| properties `usage`, `last_usage`, `last_turn_usage`, `usage_records`; attributes `client`, `name`, `model`, `tools`, `server_tools`, `memory`, `files`, `index`, `last_response`, every constructor option | | same | attributes stay mutable and are read for every request (legacy tests change `agent.model` between turns). |
| `Effort`, `ThinkingMode`, `ThinkingDisplay`, `RefusalMode`, `Fallbacks` | `Literal` aliases | same | `Effort` is also the generic effort type. |
| `ToolCall(id, name, input, agent)` | frozen dataclass | same | passed to `before_tool`/`after_tool` for **all** providers. |

## Tools and content

| Symbol | Orkexa | Strategy / notes |
| --- | --- | --- |
| `Tool(function=None, *, name=None, strict=True)` with `schema()`, `execute()`, `aexecute()` | `orkexa.Tool` | **extended**: added `definition()` returning the provider-neutral `ToolDefinition`; `schema()` keeps the Anthropic format. |
| `ToolContent`, `ContentBlock`, `Prompt` | `orkexa.content` | **same** (dict blocks are Orkexa's provider-neutral *input* block format). |
| `image`, `document`, `uploaded_file`, `text`, `to_content`, `to_tool_content`, `describe_content` | `orkexa.content` | **same** return values. `uploaded_file()` gains an optional `provider=` tag (default `"anthropic"`, the only provider whose file ids it produced in 0.8.0). |

## Exceptions

| Symbol | Orkexa | Notes |
| --- | --- | --- |
| `ClaudeAgentError` | alias of `orkexa.OrkexaError` | `except ClaudeAgentError` catches every Orkexa error. |
| `ToolDefinitionError(TypeError)`, `ToolExecutionError(RuntimeError)`, `ToolInputError`, `ToolResultError(RuntimeError)`, `MemoryToolError`, `IncompleteResponseError`, `MaxToolRoundsError`, `RefusalError(message, *, stop_details)` + `.category`, `UnexpectedResponseError`, `EvaluationFailedError(message, *, evaluation)` | `orkexa.errors` | **same** classes and builtin bases. |
| new | `ConfigurationError(ValueError)`, `UnsupportedCapabilityError(ValueError)`, `UnknownCapabilityError`, `HistoryConversionError(ValueError)`, `ToolSchemaError(ToolDefinitionError)`, `StructuredOutputError(UnexpectedResponseError)`, `ConcurrentTurnError(RuntimeError)`, `MissingDependencyError(ImportError)`, `ProviderError` + `AuthenticationError`, `RateLimitError`, `InvalidRequestError`, `ProviderConnectionError`, `ProviderServerError` | additions only. |

## Usage and pricing

| Symbol | Orkexa | Strategy / notes |
| --- | --- | --- |
| `UsageRecord` (14 fields) | `orkexa.UsageRecord` | **extended**: new trailing defaulted fields `provider`, `turn_id`, `workflow`, `pricing_source`; positional/keyword construction of 0.8.0 unchanged; `total_input_tokens`/`total_tokens` unchanged. |
| `UsageSummary` (12 fields) | same | **same** fields and semantics (`estimated_cost_usd is None` if any request is unpriced). |
| `UsageTracker()` with `add`, `records`, `usage`, `usage_by_agent`, `reset` | same | **extended**: `usage_by_provider()`, `usage_by_model()`, `usage_by_turn()`, `usage_by_workflow()`, `check()` (no-op); `orkexa.usage.workflow(label)` labels requests. |
| `record_api_usage(response, *, agent_name, default_model, custom_pricing=None, batch=False)` | `orkexa.usage` | **same**; used by the Anthropic adapter. |
| `summarize_usage(records)` | same | **same**. |
| `ModelPricing`, `ModelPricing.of`, `MODEL_PRICING`, `resolve_pricing(model, total_input_tokens=0)` | `orkexa.pricing` | **extended**: `resolve_pricing(..., provider="anthropic")`; `PROVIDER_PRICING` catalog per provider (`MODEL_PRICING` is its Anthropic table). |

## Memory, files, retrieval

| Symbol | Orkexa | Notes |
| --- | --- | --- |
| `MemoryBackend` (+ `definition`, `execute`, commands), `EphemeralMemory`, `FileMemory`, `SQLiteMemory`, `MEMORY_ROOT`, `MEMORY_TOOL_NAME` | `orkexa.memory` | **extended**: `tool()` returns a standard `@Tool` with the same commands for providers without the native memory tool. |
| `FileStore`, `LocalFileStore`, `SQLiteFileStore`, `StoredFile`, `FILES_TOOL_NAME` | `orkexa.files` | **same**; `files_api=` stays an Anthropic-specific extension. |
| `Index`, `SQLiteIndex`, `Chunk`, `Hit`, `Embedder`, `SEARCH_TOOL_NAME`, `split_text` | `orkexa.retrieval` | **same**. |
| `Citation(source, title, cited_text, search_result_index)` | `orkexa.retrieval` | **extended**: `kind` (`"api_native"`/`"retrieval"`), `provider`, `url`. |
| `citations(response)` | same | **generalized**: accepts a native Anthropic response (0.8.0 behavior), an Orkexa `ModelResponse` or an agent. |

## Evaluation

| Symbol | Orkexa | Notes |
| --- | --- | --- |
| `Criterion`, `CriterionResult`, `Evaluation`, `EvaluationCase`, `Refinement`, `DEFAULT_EVALUATOR_PROMPT`, `render` | `orkexa.evaluation` | **same**. |
| `Evaluator(criteria=None, *, model="claude-opus-5-5", system_prompt, effort="medium", max_tokens=4096, client=None, api_key=None, usage_tracker=None, name="Evaluator")` | same | **extended**: `provider=`, `options=`; `model` may be provider-qualified. `evaluator.agent` is an `orkexa.Agent` (was `ClaudeAgent`). |
| `Evaluator.evaluate_many(cases, *, batch=True, poll_interval=30.0, timeout=None)` | same | Anthropic: native Message Batches, unchanged. Providers without a native batch adapter: `batch=True` raises `UnsupportedCapabilityError`; `batch=False` evaluates sequentially and honors `timeout`. |
| `refine(agent, prompt, evaluator, ...)`, `tool_results_of_last_turn(agent)` | same | **generalized** to every Orkexa agent. |

## Dynamic agents

All 48 symbols of `claude_agent.dynamic` are moved to `orkexa.dynamic`.

| Symbol | Notes |
| --- | --- |
| `CodingAgent`, `AgentArchitect`, `Orchestrator` | **extended**: `provider=`, `options=`; internal agent is an `orkexa.Agent`. |
| `AgentFactory(tools, output_formats=None, policy=None, *, delegation_tools=None, client=None, api_key=None, usage_tracker=None)` | **extended**: `credentials=` maps provider names to `ProviderConfig`; the legacy `client`/`api_key` only apply to agents of the default provider (`anthropic`) and are never handed to agents of other providers. Returns an `orkexa.Agent`. |
| `AgentSpec.model`, `DynamicPolicy.allowed_models` | may be provider-qualified (`"openai:gpt-5.5"`); unqualified ids mean `anthropic:` as before. |
| `BudgetTracker(limit_usd=None)` | **extended**: `on_unpriced="ignore" \| "warn" \| "reject"` (default `"ignore"` = 0.8.0 behavior). |
| `ToolBuilder.attach/revoke/as_tools` | **generalized** to every Orkexa agent. |
| `DockerSandbox(image=...)` | default image renamed to `orkexa-sandbox`; package mounted at `/opt/orkexa/orkexa`. |
| everything else (`ToolRegistry`, `ToolRecord`, `AgentRegistry`, `AgentRecord`, `SandboxedTool`, `Sandbox`, `SubprocessSandbox`, `ValidationResult`, `SandboxTestResult`, `StaticReport`, `check_tool_source`, `check_tool_tests`, approvers, `ApprovalRequest`, review functions/criteria, `OutputFormatRegistry`, `RESERVED_TOOL_NAMES`, errors) | **same**. |

## Legacy import path (`claude_agent`)

The optional distribution `orkexa-claude-compat` installs the top-level package
`claude_agent` that contains only re-exports (see
[ADR 0004](decisions/0004-legacy-import-distribution.md)). Submodules with a 1:1
counterpart are *aliases* (`claude_agent.retrieval is orkexa.retrieval`), so
monkeypatching and private helpers used by existing tests keep working.

## Known, deliberate deviations

| Deviation | Reason | Test |
| --- | --- | --- |
| Sandbox temporary directory prefix `orkexa-sandbox-` (was `claude-agent-sandbox-`); Docker mount `/opt/orkexa/orkexa`; default image `orkexa-sandbox` | the spec requires migrating hardcoded `claude_agent` paths | `tests/legacy/test_dynamic_sandbox.py::test_working_directory_is_temporary` asserts the new prefix (the only edited legacy assertion); `tests/test_sandbox_migration.py` |
| `evaluator.agent`, `coding_agent.agent`, `architect.agent`, `orchestrator.agent` and `AgentFactory.create()` return `orkexa.Agent` instead of `ClaudeAgent` | dynamic components must accept any provider | their `.messages` are Orkexa messages, not Messages-API dicts; all attributes used by 0.8.0 (`usage_tracker`, `model`, `effort`, `max_tokens`, `tools`, …) are identical |
| A turn cannot run concurrently with another turn on the same agent, and `reset()`/`load_history()` are refused while a turn runs (`ConcurrentTurnError`) | 0.8.0 documented "do not run two `ask()` calls of the same agent at the same time" but did not enforce it; the running turn would commit the old history again | `tests/test_runtime.py`, `tests/test_failure_paths.py` |
| The usage of an answered request is recorded even if `on_response` raises | 0.8.0 called `on_response` before recording, so a failing hook lost the usage | `tests/test_failure_paths.py` |
| A shared `BudgetTracker` that is used up (`spent_usd >= limit_usd`, e.g. a limit of 0) stops the *next* model request before it is sent; `check()` still raises only above the limit | budgets must be enforced before requests where possible; 0.8.0 sent one more request and then raised | `tests/test_usage_budget.py` |
| `AgentRegistry.register()` rejects names outside `[A-Za-z][A-Za-z0-9_-]{0,63}` (the rule `AgentFactory.validate()` already applied) | 0.8.0 joined the name to the registry path unchecked | `tests/test_security.py` |
| `ToolRegistry.add_generated()` rejects names outside `[A-Za-z0-9_-]{1,128}` with `RegistryError`; such stored records are skipped on load | 0.8.0 joined the name to the registry path unchecked (`../x` wrote outside the directory) | `tests/test_security.py` |
| `request_options` with `temperature`, `top_p` or `top_k` raise `ConfigurationError` (a `ValueError`) when the options are built | Anthropic API version 2026-10-21, which `anthropic>=2.0` sends, removed them (HTTP 400; the SDK no longer takes them as arguments). A client pinned to `anthropic-version: 2023-06-01` can send them via `request_options={"extra_body": {...}}` | `tests/test_api_versions.py` |
| Reasoning of another model of the same provider is not replayed (`HistoryConversionWarning`); `Message.model` is the requested model id | reasoning signatures are model-bound | `tests/test_history_models.py` |
