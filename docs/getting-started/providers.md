# Providers and capabilities

| Provider name | API | SDK (extra) | Default endpoint | Credentials |
| --- | --- | --- | --- | --- |
| `anthropic` (default) | Messages API | `anthropic` | Anthropic | `ANTHROPIC_API_KEY` or `api_key=` |
| `openai` | Responses API | `openai` | `https://api.openai.com/v1` | `OPENAI_API_KEY` or `api_key=` |
| `gemini` | Gemini API / Vertex AI (`google-genai`) | `gemini` | Google | `GEMINI_API_KEY`/`GOOGLE_API_KEY` or `api_key=` |
| `compatible` | Chat Completions (default) or Responses | `openai` | `base_url=` (required) | `api_key=` only |
| `ollama` | as `compatible` | `openai` | `http://localhost:11434/v1` | `api_key=` only |
| `vllm` | as `compatible` | `openai` | `http://localhost:8000/v1` | `api_key=` only |

Own providers are added with `orkexa.register_provider(name, factory)`; the factory is
a `Provider` subclass (or a callable `(config, asynchronous) -> Provider`).

## Capabilities

Every adapter reports a `ModelCapabilities` per model, each entry `supported`,
`unsupported` or `unknown`. Before a request the runtime checks every feature the
request uses; nothing is silently degraded:

* `unsupported` raises `UnsupportedCapabilityError` with the reason
  (e.g. *"Orkexa does not implement the OpenAI Batch API"*);
* `unknown` raises `UnknownCapabilityError` naming the override, e.g.
  `capabilities={"tools": "supported"}`.

`Agent(..., capabilities={...})` overrides entries; the adapters also use the
effective values for encoding decisions (e.g. strict schemas on a compatible endpoint).
`agent.capabilities` shows the result.

| Capability | anthropic | openai | gemini | compatible (Chat) | ollama / vllm |
| --- | --- | --- | --- | --- | --- |
| tools | ✓ | ✓ | ✓ | unknown | unknown |
| forced_tool_choice | ✓ (not with every thinking mode/model) | ✓ | ✓ (`required` / named tool) | unknown | unknown |
| parallel_tool_calls | ✓ | ✓ | ✓ | unknown | unknown |
| strict_tool_schemas | ✓ | ✓ | ✗ (validated locally) | unknown | unknown |
| structured_output | ✓ | ✓ (strict JSON schema) | ✓ (response JSON schema) | unknown | ✓ |
| streaming | ✓ | ✓ | ✓ | unknown | ✓ |
| image_input / document_input | ✓ | ✓ for multimodal model families, else unknown | ✓ | unknown | unknown |
| reasoning (`effort`) | ✓ | ✓ reasoning models (`gpt-6*`, `gpt-5*`, `o*`), ✗ classic models, else unknown | ✓ Gemini 3 and 4 (`thinking_level`), ✗ Gemini 2.x (use `thinking_budget`) | unknown (`send_reasoning_effort`) | unknown |
| caching | ✓ (`cache_control`) | ✓ (automatic) | ✓ (implicit) | unknown | unknown |
| server_tools | ✓ | ✓ (hosted tools as native dicts) | ✓ (native tool dicts) | ✗ in Chat mode | ✗ in Chat mode |
| native_memory_tool | ✓ | ✗ (memory is a function tool) | ✗ (function tool) | ✗ | ✗ |
| token_counting | ✓ | ✓ (`responses.input_tokens.count`) | ✓ | ✗ | ✗ |
| batch | ✓ (Message Batches) | ✗ | ✗ | ✗ | ✗ |
| system_messages | ✓ (model dependent) | ✓ | ✗ (use `system_prompt`) | unknown | ✓ |

## What each adapter does

**Anthropic** encodes requests exactly like `claude-agent` 0.8.0 (prompt caching,
thinking modes, betas through `client.beta.messages`, fallbacks, eager input streaming,
`request_options`). History parts keep every native field, including thinking
signatures; unknown block types are preserved as native parts. `anthropic>=2.0` uses
API version 2026-10-21, which removed `temperature`, `top_p` and `top_k`; Orkexa
rejects them in `request_options` before anything is sent (a client pinned to
`anthropic-version: 2023-06-01` can still send them via `extra_body`).

**OpenAI** uses the Responses API statelessly: `store=False` by default, full input on
every request, and for reasoning models `include=["reasoning.encrypted_content"]` so
reasoning items are replayed unchanged in later requests. Tools are strict functions:
optional parameters become nullable and the `null` values are removed again before
local validation, so the Python defaults apply. Structured output uses the same strict
schema transformation in `text.format` and is validated locally. Refusals map to
`RefusalError`; `max_output_tokens` truncation to `IncompleteResponseError` (also when a
stream ends with `response.incomplete`); `failed` responses to `ProviderServerError`.
Only `completed` responses are answers: `queued`/`in_progress` and `cancelled` raise
`UnexpectedResponseError` and leave the history unchanged. Background mode
(`request_options={"background": True}`) is rejected, because an agent needs each
response to continue its turn and does not poll.

**Gemini** builds `google-genai` `Content`/`Part` objects, always disables the SDK's
automatic function calling (Orkexa executes tools after `before_tool`), keeps
`thought_signature` values of function-call parts and replays them, maps `effort`
to `thinking_level` on Gemini 3 and 4, inlines `$ref`s of response schemas, and maps
`SAFETY`/`PROHIBITED_CONTENT` endings to refusals. Function calls without an id get
a synthetic id that is never sent back.

**Compatible endpoints** speak Chat Completions by default (`CompatibleOptions(api=
"responses")` selects the Responses API; there is no automatic fallback between
protocols). Every capability is `unknown` until declared, except batch, token counting
and the native memory tool, which are `unsupported`. Structured output uses
`response_format={"type": "json_schema"}` (strict only if `strict_tool_schemas` is
declared) or, with `CompatibleOptions(structured_output="json_object")`, JSON mode plus
the schema in the system prompt. Reasoning content returned by such servers is never
sent back. `OPENAI_API_KEY` is never read.

## Known API limits

* Batch processing is implemented for Anthropic only; `Evaluator.evaluate_many(...,
  batch=True)` raises `UnsupportedCapabilityError` elsewhere; `batch=False` evaluates
  sequentially with the same timeout semantics. Batch items are built like normal
  requests (thinking, effort, cache TTL, server tools, `tool_choice`,
  `request_options`; betas use the beta batch API). Local tools are rejected, since a
  batch item is a single request in which they cannot run.
* Provider file uploads (`uploaded_file()`, Anthropic Files API) are Anthropic-only;
  inline images and documents work on every provider with media input.
* Reasoning and thinking parts are provider-specific. Continuing a conversation with
  another provider requires `history_policy="convert"` (they are dropped with a
  `HistoryConversionWarning`); the default `"strict"` raises `HistoryConversionError`.
* Orkexa ships prices for Anthropic models only; costs of other providers are `None`
  until registered (see [Usage and budgets](../guides/usage-and-budgets.md)).
* The model ids in this documentation are examples; use ids your account can access.
