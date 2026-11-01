# Providers and capabilities

| Provider name | API | SDK (extra) | Default endpoint | Credentials |
| --- | --- | --- | --- | --- |
| `anthropic` (default) | Messages API | `anthropic` | Anthropic | `ANTHROPIC_API_KEY` or `api_key=` |
| `openai` | Responses API | `openai` | `https://api.openai.com/v1` | `OPENAI_API_KEY` or `api_key=` |
| `gemini` | Gemini API / Vertex AI (`google-genai`) | `gemini` | Google | `GEMINI_API_KEY`/`GOOGLE_API_KEY` or `api_key=` |
| `bedrock` | Amazon Bedrock Converse / ConverseStream | `bedrock` (`boto3`) | `https://bedrock-runtime.<region>.amazonaws.com` | AWS credential chain; `api_key=` is rejected |
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

| Capability | anthropic | openai | gemini | bedrock | compatible (Chat) | ollama / vllm |
| --- | --- | --- | --- | --- | --- | --- |
| tools | ✓ | ✓ | ✓ | ✓ Claude, Nova | unknown | unknown |
| forced_tool_choice | ✓ (not with every thinking mode/model) | ✓ | ✓ (`required` / named tool) | ✓ Claude 4.5/4.6 and older, Nova; ✗ Claude Sonnet 5.5, Opus 5.5, Fable 5.1, Mythos 5.1 | unknown | unknown |
| parallel_tool_calls | ✓ | ✓ | ✓ | ✓ Claude | unknown | unknown |
| strict_tool_schemas | ✓ | ✓ | ✗ (validated locally) | ✓ Claude 4.5 and later | unknown | unknown |
| structured_output | ✓ | ✓ (strict JSON schema) | ✓ (response JSON schema) | ✓ Claude 4.5 and later, ✗ Nova | unknown | ✓ |
| streaming | ✓ | ✓ | ✓ | ✓ Claude, Nova | unknown | ✓ |
| image_input / document_input | ✓ | ✓ for multimodal model families, else unknown | ✓ | ✓ Claude (no images on Claude 3.5 Haiku), Nova Pro/Lite/2 Lite; ✗ Nova Micro | unknown | unknown |
| reasoning (`effort`) | ✓ | ✓ reasoning models (`gpt-6*`, `gpt-5*`, `o*`), ✗ classic models, else unknown | ✓ Gemini 3 and 4 (`thinking_level`), ✗ Gemini 2.x (use `thinking_budget`) | ✓ Claude 5.x, else unknown | unknown (`send_reasoning_effort`) | unknown |
| caching | ✓ (`cache_control`) | ✓ (automatic) | ✓ (implicit) | ✓ Claude 3.7 Sonnet and later, Nova (`prompt_caching=True`) | unknown | unknown |
| server_tools | ✓ | ✓ (hosted tools as native dicts) | ✓ (native tool dicts) | ✗ | ✗ in Chat mode | ✗ in Chat mode |
| native_memory_tool | ✓ | ✗ (memory is a function tool) | ✗ (function tool) | ✗ (function tool) | ✗ | ✗ |
| token_counting | ✓ | ✓ (`responses.input_tokens.count`) | ✓ | ✓ Claude 4.5/4.6, ✗ Claude 4.7 and later | ✗ | ✗ |
| batch | ✓ (Message Batches) | ✗ | ✗ | ✗ | ✗ | ✗ |
| system_messages | ✓ (model dependent) | ✓ | ✗ (use `system_prompt`) | ✗ (use `system_prompt`) | unknown | ✓ |

For `bedrock`, the capabilities follow from the foundation model behind the id
(`eu.anthropic.claude-sonnet-5-5` and its ARN are treated like
`anthropic.claude-sonnet-5-5`). Every model not listed, and ids that do not name
their model (application inference profiles, provisioned throughput, custom and
imported models) are `unknown`: declare what the model supports with `capabilities=`.

## What each adapter does

**Anthropic** supports prompt caching, thinking modes, betas through
`client.beta.messages`, fallbacks, eager input streaming and `request_options`. History parts keep every native field, including thinking
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

**Bedrock** uses the model-independent Converse and ConverseStream operations of
`bedrock-runtime` through `boto3`; see [Amazon Bedrock](#amazon-bedrock) below.

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
* Orkexa ships prices for Anthropic models only; costs of other providers, including
  Claude on Bedrock, are `None` until registered (see
  [Usage and budgets](../guides/usage-and-budgets.md) and
  [Bedrock prices](../guides/pricing.md#amazon-bedrock)).
* The model ids in this documentation are examples; use ids your account can access.

## Amazon Bedrock

The `bedrock` provider (`pip install "orkexa[bedrock]"`) calls the Bedrock
[Converse API](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html)
with `boto3`. It runs Claude, Amazon Nova and other Bedrock models with the same
agent code as every other provider.

```python
from orkexa import Agent, BedrockOptions

agent = Agent(
    provider="bedrock",
    model="eu.anthropic.claude-sonnet-5-5",
    options=BedrockOptions(region_name="eu-central-1", profile_name="my-profile"),
)
# or as one string; everything after the first colon is the model id:
agent = Agent(
    model="bedrock:anthropic.claude-sonnet-4-5-20250929-v1:0", options=BedrockOptions(region_name="eu-west-1")
)
```

### Credentials and region

* AWS credentials come only from boto3's credential chain: an IAM role, IAM Identity
  Center (SSO), a named profile (`BedrockOptions(profile_name=...)` or `AWS_PROFILE`),
  the `AWS_*` key variables, or a Bedrock API key in `AWS_BEARER_TOKEN_BEDROCK`. You
  can also pass `BedrockOptions(session=boto3.Session(...))` or a `bedrock-runtime`
  client as `client=`. See [Installation](installation.md#aws-credentials-for-amazon-bedrock).
* `api_key=` is rejected with `ConfigurationError`, and the provider never reads
  `ANTHROPIC_API_KEY` or another provider's key.
* The region is `BedrockOptions(region_name=...)`, else the session's region
  (`AWS_DEFAULT_REGION`, the profile). It is the **source region**: the endpoint Orkexa
  calls, the region AWS bills, and the region Orkexa reports for pricing.
* Endpoint URLs from the environment or the AWS config file (`AWS_ENDPOINT_URL*`) are
  ignored, so signed requests go only to the regional AWS endpoint. A `base_url=`
  you pass explicitly (for example an interface VPC endpoint) must use `https`; plain
  `http` is accepted only for `localhost`.
* The model must be enabled for your account in that region, and the caller needs
  `bedrock:InvokeModel` (Converse) and `bedrock:InvokeModelWithResponseStream`
  (ConverseStream) on the model or inference profile.

### Model ids

`model` is whatever Converse accepts as `modelId`:

| Kind | Example | Where it runs |
| --- | --- | --- |
| Foundation model id | `anthropic.claude-sonnet-4-6`, `amazon.nova-pro-v1:0` | In the source region (not every model offers in-region inference) |
| Geographic cross-Region inference profile | `eu.anthropic.claude-sonnet-5-5` | One of the profile's destination regions in that geography |
| Global cross-Region inference profile | `global.anthropic.claude-sonnet-5-5` | Any commercial AWS region |
| ARN | `arn:aws:bedrock:eu-central-1::foundation-model/...`, `arn:aws:bedrock:eu-central-1:111122223333:inference-profile/eu.anthropic...`, application inference profiles, provisioned throughput, custom and imported models | Depends on the resource |

Many current Claude models are offered only through cross-Region inference profiles
([supported Regions and models](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-support.html));
check the model card for your region. Model ids contain colons
(`...-v1:0`); `"bedrock:<id>"` keeps them intact.

### Options

`BedrockOptions` maps to Converse fields: `temperature`, `top_p` and
`stop_sequences` (`inferenceConfig`), `guardrail` (`guardrailConfig`),
`service_tier` (`"default"`, `"priority"`, `"flex"`), `performance_latency`,
`prompt_caching` with `cache_ttl` (cache points after the system prompt, the tools
and the last message), `additional_model_request_fields` for model-specific fields
(fields the runtime manages are rejected) and `request_metadata` (tags in invocation
logs). `effort` maps to `outputConfig.effort` on Claude 5.x. See the
[API reference](../reference/providers.md#orkexa.BedrockOptions).

### What the adapter does

* Tools always run in Orkexa; Converse has no `tool_choice` `"none"` (remove the
  tools instead). Parallel tool calls work where the model returns several `toolUse`
  blocks; their results go back in one user message.
* Claude reasoning blocks are kept unchanged in the history and replayed to the same
  model, which Claude requires in tool loops.
* Images (PNG, JPEG, GIF, WebP; at most 20 per message, 3.75 MB each) and documents
  (PDF, CSV, DOC, DOCX, XLS, XLSX, HTML, TXT, MD; at most 5 per message, 4.5 MB each)
  are sent as bytes and checked before the request
  ([Message](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Message.html)).
  `s3://` sources work only on Amazon Nova; Bedrock does not fetch `http(s)` URLs, and
  Orkexa does not download them. Document citations are not supported.
* Usage: `input_tokens` excludes cache reads and writes, which are counted
  separately, so nothing is counted twice.
* AWS error codes map to Orkexa errors: `AccessDeniedException` and expired
  credentials to `AuthenticationError`, throttling to `RateLimitError`,
  `ValidationException` to `InvalidRequestError`; secrets (session tokens, Bedrock API
  keys, signatures) are redacted from messages.

### Async and cancellation

`boto3` is synchronous. `AsyncAgent` runs every Bedrock call in its own worker thread
(not the event loop's shared executor), so the event loop never blocks. Cancelling a
streamed request closes the stream. A non-streamed call cannot be aborted inside
boto3: after cancellation it finishes in its thread and its result and usage are
discarded; AWS may still bill it. Use streaming when you need to stop requests early.

### Data residency

`BedrockOptions(data_residency="eu_only")` verifies, before the first paid request,
that the client region and every region the model or inference profile can route to
are EU regions, and raises `DataResidencyError` (nothing is sent) when they are not or
cannot be checked. `allowed_regions=(...)` sets an explicit allowlist. See
[Security: Amazon Bedrock data residency](../guides/security.md#amazon-bedrock-data-residency).

### Anthropic API or Claude on Bedrock?

Claude through `provider="anthropic"` and Claude through `provider="bedrock"` are two
different providers in Orkexa, with different contracts:

| | `anthropic` (Claude API) | `bedrock` (Claude on Amazon Bedrock) |
| --- | --- | --- |
| Account and credentials | Anthropic account, `ANTHROPIC_API_KEY` | AWS account, IAM principal (role, SSO, profile) or Bedrock API key |
| Where data is processed | Under Anthropic's terms and data handling | In the AWS region(s) of the model id or inference profile, under your AWS agreements |
| Prices and billing | Anthropic's list prices, billed by Anthropic; Orkexa ships these prices | AWS prices per region and inference profile, on your AWS bill ([model card](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-sonnet-5-5.html)); Orkexa ships no Bedrock prices and never uses Anthropic's |
| API | Messages API, all Anthropic features: server tools (web search, code execution, ...), Files API, Message Batches, citations, native memory tool, token counting, `ClaudeAgent` | Converse API, the feature subset in the table above: no server tools, Files API, batches, citations or native memory tool; token counting and forced tool choice depend on the model |
| Model ids | `claude-sonnet-5-5` | `anthropic.claude-sonnet-5-5`, `eu.anthropic.claude-sonnet-5-5`, ARNs |

Moving a conversation between them is a cross-provider continuation like any other:
Claude thinking blocks, signatures and native parts of one provider are not replayed
to the other. With the default `history_policy="strict"` this raises
`HistoryConversionError`; `history_policy="convert"` drops them with a warning (see
[History and provider switching](../guides/history.md)). Text, images, documents,
tool calls and tool results carry over; files uploaded to Anthropic's Files API do not
(Bedrock cannot read them under either policy).
