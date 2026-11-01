# Troubleshooting and FAQ

Every Orkexa exception derives from `OrkexaError` and says what failed and what to
change. This page collects the common cases.

## Errors and what to do

| Exception | Usual cause | Fix |
| --- | --- | --- |
| `MissingDependencyError` | The provider's SDK is not installed | Install the extra named in the message, e.g. `pip install "orkexa[openai]"` |
| `AuthenticationError` | Missing, invalid or insufficient key (HTTP 401/403) | Set the provider's environment variable or pass `api_key=`; for a custom `base_url`, an explicit `api_key=` is required |
| `ConfigurationError` | Contradicting settings, e.g. a custom `base_url` without a key, `request_options` that collide with agent arguments, removed API parameters, OpenAI background mode | Follow the message; it names the argument |
| `UnsupportedCapabilityError` | The model or provider cannot do what you asked (e.g. `effort` on a classic model, forced tool choice on Gemini) | Remove the feature or pick another model; `exc.capability` names the feature |
| `UnknownCapabilityError` | Orkexa does not know whether the endpoint supports the feature (every compatible endpoint, new model families) | Declare it if you know it works: `capabilities={"tools": "supported"}` |
| `RateLimitError`, `ProviderServerError`, `ProviderConnectionError` | Transient provider problems | `exc.retryable` is `True`; retry with backoff. The SDK already retries (`max_retries=`) |
| `InvalidRequestError` | The provider rejected the request (HTTP 400/404/409/413/422) | Check the model id, input sizes and options; `exc.request_id` helps the provider's support |
| `IncompleteResponseError` | The output token limit or context window was reached | Raise `max_tokens`, shorten the history (`reset()`), or ask for shorter output |
| `RefusalError` | The model declined | Rephrase; or `on_refusal="return"` to get the text instead of an exception |
| `StructuredOutputError` | The output does not validate against `output_format` | Simplify the schema, describe fields better, or use a model with native structured output |
| `MaxToolRoundsError` | The model kept calling tools | Raise `max_tool_rounds` or improve the tool descriptions |
| `HistoryConversionError` | A history with provider-specific parts was sent to another provider | `history_policy="convert"` drops what cannot be converted, with a warning |
| `ConcurrentTurnError` | A second `ask()`, `reset()` or `load_history()` while a turn runs | Use one agent per concurrent conversation |
| `DataResidencyError` | `bedrock` with `data_residency="eu_only"` or `allowed_regions`: the region, an inference profile destination, the endpoint or the guardrail is outside the allowlist, or it could not be verified | See [Amazon Bedrock](#amazon-bedrock) below; nothing was sent |
| `UnpricedModelError` (`orkexa.dynamic`) | `BudgetTracker(on_unpriced="reject")` and no price for the model | `register_pricing()` or `pricing=` on the agent |

Provider errors are redacted: messages never contain API keys. The SDK exception is not
attached (`exc.__cause__` is `None`, so tracebacks cannot print request headers); use
`exc.provider`, `exc.status_code`, `exc.request_id` and `exc.retryable`, and quote the
`request_id` when you contact the provider.

## Amazon Bedrock

| Message or error | Usual cause | Fix |
| --- | --- | --- |
| `ConfigurationError`: no AWS region configured | Neither `BedrockOptions(region_name=...)` nor `AWS_DEFAULT_REGION` / the profile's region is set | Set the region explicitly |
| `ConfigurationError`: not with `api_key=` | `api_key=` was passed to a `bedrock` agent | Remove it; use AWS credentials (profile, SSO, role) or `AWS_BEARER_TOKEN_BEDROCK` |
| `AuthenticationError` with `ExpiredToken`, `UnrecognizedClientException` or a credentials error | No or expired AWS credentials | `aws sso login --profile ...`, check `AWS_PROFILE`, or refresh the role session |
| `AuthenticationError` with `AccessDeniedException` | The IAM principal lacks `bedrock:InvokeModel` / `bedrock:InvokeModelWithResponseStream` on the profile or on the foundation model in one of the profile's destination regions, an SCP denies the region, or model access is not enabled for the account | Grant the permissions for every destination region (see [required IAM permissions](guides/security.md#required-iam-permissions)); check SCPs and model access in the Bedrock console for the source region |
| `InvalidRequestError` with `ValidationException` naming the model id | The model is not offered in this region, or only through a cross-Region inference profile | Use the profile id (`eu.anthropic...`) or another region; check the model card |
| `InvalidRequestError` with `ResourceNotFoundException` | Wrong model id, profile id or ARN, or an ARN of another account or region | Copy the id from the Bedrock console or `aws bedrock list-inference-profiles` for the same region |
| `UnsupportedCapabilityError` for `forced_tool_choice` | Claude Sonnet 5.5, Opus 5.5, Fable 5.1 and Mythos 5.1 on Bedrock accept only automatic tool choice | Use `tool_choice="auto"` and describe the expected tool in the prompt |
| `DataResidencyError`: global cross-Region inference profile | A `global.` profile routes to all commercial regions | Use an `eu.` profile or a foundation model id in an EU region |
| `DataResidencyError`: can route requests to eu-west-2 / eu-central-2 | The EU profile lists London or Zurich as destinations for this source region; they are not EU regions | Call it from another EU source region, or use a model with an EU-only destination list; `aws bedrock get-inference-profile` shows the destinations |
| `DataResidencyError`: could not be verified ... needs `bedrock:GetInferenceProfile` | The principal may not read the inference profile (or `GetGuardrail` for a guardrail) | Grant `bedrock:GetInferenceProfile` (and `bedrock:GetGuardrail`) in the source region |
| `DataResidencyError` with an injected `client=` | The residency check needs a control-plane client | Pass `BedrockOptions(control_client=boto3.client("bedrock", region_name=...))` |
| `DataResidencyError`: S3 and URL media sources | `s3://` or URL images and documents cannot be checked | Send the content as bytes (`image(path)`, `document(path)`) |
| `estimated_cost_usd` is `None` with `tariff:aws_region=...` or `price` missing | Orkexa ships no Bedrock prices; a price must list the source region | Register a price with an `aws_region` tariff (see [Bedrock prices](guides/pricing.md#amazon-bedrock)) |

## FAQ

**Which provider SDK versions are supported?** The ranges in `pyproject.toml`
(`anthropic>=2.0,<3`, `openai>=3.0,<4`, `google-genai>=2.0,<3`, `boto3>=1.43.80,<2`).
CI runs the suite with the newest versions and with the lowest supported ones.

**Why does my compatible endpoint raise `UnknownCapabilityError` for tools?** Servers
differ, so Orkexa assumes nothing. Declare what your server supports, e.g.
`capabilities={"tools": "supported", "structured_output": "supported"}`.

**Why is `OPENAI_API_KEY` not used for my Ollama or vLLM server?** On purpose: a
compatible endpoint is another host, and the key belongs to OpenAI. Pass `api_key=`
if your server needs one.

**Does Orkexa store my conversations?** No, not unless you configure a store or call
`save_history()`. See [Data flows and retention](compliance/data-flows.md).

**Can I switch providers in the middle of a conversation?** Yes, with
`history_policy="convert"`; see [History and provider switching](guides/history.md).

**Why is my Claude conversation not continued when I switch from `anthropic` to
`bedrock`?** They are different providers: Claude thinking blocks and native parts of
one are not replayed to the other. Use `history_policy="convert"`; see
[Anthropic API or Claude on Bedrock?](getting-started/providers.md#anthropic-api-or-claude-on-bedrock).

**How do I run the tests without API keys?** `pytest` runs offline by default; tests
that call real APIs are marked `live` and skip themselves without keys.

**Is `SubprocessSandbox` safe for untrusted code?** No. It is not a security boundary;
use `DockerSandbox` (the default of `ToolRegistry`) or stronger isolation
([Security](guides/security.md)).

**Why does my generated tool fail with "Cannot start docker"?** `ToolRegistry` runs
generated code in `DockerSandbox` by default and refuses to run it without Docker.
Build the image (`docker build -t orkexa-sandbox -f docker/sandbox.Dockerfile .`), or,
for code you trust, pass `sandbox=SubprocessSandbox(acknowledge_unsafe=True)`.
