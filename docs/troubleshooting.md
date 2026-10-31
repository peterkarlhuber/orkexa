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
| `UnpricedModelError` (`orkexa.dynamic`) | `BudgetTracker(on_unpriced="reject")` and no price for the model | `register_pricing()` or `pricing=` on the agent |

Provider errors are redacted: messages never contain API keys. The SDK exception is not
attached (`exc.__cause__` is `None`, so tracebacks cannot print request headers); use
`exc.provider`, `exc.status_code`, `exc.request_id` and `exc.retryable`, and quote the
`request_id` when you contact the provider.

## FAQ

**Which provider SDK versions are supported?** The ranges in `pyproject.toml`
(`anthropic>=2.0,<3`, `openai>=3.0,<4`, `google-genai>=2.0,<3`). CI runs the suite with
the newest versions and with the lowest supported ones.

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

**How do I run the tests without API keys?** `pytest` runs offline by default; tests
that call real APIs are marked `live` and skip themselves without keys.

**Is `SubprocessSandbox` safe for untrusted code?** No. It is not a security boundary;
use `DockerSandbox` (the default of `ToolRegistry`) or stronger isolation
([Security](guides/security.md)).

**Why does my generated tool fail with "Cannot start docker"?** `ToolRegistry` runs
generated code in `DockerSandbox` by default and refuses to run it without Docker.
Build the image (`docker build -t orkexa-sandbox -f docker/sandbox.Dockerfile .`), or,
for code you trust, pass `sandbox=SubprocessSandbox(acknowledge_unsafe=True)`.
