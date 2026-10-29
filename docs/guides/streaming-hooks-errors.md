# Streaming, hooks and errors

## Streaming

A request is streamed when `stream=True`, when `on_text`, `on_thinking` or `on_event`
is set, or (Anthropic, `stream=None`) when `max_tokens` exceeds 16,000. The committed
history of a streamed turn is identical to the non-streamed one (contract-tested for
every adapter).

## Hooks

| Hook | Receives |
| --- | --- |
| `on_text(text)` | streamed text deltas |
| `on_thinking(text)` | streamed reasoning deltas (Anthropic thinking, OpenAI reasoning summaries, Gemini thoughts) |
| `on_event(event)` | neutral events: `TextDelta`, `ReasoningDelta`, `ToolCallStarted`, `ToolCallDelta`, `ResponseCompleted`, `UsageReported`, `TurnCompleted`, `TurnFailed` |
| `on_request(args)` | the native request arguments |
| `on_response(response)` | the native SDK response |
| `before_tool(call)` | `None`/`True` runs the call, a string/`False` denies it (the reason goes back to the model) |
| `after_tool(call, content, is_error)` | the result sent back |

`AsyncAgent` accepts coroutine functions for every hook.

## Errors

| Situation | Exception |
| --- | --- |
| refusal (`stop_reason="refusal"`, Gemini safety block, OpenAI refusal) | `RefusalError` (or text with `on_refusal="return"`) |
| output token limit reached | `IncompleteResponseError` (usage is recorded) |
| structured output invalid | `StructuredOutputError` |
| tool rounds exhausted | `MaxToolRoundsError` |
| second `ask()` while one runs | `ConcurrentTurnError` |
| provider API errors (generic agents) | `AuthenticationError`, `RateLimitError`, `InvalidRequestError`, `ProviderConnectionError`, `ProviderServerError` (all `ProviderError`) |

`ProviderError` has `provider`, `status_code`, `request_id` and `retryable`; its message
is redacted (API keys never appear) and the SDK exception is the `__cause__`.
`ClaudeAgent` keeps raising the Anthropic SDK exceptions, as in 0.8.0. The
conversation is committed only after a complete turn; a failed turn leaves the
history unchanged.
