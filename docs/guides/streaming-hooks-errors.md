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

An exception raised by a hook ends the turn with that exception, and the history is
not committed. A request that was already answered is still recorded:
when `on_response` or the hooks of the final `ResponseCompleted` event raise, its usage
reaches `agent.usage` and the usage tracker (and counts against a budget). The
history is committed just before `TurnCompleted` is emitted, so an exception from
that last event leaves the completed turn in the history.

## Errors

| Situation | Exception |
| --- | --- |
| refusal (`stop_reason="refusal"`, Gemini safety block, OpenAI refusal) | `RefusalError` (or text with `on_refusal="return"`) |
| output token limit reached | `IncompleteResponseError` (usage is recorded) |
| structured output invalid | `StructuredOutputError` |
| tool rounds exhausted | `MaxToolRoundsError` |
| second `ask()`, `reset()` or `load_history()` while a turn runs | `ConcurrentTurnError` |
| provider API errors (generic agents) | `AuthenticationError`, `RateLimitError`, `InvalidRequestError`, `ProviderConnectionError`, `ProviderServerError` (all `ProviderError`) |

`ProviderError` has `provider`, `status_code`, `request_id` and `retryable`; its message
is redacted (API keys never appear) and names the SDK exception type. The SDK exception
itself is not attached: `__cause__` and `__context__` are `None`, so a traceback or
`logging.exception()` cannot print the request or its headers.
`ClaudeAgent` keeps raising the Anthropic SDK exceptions, as in 0.8.0. The
conversation is committed only after a complete turn; a failed turn leaves the
history unchanged.
