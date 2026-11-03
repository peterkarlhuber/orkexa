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

### Pausing a turn at a tool call

A `before_tool` hook can also raise `ToolCallSuspended(reason, ref=...)` when a call
needs a decision that is not available yet, for example a person's approval. The call
does not run. Once every other call of that round has finished, `ask()` raises
`TurnSuspended`. Its `checkpoint` (a `TurnCheckpoint`, plain JSON) holds the turn's
conversation, the results of the calls that already ran and the waiting calls; the
agent's committed conversation is unchanged.

```python
try:
    answer = agent.ask("Pay the invoice")
except TurnSuspended as paused:
    store(paused.checkpoint.model_dump(mode="json"))

# later, possibly in another process, with an agent configured the same way:
answer = agent.resume(load())
```

`resume()` passes only the waiting calls through `before_tool` again, with
`ToolCall.resumed=True`: the hook now decides (allow or deny) and must not ask again.
The model is called next with all results of the round, so it is never asked twice for
the same calls, and usage keeps the turn's `turn_id`. A checkpoint is bound to the
agent's name, model, system prompt, tool definitions, output format and tool round
limit; anything else raises `CheckpointError`. Where the checkpoint is stored, how long
it is valid and who may decide is the application's responsibility.

!!! warning "Resuming is not single-use"
    Every `resume()` of the same checkpoint runs its waiting calls again, each time with
    `ToolCall.resumed=True`. A double approval, a retried job or a crash between `resume()`
    and deleting the checkpoint can therefore run an approved side-effecting call twice.
    Either consume the checkpoint atomically in your store before you resume it, or pass
    `claim=`: a callable that records `checkpoint.id` and returns `False` if it was
    recorded before (`CheckpointClaims()` does this in memory, for one process). A claimed
    checkpoint is refused with `CheckpointError` before anything runs. Applications that
    deliberately resume again after their own crash (with idempotent tools or a journal
    that asks again) pass no claim.

    ```python
    claims = CheckpointClaims()  # or your own database-backed check
    answer = agent.resume(load(), claim=claims)
    ```

A checkpoint carries a `digest` over its conversation, every call with its arguments, the
stored results and the waiting calls; a checkpoint changed after it was issued is refused.
Without a key the digest is a plain SHA-256, which detects accidental or inconsistent
edits but can be recomputed by anyone who can write the checkpoint store. Give the agent
`checkpoint_key=` (a secret of at least 16 bytes, the same in every process that resumes)
to sign checkpoints with an HMAC: unsigned or altered checkpoints are then refused, so the
arguments of an approved call cannot be changed in storage. Bind approvals to the call
they were given for (`call.id`, `call.name` and `call.input` in `before_tool`), not only
to the checkpoint.

### Pausing before a request

A usage tracker can pause a turn *before* a model request in the same way: when
`check_request()`, `before_request()` or `try_reserve()` raises
`RequestSuspended(reason, ref=...)`, the request is not sent, recorded or reserved,
and `ask()` raises `TurnSuspended`. Its checkpoint has `stage="request"`: `round` is
the round whose request was about to be sent, `history` is exactly the conversation
that request would have sent (it may end with the prompt, a system message, tool
results or a paused assistant message), `calls`, `results` and `waiting` are empty,
and `reason` and `ref` come from the exception. The committed conversation is
unchanged. Tool-stage checkpoints have `stage="tools"` (the default, so checkpoints
written before the field existed still resume).

A typical use is a budget that a person may raise. A `BudgetTracker` subclass turns
the refusal into a pause, and `BudgetTracker.set_limit()` raises the limit later:

```python
from orkexa import RequestSuspended, TurnSuspended
from orkexa.dynamic import BudgetExceededError, BudgetTracker


class PausingBudget(BudgetTracker):
    def try_reserve(self, *, agent, provider, model, amount):
        try:
            return super().try_reserve(agent=agent, provider=provider, model=model, amount=amount)
        except BudgetExceededError as exc:
            # exc.reason, exc.limit_usd, exc.spent_usd, exc.requested_usd, exc.tracker
            raise RequestSuspended(f"needs up to ${exc.requested_usd}", ref=ask_person(exc)) from exc


budget = PausingBudget("1.00", on_oversized="reject")
agent = Agent(model="anthropic:claude-sonnet-5-5", usage_tracker=budget)
try:
    answer = agent.ask("Write the report")
except TurnSuspended as paused:
    store(paused.checkpoint.model_dump(mode="json"))

# later, once the person agreed:
budget.set_limit("5.00")
answer = agent.resume(load())
```

`resume()` builds the suspended request from the checkpoint's conversation and admits
it again: the tracker decides anew, and if it suspends again, a new request-stage
checkpoint results. Requests the turn already sent are not sent again, and usage keeps
the turn's `turn_id`. The same agent binding (`CheckpointError`) applies. Native batch
requests (`Evaluator.evaluate_many(batch=True)`) cannot be suspended: there,
`RequestSuspended` propagates to the caller.

## Errors

| Situation | Exception |
| --- | --- |
| refusal (`stop_reason="refusal"`, Gemini safety block, OpenAI refusal) | `RefusalError` (or text with `on_refusal="return"`) |
| output token limit reached | `IncompleteResponseError` (usage is recorded) |
| structured output invalid | `StructuredOutputError` |
| tool rounds exhausted | `MaxToolRoundsError` |
| second `ask()`, `reset()` or `load_history()` while a turn runs | `ConcurrentTurnError` |
| a tool call waits (`ToolCallSuspended`) or the usage tracker pauses a request (`RequestSuspended`) | `TurnSuspended` (with a `checkpoint`) |
| provider API errors (generic agents) | `AuthenticationError`, `RateLimitError`, `InvalidRequestError`, `ProviderConnectionError`, `ProviderServerError` (all `ProviderError`) |

`ProviderError` has `provider`, `status_code`, `request_id` and `retryable`; its message
is redacted (API keys never appear) and names the SDK exception type. The SDK exception
itself is not attached: `__cause__` and `__context__` are `None`, so a traceback or
`logging.exception()` cannot print the request or its headers.
`ClaudeAgent` raises the Anthropic SDK exceptions unchanged. The
conversation is committed only after a complete turn; a failed turn leaves the
history unchanged.
