# Long conversations

A conversation grows with every turn, but a model reads at most its context window (input and output together).
`orkexa.core.compaction` fits a conversation into a token budget before it goes to the model. It works on plain text
turns, so any application that keeps its own conversation history can use it.

```python
from orkexa.core.compaction import SUMMARY_INSTRUCTIONS, Turn, compact, context_budget

turns = [Turn(role, text, ref=number) for number, (role, text) in enumerate(history, start=1)]
turns.append(Turn("user", new_message, ref=len(turns) + 1))

budget = context_budget(limits.context_window, output_reserve=max_tokens, reserved=system_prompt_tokens)
result = compact(turns, budget=budget)  # trim (the default)
result = compact(turns, budget=budget, strategy="summarize", summarize=ask)  # summarize with a helper model
```

## The budget

`context_budget(window, output_reserve=..., reserved=..., margin=0.1)` is the context window minus the output
reserve (the request's `max_tokens`), less a 10 % margin for the estimate's error and the tokens of the system
prompt and tools (`reserved`). An unknown window gives `None`: the caller decides (e.g. keeps a fixed number of
turns with `max_turns`).

Tokens are estimated with `estimate_tokens` (one token per three bytes of UTF-8, which is generous for English and
fits most other scripts), so the result is deterministic. Pass `count=` to use a real tokenizer.

## Trim and summarize

`compact` keeps the newest turns that fit; the newest turn (the new message) always stays, even when it alone is
larger than the budget (`result.fits` is then false). The older turns are replaced by one turn of role `summary`
at the start:

| `result.method` | What the summary turn holds |
|---|---|
| `none` | Nothing was replaced; `result.turns` are the given turns. |
| `trimmed` | `[Summary of turns 1 to 12 (trimmed)]` and a note that they were left out. No model is asked. |
| `summarized` | `[Summary of turns 1 to 12 (summarized)]`, the helper model's summary and the kept untrusted data. |

The summary turn carries its source range as `first` and `ref` (the caller's numbers of the first and last turn it
replaces). Give the next call the summary turn first and the newer turns after it: the summary is summarized again
with the turns that no longer fit, and its range carries over (`first` stays 1).

With `strategy="summarize"` the helper is any callable `summarize(transcript, max_tokens) -> str` (`acompact`
takes an async one), typically one request to a small or local model with `SUMMARY_INSTRUCTIONS` as its system
prompt. It gets at most a quarter of the budget; a longer answer is cut. When the helper raises, the result is
`trimmed` and `result.error` names the exception type, so the conversation goes on.

## Untrusted data stays framed

Text that no person wrote (a web page, a tool or MCP result) reaches a prompt only inside an `<untrusted_data>`
frame (`orkexa.core.framing.frame_untrusted`). Compaction keeps that promise:

- The helper model never sees framed data. Each frame in the summarized turns becomes a placeholder (`[data 1]`) in
  its transcript, so instructions hidden in the data cannot steer the summary.
- The summary keeps an excerpt of each frame (at most `EXCERPT_TOKENS` each, shorter when many frames share the
  budget) inside a new frame with the original `source`, one frame per source: `result.sources` lists them.
- The helper's text has every spelling of the frame tag escaped (`&lt;`), so it can neither open a frame nor close
  one. A stray frame tag in a user's or assistant's text is escaped before it reaches the helper as well.

Framed data is never unwrapped into instructions, also not after several rounds of compaction.
