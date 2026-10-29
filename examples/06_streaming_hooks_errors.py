"""6. Streaming, hooks and events, refusals and provider-neutral error handling."""

from __future__ import annotations

from typing import Any

from _setup import model

from orkexa import (
    Agent,
    IncompleteResponseError,
    ProviderError,
    RateLimitError,
    RefusalError,
    TextDelta,
    Tool,
    ToolCall,
    TurnCompleted,
    UsageReported,
)


@Tool
def delete_everything(confirm: bool) -> str:
    """Delete all data (a dangerous tool the hook below refuses).

    Args:
        confirm: Must be true.

    Returns:
        A status message.
    """
    return "deleted"


def guard(call: ToolCall) -> str | None:
    return "Denied by policy: destructive tools need a human." if call.name == "delete_everything" else None


def on_event(event: Any) -> None:
    if isinstance(event, UsageReported):
        print(f"\n  [usage] {event.record.provider} {event.record.output_tokens} output tokens")
    elif isinstance(event, TurnCompleted):
        print("  [turn completed]")


def main() -> None:
    name = model()
    agent = Agent(
        model=name,
        tools=[delete_everything],
        before_tool=guard,
        on_text=lambda text: print(text, end="", flush=True),
        on_event=on_event,
    )
    agent.ask("Write a haiku about rivers, then try to delete everything.")

    deltas: list[str] = []
    counting = Agent(
        model=name, on_event=lambda e: deltas.append(e.text) if isinstance(e, TextDelta) else None
    )
    counting.ask("Count to five.")
    print(f"\n{len(deltas)} text deltas: {''.join(deltas)!r}")

    short = Agent(model=name, max_tokens=16, on_refusal="raise")
    try:
        short.ask("Write a long essay about the history of Europe.")
    except IncompleteResponseError as exc:
        print(f"truncated: {exc}")
    except RefusalError as exc:
        print(f"refused: {exc}")
    except RateLimitError as exc:
        print(f"rate limited (retryable={exc.retryable}): {exc}")
    except ProviderError as exc:  # every provider maps its SDK errors to this hierarchy
        print(f"{exc.provider} failed with status {exc.status_code}: {exc}")


if __name__ == "__main__":
    main()
