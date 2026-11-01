"""Turn checkpoints: a turn that stopped at a tool call waiting for a decision (e.g. a person's approval).

A ``before_tool`` hook raises :class:`~orkexa.errors.ToolCallSuspended` instead of
deciding. The turn then ends with :class:`~orkexa.errors.TurnSuspended`, whose
:class:`TurnCheckpoint` is plain JSON. ``agent.resume(checkpoint)`` continues exactly
there: the waiting calls go through ``before_tool`` again (now with
``ToolCall.resumed``), the calls that already ran are not run again, and the model is
called next only with all results of that round. Nothing is committed to the agent's
conversation until the turn completes.

The checkpoint is bound to the agent: its name, model, system prompt, tool definitions,
output format and tool round limit (``fingerprint``). A different agent, or the same
agent with changed tools or prompt, cannot resume it (:class:`~orkexa.errors.CheckpointError`).
Storing it (and deleting it with the run that owns it) is the application's task.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .types import Effort, ToolCallPart, ToolResultPart

__all__ = ["CHECKPOINT_FORMAT", "CHECKPOINT_VERSION", "TurnCheckpoint", "WaitingCall"]

CHECKPOINT_FORMAT = "orkexa.turn-checkpoint"
CHECKPOINT_VERSION = 1


class WaitingCall(BaseModel):
    """A requested tool call that has not run because it waits."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    call_id: str
    name: str
    reason: str = ""
    #: The application's reference from :class:`~orkexa.errors.ToolCallSuspended`.
    ref: str | None = None


class TurnCheckpoint(BaseModel):
    """Everything needed to continue a suspended turn (JSON: ``model_dump(mode="json")``).

    Attributes:
        format: Always :data:`CHECKPOINT_FORMAT`.
        version: :data:`CHECKPOINT_VERSION`; another version cannot be resumed.
        agent: The agent's name.
        model: The model id the turn used.
        fingerprint: Digest of what the turn depends on besides the conversation.
        turn_id: The id usage records of this turn carry; the resumed turn keeps it.
        round: The tool round whose calls wait.
        history: The turn's conversation up to the assistant message with the calls.
        calls: All tool calls of that assistant message, in order.
        results: The results of the calls that already ran.
        waiting: The calls that wait.
        effort: The turn's effort override.
        max_tokens: The turn's ``max_tokens`` override.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    format: Literal["orkexa.turn-checkpoint"] = "orkexa.turn-checkpoint"
    version: int = CHECKPOINT_VERSION
    agent: str
    model: str
    fingerprint: str
    turn_id: str | None = None
    round: int = Field(ge=0)
    history: list[dict[str, Any]]
    calls: list[ToolCallPart] = Field(min_length=1)
    results: list[ToolResultPart] = Field(default_factory=list)
    waiting: list[WaitingCall] = Field(min_length=1)
    effort: Effort | None = None
    max_tokens: int | None = None
