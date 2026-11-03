"""Turn checkpoints: a turn that stopped to wait for a decision (e.g. a person's approval).

A turn can stop at two points (``TurnCheckpoint.stage``):

* ``"tools"``: a ``before_tool`` hook raises :class:`~orkexa.errors.ToolCallSuspended`
  instead of deciding. ``agent.resume(checkpoint)`` continues exactly there: the
  waiting calls go through ``before_tool`` again (now with ``ToolCall.resumed``), the
  calls that already ran are not run again, and the model is called next only with
  all results of that round.
* ``"request"``: the usage tracker raises :class:`~orkexa.errors.RequestSuspended`
  while admitting the next model request (``check_request``, ``before_request`` or
  ``try_reserve``), e.g. because the request does not fit a budget. The request is
  not sent, recorded or reserved. ``agent.resume(checkpoint)`` admits the same request
  again (the tracker decides anew) and continues the turn; earlier requests of the
  turn are not sent again. If the tracker suspends again, a new checkpoint results.

In both cases the turn ends with :class:`~orkexa.errors.TurnSuspended`, whose
:class:`TurnCheckpoint` is plain JSON, and nothing is committed to the agent's
conversation until the turn completes. Native batch requests cannot be suspended.

The checkpoint is bound to the agent: its name, model, system prompt, tool definitions,
output format and tool round limit (``fingerprint``). A different agent, or the same
agent with changed tools or prompt, cannot resume it (:class:`~orkexa.errors.CheckpointError`).
Storing it (and deleting it with the run that owns it) is the application's task.

**Integrity.** ``digest`` covers the whole checkpoint: the conversation, every call with
its arguments, the stored results and the waiting calls. A checkpoint changed after it
was issued is refused. Without a key (``checkpoint_key=`` of the agent) the digest is a
plain SHA-256 and only catches accidental or inconsistent edits: whoever can write the
checkpoint store can recompute it. With a key it is an HMAC, and a checkpoint without a
valid HMAC is refused, so approved call arguments cannot be altered in storage.

**Resuming is not single-use by itself.** Every :meth:`~orkexa.Agent.resume` of the same
checkpoint runs its waiting calls again (``ToolCall.resumed`` is ``True`` each time). An
application that must run an approved side-effecting call at most once either consumes
the checkpoint atomically before it resumes (e.g. deletes or marks it in the same
transaction that reads it) or passes ``claim=`` (e.g. a shared :class:`CheckpointClaims`
or its own check against a database, keyed by :attr:`TurnCheckpoint.id`). An application
that deliberately resumes a checkpoint again after its own crash (and makes its tools
idempotent or asks again, as Nodestro does) passes no claim.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .types import Effort, ToolCallPart, ToolResultPart

__all__ = [
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_VERSION",
    "CheckpointClaims",
    "TurnCheckpoint",
    "WaitingCall",
    "checkpoint_digest",
]

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
        round: ``"tools"``: the tool round whose calls wait. ``"request"``: the round
            whose model request was about to be sent.
        history: The turn's conversation: up to the assistant message with the calls
            (``"tools"``), or exactly what the suspended request would have sent
            (``"request"``; it may end with a user, system or assistant message).
        calls: All tool calls of that assistant message, in order (empty for ``"request"``).
        results: The results of the calls that already ran (empty for ``"request"``).
        waiting: The calls that wait (empty for ``"request"``).
        effort: The turn's effort override.
        max_tokens: The turn's ``max_tokens`` override.
        stage: Where the turn stopped: ``"tools"`` (a tool call waits; the default, so
            checkpoints written before this field existed parse unchanged) or
            ``"request"`` (the next model request waits).
        reason: ``"request"``: the reason from :class:`~orkexa.errors.RequestSuspended`.
        ref: ``"request"``: the application's reference from
            :class:`~orkexa.errors.RequestSuspended`.
        nonce: Random per issued checkpoint, so that a turn that waits again at the same
            point gets a checkpoint with a new :attr:`id` (``None`` in older checkpoints).
        digest: :func:`checkpoint_digest` of everything above (``None`` in older
            checkpoints; accepted only by an agent without ``checkpoint_key``).
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
    calls: list[ToolCallPart] = Field(default_factory=list)
    results: list[ToolResultPart] = Field(default_factory=list)
    waiting: list[WaitingCall] = Field(default_factory=list)
    effort: Effort | None = None
    max_tokens: int | None = None
    stage: Literal["tools", "request"] = "tools"
    reason: str = ""
    ref: str | None = None
    nonce: str | None = None
    digest: str | None = None

    @property
    def id(self) -> str:
        """A stable identity of this issued checkpoint (SHA-256 of its content without ``digest``).

        Use it to record that a checkpoint was resumed (``claim=`` of ``resume()``)."""
        return hashlib.sha256(_canonical(self)).hexdigest()

    @model_validator(mode="after")
    def _check_stage(self) -> TurnCheckpoint:
        if self.stage == "tools":
            if not self.calls or not self.waiting:
                raise ValueError("a 'tools' checkpoint needs at least one call and one waiting call")
            if self.reason or self.ref is not None:
                raise ValueError("reason and ref belong to the waiting calls of a 'tools' checkpoint")
        elif self.calls or self.results or self.waiting:
            raise ValueError("a 'request' checkpoint has no calls, results or waiting calls")
        return self


def _canonical(checkpoint: TurnCheckpoint) -> bytes:
    content = checkpoint.model_dump(mode="json", exclude={"digest"})
    return json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def checkpoint_digest(checkpoint: TurnCheckpoint, key: bytes | None = None) -> str:
    """The digest of ``checkpoint`` without its ``digest`` field: ``"hmac-sha256:..."`` with
    ``key``, else ``"sha256:..."``."""
    content = _canonical(checkpoint)
    if key is not None:
        return "hmac-sha256:" + hmac.new(key, content, hashlib.sha256).hexdigest()
    return "sha256:" + hashlib.sha256(content).hexdigest()


class CheckpointClaims:
    """A thread-safe, process-local record of resumed checkpoints, for ``resume(..., claim=claims)``.

    The first claim of a :attr:`TurnCheckpoint.id` succeeds, every later one fails, so a
    checkpoint is resumed at most once by the agents that share this object. It lives in
    memory only: across processes or restarts, consume the checkpoint in the application's
    own store instead (or pass a ``claim`` callable backed by it).
    """

    def __init__(self) -> None:
        self._claimed: set[str] = set()
        self._lock = threading.Lock()

    def __call__(self, checkpoint_id: str) -> bool:
        """Claim ``checkpoint_id``; ``False`` if it was claimed before."""
        with self._lock:
            if checkpoint_id in self._claimed:
                return False
            self._claimed.add(checkpoint_id)
            return True

    def __contains__(self, checkpoint_id: object) -> bool:
        with self._lock:
            return checkpoint_id in self._claimed
