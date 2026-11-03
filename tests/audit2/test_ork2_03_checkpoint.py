"""ORK2-03: turn checkpoints bind call arguments and stored results; single-use resume is opt-in.

Before the fix a checkpoint could be resumed any number of times (each resume ran the approved call
again) and its fingerprint covered only the agent configuration, so a stored call's arguments or result
could be changed in storage and the hook saw the changed call as "resumed".
"""

from __future__ import annotations

import asyncio
import copy
import json
from typing import Any

import pytest

from orkexa import (
    Agent,
    AsyncAgent,
    CheckpointClaims,
    CheckpointError,
    Tool,
    ToolCall,
    ToolCallSuspended,
    TurnCheckpoint,
    TurnSuspended,
)
from orkexa.core.checkpoint import checkpoint_digest
from orkexa.providers.fake import FakeModel, text, tool_calls

KEY = b"0123456789abcdef-test-key"


def _agent(script: list[Any], transfers: list[int], cls: Any = Agent, **kwargs: Any) -> Any:
    @Tool
    def transfer(amount: int) -> str:
        """Transfer money.

        Args:
            amount: How much.

        Returns:
            ok.
        """
        transfers.append(amount)
        return "ok"

    @Tool
    def lookup(account: str) -> str:
        """Look up an account.

        Args:
            account: Which.

        Returns:
            The balance.
        """
        return "100"

    def gate(call: ToolCall) -> Any:
        if call.name == "transfer" and not call.resumed:
            raise ToolCallSuspended("needs approval", ref=f"approval-{call.id}")
        return None

    return cls(
        model="fake:m",
        client=FakeModel(script),
        tools=[transfer, lookup],
        before_tool=gate,
        name="payer",
        **kwargs,
    )


def _suspend(**kwargs: Any) -> dict[str, Any]:
    script = [tool_calls(("lookup", {"account": "a"}), ("transfer", {"amount": 100}))]
    with pytest.raises(TurnSuspended) as caught:
        _agent(script, [], **kwargs).ask("pay")
    return json.loads(json.dumps(caught.value.checkpoint.model_dump(mode="json")))  # type: ignore[no-any-return]


def _altered_arguments(data: dict[str, Any]) -> dict[str, Any]:
    changed = copy.deepcopy(data)
    changed["calls"][1]["arguments"] = {"amount": 1_000_000}
    for part in changed["history"][-1]["content"]:
        if part.get("type") == "tool_call" and part["name"] == "transfer":
            part["arguments"] = {"amount": 1_000_000}
    return changed


def test_changed_call_arguments_are_refused() -> None:
    data = _suspend()
    transfers: list[int] = []
    with pytest.raises(CheckpointError, match="changed"):
        _agent([text("done")], transfers).resume(_altered_arguments(data))
    assert transfers == []


def test_changed_stored_results_are_refused() -> None:
    data = _suspend()
    changed = copy.deepcopy(data)
    changed["results"][0]["content"] = "999999"
    with pytest.raises(CheckpointError, match="changed"):
        _agent([text("done")], []).resume(changed)


def test_calls_must_match_the_conversation() -> None:
    data = _suspend()
    changed = copy.deepcopy(data)
    changed["calls"][1]["arguments"] = {"amount": 5}  # only the copy outside the conversation
    with pytest.raises(CheckpointError, match="differ"):
        _agent([text("done")], []).resume(changed)


def test_without_a_key_a_recomputed_digest_is_accepted_and_with_a_key_it_is_not() -> None:
    # The plain digest catches accidental edits only; a key makes it an HMAC that storage cannot forge.
    forged = TurnCheckpoint.model_validate(_altered_arguments(_suspend()))
    forged = forged.model_copy(update={"digest": checkpoint_digest(forged)})
    transfers: list[int] = []
    _agent([text("done")], transfers).resume(forged)
    assert transfers == [1_000_000]

    signed = _suspend(checkpoint_key=KEY)
    assert signed["digest"].startswith("hmac-sha256:")
    forged = TurnCheckpoint.model_validate(_altered_arguments(signed))
    for digest in (checkpoint_digest(forged), checkpoint_digest(forged, b"another-key-0123456789"), None):
        with pytest.raises(CheckpointError):
            _agent([text("done")], [], checkpoint_key=KEY).resume(
                forged.model_copy(update={"digest": digest})
            )
    transfers = []
    assert _agent([text("done")], transfers, checkpoint_key=KEY).resume(signed) == "done"
    assert transfers == [100]
    with pytest.raises(CheckpointError, match="changed"):  # a signed checkpoint needs the key
        _agent([text("done")], []).resume(signed)
    with pytest.raises(ValueError, match="16 bytes"):
        _agent([], [], checkpoint_key=b"short")


def test_older_checkpoints_without_digest_still_resume_without_a_key() -> None:
    data = _suspend()
    legacy = {name: value for name, value in data.items() if name not in ("digest", "nonce")}
    transfers: list[int] = []
    assert _agent([text("done")], transfers).resume(legacy) == "done"
    assert transfers == [100]


def test_claims_make_resume_single_use() -> None:
    data = _suspend()
    claims = CheckpointClaims()
    transfers: list[int] = []
    assert _agent([text("done")], transfers).resume(data, claim=claims) == "done"
    assert TurnCheckpoint.model_validate(data).id in claims
    with pytest.raises(CheckpointError, match="already resumed"):
        _agent([text("done")], transfers).resume(data, claim=claims)  # a fresh agent, same claims
    assert transfers == [100]

    async def run() -> None:
        agent = _agent([text("x")], transfers, cls=AsyncAgent)
        with pytest.raises(CheckpointError, match="already resumed"):
            await agent.resume(data, claim=claims)
        assert await _agent([text("x")], transfers, cls=AsyncAgent).resume(_suspend(), claim=claims) == "x"

    asyncio.run(run())
    assert transfers == [100, 100]


def test_without_a_claim_a_checkpoint_can_be_resumed_again() -> None:
    # Nodestro resumes a stored checkpoint again after its own crash (its journal and gate decide).
    data = _suspend()
    transfers: list[int] = []
    _agent([text("done")], transfers).resume(data)
    _agent([text("done")], transfers).resume(data)
    assert transfers == [100, 100]


def test_a_turn_that_waits_again_gets_a_new_checkpoint_id() -> None:
    data = _suspend()

    def still_waiting(call: ToolCall) -> Any:
        if call.name == "transfer":
            raise ToolCallSuspended("needs approval", ref=f"approval-{call.id}")
        return None

    agent = _agent([], [])
    agent.before_tool = still_waiting
    claims = CheckpointClaims()
    with pytest.raises(TurnSuspended) as caught:
        agent.resume(data, claim=claims)
    again = caught.value.checkpoint
    assert again.id != TurnCheckpoint.model_validate(data).id
    assert claims(again.id)
