"""7. Conversation save/load and migration of Messages API histories between providers.

Runs offline: it writes a plain Messages API history file (the format
``ClaudeAgent.save_history`` writes), loads it into a provider-neutral agent,
shows what the history-conversion policy does and writes the versioned Orkexa
history format. If ``ORKEXA_MODEL`` (or a provider key) is set, it continues the
conversation with that provider.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import warnings

from _setup import available_models

from orkexa import Agent, ClaudeAgent, HistoryConversionError, HistoryConversionWarning
from orkexa.core.conversion import prepare_history

MESSAGES_API_HISTORY = [
    {"role": "user", "content": "What is 17 * 23?"},
    {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "Use the tool.", "signature": "EqQBCkYIBxgC..."},
            {"type": "tool_use", "id": "toolu_01", "name": "multiply", "input": {"a": 17, "b": 23}},
        ],
    },
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_01", "content": "391"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "17 * 23 = 391."}]},
]


def main() -> None:
    directory = Path(tempfile.mkdtemp(prefix="orkexa-history-"))
    plain = directory / "messages_api_history.json"
    plain.write_text(json.dumps(MESSAGES_API_HISTORY), encoding="utf-8")

    # 1. A ClaudeAgent reads the file unchanged (its own format, same model family).
    claude = ClaudeAgent(api_key="not-needed-offline")
    claude.load_history(plain)
    print("ClaudeAgent messages:", len(claude.messages))

    # 2. A neutral agent reads it too; the assistant turns are marked as Anthropic output.
    agent = Agent(model="openai:gpt-6.1-sol", api_key="not-needed-offline", history_policy="convert")
    agent.load_history(plain)
    for message in agent.conversation():
        print(f"  {message.role:<9} provider={message.provider} parts={[p.type for p in message.parts()]}")

    # 3. The policy decides what happens to Anthropic-only parts (the signed thinking block).
    try:
        prepare_history(agent.conversation(), "openai", "strict")
    except HistoryConversionError as exc:
        print("strict:", exc)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", HistoryConversionWarning)
        converted = prepare_history(agent.conversation(), "openai", "convert")
    print("convert:", [p.type for m in converted for p in m.parts()], f"({len(caught)} warning)")

    # 4. The versioned Orkexa history format (readable by every agent class).
    target = directory / "orkexa_history.json"
    agent.save_history(target)
    document = json.loads(target.read_text(encoding="utf-8"))
    print("saved:", document["format"], "version", document["version"], "->", target)

    # 5. Continue with a real provider, if one is configured.
    model = os.environ.get("ORKEXA_MODEL") or next(iter(available_models()), None)
    if model is None:
        print("Set a provider key or ORKEXA_MODEL to continue the conversation live.")
        return
    live = Agent(model=model, history_policy="convert")
    live.load_history(target)
    print(model, "->", live.ask("And what is that result plus 9?"))


if __name__ == "__main__":
    main()
