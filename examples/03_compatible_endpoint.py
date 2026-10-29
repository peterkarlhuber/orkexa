"""3. An OpenAI-compatible endpoint with configurable URL and protocol.

    ollama pull llama3.1:8b && ollama serve
    pip install "orkexa[openai]"
    python examples/03_compatible_endpoint.py

    # vLLM or a gateway, Responses API:
    ORKEXA_BASE_URL=http://localhost:8000/v1 ORKEXA_COMPATIBLE_API=responses \\
        ORKEXA_COMPATIBLE_MODEL=Qwen/Qwen3-8B python examples/03_compatible_endpoint.py

``OPENAI_API_KEY`` is never sent to such an endpoint; pass ``api_key=`` (here
``ORKEXA_COMPATIBLE_API_KEY``) if the server needs one. Capabilities of a
compatible endpoint are unknown until declared: tool calling depends on the model
and the server configuration, so this example declares it explicitly.
"""

from __future__ import annotations

import os
from typing import Literal, cast

from orkexa import Agent, CompatibleOptions, Tool


@Tool
def word_count(text: str) -> int:
    """Count the words of a text.

    Args:
        text: The text.

    Returns:
        The number of whitespace-separated words.
    """
    return len(text.split())


def main() -> None:
    api = cast(
        Literal["chat_completions", "responses"], os.environ.get("ORKEXA_COMPATIBLE_API", "chat_completions")
    )
    agent = Agent(
        provider="compatible",
        model=os.environ.get("ORKEXA_COMPATIBLE_MODEL", "llama3.1:8b"),
        base_url=os.environ.get("ORKEXA_BASE_URL", "http://localhost:11434/v1"),
        api_key=os.environ.get("ORKEXA_COMPATIBLE_API_KEY"),
        options=CompatibleOptions(api=api),
        capabilities={"tools": "supported", "streaming": "supported", "system_messages": "supported"},
        tools=[word_count],
    )
    print(agent.capabilities.tools, agent.capabilities.structured_output)  # supported, unknown
    print(agent.ask("How many words are in 'the quick brown fox'? Use the tool."))


if __name__ == "__main__":
    main()
