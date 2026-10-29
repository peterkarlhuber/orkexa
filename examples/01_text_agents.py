"""1. The same text agent on Anthropic, OpenAI and Gemini.

pip install "orkexa[all]"
export ANTHROPIC_API_KEY=... OPENAI_API_KEY=... GEMINI_API_KEY=...
python examples/01_text_agents.py
"""

from __future__ import annotations

from _setup import available_models

from orkexa import Agent


def main() -> None:
    for model in available_models():
        agent = Agent(model=model, system_prompt="Answer in one sentence.")
        print(f"{model}: {agent.ask('What is an AI agent?')}")
        print(f"  tokens: {agent.usage.total_tokens}, provider: {agent.provider}")


if __name__ == "__main__":
    main()
