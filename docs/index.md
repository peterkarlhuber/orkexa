# Orkexa

Orkexa is a Python framework for AI agents and multi-agent systems with **one**
agent runtime for Anthropic Claude, OpenAI (Responses API), Google Gemini and
OpenAI-compatible endpoints (Ollama, vLLM, gateways). Providers are adapters; tools,
history, structured output, streaming, memory, RAG, evaluation, dynamic agents and
cost tracking are shared.

```python
from orkexa import Agent, Tool


@Tool
def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: First summand.
        b: Second summand.

    Returns:
        The sum.
    """
    return a + b


for model in ("anthropic:claude-sonnet-5-5", "openai:gpt-5", "gemini:gemini-2.5-flash"):
    print(Agent(model=model, tools=[add]).ask("What is 17 + 25?"))
```

Orkexa grew out of [`claude-agent`](https://github.com/peterkarlhuber/claude-agent)
0.8.0. Its API lives on unchanged as `ClaudeAgent`/`AsyncClaudeAgent`, and unchanged
`import claude_agent` code runs through the separate distribution
`orkexa-claude-compat` (see [Migrating from claude-agent](guides/migration.md)).

* [Installation](getting-started/installation.md) and [Quickstart](getting-started/quickstart.md)
* [Providers and capabilities](getting-started/providers.md): what works where
* [Feature matrix](feature-matrix.md): every claude-agent feature and its status
* [Security](guides/security.md): defaults and remaining risks
