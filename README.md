# Orkexa

**One agent runtime for Anthropic Claude, OpenAI, Google Gemini and OpenAI-compatible
endpoints.** Strict tools, structured output, streaming, memory, RAG with verified
sources, evaluation, dynamic agents, multi-agent orchestration and cost tracking work
the same on every provider; providers are adapters.

```bash
pip install "orkexa[all]"        # or orkexa[anthropic] / orkexa[openai] / orkexa[gemini]
```

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
    print(model, Agent(model=model, tools=[add]).ask("What is 17 + 25?"))

local = Agent(provider="ollama", model="llama3.1:8b", capabilities={"tools": "supported"}, tools=[add])
```

## Highlights

* **One runtime** (`Agent`, `AsyncAgent`): tool loop with parallel tools, timeouts and
  hooks; history committed only after a complete turn; usage recorded per request.
* **Native adapters**: Anthropic Messages API, OpenAI Responses API (stateless,
  encrypted reasoning replay), Gemini via `google-genai` (thought signatures, automatic
  function calling disabled), OpenAI-compatible Chat Completions or Responses (Ollama,
  vLLM, gateways).
* **Capabilities instead of silent degradation**: unsupported features raise
  `UnsupportedCapabilityError`, unknown ones `UnknownCapabilityError` with the override.
* **Provider-neutral history** with lossless native metadata; versioned history files;
  `history_policy="convert"` for switching providers mid-conversation.
* **Memory, files, RAG**: native memory tool on Anthropic, function tools elsewhere;
  `retrieval_sources()` lists the passages the search tool really delivered.
* **Usage and budgets** across providers; unknown cost is never silently zero;
  `BudgetTracker(on_unpriced="ignore" | "warn" | "reject")`.
* **Multi-agent**: mixed-provider delegation, judges of another provider, dynamic tool
  and agent creation with static checks, sandboxes, approval and per-provider
  credentials.
* **claude-agent compatible**: `ClaudeAgent`/`AsyncClaudeAgent` keep the 0.8.0 API;
  unchanged `import claude_agent` code runs with `pip install orkexa-claude-compat`.

## Documentation

* [Installation](docs/getting-started/installation.md) ·
  [Quickstart](docs/getting-started/quickstart.md) ·
  [Providers and capabilities](docs/getting-started/providers.md)
* [Migrating from claude-agent](docs/guides/migration.md) ·
  [Feature matrix](docs/feature-matrix.md) · [Security](docs/guides/security.md)
* Guides: [tools](docs/guides/tools-and-structured-output.md),
  [streaming and errors](docs/guides/streaming-hooks-errors.md),
  [history](docs/guides/history.md), [memory, files, RAG](docs/guides/memory-files-rag.md),
  [usage and budgets](docs/guides/usage-and-budgets.md),
  [multi-agent](docs/guides/multi-agent.md)
* [Examples](examples/README.md) (12 scripts, all run offline in the test suite)
* [Architecture and ADRs](docs/architecture/target-architecture.md)
* API reference: `mkdocs serve` (needs `pip install -e ".[docs]"`)

## Development

```bash
pip install -e ".[dev]"
pip install --no-deps -e compat/orkexa-claude-compat
ruff check . && ruff format --check .
mypy --strict src && mypy --strict examples
pytest                      # offline: no API keys, no network
pytest -m packaging         # builds wheels, installs them into fresh virtual environments
```

## Origin and license

Orkexa is the successor of [claude-agent](https://github.com/peterkarlhuber/claude-agent)
0.8.0 by the same author (see [NOTICE](NOTICE)). MIT License.
