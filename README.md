<p align="center">
  <img src="assets/branding/orkexa-logo.png" alt="Orkexa" width="280">
</p>

<p align="center">
  <strong>One agent runtime for Anthropic Claude, OpenAI, Google Gemini and OpenAI-compatible endpoints.</strong>
</p>

<p align="center">
  <a href="https://github.com/peterkarlhuber/orkexa/actions/workflows/ci.yml"><img src="https://github.com/peterkarlhuber/orkexa/actions/workflows/ci.yml/badge.svg?branch=master" alt="CI"></a>
  <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-16243B" alt="Python 3.10 | 3.11 | 3.12 | 3.13">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-0079FE" alt="MIT License"></a>
</p>

Orkexa is a Python library for tool-using AI agents and multi-agent systems. You write
an agent once: tools, structured output, streaming, history, memory, retrieval,
evaluation, dynamic agents and cost tracking behave the same on every provider, and the
provider is an adapter you choose with a model string. Where a provider cannot do what
you asked for, Orkexa raises an error instead of silently doing something else.

> **Status:** alpha (0.1.0, unreleased). The API may still change. Orkexa is not yet
> published on PyPI; install it from source as shown below.

## Installation

```bash
pip install "orkexa[all] @ git+https://github.com/peterkarlhuber/orkexa"
```

The core needs only Pydantic. Each provider SDK is an extra: `orkexa[anthropic]`,
`orkexa[openai]` (also for compatible endpoints), `orkexa[gemini]`, or `orkexa[all]`;
`orkexa[pdf]` and `orkexa[vector]` add PDF extraction and vector search.
`import orkexa` never imports a provider SDK.

## Quickstart

Set the key of every provider you want to use. The SDKs read these variables; Orkexa
never sends one provider's key to another provider or host.

```bash
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
export GEMINI_API_KEY=...       # or GOOGLE_API_KEY
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


for model in ("anthropic:claude-sonnet-5-5", "openai:gpt-6.1-sol", "gemini:gemini-3.8-flash"):
    print(model, Agent(model=model, tools=[add]).ask("What is 17 + 25?"))

# A local OpenAI-compatible server: no key is read from the environment.
local = Agent(provider="ollama", model="llama3.1:8b", capabilities={"tools": "supported"}, tools=[add])
```

The model ids are examples; use any model your account can access. Every call above is
a paid API request; the test suite and the [offline examples](examples/README.md) need
no key.

## Provider support

| Provider | Kind | API | Install | Credentials |
|----------|------|-----|---------|-------------|
| `anthropic` | native | Messages API (`anthropic` SDK) | `orkexa[anthropic]` | `ANTHROPIC_API_KEY` or `api_key=` |
| `openai` | native | Responses API, stateless (`openai` SDK) | `orkexa[openai]` | `OPENAI_API_KEY` or `api_key=` |
| `gemini` | native | Gemini API (`google-genai` SDK) | `orkexa[gemini]` | `GEMINI_API_KEY` / `GOOGLE_API_KEY` or `api_key=` |
| `compatible` | compatible | Chat Completions or Responses at any `base_url` | `orkexa[openai]` | explicit `api_key=` only |
| `ollama`, `vllm` | compatible (presets) | as `compatible`, with local default URLs | `orkexa[openai]` | explicit `api_key=` only |

Native adapters know their provider's features and report them as capabilities.
Compatible endpoints differ from server to server, so every capability is `unknown`
until you declare it (`capabilities={"tools": "supported"}`). The full matrix is in
[Providers and capabilities](docs/getting-started/providers.md).

## Features

* **One runtime** (`Agent`, `AsyncAgent`): multi-round tool loop with parallel tools,
  timeouts and hooks; history is committed only after a complete turn.
* **Explicit capabilities**: unsupported features raise `UnsupportedCapabilityError`,
  unknown ones `UnknownCapabilityError` naming the override.
* **Strict tools and structured output** from type hints and Pydantic models, validated
  locally on every provider.
* **Provider-neutral history** with lossless native metadata, versioned history files
  and `history_policy="convert"` for switching providers mid-conversation.
* **Memory, files and RAG**: native memory tool on Anthropic, function tools elsewhere;
  `retrieval_sources()` lists the passages the search tool actually returned.
* **Usage and budgets** across providers; an unknown price is `None`, never zero.
* **Multi-agent and dynamic agents**: mixed-provider delegation, judges from another
  provider, generated tools with static checks, sandboxes and approval.
* **Typed errors** (`RateLimitError`, `RefusalError`, `IncompleteResponseError`, ...)
  with secrets redacted from messages.
* **claude-agent compatible**: `ClaudeAgent`/`AsyncClaudeAgent` keep the 0.8.0 API, and
  unchanged `import claude_agent` code runs with `orkexa-claude-compat`.

## Documentation

| | |
|---|---|
| Getting started | [Installation](docs/getting-started/installation.md) · [Quickstart](docs/getting-started/quickstart.md) · [Providers and capabilities](docs/getting-started/providers.md) |
| Guides | [Tools and structured output](docs/guides/tools-and-structured-output.md) · [Streaming, hooks and errors](docs/guides/streaming-hooks-errors.md) · [History](docs/guides/history.md) · [Memory, files and RAG](docs/guides/memory-files-rag.md) · [Usage and budgets](docs/guides/usage-and-budgets.md) · [Multi-agent](docs/guides/multi-agent.md) |
| Operations | [Security](docs/guides/security.md) · [Data protection and AI regulation](docs/compliance/index.md) · [Troubleshooting and FAQ](docs/troubleshooting.md) |
| Migration | [From claude-agent 0.8.0](docs/guides/migration.md) · [Feature matrix](docs/feature-matrix.md) |
| Internals | [Architecture and ADRs](docs/architecture/target-architecture.md) · [Changelog](CHANGELOG.md) |

The documentation is a MkDocs site with an API reference generated from the
docstrings: `pip install -e ".[docs]" && mkdocs serve`.

## Security and compliance

Orkexa never lets a provider SDK execute tools on its own, keeps credentials per
provider, redacts secrets from errors and tracebacks, stores histories, memory and
registries owner-private (`0600`) and runs generated code in `DockerSandbox` by
default, failing closed without Docker. `SubprocessSandbox` is **not** a security
boundary and, like `AutoApprover`, an explicit unsafe opt-in; see the
[security guide](docs/guides/security.md). Report
vulnerabilities as described in [SECURITY.md](SECURITY.md).

Orkexa is a library, not a certified product. It provides technical controls that help
integrators meet obligations under the GDPR, the EU AI Act or ISO/IEC 42001; whether a
system built with it complies depends on that system. See
[Data protection and AI regulation](docs/compliance/index.md).

## Development

```bash
pip install -e ".[dev]"
pip install --no-deps -e compat/orkexa-claude-compat
ruff check . && ruff format --check .
mypy --strict src && mypy --strict examples
pytest --cov=orkexa            # offline: no API keys, no network
pytest -m packaging tests/packaging --noconftest   # builds and installs into fresh venvs
mkdocs build --strict          # needs .[docs]
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow.

## Project

* **License:** [MIT](LICENSE).
* **Origin:** Orkexa is the provider-independent successor of
  [claude-agent](https://github.com/peterkarlhuber/claude-agent) 0.8.0 by the same
  author ([NOTICE](NOTICE)); its API lives on as `ClaudeAgent` and in
  [`orkexa-claude-compat`](compat/orkexa-claude-compat/README.md).
* **Independence:** Orkexa is an independent open-source project. It is not affiliated
  with, endorsed by or sponsored by Anthropic, OpenAI or Google; their names identify
  the APIs Orkexa connects to.
* **Support and contributing:** [SUPPORT.md](SUPPORT.md) ·
  [CONTRIBUTING.md](CONTRIBUTING.md) · [Code of Conduct](CODE_OF_CONDUCT.md)
