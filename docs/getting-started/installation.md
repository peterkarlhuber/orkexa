# Installation

Orkexa needs Python 3.10 or newer. The core depends only on Pydantic; each provider
SDK is an extra:

```bash
pip install "orkexa[anthropic]"   # Anthropic Claude (anthropic SDK)
pip install "orkexa[openai]"      # OpenAI and OpenAI-compatible endpoints (openai SDK)
pip install "orkexa[gemini]"      # Google Gemini (google-genai SDK)
pip install "orkexa[all]"         # all three
pip install "orkexa[pdf]"         # PDF text extraction for the index (pypdf)
pip install "orkexa[vector]"      # vector/hybrid search (sqlite-vec)
```

`import orkexa` never imports a provider SDK. Using a provider whose SDK is missing
raises `MissingDependencyError` naming the extra, e.g. `pip install 'orkexa[openai]'`.

## API keys

| Provider | Variable read by the SDK | Explicit argument |
| --- | --- | --- |
| `anthropic` | `ANTHROPIC_API_KEY` | `api_key=` |
| `openai` | `OPENAI_API_KEY` | `api_key=` |
| `gemini` | `GEMINI_API_KEY` / `GOOGLE_API_KEY` | `api_key=` |
| `compatible`, `ollama`, `vllm` | **none** | `api_key=` (only if the server needs one) |

A custom `base_url` for `anthropic`, `openai` or `gemini` requires an explicit
`api_key=`: Orkexa never sends a provider's environment key to another host.
OpenAI's `OPENAI_BASE_URL` variable is ignored for the same reason.

## Existing claude-agent code

```bash
pip uninstall claude-agent
pip install orkexa-claude-compat     # provides `import claude_agent`, depends on orkexa[anthropic]
python -m orkexa.compat              # shows which distribution serves claude_agent
```

Both distributions contain the package `claude_agent`, so they must not be
installed together; the shim refuses to import if the original is present and
`python -m orkexa.compat` exits with status 1. Details: [Migration](../guides/migration.md).
