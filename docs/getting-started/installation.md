# Installation

Orkexa needs Python 3.10 or newer. The core depends only on Pydantic; each provider
SDK is an extra.

!!! note "Not yet on PyPI"
    Orkexa 0.1.0 is unreleased. Until it is published, install it from the repository:

    ```bash
    pip install "orkexa[all] @ git+https://github.com/peterkarlhuber/orkexa"
    # or, from a clone, for development:
    pip install -e ".[dev]"
    ```

    The commands below show the extras as they will be installed from PyPI.

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

Set the variable of each provider you use, e.g. in your shell profile or a secrets
manager (never in code or in the repository):

```bash
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
export GEMINI_API_KEY=...
```

| Provider | Variable read by the SDK | Explicit argument |
| --- | --- | --- |
| `anthropic` | `ANTHROPIC_API_KEY` | `api_key=` |
| `openai` | `OPENAI_API_KEY` | `api_key=` |
| `gemini` | `GEMINI_API_KEY` / `GOOGLE_API_KEY` | `api_key=` |
| `compatible`, `ollama`, `vllm` | **none** | `api_key=` (only if the server needs one) |

A custom `base_url` for `anthropic`, `openai` or `gemini` requires an explicit
`api_key=`: Orkexa never sends a provider's environment key to another host.
OpenAI's `OPENAI_BASE_URL` variable is ignored for the same reason.
