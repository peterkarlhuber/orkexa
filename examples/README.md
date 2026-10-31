# Orkexa examples

Every example is a standalone script that you run from the repository root, e.g.
`python examples/01_text_agents.py`. They are ordered from first steps to advanced
setups.

* **Live** examples call real model APIs: they need API keys and **cost money**.
  Providers without a key are skipped.
* **Local** examples talk to a server you run yourself (Ollama, vLLM, a gateway).
* **Offline** examples need neither.

Install the extras first: `pip install -e ".[all]"` from a clone (or
`pip install "orkexa[all] @ git+https://github.com/peterkarlhuber/orkexa"`).

## Beginner

| # | Script | Mode | What it shows | What you see |
|---|--------|------|---------------|--------------|
| 1 | `01_text_agents.py` | Live | The same text agent on Anthropic, OpenAI and Gemini | One short answer per provider with a key |
| 2 | `02_same_tool_three_providers.py` | Live | One `@Tool` rendered for three native providers; Orkexa runs the tool | Each provider's answer, computed with the tool |
| 5 | `05_structured_output.py` | Live (one key) | A Pydantic model as `output_format`, validated locally | The extracted object as JSON |
| 7 | `07_history_migration.py` | Offline, optionally live | Saving and loading histories, moving a Messages API history to another provider, the conversion policy | Message counts, the strict policy's error, the converted parts and the saved file; with a key, one more live turn |

## Intermediate

| # | Script | Mode | What it shows | What you see |
|---|--------|------|---------------|--------------|
| 3 | `03_compatible_endpoint.py` | Local | An OpenAI-compatible endpoint, Chat Completions or Responses, declared capabilities | The declared capabilities and the local model's answer using a tool |
| 4 | `04_tool_loop_sync_async.py` | Live (one key) | Multi-round tool calling, synchronous and asynchronous, parallel tools | Tool calls as they run, then the final answers |
| 6 | `06_streaming_hooks_errors.py` | Live (one key) | Streaming, hooks and events, a `before_tool` hook that refuses a dangerous tool, error handling | Streamed text, usage and turn events, the answer without the refused tool, error handling |
| 8 | `08_memory_files_rag.py` | Live (one key) | Memory, a file store, image input, RAG with verified sources | The answer and the passages the search tool really delivered |
| 9 | `09_usage_budget.py` | Live | Usage per provider, custom prices, a budget policy for unpriced models | Usage records per provider and the budget spent; unpriced requests are refused or reported, depending on the policy |

## Advanced

| # | Script | Mode | What it shows | What you see |
|---|--------|------|---------------|--------------|
| 10 | `10_mixed_provider_delegation.py` | Live (two or three keys) | A Claude agent delegating to OpenAI and Gemini agents, a judge of another provider, a shared usage tracker | The delegated answer, whether the judge passed it, tokens per provider |
| 11 | `11_dynamic_tools_sandbox.py` | Live (one key), interactive | A coding agent writes a tool; static check, sandbox tests and **your** approval before registration | The proposed tool, an approval prompt (`approve? [y/N]`), then the answer using the new tool |

## Prerequisites and limitations

* **Keys:** `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY` (or
  `GOOGLE_API_KEY`). Each agent receives only its own provider's key.
* **Models:** `ORKEXA_ANTHROPIC_MODEL` (default `claude-sonnet-5-5`),
  `ORKEXA_OPENAI_MODEL` (default `gpt-6.1-sol`), `ORKEXA_GEMINI_MODEL` (default
  `gemini-3.8-flash`). The defaults are examples; use any model your account can
  access.
* **Single-provider examples** (4, 5, 6, 8, 11) use `ORKEXA_MODEL`, e.g.
  `openai:gpt-6.1-sol` (default: the first provider with a key).
* **Example 3:** `ORKEXA_BASE_URL` (default `http://localhost:11434/v1`, Ollama),
  `ORKEXA_COMPATIBLE_MODEL` (default `llama3.1:8b`), `ORKEXA_COMPATIBLE_API`
  (`chat_completions` or `responses`), `ORKEXA_COMPATIBLE_API_KEY` if the server needs
  one. `OPENAI_API_KEY` is never sent to such a server. Whether tool calling works
  depends on the model and the server.
* **Example 11** uses `DockerSandbox` when the `orkexa-sandbox` image exists
  (`docker build -t orkexa-sandbox -f docker/sandbox.Dockerfile .`). Without Docker it
  stops before any model call, unless you set `ORKEXA_ALLOW_SUBPROCESS_SANDBOX=1`; then
  it uses `SubprocessSandbox`, which is **not a security boundary**. Approve only code
  you have read.
* Model output varies between runs and providers; the output described above is what
  to expect, not an exact transcript.

## How the examples are tested

`tests/test_examples.py` runs every example end to end without network access or API
keys: the SDK client classes are replaced by the scripted clients of the contract
harnesses, so the examples exercise the real adapters, runtime and public API. A test
also fails if an example is added without a test. CI type-checks the examples with
`mypy --strict` and lints them with `ruff`. Live runs are not part of the automated
tests.
