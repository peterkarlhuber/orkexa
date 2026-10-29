# Orkexa examples

Every example is a standalone script: `python examples/01_text_agents.py`.

| # | Script | Shows | Needs |
|---|--------|-------|-------|
| 1 | `01_text_agents.py` | the same text agent on Anthropic, OpenAI, Gemini | `orkexa[all]`, one or more API keys |
| 2 | `02_same_tool_three_providers.py` | one `@Tool` used by three native providers | `orkexa[all]`, API keys |
| 3 | `03_compatible_endpoint.py` | OpenAI-compatible endpoint, Chat Completions or Responses | `orkexa[openai]`, a running Ollama/vLLM/gateway |
| 4 | `04_tool_loop_sync_async.py` | multi-round tool calling, sync and async, parallel tools | one API key |
| 5 | `05_structured_output.py` | Pydantic structured output | one API key |
| 6 | `06_streaming_hooks_errors.py` | streaming, hooks, events, refusal and error handling | one API key |
| 7 | `07_history_migration.py` | save/load, migrating a `claude-agent` 0.8.0 history | runs offline; continuing needs a key |
| 8 | `08_memory_files_rag.py` | memory, file store, images, RAG with verified sources | one API key |
| 9 | `09_usage_budget.py` | usage per provider, custom pricing, budget policy | API keys |
| 10 | `10_mixed_provider_delegation.py` | Claude delegates to OpenAI/Gemini, judge of another provider | API keys of two or three providers |
| 11 | `11_dynamic_tools_sandbox.py` | tool generation, static check, sandbox, approval | one API key (Docker optional) |
| 12 | `12_unchanged_claude_agent_script.py` | an unchanged `claude_agent` 0.8.0 script | `orkexa-claude-compat`, `ANTHROPIC_API_KEY` |

## Environment variables

* `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY` (or `GOOGLE_API_KEY`): the
  provider keys. Examples skip providers whose key is missing.
* `ORKEXA_ANTHROPIC_MODEL` (default `claude-sonnet-5-5`), `ORKEXA_OPENAI_MODEL`
  (default `gpt-6.1-sol`), `ORKEXA_GEMINI_MODEL` (default `gemini-3.8-flash`): model ids;
  the defaults are examples, use any model your account can access.
* `ORKEXA_MODEL`: the model of the single-provider examples (4, 5, 6, 8, 11), e.g.
  `openai:gpt-6.1-sol` (default: the first provider with a key).
* `ORKEXA_BASE_URL`, `ORKEXA_COMPATIBLE_MODEL`, `ORKEXA_COMPATIBLE_API`: example 3.

## How the examples are tested

`tests/test_examples.py` imports every example (no network on import) and the CI runs
`mypy --strict` and `ruff` over them, so every snippet uses the real public API.
Example 7 runs completely offline in the test suite. Live runs need API keys and cost
money; they are not part of the automated tests.
