# Feature parity matrix

Status values: `tested` (implemented and covered by automated offline tests),
`implemented` (implemented, no automated test), `not supported` (deliberately absent,
documented). Live API tests were not run (no API keys in the build environment).

| # | claude-agent 0.8.0 feature | Orkexa component | Compatibility strategy | Tests | Status |
| --- | --- | --- | --- | --- | --- |
| 1 | `ClaudeAgent`/`AsyncClaudeAgent` API | `orkexa.claude` | facade over the shared runtime | `tests/legacy` (all files) | tested |
| 2 | request building, betas, effort, thinking, caching, fallbacks | `providers.anthropic` | identical request kwargs | `tests/legacy/test_agent.py`, `test_features.py`, `test_turn_options.py` | tested |
| 3 | tool loop, rounds, parallel tools, timeouts, hooks | `core.runtime`, drivers | one state machine for every provider | legacy suite, `tests/test_runtime.py`, `tests/contract` | tested |
| 4 | streaming hooks | adapters → `StreamEvent` | `on_text`/`on_thinking` unchanged, `on_event` added | legacy suite, contract streaming parity (5 adapters) | tested |
| 5 | structured output | adapters + local validation | `ask()` returns the model instance | legacy suite, contract tests, `test_providers.py` | tested |
| 6 | refusals, incomplete responses | runtime | same exceptions | legacy suite, contract tests | tested |
| 7 | history, save/load, token counting | `core.conversion`, codecs | 0.8.0 files readable; versioned Orkexa format | legacy suite, contract save/load + cross-provider, `test_examples.py::test_example_07` | tested |
| 8 | usage/pricing | `usage`, `pricing` | `UsageRecord` extended with trailing fields | `tests/legacy/test_usage.py`, contract usage normalization, mixed-provider budget tests | tested |
| 9 | `@Tool`, content helpers | `tools`, `content` | unchanged | `tests/legacy/test_tools.py`, contract tests | tested |
| 10 | memory backends + native memory tool | `memory` | native on Anthropic, function tool elsewhere | `tests/legacy/test_memory.py`, `test_mixed_providers.py::test_memory_is_a_function_tool_for_openai` | tested |
| 11 | file stores | `files` | same tool for every provider; Files API Anthropic-only | `tests/legacy/test_files.py`, `test_mixed_providers.py` (OpenAI image, Gemini document) | tested |
| 12 | index, RAG, citations | `retrieval` | neutral search results; `retrieval_sources()` added | `tests/legacy/test_retrieval.py`, `test_mixed_providers.py` (Gemini) | tested |
| 13 | evaluator, refine | `evaluation` | judge of any provider | `tests/legacy/test_evaluation.py`, `test_mixed_providers.py` | tested |
| 13a | batch evaluation | `evaluation` + `run_batch` | Anthropic native; others `UnsupportedCapabilityError` or sequential | legacy suite, `test_evaluate_many_without_native_batch_is_explicit` | tested |
| 14 | dynamic package | `dynamic` | generic agents, `ProviderSettings`, qualified models | `tests/legacy/test_dynamic_*.py`, `test_mixed_providers.py` | tested |
| 15 | sandbox paths | `dynamic.sandbox`, `_sandbox_runner` | migrated to `orkexa` paths | `tests/legacy/test_dynamic_sandbox.py`, `tests/test_sandbox_migration.py` | tested (Docker run itself: skipped, no Docker) |
| 16 | legacy imports | `orkexa-claude-compat` | alias distribution, conflict guard | `tests/packaging` (fresh venvs) | tested |
| 17 | live API test | `tests/legacy/test_live.py` | unchanged | needs `ANTHROPIC_API_KEY` | skipped |

Documented deviations of the copied 0.8.0 suite: one assertion (sandbox temp directory
prefix `orkexa-sandbox-`) and the image name of the skipped Docker test helper.
