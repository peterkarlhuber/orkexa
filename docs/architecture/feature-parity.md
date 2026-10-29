# Feature parity matrix

Status values: `planned`, `implemented`, `tested`. This initial version was written
before the core was replaced (M0); the final status is maintained in the same file.

| # | 0.8.0 feature | Orkexa component | Compatibility strategy | Test | Status |
| --- | --- | --- | --- | --- | --- |
| 1 | `ClaudeAgent`/`AsyncClaudeAgent` API | `orkexa.claude` | facade over `Agent`/`AsyncAgent` | legacy suite | planned |
| 2 | request building, betas, effort, thinking, caching, fallbacks | `providers.anthropic` | same request dicts | legacy suite | planned |
| 3 | tool loop, rounds, parallel tools, hooks | `core.runtime` | shared state machine | legacy suite + contract tests | planned |
| 4 | streaming hooks | adapters → `StreamEvent` | `on_text`/`on_thinking` unchanged | legacy suite + contract tests | planned |
| 5 | structured output | adapters + runtime | `ask()` returns model instance | legacy suite + contract tests | planned |
| 6 | refusals, incomplete responses | runtime | same exceptions | legacy suite + contract tests | planned |
| 7 | history, save/load, token counting | `core.conversion`, adapters | Anthropic codec, legacy files | legacy suite + history tests | planned |
| 8 | usage/pricing | `usage`, `pricing` | extended dataclasses | legacy suite + usage tests | planned |
| 9 | `@Tool`, content helpers | `tools`, `content` | same | legacy suite | planned |
| 10 | memory backends + native memory tool | `memory` | native on Anthropic, function tool elsewhere | legacy suite + cross-provider tests | planned |
| 11 | file stores | `files` | same; Files API Anthropic-only | legacy suite | planned |
| 12 | index, RAG, citations | `retrieval` | neutral search results/citations | legacy suite + cross-provider tests | planned |
| 13 | evaluator, refine, batch | `evaluation` | any provider; Anthropic batch native | legacy suite + mixed-provider tests | planned |
| 14 | dynamic package | `dynamic` | generic agent contract | legacy suite + mixed-provider tests | planned |
| 15 | sandbox paths | `dynamic.sandbox` | migrated to orkexa | sandbox tests | planned |
| 16 | legacy imports | `orkexa-claude-compat` | re-export distribution | packaging tests | planned |
