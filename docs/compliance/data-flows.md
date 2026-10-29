# Data flows and retention

## What leaves the process

Orkexa itself sends data only to the model endpoint you configure, and only when an
agent makes a request (`ask()`, `count_tokens()`, a batch). Importing Orkexa makes no
network request.

| Data | Sent to | When |
| --- | --- | --- |
| System prompt, conversation history, tool definitions, tool results, attached images and documents | The provider of the agent's model (`anthropic`, `openai`, `gemini`) or the `base_url` of a compatible endpoint | Every model request of a turn; the whole history is sent each time, because requests are stateless |
| Search hits and memory contents | Same provider, as tool results | When the model calls the `search` or `memory` tool |
| Files with `files_api=client` | Anthropic's Files API | The first time a file is loaded; later requests reference its id |
| Queries of server tools (e.g. Anthropic web search) | The provider and the services it uses for the tool | When you enable a server tool and the model uses it |
| Generated tool code and its tests | The configured sandbox (a local subprocess or a Docker container) | When `ToolBuilder` validates or a registry runs a generated tool |
| Delegated tasks | The provider of each delegated agent, which may differ from the caller's | In multi-agent setups, per delegated agent |

Credentials go only to the provider they belong to. A custom `base_url` for a
first-party provider requires an explicit key, so an environment key is never sent to
a third-party host. Sandboxes start with an empty environment.

## What is stored, and where

| Component | Storage | Default |
| --- | --- | --- |
| Agent history | Process memory | Not persisted; `save_history(path)` writes a JSON file you control |
| `EphemeralMemory` | Process memory | Gone when the process ends |
| `FileMemory`, `LocalFileStore` | Files below the directory you pass | Only if you create one |
| `SQLiteMemory`, `SQLiteFileStore`, `SQLiteIndex` | The SQLite database you pass | Only if you create one |
| `UsageTracker` | Process memory | Metadata only: counts, model, provider, ids, cost |
| `ToolRegistry`, `AgentRegistry` | Files below the directory you pass | Generated code, specs, approval status |
| Python `logging` | Wherever your logging configuration sends it | Orkexa adds no handlers. The dynamic package logs registry problems (file paths) and, at INFO level, the sandbox feedback of failed tool proposals |

## Deleting data

See [Deleting data](../guides/memory-files-rag.md#deleting-data) for what each store's
`delete()` removes and what remains. In short: in-memory state disappears with
`reset()`, `clear()` or the process; SQLite stores overwrite deleted rows
(`SQLiteIndex` after `optimize()`); file-based stores unlink files, which does not
guarantee overwriting on every storage medium.

Orkexa cannot delete what a provider keeps. Requests to model providers are governed
by their retention policies and your agreements with them. OpenAI requests use
`store=False` by default (`OpenAIOptions(store=True)` changes that); check each
provider's current data usage and retention terms, including abuse-monitoring
retention and the options available to your account.

## Third parties and international transfers

Every model provider, and every compatible endpoint you configure, receives the data
listed above. Depending on your setup they may act as processors, sub-processors or
independent controllers, and they may process data outside the European Economic Area.
Choosing a provider, a region or a data residency option, concluding a data processing
agreement and documenting a transfer mechanism are the integrator's tasks. A local
server (`ollama`, `vllm`) keeps the data on infrastructure you operate.
