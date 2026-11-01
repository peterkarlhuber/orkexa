# Data flows and retention

## What leaves the process

Orkexa itself sends data only to the model endpoint you configure, and only when an
agent makes a request (`ask()`, `count_tokens()`, a batch). Importing Orkexa makes no
network request.

| Data | Sent to | When |
| --- | --- | --- |
| System prompt, conversation history, tool definitions, tool results, attached images and documents | The provider of the agent's model (`anthropic`, `openai`, `gemini`, `bedrock`) or the `base_url` of a compatible endpoint | Every model request of a turn; the whole history is sent each time, because requests are stateless |
| Search hits and memory contents | Same provider, as tool results | When the model calls the `search` or `memory` tool |
| Files with `files_api=client` | Anthropic's Files API | The first time a file is loaded; later requests reference its id |
| Queries of server tools (e.g. Anthropic web search) | The provider and the services it uses for the tool | When you enable a server tool and the model uses it |
| Generated tool code and its tests | The configured sandbox (a Docker container by default; a local subprocess only after an explicit opt-in) | When `ToolBuilder` validates or a registry runs a generated tool |
| Model and inference profile ids, guardrail ids | The Bedrock control plane (`GetInferenceProfile`, `GetGuardrail`) in the source region; no conversation content | Before the first request of a `bedrock` agent with `data_residency` or `allowed_regions`, then every 300 seconds |
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

Files and databases Orkexa creates are readable only by the user that runs it (`0600`,
directories `0700`); see [Stored data](../guides/security.md#stored-data).

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

On Amazon Bedrock, what AWS keeps depends on your account and model: model invocation
logging (off by default) stores full prompts and responses in CloudWatch Logs or S3
when an operator enables it, and abuse detection can retain inputs and outputs of some
models for up to 30 days. See
[Amazon Bedrock data residency](../guides/security.md#amazon-bedrock-data-residency).

## Third parties and international transfers

Every model provider, and every compatible endpoint you configure, receives the data
listed above. Depending on your setup they may act as processors, sub-processors or
independent controllers, and they may process data outside the European Economic Area.
Choosing a provider, a region or a data residency option, concluding a data processing
agreement and documenting a transfer mechanism are the integrator's tasks. A local
server (`ollama`, `vllm`) keeps the data on infrastructure you operate.

Claude through `anthropic` and Claude through `bedrock` are different recipients:
the first sends data to Anthropic under Anthropic's terms, the second to AWS in the
regions of the model id or inference profile, under your AWS agreements. For Bedrock,
`BedrockOptions(data_residency="eu_only")` checks before each new model or profile
that every region it can route to is an EU region and refuses to send otherwise.
This verifies Orkexa's configuration; it does not guarantee where AWS processes data.
Enforce it with IAM and service control policies (see
[Amazon Bedrock data residency](../guides/security.md#amazon-bedrock-data-residency)).
