# Security

## Defaults

* **Credentials** are never logged, put into exceptions (messages are redacted),
  serialized into histories, passed into sandboxes or shared between providers.
  `ProviderConfig`/`ProviderSettings` hide keys in `repr`, and credentials in a
  `base_url` (user info such as `https://user:pass@host`, query parameters such as
  `api_key=`, `token=`, `sig=`) are shown as `[REDACTED]`; the URL used for requests
  is unchanged.
* **Provider errors** of generic agents (`Agent`, `AsyncAgent`) are raised without a
  link to the SDK exception: `__cause__` and `__context__` are `None`, so neither a
  traceback nor `logging.exception()` can print the SDK exception, its request or its
  headers. The error keeps the safe diagnostics `provider`, `status_code`,
  `request_id`, `retryable` and the redacted message (which names the SDK exception
  type). `ClaudeAgent` raises the Anthropic SDK exceptions unchanged.
* **Stored data is owner-private** (see [below](#stored-data)).
* **Endpoints**: a custom `base_url` for `anthropic`, `openai` or `gemini` requires an
  explicit `api_key=`; environment keys are only sent to the provider's own API.
  `OPENAI_BASE_URL` is ignored. `compatible`/`ollama`/`vllm` never read
  `OPENAI_API_KEY`. Treat a configurable `base_url` as security-relevant: it receives
  the full conversation.
* **Tools** run only in Orkexa, after `before_tool`; SDK automatic function calling is
  disabled. Tool inputs are validated; tool rounds, parallelism and timeouts are limited.
* **OpenAI** requests use `store=False` by default.
* **No network on import** and no live API calls in the default test suite.

## Stored data

Histories, memory, file stores, search indexes and registry records can contain
personal data. Orkexa creates them readable only by the user that runs it, without
changing the process umask:

| What | Mode |
| --- | --- |
| `save_history()` files (`Agent`, `AsyncAgent`, `ClaudeAgent`), `ToolRegistry` and `AgentRegistry` records | `0600` on every write; replaced atomically (temporary file, `fsync`, rename); a symbolic link is replaced, not followed |
| `SQLiteMemory`, `SQLiteFileStore`, `SQLiteIndex` databases | `0600` when Orkexa creates them; SQLite gives the journal, `-wal` and `-shm` files the same mode |
| `FileMemory`, `LocalFileStore` | files `0600`, directories `0700` |
| directories Orkexa creates (including missing parents) | `0700` |

Existing files and directories are not made more permissive, and an existing SQLite
database or directory keeps its mode, so a deliberately shared setup keeps working.
To make an existing database private, run `chmod 600 store.db*`.

On Windows, mode bits only control the read-only flag; access is governed by the ACLs
a new file inherits from its directory (private below the user profile by default).
Keep stores in a directory with suitable ACLs.

## Dynamic code

* **Generated code runs in `DockerSandbox` by default.** `ToolRegistry()` without a
  `sandbox` uses `DockerSandbox()`, which runs code without network, with a read-only
  file system, without capabilities, as an unprivileged user, with memory, CPU and PID
  limits. If Docker or the image is missing, every validation or call fails with a
  `SandboxError` (wrapped in `ToolExecutionError` for tool calls) that explains the
  options; no code runs. Build the image with
  `docker build -t orkexa-sandbox -f docker/sandbox.Dockerfile .`.
* **`SubprocessSandbox` is an explicit, unsafe opt-in.** It runs code in a separate
  process with an empty environment, a private temporary directory (`0700`), a timeout
  and resource limits, but **it is not a security boundary**: the process can reach
  the network and read every file the user can read. Creating one emits
  `UnsafeOptInWarning` unless created with `SubprocessSandbox(acknowledge_unsafe=True)`.
  Use it only for code you trust or have reviewed, e.g. on a development machine.
* The static check rejects imports outside an allow list, dunder access,
  `eval`/`exec`, file and network access patterns and access to modules through the
  attributes of allowed modules (`calendar.sys`, `json.codecs`, `enum.bltns` ...). A
  test walks every module reachable from the allow list, so a new Python or dependency
  version that exposes another module fails CI until it is reviewed. **It is a filter,
  not a sandbox**: it reduces what reaches the sandbox, it does not replace it.
* `AutoApprover` is an **unsafe opt-in**: it emits `UnsafeOptInWarning` unless created
  with `AutoApprover(acknowledge_unsafe=True)`.
* `ToolRegistry` and `AgentRegistry` treat names as untrusted: only tool names
  `[A-Za-z0-9_-]{1,128}` and agent names `[A-Za-z][A-Za-z0-9_-]{0,63}` are accepted,
  records are written only inside the registry directory, and stored records whose
  name does not match their directory are skipped on load.
* `DynamicPolicy.allowed_agent_tools` limits the tools of generated agents and the
  tools a model can attach to itself with `ToolBuilder.as_tools()`' `load_tool`.

## Production configuration for generated code

* Use `DockerSandbox` (the default) or stronger isolation such as a VM for both the
  tests and the calls of generated tools. Use `SubprocessSandbox(acknowledge_unsafe=True)`
  only on trusted development machines.
* Use a human or LLM-plus-human `Approver`, never a bare `AutoApprover`.
* Set `DynamicPolicy(allowed_models=..., allowed_agent_tools=..., max_generated_tools=...,
  budget_usd=...)` and a `BudgetTracker(on_unpriced="reject")` with registered prices.
* Register only side-effect-free functions as trusted tools that models may load, and
  guard side-effecting tools with `before_tool`.
* A tool timeout does not stop a running tool thread: Python threads cannot be killed,
  so a timed-out tool may still finish its side effects. Put tools with risky side
  effects behind a process boundary (a sandbox or a service) with its own timeout.

## Migrating to the secure defaults

| You relied on | Now | What to do |
| --- | --- | --- |
| `ToolRegistry()` running generated tools in a subprocess | it uses `DockerSandbox()` and fails closed without Docker | build the image (see above); for trusted or reviewed code only, pass `ToolRegistry(path, sandbox=SubprocessSandbox(acknowledge_unsafe=True))` |
| `SubprocessSandbox()` without arguments | works, but emits `UnsafeOptInWarning` | pass `acknowledge_unsafe=True` after deciding that the code is trusted |
| `exc.__cause__` of a `ProviderError` to inspect the SDK exception | `None` | use `provider`, `status_code`, `request_id`, `retryable` and the message; for SDK details, log them in your own code before they reach Orkexa (e.g. an `httpx` event hook) and keep credentials out |
| other users or groups reading histories, registries or new databases | `0600` / `0700` | grant access explicitly (`chmod`, group ownership) or share through an application |
| generated code that reaches modules through attributes of allowed modules | rejected by the static check | import only allowed modules directly |

## Remaining risks

* The static check is a deny list on top of an import allow list; it is not a proof
  that code cannot escape. Only the sandbox is the boundary.
* Unmapped exceptions (programming errors, `KeyboardInterrupt`) and every exception of
  `ClaudeAgent` are raised as they are; an SDK exception there can still print request
  details in a traceback. Do not log tracebacks of `ClaudeAgent` errors verbatim where
  logs are less protected than the API key.
* Existing SQLite databases, histories and registry directories keep their mode until
  they are rewritten (histories and records) or changed by hand (databases,
  directories).
* Budgets reserve the most expensive outcome of every request in flight, so concurrent
  agents cannot overshoot a shared limit together. A single oversized request (with
  the default `on_oversized="exclusive"`), requests without a cost bound (no price,
  native tools, images, binary documents) and every item of an admitted native batch
  can still exceed it; `hard_maximum` tells whether that happened (see
  [Usage and budgets](usage-and-budgets.md)).
* `DockerSandbox` isolation has not been audited; container escapes are out of scope
  of Orkexa.
* Models from unknown endpoints may ignore schemas; Orkexa validates locally but cannot
  prevent a server from logging prompts.
* Prompt injection through tool results, search hits or files is not prevented by
  Orkexa; keep `before_tool` policies for side-effecting tools.
