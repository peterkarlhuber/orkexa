# Security

## Defaults

* **Credentials** are never logged, put into exceptions (messages are redacted),
  serialized into histories, passed into sandboxes or shared between providers.
  `ProviderConfig`/`ProviderSettings` hide keys in `repr`.
* **Endpoints**: a custom `base_url` for `anthropic`, `openai` or `gemini` requires an
  explicit `api_key=`; environment keys are only sent to the provider's own API.
  `OPENAI_BASE_URL` is ignored. `compatible`/`ollama`/`vllm` never read
  `OPENAI_API_KEY`. Treat a configurable `base_url` as security-relevant: it receives
  the full conversation.
* **Tools** run only in Orkexa, after `before_tool`; SDK automatic function calling is
  disabled. Tool inputs are validated; tool rounds, parallelism and timeouts are limited.
* **OpenAI** requests use `store=False` by default.
* **No network on import** and no live API calls in the default test suite.

## Dynamic code

* The static check rejects imports outside an allow list, dunder access, `eval`/`exec`,
  file and network access patterns; it is a filter, not a proof of safety.
* `SubprocessSandbox` runs code in a separate process with an empty environment, a
  temporary directory, a timeout and resource limits. **It is not a security
  boundary**: the process can reach the network and read files of the user.
* `DockerSandbox` runs code without network, read-only, without capabilities, as an
  unprivileged user, with memory/CPU/PID limits. Use it (or stronger isolation) for
  untrusted code in production. Build the image with
  `docker build -t orkexa-sandbox -f docker/sandbox.Dockerfile .`.
* `AutoApprover` is an **unsafe opt-in**: it emits `UnsafeOptInWarning` unless created
  with `AutoApprover(acknowledge_unsafe=True)`.
* `ToolRegistry` and `AgentRegistry` treat names as untrusted: only tool names
  `[A-Za-z0-9_-]{1,128}` and agent names `[A-Za-z][A-Za-z0-9_-]{0,63}` are accepted,
  records are written only inside the registry directory, and stored records whose
  name does not match their directory are skipped on load.
* `DynamicPolicy.allowed_agent_tools` limits the tools of generated agents and the
  tools a model can attach to itself with `ToolBuilder.as_tools()`' `load_tool`.

## Production configuration for generated code

* Use `DockerSandbox` (or stronger isolation such as a VM) for both the tests and the
  calls of generated tools: `ToolRegistry(path, sandbox=DockerSandbox())`. Use
  `SubprocessSandbox` only on trusted development machines.
* Use a human or LLM-plus-human `Approver`, never a bare `AutoApprover`.
* Set `DynamicPolicy(allowed_models=..., allowed_agent_tools=..., max_generated_tools=...,
  budget_usd=...)` and a `BudgetTracker(on_unpriced="reject")` with registered prices.
* Register only side-effect-free functions as trusted tools that models may load, and
  guard side-effecting tools with `before_tool`.
* A tool timeout does not stop a running tool thread: Python threads cannot be killed,
  so a timed-out tool may still finish its side effects. Put tools with risky side
  effects behind a process boundary (a sandbox or a service) with its own timeout.

## Remaining risks

* Budgets are soft limits: the last admitted request, one request per concurrently
  running agent, and every item of an admitted native batch can exceed the limit (see
  [Usage and budgets](usage-and-budgets.md)).
* `DockerSandbox` isolation has not been audited; container escapes are out of scope
  of Orkexa.
* Models from unknown endpoints may ignore schemas; Orkexa validates locally but cannot
  prevent a server from logging prompts.
* Prompt injection through tool results, search hits or files is not prevented by
  Orkexa; keep `before_tool` policies for side-effecting tools.
