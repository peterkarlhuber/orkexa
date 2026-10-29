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

## Remaining risks

* Budgets can only be enforced after a response; one request can exceed the limit.
* Models from unknown endpoints may ignore schemas; Orkexa validates locally but cannot
  prevent a server from logging prompts.
* Prompt injection through tool results, search hits or files is not prevented by
  Orkexa; keep `before_tool` policies for side-effecting tools.
