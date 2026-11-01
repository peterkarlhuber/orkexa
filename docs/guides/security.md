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
  `OPENAI_API_KEY`. `bedrock` uses only AWS credentials (it rejects `api_key=`),
  ignores `AWS_ENDPOINT_URL*` from the environment and the AWS config file, and
  accepts only an `https` `base_url` (signed requests carry the access key id and
  session token). Treat a configurable `base_url` as security-relevant: it receives
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

## Amazon Bedrock data residency

`BedrockOptions(data_residency="eu_only")` makes the `bedrock` provider check, before
the first paid request, where the request would be processed. It is a configuration
check that fails closed: whatever cannot be verified raises `DataResidencyError`
(a `ConfigurationError`) and nothing is sent to the model.

```python
from orkexa import Agent, BedrockOptions

agent = Agent(
    provider="bedrock",
    model="eu.anthropic.claude-sonnet-5-5",
    options=BedrockOptions(region_name="eu-central-1", data_residency="eu_only"),
)
```

### What is verified

* **Region and endpoint.** The `bedrock-runtime` client's region must be on the
  allowlist. Only AWS endpoints of that region are accepted: the default regional
  endpoint or an interface VPC endpoint
  (`*.bedrock-runtime.<region>.vpce.amazonaws.com`) passed as `base_url`, over `https`.
* **Model id.** A foundation model id or ARN, provisioned throughput, custom model
  deployments and imported models run in their region. For a cross-Region or
  application inference profile, Orkexa calls `GetInferenceProfile` in the source
  region and requires the profile to be `ACTIVE` and **every** destination region
  (the region of each `models[].modelArn`) to be on the allowlist; AWS recommends
  exactly this instead of trusting the `eu.` prefix
  ([supported Regions and models for inference profiles](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-support.html)).
  The prefix is never trusted on its own.
* **Global profiles** (`global.anthropic...`) route to all commercial AWS regions and
  are always rejected, before any AWS call.
* **ARNs of another region** than the client's are rejected. Prompt resources, prompt
  routers and SageMaker endpoints cannot be verified and are rejected.
* **Guardrails.** A configured guardrail is read with `GetGuardrail`; a guardrail in
  another region or with a cross-Region guardrail profile is rejected.
* **Media.** `s3://` and URL sources of images and documents are rejected; send the
  content as bytes.

A verified model or guardrail is cached for 300 seconds per region and id
(`RESIDENCY_CACHE_SECONDS`), then checked again. AWS documents that the destination
regions of a geographic profile never change, while those of global profiles can
grow over time.

With an injected `client=`, pass the control-plane client for these checks as
`BedrockOptions(control_client=boto3.client("bedrock", region_name=...))`.

### The allowlist

`eu_only` allows the AWS commercial regions in EU member states
(`orkexa.providers.bedrock.EU_REGIONS`): `eu-central-1` (Frankfurt), `eu-north-1`
(Stockholm), `eu-south-1` (Milan), `eu-south-2` (Spain), `eu-west-1` (Ireland) and
`eu-west-3` (Paris).

* **London (`eu-west-2`) and Zurich (`eu-central-2`) are in Europe but not in the
  EU** and are not allowed. EU inference profiles can list them as destinations for
  some models and source regions, and such a profile is then rejected.
* The AWS European Sovereign Cloud is a separate partition and is not covered.
* `allowed_regions=("eu-central-1", "eu-west-1")` narrows the list (with `eu_only`
  it may only contain EU regions). Without `data_residency`, `allowed_regions`
  enforces an allowlist for any geography.

### Required IAM permissions

With `eu_only` and a geographic inference profile, the caller needs:

* `bedrock:InvokeModel` (Converse) and `bedrock:InvokeModelWithResponseStream`
  (ConverseStream) on the inference profile **and** on the foundation model in the
  source region and in **all** destination regions of the profile
  ([geographic cross-Region inference](https://docs.aws.amazon.com/bedrock/latest/userguide/geographic-cross-region-inference.html),
  [prerequisites for inference profiles](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-prereq.html));
* `bedrock:GetInferenceProfile` on the profile (for the residency check);
* `bedrock:GetGuardrail` on the guardrail, if you configure one.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "InvokeEuProfile",
      "Effect": "Allow",
      "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
      "Resource": [
        "arn:aws:bedrock:eu-central-1:111122223333:inference-profile/eu.anthropic.claude-sonnet-5-5",
        "arn:aws:bedrock:eu-*::foundation-model/anthropic.claude-sonnet-5-5"
      ]
    },
    {
      "Sid": "VerifyResidency",
      "Effect": "Allow",
      "Action": ["bedrock:GetInferenceProfile"],
      "Resource": "arn:aws:bedrock:eu-central-1:111122223333:inference-profile/eu.anthropic.claude-sonnet-5-5"
    }
  ]
}
```

A missing `GetInferenceProfile` permission makes the check fail with
`DataResidencyError`; it is never skipped.

### Enforcing residency on the AWS side

Orkexa's check protects one application's configuration. The enforcement for an AWS
account or organization is IAM and service control policies (SCPs). The following SCP
is an **example to adapt and test**, not a ready-made policy: it denies Bedrock
requests outside the six EU regions and requests through global inference profiles.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "DenyBedrockOutsideEU",
      "Effect": "Deny",
      "Action": "bedrock:*",
      "Resource": "*",
      "Condition": {
        "StringNotEquals": {
          "aws:RequestedRegion": [
            "eu-central-1", "eu-north-1", "eu-south-1",
            "eu-south-2", "eu-west-1", "eu-west-3"
          ]
        }
      }
    },
    {
      "Sid": "DenyGlobalInferenceProfiles",
      "Effect": "Deny",
      "Action": "bedrock:*",
      "Resource": "*",
      "Condition": {
        "StringEquals": { "aws:RequestedRegion": "unspecified" },
        "ArnLike": { "bedrock:InferenceProfileArn": "arn:aws:bedrock:*:*:inference-profile/global.*" }
      }
    }
  ]
}
```

* A global profile is authorized against a region-less foundation model ARN, for
  which `aws:RequestedRegion` is `unspecified`. The first statement does not list
  `unspecified` and therefore already denies it; the second statement is the deny
  statement AWS documents for disabling global cross-Region inference
  ([global cross-Region inference](https://docs.aws.amazon.com/bedrock/latest/userguide/global-cross-region-inference.html)).
* A geographic profile is authorized against the foundation model in every candidate
  destination region, so the first statement also denies EU profiles that can route
  to London or Zurich. Allow those regions only if your requirements permit them.
* Cross-Region inference can route requests to regions that are not enabled in your
  account ([cross-Region inference](https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html));
  only policies like the one above restrict them.
* SCPs do not apply to the management account. Check the policy against your other
  services before you attach it, and keep it in place even when every application
  uses `eu_only`.
* CloudTrail in the source region records every cross-Region request;
  `additionalEventData.inferenceRegion` shows where it was processed. Use it to audit
  the actual processing regions.

### Model invocation logging

[Model invocation logging](https://docs.aws.amazon.com/bedrock/latest/userguide/model-invocation-logging.html)
is **disabled by default**. When an account enables it, Bedrock stores the **full
request and response data** of Converse and ConverseStream calls, including prompts,
tool results, documents and images, with metadata, in CloudWatch Logs and/or S3 of
the same account and region. Orkexa does not check whether it is enabled and cannot
redact what Bedrock logs. Operators must disable it or secure it like any other store
of personal data: encryption, access control, retention periods, deletion.
`BedrockOptions(request_metadata=...)` is written to these logs; keep personal data
out of it.

### Abuse detection and retention

Amazon Bedrock may use automated abuse detection and, in certain limited
circumstances, store or review model inputs and outputs
([abuse detection](https://docs.aws.amazon.com/bedrock/latest/userguide/abuse-detection.html)).
AWS documents exceptions with retention for some models, for example:

* Claude Fable 5 and Claude Fable 5.1: all traffic is kept for up to 30 days, and
  flagged traffic may be reviewed by AWS staff;
* OpenAI models other than the gpt-oss family: traffic flagged by classifiers is kept
  for up to 30 days.

With cross-Region inference, retained inputs and outputs are stored in the
destination regions. AWS also documents per-region
[data retention settings](https://docs.aws.amazon.com/bedrock/latest/userguide/data-retention.html)
for some APIs; check which apply to your models and to the Converse API. These terms
change; read the current AWS pages before choosing a model for personal data.

### What Orkexa cannot guarantee

Orkexa verifies the configuration it sends requests with. It **cannot guarantee or
certify** where AWS processes or stores data, and `eu_only` does not make a system
GDPR compliant. The enforcement point is your AWS organization (SCPs, IAM,
invocation-logging settings, your agreements with AWS); the legal assessment is
yours. See [Data flows and retention](../compliance/data-flows.md) and
[GDPR](../compliance/gdpr.md).

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
