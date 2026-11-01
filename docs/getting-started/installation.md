# Installation

Orkexa needs Python 3.10 or newer. The core depends only on Pydantic; each provider
SDK is an extra.

!!! note "Not yet on PyPI"
    Orkexa 0.1.0 is unreleased. Until it is published, install it from the repository:

    ```bash
    pip install "orkexa[all] @ git+https://github.com/peterkarlhuber/orkexa"
    # or, from a clone, for development:
    pip install -e ".[dev]"
    ```

    The commands below show the extras as they will be installed from PyPI.

```bash
pip install "orkexa[anthropic]"   # Anthropic Claude (anthropic SDK)
pip install "orkexa[openai]"      # OpenAI and OpenAI-compatible endpoints (openai SDK)
pip install "orkexa[gemini]"      # Google Gemini (google-genai SDK)
pip install "orkexa[bedrock]"     # Amazon Bedrock (boto3)
pip install "orkexa[all]"         # all provider SDKs, including boto3
pip install "orkexa[pdf]"         # PDF text extraction for the index (pypdf)
pip install "orkexa[vector]"      # vector/hybrid search (sqlite-vec)
```

`import orkexa` never imports a provider SDK. Using a provider whose SDK is missing
raises `MissingDependencyError` naming the extra, e.g. `pip install 'orkexa[openai]'`.

## API keys

Set the variable of each provider you use, e.g. in your shell profile or a secrets
manager (never in code or in the repository):

```bash
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
export GEMINI_API_KEY=...
```

| Provider | Variable read by the SDK | Explicit argument |
| --- | --- | --- |
| `anthropic` | `ANTHROPIC_API_KEY` | `api_key=` |
| `openai` | `OPENAI_API_KEY` | `api_key=` |
| `gemini` | `GEMINI_API_KEY` / `GOOGLE_API_KEY` | `api_key=` |
| `bedrock` | boto3's AWS credential chain (see below) | none: `api_key=` is rejected |
| `compatible`, `ollama`, `vllm` | **none** | `api_key=` (only if the server needs one) |

A custom `base_url` for `anthropic`, `openai` or `gemini` requires an explicit
`api_key=`: Orkexa never sends a provider's environment key to another host.
OpenAI's `OPENAI_BASE_URL` variable is ignored for the same reason.

## AWS credentials for Amazon Bedrock

The `bedrock` provider has no API key of its own. `boto3` finds AWS credentials the
usual way, in this order of preference for production:

* an IAM role of the runtime (ECS task role, EKS pod identity or IRSA, EC2 instance
  profile, Lambda execution role);
* IAM Identity Center (SSO): `aws sso login --profile my-profile`, then
  `BedrockOptions(profile_name="my-profile")` or `AWS_PROFILE=my-profile`;
* a named profile in `~/.aws/config` / `~/.aws/credentials`, including profiles that
  assume a role (`role_arn`, `source_profile`);
* `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and `AWS_SESSION_TOKEN`;
* a Bedrock API key in `AWS_BEARER_TOKEN_BEDROCK`, which boto3 sends as a bearer token
  ([Use an Amazon Bedrock API key](https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys-use.html)).
  AWS recommends short-term keys; long-term keys create an IAM user and are meant
  for exploration only.

You can also pass a preconfigured `boto3.Session` (`BedrockOptions(session=...)`) or a
`bedrock-runtime` client (`client=`). Set the region with
`BedrockOptions(region_name="eu-central-1")` or `AWS_DEFAULT_REGION`; without one the first
request raises `ConfigurationError`.

```bash
export AWS_PROFILE=my-profile
export AWS_DEFAULT_REGION=eu-central-1
```

Orkexa never reads `ANTHROPIC_API_KEY` or any other provider's key for Bedrock, and it
ignores endpoint URLs from the environment or the AWS config file
(`AWS_ENDPOINT_URL`, `AWS_ENDPOINT_URL_BEDROCK_RUNTIME`, `endpoint_url` in a profile),
so signed requests go only to the regional AWS endpoint or to a `base_url=` you pass
explicitly (which must use `https`). See
[Providers and capabilities](providers.md#amazon-bedrock) for regions, model ids and
permissions.
