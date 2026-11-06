# ADR 0013 – Amazon Bedrock: credentials, endpoints and data residency

Status: accepted (records the design of the Bedrock provider of 2026-11-01)

## Context

The `bedrock` provider runs Claude, Amazon Nova and other models through the Converse and
ConverseStream operations of `bedrock-runtime` with `boto3`. Three things differ from the
key-based providers of [ADR 0007](0007-credentials-and-endpoints.md):

* Requests are signed with AWS credentials (access key id, session token, signature), not
  sent with one API key, and boto3 finds credentials in many places on its own.
* boto3 also takes endpoint URLs from the environment and the AWS config file
  (`AWS_ENDPOINT_URL*`), so a variable could send signed requests to another host.
* Where a request is processed depends on the model id: a foundation model runs in the
  source region, a cross-Region inference profile routes to one of its destination
  regions, a global profile to any commercial region. A user who needs EU-only
  processing cannot tell from the id alone (`eu.` is a naming convention).

## Decision

**Credentials.** The provider uses only boto3's credential chain (role, IAM Identity
Center, named profile, the `AWS_*` variables, a Bedrock API key in
`AWS_BEARER_TOKEN_BEDROCK`), a `boto3.Session` or a ready `bedrock-runtime` client the
caller passes. `api_key=` is rejected, and no other provider's key is ever read. Hosts
that serve several tenants pass a session per call (Nodestro enforces this with its
`require_session` guard; a library-level fail-closed flag is ORK-005).

**Endpoints.** `AWS_ENDPOINT_URL*` from the environment and the config file are ignored,
so signed requests go to the regional AWS endpoint. An explicit `base_url` must be
`https` (`http` only for `localhost`). The region of the client is the source region:
the endpoint, the region AWS bills and the region reported for pricing.

**Data residency.** `BedrockOptions(data_residency="eu_only")` or `allowed_regions=(...)`
checks, before the first paid request, that the source region and every region the model
or inference profile can route to are allowed (`GetInferenceProfile` lists the
destination regions; foundation models, provisioned throughput, custom and imported
models run in the region of their ARN). Global profiles, prompt resources, prompt routers
and anything whose routing cannot be determined are refused. Every check fails closed
with `DataResidencyError`; nothing is sent. The checks live in
`orkexa.providers.bedrock_residency`, which does not import boto3 and is tested with
scripted control-plane clients.

**Prices.** Orkexa ships no Bedrock prices and never applies Anthropic's: Bedrock prices
differ by region, profile and tier ([ADR 0008](0008-pricing-engine.md)).

**Capabilities.** Features per model family come from the AWS documentation
(`bedrock_capabilities()`); a model the table does not know reports `unknown`, never a
guess ([ADR 0003](0003-capabilities.md)). The table is dated and reviewed with each boto3
update and before a release ([Releasing](../../development/releasing.md#upkeep)).

## Consequences

* A misconfigured environment cannot redirect signed requests or leak another provider's
  key into a Bedrock call.
* EU-only processing is verified, not assumed; the check costs one control-plane call
  per model and region, reused for five minutes per provider instance
  (`RESIDENCY_CACHE_SECONDS`).
* Out of scope, refused with a clear error or documented as not covered (ORK-011): batch
  inference, citations and server tools, custom or VPC endpoints beyond an explicit
  `https` `base_url`, and the AWS European Sovereign Cloud partition.
