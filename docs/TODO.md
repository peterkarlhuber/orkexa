# TODO: orkexa

> Open work for this repository: release blockers, known bugs, security follow-ups, technical debt and planned improvements.
> GitHub issues are the source of truth for details; this file is the entry point and summary.
>
> **Last full review:** 2026-11-03 against `master` at `0861c1b` (internal backend audit round 2; fixes merged
> in #40 at `bc8c4ed`).

## Summary

| Priority | Count |
|---|---|
| P0 Critical | 0 |
| P1 High | 0 |
| P2 Medium | 2 |
| P3 Low | 13 |
| P4 Optional | 11 |

**Release blockers:** None for this repository. Orkexa 0.1.0 is unreleased and no release is planned yet;
Nodestro consumes it by commit SHA.

The library is in good shape: no open issues or pull requests before this review, no TODO/FIXME markers in the
code, and offline tests, strict typing, packaging, docs and dependency audits run in CI. The open work is mostly
verification against real providers and Docker hosts, follow-ups in Nodestro for the 2026-11-03 audit fixes, and
release and documentation upkeep.

## Priorities and statuses

- **P0 Critical**: data loss, security exposure, or broken core function; blocks any release.
- **P1 High**: blocks a planned release or a promised capability, or a significant risk.
- **P2 Medium**: real defect or gap with a workaround; schedule soon.
- **P3 Low**: minor defect, polish, or small debt.
- **P4 Optional**: improvement idea; not a defect.

Statuses: `Open`, `In progress` (PR linked), `Blocked` (names the blocker), `Needs decision` (owner decision pending),
`Needs external review`, `Accepted risk` (deliberately not fixed; reason stated).

## Release blockers and critical items

None known.

## Security

### ORK-003 · Redirect handling on OpenAI-compatible endpoints

- **Priority:** P3 Low · **Status:** Open · **Refs:** internal backend audit 2026-11-01, follow-up to finding A-05
- **What:** Hardening item. The fix for A-05 lives only in Nodestro's server path; code that uses `orkexa.Agent`
  with a `compatible` endpoint directly still relies on the HTTP client's default redirect handling.
- **Next step:** Decide on a library-level default or option and document it in the security guide (details are
  in the internal audit, not here).

### ORK-004 · Independent review of the static check

- **Priority:** P3 Low · **Status:** Needs external review · **Refs:** [Security guide](guides/security.md#remaining-risks)
- **What:** `DockerSandbox` isolation was reviewed (security reviews of 2026-11-02, DS and R findings, fixes in
  #31/#34; re-checked by the internal audit of 2026-11-03). The static check (`dynamic/static_check.py`) is still
  a deny list on top of an import allow list, not a proof, and has had no independent review.
- **Next step:** Include the static check and the approval flow in the next independent security review that
  covers Nodestro tool execution.

### ORK-028 · Nodestro: adopt signed checkpoints and decide on single-use resume

- **Priority:** P3 Low · **Status:** Open · **Refs:** internal backend audit 2026-11-03 (ORK2-03),
  [Pausing a turn](guides/streaming-hooks-errors.md#pausing-a-turn-at-a-tool-call)
- **What:** Turn checkpoints now carry a digest over their content, and agents accept `checkpoint_key=` (HMAC) and
  `resume(..., claim=)`. Without a key the digest only detects accidental changes. nodestro-runtime stores
  checkpoints and resumes them again after its own crash (by design), so it passes neither yet.
- **Next step:** After the pin update, nodestro-runtime passes a per-installation `checkpoint_key` (from its secret
  store) and documents why it does not pass a claim (its journal and approval gate decide on a repeat).

### ORK-005 · Optional fail-closed mode for Bedrock credentials

- **Priority:** P4 Optional · **Status:** Open · **Refs:** Bedrock integration review 2026-11-01, finding L1
- **What:** Without a session or client, the Bedrock provider uses boto3's default credential chain. Multi-tenant
  hosts must make sure they always pass a session; Nodestro already enforces this in the runtime
  (`require_session`). A library flag would make the guard independent of each call site.
- **Next step:** Consider a `BedrockOptions` setting that refuses to build a default session.

## Bugs and reliability

### ORK-002 · Registries do not see records written by other processes

- **Priority:** P3 Low · **Status:** Open · **Refs:** #24, S-2 pilot finding F1 (worked around in nodestro-runtime #57)
- **What:** `ToolRegistry` and `AgentRegistry` read their directory once in `__init__`; a long-running process keeps
  stale tool versions until it creates a new registry.
- **Next step:** Add a `refresh()` or document registries as single-process snapshots.

## Architecture and modularity

None known.

## Performance

None known.

## Features and enhancements

### ORK-008 · First public release to PyPI

- **Priority:** P3 Low · **Status:** Needs decision · **Refs:** [Releasing](development/releasing.md)
- **What:** The release workflow is ready, but the one-time owner setup is not done or not confirmed: PyPI trusted
  publisher, protected `pypi` environment, branch protection, private vulnerability reporting (referenced by
  `SECURITY.md`), and whether to deploy the docs site. (The actions are SHA-pinned, #20; `releasing.md` says so.)
- **Next step:** Owner decides when to publish 0.1.0; then complete the setup.

### ORK-009 · Keep the Bedrock capability table current

- **Priority:** P3 Low · **Status:** Open · **Refs:** Bedrock provider report 2026-11-01, follow-up 4
- **What:** Bedrock capabilities per model family come from the AWS documentation of 2026-11-01
  (`bedrock_capabilities()` in `src/orkexa/providers/bedrock.py`). New models report `unknown` until the table is
  updated, which disables features in Nodestro.
- **Next step:** Define a review cadence (for example with each provider SDK update) and a short checklist in the
  provider docs.

### ORK-011 · Deferred Bedrock features

- **Priority:** P4 Optional · **Status:** Open · **Refs:** Bedrock provider and integration reports 2026-11-01
- **What:** Deliberately out of scope so far: batch inference, citations and server tools, custom or VPC endpoints,
  and the AWS European Sovereign Cloud partition. All are refused with a clear error or documented as not covered.
- **Next step:** Revisit when there is a concrete need.

### ORK-022 · Suspending native batches and delegated requests

- **Priority:** P4 Optional · **Status:** Open · **Refs:** [Pausing before a request](guides/streaming-hooks-errors.md#pausing-before-a-request)
- **What:** A usage tracker can pause a turn before a model request (`RequestSuspended`, request-stage
  `TurnCheckpoint`), but a native batch (`Evaluator.evaluate_many(batch=True)`) has no turn to suspend:
  `RequestSuspended` propagates to the caller. A delegated agent that runs as a tool of another agent (`run_agent`)
  suspends only its own turn: the outer turn sees its `TurnSuspended` like any other tool exception and gets no
  checkpoint for it (this applies to tool-stage suspension as well).
- **Next step:** Decide whether batches need a resumable form, and whether delegated agents should map their
  suspension to the outer turn (as `Pending` delegations do for orchestrations).

### ORK-029 · Nodestro Studio: messages for the new Orkexa errors

- **Priority:** P4 Optional · **Status:** Open · **Refs:** internal backend audit 2026-11-03 (ORK2-04, DX-01)
- **What:** nodestro-runtime reports agent errors as `run.agent.<ErrorClass>`. `MaxToolCallsError` (more than
  `max_tool_calls_per_round`, default 64, calls in one response) has no Studio message or fix hint yet and shows the
  generic text; the nodestro-api assistant already handles it as a `MaxToolRoundsError`. `MissingCredentialsError`
  is an `AuthenticationError` and already maps to `run.provider_auth`.
- **Next step:** With the pin update, add `run.agent.MaxToolCallsError` to the Studio translations and to the
  "advanced fix" set (`RunProblem.tsx`).

### ORK-031 · Signed offline model packages

- **Priority:** P3 Low · **Status:** Open · **Refs:** [Local models](guides/local-models.md#getting-models-offline)
- **What:** Offline model packages (`orkexa.local.export_package`) carry a SHA-256 for every file and the model
  digests, which `import_package` checks; they have no signature, so integrity is proven but origin is not. The
  model digests can be compared with the registry (Nodestro's catalog pins them).
- **Next step:** Decide whether a signature (e.g. Sigstore or a minisign key of the exporting organization) is needed
  for air-gapped deployments.

### ORK-032 · Approximate vector index for large corpora

- **Priority:** P4 Optional · **Status:** Open · **Refs:** `SQLiteIndex` vector spaces (2026-11-03)
- **What:** Vector search ranks by an exact cosine scan of the namespace's (or the allowed sources') vectors with
  sqlite-vec's scalar function. That is exact and scoped before ranking, but linear in the number of chunks.
- **Next step:** Measure on large corpora; add an approximate index only if the scan becomes the bottleneck.

### ORK-033 · Rerankers and further embedding providers

- **Priority:** P4 Optional · **Status:** Open · **Refs:** `orkexa.embeddings`
- **What:** `EmbeddingModel` has one built-in implementation (`OllamaEmbeddings`) plus `CallableEmbeddings`.
  Rerankers are not covered (Ollama has no official reranker models).
- **Next step:** Add cloud embedding providers or a reranker step when there is a concrete need.

### ORK-034 · Updating the pinned Ollama runtime

- **Priority:** P3 Low · **Status:** Open · **Refs:** `src/orkexa/local/install.py`
- **What:** `install_runtime` accepts only the archives of Ollama 0.40.14 by SHA-256. A newer Ollama needs the
  table updated from the release's `sha256sum.txt`, and the live tests rerun.
- **Next step:** Add the update to the release checklist; review with each Nodestro release.

## Tests and quality assurance

### ORK-030 · Live tests of the larger local models

- **Priority:** P2 Medium · **Status:** Open · **Refs:** `tests/test_local.py` (live tests with `ORKEXA_TEST_OLLAMA_URL`)
- **What:** Real answers, tool calls and embeddings were tested with `qwen3.5:4b` and `qwen3-embedding:0.6b` on a
  4-core CPU machine with 16 GB (no GPU). `qwen3.5:9b`, `qwen3.6:35b-a3b`, `qwen3.8:27b` and the larger embedding
  models were verified only by registry metadata; their tool calling and structured output are untested here, and
  macOS/Windows runs of `ManagedOllama` are untested.
- **Next step:** Run the live tests on hardware with 32+ GB (and a GPU, and macOS/Windows) and record the results.

## Platform coverage

### ORK-024 · Docker sandbox hosts without CI evidence (Docker Desktop, OrbStack, rootless Podman)

- **Priority:** P3 Low · **Status:** Open · **Refs:** [Feature matrix](feature-matrix.md#docker-sandbox-hosts),
  [Docker manual verification](development/docker-manual-verification.md), ORK-021, ORK-023
- **What:** The preflight and the self-test have run in CI only on Docker Engine (rootful x86-64 and arm64, rootless
  x86-64), rootful Podman 4.9.3 (crun 1.14.1, Ubuntu 24.04) and, non-blocking, Colima on macOS x86-64. **Docker
  Desktop has never run in CI** (macOS Apple silicon and Intel, Windows x64 and ARM64), nor have OrbStack, Colima on
  Apple silicon or `podman machine`: their `docker info` outputs in the unit tests are fixtures, not CI runs.
  Rootless Podman 4.9.3 (crun 1.14.1, Ubuntu 24.04) ran once in CI on 2026-11-02 (non-blocking job): its mount table
  passes `check_mount_table()` (now the recorded fixture `podman-rootless.txt`), but the self-test rightly refuses it:
  the `--ulimit` limits do not reach the container (`RLIMIT_NOFILE` 65536, `RLIMIT_FSIZE` unlimited, core hard limit
  unlimited) and crun runs unconfined by AppArmor (`crun (unconfined)`). The mount allow list was reviewed against
  what rootless Podman with crun adds (`/run/.containerenv`, `/etc/hostname`, `/etc/hosts`, `/etc/resolv.conf` from its state directory,
  `/dev/shm`, a read-only `cgroup2`, device nodes bound from the host's `/dev`, `/dev/null` and empty read-only tmpfs
  masks, read-only `/proc` views, `/run/podman-init`): every one is allowed, so no gap was found and nothing was
  widened. Refused by design (host-backed mounts): `mounts.conf` entries such as RHEL's subscription secrets on
  `/run/secrets`, containers.conf `volumes`, a `tz` default (`/etc/localtime`), and a generated `/etc/passwd` or
  `/etc/group` (Podman adds them only when the image lacks the sandbox user 65534, which the sandbox images have).
- **Next step:** Find out whether a newer rootless Podman (5.x, or containers.conf `default_ulimits`) applies the
  limits and an AppArmor profile; until then the non-blocking job `podman rootless is refused` asserts the refusal
  (the checks are not relaxed). Run the manual checklist on Docker Desktop (macOS arm64 and x86-64, Windows x64 and ARM64), OrbStack and `podman machine`
  and record the results in the feature matrix.

### ORK-012 · Real providers are not exercised

- **Priority:** P2 Medium · **Status:** Needs decision · **Refs:** [Feature matrix](feature-matrix.md),
  Bedrock provider report 2026-11-01
- **What:** All automated tests are offline. Opt-in live tests exist only for Anthropic and Bedrock and have not
  been run for the current state; the Bedrock live smoke test (including CountTokens with inference profile ids)
  is prepared but needs an AWS account, model access and owner approval for paid calls. OpenAI, Gemini, Ollama and
  vLLM have no live tests.
- **Next step:** Owner approves a small budget and credentials; add live smoke tests for the other providers and
  run all of them before a release.

### ORK-015 · No native sandbox on Windows

- **Priority:** P3 Low · **Status:** Open
- **What:** Windows has no isolating backend besides Docker. Options evaluated 2026-11-02: AppContainer + Job
  Object of our own, or Microsoft's MXC (`wxc-exec`, Windows 11 24H2+) inside a Job Object.
- **Next step:** Docker-first (ADR 0012): Docker is the Windows path; a native backend needs a new decision.

## Documentation

### ORK-010 · Architecture page and site metadata are out of date

- **Priority:** P3 Low · **Status:** Open · **Refs:** #26
- **What:** The architecture page lacks the Bedrock provider and the orchestration package, still names
  `pricing.py`, and lists ADRs only up to 0009; no ADR records the Bedrock credential and residency design; the
  removed ADR 0004 is not explained; the site description omits Bedrock.
- **Next step:** Update the page and `mkdocs.yml`, add an ADR for Bedrock.

## Technical debt

### ORK-007 · Python 3.14 untested

- **Priority:** P3 Low · **Status:** Open · **Refs:** #25, `tests/test_sandbox_policy.py`
- **What:** Python 3.11 is the minimum since 2026-11-02 (3.10 reached end of life in October 2026). On 3.14 all
  dependencies install and the suite passes except
  `test_sandbox_policy.py::test_every_module_reachable_from_allowed_imports_is_reviewed`: `annotationlib` (new in
  3.14, reached through `dataclasses` and `typing`) is not reviewed for the sandbox.
- **Next step:** Review `annotationlib` for the sandbox policy, then add 3.14 to the CI matrix and the classifiers.

### ORK-025 · Small leftovers

- **Priority:** P4 Optional · **Status:** Open
- **What:** The sdist still excludes the removed `compat/` directory (`pyproject.toml`), and
  `examples/bedrock_agent.py` is outside the numbered example scheme.
- **Next step:** Clean up with the next related change.

## Later optimizations

### ORK-026 · Freshness check for built-in prices

- **Priority:** P4 Optional · **Status:** Open · **Refs:** [Pricing engine](guides/pricing.md#known-limitations)
- **What:** The built-in Anthropic prices are a manual snapshot (`orkexa-builtin-2026-10-31`); nothing warns when
  it ages. Nodestro syncs its own prices, so the impact is limited to direct library users.
- **Next step:** Add a test or release checklist step that flags snapshots older than a set age.

### ORK-027 · Sandbox image upkeep: build behind a TLS-intercepting proxy, digest refresh

- **Priority:** P4 Optional · **Status:** Open · **Refs:** internal backend audit 2026-11-03 (ORK2-10), ORK-023
- **What:** `docker/sandbox.Dockerfile` cannot be built behind a proxy that intercepts TLS (pip fails) without
  editing it; there is no build secret or argument for an extra CA. Separately, the pinned base image digest and the
  hashed requirements need a refresh with each Python patch release.
- **Next step:** Optional `--mount=type=secret` CA support in the Dockerfile (documented); add the digest refresh to
  the release checklist.

## Accepted risks

### ORK-016 · Budgets are soft limits for some requests

- **Priority:** P3 Low · **Status:** Accepted risk · **Refs:** [Usage and budgets](guides/usage-and-budgets.md)
- **What:** With the default `on_oversized="exclusive"`, a single oversized request, requests without a cost bound
  and items of an admitted native batch can exceed a budget. Reason: refusing them by default would block normal
  use; `on_oversized="reject"` and `hard_maximum` exist for strict callers.

### ORK-017 · Running work that cannot be interrupted

- **Priority:** P4 Optional · **Status:** Accepted risk · **Refs:** [Providers](getting-started/providers.md#async-and-cancellation),
  [Security guide](guides/security.md)
- **What:** A cancelled non-streamed async Bedrock call finishes in its worker thread (boto3 limit), and a timed-out
  or cancelled tool call cannot stop plain Python code running in its thread. Sandbox calls are stopped (process
  killed, container removed) since the 2026-11-03 audit fix ORK2-01. Reason: platform limits; both are documented.

### ORK-018 · Documented unsafe opt-ins and unredacted ClaudeAgent exceptions

- **Priority:** P4 Optional · **Status:** Accepted risk · **Refs:** `SECURITY.md`, [Security guide](guides/security.md#remaining-risks)
- **What:** `SubprocessSandbox` and `AutoApprover` are explicit unsafe opt-ins, and `ClaudeAgent` raises SDK
  exceptions as they are (tracebacks may contain request details). Reason: deliberate design, documented as out of
  scope in `SECURITY.md`.

## Cross-repository dependencies

| This repo item | Other repo | Other item | Relation |
|---|---|---|---|
| ORK-002 | nodestro-runtime | #57 (registry refresh workaround) | Runtime can drop its workaround after ORK-002 |
| ORK-005 | nodestro-runtime | `require_session` guard | Runtime already enforces it; ORK-005 would add a second layer |
| ORK-007 | nodestro-runtime | RT-022 (Python 3.14) | Runtime adds 3.14 after this review; runtime depends on Orkexa |
| ORK-028 | nodestro-runtime | checkpoint storage (`tool_checkpoints`) | Runtime passes `checkpoint_key`; resume flow unchanged |
| ORK-029 | nodestro-studio | run problem messages (`errors.json`, `RunProblem.tsx`) | Studio adds `run.agent.MaxToolCallsError` |
| Any change | nodestro-runtime | Orkexa pin (`pyproject.toml`, now `bc8c4ed`) | Changes reach Nodestro only through a coordinated pin cascade; the 2026-11-03 audit fixes (ORK2-01 to ORK2-09, DX-01, DX-02) are pinned since runtime #96 |

## Maintaining this file

- Add an entry when work is identified but not done in the current PR. Use the next free ID; IDs are never reused.
- Link a GitHub issue for anything concrete enough to be picked up; keep details in the issue, not here.
- When an item is finished, delete its entry in the PR that completes it and mention the ID in the PR description.
  Git history is the record; do not keep ticked checklists.
- Accepted risks stay listed with their reason until the risk is removed or re-evaluated.
- Re-review the whole file before each release and update the "Last full review" line.
- Security issues that are not yet fixed are described only in general terms here. Report details privately
  (see SECURITY.md or GitHub private vulnerability reporting), never in this file or in public issues.
