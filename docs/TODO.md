# TODO: orkexa

> Open work for this repository: release blockers, known bugs, security follow-ups, technical debt and planned improvements.
> GitHub issues are the source of truth for details; this file is the entry point and summary.
>
> **Last full review:** 2026-11-07 against `master` at `2da5c6c` (TODO abatement round 4).

## Summary

| Priority | Count |
|---|---|
| P0 Critical | 0 |
| P1 High | 0 |
| P2 Medium | 4 |
| P3 Low | 10 |
| P4 Optional | 7 |

**Release blockers:** None. Orkexa 0.2.0 is unreleased and no release is planned yet; Nodestro consumes it by
commit SHA. No entry blocks the frozen desktop candidate v0.2.0-preview.1 or the next Nodestro preview; every entry
below is not a release blocker.

The library is in good shape: no TODO/FIXME markers in the code, and offline tests (Python 3.11 to 3.14), strict
typing, packaging, docs and dependency audits run in CI; the live Ollama tests run weekly (`ollama-live.yml`). The
issues #24 to #26 are fixed on `ws-ork` (ORK-002, ORK-007, ORK-010) and can be closed once it is merged. The open work
is mostly verification against real providers and Docker hosts, owner decisions, and follow-ups in Nodestro.

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

Needs an independent review: ORK-004 (the static check). It belongs in the same independent review as Nodestro's
tool approvals (nodestro-runtime RT-001); `NODESTRO_TOOL_APPROVALS` stays off until then.

### ORK-003 · Loopback rule for `OllamaProvider`

- **Priority:** P3 Low · **Status:** Needs decision · **Refs:** internal backend audit 2026-11-01 (A-05 follow-up),
  internal backend audit 2026-11-04
- **What:** OpenAI-compatible providers (`compatible`, `ollama`, `vllm`) no longer follow HTTP redirects, and the
  managed Ollama server's start checks, error mapping, private log files and log rotation were hardened (2026-11-06).
  Left open: `OllamaProvider` (the `Agent` path) accepts any `base_url`, while `OllamaClient` is limited to loopback
  by default. Nodestro relies on a non-loopback Ollama in Local Docker, so a loopback default needs an agreed opt-in.
- **Next step:** Owner and Nodestro agree on an option (for example `allow_remote=True` for non-loopback hosts) and
  its default; then document it in the security guide.

### ORK-004 · Independent review of the static check

- **Priority:** P3 Low · **Status:** Needs external review · **Refs:** [Security guide](guides/security.md#remaining-risks)
- **What:** `DockerSandbox` isolation was reviewed (security reviews of 2026-11-02, DS and R findings, fixes in
  #31/#34; re-checked by the internal audit of 2026-11-03). The static check (`dynamic/static_check.py`) is still
  a deny list on top of an import allow list, not a proof, and has had no independent review.
- **Next step:** Include the static check and the approval flow in the next independent security review that
  covers Nodestro tool execution (nodestro-runtime RT-001).

### ORK-028 · Nodestro: adopt signed checkpoints and decide on single-use resume

- **Priority:** P3 Low · **Status:** Open · **Refs:** internal backend audit 2026-11-03 (ORK2-03),
  [Pausing a turn](guides/streaming-hooks-errors.md#pausing-a-turn-at-a-tool-call)
- **What:** Turn checkpoints now carry a digest over their content, and agents accept `checkpoint_key=` (HMAC) and
  `resume(..., claim=)`. Without a key the digest only detects accidental changes. nodestro-runtime stores
  checkpoints and resumes them again after its own crash (by design), so it passes neither yet.
- **Next step:** After the pin update, nodestro-runtime passes a per-installation `checkpoint_key` (from its secret
  store) and documents why it does not pass a claim (its journal and approval gate decide on a repeat).

## Bugs and reliability

None known.

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
- **Next step:** The release rules (MUP-ONA B0, [Versioning and releases](development/releasing.md#versioning-and-releases))
  say no PyPI. If the owner decides to publish after all: complete the setup and make the workflow's tag check accept
  `orkexa-vX.Y.Z`.

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

### ORK-035 · Source of the mDenseOn tokenizer metadata

- **Priority:** P3 Low · **Status:** Open · **Refs:** `src/orkexa/local/convert.py` (`MDENSEON`), 2026-11-03
- **What:** The `MDENSEON` recipe takes its tokenizer and hyperparameter metadata from the 6 MB header of a community
  llama.cpp conversion of mmBERT-base (`mykor/mmBERT-base-GGUF`, MIT, pinned by revision and SHA-256), because
  mDenseOn ships no GGUF and the converter was not run here. That conversion sets `add_space_prefix=false`, which
  does not match the Hugging Face tokenizer; the recipe corrects it and the reference-vector check guards it.
- **Next step:** On a machine where running the converter is allowed (it was not run in the 2026-11-06 round),
  run llama.cpp's `convert_hf_to_gguf.py` on mDenseOn at a pinned llama.cpp commit, compare the
  metadata with the template, report the `add_space_prefix` mismatch upstream, and switch the recipe's template to
  our own conversion or to an official LightOn GGUF if one is published.

### ORK-036 · Synthetic structured-answer request in the stored conversation

- **Priority:** P4 Optional · **Status:** Needs decision · **Refs:** internal backend audit 2026-11-04 (ORK3-08)
- **What:** With providers that split tools and structured output (Ollama), the agent adds a user message that asks
  for the JSON answer and keeps it in the conversation history, although the user never wrote it.
- **Next step:** Mark it as synthetic or drop it on commit; agree with Nodestro how chat views show it.

## Tests and quality assurance

### ORK-030 · Live tests of the larger local models

- **Priority:** P2 Medium · **Status:** Open · **Refs:** `tests/test_local.py` (live tests with
  `ORKEXA_TEST_OLLAMA_URL`); internal backend audit round 3 (ORK3-04, ORK3-05, OPS3-04)
- **What:** Real answers, tool calls and embeddings were tested with `qwen3.5:4b` and `qwen3-embedding:0.6b` on a
  4-core CPU machine with 16 GB (no GPU). `qwen3.5:9b`, `qwen3.6:35b-a3b`, `qwen3.8:27b` and the larger embedding
  models were verified only by registry metadata; their tool calling and structured output are untested here, and
  macOS/Windows runs of `ManagedOllama` are untested. The IPv6 loopback path (`host="::1"`) and the orphan reclaim
  after a host crash (POSIX only; Windows has none) were added on 2026-11-04 (#49) without a CI run on those
  platforms. The weekly `ollama-live.yml` job runs the live tests with the small models on Linux CPU only.
  Since 2026-11-07 the managed server ends with its application (`orkexa/local/_lifeline.py`: a watcher process on
  POSIX, tested on Linux; a job object with kill-on-close on Windows, whose tests run only in `platforms.yml` and
  have not run yet); macOS has no local run of it either.
- **Next step:** Run the live tests on hardware with 32+ GB (and a GPU, and macOS/Windows) and record the results;
  check the first `platforms.yml` run of `tests/test_managed_server_lifetime.py` on Windows and macOS.

## Platform coverage

### ORK-024 · Docker sandbox hosts without CI evidence (Docker Desktop, OrbStack, rootless Podman)

- **Priority:** P3 Low · **Status:** Open · **Refs:** [Feature matrix](feature-matrix.md#docker-sandbox-hosts),
  [Docker manual verification](development/docker-manual-verification.md), ORK-021 (done, #31), ORK-023 (done, #34)
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
- **What:** All automated tests are offline. Opt-in live smoke tests (`pytest -m live`) exist for every provider:
  Anthropic (`tests/anthropic/test_live.py`), Bedrock (`tests/test_bedrock_live.py`, including CountTokens with
  inference profile ids), Ollama (`tests/test_ollama_live.py`, weekly in `ollama-live.yml`) and, since 2026-11-07,
  OpenAI, Gemini and vLLM (`tests/test_provider_live.py`: text, one tool call, structured answer; the model is named
  by a variable, none is built in). The paid ones and vLLM have never run for the current state: they need keys,
  model access (AWS), a vLLM server with tool calling, and owner approval for paid calls.
- **Next step:** Owner approves a small budget and credentials; run all live smoke tests before a release and
  record the results in the feature matrix.

### ORK-015 · No native sandbox on Windows

- **Priority:** P3 Low · **Status:** Open
- **What:** Windows has no isolating backend besides Docker. Options evaluated 2026-11-02: AppContainer + Job
  Object of our own, or Microsoft's MXC (`wxc-exec`, Windows 11 24H2+) inside a Job Object.
- **Next step:** Docker-first (ADR 0012): Docker is the Windows path; a native backend needs a new decision.

## Documentation

None known.

## Technical debt

### ORK-038 · Dependency updates are merged by hand; branch protection decided

- **Priority:** P3 Low · **Status:** Decided (owner, 2026-11-05), not implemented · **Refs:** `.github/dependabot.yml`,
  dependency round 2026-11-05, nodestro-studio ST-052
- **What:** Dependabot opens weekly grouped pull requests (pip minor/patch, GitHub Actions; majors of the provider
  SDKs and pydantic are ignored and reviewed by hand) with a cooldown of 3 days (7 for pip majors); security updates are
  not delayed. Auto-merge is off. A new pip range in `pyproject.toml` changes what every Nodestro install resolves,
  so it needs CI here and the next studio pin bump.
  **Owner decision 2026-11-05:** protect `master` with required CI checks and block force pushes and deletion,
  without blocking the agent and pull request flow; auto-merge stays off for now.
- **Next step:** Separate work order following nodestro-studio ST-052: required checks only from jobs that always
  report on pull requests (the macOS and Windows jobs of `platforms.yml` only if they run on every pull request),
  no required approvals, "up to date" off, merge commits allowed.

## Later optimizations

None known.

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

## MUP-ONA

Packages of the MUP-ONA plan (security and database backends for Orkexa, Nodestro and Axentra; [plan](https://github.com/peterkarlhuber/nodestro-studio/blob/master/docs/architecture/mup-ona/MUP-ONA.md))
that change this repository. The baseline (MP-A0: legacy fixtures FX-SQ-0 and FX-RANK-0, skip expectations), MP-A10 (SQLite
robustness), the Orkexa part of MP-A1 (PDF budgets and isolation), the Orkexa part of MP-B0 (version 0.2.0.dev0,
API snapshot, import contracts, release rules), the Orkexa part of MP-B1 (storage contracts, public conformance
suite `orkexa.testing.conformance`, `ScriptedClient`), MP-B2 (SQLite backends in `orkexa.backends.sqlite`) and the
rule engine of MP-B5 (`orkexa.testing.sqlcheck`) are in place; each entry is removed in the change that completes its package.

### ORK-MUP-B5 · `orkexa.backends.postgres` and `orkexa.testing.sqlcheck`

- **Priority:** P2 Medium · **Status:** Open · **Refs:** [MUP-ONA MP-B5](https://github.com/peterkarlhuber/nodestro-studio/blob/master/docs/architecture/mup-ona/MUP-ONA.md)
- **What:** The generic PostgreSQL backend still lives in nodestro-runtime. The tenant SQL rule engine is in place
  (PR 1: `orkexa.testing.sqlcheck`, extra `sqlcheck`, T-SQLC-1, T-SQLCHECK).
- **Next step:** PR 2 (`mup/b5-postgres-backend`): extra `postgres`, `orkexa.backends.postgres` with statement
  catalog and partition guard, schema v1, `verify_schema`/`migrate`, T-PART/T-CAT/T-GUARD/T-PAR-2, CI jobs `pg-a`,
  `pg-b`, `core-no-extras`.

## Cross-repository dependencies

| This repo item | Other repo | Other item | Relation |
|---|---|---|---|
| ORK-003 | nodestro-runtime | Local Docker Ollama URL (non-loopback) | A loopback default needs the runtime's opt-in first |
| ORK-028 | nodestro-runtime | checkpoint storage (`tool_checkpoints`) | Runtime passes `checkpoint_key`; resume flow unchanged |
| ORK-030 | nodestro-runtime, nodestro-desktop | RT-049, DSK-022 (managed Ollama on macOS/Windows) | The managed server lives here; runtime and desktop rerun their tests |
| (done 2026-11-07) | nodestro-desktop | DSK-022 (the managed Ollama survives a killed backend) | With the next pin the server ends with its application (watcher on POSIX, job object on Windows); the desktop adds its kill test |
| ORK-036 | nodestro-runtime | chat views of stored conversations | Agree how chat views show the synthetic request |
| (done 2026-11-06) | nodestro-runtime | #57 workaround, `require_session` guard, RT-022, Bedrock caching option | With the next pin: `ToolRegistry.refresh()`/`AgentRegistry.refresh()`, `BedrockOptions(require_session=True)`, Python 3.14, `BedrockOptions(prompt_caching=True)` are available |
| Any change | nodestro-runtime | Orkexa pin (`pyproject.toml`, `e9c2147` since runtime #113) | Changes reach Nodestro only through a coordinated pin cascade; the round-3 audit fixes (orkexa #49) are pinned since runtime #113 |

## Maintaining this file

- Add an entry when work is identified but not done in the current PR. Use the next free ID; IDs are never reused.
- Link a GitHub issue for anything concrete enough to be picked up; keep details in the issue, not here.
- When an item is finished, delete its entry in the PR that completes it and mention the ID in the PR description.
  Git history is the record; do not keep ticked checklists.
- Accepted risks stay listed with their reason until the risk is removed or re-evaluated.
- Re-review the whole file before each release and update the "Last full review" line.
- Security issues that are not yet fixed are described only in general terms here. Report details privately
  (see SECURITY.md or GitHub private vulnerability reporting), never in this file or in public issues.
