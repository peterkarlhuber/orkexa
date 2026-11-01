# TODO: orkexa

> Open work for this repository: release blockers, known bugs, security follow-ups, technical debt and planned improvements.
> GitHub issues are the source of truth for details; this file is the entry point and summary.
>
> **Last full review:** 2026-11-01 against `master` at `3467940`.

## Summary

| Priority | Count |
|---|---|
| P0 Critical | 0 |
| P1 High | 0 |
| P2 Medium | 3 |
| P3 Low | 9 |
| P4 Optional | 6 |

**Release blockers:** None for this repository. Orkexa 0.1.0 is unreleased and no release is planned yet;
Nodestro consumes it by commit SHA.

The library is in good shape: no open issues or pull requests before this review, no TODO/FIXME markers in the
code, and offline tests, strict typing, packaging, docs and dependency audits run in CI. The open work is mostly
sandbox cancellation and container test coverage, verification against real providers and platforms, and
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

### ORK-004 · Independent review of DockerSandbox isolation and the static check

- **Priority:** P3 Low · **Status:** Needs external review · **Refs:** [Security guide](guides/security.md#remaining-risks)
- **What:** `DockerSandbox` is the default boundary for generated code, but its isolation has not been audited, and
  the static check is a deny list on top of an import allow list, not a proof.
- **Next step:** Include `orkexa.dynamic` (sandboxes, static check, approval) in the next independent security
  review that covers Nodestro tool execution.

### ORK-005 · Optional fail-closed mode for Bedrock credentials

- **Priority:** P4 Optional · **Status:** Open · **Refs:** Bedrock integration review 2026-11-01, finding L1
- **What:** Without a session or client, the Bedrock provider uses boto3's default credential chain. Multi-tenant
  hosts must make sure they always pass a session; Nodestro already enforces this in the runtime
  (`require_session`). A library flag would make the guard independent of each call site.
- **Next step:** Consider a `BedrockOptions` setting that refuses to build a default session.

## Bugs and reliability

### ORK-001 · Sandbox calls cannot be cancelled

- **Priority:** P2 Medium · **Status:** Open · **Refs:** #22, internal backend audit 2026-11-01, finding B-03
- **What:** `SubprocessSandbox` and `DockerSandbox` run until their timeout after the caller cancels; the `Sandbox`
  base class has no cancellation hook (`src/orkexa/dynamic/sandbox.py`). Nodestro's own Linux sandbox stops at once.
- **Next step:** Add a cancellation signal to the sandbox API, kill the process and remove the container, with tests.
- **Depends on:** nodestro-runtime must pass its cancellation into Orkexa sandboxes afterwards.

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
  `SECURITY.md`), and whether to deploy the docs site. Step 4 of the setup still lists SHA-pinning the actions as
  open, although #20 did that.
- **Next step:** Owner decides when to publish 0.1.0; then complete the setup and fix the stale step in
  `releasing.md`.

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

## Tests and quality assurance

### ORK-006 · DockerSandbox end-to-end test never runs in CI

- **Priority:** P2 Medium · **Status:** Open · **Refs:** #23, internal backend audit 2026-11-01 (B-01, test gaps)
- **What:** The only real-container test is skipped because no CI job builds the `orkexa-sandbox` image; the
  container paths of bounded output and cleanup are tested only against a mocked `docker` CLI.
- **Next step:** Add a CI job that builds `docker/sandbox.Dockerfile` and runs the Docker tests.

## Platform coverage

### ORK-012 · Real providers are not exercised

- **Priority:** P2 Medium · **Status:** Needs decision · **Refs:** [Feature matrix](feature-matrix.md),
  Bedrock provider report 2026-11-01
- **What:** All automated tests are offline. Opt-in live tests exist only for Anthropic and Bedrock and have not
  been run for the current state; the Bedrock live smoke test (including CountTokens with inference profile ids)
  is prepared but needs an AWS account, model access and owner approval for paid calls. OpenAI, Gemini, Ollama and
  vLLM have no live tests.
- **Next step:** Owner approves a small budget and credentials; add live smoke tests for the other providers and
  run all of them before a release.

### ORK-013 · CI runs on Linux only

- **Priority:** P3 Low · **Status:** Open
- **What:** CI uses Ubuntu only. macOS (no reliable `RLIMIT_AS`, sandbox limits) and Windows (POSIX permission
  tests are skipped; owner-private storage relies on ACLs) are untested.
- **Next step:** Add a macOS and a Windows job for the offline suite, at least before the first release.

## Documentation

### ORK-010 · Architecture page and site metadata are out of date

- **Priority:** P3 Low · **Status:** Open · **Refs:** #26
- **What:** The architecture page lacks the Bedrock provider and the orchestration package, still names
  `pricing.py`, and lists ADRs only up to 0009; no ADR records the Bedrock credential and residency design; the
  removed ADR 0004 is not explained; the site description omits Bedrock.
- **Next step:** Update the page and `mkdocs.yml`, add an ADR for Bedrock.

## Technical debt

### ORK-007 · Python version support (3.10 end of life, 3.14 untested)

- **Priority:** P3 Low · **Status:** Needs decision · **Refs:** #25
- **What:** Python 3.10 reached end of life in October 2026 and still sets the lowest supported version (and holds
  back the docs toolchain pins); Python 3.14 is not in the CI matrix or classifiers.
- **Next step:** Add 3.14; decide with nodestro-runtime whether to drop 3.10.

### ORK-014 · Small leftovers

- **Priority:** P4 Optional · **Status:** Open
- **What:** The sdist still excludes the removed `compat/` directory (`pyproject.toml`), and
  `examples/bedrock_agent.py` is outside the numbered example scheme.
- **Next step:** Clean up with the next related change.

## Later optimizations

### ORK-015 · Freshness check for built-in prices

- **Priority:** P4 Optional · **Status:** Open · **Refs:** [Pricing engine](guides/pricing.md#known-limitations)
- **What:** The built-in Anthropic prices are a manual snapshot (`orkexa-builtin-2026-10-31`); nothing warns when
  it ages. Nodestro syncs its own prices, so the impact is limited to direct library users.
- **Next step:** Add a test or release checklist step that flags snapshots older than a set age.

## Accepted risks

### ORK-016 · Budgets are soft limits for some requests

- **Priority:** P3 Low · **Status:** Accepted risk · **Refs:** [Usage and budgets](guides/usage-and-budgets.md)
- **What:** With the default `on_oversized="exclusive"`, a single oversized request, requests without a cost bound
  and items of an admitted native batch can exceed a budget. Reason: refusing them by default would block normal
  use; `on_oversized="reject"` and `hard_maximum` exist for strict callers.

### ORK-017 · Running work that cannot be interrupted

- **Priority:** P4 Optional · **Status:** Accepted risk · **Refs:** [Providers](getting-started/providers.md#async-and-cancellation),
  [Security guide](guides/security.md)
- **What:** A cancelled non-streamed async Bedrock call finishes in its worker thread (boto3 limit), and a tool
  timeout does not stop a running Python tool thread. Reason: platform limits; both are documented.

### ORK-018 · Documented unsafe opt-ins and unredacted ClaudeAgent exceptions

- **Priority:** P4 Optional · **Status:** Accepted risk · **Refs:** `SECURITY.md`, [Security guide](guides/security.md#remaining-risks)
- **What:** `SubprocessSandbox` and `AutoApprover` are explicit unsafe opt-ins, and `ClaudeAgent` raises SDK
  exceptions as they are (tracebacks may contain request details). Reason: deliberate design, documented as out of
  scope in `SECURITY.md`.

## Cross-repository dependencies

| This repo item | Other repo | Other item | Relation |
|---|---|---|---|
| ORK-001 | nodestro-runtime | audit B-03 (runtime side) | Runtime passes cancellation into Orkexa sandboxes once ORK-001 lands |
| ORK-002 | nodestro-runtime | #57 (registry refresh workaround) | Runtime can drop its workaround after ORK-002 |
| ORK-005 | nodestro-runtime | `require_session` guard | Runtime already enforces it; ORK-005 would add a second layer |
| ORK-007 | nodestro-runtime | supported Python versions | Decide together; runtime depends on Orkexa |
| Any change | nodestro-runtime | Orkexa pin (`pyproject.toml`, now `3467940`) | Changes reach Nodestro only through a coordinated pin cascade |

## Maintaining this file

- Add an entry when work is identified but not done in the current PR. Use the next free ID; IDs are never reused.
- Link a GitHub issue for anything concrete enough to be picked up; keep details in the issue, not here.
- When an item is finished, delete its entry in the PR that completes it and mention the ID in the PR description.
  Git history is the record; do not keep ticked checklists.
- Accepted risks stay listed with their reason until the risk is removed or re-evaluated.
- Re-review the whole file before each release and update the "Last full review" line.
- Security issues that are not yet fixed are described only in general terms here. Report details privately
  (see SECURITY.md or GitHub private vulnerability reporting), never in this file or in public issues.
