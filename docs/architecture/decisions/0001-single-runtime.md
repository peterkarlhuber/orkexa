# ADR 0001 – One sans-I/O runtime for sync and async agents

Status: accepted

## Context

`claude-agent` 0.8.0 kept configuration and response handling in `_AgentBase` but
duplicated the request/tool loop in `ClaudeAgent.ask` and `AsyncClaudeAgent.ask`.
Orkexa must support four integration types with one loop and no per-provider loops.

## Decision

The complete turn logic is a generator, `AgentCore._turn()`, that yields effects
(`CallModel`, `RunTools`) and receives their results. `Agent` and `AsyncAgent` only
perform effects (blocking or awaiting). Providers are stateless translators between
`ModelRequest`/`ModelResponse` and SDK calls.

## Consequences

* State transitions (copy, commit, rollback, limits, refusal, usage) exist once and
  are tested once with the SDK-free `FakeProvider`, then re-run for every adapter.
* Closing the generator in `finally` guarantees per-turn cleanup on errors and on
  `asyncio` cancellation.
* Adding a provider never touches the runtime (`register_provider`).
