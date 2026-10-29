# ADR 0004 – Legacy imports through the `orkexa-claude-compat` distribution

Status: accepted

## Context

Unchanged code must keep `from claude_agent import ClaudeAgent` and
`from claude_agent.dynamic import ...`. pip has no file-conflict detection: two
distributions shipping `claude_agent/` overwrite each other silently, and uninstalling
one breaks the other.

## Decision

* `orkexa` itself never ships `claude_agent`.
* A separate distribution `orkexa-claude-compat` (directory `compat/orkexa-claude-compat`)
  ships only the package `claude_agent`, consisting of re-exports and module aliases
  (`sys.modules["claude_agent.retrieval"] = orkexa.retrieval`); it depends on
  `orkexa[anthropic]` of the same version. No business logic is duplicated.
* Installation: `pip uninstall claude-agent && pip install orkexa-claude-compat`.
* Guard: importing the shim raises an `ImportError` explaining the conflict if the
  distribution metadata of the original `claude-agent` is installed.
  `python -m orkexa.compat` reports which distribution owns `claude_agent`.
* Both directions (shim after original, original after shim) and uninstallation are
  tested in fresh virtual environments (`tests/packaging/`).

## Alternatives considered

Publishing the shim as `claude-agent` 1.0 would let pip replace the original cleanly,
but it takes over the original project name; this is left to the maintainer.
