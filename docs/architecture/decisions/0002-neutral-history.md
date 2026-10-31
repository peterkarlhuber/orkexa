# ADR 0002 – Neutral conversation model with lossless provider codecs

Status: accepted

## Context

Anthropic stores Messages-API dicts. OpenAI Responses needs item ids, encrypted
reasoning and `call_id`s; Gemini needs thought signatures on parts; Chat Completions
needs role `tool` messages. `ClaudeAgent` exposes and persists the Anthropic format.

## Decision

* The runtime works on `orkexa.core.types.Message`/`Part` only.
* Every part has `meta: {provider: {...}}`. Codecs store every field they do not model
  in the meta of their own provider and restore it when encoding for that provider,
  so `anthropic dict -> Message -> anthropic dict` is the identity (unknown block types
  become `NativePart`).
* Assistant messages record their origin provider/model. Encoding for another provider
  ignores foreign meta; foreign reasoning and native-only parts are a
  `HistoryConversionError` unless `history_policy="convert"` (drop + warning);
  provider file references are never converted.
* History files: versioned `{"format": "orkexa.history", "version": 1, ...}`; plain
  Messages API lists are detected and decoded with the Anthropic codec. `ClaudeAgent`
  stores and writes the plain list.

## Consequences

`ClaudeAgent.messages` remains a plain mutable list (converted to neutral at turn
start, written back on commit). Provider switches are explicit and never silently
discard state.
