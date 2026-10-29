# ADR 0003 – Capabilities: `unknown` is not `supported`

Status: accepted

## Decision

Each feature has `supported`, `unsupported` or `unknown` per provider/model/endpoint.
The runtime calls `require(feature)` before using it: `unsupported` raises
`UnsupportedCapabilityError`, `unknown` raises `UnknownCapabilityError` with the exact
override (`capabilities={"tools": "supported"}`) to use. Protocol-level features of
first-party APIs (tools and streaming on Anthropic/OpenAI/Gemini) are `supported`;
model-dependent features use documented model families; everything on an unknown
compatible endpoint starts as `unknown`, except streaming (protocol feature).
Presets (`ollama`, `vllm`) set typical values and are documented as such.

## Consequences

No silent degradation: requests that would be rejected or misinterpreted fail before
they are sent, with an actionable message.
