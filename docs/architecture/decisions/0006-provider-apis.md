# ADR 0006 – Provider SDKs, API families and versions

Status: accepted (verified against the SDK versions installed on 2026-10-29)

| Integration | SDK (tested) | API family | Notes |
| --- | --- | --- | --- |
| Anthropic | `anthropic` 2.3 (`>=2.0,<3`) | Messages API (`messages.create/parse/stream/count_tokens`, `beta.messages`, `messages.batches`) | native blocks kept losslessly |
| OpenAI | `openai` 3.41 (`>=3.0,<4`) | **Responses API** (`responses.create/parse/stream`, `responses.input_tokens.count`) | stateless by default (`store=False`, encrypted reasoning replayed) so Orkexa owns the history; `previous_response_id` is not used |
| Gemini | `google-genai` 2.34 (`>=2.0,<3`) | `models.generate_content` / `generate_content_stream` / `count_tokens` on the Gemini Developer API or Vertex AI (same SDK surface, selected by `GeminiOptions(vertexai=...)`) | not the OpenAI-compatible Gemini endpoint and not the Interactions API; function declarations with `parameters_json_schema`, `automatic_function_calling.disable=True`, thought signatures preserved |
| Compatible | `openai` 3.x client with custom `base_url` | Chat Completions (`chat.completions.create`) or Responses | protocol explicit; `auto` only from static preset data |

Version ranges are bounded by the next major version because the adapters use typed
response fields of these majors.
